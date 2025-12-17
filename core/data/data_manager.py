import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Literal
from dataclasses import dataclass

# from config import S3_CONFIG, DATA_DIR

DATA_DIR = Path("../store") 
S3_CONFIG = {
    "bucket": "data-collector-hft",
    "prefix": "",
    "endpoint_url": "https://storage.yandexcloud.net",
    "aws_access_key_id": "YCAJEVJeIO1bNwwm7wm9o9by1",
    "aws_secret_access_key": "YCNJ_6Frr8TWLFZxASFW47ZeYGFTtEawQaE0gwXa",
    "aws_region": "ru-central1",
}

@dataclass
class MarketData:
    """Контейнер для данных разных типов"""
    orderbook: pl.DataFrame | None = None
    trades: pl.DataFrame | None = None
    
    def __repr__(self) -> str:
        ob_rows = len(self.orderbook) if self.orderbook is not None else 0
        tr_rows = len(self.trades) if self.trades is not None else 0
        return f"MarketData(orderbook={ob_rows:,} rows, trades={tr_rows:,} rows)"


# Стандартные схемы колонок (формат Binance/Bybit)
COLUMN_SCHEMAS = {
    "orderbook": ["event_time", "update_id", "bid_price", "bid_qty", "ask_price", "ask_qty"],
    "trades": ["event_time", "trade_id", "price", "qty", "trade_time", "is_maker"],
}

# Схемы для Gate.io (qty со знаком, без is_maker)
GATE_COLUMN_SCHEMAS = {
    "orderbook": ["event_time", "update_id", "bid_price", "bid_qty", "ask_price", "ask_qty"],
    "trades": ["event_time", "trade_id", "price", "qty", "trade_time"],
}


def to_utc(dt: datetime | str) -> datetime:
    """
    Конвертирует datetime в UTC.
    Если передана строка — парсит и добавляет UTC.
    Если datetime без tzinfo — считаем что это уже UTC.
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    
    if dt.tzinfo is None:
        # Naive datetime — считаем UTC
        return dt.replace(tzinfo=timezone.utc)
    else:
        # Aware datetime — конвертируем в UTC
        return dt.astimezone(timezone.utc)


class DataManager:
    """Клиент для работы с S3. Все операции в UTC."""
    
    DATA_TYPES = {
        "orderbook": "_orderbook.gz",
        "trades": "_trades.gz",
    }
    
    def __init__(self):
        self.bucket = S3_CONFIG["bucket"]
        self.prefix = S3_CONFIG["prefix"]
        
        self.client = boto3.client(
            "s3",
            endpoint_url=S3_CONFIG.get("endpoint_url"),
            aws_access_key_id=S3_CONFIG["aws_access_key_id"],
            aws_secret_access_key=S3_CONFIG["aws_secret_access_key"],
            region_name=S3_CONFIG["aws_region"],
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )
        
        self.transfer_config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=8 * 1024 * 1024,
            max_concurrency=10,
            multipart_chunksize=8 * 1024 * 1024,
        )
    
    # ==================== Базовые операции ====================
    
    def list_objects(self, prefix: str = "") -> list[dict]:
        """Список объектов в бакете"""
        full_prefix = f"{self.prefix}{prefix}"
        
        objects = []
        paginator = self.client.get_paginator("list_objects_v2")
        
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    objects.append({
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "modified": obj["LastModified"],
                    })
        
        return objects
    
    def exists(self, key: str) -> bool:
        """Проверить существование объекта"""
        try:
            self.client.head_object(Bucket=self.bucket, Key=f"{self.prefix}{key}")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise
    
    def download_bytes(self, key: str) -> bytes:
        """Скачать как байты"""
        response = self.client.get_object(
            Bucket=self.bucket,
            Key=key,
        )
        return response["Body"].read()
    
    # ==================== Детекция и нормализация ====================
    
    def _detect_data_type(self, key: str) -> str | None:
        """Определяет тип данных по имени файла"""
        for dtype, suffix in self.DATA_TYPES.items():
            if suffix in key:
                return dtype
        return None
    
    def _detect_exchange(self, key: str) -> str | None:
        """Определяет биржу по пути файла"""
        key_lower = key.lower()
        if "/binance/" in key_lower:
            return "binance"
        elif "/bybit/" in key_lower:
            return "bybit"
        elif "/gate/" in key_lower:
            return "gate"
        return None
    
    def _normalize_gate_trades(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Нормализует trades Gate.io к формату Binance.
        
        Gate.io: qty < 0 = тейкер sell, qty > 0 = тейкер buy
        Binance: is_maker = 1 если buyer is maker
        """
        return df.with_columns([
            (pl.col("qty") < 0).cast(pl.Int8).alias("is_maker"),
            pl.col("qty").abs().alias("qty"),
        ])
    
    # ==================== Загрузка файлов ====================
    
    def load_csv_gz(self, key: str, cache_dir: Path | str | None = None) -> pl.DataFrame:
        """
        Загрузить CSV.GZ файл и вернуть DataFrame с именованными колонками.
        """
        if cache_dir is None:
            cache_dir = Path(DATA_DIR)
        else:
            cache_dir = Path(cache_dir)
        
        local_cache_path = cache_dir / key
        
        if not local_cache_path.exists():
            print(f"📥 Файл не найден локально, скачиваю из S3: {key}")
            
            try:
                local_cache_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise OSError(f"❌ Невозможно создать директорию: {e}") from e
            
            full_key = f"{self.prefix}{key}" if self.prefix else key
            
            try:
                data = self.download_bytes(full_key)
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    raise FileNotFoundError(f"❌ Файл не найден в S3: {full_key}") from e
                raise
            
            try:
                with open(local_cache_path, 'wb') as f:
                    f.write(data)
                print(f"✅ Файл сохранен: {local_cache_path}")
            except OSError as e:
                raise OSError(f"❌ Невозможно сохранить файл: {e}") from e
        else:
            print(f"📁 Используется локальный файл: {local_cache_path}")
        
        df = pl.read_csv(local_cache_path, has_header=False)
        
        data_type = self._detect_data_type(key)
        exchange = self._detect_exchange(key)
        
        if exchange == "gate" and data_type == "trades":
            expected_columns = GATE_COLUMN_SCHEMAS["trades"]
        elif data_type and data_type in COLUMN_SCHEMAS:
            expected_columns = COLUMN_SCHEMAS[data_type]
        else:
            expected_columns = None
        
        if expected_columns:
            if len(df.columns) == len(expected_columns):
                df = df.rename(dict(zip(df.columns, expected_columns)))
            else:
                print(f"⚠️ Несоответствие колонок в {key}: "
                      f"ожидается {len(expected_columns)}, получено {len(df.columns)}")
        
        if exchange == "gate" and data_type == "trades":
            df = self._normalize_gate_trades(df)
        
        return df
    
    # ==================== Работа с временными диапазонами ====================
    
    def _generate_hourly_keys(
        self,
        exchange: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        data_type: Literal["orderbook", "trades", "all"] = "all",
        market_type: Literal["futures", "spot"] = "futures"
    ) -> dict[str, list[str]]:
        """Генерирует список ключей для почасовых файлов."""
        if data_type == "all":
            types_to_load = list(self.DATA_TYPES.keys())
        else:
            types_to_load = [data_type]
        
        keys_by_type: dict[str, list[str]] = {t: [] for t in types_to_load}
        
        current = start_time.replace(minute=0, second=0, microsecond=0)
        end_hour = end_time.replace(minute=0, second=0, microsecond=0)
        
        while current <= end_hour:
            date_str = current.strftime("%Y%m%d")
            hour_str = str(current.hour)
            base_path = f"{market_type}/{exchange.lower()}/{symbol.lower()}/{date_str}/{hour_str}"
            
            for dtype in types_to_load:
                suffix = self.DATA_TYPES[dtype]
                key = f"{base_path}{suffix}"
                keys_by_type[dtype].append(key)
            
            current += timedelta(hours=1)
        
        return keys_by_type
    
    def _load_and_filter_files(
        self,
        keys: list[str],
        cache_dir: Path,
        start_ms: int,
        end_ms: int,
        timestamp_column: str = "event_time"
    ) -> pl.DataFrame | None:
        """Загружает файлы и фильтрует по времени (UTC)"""
        dfs = []
        
        for key in keys:
            try:
                df = self.load_csv_gz(key, cache_dir=cache_dir)
                dfs.append(df)
            except FileNotFoundError:
                print(f"⏭️  Пропускаю отсутствующий файл: {key}")
                continue
            except Exception as e:
                print(f"❌ Ошибка при загрузке {key}: {e}")
                continue
        
        if not dfs:
            return None
        
        combined_df = pl.concat(dfs, how="vertical")
        
        if timestamp_column in combined_df.columns:
            filtered_df = combined_df.filter(
                (pl.col(timestamp_column) >= start_ms) &
                (pl.col(timestamp_column) <= end_ms)
            ).sort(timestamp_column)
            
            print(f"   📊 До фильтрации: {len(combined_df):,}, после: {len(filtered_df):,}")
            return filtered_df
        else:
            print(f"   ⚠️ Колонка '{timestamp_column}' не найдена. "
                  f"Доступные: {combined_df.columns}")
            return combined_df
    
    def load_timerange(
        self,
        exchange: str,
        symbol: str,
        start_time: datetime | str,
        end_time: datetime | str,
        data_type: Literal["orderbook", "trades", "all"] = "all",
        market_type: Literal["futures", "spot"] = "futures",
        cache_dir: Path | str | None = None,
        timestamp_column: str = "event_time"
    ) -> MarketData | pl.DataFrame:
        """
        Загрузить данные за временной диапазон.
        
        ⚠️ Все времена интерпретируются как UTC!
        
        Args:
            exchange: биржа ("binance", "bybit", "gate")
            symbol: символ ("btcusdt", "ethusdt")
            start_time: начало периода UTC (datetime или строка ISO)
            end_time: конец периода UTC (datetime или строка ISO)
            data_type: "orderbook", "trades" или "all"
            market_type: "futures" или "spot"
            cache_dir: директория для кэша
            timestamp_column: колонка с временной меткой
            
        Returns:
            MarketData или DataFrame
            
        Example:
            >>> data = dataManager.load_timerange(
            ...     "binance", "btcusdt", 
            ...     "2025-12-17 10:00:00",  # 10:00 UTC
            ...     "2025-12-17 11:00:00",  # 11:00 UTC
            ... )
        """
        # Конвертируем в UTC
        start_time = to_utc(start_time)
        end_time = to_utc(end_time)
        
        print(f"\n🔍 Загрузка данных для {symbol.upper()} на {exchange.upper()}")
        print(f"📅 Период: {start_time.strftime('%Y-%m-%d %H:%M:%S')} → {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"📊 Тип данных: {data_type}")
        
        keys_by_type = self._generate_hourly_keys(
            exchange, symbol, start_time, end_time, data_type, market_type
        )
        
        total_files = sum(len(keys) for keys in keys_by_type.values())
        print(f"📦 Требуется файлов: {total_files}")
        
        cache_dir = Path(cache_dir) if cache_dir else Path(DATA_DIR)
        
        # Конвертируем в миллисекунды UTC
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        all_keys = [key for keys in keys_by_type.values() for key in keys]
        missing_locally = [
            key for key in all_keys
            if not (cache_dir / key).exists()
        ]
        
        if missing_locally:
            print(f"📥 Нужно скачать из S3: {len(missing_locally)} файлов")
        else:
            print(f"✅ Все файлы уже есть локально")
        
        if data_type != "all":
            keys = keys_by_type[data_type]
            df = self._load_and_filter_files(
                keys, cache_dir, start_ms, end_ms, timestamp_column
            )
            
            if df is None:
                raise ValueError(f"❌ Не удалось загрузить данные {data_type} для {symbol}")
            
            print(f"\n📈 Загружено {data_type}: {len(df):,} строк")
            return df
        
        result = MarketData()
        
        for dtype, keys in keys_by_type.items():
            print(f"\n📂 Загружаю {dtype}...")
            
            df = self._load_and_filter_files(
                keys, cache_dir, start_ms, end_ms, timestamp_column
            )
            
            if df is not None:
                print(f"   ✅ {dtype}: {len(df):,} строк")
                setattr(result, dtype, df)
            else:
                print(f"   ⚠️ {dtype}: нет данных")
        
        print(f"\n{result}")
        return result

    # ==================== Утилиты ====================
    
    def _full_prefix(self, *parts: str) -> str:
        """Собирает полный путь"""
        all_parts = [self.prefix] if self.prefix else []
        all_parts.extend(parts)
        return "/".join(p.strip("/") for p in all_parts if p) + "/"
    
    def _list_prefixes(self, prefix: str) -> list[str]:
        """Получает список 'папок' на уровне prefix."""
        prefixes = []
        paginator = self.client.get_paginator("list_objects_v2")
        
        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=prefix,
            Delimiter="/",
        ):
            for cp in page.get("CommonPrefixes", []):
                folder = cp["Prefix"].rstrip("/").split("/")[-1]
                prefixes.append(folder)
        
        return prefixes
    
    @staticmethod
    def now_utc() -> datetime:
        """Текущее время в UTC"""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def ms_to_utc(ms: int) -> datetime:
        """Конвертирует миллисекунды в datetime UTC"""
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    
    @staticmethod
    def utc_to_ms(dt: datetime | str) -> int:
        """Конвертирует datetime в миллисекунды UTC"""
        dt = to_utc(dt)
        return int(dt.timestamp() * 1000)


dataManager = DataManager()


# if __name__ == "__main__":
#     # Теперь всё в UTC!
#     data = dataManager.load_timerange(
#         exchange="binance",
#         symbol="fheusdt",
#         start_time="2025-12-17 10:00:00",  # Это UTC
#         end_time="2025-12-17 11:00:00",    # Это UTC
#         data_type="all",
#         market_type="futures"
#     )
#     print(data)