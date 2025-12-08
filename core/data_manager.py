# core/s3_client.py
import gzip
import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
from typing import Literal
from dataclasses import dataclass

from config import S3_CONFIG, DATA_DIR


@dataclass
class MarketData:
    """Контейнер для данных разных типов"""
    orderbook: pl.DataFrame | None = None
    trades: pl.DataFrame | None = None
    
    def __repr__(self) -> str:
        ob_rows = len(self.orderbook) if self.orderbook is not None else 0
        tr_rows = len(self.trades) if self.trades is not None else 0
        return f"MarketData(orderbook={ob_rows:,} rows, trades={tr_rows:,} rows)"


class DataManager:
    """Клиент для работы с S3"""
    
    # Типы данных и их суффиксы в именах файлов
    DATA_TYPES = {
        "orderbook": "_orderbook.gz",
        "trades": "_trades.gz",
    }
    
    def __init__(self):
        self.bucket = S3_CONFIG["bucket"]
        self.prefix = S3_CONFIG["prefix"]
        
        # Создаём клиент
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
        
        # Для upload/download больших файлов
        self.transfer_config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=8 * 1024 * 1024,  # 8MB
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
    
    
    # ==================== Загрузка/Скачивание ====================
    
    def download_bytes(self, key: str) -> bytes:
        """Скачать как байты"""
        response = self.client.get_object(
            Bucket=self.bucket,
            Key=key,
        )
        return response["Body"].read()
    
    def load_csv_gz(self, key: str, cache_dir: Path | str | None = None) -> pl.DataFrame:
        """
        Загрузить CSV.GZ файл и вернуть DataFrame.
        Если файл есть локально - использует его, иначе скачивает из S3.
        
        Args:
            key: путь к файлу в S3 (например, "futures/binance/btcusdt/20251204/20_orderbook.gz")
            cache_dir: директория для кэша (по умолчанию из DATA_DIR)
            
        Returns:
            pl.DataFrame
        """
        # Определяем директорию кэша
        if cache_dir is None:
            cache_dir = Path(DATA_DIR)
        else:
            cache_dir = Path(cache_dir)
        
        # Локальный путь для кэша
        local_cache_path = cache_dir / key
        
        # Проверяем наличие файла локально
        if not local_cache_path.exists():
            print(f"📥 Файл не найден локально, скачиваю из S3: {key}")
            
            try:
                # Создаем директории
                local_cache_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise OSError(
                    f"❌ Невозможно создать директорию {local_cache_path.parent}.\n"
                    f"Ошибка: {e}\n"
                    f"Проверьте права доступа или измените DATA_DIR в config.py"
                ) from e
            
            # Скачиваем файл
            full_key = f"{self.prefix}{key}" if self.prefix else key
            
            try:
                data = self.download_bytes(full_key)
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    raise FileNotFoundError(f"❌ Файл не найден в S3: {full_key}") from e
                raise
            
            # Сохраняем
            try:
                with open(local_cache_path, 'wb') as f:
                    f.write(data)
                print(f"✅ Файл сохранен: {local_cache_path}")
            except OSError as e:
                raise OSError(
                    f"❌ Невозможно сохранить файл {local_cache_path}.\n"
                    f"Ошибка: {e}\n"
                    f"Проверьте права доступа"
                ) from e
        else:
            print(f"📁 Используется локальный файл: {local_cache_path}")
        
        # Polars автоматически распаковывает .gz
        return pl.read_csv(local_cache_path, has_header=False)
    
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
        """
        Генерирует список ключей для почасовых файлов.
        
        Args:
            exchange: "binance", "bybit"
            symbol: "btcusdt", "ethusdt"
            start_time: начало периода
            end_time: конец периода
            data_type: "orderbook", "trades" или "all" для обоих
            market_type: тип рынка
            
        Returns:
            Словарь {data_type: [keys]}
        """
        # Определяем какие типы данных нужны
        if data_type == "all":
            types_to_load = list(self.DATA_TYPES.keys())
        else:
            types_to_load = [data_type]
        
        keys_by_type: dict[str, list[str]] = {t: [] for t in types_to_load}
        
        # Начинаем с начала часа
        current = start_time.replace(minute=0, second=0, microsecond=0)
        
        # Идем до конца часа, в котором находится end_time
        end_hour = end_time.replace(minute=0, second=0, microsecond=0)
        
        while current <= end_hour:
            # Формат: futures/binance/btcusdt/20251204/20_orderbook.gz
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
        timestamp_column: str
    ) -> pl.DataFrame | None:
        """Загружает файлы и фильтрует по времени"""
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
        
        # Объединяем
        combined_df = pl.concat(dfs, how="vertical")
        
        # Фильтруем по времени если есть колонка timestamp
        if timestamp_column in combined_df.columns:
            filtered_df = combined_df.filter(
                (pl.col(timestamp_column) >= start_ms) &
                (pl.col(timestamp_column) <= end_ms)
            )
            return filtered_df
        
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
        timestamp_column: str = "timestamp"
    ) -> MarketData | pl.DataFrame:
        """
        Загрузить данные за временной диапазон.
        
        Args:
            exchange: биржа ("binance", "bybit")
            symbol: символ ("btcusdt", "ethusdt")
            start_time: начало периода (datetime или строка ISO)
            end_time: конец периода (datetime или строка ISO)
            data_type: "orderbook", "trades" или "all" для обоих типов
            market_type: "futures" или "spot"
            cache_dir: директория для кэша
            timestamp_column: название колонки с временной меткой
            
        Returns:
            MarketData с orderbook и trades DataFrame'ами (если data_type="all")
            или один DataFrame (если указан конкретный тип)
            
        Example:
            >>> # Загрузить оба типа данных
            >>> data = s3_client.load_timerange(
            ...     "binance", "btcusdt", 
            ...     "2025-12-04 13:50:00", "2025-12-04 14:25:00"
            ... )
            >>> print(data.orderbook)
            >>> print(data.trades)
            
            >>> # Загрузить только orderbook
            >>> df = s3_client.load_timerange(
            ...     "binance", "btcusdt",
            ...     "2025-12-04 13:50:00", "2025-12-04 14:25:00",
            ...     data_type="orderbook"
            ... )
        """
        # Конвертируем строки в datetime
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time)
        
        print(f"\n🔍 Загрузка данных для {symbol.upper()} на {exchange.upper()}")
        print(f"📅 Период: {start_time} → {end_time}")
        print(f"📊 Тип данных: {data_type}")
        
        # Генерируем список файлов по типам
        keys_by_type = self._generate_hourly_keys(
            exchange, symbol, start_time, end_time, data_type, market_type
        )
        
        total_files = sum(len(keys) for keys in keys_by_type.values())
        print(f"📦 Требуется файлов: {total_files}")
        
        # Подготовка
        cache_dir = Path(cache_dir) if cache_dir else Path(DATA_DIR)
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        # Проверяем локальное наличие для всех файлов
        all_keys = [key for keys in keys_by_type.values() for key in keys]
        missing_locally = [
            key for key in all_keys
            if not (cache_dir / key).exists()
        ]
        
        if missing_locally:
            print(f"📥 Нужно скачать из S3: {len(missing_locally)} файлов")
        else:
            print(f"✅ Все файлы уже есть локально")
        
        # Если запрошен конкретный тип - возвращаем DataFrame
        if data_type != "all":
            keys = keys_by_type[data_type]
            df = self._load_and_filter_files(
                keys, cache_dir, start_ms, end_ms, timestamp_column
            )
            
            if df is None:
                raise ValueError(f"❌ Не удалось загрузить ни одного файла {data_type} для {symbol}")
            
            print(f"\n📈 Загружено {data_type}: {len(df):,} строк")
            return df
        
        # Загружаем все типы данных
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
        """
        Получает список 'папок' на уровне prefix.
        Использует delimiter — не скачивает все объекты.
        """
        prefixes = []
        paginator = self.client.get_paginator("list_objects_v2")
        
        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=prefix,
            Delimiter="/",
        ):
            # CommonPrefixes содержит "папки"
            for cp in page.get("CommonPrefixes", []):
                # Извлекаем имя папки
                # "futures/binance/btcusdt/" -> "btcusdt"
                folder = cp["Prefix"].rstrip("/").split("/")[-1]
                prefixes.append(folder)
        
        return prefixes


# Синглтон
dataManager = DataManager()


# ==================== Примеры использования ====================
if __name__ == "__main__":
    

    

    df_trades = dataManager.load_timerange(
        exchange="binance",
        symbol="cvcusdt",
        start_time="2025-12-05 12:00:00",
        end_time="2025-12-05 12:00:00",
        data_type="all",
        market_type="futures"
    )
    print(f"\nTrades only: {df_trades.orderbook}")
    
    # Пример 4: Одиночный файл (старый формат тоже работает)
    # df = s3_client.load_csv_gz("futures/binance/cvcusdt/20251205/12_orderbook.gz")
    # print(df)