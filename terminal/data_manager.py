import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Literal
from dataclasses import dataclass
import io

from config import S3_CONFIG

# from config import S3_CONFIG, DATA_DIR

DATA_DIR = Path("../store")

@dataclass
class MarketData:
    """Контейнер для данных разных типов"""
    trades:      pl.DataFrame | None = None
    depth:       pl.DataFrame | None = None
    ob_snapshot: pl.DataFrame | None = None

    def __repr__(self) -> str:
        def rows(df): return f"{len(df):,}" if df is not None else "—"
        return (
            f"MarketData("
            f"trades={rows(self.trades)} rows, "
            f"depth={rows(self.depth)} rows, "
            f"ob_snapshot={rows(self.ob_snapshot)} rows)"
        )


def to_utc(dt: datetime | str) -> datetime:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class DataManager:
    """Клиент для работы с S3. Все операции в UTC."""

    # Суффиксы файлов после мерджа (без serverID)
    DATA_TYPES = {
        "trades":      "_trades.parquet",
        "depth":       "_depth.parquet",
        "ob_snapshot": "_ob_snapshot.parquet",
    }

    # Колонки с временной меткой для каждого типа
    TIMESTAMP_COLUMNS = {
        "trades":      "E",
        "depth":       "E",
        "ob_snapshot": "ts",
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
        try:
            self.client.head_object(Bucket=self.bucket, Key=f"{self.prefix}{key}")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def download_bytes(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    # ==================== Загрузка файлов ====================

    def load_parquet(
        self,
        key: str,
        cache_dir: Path | str | None = None,
    ) -> pl.DataFrame:
        """
        Загрузить Parquet файл из S3 (с локальным кэшем).
        Файлы хранятся без сжатия на уровне контейнера — внутреннее zstd
        прозрачно обрабатывается библиотекой.
        """
        cache_dir = Path(cache_dir) if cache_dir else Path(DATA_DIR)
        local_path = cache_dir / key

        if not local_path.exists():
            print(f"📥 Скачиваю из S3: {key}")
            local_path.parent.mkdir(parents=True, exist_ok=True)

            full_key = f"{self.prefix}{key}" if self.prefix else key
            try:
                data = self.download_bytes(full_key)
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    raise FileNotFoundError(f"❌ Файл не найден в S3: {full_key}") from e
                raise

            local_path.write_bytes(data)
            print(f"✅ Сохранён: {local_path}")
        else:
            print(f"📁 Локальный кэш: {local_path}")

        return pl.read_parquet(local_path)

    # ==================== Генерация ключей ====================

    def _generate_hourly_keys(
        self,
        exchange: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        data_type: Literal["trades", "depth", "ob_snapshot", "all"] = "all",
        market_type: Literal["futures", "spot"] = "futures",
    ) -> dict[str, list[str]]:
        """
        Генерирует список ключей для почасовых файлов.

        Структура ключа:
          {exchange}/{market}/{symbol}/{date}/{hour}_{dtype}.parquet
        """
        types_to_load = list(self.DATA_TYPES.keys()) if data_type == "all" else [data_type]
        keys_by_type: dict[str, list[str]] = {t: [] for t in types_to_load}

        current = start_time.replace(minute=0, second=0, microsecond=0)
        end_hour = end_time.replace(minute=0, second=0, microsecond=0)

        while current <= end_hour:
            date_str = current.strftime("%Y%m%d")
            hour_str = f"{current.hour:02d}"
            base = f"{exchange.lower()}/{market_type}/{symbol.lower()}/{date_str}/{hour_str}"

            for dtype in types_to_load:
                keys_by_type[dtype].append(f"{base}{self.DATA_TYPES[dtype]}")

            current += timedelta(hours=1)

        return keys_by_type

    # ==================== Загрузка с фильтрацией ====================

    def _load_and_filter_files(
        self,
        keys: list[str],
        cache_dir: Path,
        start_ms: int,
        end_ms: int,
        timestamp_column: str,
    ) -> pl.DataFrame | None:
        dfs = []
        for key in keys:
            try:
                dfs.append(self.load_parquet(key, cache_dir=cache_dir))
            except FileNotFoundError:
                print(f"⏭️  Пропускаю отсутствующий файл: {key}")
            except Exception as e:
                print(f"❌ Ошибка при загрузке {key}: {e}")

        if not dfs:
            return None

        combined = pl.concat(dfs, how="vertical")

        if timestamp_column in combined.columns:
            filtered = combined.filter(
                (pl.col(timestamp_column) >= start_ms) &
                (pl.col(timestamp_column) <= end_ms)
            ).sort(timestamp_column)
            print(f"   📊 До фильтрации: {len(combined):,}, после: {len(filtered):,}")
            return filtered
        else:
            print(f"   ⚠️ Колонка '{timestamp_column}' не найдена. "
                  f"Доступные: {combined.columns}")
            return combined

    # ==================== Публичный API ====================

    def load_timerange(
        self,
        exchange: str,
        symbol: str,
        start_time: datetime | str,
        end_time: datetime | str,
        data_type: Literal["trades", "depth", "ob_snapshot", "all"] = "all",
        market_type: Literal["futures", "spot"] = "futures",
        cache_dir: Path | str | None = None,
    ) -> MarketData | pl.DataFrame:
        """
        Загрузить данные за временной диапазон.

        ⚠️ Все времена интерпретируются как UTC!

        Args:
            exchange:    биржа ("binance")
            symbol:      символ ("btcusdt")
            start_time:  начало периода UTC (datetime или строка ISO)
            end_time:    конец периода UTC (datetime или строка ISO)
            data_type:   "trades", "depth", "ob_snapshot" или "all"
            market_type: "futures" или "spot"
            cache_dir:   директория для локального кэша

        Returns:
            MarketData если data_type="all", иначе DataFrame

        Example:
            >>> data = dataManager.load_timerange(
            ...     "binance", "btcusdt",
            ...     "2026-03-26 14:00:00",
            ...     "2026-03-26 15:00:00",
            ... )
        """
        start_time = to_utc(start_time)
        end_time   = to_utc(end_time)

        print(f"\n🔍 Загрузка данных для {symbol.upper()} на {exchange.upper()}")
        print(f"📅 Период: {start_time.strftime('%Y-%m-%d %H:%M:%S')} → "
              f"{end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"📊 Тип данных: {data_type}, рынок: {market_type}")

        keys_by_type = self._generate_hourly_keys(
            exchange, symbol, start_time, end_time, data_type, market_type
        )

        total_files = sum(len(v) for v in keys_by_type.values())
        print(f"📦 Файлов для загрузки: {total_files}")

        cache_dir  = Path(cache_dir) if cache_dir else Path(DATA_DIR)
        start_ms   = int(start_time.timestamp() * 1000)
        end_ms     = int(end_time.timestamp() * 1000)

        # Один тип — возвращаем DataFrame
        if data_type != "all":
            ts_col = self.TIMESTAMP_COLUMNS[data_type]
            df = self._load_and_filter_files(
                keys_by_type[data_type], cache_dir, start_ms, end_ms, ts_col
            )
            if df is None:
                raise ValueError(f"❌ Нет данных {data_type} для {symbol} за указанный период")
            print(f"\n📈 Загружено {data_type}: {len(df):,} строк")
            return df

        # "all" — возвращаем MarketData
        result = MarketData()
        for dtype, keys in keys_by_type.items():
            print(f"\n📂 Загружаю {dtype}...")
            ts_col = self.TIMESTAMP_COLUMNS[dtype]
            df = self._load_and_filter_files(keys, cache_dir, start_ms, end_ms, ts_col)
            if df is not None:
                print(f"   ✅ {dtype}: {len(df):,} строк")
                setattr(result, dtype, df)
            else:
                print(f"   ⚠️ {dtype}: нет данных")

        print(f"\n{result}")
        return result

    # ==================== Утилиты ====================

    def _list_prefixes(self, prefix: str) -> list[str]:
        prefixes = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                prefixes.append(cp["Prefix"].rstrip("/").split("/")[-1])
        return prefixes

    def list_available_symbols(
        self,
        exchange: str,
        market_type: Literal["futures", "spot"] = "futures",
    ) -> list[str]:
        """Список доступных символов для биржи"""
        return self._list_prefixes(f"{exchange.lower()}/{market_type}/")

    def list_available_dates(
        self,
        exchange: str,
        symbol: str,
        market_type: Literal["futures", "spot"] = "futures",
    ) -> list[str]:
        """Список доступных дат для символа"""
        return self._list_prefixes(
            f"{exchange.lower()}/{market_type}/{symbol.lower()}/"
        )

    @staticmethod
    def now_utc() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def ms_to_utc(ms: int) -> datetime:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

    @staticmethod
    def utc_to_ms(dt: datetime | str) -> int:
        return int(to_utc(dt).timestamp() * 1000)


dataManager = DataManager()


if __name__ == "__main__":
    # Загрузка всех типов данных
    data = dataManager.load_timerange(
        exchange="binance",
        symbol="0gusdt",
        start_time="2026-03-26 13:00:00",
        end_time="2026-03-26 14:00:00",
        data_type="all",
        market_type="futures",
    )
    print(data)

    # Загрузка только трейдов
    trades = dataManager.load_timerange(
        exchange="binance",
        symbol="riverusdt",
        start_time="2026-03-26 13:00:00",
        end_time="2026-03-26 14:00:00",
        data_type="trades",
    )
    print(f"\n📈 Trades: {trades.shape}")
    print(trades.head())

    # Загрузка снапшотов стакана
    snapshots = dataManager.load_timerange(
        exchange="binance",
        symbol="ethusdt",
        start_time="2026-03-26 13:00:00",
        end_time="2026-03-26 14:00:00",
        data_type="ob_snapshot",
    )
    print(f"\n📸 OB Snapshots: {snapshots.shape}")
    print(snapshots.head())


# df = pl.read_parquet("../store/binance/futures/0gusdt/20260326/13_trades.parquet")
# print(df.head())
# print(df.dtypes)