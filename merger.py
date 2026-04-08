#!/usr/bin/env python3
"""
Мерджер S3: находит файлы за указанный день и рынок,
скачивает, мержит, дедуплицирует, загружает финальную версию.

nohup python m.py > log.txt 2>&1 &

Структура ключей в S3:
  {exchange}/{market}/{symbol}/{date}/{hour}_{datatype}_{serverID}.parquet

Пример:
  binance/futures/btcusdt/20260324/14_trades_1.parquet
  binance/futures/btcusdt/20260324/14_trades_2.parquet
  → binance/futures/btcusdt/20260324/14_trades.parquet

Переменные окружения:
  S3_ACCESS_KEY   — ключ доступа
  S3_SECRET_KEY   — секретный ключ
  S3_BUCKET       — имя бакета
  S3_EXCHANGE     — биржа (default: binance)
  MERGE_MARKET    — рынок: futures или spot (обязательно)
  MERGE_DATE      — дата в формате YYYYMMDD (обязательно)
  DRY_RUN         — 1 = только логировать, не загружать
"""

import os
import re
import sys
import tempfile
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from botocore.client import Config

# ── Настройки ─────────────────────────────────────────────────────────────────

ENDPOINT   = "https://s3.ru1.storage.beget.cloud"
REGION     = "ru-central1"
BUCKET     = os.environ.get("S3_BUCKET",     "34bfe174605d-temp-hft-data")
ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "S2AX0NGD22Y0TR80SUJ9")
SECRET_KEY = os.environ.get("S3_SECRET_KEY", "NWYz4HqBJ6PObIWe3sPbQi4GfwqCtMxKJxawYKYk")
EXCHANGE   = os.environ.get("S3_EXCHANGE",   "binance")
MARKET     = os.environ.get("MERGE_MARKET",  "futures")
DATE       = os.environ.get("MERGE_DATE",    "20260402")
DRY_RUN    = os.environ.get("DRY_RUN", "0") == "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── S3 клиент ─────────────────────────────────────────────────────────────────

def make_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(
            signature_version="s3v4",
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )

def list_prefix(client, prefix):
    keys = []
    kwargs = {"Bucket": BUCKET, "Prefix": prefix}
    while True:
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            keys.append(obj["Key"])
        if resp.get("IsTruncated"):
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        else:
            break
    return keys

def list_subdirs(client, prefix):
    prefixes = []
    kwargs = {"Bucket": BUCKET, "Prefix": prefix, "Delimiter": "/"}
    while True:
        resp = client.list_objects_v2(**kwargs)
        for cp in resp.get("CommonPrefixes", []):
            prefixes.append(cp["Prefix"])
        if resp.get("IsTruncated"):
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        else:
            break
    return prefixes

def list_for_date(exchange, market, date):
    client = make_client()
    market_prefix = f"{exchange}/{market}/"
    symbol_prefixes = list_subdirs(client, market_prefix)
    log.info("Найдено символов: %d", len(symbol_prefixes))

    if not symbol_prefixes:
        log.warning("Нет символов для %s", market_prefix)
        return []

    def fetch(sym_prefix):
        c = make_client()
        return list_prefix(c, f"{sym_prefix}{date}/")

    all_keys = []
    workers = min(32, len(symbol_prefixes))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch, sp): sp for sp in symbol_prefixes}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 100 == 0:
                log.info("  листинг: %d/%d символов", done, len(symbol_prefixes))
            try:
                all_keys.extend(fut.result())
            except Exception as e:
                log.error("  ошибка листинга %s: %s", futures[fut], e)

    return all_keys

# ── Группировка ───────────────────────────────────────────────────────────────

KEY_RE = re.compile(
    r"^(?P<exchange>[^/]+)/(?P<market>[^/]+)/(?P<symbol>[^/]+)/"
    r"(?P<date>\d{8})/(?P<hour>\d{2})_(?P<dtype>trades|depth|ob_snapshot)_.+\.parquet$"
)

def group_keys(keys):
    groups = defaultdict(list)
    for key in keys:
        m = KEY_RE.match(key)
        if not m:
            continue
        g = m.group
        gid = (g("exchange"), g("market"), g("symbol"), g("date"), g("hour"), g("dtype"))
        groups[gid].append(key)
    return groups

def make_final_key(exchange, market, symbol, date, hour, dtype):
    return f"{exchange}/{market}/{symbol}/{date}/{hour}_{dtype}.parquet"

# ── S3 операции ───────────────────────────────────────────────────────────────

def download(client, key, path):
    client.download_file(BUCKET, key, str(path))

def upload(client, key, path):
    client.upload_file(str(path), BUCKET, key)

def delete_batch(client, keys):
    if not keys:
        return
    client.delete_objects(
        Bucket=BUCKET,
        Delete={"Objects": [{"Key": k} for k in keys]},
    )

# ── Мерж и дедупликация (только pyarrow, без pandas) ─────────────────────────

def merge_trades(tables):
    combined = pa.concat_tables(tables)
    sort_idx = pc.sort_indices(combined, sort_keys=[("t", "ascending")])
    combined = combined.take(sort_idx)
    t_vals = combined.column("t").to_pylist()
    seen, keep = set(), []
    for i, v in enumerate(t_vals):
        if v not in seen:
            seen.add(v)
            keep.append(i)
    return combined.take(keep)

def merge_depth(tables):
    combined = pa.concat_tables(tables)
    sort_idx = pc.sort_indices(combined, sort_keys=[("E", "ascending")])
    combined = combined.take(sort_idx)
    e  = combined.column("E").to_pylist()
    u  = combined.column("U").to_pylist()
    u2 = combined.column("u").to_pylist()
    seen, keep = set(), []
    for i, (ev, uv, u2v) in enumerate(zip(e, u, u2)):
        k = (ev, uv, u2v)
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return combined.take(keep)

def merge_ob_snapshot(tables):
    combined = pa.concat_tables(tables)
    sort_idx = pc.sort_indices(combined, sort_keys=[("ts", "ascending")])
    return combined.take(sort_idx)

MERGERS = {
    "trades":      merge_trades,
    "depth":       merge_depth,
    "ob_snapshot": merge_ob_snapshot,
}

# ── Мерж одной группы ─────────────────────────────────────────────────────────

def merge_group(group_id, keys, tmpdir):
    exchange, market, symbol, date, hour, dtype = group_id
    label = f"{market}/{symbol}/{date}/{hour}_{dtype}"

    if dtype not in MERGERS:
        log.warning("Неизвестный тип: %s, пропускаем", dtype)
        return

    fkey = make_final_key(exchange, market, symbol, date, hour, dtype)
    client = make_client()

    # Синхронное скачивание
    local_paths = []
    for key in keys:
        local = tmpdir / Path(key).name
        try:
            download(client, key, local)
            local_paths.append(local)
        except Exception as e:
            log.error("  ошибка скачивания %s: %s", key, e)

    # Читаем как pyarrow Tables
    tables = []
    for path in local_paths:
        try:
            tables.append(pq.read_table(str(path)))
        except Exception as e:
            log.error("  ошибка чтения %s: %s", path.name, e)

    # Чистим скачанные файлы сразу после чтения
    for p in local_paths:
        p.unlink(missing_ok=True)

    if not tables:
        log.error("  нет данных для %s, пропускаем", label)
        return

    # Мержим
    merged = MERGERS[dtype](tables)
    tables.clear()  # освобождаем память
    log.info("  %s: %d строк из %d файлов", label, len(merged), len(keys))

    if DRY_RUN:
        log.info("  [DRY_RUN] пропускаем загрузку %s", fkey)
        return

    # Записываем финальный parquet
    out = tmpdir / f"merged_{symbol}_{hour}_{dtype}.parquet"
    pq.write_table(merged, str(out), compression="zstd", compression_level=12)
    del merged  # освобождаем память до загрузки

    # Загружаем
    log.info("  загружаем → %s", fkey)
    try:
        upload(client, fkey, out)
    except Exception as e:
        log.error("  ошибка загрузки %s: %s", fkey, e)
        out.unlink(missing_ok=True)
        return  # не удаляем исходники если upload упал

    # Удаляем исходники
    delete_batch(client, keys)
    log.info("  ✅ %s", label)

    out.unlink(missing_ok=True)

# ── Точка входа ───────────────────────────────────────────────────────────────

def main():
    errors = []
    if not ACCESS_KEY:
        errors.append("S3_ACCESS_KEY")
    if not SECRET_KEY:
        errors.append("S3_SECRET_KEY")
    if not DATE:
        errors.append("MERGE_DATE (например 20260324)")
    if not MARKET:
        errors.append("MERGE_MARKET (futures или spot)")
    if errors:
        log.error("Не заданы переменные окружения: %s", ", ".join(errors))
        sys.exit(1)

    log.info("Старт: exchange=%s market=%s date=%s dry_run=%s",
             EXCHANGE, MARKET, DATE, DRY_RUN)

    keys = list_for_date(EXCHANGE, MARKET, DATE)
    log.info("Найдено ключей за %s: %d", DATE, len(keys))

    groups = group_keys(keys)
    log.info("Групп всего: %d", len(groups))

    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    log.info("Групп с дублями (≥2): %d", len(groups))

    if not groups:
        log.info("Нечего мержить, выходим")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        total = len(groups)
        workers = min(4, total)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(merge_group, gid, ks, tmpdir): gid
                    for gid, ks in groups.items()}
            done = 0
            for fut in as_completed(futs):
                done += 1
                gid = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    log.error("ошибка мержа %s: %s", gid, e)
                if done % 50 == 0:
                    log.info("прогресс: %d/%d групп", done, total)

    log.info("Готово. Обработано %d групп", total)

if __name__ == "__main__":
    main()