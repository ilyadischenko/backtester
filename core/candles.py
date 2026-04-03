#!/usr/bin/env python3
"""
cloud_candles.py — свечной график + реконструкция стакана из снапшота + depth updates
Визуализация на Bokeh.

Использование:
    python cloud_candles.py --symbol btcusdt --market futures --date 2026-03-14 --from 6 --to 10
    python cloud_candles.py --symbol btcusdt --market futures --date 2026-03-14 --from 6 --to 10 --interval 5m
    python cloud_candles.py --symbol btcusdt --market futures --date 2026-03-14 --from 6 --to 10 --no-binance
    python cloud_candles.py --symbol btcusdt --market futures --date 2026-03-14 --from 6 --to 10 --no-ob
"""

import argparse
import io
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from bokeh.plotting import figure, output_file, show
from bokeh.models import (
    ColumnDataSource, HoverTool, CrosshairTool,
    LinearColorMapper, ColorBar, BasicTicker,
    RangeTool, Range1d, Div,
)
from bokeh.layouts import column, row, gridplot
from bokeh.palettes import Greens256, Reds256
from bokeh.transform import linear_cmap

sys.path.append(str(Path(__file__).parent))
from bct import ExCloud
from data_manager.cloud_manager import CloudManager


INTERVALS = {
    "1m":  ("1min",  "1m"),
    "3m":  ("3min",  "3m"),
    "5m":  ("5min",  "5m"),
    "15m": ("15min", "15m"),
    "30m": ("30min", "30m"),
    "1h":  ("1h",    "1h"),
    "4h":  ("4h",    "4h"),
}

BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_SPOT_URL    = "https://api.binance.com/api/v3/klines"

# Цвета
C_BG      = "#0d0d0d"
C_GRID    = "#1a1a1a"
C_UP      = "#00d4aa"
C_DOWN    = "#ff4d6d"
C_TEXT    = "#888888"
C_VOL_BUY  = "rgba(0,212,170,0.4)"
C_VOL_SELL = "rgba(255,77,109,0.4)"


# ── загрузка из облака ────────────────────────────────────────────────────────

def download_parquet(cloud: CloudManager, symbol: str, market: str, date: str,
                     hour_from: int, hour_to: int, data_type: str) -> pd.DataFrame:
    frames = []
    for hour in range(hour_from, hour_to + 1):
        hour_str = str(hour).zfill(2)
        key  = cloud._make_key(market=market, symbol=symbol, date=date,
                               hour=hour_str, data_type=data_type)
        print(f"  Скачиваю {key}...")
        data = cloud.download_bytes(key)
        if data is None:
            print(f"  ⚠️  Не найден: {key}")
            continue
        df = pq.read_table(io.BytesIO(data)).to_pandas()
        frames.append(df)
        print(f"  ✅ {len(df):,} строк")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Итого {data_type}: {len(combined):,} строк\n")
    return combined


# ── реконструкция стакана ─────────────────────────────────────────────────────

def reconstruct_orderbook(snap_df: pd.DataFrame, depth_df: pd.DataFrame,
                           interval: str = "1m") -> pd.DataFrame:
    if snap_df.empty:
        print("  ⚠️  Нет снапшотов, реконструкция невозможна")
        return pd.DataFrame()

    freq     = INTERVALS[interval][0]
    snap_df  = snap_df.sort_values("ts").reset_index(drop=True)
    depth_df = depth_df.sort_values("E").reset_index(drop=True) if not depth_df.empty else depth_df

    snap           = snap_df.iloc[0]
    last_update_id = int(snap["lastUpdateId"])

    bids: dict[float, float] = {p: q for p, q in zip(snap["b_p"], snap["b_q"]) if q > 0}
    asks: dict[float, float] = {p: q for p, q in zip(snap["a_p"], snap["a_q"]) if q > 0}

    print(f"  Снапшот: lastUpdateId={last_update_id}, bids={len(bids)}, asks={len(asks)}")

    results: list[dict] = [{
        "time": pd.Timestamp(int(snap["ts"]), unit="ms", tz="UTC"),
        "bids": dict(bids),
        "asks": dict(asks),
    }]

    if not depth_df.empty:
        applied = skipped = 0
        for _, upd in depth_df.iterrows():
            u = int(upd["u"])
            E = int(upd["E"])

            if u <= last_update_id:
                skipped += 1
                continue

            for price, qty in zip(upd["b_p"], upd["b_q"]):
                if qty == 0.0: bids.pop(price, None)
                else:          bids[price] = qty

            for price, qty in zip(upd["a_p"], upd["a_q"]):
                if qty == 0.0: asks.pop(price, None)
                else:          asks[price] = qty

            last_update_id = u
            applied += 1
            results.append({
                "time": pd.Timestamp(E, unit="ms", tz="UTC"),
                "bids": dict(bids),
                "asks": dict(asks),
            })

        print(f"  Применено: {applied} depth updates, пропущено: {skipped}")

    ob_series = pd.DataFrame(results).set_index("time").sort_index()
    time_grid = pd.date_range(
        start=ob_series.index.min().floor(freq),
        end=ob_series.index.max().ceil(freq),
        freq=freq,
    )

    resampled = []
    for t in time_grid:
        mask = ob_series.index <= t
        if not mask.any():
            continue
        row_data = ob_series[mask].iloc[-1]
        resampled.append({"time": t, "bids": row_data["bids"], "asks": row_data["asks"]})

    print(f"  Бинов стакана: {len(resampled)}\n")
    return pd.DataFrame(resampled)


def build_ob_heatmap_data(ob_states: pd.DataFrame):
    """Возвращает данные для двух heatmap (bid/ask) в формате Bokeh image."""
    if ob_states.empty:
        return None

    all_prices: set[float] = set()
    for _, r in ob_states.iterrows():
        all_prices.update(r["bids"].keys())
        all_prices.update(r["asks"].keys())

    if not all_prices:
        return None

    price_grid = np.array(sorted(all_prices))
    time_grid  = ob_states["time"].values
    price_idx  = {p: i for i, p in enumerate(price_grid)}

    n_prices = len(price_grid)
    n_times  = len(time_grid)

    bid_matrix = np.zeros((n_prices, n_times))
    ask_matrix = np.zeros((n_prices, n_times))

    for t_idx, (_, r) in enumerate(ob_states.iterrows()):
        for price, qty in r["bids"].items():
            i = price_idx.get(price)
            if i is not None:
                bid_matrix[i, t_idx] = qty
        for price, qty in r["asks"].items():
            i = price_idx.get(price)
            if i is not None:
                ask_matrix[i, t_idx] = qty

    def log_norm(m):
        m = np.log1p(m)
        mx = m.max()
        return m / mx if mx > 0 else m

    return time_grid, price_grid, log_norm(bid_matrix), log_norm(ask_matrix)


# ── трейды ────────────────────────────────────────────────────────────────────

def prepare_trades(df: pd.DataFrame) -> pd.DataFrame:
    df["time"]  = pd.to_datetime(df["E"], unit="ms", utc=True)
    df["price"] = df["p"].astype(float)
    df["qty"]   = df["q"].astype(str).str.lstrip("-").astype(float)
    df["side"]  = df["q"].astype(str).str.startswith("-").map({True: "sell", False: "buy"})
    return df.sort_values("time")


def build_candles(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    freq = INTERVALS[interval][0]
    df   = df.set_index("time")

    candles             = df["price"].resample(freq).ohlc()
    candles["volume"]   = df["qty"].resample(freq).sum()
    candles["buy_vol"]  = df[df["side"] == "buy"]["qty"].resample(freq).sum().fillna(0)
    candles["sell_vol"] = df[df["side"] == "sell"]["qty"].resample(freq).sum().fillna(0)

    return candles.dropna(subset=["open"]).reset_index()


# ── Binance API ───────────────────────────────────────────────────────────────

def fetch_binance_candles(symbol: str, market: str, date: str,
                           hour_from: int, hour_to: int, interval: str) -> pd.DataFrame:
    binance_interval = INTERVALS[interval][1]
    url = BINANCE_FUTURES_URL if market == "futures" else BINANCE_SPOT_URL

    dt_from  = datetime(int(date[:4]), int(date[5:7]), int(date[8:10]),
                        hour_from, 0, 0, tzinfo=timezone.utc)
    dt_to    = datetime(int(date[:4]), int(date[5:7]), int(date[8:10]),
                        hour_to, 59, 59, tzinfo=timezone.utc)
    start_ms = int(dt_from.timestamp() * 1000)
    end_ms   = int(dt_to.timestamp()   * 1000)

    print(f"Загружаю свечи с Binance API ({market}, {binance_interval})...")
    frames, current = [], start_ms

    while current < end_ms:
        resp = requests.get(url, params={
            "symbol":    symbol.upper(),
            "interval":  binance_interval,
            "startTime": current,
            "endTime":   end_ms,
            "limit":     1000,
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        frames.extend(data)
        current = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(0.1)

    if not frames:
        print("⚠️  Binance API не вернул данных")
        return pd.DataFrame()

    df = pd.DataFrame(frames, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_vol",
        "taker_buy_quote_vol", "ignore",
    ])
    df["time"]     = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "taker_buy_vol"]:
        df[col] = df[col].astype(float)
    df["buy_vol"]  = df["taker_buy_vol"]
    df["sell_vol"] = df["volume"] - df["buy_vol"]

    print(f"✅ {len(df):,} свечей с Binance\n")
    return df.reset_index(drop=True)


# ── тема ──────────────────────────────────────────────────────────────────────

def apply_dark_theme(p):
    p.background_fill_color  = C_BG
    p.border_fill_color      = C_BG
    p.outline_line_color     = C_GRID
    p.title.text_color       = C_TEXT
    p.title.text_font        = "Courier New"
    p.title.text_font_size   = "12px"
    p.xaxis.axis_label_text_color  = C_TEXT
    p.yaxis.axis_label_text_color  = C_TEXT
    p.xaxis.major_label_text_color = C_TEXT
    p.yaxis.major_label_text_color = C_TEXT
    p.xaxis.axis_line_color        = C_GRID
    p.yaxis.axis_line_color        = C_GRID
    p.xaxis.major_tick_line_color  = C_GRID
    p.yaxis.major_tick_line_color  = C_GRID
    p.xaxis.minor_tick_line_color  = None
    p.yaxis.minor_tick_line_color  = None
    p.grid.grid_line_color         = C_GRID
    p.grid.grid_line_alpha         = 0.5
    p.toolbar.logo                 = None


# ── рисование свечей ──────────────────────────────────────────────────────────

def make_candle_panel(candles: pd.DataFrame, title: str,
                      x_range=None, width: int = 900) -> tuple:
    """
    Возвращает (candle_fig, volume_fig) с общим x_range.
    """
    # Bokeh datetime — нужны миллисекунды
    candles = candles.copy()
    candles["time_ms"] = candles["time"].astype(np.int64) // 10**6

    # Ширина свечи = 80% от интервала
    if len(candles) > 1:
        diffs    = candles["time_ms"].diff().dropna()
        bar_w_ms = int(diffs.median() * 0.8)
    else:
        bar_w_ms = 60_000

    up   = candles[candles["close"] >= candles["open"]]
    down = candles[candles["close"] <  candles["open"]]

    x_range_arg = x_range if x_range is not None else (
        candles["time_ms"].min() - bar_w_ms,
        candles["time_ms"].max() + bar_w_ms,
    )

    # ── Свечи ──
    p = figure(
        title=title,
        x_axis_type="datetime",
        x_range=x_range_arg,
        height=400, width=width,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    apply_dark_theme(p)
    p.add_tools(CrosshairTool())

    # Тени
    p.segment(
        x0=up["time_ms"], y0=up["low"],  x1=up["time_ms"], y1=up["high"],
        color=C_UP, line_width=1,
    )
    p.segment(
        x0=down["time_ms"], y0=down["low"], x1=down["time_ms"], y1=down["high"],
        color=C_DOWN, line_width=1,
    )

    # Тела
    src_up   = ColumnDataSource(up)
    src_down = ColumnDataSource(down)

    p.vbar(
        x="time_ms", top="close", bottom="open", width=bar_w_ms,
        source=src_up, fill_color=C_UP, line_color=C_UP,
    )
    p.vbar(
        x="time_ms", top="open", bottom="close", width=bar_w_ms,
        source=src_down, fill_color=C_DOWN, line_color=C_DOWN,
    )

    p.add_tools(HoverTool(tooltips=[
        ("Время",  "@time{%F %T}"),
        ("O",      "@open{0.0000}"),
        ("H",      "@high{0.0000}"),
        ("L",      "@low{0.0000}"),
        ("C",      "@close{0.0000}"),
        ("Объём",  "@volume{0,0.000}"),
    ], formatters={"@time": "datetime"}, renderers=[
        p.vbar(x="time_ms", top="close", bottom="open", width=bar_w_ms,
               source=ColumnDataSource(candles), alpha=0, line_alpha=0),
    ]))

    p.yaxis.axis_label = "Цена"

    # ── Объём ──
    pv = figure(
        x_axis_type="datetime",
        x_range=p.x_range,
        height=120, width=width,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,reset",
    )
    apply_dark_theme(pv)

    src_vol = ColumnDataSource(candles)
    pv.vbar(x="time_ms", top="buy_vol",  width=bar_w_ms,
            source=src_vol, fill_color=C_UP,   line_color=C_UP,   alpha=0.5,
            legend_label="Buy")
    pv.vbar(x="time_ms", top="sell_vol", width=bar_w_ms,
            source=src_vol, fill_color=C_DOWN, line_color=C_DOWN, alpha=0.5,
            legend_label="Sell")

    pv.legend.location              = "top_left"
    pv.legend.background_fill_alpha = 0.3
    pv.legend.label_text_color      = C_TEXT
    pv.legend.border_line_color     = C_GRID
    pv.yaxis.axis_label             = "Объём"

    return p, pv


# ── стакан heatmap ────────────────────────────────────────────────────────────

def make_ob_panel(ob_data, x_range=None, width: int = 900) -> list:
    """
    Рисует heatmap стакана через image — два overlapping image (bid/ask).
    Возвращает список figures.
    """
    time_grid, price_grid, bid_norm, ask_norm = ob_data

    # Конвертируем время в миллисекунды
    times_ms = pd.to_datetime(time_grid).astype(np.int64) // 10**6

    x_start = float(times_ms[0])
    x_end   = float(times_ms[-1])
    y_start = float(price_grid[0])
    y_end   = float(price_grid[-1])
    dw      = x_end - x_start
    dh      = y_end - y_start

    x_range_arg = x_range if x_range is not None else Range1d(x_start, x_end)

    p = figure(
        title="Стакан (реконструкция: снапшот + depth)",
        x_axis_type="datetime",
        x_range=x_range_arg,
        height=400, width=width,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    apply_dark_theme(p)
    p.add_tools(CrosshairTool())

    # Bid heatmap (зелёный)
    bid_mapper = LinearColorMapper(
        palette=["rgba(0,0,0,0)"] + [
            f"rgba(0,{int(80 + 132 * i / 254)},{int(60 + 110 * i / 254)},{0.3 + 0.7 * i / 254:.2f})"
            for i in range(255)
        ],
        low=0.0, high=1.0,
    )
    p.image(
        image=[bid_norm],
        x=x_start, y=y_start, dw=dw, dh=dh,
        color_mapper=bid_mapper,
    )

    # Ask heatmap (красный)
    ask_mapper = LinearColorMapper(
        palette=["rgba(0,0,0,0)"] + [
            f"rgba({int(180 + 75 * i / 254)},{int(20 + 57 * i / 254)},{int(40 + 69 * i / 254)},{0.3 + 0.7 * i / 254:.2f})"
            for i in range(255)
        ],
        low=0.0, high=1.0,
    )
    p.image(
        image=[ask_norm],
        x=x_start, y=y_start, dw=dw, dh=dh,
        color_mapper=ask_mapper,
    )

    p.yaxis.axis_label = "Цена"
    p.xaxis.axis_label = "Время"

    return p


# ── финальная сборка ──────────────────────────────────────────────────────────

def plot(our: pd.DataFrame, binance: pd.DataFrame, ob_data, title: str):
    output_file("candles.html", title=title)

    # Общий x_range — берём из наших данных
    our_ms   = our["time"].astype(np.int64) // 10**6
    x_range  = Range1d(
        start=float(our_ms.min()) - 60_000,
        end=float(our_ms.max())   + 60_000,
    )

    panels = []

    # ── Наши свечи ──
    p_our, pv_our = make_candle_panel(our, "Наши данные", x_range=x_range)
    panels.append(p_our)
    panels.append(pv_our)

    # ── Binance свечи ──
    if not binance.empty:
        p_bin, pv_bin = make_candle_panel(binance, "Binance API", x_range=x_range)
        panels.append(p_bin)
        panels.append(pv_bin)

    # ── Стакан ──
    if ob_data is not None:
        p_ob = make_ob_panel(ob_data, x_range=x_range)
        panels.append(p_ob)

    layout = column(*panels, sizing_mode="stretch_width")
    out_path = Path("candles.html")
    print(f"Сохранено → {out_path.resolve()}")
    show(layout)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",     required=True)
    parser.add_argument("--market",     required=True, help="futures / spot")
    parser.add_argument("--date",       required=True, help="2026-03-14")
    parser.add_argument("--from",       dest="hour_from", required=True, type=int)
    parser.add_argument("--to",         dest="hour_to",   required=True, type=int)
    parser.add_argument("--interval",   default="1m", choices=INTERVALS.keys())
    parser.add_argument("--no-binance", action="store_true")
    parser.add_argument("--no-ob",      action="store_true")
    args = parser.parse_args()

    print(f"\n{args.symbol.upper()} · {args.market} · {args.date} "
          f"{args.hour_from:02d}:00–{args.hour_to:02d}:59 · {args.interval}\n")

    cloud = ExCloud()

    # ── Трейды ───────────────────────────────────────────────────────────────
    print("── Трейды ──────────────────────────────")
    trades_df = download_parquet(cloud, args.symbol, args.market, args.date,
                                 args.hour_from, args.hour_to, "trades")
    if trades_df.empty:
        print("Нет данных трейдов")
        sys.exit(1)

    trades_df   = prepare_trades(trades_df)
    our_candles = build_candles(trades_df, args.interval)
    print(f"Свечей: {len(our_candles):,}\n")

    # ── Binance ───────────────────────────────────────────────────────────────
    binance_candles = pd.DataFrame()
    if not args.no_binance:
        binance_candles = fetch_binance_candles(
            args.symbol, args.market, args.date,
            args.hour_from, args.hour_to, args.interval,
        )

    # ── Стакан ───────────────────────────────────────────────────────────────
    ob_data = None
    if not args.no_ob:
        print("── Стакан ──────────────────────────────")

        print("  Снапшоты:")
        snap_df = download_parquet(cloud, args.symbol, args.market, args.date,
                                   args.hour_from, args.hour_to, "ob_snapshot")

        print("  Depth updates:")
        depth_df = download_parquet(cloud, args.symbol, args.market, args.date,
                                    args.hour_from, args.hour_to, "depth")

        if not snap_df.empty:
            print("  Реконструирую стакан...")
            ob_states = reconstruct_orderbook(snap_df, depth_df, args.interval)
            if not ob_states.empty:
                print("  Строю heatmap...")
                ob_data = build_ob_heatmap_data(ob_states)
                if ob_data:
                    _, price_grid, _, _ = ob_data
                    print(f"  ✅ Heatmap: {len(price_grid)} ценовых уровней\n")

    title = (f"{args.symbol.upper()} · {args.market} · {args.date} "
             f"{args.hour_from:02d}:00–{args.hour_to:02d}:59 · {args.interval}")
    plot(our_candles, binance_candles, ob_data, title)


if __name__ == "__main__":
    main()