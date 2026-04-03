#!/usr/bin/env python3
"""
ob_trades_chart.py — Depth heatmap + trades (▲▼) via PyQtGraph.

Использование:
    python ob_trades_chart.py \
        --exchange binance --symbol btcusdt --market futures \
        --start "2026-03-14 06:00:00" --end "2026-03-14 06:15:00" \
        --levels 60 --price-step 1.0 --bucket-ms 100 --min-qty 0.5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QSlider, QSizePolicy,
)
from PyQt6.QtCore import Qt
import pyqtgraph as pg

sys.path.append(str(Path(__file__).parent))
from data.data_manager import dataManager


# ─── Палитра ──────────────────────────────────────────────────────────────────

C_BG   = "#0d0d0d"
C_GRID = "#1a1a1a"
C_TEXT = "#666666"
C_MID  = "#f5c542"
C_BUY  = "#00d4aa"
C_SELL = "#ff4d6d"

pg.setConfigOptions(antialias=False, useOpenGL=True)


# ─── Tick-by-tick реконструкция ───────────────────────────────────────────────

def reconstruct_tick_by_tick(
    snap_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    bucket_ms: int = 0,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    if hasattr(snap_df, "to_pandas"):
        snap_df = snap_df.to_pandas()
    if hasattr(depth_df, "to_pandas"):
        depth_df = depth_df.to_pandas()

    if snap_df.empty:
        raise ValueError("Нет снапшотов")

    snap_df  = snap_df.sort_values("ts").reset_index(drop=True)
    snap     = snap_df.iloc[0]
    last_uid = int(snap["lastUpdateId"])

    bids: dict[float, float] = {p: q for p, q in zip(snap["b_p"], snap["b_q"]) if q > 0}
    asks: dict[float, float] = {p: q for p, q in zip(snap["a_p"], snap["a_q"]) if q > 0}
    print(f"  Снапшот uid={last_uid}, bids={len(bids)}, asks={len(asks)}")

    if depth_df.empty:
        t = int(snap["ts"])
        return np.array([t], dtype=np.int64), [dict(bids)], [dict(asks)]

    depth_df = depth_df[depth_df["u"] > last_uid].sort_values("E").reset_index(drop=True)
    print(f"  Depth апдейтов: {len(depth_df)}")

    times_ms: list[int]  = []
    bid_cols: list[dict] = []
    ask_cols: list[dict] = []

    if bucket_ms <= 0:
        for _, upd in depth_df.iterrows():
            _apply_update(bids, asks, upd)
            times_ms.append(int(upd["E"]))
            bid_cols.append(dict(bids))
            ask_cols.append(dict(asks))
    else:
        t0         = int(depth_df["E"].iloc[0])
        t_end      = int(depth_df["E"].iloc[-1])
        n_buckets  = (t_end - t0) // bucket_ms + 2
        boundaries = [t0 + i * bucket_ms for i in range(n_buckets)]
        b_ptr = 0

        for _, upd in depth_df.iterrows():
            E = int(upd["E"])
            while b_ptr + 1 < len(boundaries) and boundaries[b_ptr + 1] <= E:
                b_ptr += 1
            _apply_update(bids, asks, upd)
            bucket_t = boundaries[b_ptr]
            if times_ms and times_ms[-1] == bucket_t:
                bid_cols[-1] = dict(bids)
                ask_cols[-1] = dict(asks)
            else:
                times_ms.append(bucket_t)
                bid_cols.append(dict(bids))
                ask_cols.append(dict(asks))

    print(f"  Столбцов heatmap: {len(times_ms)}"
          + (f" (bucket={bucket_ms}ms)" if bucket_ms > 0 else " (tick-by-tick)"))
    return np.array(times_ms, dtype=np.int64), bid_cols, ask_cols


def _apply_update(bids: dict, asks: dict, upd) -> None:
    for price, qty in zip(upd["b_p"], upd["b_q"]):
        if qty == 0.0:
            bids.pop(price, None)
        else:
            bids[price] = qty
    for price, qty in zip(upd["a_p"], upd["a_q"]):
        if qty == 0.0:
            asks.pop(price, None)
        else:
            asks[price] = qty


# ─── Построение матриц ────────────────────────────────────────────────────────

def build_matrices(
    times_ms: np.ndarray,
    bid_cols: list[dict],
    ask_cols: list[dict],
) -> dict:
    """
    Ось Y = все уникальные цены из всех bid/ask апдейтов.
    Никакого price_step, никакого max_levels.
    """
    # Собираем все уникальные цены
    all_prices: set[float] = set()
    for col in bid_cols:
        all_prices.update(col.keys())
    for col in ask_cols:
        all_prices.update(col.keys())

    if not all_prices:
        return {}

    price_grid = np.array(sorted(all_prices), dtype=np.float64)
    price_idx  = {p: i for i, p in enumerate(price_grid)}
    n_prices   = len(price_grid)
    n_times    = len(times_ms)

    print(f"  Уникальных price-levels: {n_prices}")
    print(f"  Диапазон: {price_grid[0]:.4f} … {price_grid[-1]:.4f}")

    bid_mat = np.zeros((n_times, n_prices), dtype=np.float32)
    ask_mat = np.zeros((n_times, n_prices), dtype=np.float32)

    for t_idx, (b_col, a_col) in enumerate(zip(bid_cols, ask_cols)):
        for price, qty in b_col.items():
            i = price_idx.get(price)
            if i is not None:
                bid_mat[t_idx, i] = qty
        for price, qty in a_col.items():
            i = price_idx.get(price)
            if i is not None:
                ask_mat[t_idx, i] = qty

    def log_norm(m: np.ndarray) -> np.ndarray:
        m = np.log1p(m)
        mx = m.max()
        return (m / mx).astype(np.float32) if mx > 0 else m

    mid_idxs, mid_prices = [], []
    for t_idx, (b, a) in enumerate(zip(bid_cols, ask_cols)):
        if b and a:
            mid_idxs.append(float(t_idx))
            mid_prices.append((max(b.keys()) + min(a.keys())) / 2)

    pmin = float(price_grid[0])
    pmax = float(price_grid[-1])
    # Средний шаг для ImageItem (он рендерит равномерно)
    y_step = (pmax - pmin) / (n_prices - 1) if n_prices > 1 else 1.0

    return {
        "times_ms":   times_ms,
        "price_grid": price_grid,
        "price_step": y_step,   # только для отступа трейдов
        "bid_mat":    log_norm(bid_mat),
        "ask_mat":    log_norm(ask_mat),
        "mid_idxs":   mid_idxs,
        "mid_prices": mid_prices,
        "pmin":       pmin,
        "pmax":       pmax,
        "n_times":    n_times,
        "n_prices":   n_prices,
    }

# ─── RGBA-кубы: чёрный → цвет ────────────────────────────────────────────────

def _make_rgba(
    mat: np.ndarray,
    r: int, g: int, b: int,
) -> np.ndarray:
    """
    mat: float32 (n_times, n_prices), значения 0..1
    0   → (0, 0, 0, 255)        полностью чёрный, непрозрачный
    1.0 → (r, g, b, 255)        полный цвет
    Возвращает uint8 (n_times, n_prices, 4) RGBA.
    """
    rgba = np.zeros((*mat.shape, 4), dtype=np.uint8)
    rgba[..., 0] = (mat * r).clip(0, 255).astype(np.uint8)
    rgba[..., 1] = (mat * g).clip(0, 255).astype(np.uint8)
    rgba[..., 2] = (mat * b).clip(0, 255).astype(np.uint8)
    rgba[..., 3] = 255  # всегда непрозрачный — чёрный фон через цвет
    return rgba


# ─── Кастомный AxisItem: индекс колонки → datetime ───────────────────────────

class TimeAxisItem(pg.AxisItem):
    def __init__(self, times_ms: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._times = times_ms

    def tickStrings(self, values, scale, spacing):
        result = []
        n = len(self._times)
        for v in values:
            idx = int(round(v))
            if 0 <= idx < n:
                ts = pd.Timestamp(self._times[idx], unit="ms", tz="UTC")
                result.append(ts.strftime("%H:%M:%S.%f")[:-3])
            else:
                result.append("")
        return result


# ─── Трейды ───────────────────────────────────────────────────────────────────

def prepare_trades(df, times_ms: np.ndarray) -> pd.DataFrame:
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    df = df.copy()
    df["time_ms"] = df["E"].astype(np.int64)
    df["price"]   = df["p"].astype(float)
    raw           = df["q"].astype(str)
    df["qty"]     = raw.str.lstrip("-").astype(float)
    df["side"]    = raw.str.startswith("-").map({True: "sell", False: "buy"})
    df = df.sort_values("time_ms").reset_index(drop=True)

    idxs = np.searchsorted(times_ms, df["time_ms"].values, side="left")
    idxs = idxs.clip(0, len(times_ms) - 1)
    df["col_idx"] = idxs.astype(float)
    return df


# ─── Главное окно ─────────────────────────────────────────────────────────────

class OBWindow(QMainWindow):
    def __init__(self, hm: dict, trades: pd.DataFrame,
                 title: str, min_qty: float = 0.0):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1600, 800)
        self._hm      = hm
        self._trades  = trades
        self._min_qty = min_qty

        self._build_ui()
        self._plot_heatmap()
        self._plot_mid()
        self._plot_trades()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        central.setStyleSheet(f"background:{C_BG};")

        hm = self._hm
        header = QLabel(
            f"  {self.windowTitle()}"
            f"  │  cols: {hm['n_times']:,}"
            f"  │  price_step: {hm['price_step']}"
            f"  │  levels: {hm['n_prices']}"
            f"  │  trades: {len(self._trades):,}"
        )
        header.setStyleSheet(
            f"color:{C_TEXT}; background:{C_BG}; font-family:monospace;"
            f"font-size:11px; padding:6px 10px;"
            f"border-bottom:1px solid {C_GRID};"
        )
        layout.addWidget(header)

        # Слайдеры яркости
        ctrl = QWidget()
        ctrl.setStyleSheet(f"background:{C_BG};")
        ctrl_layout = QHBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(10, 4, 10, 4)
        self._bid_slider = self._make_slider("Bid", C_BUY,  ctrl_layout, self._on_bid)
        self._ask_slider = self._make_slider("Ask", C_SELL, ctrl_layout, self._on_ask)
        ctrl_layout.addStretch()
        layout.addWidget(ctrl)

        # График
        times_ms = hm["times_ms"]
        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setBackground(C_BG)
        self._gw.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._plot = self._gw.addPlot(
            axisItems={"bottom": TimeAxisItem(times_ms, orientation="bottom")}
        )
        self._style_plot(self._plot)
        layout.addWidget(self._gw)

    def _make_slider(self, label, color, parent_layout, callback):
        lbl = QLabel(f"{label} brightness")
        lbl.setStyleSheet(
            f"color:{color}; font-family:monospace; font-size:11px; margin-right:4px;"
        )
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(1, 100)
        sl.setValue(80)
        sl.setFixedWidth(200)
        sl.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height:4px; background:{C_GRID}; border-radius:2px;
            }}
            QSlider::handle:horizontal {{
                width:12px; height:12px; margin:-4px 0;
                background:{color}; border-radius:6px;
            }}
            QSlider::sub-page:horizontal {{
                background:{color}; border-radius:2px;
            }}
        """)
        sl.valueChanged.connect(callback)
        parent_layout.addWidget(lbl)
        parent_layout.addWidget(sl)
        parent_layout.addSpacing(20)
        return sl

    def _style_plot(self, p: pg.PlotItem):
        p.showGrid(x=True, y=True, alpha=0.15)
        p.getAxis("bottom").setTextPen(pg.mkPen(C_TEXT))
        p.getAxis("left").setTextPen(pg.mkPen(C_TEXT))
        p.getAxis("bottom").setPen(pg.mkPen(C_GRID))
        p.getAxis("left").setPen(pg.mkPen(C_GRID))
        p.getViewBox().setBackgroundColor(C_BG)

        vline = pg.InfiniteLine(angle=90, movable=False,
                                pen=pg.mkPen("#2a2a2a", width=1))
        hline = pg.InfiniteLine(angle=0,  movable=False,
                                pen=pg.mkPen("#2a2a2a", width=1))
        p.addItem(vline, ignoreBounds=True)
        p.addItem(hline, ignoreBounds=True)

        def on_mouse(pos):
            if p.sceneBoundingRect().contains(pos):
                mp = p.vb.mapSceneToView(pos)
                vline.setPos(mp.x())
                hline.setPos(mp.y())

        self._gw.scene().sigMouseMoved.connect(on_mouse)
        p.setLabel("bottom", "Время")
        p.setLabel("left",   "Цена")

    # ── Heatmap ───────────────────────────────────────────────────────────────

    def _rect(self):
        hm = self._hm
        return pg.QtCore.QRectF(
            0,           hm["pmin"],
            hm["n_times"], hm["n_prices"] * hm["price_step"],
        )

    def _plot_heatmap(self):
        # Два ImageItem: bid поверх ask, оба непрозрачные (чёрный = 0 объём).
        # Bid рисуется ПОСЛЕ ask → перекрывает там, где оба ненулевые
        # (спред — только один из двух ненулевой, поэтому коллизий нет).
        self._ask_img = pg.ImageItem()
        self._bid_img = pg.ImageItem()

        self._plot.addItem(self._ask_img)
        self._plot.addItem(self._bid_img)

        rect = self._rect()
        self._ask_img.setRect(rect)
        self._bid_img.setRect(rect)

        # Bid поверх — используем CompositionMode_SourceOver
        # (пиксели bid c alpha=255 перекроют ask только там где bid > 0).
        # Чтобы bid не закрывал чёрным нулевые ask-ячейки,
        # делаем нулевые bid-ячейки прозрачными.
        self._apply_bid(self._bid_slider.value())
        self._apply_ask(self._ask_slider.value())

    def _bid_rgba(self, brightness: float) -> np.ndarray:
        """brightness: 0..1. Нулевые ячейки — прозрачные."""
        mat  = self._hm["bid_mat"]
        rgba = np.zeros((*mat.shape, 4), dtype=np.uint8)
        scaled = mat * brightness
        rgba[..., 1] = (scaled * 180).clip(0, 255).astype(np.uint8)   # G
        rgba[..., 3] = np.where(mat > 1e-6, 255, 0).astype(np.uint8)  # A
        return rgba

    def _ask_rgba(self, brightness: float) -> np.ndarray:
        mat  = self._hm["ask_mat"]
        rgba = np.zeros((*mat.shape, 4), dtype=np.uint8)
        scaled = mat * brightness
        rgba[..., 0] = (scaled * 220).clip(0, 255).astype(np.uint8)   # R
        rgba[..., 3] = np.where(mat > 1e-6, 255, 0).astype(np.uint8)  # A
        return rgba

    def _apply_bid(self, v: int):
        rgba = self._bid_rgba(v / 100.0)
        self._bid_img.setImage(rgba, autoLevels=False)

    def _apply_ask(self, v: int):
        rgba = self._ask_rgba(v / 100.0)
        self._ask_img.setImage(rgba, autoLevels=False)

    def _on_bid(self, v: int): self._apply_bid(v)
    def _on_ask(self, v: int): self._apply_ask(v)

    # ── Mid-price ─────────────────────────────────────────────────────────────

    def _plot_mid(self):
        hm = self._hm
        if not hm["mid_idxs"]:
            return
        self._plot.plot(
            x=hm["mid_idxs"],
            y=hm["mid_prices"],
            pen=pg.mkPen(C_MID, width=1),
            name="Mid",
        )

    # ── Трейды ────────────────────────────────────────────────────────────────

    def _plot_trades(self):
        hm = self._hm
        t  = self._trades
        if t.empty:
            return
        if self._min_qty > 0:
            t = t[t["qty"] >= self._min_qty]
        if t.empty:
            return

        step     = hm["price_step"]
        qty_log  = np.log1p(t["qty"].values)
        qty_max  = qty_log.max() if qty_log.max() > 0 else 1.0
        sizes    = (6 + 14 * qty_log / qty_max).clip(6, 20)

        buys  = t[t["side"] == "buy"]
        sells = t[t["side"] == "sell"]

        if not buys.empty:
            sz_b = sizes[t["side"] == "buy"]
            sc = pg.ScatterPlotItem()
            sc.addPoints([
                {"pos": (x, y - step * 2), "size": s,
                 "symbol": "t1",
                 "pen": pg.mkPen(None),
                 "brush": pg.mkBrush(C_BUY)}
                for x, y, s in zip(buys["col_idx"], buys["price"], sz_b)
            ])
            self._plot.addItem(sc)
            print(f"  Buy trades: {len(buys):,}")

        if not sells.empty:
            sz_s = sizes[t["side"] == "sell"]
            sc = pg.ScatterPlotItem()
            sc.addPoints([
                {"pos": (x, y + step * 2), "size": s,
                 "symbol": "t",
                 "pen": pg.mkPen(None),
                 "brush": pg.mkBrush(C_SELL)}
                for x, y, s in zip(sells["col_idx"], sells["price"], sz_s)
            ])
            self._plot.addItem(sc)
            print(f"  Sell trades: {len(sells):,}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange",    default="binance")
    ap.add_argument("--symbol",      required=True)
    ap.add_argument("--market",      default="futures")
    ap.add_argument("--start",       required=True)
    ap.add_argument("--end",         required=True)
    ap.add_argument("--levels",      default=60,  type=int,
                    help="Уровней выше и ниже mid")
    ap.add_argument("--price-step",  default=1.0, type=float,
                    help="Шаг ценовой сетки (напр. 0.1 для BTC)")
    ap.add_argument("--bucket-ms",   default=0,   type=int,
                    help="0 = tick-by-tick, >0 = батч в мс")
    ap.add_argument("--min-qty",     default=0.0, type=float)
    args = ap.parse_args()

    print(f"\n{'─'*62}")
    print(f"  {args.symbol.upper()} · {args.market}")
    print(f"  {args.start}  →  {args.end}")
    print(f"  levels={args.levels}  price_step={args.price_step}"
          f"  bucket_ms={args.bucket_ms}  min_qty={args.min_qty}")
    print(f"{'─'*62}\n")

    print("── Загрузка ─────────────────────────────────────────────")
    data = dataManager.load_timerange(
        exchange=args.exchange, symbol=args.symbol,
        start_time=args.start,  end_time=args.end,
        data_type="all",        market_type=args.market,
    )
    if data.trades is None:
        print("❌ Нет трейдов"); sys.exit(1)
    if data.ob_snapshot is None:
        print("❌ Нет снапшотов"); sys.exit(1)

    print("\n── Tick-by-tick стакан ──────────────────────────────────")
    snap_df  = data.ob_snapshot
    depth_df = data.depth if data.depth is not None else pd.DataFrame()

    times_ms, bid_cols, ask_cols = reconstruct_tick_by_tick(
        snap_df, depth_df, bucket_ms=args.bucket_ms
    )

    print("\n── Матрицы heatmap ──────────────────────────────────────")
    hm = build_matrices(
        times_ms, bid_cols, ask_cols,
    )
    if not hm:
        print("❌ Не удалось построить матрицы"); sys.exit(1)

    print("\n── Трейды ───────────────────────────────────────────────")
    trades = prepare_trades(data.trades, times_ms)
    print(f"  Всего: {len(trades):,}")

    title = (f"{args.symbol.upper()} · {args.market} · "
             f"{args.start} → {args.end}")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = OBWindow(hm, trades, title=title, min_qty=args.min_qty)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()