import sys
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout,
    QLineEdit, QDateTimeEdit, QPushButton, QLabel, QCheckBox
)
from PyQt6.QtCore import QDateTime, QThread, pyqtSignal, QTimer
from numba import njit

from logic.figure_heatmap import LiquidityRenderer, build_segments
from data_manager import DataManager

data_manager = DataManager()


# ══════════════════════════════════════════════════════════════════════════════
# TRADES DOWNSAMPLE
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _downsample_trades(ts, price, is_sell, max_points):
    n = len(ts)
    if n <= max_points:
        return ts, price, is_sell

    bucket = n // max_points
    ts_out    = np.empty(max_points, dtype=ts.dtype)
    price_out = np.empty(max_points, dtype=price.dtype)
    sell_out  = np.empty(max_points, dtype=np.bool_)

    for i in range(max_points):
        start = i * bucket
        end   = start + bucket
        mid   = start + bucket // 2

        ts_out[i] = ts[mid]

        bucket_prices = price[start:end].copy()
        bucket_prices.sort()
        price_out[i] = bucket_prices[bucket // 2]

        sell_count = 0
        for j in range(start, end):
            if is_sell[j]:
                sell_count += 1
        sell_out[i] = sell_count >= bucket // 2

    return ts_out, price_out, sell_out


def downsample(ts, price, is_sell, max_points=3000):
    if len(ts) == 0:
        return ts, price, is_sell
    return _downsample_trades(ts, price, is_sell, max_points)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

class DataLoader(QThread):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, symbol, start_time, end_time):
        super().__init__()
        self.symbol     = symbol
        self.start_time = start_time
        self.end_time   = end_time

    def run(self):
        try:
            self.progress.emit("Загрузка трейдов...")
            market_data = data_manager.load_timerange(
                exchange="binance", symbol=self.symbol,
                start_time=self.start_time, end_time=self.end_time,
            )  # возвращает MarketData

            trades_df = market_data.trades
            depth_df  = market_data.depth

            trades_cache = {
                "ts":      trades_df["E"].to_numpy().astype(np.float64) / 1000,
                "price":   trades_df["p"].cast(float).to_numpy(),
                "is_sell": trades_df["q"].to_numpy() < 0,
                "count":   len(trades_df),
            }

            # self.progress.emit("Строим heatmap (numba)...")
            # depth_cache = build_heatmap(depth_df)

            self.finished.emit({
                "trades":   trades_cache,
                # "depth":    depth_cache,
                "depth_df": depth_df,   # ← сырой, для build_segments
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trades + Depth Viewer")
        self.resize(1400, 800)

        self._trades_cache  = None
        self._depth_cache   = None
        self._loader        = None
        self._pending_range = None

        self._build_ui()  # plot_widget создаётся здесь
        # self._heatmap = HeatmapRenderer(self.plot_widget)  # после _build_ui
        self._liquidity = LiquidityRenderer(self.plot_widget)

    def _build_ui(self):
        # ── Контролы ──
        self.symbol_input = QLineEdit("riverusdt")
        self.symbol_input.setFixedWidth(100)

        self.start_dt = QDateTimeEdit(
            QDateTime.fromString("2026-03-27 13:00:00", "yyyy-MM-dd HH:mm:ss"))
        self.end_dt = QDateTimeEdit(
            QDateTime.fromString("2026-03-27 13:05:00", "yyyy-MM-dd HH:mm:ss"))
        for w in (self.start_dt, self.end_dt):
            w.setDisplayFormat("yyyy-MM-dd HH:mm:ss")

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.on_load)

        self.heatmap_check = QCheckBox("Depth Heatmap")
        self.heatmap_check.setChecked(True)
        self.heatmap_check.toggled.connect(self._toggle_heatmap)
        self.min_qty_input = QLineEdit("50")
        self.min_qty_input.setFixedWidth(60)


        ctrl = QWidget()
        hl = QHBoxLayout(ctrl)
        hl.setContentsMargins(4, 4, 4, 4)
        for w in [QLabel("Symbol:"), self.symbol_input,
                  QLabel("From:"), self.start_dt,
                  QLabel("To:"), self.end_dt,
                  load_btn, self.heatmap_check, self.min_qty_input]:
            hl.addWidget(w)
        hl.addStretch()
        # hl.addWidget(QLabel("Min qty:"))
        # hl.addWidget(self.min_qty_input)

        # ── График ──
        pg.setConfigOptions(antialias=False, useOpenGL=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#0e0e0e")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        axis = pg.DateAxisItem(orientation="bottom")
        self.plot_widget.getPlotItem().setAxisItems({"bottom": axis})

        # Scatter (трейды) — создаём здесь, heatmap-items — в HeatmapRenderer
        self.scatter_buy = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush(40, 200, 100, 200), size=3, zValue=10)
        self.scatter_sell = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush(220, 70, 50, 200), size=3, zValue=10)
        self.plot_widget.addItem(self.scatter_buy)
        self.plot_widget.addItem(self.scatter_sell)

        # ── Дебаунс зума ──
        self._redraw_timer = QTimer()
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self._redraw_visible)
        self.plot_widget.getViewBox().sigXRangeChanged.connect(
            self._on_x_range_changed)

        # ── Статусбар ──
        self.status = QLabel("Готово")
        self.statusBar().addWidget(self.status)

        # ── Layout ──
        central = QWidget()
        vl = QVBoxLayout(central)
        vl.setContentsMargins(4, 4, 4, 4)
        vl.addWidget(ctrl)
        vl.addWidget(self.plot_widget)
        self.setCentralWidget(central)

    # ── Загрузка ──────────────────────────────────────────────────────────────

    def on_load(self):
        symbol = self.symbol_input.text().strip().lower()
        start  = self.start_dt.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        end    = self.end_dt.dateTime().toString("yyyy-MM-dd HH:mm:ss")

        if self._loader and self._loader.isRunning():
            return

        self._trades_cache = None
        self._depth_cache  = None
        self._loader = DataLoader(symbol, start, end)
        self._loader.progress.connect(self.status.setText)
        self._loader.finished.connect(self._on_data_loaded)
        self._loader.error.connect(lambda e: self.status.setText(f"Ошибка: {e}"))
        self._loader.start()

    def _on_data_loaded(self, data):
        self._trades_cache = data["trades"]
        # self._depth_cache  = data["depth"]

        # if self._depth_cache is not None:
        #     self._heatmap.load(self._depth_cache)

        # Строим сегменты ликвидности
        depth_df = data.get("depth_df")
        if depth_df is not None:
            min_qty = float(self.min_qty_input.text() or "50")
            segs = build_segments(depth_df, threshold=min_qty)
            if segs is not None:
                self._liquidity.load(segs)

        self._redraw_full()

        c, d = self._trades_cache, self._depth_cache
        if d:
            self.status.setText(
                f"Трейдов: {c['count']:,}  |  "
                f"Depth: {d['matrix'].shape[0]:,}×{d['matrix'].shape[1]:,}  |  "
                f"tick={d['tick_size']}  |  "
                f"{self.symbol_input.text().upper()}")
        else:
            self.status.setText(
                f"Трейдов: {c['count']:,}  |  Depth: нет данных")

    # ── Рендер ────────────────────────────────────────────────────────────────

    def _redraw_full(self):
        if self._trades_cache is None:
            return

        c = self._trades_cache
        ts_d, price_d, sell_d = downsample(c["ts"], c["price"], c["is_sell"])
        self._set_scatter(ts_d, price_d, sell_d)

        # if self._depth_cache is not None and self.heatmap_check.isChecked():
        #     self._heatmap.draw(self._depth_cache)

        self.plot_widget.autoRange()

    def _on_x_range_changed(self, _view, x_range):
        self._pending_range = x_range
        self._redraw_timer.start(50)

    def _redraw_visible(self):
        if self._trades_cache is None or self._pending_range is None:
            return
        x_min, x_max = self._pending_range
        c = self._trades_cache
        mask = (c["ts"] >= x_min) & (c["ts"] <= x_max)
        ts_d, price_d, sell_d = downsample(
            c["ts"][mask], c["price"][mask], c["is_sell"][mask])
        self._set_scatter(ts_d, price_d, sell_d)

    def _set_scatter(self, ts, price, sell):
        buy_mask = ~sell
        self.plot_widget.setUpdatesEnabled(False)
        self.scatter_buy.setData(x=ts[buy_mask], y=price[buy_mask])
        self.scatter_sell.setData(x=ts[sell],    y=price[sell])
        self.plot_widget.setUpdatesEnabled(True)

    def _toggle_heatmap(self, checked):
        self._liquidity.set_visible(checked)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


# if __name__ == "__main__":

#     dl = DataManager()

#     df = dl.load_timerange(
#         exchange="binance",
#         symbol="riverusdt",
#         start_time="2026-03-27 13:00:00",
#         end_time="2026-03-27 13:05:00",
#         data_type="depth"
#     )
#     print(df.dtypes)
#     print(df.schema)
#     print(df.head(2))
    