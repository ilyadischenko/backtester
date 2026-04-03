# heatmap.py
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGraphicsSimpleTextItem
from numba import njit, prange


# ══════════════════════════════════════════════════════════════════════════════
# СТАРЫЙ ПАЙПЛАЙН (ImageItem heatmap) — сохранён
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _build_matrix(n_times, n_prices, price_min, inv_tick,
                  bid_p, bid_q, bid_starts, bid_ends,
                  ask_p, ask_q, ask_starts, ask_ends):
    matrix = np.zeros((n_times, n_prices), dtype=np.float32)
    book = np.zeros(n_prices, dtype=np.float32)

    for i in range(n_times):
        for k in range(bid_starts[i], bid_ends[i]):
            j = int(round((bid_p[k] - price_min) * inv_tick))
            if 0 <= j < n_prices:
                book[j] = bid_q[k]

        for k in range(ask_starts[i], ask_ends[i]):
            j = int(round((ask_p[k] - price_min) * inv_tick))
            if 0 <= j < n_prices:
                q = ask_q[k]
                book[j] = -q if q > 0 else 0.0

        matrix[i, :] = book

    return matrix


@njit(cache=True)
def _compute_mid_prices(matrix, price_min, tick_size):
    n_times, n_prices = matrix.shape
    mid_prices = np.empty(n_times, dtype=np.float64)

    for i in range(n_times):
        best_bid_idx = -1
        best_ask_idx = -1

        for j in range(n_prices - 1, -1, -1):
            if matrix[i, j] > 0:
                best_bid_idx = j
                break

        for j in range(n_prices):
            if matrix[i, j] < 0:
                best_ask_idx = j
                break

        if best_bid_idx >= 0 and best_ask_idx >= 0:
            best_bid = price_min + best_bid_idx * tick_size
            best_ask = price_min + best_ask_idx * tick_size
            mid_prices[i] = (best_bid + best_ask) * 0.5
        else:
            mid_prices[i] = np.nan

    return mid_prices


@njit(cache=True, parallel=True)
def _downsample_matrix(matrix, step):
    n_times = matrix.shape[0]
    n_prices = matrix.shape[1]
    new_n = (n_times + step - 1) // step
    result = np.empty((new_n, n_prices), dtype=np.float32)
    for i in prange(new_n):
        result[i, :] = matrix[i * step, :]
    return result


def prepare_depth_arrays(rows):
    n = len(rows)
    total_bids = sum(len(r[1]) for r in rows)
    total_asks = sum(len(r[3]) for r in rows)

    timestamps  = np.empty(n, dtype=np.float64)
    bid_p       = np.empty(total_bids, dtype=np.float64)
    bid_q       = np.empty(total_bids, dtype=np.float64)
    bid_starts  = np.empty(n, dtype=np.int64)
    bid_ends    = np.empty(n, dtype=np.int64)
    ask_p       = np.empty(total_asks, dtype=np.float64)
    ask_q       = np.empty(total_asks, dtype=np.float64)
    ask_starts  = np.empty(n, dtype=np.int64)
    ask_ends    = np.empty(n, dtype=np.int64)

    bid_idx = ask_idx = 0

    for i, (E, b_prices, b_qtys, a_prices, a_qtys) in enumerate(rows):
        timestamps[i] = E
        bid_starts[i] = bid_idx
        for p, q in zip(b_prices, b_qtys):
            bid_p[bid_idx] = p
            bid_q[bid_idx] = q
            bid_idx += 1
        bid_ends[i] = bid_idx

        ask_starts[i] = ask_idx
        for p, q in zip(a_prices, a_qtys):
            ask_p[ask_idx] = p
            ask_q[ask_idx] = q
            ask_idx += 1
        ask_ends[i] = ask_idx

    return (timestamps,
            bid_p, bid_q, bid_starts, bid_ends,
            ask_p, ask_q, ask_starts, ask_ends)

def build_heatmap(df, max_times=3000, tick_size=0.001):
    rows = list(df.select(["E", "b_p", "b_q", "a_p", "a_q"]).iter_rows())
    if not rows:
        return None

    print(f"[build_heatmap] Подготовка {len(rows)} апдейтов...")

    all_prices = set()
    for _, b_p, b_q, a_p, a_q in rows:
        all_prices.update(p for p, q in zip(b_p, b_q) if q != 0)
        all_prices.update(p for p, q in zip(a_p, a_q) if q != 0)

    if len(all_prices) < 2:
        return None

    sorted_prices = np.array(sorted(all_prices))
    price_min = float(sorted_prices[0])
    price_max = float(sorted_prices[-1])

    n_prices = int(round((price_max - price_min) / tick_size)) + 1
    n_times  = len(rows)
    inv_tick = 1.0 / tick_size

    print(f"[build_heatmap] price_range=[{price_min}, {price_max}], "
          f"tick={tick_size}, levels={n_prices}")

    (timestamps,
     bid_p, bid_q, bid_starts, bid_ends,
     ask_p, ask_q, ask_starts, ask_ends) = prepare_depth_arrays(rows)

    timestamps = timestamps / 1000.0

    print(f"[build_heatmap] Building matrix {n_times}x{n_prices}...")
    matrix = _build_matrix(
        n_times, n_prices, price_min, inv_tick,
        bid_p, bid_q, bid_starts, bid_ends,
        ask_p, ask_q, ask_starts, ask_ends)

    if n_times > max_times:
        step      = n_times // max_times
        matrix    = _downsample_matrix(matrix, step)
        timestamps = timestamps[::step].copy()
        print(f"[build_heatmap] Downsampled to {matrix.shape[0]} snapshots")

    return {
        "matrix":     matrix,
        "timestamps": timestamps,
        "price_min":  price_min,
        "price_max":  price_max,
        "tick_size":  tick_size,
    }

def make_rg_colormap():
    colors = [
        (0.0,  (180, 40,  40,  220)),
        (0.45, (30,  10,  10,  180)),
        (0.5,  (14,  14,  14,  0)),
        (0.55, (10,  30,  10,  180)),
        (1.0,  (40,  180, 80,  220)),
    ]
    pos  = np.array([c[0] for c in colors])
    rgba = np.array([c[1] for c in colors], dtype=np.ubyte)
    return pg.ColorMap(pos, rgba)


class HeatmapRenderer:
    def __init__(self, plot_widget):
        self.plot_widget = plot_widget
        self._mid_prices_cache = None

        self.image_item = pg.ImageItem()
        self.image_item.setColorMap(make_rg_colormap())
        self.image_item.setZValue(-10)
        plot_widget.addItem(self.image_item)

        self.mid_line = pg.PlotDataItem(
            pen=pg.mkPen("#ffff00", width=1,
                         style=pg.QtCore.Qt.PenStyle.DotLine))
        plot_widget.addItem(self.mid_line)

    def set_visible(self, visible: bool):
        self.image_item.setVisible(visible)
        self.mid_line.setVisible(visible)

    def load(self, depth_cache: dict):
        dc = depth_cache
        self._mid_prices_cache = _compute_mid_prices(
            dc["matrix"], dc["price_min"], dc["tick_size"])

    def draw(self, depth_cache: dict):
        if depth_cache is None:
            return
        dc         = depth_cache
        matrix     = dc["matrix"]
        timestamps = dc["timestamps"]
        price_min  = dc["price_min"]
        tick_size  = dc["tick_size"]

        if matrix.size == 0:
            return

        nonzero = matrix[matrix != 0]
        if len(nonzero) == 0:
            return
        vmax = float(np.percentile(np.abs(nonzero), 95))
        if vmax == 0:
            return

        self.image_item.setImage(matrix, levels=(-vmax, vmax), autoLevels=False)

        n_t = matrix.shape[0]
        t0  = float(timestamps[0])
        dt  = float(timestamps[-1] - timestamps[0]) / max(n_t - 1, 1)

        tr = pg.QtGui.QTransform()
        tr.translate(t0 - 0.5 * dt, price_min - 0.5 * tick_size)
        tr.scale(dt, tick_size)
        self.image_item.setTransform(tr)

        if self._mid_prices_cache is not None:
            self.mid_line.setData(x=timestamps, y=self._mid_prices_cache)


# ══════════════════════════════════════════════════════════════════════════════
# НОВЫЙ ПАЙПЛАЙН — уровни ликвидности линиями
# ══════════════════════════════════════════════════════════════════════════════

def build_segments(df, min_qty: float = 10.0):
    """
    Строит список отрезков для уровней с объёмом >= min_qty.

    Каждый апдейт уровня = новый отрезок длиной до следующего апдейта.
    Возвращает dict с раздельными массивами для bid и ask (для разного цвета).

    Формат сегмента:
        t_start, t_end, price, qty, side  (side: 1=bid, -1=ask)
    """
    rows = list(df.select(["E", "b_p", "b_q", "a_p", "a_q"]).iter_rows())
    if not rows:
        return None

    print(f"[build_segments] Обрабатываем {len(rows)} апдейтов, min_qty={min_qty}")

    # book хранит последний апдейт уровня:
    # price -> [t_start, qty, side]  (side: 1=bid, -1=ask)
    book = {}

    # Накапливаем сегменты в списки
    seg_t0    = []
    seg_t1    = []
    seg_price = []
    seg_qty   = []
    seg_side  = []  # 1=bid, -1=ask

    last_t = rows[-1][0] / 1000.0

    for E, b_p, b_q, a_p, a_q in rows:
        t = E / 1000.0

        # ── Bid ──
        for price, qty in zip(b_p, b_q):
            key = (price, 1)

            if key in book:
                # Закрываем предыдущий отрезок
                prev_t, prev_qty = book[key]
                if prev_qty >= min_qty:
                    seg_t0.append(prev_t)
                    seg_t1.append(t)
                    seg_price.append(price)
                    seg_qty.append(prev_qty)
                    seg_side.append(1)

            if qty >= min_qty:
                book[key] = (t, qty)
            else:
                # Уровень снят или ниже порога — удаляем
                book.pop(key, None)

        # ── Ask ──
        for price, qty in zip(a_p, a_q):
            key = (price, -1)

            if key in book:
                prev_t, prev_qty = book[key]
                if prev_qty >= min_qty:
                    seg_t0.append(prev_t)
                    seg_t1.append(t)
                    seg_price.append(price)
                    seg_qty.append(prev_qty)
                    seg_side.append(-1)

            if qty >= min_qty:
                book[key] = (t, qty)
            else:
                book.pop(key, None)

    # ── Закрываем незакрытые уровни по последнему timestamp ──
    for (price, side), (t_start, qty) in book.items():
        if qty >= min_qty:
            seg_t0.append(t_start)
            seg_t1.append(last_t)
            seg_price.append(price)
            seg_qty.append(qty)
            seg_side.append(side)

    if not seg_t0:
        print("[build_segments] Нет сегментов выше порога")
        return None

    result = {
        "t0":    np.array(seg_t0,    dtype=np.float64),
        "t1":    np.array(seg_t1,    dtype=np.float64),
        "price": np.array(seg_price, dtype=np.float64),
        "qty":   np.array(seg_qty,   dtype=np.float64),
        "side":  np.array(seg_side,  dtype=np.int8),
    }

    bid_count = int((result["side"] == 1).sum())
    ask_count = int((result["side"] == -1).sum())
    print(f"[build_segments] Сегментов: {len(seg_t0)} "
          f"(bid={bid_count}, ask={ask_count})")

    return result


def _segments_to_lines(segments: dict):
    """
    Конвертирует сегменты в формат для pg.PlotDataItem с connect='pairs'.

    x = [t0, t1, t0, t1, ...]
    y = [p,  p,  p,  p,  ...]
    connect = 'pairs'
    """
    t0    = segments["t0"]
    t1    = segments["t1"]
    price = segments["price"]

    n = len(t0)
    x = np.empty(n * 2, dtype=np.float64)
    y = np.empty(n * 2, dtype=np.float64)

    x[0::2] = t0
    x[1::2] = t1
    y[0::2] = price
    y[1::2] = price

    return x, y


class LiquidityRenderer:
    BID_COLOR = (40,  200, 100)
    ASK_COLOR = (220, 70,  50)

    def __init__(self, plot_widget):
        self.plot_widget = plot_widget
        self._segments   = None
        self._visible    = True
        self._items      = []

        self._tooltip = pg.TextItem(
            text="", color="#ffffff",
            fill=pg.mkBrush(30, 30, 30, 200),
            anchor=(0, 1))
        self._tooltip.setZValue(100)
        self._tooltip.setVisible(False)
        plot_widget.addItem(self._tooltip)
        plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def set_visible(self, visible: bool):
        self._visible = visible
        for item in self._items:
            item.setVisible(visible)
        if not visible:
            self._tooltip.setVisible(False)

    def load(self, segments: dict):
        self._segments = segments
        self._draw()

    def _clear_items(self):
        for item in self._items:
            self.plot_widget.removeItem(item)
        self._items.clear()

    def _draw(self):
        self._clear_items()
        if self._segments is None:
            return

        seg   = self._segments
        t0    = seg["t0"]
        t1    = seg["t1"]
        price = seg["price"]
        qty   = seg["qty"]
        sides = seg["side"]

        q95 = float(np.percentile(qty, 95))
        if q95 == 0:
            q95 = float(qty.max()) or 1.0

        alpha_norm = np.clip(qty / q95, 0.15, 1.0)

        for i in range(len(t0)):
            r, g, b = self.BID_COLOR if sides[i] == 1 else self.ASK_COLOR
            a = int(alpha_norm[i] * 220 + 35)
            pen = pg.mkPen(color=(r, g, b, a), width=1.5)

            item = pg.PlotCurveItem(
                x=np.array([t0[i], t1[i]]),
                y=np.array([price[i], price[i]]),
                pen=pen)
            item.setZValue(5)
            self.plot_widget.addItem(item)
            self._items.append(item)

    def _on_mouse_moved(self, scene_pos):
        if not self._visible or self._segments is None:
            return

        vb = self.plot_widget.getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos):
            self._tooltip.setVisible(False)
            return

        mouse = vb.mapSceneToView(scene_pos)
        mt, mp = mouse.x(), mouse.y()

        seg   = self._segments
        t0, t1, price, qty, side = (
            seg["t0"], seg["t1"], seg["price"], seg["qty"], seg["side"])

        x_range = vb.viewRange()[0]
        y_range = vb.viewRange()[1]
        t_tol = (x_range[1] - x_range[0]) * 0.005
        p_tol = (y_range[1] - y_range[0]) * 0.003

        mask = (t0 <= mt) & (mt <= t1) & (np.abs(price - mp) <= p_tol)
        hits = np.where(mask)[0]

        if len(hits) == 0:
            self._tooltip.setVisible(False)
            return

        idx      = hits[np.argmin(np.abs(price[hits] - mp))]
        hit_side = "BID" if side[idx] == 1 else "ASK"
        text = (f"{hit_side}  {price[idx]:.3f}\n"
                f"qty:  {qty[idx]:.1f}\n"
                f"dur:  {t1[idx] - t0[idx]:.1f}s")

        self._tooltip.setText(text)
        self._tooltip.setPos(mt, price[idx])
        self._tooltip.setVisible(True)

