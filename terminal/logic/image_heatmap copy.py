# heatmap.py
import numpy as np
import pyqtgraph as pg
from numba import njit, prange




# ── Numba функции ──────────────────────────────────────────────────────────────

@njit(cache=True)
def _build_matrix(n_times, n_prices, price_min, inv_tick,
                  bid_p, bid_q, bid_starts, bid_ends,
                  ask_p, ask_q, ask_starts, ask_ends):
    """
    Строит heatmap матрицу из плоских массивов апдейтов.
    book[j] > 0 = bid, book[j] < 0 = ask
    """
    matrix = np.zeros((n_times, n_prices), dtype=np.float32)
    book = np.zeros(n_prices, dtype=np.float32)

    for i in range(n_times):
        # Применяем bid-апдейты
        for k in range(bid_starts[i], bid_ends[i]):
            j = int(round((bid_p[k] - price_min) * inv_tick))
            if 0 <= j < n_prices:
                book[j] = bid_q[k]  # 0 = удаление, >0 = установка

        # Применяем ask-апдейты
        for k in range(ask_starts[i], ask_ends[i]):
            j = int(round((ask_p[k] - price_min) * inv_tick))
            if 0 <= j < n_prices:
                q = ask_q[k]
                book[j] = -q if q > 0 else 0.0

        # Копируем состояние book в матрицу
        matrix[i, :] = book

    return matrix


@njit(cache=True)
def _compute_mid_prices(matrix, price_min, tick_size):
    """Вычисляет mid price для каждого снапшота."""
    n_times, n_prices = matrix.shape
    mid_prices = np.empty(n_times, dtype=np.float64)

    for i in range(n_times):
        best_bid_idx = -1
        best_ask_idx = -1

        # Ищем лучший bid (максимальный индекс где > 0)
        for j in range(n_prices - 1, -1, -1):
            if matrix[i, j] > 0:
                best_bid_idx = j
                break

        # Ищем лучший ask (минимальный индекс где < 0)
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
    """Даунсэмплинг матрицы по времени (берём каждый step-й снапшот)."""
    n_times = matrix.shape[0]
    n_prices = matrix.shape[1]
    new_n = (n_times + step - 1) // step
    
    result = np.empty((new_n, n_prices), dtype=np.float32)
    for i in prange(new_n):
        result[i, :] = matrix[i * step, :]
    
    return result



# ── Подготовка данных ──────────────────────────────────────────────────────────


def prepare_depth_arrays(rows):
    """
    Конвертирует list of tuples в плоские numpy массивы для numba.
    Возвращает: timestamps, bid/ask массивы с индексами начала/конца для каждой строки.
    """
    n = len(rows)

    # Подсчитываем размеры
    total_bids = sum(len(r[1]) for r in rows)
    total_asks = sum(len(r[3]) for r in rows)

    timestamps = np.empty(n, dtype=np.float64)

    bid_p = np.empty(total_bids, dtype=np.float64)
    bid_q = np.empty(total_bids, dtype=np.float64)
    bid_starts = np.empty(n, dtype=np.int64)
    bid_ends = np.empty(n, dtype=np.int64)

    ask_p = np.empty(total_asks, dtype=np.float64)
    ask_q = np.empty(total_asks, dtype=np.float64)
    ask_starts = np.empty(n, dtype=np.int64)
    ask_ends = np.empty(n, dtype=np.int64)

    bid_idx = 0
    ask_idx = 0

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

def build_heatmap(df, max_times=3000):
    """
    Строит heatmap матрицу из depth апдейтов.
    Каждый ценовой уровень = отдельная строка (без батчинга).
    """
    rows = list(df.select(["E", "b_p", "b_q", "a_p", "a_q"]).iter_rows())
    if not rows:
        return None

    print(f"[build_heatmap] Подготовка {len(rows)} апдейтов...")

    # ── Определяем ценовой диапазон и tick_size ──
    all_prices = set()
    for _, b_p, b_q, a_p, a_q in rows:
        all_prices.update(p for p, q in zip(b_p, b_q) if q != 0)
        all_prices.update(p for p, q in zip(a_p, a_q) if q != 0)

    if len(all_prices) < 2:
        return None

    sorted_prices = np.array(sorted(all_prices))
    price_min = float(sorted_prices[0])
    price_max = float(sorted_prices[-1])

    # Tick size = минимальная ненулевая разница
    diffs = np.diff(sorted_prices)
    diffs = diffs[diffs > 1e-12]
    # tick_size = float(np.min(diffs)) if len(diffs) > 0 else 0.01
    tick_size = 0.001


    n_prices = int(round((price_max - price_min) / tick_size)) + 1
    n_times = len(rows)
    inv_tick = 1.0 / tick_size

    print(f"[build_heatmap] price_range=[{price_min}, {price_max}], "
          f"tick={tick_size}, levels={n_prices}")

    # ── Подготавливаем плоские массивы ──
    (timestamps,
     bid_p, bid_q, bid_starts, bid_ends,
     ask_p, ask_q, ask_starts, ask_ends) = prepare_depth_arrays(rows)

    timestamps = timestamps / 1000.0  # ms -> s

    # ── Строим матрицу через numba ──
    print(f"[build_heatmap] Building matrix {n_times}x{n_prices}...")
    matrix = _build_matrix(
        n_times, n_prices, price_min, inv_tick,
        bid_p, bid_q, bid_starts, bid_ends,
        ask_p, ask_q, ask_starts, ask_ends
    )

    # ── Даунсэмплинг по времени ──
    if n_times > max_times:
        step = n_times // max_times
        matrix = _downsample_matrix(matrix, step)
        timestamps = timestamps[::step].copy()
        print(f"[build_heatmap] Downsampled to {matrix.shape[0]} snapshots")

    return {
        "matrix": matrix,
        "timestamps": timestamps,
        "price_min": price_min,
        "price_max": price_max,
        "tick_size": tick_size,
    }



# ── Colormap ───────────────────────────────────────────────────────────────────

def make_rg_colormap():
    colors = [
        (0.0,  (180, 40,  40,  220)),
        (0.45, (30,  10,  10,  180)),
        (0.5,  (14,  14,  14,  0)),
        (0.55, (10,  30,  10,  180)),
        (1.0,  (40,  180, 80,  220)),
    ]
    pos = np.array([c[0] for c in colors])
    rgba = np.array([c[1] for c in colors], dtype=np.ubyte)
    return pg.ColorMap(pos, rgba)

# ── Класс-обёртка для рендера ──────────────────────────────────────────────────

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
        """Принимает готовый depth_cache из DataLoader, предвычисляет mid."""
        dc = depth_cache
        self._mid_prices_cache = _compute_mid_prices(
            dc["matrix"], dc["price_min"], dc["tick_size"])

    def draw(self, depth_cache: dict):
        """Рисует heatmap и mid line."""
        if depth_cache is None:
            return
        dc = depth_cache
        matrix = dc["matrix"]
        timestamps = dc["timestamps"]
        price_min = dc["price_min"]
        tick_size = dc["tick_size"]

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
        t0 = float(timestamps[0])
        dt = float(timestamps[-1] - timestamps[0]) / max(n_t - 1, 1)

        tr = pg.QtGui.QTransform()
        tr.translate(t0 - 0.5 * dt, price_min - 0.5 * tick_size)
        tr.scale(dt, tick_size)
        self.image_item.setTransform(tr)

        if self._mid_prices_cache is not None:
            self.mid_line.setData(x=timestamps, y=self._mid_prices_cache)