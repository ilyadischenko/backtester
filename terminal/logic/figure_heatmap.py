# logic/figure_heatmap.py

import polars as pl
import numpy as np
import pyqtgraph as pg
from numba import njit
from PyQt6 import QtGui
from PyQt6 import QtCore


# ══════════════════════════════════════════════════════════════════════════════
# NUMBA: построение сегментов уровней
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _build_segments(
    timestamps_ms,   # (N,)  int64
    bid_p, bid_q,    # плоские массивы bid
    bid_row, bid_ptr,
    ask_p, ask_q,    # плоские массивы ask
    ask_row, ask_ptr,
    threshold,       # float — минимальный объём для сегмента
    price_min,
    inv_tick,
    n_prices,
):
    """
    Stateful проход по апдейтам стакана.
    Для каждого ценового уровня отслеживаем:
      - текущий объём в book (+ bid, - ask, 0 пусто)
      - время начала активного сегмента (qty >= threshold)

    Выход: массивы сегментов (t_start, t_end, price_idx, qty)
    """
    N = len(timestamps_ms)

    # Рабочие массивы состояния book
    book_qty       = np.zeros(n_prices, dtype=np.float64)   # текущий объём (+ bid, - ask)
    seg_start      = np.full(n_prices, -1.0, dtype=np.float64)  # -1 = нет активного сегмента
    seg_start_qty  = np.zeros(n_prices, dtype=np.float64)

    # Предвыделяем с запасом (макс. N * среднее кол-во уровней — урежем потом)
    max_segs = N * 20
    out_t0  = np.empty(max_segs, dtype=np.float64)
    out_t1  = np.empty(max_segs, dtype=np.float64)
    out_pi  = np.empty(max_segs, dtype=np.int32)
    out_qty = np.empty(max_segs, dtype=np.float64)
    n_segs  = 0

    def _close_seg(pi, t_end):
        nonlocal n_segs
        if seg_start[pi] >= 0 and n_segs < max_segs:
            out_t0[n_segs]  = seg_start[pi]
            out_t1[n_segs]  = t_end
            out_pi[n_segs]  = pi
            out_qty[n_segs] = seg_start_qty[pi]
            n_segs += 1
            seg_start[pi] = -1.0

    def _open_seg(pi, t_start, qty):
        seg_start[pi]     = t_start
        seg_start_qty[pi] = qty

    for i in range(N):
        t = timestamps_ms[i] / 1000.0

        # --- обновляем bid уровни ---
        for k in range(bid_row[i], bid_ptr[i]):
            raw_qty = bid_q[k]
            pi = int((bid_p[k] - price_min) * inv_tick + 0.5)
            if pi < 0 or pi >= n_prices:
                continue

            prev = book_qty[pi]
            book_qty[pi] = raw_qty  # 0 = удаление уровня

            prev_active = abs(prev) >= threshold
            curr_active = raw_qty >= threshold  # bid всегда >= 0

            if prev_active and not curr_active:
                _close_seg(pi, t)
            elif not prev_active and curr_active:
                _open_seg(pi, t, raw_qty)
            elif prev_active and curr_active:
                # объём изменился — закрываем старый, открываем новый
                if raw_qty != prev:
                    _close_seg(pi, t)
                    _open_seg(pi, t, raw_qty)

        # --- обновляем ask уровни (храним как отрицательные) ---
        for k in range(ask_row[i], ask_ptr[i]):
            raw_qty = ask_q[k]
            pi = int((ask_p[k] - price_min) * inv_tick + 0.5)
            if pi < 0 or pi >= n_prices:
                continue

            stored_qty = -raw_qty  # отрицательное = ask
            prev = book_qty[pi]

            # если уровень перешёл из bid в ask или обратно — сбрасываем
            if prev > 0:
                _close_seg(pi, t)
                book_qty[pi] = 0.0
                prev = 0.0

            book_qty[pi] = stored_qty

            prev_active = abs(prev) >= threshold
            curr_active = raw_qty >= threshold

            if prev_active and not curr_active:
                _close_seg(pi, t)
            elif not prev_active and curr_active:
                _open_seg(pi, t, stored_qty)
            elif prev_active and curr_active:
                if stored_qty != prev:
                    _close_seg(pi, t)
                    _open_seg(pi, t, stored_qty)

    # Закрываем все открытые сегменты последним временем
    t_last = timestamps_ms[N - 1] / 1000.0
    for pi in range(n_prices):
        if seg_start[pi] >= 0:
            _close_seg(pi, t_last)

    return (
        out_t0[:n_segs],
        out_t1[:n_segs],
        out_pi[:n_segs],
        out_qty[:n_segs],
    )


# ══════════════════════════════════════════════════════════════════════════════
# PYTHON: подготовка flat-массивов из Polars
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_flat(df: pl.DataFrame):
    """Превращает list-колонки Polars в плоские NumPy массивы + indptr."""
    bid_lens = df["b_p"].list.len().to_numpy()
    ask_lens = df["a_p"].list.len().to_numpy()

    bid_row = np.zeros(len(bid_lens), dtype=np.int64)
    ask_row = np.zeros(len(ask_lens), dtype=np.int64)
    np.cumsum(bid_lens[:-1], out=bid_row[1:])
    np.cumsum(ask_lens[:-1], out=ask_row[1:])

    bid_ptr = bid_row + bid_lens
    ask_ptr = ask_row + ask_lens

    timestamps = df["E"].to_numpy().astype(np.int64)
    bid_p = df["b_p"].explode().to_numpy().astype(np.float64)
    bid_q = df["b_q"].explode().to_numpy().astype(np.float64)
    ask_p = df["a_p"].explode().to_numpy().astype(np.float64)
    ask_q = df["a_q"].explode().to_numpy().astype(np.float64)

    return timestamps, bid_p, bid_q, bid_row, bid_ptr, ask_p, ask_q, ask_row, ask_ptr


def build_segments(df: pl.DataFrame, threshold: float = 50.0, tick_size: float = 0.001):
    df = df.select(["E", "b_p", "b_q", "a_p", "a_q"])
    if df.is_empty():
        return None

    print(f"[build_segments] {len(df)} апдейтов, threshold={threshold}")

    (timestamps, bid_p, bid_q, bid_row, bid_ptr,
     ask_p, ask_q, ask_row, ask_ptr) = _prepare_flat(df)

    # Фильтруем ненулевые И не-NaN цены
    bid_mask = (bid_q != 0) & np.isfinite(bid_p) & np.isfinite(bid_q)
    ask_mask = (ask_q != 0) & np.isfinite(ask_p) & np.isfinite(ask_q)
    all_prices = np.concatenate([bid_p[bid_mask], ask_p[ask_mask]])

    if len(all_prices) == 0:
        return None

    # Отсекаем выбросы
    p1, p99 = np.percentile(all_prices, 1), np.percentile(all_prices, 99)
    clean = all_prices[(all_prices >= p1) & (all_prices <= p99)]
    if len(clean) == 0:
        return None

    median_price = float(np.median(clean))
    price_range_pct = 0.05  # ±5% от медианы

    price_min = round(median_price * (1 - price_range_pct), 6)
    price_max = round(median_price * (1 + price_range_pct), 6)
    n_prices  = int(round((price_max - price_min) / tick_size)) + 1
    inv_tick  = 1.0 / tick_size

    print(f"[build_segments] median={median_price:.4f}, "
          f"price=[{price_min:.4f}, {price_max:.4f}], "
          f"levels={n_prices}, tick={tick_size}")

    t0, t1, pi, qty = _build_segments(
        timestamps,
        bid_p, bid_q, bid_row, bid_ptr,
        ask_p, ask_q, ask_row, ask_ptr,
        float(threshold),
        price_min, inv_tick, n_prices,
    )

    print(f"[build_segments] сегментов: {len(t0)}")

    return {
        "t0": t0,
        "t1": t1,
        "pi": pi,
        "qty": qty,
        "price_min": price_min,
        "tick_size": tick_size,
    }


# ══════════════════════════════════════════════════════════════════════════════
# РЕНДЕР: LiquidityRenderer
# ══════════════════════════════════════════════════════════════════════════════

class LiquidityRenderer:
    """
    Отрисовывает сегменты ликвидности как горизонтальные линии.
    Яркость = нормированный объём.
    Цвет: зелёный (bid) / красный (ask).
    """

    def __init__(self, plot_widget: pg.PlotWidget):
        self.plot_widget = plot_widget
        self._item: pg.GraphicsObject | None = None
        self._segs = None

    # ── публичный API ──────────────────────────────────────────────────────

    def load(self, segs: dict):
        self._segs = segs
        self._rebuild()

    def set_visible(self, v: bool):
        if self._item is not None:
            self._item.setVisible(v)

    def clear(self):
        if self._item is not None:
            self.plot_widget.removeItem(self._item)
            self._item = None

    # ── внутренние методы ──────────────────────────────────────────────────

    def _rebuild(self):
        self.clear()
        segs = self._segs
        if segs is None or len(segs["t0"]) == 0:
            return

        t0   = segs["t0"]
        t1   = segs["t1"]
        pi   = segs["pi"]
        qty  = segs["qty"]
        pmin = segs["price_min"]
        tick = segs["tick_size"]

        prices = pmin + pi * tick

        # Нормируем объём для яркости (0..1)
        abs_qty  = np.abs(qty)
        vmax     = float(np.percentile(abs_qty, 98)) if len(abs_qty) else 1.0
        if vmax == 0:
            vmax = 1.0
        alpha_f  = np.clip(abs_qty / vmax, 0.05, 1.0)  # float 0..1

        is_bid = qty > 0

        item = _SegmentItem(t0, t1, prices, alpha_f, is_bid)
        item.setZValue(5)
        self.plot_widget.addItem(item)
        self._item = item


class _SegmentItem(pg.GraphicsObject):
    """
    Кастомный GraphicsObject: рисует все сегменты за один QPainter-проход.
    """

    def __init__(self, t0, t1, prices, alpha_f, is_bid):
        super().__init__()
        self._t0      = t0
        self._t1      = t1
        self._prices  = prices
        self._alpha_f = alpha_f          # float array 0..1
        self._is_bid  = is_bid

        # Предвычисляем bounding rect
        if len(t0):
            self._br = QtCore.QRectF(
                float(t0.min()), float(prices.min()),
                float(t1.max() - t0.min()),
                float(prices.max() - prices.min()) + 0.01,
            )
        else:
            self._br = QtCore.QRectF()


    def boundingRect(self):
        return self._br

    def paint(self, painter, option, widget=None):
        if len(self._t0) == 0:
            return

        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)

        # Группируем по is_bid для меньшего числа смен пера
        for side in (True, False):
            mask = self._is_bid == side
            if not mask.any():
                continue

            t0s    = self._t0[mask]
            t1s    = self._t1[mask]
            prs    = self._prices[mask]
            alphas = self._alpha_f[mask]

            # Сортируем по alpha чтобы менять перо только при изменении цвета
            order  = np.argsort(alphas)
            t0s    = t0s[order]
            t1s    = t1s[order]
            prs    = prs[order]
            alphas = alphas[order]

            prev_alpha = -1.0
            for idx in range(len(t0s)):
                a = alphas[idx]
                if abs(a - prev_alpha) > 0.02:          # меняем перо только при заметном изменении
                    alpha_byte = int(a * 220)
                    if side:    # bid — зелёный
                        color = QtGui.QColor(40, 200, 100, alpha_byte)
                    else:       # ask — красный
                        color = QtGui.QColor(220, 60, 50, alpha_byte)
                    painter.setPen(QtGui.QPen(color, 0))
                    prev_alpha = a

                painter.drawLine(
                    QtCore.QPointF(t0s[idx], prs[idx]),
                    QtCore.QPointF(t1s[idx], prs[idx]),
                )

        painter.restore()