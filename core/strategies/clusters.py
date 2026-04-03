from typing import Optional
from collections import deque
from dataclasses import dataclass
import numpy as np
from numba import njit

from visualization.plot_recorder import PlotRecorder


@njit(cache=True, fastmath=True, nogil=True)
def calc_cluster(prices: np.ndarray, volumes: np.ndarray, num_bins: int, percentile: float):
    """Расчёт кластера."""
    n = len(prices)
    if n == 0:
        return 0.0, 0.0, 0.0

    pmin, pmax = prices[0], prices[0]
    for i in range(1, n):
        if prices[i] < pmin: pmin = prices[i]
        if prices[i] > pmax: pmax = prices[i]

    if pmax <= pmin:
        return pmin, pmax, pmin

    bw = (pmax - pmin) / num_bins
    bins = np.zeros(num_bins, dtype=np.float64)

    for i in range(n):
        idx = int((prices[i] - pmin) / bw)
        if idx >= num_bins: idx = num_bins - 1
        bins[idx] += volumes[i]

    total = np.sum(bins)
    target = total * (percentile / 100.0)

    best_s, best_e, best_w = 0, num_bins - 1, num_bins
    left, cur = 0, 0.0

    for right in range(num_bins):
        cur += bins[right]
        while cur - bins[left] >= target and left < right:
            cur -= bins[left]
            left += 1
        if cur >= target and (right - left + 1) < best_w:
            best_w = right - left + 1
            best_s, best_e = left, right

    low = pmin + best_s * bw
    high = pmin + (best_e + 1) * bw

    max_v, poc_bin = 0.0, best_s
    for i in range(best_s, best_e + 1):
        if bins[i] > max_v:
            max_v = bins[i]
            poc_bin = i

    return low, high, pmin + (poc_bin + 0.5) * bw


@njit(cache=True, fastmath=True, nogil=True)
def calc_trend(prices: np.ndarray, times: np.ndarray):
    """Расчёт тренда: slope_pct, r_squared."""
    n = len(prices)
    if n < 2:
        return 0.0, 0.0

    t0 = times[0]
    sx, sy, sxx, sxy, syy = 0.0, 0.0, 0.0, 0.0, 0.0

    for i in range(n):
        x = (times[i] - t0) / 1000.0
        y = prices[i]
        sx += x
        sy += y
        sxx += x * x
        sxy += x * y
        syy += y * y

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, 0.0

    slope = (n * sxy - sx * sy) / denom
    y_mean = sy / n
    slope_pct = (slope / y_mean) * 100.0 if y_mean > 0 else 0.0

    ss_tot = syy - sy * sy / n
    ss_res = syy - slope * sxy - (sy / n) * (sy - slope * sx)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return slope_pct, max(0.0, min(1.0, r_sq))


@dataclass(slots=True)
class Trade:
    time: int
    price: float
    volume: float
    is_buy: bool


class ClusterMeanReversionStrategy:
    """
    Кластерная стратегия mean-reversion под новый движок.

    Источники данных из движка:
      - event_type="depth"  → engine.ob.best_bid / best_ask → mid_price
      - event_type="trade"  → поля p (price), q (qty, отрицательный = sell taker)
    """

    def __init__(
        self,
        cluster_window_sec: float = 60 * 60,
        trend_window_sec: float = 60.0,
        cluster_percentile: float = 70.0,
        num_bins: int = 40,
        breakout_pct: float = 0.1,
        slowdown_pct: float = 50.0,
        activity_window: int = 100,
        activity_baseline: int = 200,
        max_slope_pct: float = 0.01,
        min_r_squared: float = 0.6,
        max_vol_imbalance: float = 0.5,
        order_size: float = 50.0,
        max_position: float = 500.0,
        max_adds: int = 4,
        stop_pct: float = 0.5,
        take_pct: float = 0.75,
        add_delay_ms: int = 30_000,
        add_improve_pct: float = 0.1,
        min_trades: int = 100,
        recalc_ms: int = 500,
        cluster_update_ms: int = 5000,
    ):
        self.initial_balance = 1000.0

        # Параметры
        self.cluster_window_ms  = int(cluster_window_sec * 1000)
        self.trend_window_ms    = int(trend_window_sec * 1000)
        self.cluster_percentile = cluster_percentile
        self.num_bins           = num_bins
        self.breakout_pct       = breakout_pct
        self.slowdown_pct       = slowdown_pct
        self.activity_window    = activity_window
        self.activity_baseline  = activity_baseline
        self.max_slope_pct      = max_slope_pct
        self.min_r_squared      = min_r_squared
        self.max_vol_imbalance  = max_vol_imbalance
        self.order_size         = order_size
        self.max_position       = max_position
        self.max_adds           = max_adds
        self.stop_pct           = stop_pct
        self.take_pct           = take_pct
        self.add_delay_ms       = add_delay_ms
        self.add_improve_pct    = add_improve_pct
        self.min_trades         = min_trades
        self.recalc_ms          = recalc_ms
        self.cluster_update_ms  = cluster_update_ms

        # Данные
        self.trades: deque[Trade]         = deque()
        self.prices: deque[tuple[int, float]] = deque()  # (time, mid_price)

        # Кэш кластера
        self.cluster_low      = 0.0
        self.cluster_high     = 0.0
        self.cluster_poc      = 0.0
        self.baseline_activity = 0.0

        # Кэш тренда
        self.trend_slope   = 0.0
        self.trend_r2      = 0.0
        self.vol_imbalance = 0.0

        # Текущая цена
        self.mid_price = 0.0

        # Состояние
        self.breakout_side    = 0
        self.waiting_slowdown = False

        # Позиция
        self.in_position      = False
        self.position_side    = 0
        self.entry_price      = 0.0
        self.first_entry_time = 0
        self.stop_price       = 0.0
        self.take_price       = 0.0
        self.add_count        = 0
        self.total_size       = 0.0
        self.take_order_id: Optional[int] = None

        # Тайминги
        self.last_recalc         = 0
        self.last_cluster_update = 0

        self.plot = PlotRecorder()

        # Прогрев numba
        _p = np.array([1.0, 2.0], dtype=np.float64)
        _t = np.array([0, 1000], dtype=np.int64)
        _ = calc_cluster(_p, _p, 10, 70.0)
        _ = calc_trend(_p, _t)

    # ═══════════════════════════════════════════════════════════
    # ГЛАВНЫЙ ОБРАБОТЧИК
    # ═══════════════════════════════════════════════════════════

    def on_tick(self, event: dict, engine):
        t     = event["event_time"]
        etype = event.get("event_type")

        # ── Обновление mid_price из стакана ──────────────────
        # depth и ob_snapshot уже применены к engine.ob до вызова on_tick,
        # поэтому просто читаем best_bid/best_ask напрямую.
        if etype in ("depth", "ob_snapshot"):
            if engine.ob.is_ready:
                self.mid_price = (engine.ob.best_bid + engine.ob.best_ask) / 2
                self.prices.append((t, self.mid_price))

        # ── Сбор трейдов ──────────────────────────────────────
        # В новом движке: p = цена, q = объём (q < 0 → sell taker, q > 0 → buy taker)
        elif etype == "trade":
            p = event.get("p", 0.0)
            q = event.get("q", 0.0)
            if p > 0 and q != 0:
                is_buy = q > 0          # buy taker = агрессивная покупка
                self.trades.append(Trade(t, p, abs(q), is_buy))

        # ── Очистка старых данных ─────────────────────────────
        self._cleanup(t)

        # ── Гейты ────────────────────────────────────────────
        if self.mid_price <= 0:
            return
        if len(self.trades) < self.min_trades:
            return
        if t - self.last_recalc < self.recalc_ms:
            return

        # ── Пересчёт кластера и тренда ────────────────────────
        if t - self.last_cluster_update >= self.cluster_update_ms:
            self._update_cluster()
            self._update_trend()
            self.last_cluster_update = t

        if self.cluster_low <= 0:
            self.last_recalc = t
            return

        # ── Синхронизация позиции ─────────────────────────────
        pos = self._get_position(engine)
        inv = pos.size if pos else 0.0

        if inv == 0.0 and self.in_position:
            self._reset()

        # ── Торговая логика ───────────────────────────────────
        if self.in_position:
            self._manage(engine, inv, t)
        else:
            self._look_for_entry(engine, inv, t)

        self._draw(t)
        self.last_recalc = t

    # ═══════════════════════════════════════════════════════════
    # ДАННЫЕ
    # ═══════════════════════════════════════════════════════════

    def _cleanup(self, t: int):
        cutoff_cluster = t - self.cluster_window_ms
        cutoff_trend   = t - self.trend_window_ms

        while self.trades and self.trades[0].time < cutoff_cluster:
            self.trades.popleft()
        while self.prices and self.prices[0][0] < cutoff_trend:
            self.prices.popleft()

    def _update_cluster(self):
        if len(self.trades) < self.min_trades:
            return

        prices  = np.array([tr.price  for tr in self.trades], dtype=np.float64)
        volumes = np.array([tr.volume for tr in self.trades], dtype=np.float64)

        self.cluster_low, self.cluster_high, self.cluster_poc = calc_cluster(
            prices, volumes, self.num_bins, self.cluster_percentile
        )

        if len(volumes) >= self.activity_baseline:
            self.baseline_activity = float(np.mean(volumes[-self.activity_baseline:]))

    def _update_trend(self):
        if len(self.prices) < 10:
            self.trend_slope = 0.0
            self.trend_r2    = 0.0
            return

        times  = np.array([p[0] for p in self.prices], dtype=np.int64)
        prices = np.array([p[1] for p in self.prices], dtype=np.float64)

        self.trend_slope, self.trend_r2 = calc_trend(prices, times)

        buy_vol  = sum(tr.volume for tr in self.trades if tr.is_buy)
        sell_vol = sum(tr.volume for tr in self.trades if not tr.is_buy)
        total    = buy_vol + sell_vol
        self.vol_imbalance = (buy_vol - sell_vol) / total if total > 0 else 0.0

    # ═══════════════════════════════════════════════════════════
    # ФИЛЬТРЫ
    # ═══════════════════════════════════════════════════════════

    def _is_trending(self) -> bool:
        return abs(self.trend_slope) > self.max_slope_pct and self.trend_r2 > self.min_r_squared

    def _safe_to_enter(self, side: int) -> bool:
        if self._is_trending():
            trend_dir = 1 if self.trend_slope > 0 else -1
            if trend_dir != side:
                return False

        if abs(self.vol_imbalance) > self.max_vol_imbalance:
            vol_dir = 1 if self.vol_imbalance > 0 else -1
            if vol_dir != side:
                return False

        return True

    # ═══════════════════════════════════════════════════════════
    # ПОИСК ВХОДА
    # ═══════════════════════════════════════════════════════════

    def _look_for_entry(self, engine, inv: float, t: int):
        if abs(inv) >= self.max_position:
            return

        rng = self.cluster_high - self.cluster_low
        buf = rng * (self.breakout_pct / 100)

        if self.mid_price > self.cluster_high + buf:
            new_side = 1
        elif self.mid_price < self.cluster_low - buf:
            new_side = -1
        else:
            self.breakout_side    = 0
            self.waiting_slowdown = False
            return

        # Фиксируем пробой
        if self.breakout_side == 0:
            self.breakout_side    = new_side
            self.waiting_slowdown = True
            self.plot.marker("Breakout", self.mid_price, t,
                             marker="diamond", color="#00BCD4", size=10)
            return

        # Ждём замедления активности
        if self.waiting_slowdown and self.baseline_activity > 0:
            recent = [tr.volume for tr in list(self.trades)[-self.activity_window:]]
            if recent:
                current  = sum(recent) / len(recent)
                slowdown = (1 - current / self.baseline_activity) * 100

                if slowdown >= self.slowdown_pct:
                    entry_side = -self.breakout_side

                    if not self._safe_to_enter(entry_side):
                        self.plot.marker("Blocked", self.mid_price, t,
                                         marker="x", color="#FF9800", size=8)
                        self.breakout_side    = 0
                        self.waiting_slowdown = False
                        return

                    self._enter(engine, entry_side, t)
                    self.waiting_slowdown = False

    # ═══════════════════════════════════════════════════════════
    # УПРАВЛЕНИЕ ПОЗИЦИЕЙ
    # ═══════════════════════════════════════════════════════════

    def _enter(self, engine, side: int, t: int):
        size = self.order_size * side
        oid  = engine.place_order("market", price=self.mid_price, size=size)

        if oid:
            self.in_position      = True
            self.position_side    = side
            self.entry_price      = self.mid_price
            self.first_entry_time = t
            self.add_count        = 0
            self.total_size       = self.order_size

            self._update_stops()
            self._place_take(engine)

            label = "LONG" if side == 1 else "SHORT"
            self.plot.marker(f"ENTRY {label}", self.mid_price, t,
                             marker="star",
                             color="#00E676" if side == 1 else "#FF5252",
                             size=15)

    def _manage(self, engine, inv: float, t: int):
        # Стоп-лосс
        hit_stop = (
            (self.position_side ==  1 and self.mid_price <= self.stop_price) or
            (self.position_side == -1 and self.mid_price >= self.stop_price)
        )
        if hit_stop:
            self._close(engine, inv, "STOP", t)
            return

        # Тейк-профит (лимитный ордер исполнился)
        if self.take_order_id:
            order = self._find_order(engine, self.take_order_id)
            if order and order.status == "filled":
                self.plot.marker("TAKE", self.take_price, t,
                                 marker="star", color="#00E676", size=15)
                self._reset()
                return

        # Добор в убыток (усреднение)
        if self.add_count < self.max_adds and abs(inv) < self.max_position:
            elapsed = t - self.first_entry_time
            delay   = self.add_delay_ms * (self.add_count + 1)

            if elapsed >= delay:
                thresh   = self.entry_price * (self.add_improve_pct / 100)
                improved = (
                    (self.position_side ==  1 and self.mid_price < self.entry_price - thresh) or
                    (self.position_side == -1 and self.mid_price > self.entry_price + thresh)
                )
                if improved and self._safe_to_enter(self.position_side):
                    self._add(engine, t)

        # Рисуем уровни
        self.plot.line("Stop", self.stop_price, t,
                       color="#FF0000", linewidth=2, linestyle="dotted", alpha=0.8)
        self.plot.line("Take", self.take_price, t,
                       color="#00E676", linewidth=2, linestyle="dotted", alpha=0.8)

    def _add(self, engine, t: int):
        size = self.order_size * self.position_side
        oid  = engine.place_order("market", price=self.mid_price, size=size)

        if oid:
            old_notional    = self.entry_price * self.total_size
            self.total_size += self.order_size
            self.entry_price = (old_notional + self.mid_price * self.order_size) / self.total_size
            self.add_count  += 1

            self._update_stops()
            self._update_take(engine)

            self.plot.marker("ADD", self.mid_price, t,
                             marker="triangle", color="#2196F3", size=10)

    def _close(self, engine, inv: float, reason: str, t: int):
        # Снимаем тейк
        if self.take_order_id:
            order = self._find_order(engine, self.take_order_id)
            if order and order.status in ("new", "partially_filled"):
                engine.cancel_order(self.take_order_id)

        engine.place_order("market", price=self.mid_price, size=-inv)

        pnl = (self.mid_price - self.entry_price) / self.entry_price * 100
        if self.position_side == -1:
            pnl = -pnl

        color = "#FF5252" if pnl < 0 else "#00E676"
        self.plot.marker(f"{reason} ({pnl:+.2f}%)", self.mid_price, t,
                         marker="x", color=color, size=15)
        self._reset()
        self.breakout_side = 0

    # ═══════════════════════════════════════════════════════════
    # ОРДЕРА / СТОПЫ / ТЕЙКИ
    # ═══════════════════════════════════════════════════════════

    def _update_stops(self):
        if self.position_side == 1:
            self.stop_price = self.entry_price * (1 - self.stop_pct / 100)
            self.take_price = self.entry_price * (1 + self.take_pct / 100)
        else:
            self.stop_price = self.entry_price * (1 + self.stop_pct / 100)
            self.take_price = self.entry_price * (1 - self.take_pct / 100)

    def _place_take(self, engine):
        size = -self.total_size if self.position_side == 1 else self.total_size
        self.take_order_id = engine.place_order("limit", price=self.take_price, size=size)

    def _update_take(self, engine):
        if self.take_order_id:
            order = self._find_order(engine, self.take_order_id)
            if order and order.status in ("new", "partially_filled"):
                engine.cancel_order(self.take_order_id)
        self._place_take(engine)

    # ═══════════════════════════════════════════════════════════
    # ВИЗУАЛИЗАЦИЯ
    # ═══════════════════════════════════════════════════════════

    def _draw(self, t: int):
        if self.cluster_low > 0:
            self.plot.band("Cluster", self.cluster_high, self.cluster_low, t,
                           color="#4CAF50", alpha=0.2)
            self.plot.line("POC", self.cluster_poc, t,
                           color="#9C27B0", linewidth=2, alpha=0.9)

        if self.in_position:
            self.plot.line("Entry", self.entry_price, t,
                           color="#FF9800", linewidth=2, linestyle="dotted", alpha=0.9)

    # ═══════════════════════════════════════════════════════════
    # ХЕЛПЕРЫ
    # ═══════════════════════════════════════════════════════════

    def _reset(self):
        self.in_position   = False
        self.position_side = 0
        self.entry_price   = 0.0
        self.add_count     = 0
        self.total_size    = 0.0
        self.take_order_id = None

    def _get_position(self, engine):
        for p in engine.positions:
            if p.status == "open" and p.size != 0:
                return p
        return None

    def _find_order(self, engine, oid: int):
        return engine.orders_by_id.get(oid)   # O(1) через словарь движка