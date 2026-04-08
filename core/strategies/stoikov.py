from typing import Optional
from collections import deque
import numpy as np
from numba import njit

from core.visualization.plot_recorder import PlotRecorder


# ══════════════════════════════════════════════════════════════
# NUMBA — быстрые вычисления
# ══════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True, nogil=True)
def calc_volatility(prices: np.ndarray, window: int) -> float:
    """Волатильность как std log-returns на последних window тиках."""
    n = len(prices)
    if n < 2:
        return 0.0
    subset = prices[-window:] if n > window else prices
    diffs = np.diff(subset)
    means = subset[:-1]
    returns = diffs / means
    return np.std(returns)


@njit(cache=True, fastmath=True, nogil=True)
def calc_reservation_price(
    mid: float, q: float, gamma: float, sigma: float, T: float, t_frac: float
) -> float:
    """
    Классическая формула Avellaneda-Stoikov (2008):
        r = s - q * gamma * sigma^2 * (T - t)

    s      — mid price
    q      — нормализованный инвентарь (позиция / order_amount)
    gamma  — неприятие риска
    sigma  — волатильность
    T - t  — оставшееся "время" (здесь — нормированная длина сессии)
    """
    return mid - q * gamma * sigma * sigma * (T - t_frac)


@njit(cache=True, fastmath=True, nogil=True)
def calc_half_spread(
    gamma: float, sigma: float, k: float, T: float, t_frac: float
) -> float:
    """
    Оптимальный полуспред Avellaneda-Stoikov:
        delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

    k — интенсивность пришествия ордеров (order arrival rate).
    Чем меньше k, тем меньше ордеров приходит → спред шире.
    """
    time_term = gamma * sigma * sigma * (T - t_frac)
    # Защита от вырождения при gamma -> 0
    if gamma < 1e-10:
        arrival_term = 1.0 / k if k > 1e-10 else 0.0
    else:
        arrival_term = (2.0 / gamma) * np.log(1.0 + gamma / k) if k > 1e-10 else 0.0
    half = time_term + arrival_term
    return half


@njit(cache=True, fastmath=True, nogil=True)
def calc_book_imbalance_depth(
    bid_prices: np.ndarray,
    bid_qtys: np.ndarray,
    ask_prices: np.ndarray,
    ask_qtys: np.ndarray,
    depth_levels: int,
) -> float:
    """
    Взвешенный имбаланс стакана по depth_levels уровням.
    Возвращает [-1, 1]:  > 0 → давление покупателей, < 0 → продавцов.

    Вес уровня убывает линейно: top-of-book весит depth_levels,
    последний уровень — 1. Так top-of-book важнее.
    """
    n_bid = min(len(bid_prices), depth_levels)
    n_ask = min(len(ask_prices), depth_levels)

    bid_vol = 0.0
    ask_vol = 0.0

    for i in range(n_bid):
        w = float(depth_levels - i)          # линейный спад веса
        bid_vol += w * bid_qtys[n_bid - 1 - i]   # bids отсортированы по убыванию

    for i in range(n_ask):
        w = float(depth_levels - i)
        ask_vol += w * ask_qtys[i]

    total = bid_vol + ask_vol
    if total < 1e-12:
        return 0.0
    return (bid_vol - ask_vol) / total


# ══════════════════════════════════════════════════════════════
# СТРАТЕГИЯ
# ══════════════════════════════════════════════════════════════

class AvellanedaStoikov:
    """
    Market-making стратегия Avellaneda-Stoikov для движка ExchangeEngine.

    Формат событий (event_type):
        "ob_snapshot" — полный снапшот стакана (b_p, b_q, a_p, a_q)
        "depth"       — инкрементальное обновление стакана
        "trade"       — сделка (p, q, t)

    Стакан доступен через engine.ob (OrderBook с SortedDict).

    Логика запускается на каждом depth/snapshot тике (mid_price меняется).
    trade-события используются только для обновления arrival_rate.

    Параметры:
        risk_aversion   — gamma: интенсивность штрафа за инвентарь
        order_amount    — размер каждого ордера в базовой валюте (монеты)
        max_inventory   — макс. абс. позиция (монеты). Выше — ордер со стороны
                          перекоса не выставляется.
        arrival_rate    — k: оценочная интенсивность исполнения ордеров.
                          Можно подбирать по истории или оставить как гиперпараметр.
        session_seconds — длина торговой сессии в секундах (для T - t).
                          Не критично: меняет только масштаб спреда.
        vol_window      — кол-во тиков mid_price для расчёта волатильности
        min_half_spread_pct — нижний клэмп полуспреда (% от mid). Защита от нуля.
        max_half_spread_pct — верхний клэмп (защита от взрывного спреда при vol spike)
        imbalance_weight    — как сильно имбаланс стакана сдвигает reservation price
        imbalance_depth     — сколько уровней стакана брать для имбаланса
        requote_threshold_pct — минимальный сдвиг цены для перевыставления (% от mid)
    """

    def __init__(
        self,
        risk_aversion: float = 0.3,
        order_amount: float = 0.01,
        max_inventory: float = 0.1,
        arrival_rate: float = 1.5,
        session_seconds: float = 3600.0,
        vol_window: int = 200,
        min_half_spread_pct: float = 0.02,
        max_half_spread_pct: float = 0.5,
        imbalance_weight: float = 0.3,
        imbalance_depth: int = 10,
        requote_threshold_pct: float = 0.005,
    ):
        self.gamma            = risk_aversion
        self.order_amount     = order_amount
        self.max_inventory    = max_inventory
        self.k                = arrival_rate
        self.session_ms       = session_seconds * 1000.0
        self.vol_window       = vol_window
        self.min_half_spread  = min_half_spread_pct / 100.0
        self.max_half_spread  = max_half_spread_pct / 100.0
        self.imbalance_weight = imbalance_weight
        self.imbalance_depth  = imbalance_depth
        self.requote_thr      = requote_threshold_pct / 100.0

        self.plot = PlotRecorder()

        # Буфер цен (deque → numpy при расчёте)
        self.mid_prices: deque = deque(maxlen=max(vol_window * 2, 500))

        # Состояние
        self.mid_price: float  = 0.0
        self.volatility: float = 0.0
        self.inventory: float  = 0.0
        self.imbalance: float  = 0.0

        # Тайминг сессии
        self._session_start_ms: Optional[int] = None

        # Активные ордера
        self.bid_oid: Optional[int] = None
        self.ask_oid: Optional[int] = None
        self._bid_price: float = 0.0
        self._ask_price: float = 0.0

        # Счётчики для arrival_rate (опционально — авто-оценка)
        self._trade_count: int = 0
        self._trade_window_start_ms: Optional[int] = None
        self._use_dynamic_k: bool = False  # True → k оценивается динамически

        # Прогрев Numba (первый вызов компилирует)
        _warm = np.array([100.0, 101.0, 100.5, 100.8, 101.2], dtype=np.float64)
        calc_volatility(_warm, 5)
        calc_reservation_price(100.0, 0.0, 0.1, 0.01, 1.0, 0.5)
        calc_half_spread(0.1, 0.01, 1.5, 1.0, 0.5)
        calc_book_imbalance_depth(
            np.array([99.0, 98.0]), np.array([1.0, 1.0]),
            np.array([101.0, 102.0]), np.array([1.0, 1.0]), 2
        )

    # ──────────────────────────────────────────────────────────
    # ENTRY POINT
    # ──────────────────────────────────────────────────────────

    def on_tick(self, event: dict, engine) -> None:
        t = event["event_time"]
        etype = event.get("event_type")

        # Инициализируем таймер сессии при первом тике
        if self._session_start_ms is None:
            self._session_start_ms = t

        if etype in ("ob_snapshot", "depth"):
            self._on_book_update(engine, t)

        elif etype == "trade":
            self._on_trade(event, t)

    # ──────────────────────────────────────────────────────────
    # ОБНОВЛЕНИЕ СТАКАНА
    # ──────────────────────────────────────────────────────────

    def _on_book_update(self, engine, t: int) -> None:
        ob = engine.ob
        if not ob.is_ready:
            return

        bid = ob.best_bid
        ask = ob.best_ask
        self.mid_price = (bid + ask) / 2.0
        self.mid_prices.append(self.mid_price)

        # Имбаланс из реального стакана
        self.imbalance = self._calc_imbalance(ob)

        # Волатильность
        if len(self.mid_prices) >= 2:
            arr = np.array(self.mid_prices, dtype=np.float64)
            self.volatility = calc_volatility(arr, self.vol_window)

        self._run_logic(engine, t)

    # ──────────────────────────────────────────────────────────
    # ОБНОВЛЕНИЕ ТРЕЙДОВ (arrival rate)
    # ──────────────────────────────────────────────────────────

    def _on_trade(self, event: dict, t: int) -> None:
        if not self._use_dynamic_k:
            return
        self._trade_count += 1
        if self._trade_window_start_ms is None:
            self._trade_window_start_ms = t
        elapsed_s = (t - self._trade_window_start_ms) / 1000.0
        if elapsed_s >= 60.0:  # пересчитываем каждые 60 сек
            self.k = max(0.1, self._trade_count / elapsed_s)
            self._trade_count = 0
            self._trade_window_start_ms = t

    # ──────────────────────────────────────────────────────────
    # ИМБАЛАНС ИЗ ПОЛНОГО СТАКАНА
    # ──────────────────────────────────────────────────────────

    def _calc_imbalance(self, ob) -> float:
        depth = self.imbalance_depth

        # Берём N лучших уровней (bids отсортированы по возрастанию — берём с конца)
        bid_keys = list(ob.bids.keys())
        ask_keys = list(ob.asks.keys())

        # Лучшие биды — с конца (наибольшие), лучшие аски — с начала (наименьшие)
        n_bid = min(depth, len(bid_keys))
        n_ask = min(depth, len(ask_keys))

        if n_bid == 0 and n_ask == 0:
            return 0.0

        bid_prices = np.array(bid_keys[-n_bid:], dtype=np.float64)
        bid_qtys   = np.array([ob.bids[p] for p in bid_keys[-n_bid:]], dtype=np.float64)
        ask_prices = np.array(ask_keys[:n_ask], dtype=np.float64)
        ask_qtys   = np.array([ob.asks[p] for p in ask_keys[:n_ask]], dtype=np.float64)

        # Дополняем до depth нулями для Numba (фиксированный размер)
        if n_bid < depth:
            bid_prices = np.concatenate([np.zeros(depth - n_bid), bid_prices])
            bid_qtys   = np.concatenate([np.zeros(depth - n_bid), bid_qtys])
        if n_ask < depth:
            ask_prices = np.concatenate([ask_prices, np.zeros(depth - n_ask)])
            ask_qtys   = np.concatenate([ask_qtys, np.zeros(depth - n_ask)])

        return calc_book_imbalance_depth(
            bid_prices, bid_qtys, ask_prices, ask_qtys, depth
        )

    # ──────────────────────────────────────────────────────────
    # ОСНОВНАЯ ЛОГИКА (каждый book-тик)
    # ──────────────────────────────────────────────────────────

    def _run_logic(self, engine, t: int) -> None:
        if len(self.mid_prices) < 10:
            return

        # Текущая позиция
        pos = engine.get_position()
        self.inventory = pos.size if pos else 0.0

        # Нормализованный инвентарь (в единицах order_amount)
        q = self.inventory / self.order_amount if self.order_amount > 1e-12 else 0.0

        # Оставшееся "время" сессии [0, 1]
        elapsed_ms = t - self._session_start_ms
        t_frac = min(elapsed_ms / self.session_ms, 1.0) if self.session_ms > 0 else 0.5
        T = 1.0  # горизонт нормирован в [0, 1]

        sigma = max(self.volatility, 1e-8)

        # ── 1. Reservation price ──────────────────────────────
        res_price = calc_reservation_price(
            self.mid_price, q, self.gamma, sigma, T, t_frac
        )

        # ── 2. Корректировка на имбаланс стакана ─────────────
        # imbalance ∈ [-1, 1]: положительный → давление вверх
        imb_shift = self.imbalance * self.imbalance_weight * self.mid_price * self.min_half_spread
        res_price += imb_shift

        # ── 3. Оптимальный полуспред ──────────────────────────
        raw_half = calc_half_spread(self.gamma, sigma, self.k, T, t_frac)

        # Клэмп в разумные пределы
        min_hs = self.mid_price * self.min_half_spread
        max_hs = self.mid_price * self.max_half_spread
        half_spread = max(min_hs, min(raw_half * self.mid_price, max_hs))

        # ── 4. Целевые цены ───────────────────────────────────
        target_bid = res_price - half_spread
        target_ask = res_price + half_spread

        # ── 5. Управление ордерами ────────────────────────────
        self._manage_quotes(engine, target_bid, target_ask)

        self.plot.line("reservation_price", res_price, t, color="#FFD700", label="Reservation price")
        self.plot.band("spread_band", target_ask, target_bid, t, color="#4fc3f7", alpha=0.15)
        self.plot.line("bid_quote", target_bid, t, color="#00e676", linestyle="dashed", label="Bid quote")
        self.plot.line("ask_quote", target_ask, t, color="#ff5252", linestyle="dashed", label="Ask quote")

        self._manage_quotes(engine, target_bid, target_ask)

    # ──────────────────────────────────────────────────────────
    # УПРАВЛЕНИЕ КОТИРОВКАМИ
    # ──────────────────────────────────────────────────────────

    def _manage_quotes(self, engine, target_bid: float, target_ask: float) -> None:
        self._sync_order_state(engine)

        # ── BID ──────────────────────────────────────────────
        can_buy = self.inventory < self.max_inventory

        if can_buy:
            if self.bid_oid is None:
                self._place_bid(engine, target_bid)
            else:
                delta = abs(self._bid_price - target_bid) / self.mid_price
                if delta > self.requote_thr:
                    engine.cancel_order(self.bid_oid)
                    self.bid_oid = None
                    self._place_bid(engine, target_bid)
        else:
            if self.bid_oid is not None:
                engine.cancel_order(self.bid_oid)
                self.bid_oid = None

        # ── ASK ──────────────────────────────────────────────
        can_sell = self.inventory > -self.max_inventory

        if can_sell:
            if self.ask_oid is None:
                self._place_ask(engine, target_ask)
            else:
                delta = abs(self._ask_price - target_ask) / self.mid_price
                if delta > self.requote_thr:
                    engine.cancel_order(self.ask_oid)
                    self.ask_oid = None
                    self._place_ask(engine, target_ask)
        else:
            if self.ask_oid is not None:
                engine.cancel_order(self.ask_oid)
                self.ask_oid = None

    # ──────────────────────────────────────────────────────────
    # СИНХРОНИЗАЦИЯ — проверяем, не исполнился ли ордер
    # ──────────────────────────────────────────────────────────

    def _sync_order_state(self, engine) -> None:
        """
        Движок не уведомляет стратегию об исполнении.
        Проверяем статус через orders_by_id.
        """
        if self.bid_oid is not None:
            o = engine.orders_by_id.get(self.bid_oid)
            if o is None or o.status in ("filled", "canceled"):
                self.bid_oid = None
                self._bid_price = 0.0

        if self.ask_oid is not None:
            o = engine.orders_by_id.get(self.ask_oid)
            if o is None or o.status in ("filled", "canceled"):
                self.ask_oid = None
                self._ask_price = 0.0

    # ──────────────────────────────────────────────────────────
    # РАЗМЕЩЕНИЕ ОРДЕРОВ
    # ──────────────────────────────────────────────────────────

    def _place_bid(self, engine, price: float) -> None:
        if price <= 0:
            return
        oid = engine.place_order("limit", price=price, size=self.order_amount)
        self.bid_oid = oid
        self._bid_price = price

    def _place_ask(self, engine, price: float) -> None:
        if price <= 0:
            return
        oid = engine.place_order("limit", price=price, size=-self.order_amount)
        self.ask_oid = oid
        self._ask_price = price

    # ──────────────────────────────────────────────────────────
    # УТИЛИТЫ
    # ──────────────────────────────────────────────────────────

    def enable_dynamic_arrival_rate(self) -> None:
        """Включить авто-оценку k из потока трейдов."""
        self._use_dynamic_k = True

    def get_state(self) -> dict:
        return {
            "mid_price":  self.mid_price,
            "volatility": self.volatility,
            "inventory":  self.inventory,
            "imbalance":  self.imbalance,
            "bid_oid":    self.bid_oid,
            "ask_oid":    self.ask_oid,
            "bid_price":  self._bid_price,
            "ask_price":  self._ask_price,
        }