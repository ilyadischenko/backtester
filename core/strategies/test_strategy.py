from typing import Dict
from collections import deque

import numpy as np
from numba import njit

from visualization.visualization import PlotRecorder


@njit(cache=True, fastmath=True, nogil=True)
def calc_volatility_std(prices: np.ndarray) -> float:
    n = prices.shape[0]
    if n < 2:
        return 0.0

    mean = 0.0
    for i in range(n):
        mean += prices[i]
    mean /= n

    var = 0.0
    for i in range(n):
        diff = prices[i] - mean
        var += diff * diff

    var /= (n - 1)
    return np.sqrt(var)


class AvellanedaStoikovMMSlope:
    def __init__(
        self,
        initial_balance: float = 1000.0,

        # AS‑параметры
        gamma: float = 0.9,
        k_spread_multiplier: float = 5.0,

        # Настройки
        volatility_window_sec: float = 30.0,
        min_spread_bps: float = 5.0,
        max_inventory: float = 1000.0,

        order_size: float = 100.0,
        recalc_interval_ms: int = 1000,
        quote_update_threshold_bps: float = 2.0,

        min_data_points: int = 100,
    ):
        self.initial_balance = initial_balance
        self.gamma = gamma
        self.k_spread = k_spread_multiplier
        self.vol_window_ms = int(volatility_window_sec * 1000)

        self.min_spread_bps = min_spread_bps
        self.max_inventory = max_inventory

        self.order_size = order_size
        self.recalc_interval_ms = recalc_interval_ms
        self.update_threshold = quote_update_threshold_bps
        self.min_data_points = min_data_points

        # Окно цен/времени
        self.price_times = deque()
        self.price_values = deque()

        # Состояние
        self.last_recalc_time = 0
        self.sigma = 0.0
        self.mid_price = 0.0

        # Активные ордера
        self.active_orders: Dict[str, int] = {}
        self.last_quote_prices: Dict[str, float] = {"buy": 0.0, "sell": 0.0}

        # ══════════════════════════════════════════════════════
        # PlotRecorder для визуализации
        # ══════════════════════════════════════════════════════
        self.plot = PlotRecorder()

        # Прогрев Numba
        _ = calc_volatility_std(np.zeros(2, dtype=np.float64))

    # ==================================================================
    #  Основной хук из движка
    # ==================================================================
    def on_tick(self, event, engine):
        current_time = event["event_time"]

        # 1) обновляем mid и окно цен
        if event.get("event_type") == "bookticker":
            bid = event["bid_price"]
            ask = event["ask_price"]
            self.mid_price = 0.5 * (bid + ask)

            self.price_times.append(current_time)
            self.price_values.append(self.mid_price)

            cutoff = current_time - self.vol_window_ms
            while self.price_times and self.price_times[0] < cutoff:
                self.price_times.popleft()
                self.price_values.popleft()

        # Ждём достаточного количества точек
        if len(self.price_values) < self.min_data_points:
            return

        # 2) раз в recalc_interval_ms пересчитываем котировки
        if current_time - self.last_recalc_time >= self.recalc_interval_ms:
            pos = self._get_position(engine)
            inventory = pos.size if pos else 0.0

            prices_np = np.array(self.price_values, dtype=np.float64)
            self.sigma = calc_volatility_std(prices_np)

            if self.sigma > 0.0 and self.mid_price > 0.0:
                self._update_quotes(engine, inventory, current_time)

            self.last_recalc_time = current_time

    # ==================================================================
    #  Avellaneda‑Stoikov логика
    # ==================================================================
    def _update_quotes(self, engine, inventory: float, current_time: int):
        """
        inventory > 0 -> лонг
        inventory < 0 -> шорт
        """
        # 1) reservation price
        reservation_price = self.mid_price - (inventory * self.gamma * (self.sigma ** 2))

        # 2) ширина спреда
        half_spread = 0.5 * self.sigma * self.k_spread
        min_half_spread = self.mid_price * (self.min_spread_bps / 10000.0) / 2.0
        half_spread = max(half_spread, min_half_spread)

        # 3) целевые цены
        target_bid = reservation_price - half_spread
        target_ask = reservation_price + half_spread

        # ══════════════════════════════════════════════════════
        # ЗАПИСЫВАЕМ ДАННЫЕ ДЛЯ ВИЗУАЛИЗАЦИИ
        # ══════════════════════════════════════════════════════
        
        # Reservation price (основная линия стратегии)
        self.plot.line(
            "Reservation Price",
            reservation_price,
            current_time,
            color="#2196F3",  # синий
            linewidth=1.5,
            alpha=0.9
        )
        
        # Spread band (полоса вокруг reservation)
        self.plot.band(
            "Spread Band",
            upper=target_ask,
            lower=target_bid,
            time=current_time,
            color="#9C27B0",  # фиолетовый
            alpha=0.15
        )
        
        # Целевые котировки (bid/ask линии)
        self.plot.line(
            "Target Bid",
            target_bid,
            current_time,
            color="#4CAF50",  # зелёный
            linewidth=1,
            linestyle="dashed",
            alpha=0.7
        )
        
        self.plot.line(
            "Target Ask",
            target_ask,
            current_time,
            color="#F44336",  # красный
            linewidth=1,
            linestyle="dashed",
            alpha=0.7
        )

        # 4) риск по инвентарю
        current_inv = inventory
        allow_buy = current_inv < self.max_inventory
        allow_sell = current_inv > -self.max_inventory

        # 5) управление ордерами
        self._manage_order_side(engine, "buy", target_bid, allow_buy, current_time)
        self._manage_order_side(engine, "sell", target_ask, allow_sell, current_time)

    # ==================================================================
    #  Работа с ордерами
    # ==================================================================
    def _manage_order_side(self, engine, side: str, target_price: float, allow: bool, current_time: int):
        order_id = self.active_orders.get(side)

        if not allow:
            if order_id:
                self._cancel_order(engine, side)
            return

        if not order_id:
            self._place_order(engine, side, target_price, current_time)
            return

        current_price = self.last_quote_prices.get(side, 0.0)
        if current_price == 0.0:
            return

        diff_bps = abs(target_price - current_price) / current_price * 10000.0

        if diff_bps > self.update_threshold:
            existing_order = self._find_order(engine, order_id)

            if existing_order and existing_order.status == "filled":
                self.active_orders.pop(side, None)
                self._place_order(engine, side, target_price, current_time)
            else:
                self._cancel_order(engine, side)
                self._place_order(engine, side, target_price, current_time)

    def _place_order(self, engine, side: str, price: float, current_time: int):
        qty = self.order_size
        if side == "sell":
            qty = -qty

        # Псевдо post-only
        if side == "buy" and price > self.mid_price:
            price = self.mid_price - 0.01
        if side == "sell" and price < self.mid_price:
            price = self.mid_price + 0.01

        oid = engine.place_order("limit", price=price, size=qty)
        if oid:
            self.active_orders[side] = oid
            self.last_quote_prices[side] = price
            
            # ══════════════════════════════════════════════════════
            # Маркер размещения ордера
            # ══════════════════════════════════════════════════════
            if side == "buy":
                self.plot.marker(
                    "Quote Placed (Buy)",
                    price,
                    current_time,
                    marker="triangle",
                    color="#00E676",  # ярко-зелёный
                    size=6
                )
            else:
                self.plot.marker(
                    "Quote Placed (Sell)",
                    price,
                    current_time,
                    marker="inverted_triangle",
                    color="#FF5252",  # ярко-красный
                    size=6
                )

    def _cancel_order(self, engine, side: str):
        oid = self.active_orders.get(side)
        if oid is None:
            return

        o = self._find_order(engine, oid)
        if o and o.status in ("new", "partially_filled"):
            engine.cancel_order(oid)

        self.active_orders.pop(side, None)

    # ==================================================================
    #  Вспомогательное
    # ==================================================================
    def _get_position(self, engine):
        for p in engine.positions:
            if p.status == "open" and p.size != 0:
                return p
        return None

    def _find_order(self, engine, oid: int):
        for o in engine.orders:
            if o.id == oid:
                return o
        return None