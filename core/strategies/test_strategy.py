from typing import Dict, Optional
from collections import deque

import numpy as np
from numba import njit

# ═══════════════════════════════════════════════════════════════════
# 1. NUMBA: Волатильность лог-доходностей (Log Returns)
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True, nogil=True)
def calc_volatility_log_returns(prices: np.ndarray) -> float:
    """
    Считает стандартное отклонение логарифмических доходностей.
    Возвращает волатильность за период семплирования (напр. за 1 тик/интервал).
    """
    n = prices.shape[0]
    if n < 2:
        return 0.0

    # 1. Считаем среднее лог-приращение (mean return)
    # Используем однопроходный алгоритм или классический в 2 прохода
    # Для точности и простоты сделаем 2 прохода по приращениям
    
    mean_ret = 0.0
    # Нам нужно (n-1) приращений
    count = n - 1
    
    # Предварительно считаем сумму доходностей
    for i in range(count):
        # ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
        ret = np.log(prices[i+1] / prices[i])
        mean_ret += ret
    
    mean_ret /= count

    # 2. Считаем дисперсию
    var = 0.0
    for i in range(count):
        ret = np.log(prices[i+1] / prices[i])
        diff = ret - mean_ret
        var += diff * diff

    var /= (count - 1) if count > 1 else 1.0
    
    return np.sqrt(var)


# ═══════════════════════════════════════════════════════════════════
# 2. Avellaneda‑Stoikov MM
# ═══════════════════════════════════════════════════════════════════

class AvellanedaStoikovMMNew:
    def __init__(
        self,
        initial_balance: float = 1000.0,

        # AS‑параметры
        gamma: float = 0.1,             # Коэффициент неприятия риска
        k_spread_multiplier: float = 0.5, # Множитель ширины спреда от волатильности

        # Настройки
        tick_size: float = 0.01,        # Шаг цены инструмента
        volatility_window_sec: float = 60.0,
        min_spread_bps: float = 0.1,
        max_inventory: float = 1000.0,  # Макс позиция в контрактах

        order_size: float = 100.0,
        recalc_interval_ms: int = 100,
        quote_update_threshold_bps: float = 0.1,

        min_data_points: int = 100,
    ):
        self.initial_balance = initial_balance
        self.gamma = gamma
        self.k_spread = k_spread_multiplier
        self.tick_size = tick_size
        
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
        self.sigma_abs = 0.0  # Абсолютная волатильность в валюте цены
        self.mid_price = 0.0

        # Активные ордера: логический side -> order_id
        self.active_orders: Dict[str, int] = {}
        self.last_quote_prices: Dict[str, float] = {"buy": 0.0, "sell": 0.0}

        # Прогрев Numba
        dummy_data = np.array([100.0, 101.0, 100.5], dtype=np.float64)
        _ = calc_volatility_log_returns(dummy_data)

    def _round_price(self, price: float) -> float:
        """Округляет цену до шага инструмента (tick_size)."""
        if self.tick_size <= 0:
            return price
        return round(price / self.tick_size) * self.tick_size

    # ==================================================================
    #  Основной хук из движка
    # ==================================================================
    def on_tick(self, event, engine):
        current_time = event["event_time"]

        # 1) Обновляем mid и окно цен
        if event.get("event_type") == "bookticker":
            bid = event["bid_price"]
            ask = event["ask_price"]
            # Защита от нулевых цен
            if bid > 0 and ask > 0:
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

        # 2) Пересчет раз в интервал
        if current_time - self.last_recalc_time >= self.recalc_interval_ms:
            pos = self._get_position(engine)
            inventory = pos.size if pos else 0.0

            prices_np = np.array(self.price_values, dtype=np.float64)
            
            # Считаем волатильность в процентах (log returns)
            vol_pct = calc_volatility_log_returns(prices_np)
            
            # Переводим в абсолютные значения (доллары) для формулы AS
            # sigma_abs = цена * волатильность%
            self.sigma_abs = vol_pct * self.mid_price

            if self.sigma_abs > 0.0 and self.mid_price > 0.0:
                self._update_quotes(engine, inventory)

            self.last_recalc_time = current_time

    # ==================================================================
    #  Avellaneda‑Stoikov логика
    # ==================================================================
    def _update_quotes(self, engine, inventory: float):
        """
        inventory > 0 -> лонг
        inventory < 0 -> шорт
        """
        # 1. Reservation price (r)
        # r = s - q * gamma * sigma^2
        # sigma должна быть абсолютной (в единицах цены), так как r и s в единицах цены
        reservation_shift = inventory * self.gamma * (self.sigma_abs ** 2)
        reservation_price = self.mid_price - reservation_shift

        # 2. Ширина спреда
        # В упрощенной модели: spread = k * sigma
        half_spread = 0.5 * self.sigma_abs * self.k_spread
        
        # Минимальный спред (защита)
        min_half_spread_val = self.mid_price * (self.min_spread_bps / 10000.0) / 2.0
        # Также спред не может быть меньше 1 тика
        half_spread = max(half_spread, min_half_spread_val, self.tick_size)

        # 3. Целевые цены
        raw_bid = reservation_price - half_spread
        raw_ask = reservation_price + half_spread

        # Округляем до tick_size
        target_bid = self._round_price(raw_bid)
        target_ask = self._round_price(raw_ask)

        # 4. Проверка на пересечение (Crossed Market Protection)
        # Bid должен быть строго меньше Ask.
        if target_bid >= target_ask:
            # Раздвигаем их симметрично или корректируем
            diff = target_bid - target_ask + 2 * self.tick_size
            target_bid -= diff / 2
            target_ask += diff / 2
            target_bid = self._round_price(target_bid)
            target_ask = self._round_price(target_ask)

        # Финальная защита (на случай если округление опять свело их)
        if target_bid >= target_ask:
            target_ask = target_bid + self.tick_size

        # 5. Управление инвентарем
        allow_buy = inventory < self.max_inventory
        allow_sell = inventory > -self.max_inventory

        self._manage_order_side(engine, "buy", target_bid, allow_buy)
        self._manage_order_side(engine, "sell", target_ask, allow_sell)

    # ==================================================================
    #  Работа с ордерами
    # ==================================================================
    def _manage_order_side(self, engine, side: str, target_price: float, allow: bool):
        order_id = self.active_orders.get(side)

        # Если торговать сторону нельзя - снимаем ордер
        if not allow:
            if order_id:
                self._cancel_order(engine, side)
            return

        # Если ордера нет - ставим
        if not order_id:
            self._place_order(engine, side, target_price)
            return

        # Проверка порога обновления (Hysteresis)
        current_price = self.last_quote_prices.get(side, 0.0)
        if current_price == 0.0:
            return

        # Считаем разницу в bps
        diff_bps = abs(target_price - current_price) / current_price * 10000.0
        
        # Также проверяем, что цена изменилась хотя бы на 1 тик
        diff_abs = abs(target_price - current_price)

        if diff_bps > self.update_threshold and diff_abs >= self.tick_size:
            existing_order = self._find_order(engine, order_id)

            if existing_order and existing_order.status == "filled":
                # Если уже исполнен, просто забываем старый ID и ставим новый
                self.active_orders.pop(side, None)
                self._place_order(engine, side, target_price)
            else:
                # Отменяем и ставим новый
                self._cancel_order(engine, side)
                self._place_order(engine, side, target_price)

    def _place_order(self, engine, side: str, price: float):
        qty = self.order_size
        if side == "sell":
            qty = -qty

        # Post-only эмуляция (защита от немедленного исполнения по маркету)
        # Если ставим Buy выше Mid или Sell ниже Mid -> двигаем к Mid
        if side == "buy" and price >= self.mid_price:
            price = self.mid_price - self.tick_size
        if side == "sell" and price <= self.mid_price:
            price = self.mid_price + self.tick_size
            
        price = self._round_price(price)

        oid = engine.place_order("limit", price=price, size=qty)
        if oid:
            self.active_orders[side] = oid
            self.last_quote_prices[side] = price

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
        """
        Линейный поиск ордера (без оптимизации, как просили).
        """
        for o in engine.orders:
            if o.id == oid:
                return o
        return None