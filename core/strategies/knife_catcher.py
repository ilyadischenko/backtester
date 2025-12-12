# strategies/knife_catcher_ultra_fast.py
from typing import Optional, List
import numpy as np
from numba import njit


# ═══════════════════════════════════════════════════════════════════
# NUMBA ФУНКЦИИ - работают с circular buffer напрямую
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def calc_volatility_circular(
        prices: np.ndarray,
        times: np.ndarray,
        head: int,
        size: int,
        buffer_size: int,
        current_time: int,
        window_ms: int
) -> float:
    """Волатильность с circular buffer БЕЗ копирований."""
    if size < 2:
        return 0.0

    cutoff = current_time - window_ms

    # Считаем среднее и количество валидных точек
    mean = 0.0
    count = 0

    for i in range(size):
        idx = (head - size + i) % buffer_size
        if times[idx] >= cutoff:
            mean += prices[idx]
            count += 1

    if count < 2:
        return 0.0

    mean /= count

    if mean <= 0:
        return 0.0

    # Дисперсия
    variance = 0.0
    for i in range(size):
        idx = (head - size + i) % buffer_size
        if times[idx] >= cutoff:
            diff = prices[idx] - mean
            variance += diff * diff

    variance /= count
    std = np.sqrt(variance)
    return (std / mean) * 10000.0


@njit(cache=True, fastmath=True)
def calc_price_change_circular(
        prices: np.ndarray,
        times: np.ndarray,
        head: int,
        size: int,
        buffer_size: int,
        current_time: int,
        window_ms: int
) -> tuple:
    """Изменение цены с circular buffer. Возвращает (change_bps, is_falling)."""
    if size < 2:
        return 0.0, False

    cutoff = current_time - window_ms

    # Найти первую и последнюю валидные точки
    first_price = 0.0
    last_price = 0.0
    found_first = False

    for i in range(size):
        idx = (head - size + i) % buffer_size
        if times[idx] >= cutoff:
            if not found_first:
                first_price = prices[idx]
                found_first = True
            last_price = prices[idx]

    if not found_first or first_price <= 0:
        return 0.0, False

    change_bps = (last_price - first_price) / first_price * 10000.0
    return change_bps, change_bps < 0


@njit(cache=True, fastmath=True)
def calc_imbalance_circular(
        is_sell: np.ndarray,
        volumes: np.ndarray,
        times: np.ndarray,
        head: int,
        size: int,
        buffer_size: int,
        current_time: int,
        window_ms: int
) -> float:
    """Имбаланс с circular buffer."""
    if size == 0:
        return 0.5

    cutoff = current_time - window_ms

    sell_vol = 0.0
    total_vol = 0.0

    for i in range(size):
        idx = (head - size + i) % buffer_size
        if times[idx] >= cutoff:
            total_vol += volumes[idx]
            if is_sell[idx]:
                sell_vol += volumes[idx]

    if total_vol <= 0:
        return 0.5

    return sell_vol / total_vol


@njit(cache=True, fastmath=True)
def calc_avg_volatility(vol_values: np.ndarray, vol_size: int) -> float:
    """Средняя волатильность."""
    if vol_size == 0:
        return 0.0

    sum_vol = 0.0
    count = 0

    for i in range(vol_size):
        if vol_values[i] > 0:
            sum_vol += vol_values[i]
            count += 1

    if count == 0:
        return 0.0

    return sum_vol / count


@njit(cache=True, fastmath=True)
def compute_all_indicators_circular(
        price_times: np.ndarray,
        price_values: np.ndarray,
        price_head: int,
        price_size: int,
        price_buffer_size: int,
        trade_times: np.ndarray,
        trade_is_sell: np.ndarray,
        trade_volumes: np.ndarray,
        trade_head: int,
        trade_size: int,
        trade_buffer_size: int,
        vol_values: np.ndarray,
        vol_size: int,
        current_time: int,
        vol_window_ms: int,
        calm_window_ms: int,
        imbalance_window_ms: int
) -> tuple:
    """
    Все индикаторы за один проход с circular buffers.
    Возвращает: (volatility, price_change, is_falling, avg_volatility, sell_ratio)
    """

    # Волатильность
    volatility = calc_volatility_circular(
        price_values, price_times,
        price_head, price_size, price_buffer_size,
        current_time, vol_window_ms
    )

    # Изменение цены
    price_change, is_falling = calc_price_change_circular(
        price_values, price_times,
        price_head, price_size, price_buffer_size,
        current_time, calm_window_ms
    )

    # Средняя волатильность
    avg_volatility = calc_avg_volatility(vol_values, vol_size)

    # Имбаланс
    sell_ratio = calc_imbalance_circular(
        trade_is_sell, trade_volumes, trade_times,
        trade_head, trade_size, trade_buffer_size,
        current_time, imbalance_window_ms
    )

    return volatility, price_change, is_falling, avg_volatility, sell_ratio


# ═══════════════════════════════════════════════════════════════════
# СТРАТЕГИЯ
# ═══════════════════════════════════════════════════════════════════

class KnifeCatcherUltraFast:
    """
    Ultra-fast версия с настоящим circular buffer.
    НИ ОДНОГО копирования массивов.
    """

    def __init__(
            self,
            initial_balance: float = 10000.0,
            market_order_size_usd: float = 100.0,
            limit_order_1_size_usd: float = 100.0,
            limit_order_2_size_usd: float = 100.0,
            volatility_window_sec: float = 30.0,
            volatility_spike_pct: float = 150.0,
            volatility_lookback_sec: float = 300.0,
            imbalance_window_sec: float = 5.0,
            imbalance_sell_threshold: float = 0.65,
            imbalance_neutral_threshold: float = 0.55,
            calm_window_sec: float = 2.0,
            calm_price_change_bps: float = 5.0,
            min_knife_duration_sec: float = 1.0,
            max_knife_duration_sec: float = 60.0,
            limit_order_1_offset_bps: float = 20.0,
            limit_order_2_offset_bps: float = 40.0,
            stop_loss_bps: float = 60.0,
            take_profit_bps: float = 30.0,
            cooldown_after_close_sec: float = 10.0,
            min_data_points: int = 100,
            # Оптимальные параметры
            indicator_update_interval_ms: int = 50,  # Обновляем раз в 50ms
            price_buffer_size: int = 50_000,  # Храним только нужное окно
            trade_buffer_size: int = 10_000,
            vol_buffer_size: int = 1000,
    ):
        # Параметры
        self.market_order_size_usd = market_order_size_usd
        self.limit_order_1_size_usd = limit_order_1_size_usd
        self.limit_order_2_size_usd = limit_order_2_size_usd

        self.volatility_window_ms = int(volatility_window_sec * 1000)
        self.volatility_spike_pct = volatility_spike_pct
        self.volatility_lookback_ms = int(volatility_lookback_sec * 1000)

        self.imbalance_window_ms = int(imbalance_window_sec * 1000)
        self.imbalance_sell_threshold = imbalance_sell_threshold
        self.imbalance_neutral_threshold = imbalance_neutral_threshold

        self.calm_window_ms = int(calm_window_sec * 1000)
        self.calm_price_change_bps = calm_price_change_bps
        self.min_knife_duration_ms = int(min_knife_duration_sec * 1000)
        self.max_knife_duration_ms = int(max_knife_duration_sec * 1000)

        self.limit_order_1_offset_bps = limit_order_1_offset_bps
        self.limit_order_2_offset_bps = limit_order_2_offset_bps
        self.stop_loss_bps = stop_loss_bps
        self.take_profit_bps = take_profit_bps

        self.cooldown_ms = int(cooldown_after_close_sec * 1000)
        self.min_data_points = min_data_points

        self.indicator_update_interval_ms = indicator_update_interval_ms
        self.last_indicator_update_time = 0

        # ═══════════════════════════════════════════════════════
        # НАСТОЯЩИЙ Circular buffer - только head pointer
        # ═══════════════════════════════════════════════════════

        # Цены
        self.price_buffer_size = price_buffer_size
        self.price_times = np.zeros(price_buffer_size, dtype=np.int64)
        self.price_values = np.zeros(price_buffer_size, dtype=np.float64)
        self.price_head = 0  # Указатель на последний элемент
        self.price_size = 0  # Текущее количество элементов (до buffer_size)

        # Трейды
        self.trade_buffer_size = trade_buffer_size
        self.trade_times = np.zeros(trade_buffer_size, dtype=np.int64)
        self.trade_is_sell = np.zeros(trade_buffer_size, dtype=np.bool_)
        self.trade_volumes = np.zeros(trade_buffer_size, dtype=np.float64)
        self.trade_head = 0
        self.trade_size = 0

        # Волатильность (компактный буфер)
        self.vol_buffer_size = vol_buffer_size
        self.vol_values = np.zeros(vol_buffer_size, dtype=np.float64)
        self.vol_head = 0
        self.vol_size = 0

        # Состояние
        self.state: str = "IDLE"
        self.knife_start_time: int = 0
        self.knife_low_price: float = 0.0

        self.limit_order_ids: List[int] = []
        self.tp_order_id: Optional[int] = None
        self.last_tp_price: float = 0.0
        self.entry_price: float = 0.0

        self.last_close_time: int = 0
        self.data_points_count: int = 0

        self.current_bid: float = 0.0
        self.current_ask: float = 0.0
        self.current_mid: float = 0.0

        # Кэшированные индикаторы
        self._volatility: float = 0.0
        self._avg_volatility: float = 0.0
        self._sell_ratio: float = 0.5
        self._price_change: float = 0.0
        self._is_falling: bool = False

        # Кэш позиции
        self._cached_position = None
        self._position_cache_valid = False

    # ═══════════════════════════════════════════════════════════════
    # ГЛАВНЫЙ ОБРАБОТЧИК
    # ═══════════════════════════════════════════════════════════════

    def on_tick(self, event, engine):
        """Обрабатываем тик с НУЛЕВЫМИ копированиями."""
        event_type = event.get("event_type")
        current_time = event["event_time"]

        # Записываем данные (O(1) операция)
        if event_type == "bookticker":
            self._record_price_fast(event, current_time)
        elif event_type == "trade":
            self._record_trade_fast(event, current_time)

        self.data_points_count += 1
        if self.data_points_count < self.min_data_points:
            return

        # Обновляем индикаторы только периодически
        if current_time - self.last_indicator_update_time >= self.indicator_update_interval_ms:
            self._compute_indicators_ultra_fast(current_time)
            self.last_indicator_update_time = current_time

        # Проверяем позицию (кэш)
        if not self._position_cache_valid:
            self._cached_position = self._get_open_position(engine)
            self._position_cache_valid = True

        pos = self._cached_position
        pos_size = pos.size if pos else 0.0

        if pos_size > 0:
            self.state = "IN_POSITION"
            self._manage_position(engine, pos, current_time)
            return

        if self.state == "IN_POSITION" and pos_size == 0:
            self._on_position_closed(engine, current_time)
            return

        # Кулдаун
        if current_time - self.last_close_time < self.cooldown_ms:
            return

        # Машина состояний
        self._run_state_machine(engine, current_time)

    # ═══════════════════════════════════════════════════════════════
    # CIRCULAR BUFFER - O(1) запись без копирований
    # ═══════════════════════════════════════════════════════════════

    def _record_price_fast(self, event, current_time: int):
        """Записываем цену в circular buffer - O(1)."""
        self.current_bid = event["bid_price"]
        self.current_ask = event["ask_price"]
        self.current_mid = (self.current_bid + self.current_ask) / 2

        # Пишем в текущую позицию
        idx = self.price_head % self.price_buffer_size
        self.price_times[idx] = current_time
        self.price_values[idx] = self.current_mid

        # Двигаем указатель
        self.price_head += 1
        if self.price_size < self.price_buffer_size:
            self.price_size += 1

    def _record_trade_fast(self, event, current_time: int):
        """Записываем трейд в circular buffer - O(1)."""
        price = event.get("price", 0)
        qty = event.get("quantity", 0)
        is_sell = event.get("is_buyer_maker", False)

        idx = self.trade_head % self.trade_buffer_size
        self.trade_times[idx] = current_time
        self.trade_is_sell[idx] = is_sell
        self.trade_volumes[idx] = price * qty

        self.trade_head += 1
        if self.trade_size < self.trade_buffer_size:
            self.trade_size += 1

    # ═══════════════════════════════════════════════════════════════
    # ВЫЧИСЛЕНИЕ ИНДИКАТОРОВ
    # ═══════════════════════════════════════════════════════════════

    def _compute_indicators_ultra_fast(self, current_time: int):
        """Быстрое вычисление через Numba с circular buffer."""

        if self.price_size < 2:
            return

        # Сохраняем волатильность в circular buffer
        if self._volatility > 0:
            vol_idx = self.vol_head % self.vol_buffer_size
            self.vol_values[vol_idx] = self._volatility
            self.vol_head += 1
            if self.vol_size < self.vol_buffer_size:
                self.vol_size += 1

        # Один вызов Numba - все индикаторы
        result = compute_all_indicators_circular(
            self.price_times,
            self.price_values,
            self.price_head,
            self.price_size,
            self.price_buffer_size,
            self.trade_times,
            self.trade_is_sell,
            self.trade_volumes,
            self.trade_head,
            self.trade_size,
            self.trade_buffer_size,
            self.vol_values,
            self.vol_size,
            current_time,
            self.volatility_window_ms,
            self.calm_window_ms,
            self.imbalance_window_ms
        )

        (self._volatility,
         self._price_change,
         self._is_falling,
         self._avg_volatility,
         self._sell_ratio) = result

    # ═══════════════════════════════════════════════════════════════
    # МАШИНА СОСТОЯНИЙ
    # ═══════════════════════════════════════════════════════════════

    def _run_state_machine(self, engine, current_time: int):
        if self.state == "IDLE":
            if self._detect_knife():
                self.state = "KNIFE_DETECTED"
                self.knife_start_time = current_time
                self.knife_low_price = self.current_mid

        elif self.state == "KNIFE_DETECTED":
            if self.current_mid < self.knife_low_price:
                self.knife_low_price = self.current_mid

            knife_duration = current_time - self.knife_start_time
            if knife_duration > self.max_knife_duration_ms:
                self._reset_knife_state()
                return

            if knife_duration >= self.min_knife_duration_ms:
                if self._detect_calm():
                    self.state = "WAITING_CALM"

        elif self.state == "WAITING_CALM":
            if self._confirm_calm():
                self._enter_position(engine, current_time)
            elif self._detect_knife():
                self.state = "KNIFE_DETECTED"
                if self.current_mid < self.knife_low_price:
                    self.knife_low_price = self.current_mid

    def _detect_knife(self) -> bool:
        if self._avg_volatility <= 0:
            return False
        vol_spike = (self._volatility / self._avg_volatility - 1) * 100
        if vol_spike < self.volatility_spike_pct:
            return False
        if self._sell_ratio < self.imbalance_sell_threshold:
            return False
        if not self._is_falling:
            return False
        return True

    def _detect_calm(self) -> bool:
        return not self._is_falling

    def _confirm_calm(self) -> bool:
        if abs(self._price_change) > self.calm_price_change_bps:
            return False
        if self._sell_ratio > self.imbalance_neutral_threshold:
            return False
        return True

    # ═══════════════════════════════════════════════════════════════
    # УПРАВЛЕНИЕ ПОЗИЦИЕЙ
    # ═══════════════════════════════════════════════════════════════

    def _enter_position(self, engine, current_time: int):
        self.entry_price = self.current_ask

        market_size = self.market_order_size_usd / self.entry_price
        engine.place_order("market", price=0, size=market_size)

        limit_price_1 = self.entry_price * (1 - self.limit_order_1_offset_bps / 10000)
        limit_size_1 = self.limit_order_1_size_usd / limit_price_1
        order_1_id = engine.place_order("limit", price=limit_price_1, size=limit_size_1)
        if order_1_id:
            self.limit_order_ids.append(order_1_id)

        limit_price_2 = self.entry_price * (1 - self.limit_order_2_offset_bps / 10000)
        limit_size_2 = self.limit_order_2_size_usd / limit_price_2
        order_2_id = engine.place_order("limit", price=limit_price_2, size=limit_size_2)
        if order_2_id:
            self.limit_order_ids.append(order_2_id)

        self.state = "IN_POSITION"
        self._reset_knife_state()

    def _manage_position(self, engine, pos, current_time: int):
        avg_price = pos.price
        pos_size = pos.size

        tp_price = avg_price * (1 + self.take_profit_bps / 10000)

        if self.tp_order_id is None:
            self._place_take_profit(engine, tp_price, pos_size)
        else:
            self._update_take_profit(engine, tp_price, pos_size)

        sl_price = avg_price * (1 - self.stop_loss_bps / 10000)

        if self.current_bid <= sl_price:
            self._execute_stop_loss(engine, pos_size, current_time)
            return

        if self.tp_order_id:
            tp_order = self._find_order_by_id(engine, self.tp_order_id)
            if tp_order and tp_order.status == "filled":
                self._on_position_closed(engine, current_time)

    def _place_take_profit(self, engine, tp_price: float, pos_size: float):
        self.tp_order_id = engine.place_order("limit", price=tp_price, size=-pos_size)
        if self.tp_order_id:
            self.last_tp_price = tp_price

    def _update_take_profit(self, engine, new_tp_price: float, pos_size: float):
        if not self.tp_order_id:
            return

        if self.last_tp_price > 0:
            diff_bps = abs(new_tp_price - self.last_tp_price) / self.last_tp_price * 10000
            if diff_bps < 0.5:
                return

        tp_order = self._find_order_by_id(engine, self.tp_order_id)
        if not tp_order or tp_order.status == "filled":
            return

        if tp_order.status == "new":
            engine.cancel_order(self.tp_order_id)
            self._place_take_profit(engine, new_tp_price, pos_size)

    def _execute_stop_loss(self, engine, pos_size: float, current_time: int):
        engine.place_order("market", price=0, size=-pos_size)
        self._on_position_closed(engine, current_time)

    def _on_position_closed(self, engine, current_time: int):
        self._cancel_all_orders(engine)
        self.last_close_time = current_time
        self.state = "IDLE"
        self.entry_price = 0.0
        self.last_tp_price = 0.0
        self._reset_knife_state()
        self._position_cache_valid = False

    def _cancel_all_orders(self, engine):
        for order_id in self.limit_order_ids:
            order = self._find_order_by_id(engine, order_id)
            if order and order.status == "new":
                engine.cancel_order(order_id)
        self.limit_order_ids = []

        if self.tp_order_id:
            order = self._find_order_by_id(engine, self.tp_order_id)
            if order and order.status == "new":
                engine.cancel_order(self.tp_order_id)
            self.tp_order_id = None

    def _reset_knife_state(self):
        self.knife_start_time = 0
        self.knife_low_price = 0.0
        if self.state != "IN_POSITION":
            self.state = "IDLE"

    def _get_open_position(self, engine):
        for p in engine.positions:
            if p.status == "open":
                return p
        return None

    def _find_order_by_id(self, engine, order_id: int):
        for order in engine.orders:
            if order.id == order_id:
                return order
        return None