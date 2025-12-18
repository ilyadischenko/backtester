from typing import Optional, List
import numpy as np
from numba import njit

# ═══════════════════════════════════════════════════════════════════
# 1. NUMBA ЯДРО (ВЫСОКОПРОИЗВОДИТЕЛЬНЫЕ ВЫЧИСЛЕНИЯ)
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True, nogil=True)
def calc_volatility_circular(
    prices: np.ndarray, 
    times: np.ndarray,
    head: int,
    size: int,
    buffer_size: int,
    current_time: int,
    window_ms: int
) -> float:
    """Считает волатильность (Coeff of Variation) на кольцевом буфере."""
    if size < 2:
        return 0.0
    
    cutoff = current_time - window_ms
    mean = 0.0
    count = 0
    
    # 1. Считаем среднее
    for i in range(size):
        idx = (head - size + i) % buffer_size
        if times[idx] >= cutoff:
            mean += prices[idx]
            count += 1
    
    if count < 2 or mean <= 1e-8:
        return 0.0
    
    mean /= count
    
    # 2. Считаем дисперсию
    variance = 0.0
    for i in range(size):
        idx = (head - size + i) % buffer_size
        if times[idx] >= cutoff:
            diff = prices[idx] - mean
            variance += diff * diff
    
    # Используем поправку Бесселя (count - 1)
    variance /= (count - 1)
    std = np.sqrt(variance)
    
    # Возвращаем в базисных пунктах (или просто масштабированное значение)
    return (std / mean) * 10000.0


@njit(cache=True, fastmath=True, nogil=True)
def calc_price_change_circular(
    prices: np.ndarray,
    times: np.ndarray,
    head: int,
    size: int,
    buffer_size: int,
    current_time: int,
    window_ms: int
) -> tuple:
    """Считает изменение цены за окно и определяет направление."""
    if size < 2:
        return 0.0, False
    
    cutoff = current_time - window_ms
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
    
    if not found_first or first_price <= 1e-8:
        return 0.0, False
    
    change_bps = (last_price - first_price) / first_price * 10000.0
    return change_bps, change_bps < 0


@njit(cache=True, fastmath=True, nogil=True)
def compute_all_indicators(
    p_times, p_vals, p_head, p_size, p_buf_size,
    v_vals, v_size,
    cur_time, vol_win, calm_win
):
    """
    Агрегирующая функция. Вызывается из Python 1 раз,
    внутри Numba делает все расчеты, избегая оверхеда вызовов.
    """
    # 1. Текущая волатильность
    vol = calc_volatility_circular(p_vals, p_times, p_head, p_size, p_buf_size, cur_time, vol_win)
    
    # 2. Средняя волатильность (из буфера истории волатильности)
    avg_vol = 0.0
    v_count = 0
    if v_size > 0:
        for i in range(v_size):
            if v_vals[i] > 0:
                avg_vol += v_vals[i]
                v_count += 1
        if v_count > 0:
            avg_vol /= v_count

    # 3. Изменение цены
    pch, is_fall = calc_price_change_circular(p_vals, p_times, p_head, p_size, p_buf_size, cur_time, calm_win)
    
    return vol, pch, is_fall, avg_vol


# ═══════════════════════════════════════════════════════════════════
# 2. СТРАТЕГИЯ
# ═══════════════════════════════════════════════════════════════════

class KnifeCatcherVolatilityOnly:
    def __init__(
        self,
        initial_balance: float = 1000.0,
        # === Ордера ===
        market_order_size: float = 100.0,
        limit_order_1_size: float = 100.0,
        limit_order_2_size: float = 100.0,
        limit_order_1_offset_bps: float = 20.0,
        limit_order_2_offset_bps: float = 40.0,
        stop_loss_bps: float = 30.0,
        take_profit_bps: float = 70.0,
        cooldown_after_close_sec: float = 10.0,
        
        # === Логика Волатильности ===
        volatility_window_sec: float = 30.0,
        volatility_spike_pct: float = 150.0,      
        volatility_lookback_sec: float = 100.0,
        
        # === Логика Ножа ===
        min_knife_duration_sec: float = 0.5,
        max_knife_duration_sec: float = 10.0,
        min_price_drop_bps: float = 5.0,
        
        # === Логика Спокойствия ===
        calm_window_sec: float = 2.0,
        calm_price_change_bps: float = 2.0,
        
        # === Оптимизация ===
        price_buffer_size: int = 100_000,
        recalc_interval_ms: int = 10,  # Троттлинг расчетов (10мс) в спокойном режиме
        min_data_points: int = 100
    ):
        
        self.initial_balance = initial_balance
        # Конвертация в миллисекунды и integers
        self.market_order_size = market_order_size
        self.limit_order_1_size = limit_order_1_size
        self.limit_order_2_size = limit_order_2_size
        self.limit_order_1_offset_bps = limit_order_1_offset_bps
        self.limit_order_2_offset_bps = limit_order_2_offset_bps
        self.stop_loss_bps = stop_loss_bps
        self.take_profit_bps = take_profit_bps
        
        self.volatility_window_ms = int(volatility_window_sec * 1000)
        self.volatility_spike_pct = volatility_spike_pct
        self.volatility_lookback_ms = int(volatility_lookback_sec * 1000)
        
        self.calm_window_ms = int(calm_window_sec * 1000)
        self.calm_price_change_bps = calm_price_change_bps
        
        self.min_knife_duration_ms = int(min_knife_duration_sec * 1000)
        self.max_knife_duration_ms = int(max_knife_duration_sec * 1000)
        self.min_price_drop_bps = min_price_drop_bps
        
        self.cooldown_ms = int(cooldown_after_close_sec * 1000)
        self.recalc_interval_ms = recalc_interval_ms
        self.min_data_points = min_data_points
        
        # --- Буферы данных (Pre-allocated) ---
        self.price_buffer_size = price_buffer_size
        self.price_times = np.zeros(price_buffer_size, dtype=np.int64)
        self.price_values = np.zeros(price_buffer_size, dtype=np.float64)
        self.price_head = 0
        self.price_size = 0
        
        # Буфер для истории волатильности (для расчета среднего)
        self.vol_buffer_size = 2000 
        self.vol_values = np.zeros(self.vol_buffer_size, dtype=np.float64)
        self.vol_head = 0
        self.vol_size = 0
        
        # --- Состояние стратегии ---
        self.state: str = "IDLE"
        self.knife_start_time: int = 0
        self.knife_low_price: float = 0.0
        self.last_recalc_time: int = 0
        self.last_close_time: int = 0
        
        # --- Кэш индикаторов ---
        self._volatility: float = 0.0
        self._avg_volatility: float = 0.0
        self._price_change: float = 0.0
        self._is_falling: bool = False
        
        # --- Управление ордерами ---
        self.limit_order_ids: List[int] = []
        self.tp_order_id: Optional[int] = None
        self.last_tp_price: float = 0.0
        self.is_closing_position: bool = False
        
        self.current_bid: float = 0.0
        self.current_ask: float = 0.0
        self.current_mid: float = 0.0
        
        # --- Warm-up Numba (Компиляция при старте) ---
        # Вызываем функцию на пустышках, чтобы скомпилировать её сейчас, 
        # а не на первом тике данных.
        compute_all_indicators(
            self.price_times, self.price_values, 0, 0, 100, 
            self.vol_values, 0, 0, 100, 100
        )

    # ═══════════════════════════════════════════════════════════════
    # ГЛАВНЫЙ ЦИКЛ (ON TICK) - OPTIMIZED PATH
    # ═══════════════════════════════════════════════════════════════
    
    def on_tick(self, event, engine):
        current_time = event["event_time"]
        
        # 1. МАКСИМАЛЬНО БЫСТРАЯ ЗАПИСЬ ЦЕНЫ (O(1))
        # Это делается на каждом тике, чтобы не терять high/low внутри свечей
        if event.get("event_type") == "bookticker":
            self.current_bid = event["bid_price"]
            self.current_ask = event["ask_price"]
            self.current_mid = (self.current_bid + self.current_ask) * 0.5
            
            idx = self.price_head % self.price_buffer_size
            self.price_times[idx] = current_time
            self.price_values[idx] = self.current_mid
            self.price_head += 1
            if self.price_size < self.price_buffer_size:
                self.price_size += 1
        
        # Пропускаем первые тики, пока буфер не наполнится
        if self.price_size < self.min_data_points:
            return

        # 2. ПРОВЕРКА ПОЗИЦИИ
        # Если мы отправили рыночный ордер на закрытие, ждем пока он исполнится
        if self.is_closing_position:
            # Тут можно проверить engine.positions, если движок обновляет их синхронно
            pos = self._get_position(engine)
            if not pos or pos.size == 0:
                 self._on_position_closed(engine, current_time)
            return

        pos = self._get_position(engine)
        if pos and pos.size > 0:
            self.state = "IN_POSITION"
            self._manage_position(engine, pos, current_time)
            return
        elif self.state == "IN_POSITION":
            # Позиция исчезла (закрылась по TP или стоп-аут)
            self._on_position_closed(engine, current_time)
            return
            
        # Кулдаун после сделки
        if current_time - self.last_close_time < self.cooldown_ms:
            return

        # 3. ОПТИМИЗАЦИЯ РАСЧЕТОВ (Lazy Calculation)
        # Если мы IDLE, считаем раз в X мс. Если ищем нож - считаем всегда.
        time_diff = current_time - self.last_recalc_time
        should_recalc = (time_diff >= self.recalc_interval_ms) or (self.state != "IDLE")
        
        if should_recalc:
            self._update_indicators(current_time)
            self.last_recalc_time = current_time
            
            # Запускаем логику поиска входа
            self._run_state_machine(engine, current_time)

    # ═══════════════════════════════════════════════════════════════
    # ЛОГИКА
    # ═══════════════════════════════════════════════════════════════

    def _update_indicators(self, current_time):
        # Если есть рассчитанная волатильность, сохраняем её в историю
        # для расчета среднего (moving average volatility)
        if self._volatility > 0:
            v_idx = self.vol_head % self.vol_buffer_size
            self.vol_values[v_idx] = self._volatility
            self.vol_head += 1
            if self.vol_size < self.vol_buffer_size:
                self.vol_size += 1
        
        # Numba считает всё за один проход
        (self._volatility, 
         self._price_change, 
         self._is_falling, 
         self._avg_volatility) = compute_all_indicators(
            self.price_times, self.price_values, self.price_head, self.price_size, self.price_buffer_size,
            self.vol_values, self.vol_size,
            current_time, self.volatility_window_ms, self.calm_window_ms
        )

    def _run_state_machine(self, engine, current_time: int):
        # Нужна база для сравнения
        if self.vol_size < 10: return

        if self.state == "IDLE":
            if self._detect_knife():
                self.state = "KNIFE_DETECTED"
                self.knife_start_time = current_time
                self.knife_low_price = self.current_mid
        
        elif self.state == "KNIFE_DETECTED":
            # Обновляем лой
            if self.current_mid < self.knife_low_price:
                self.knife_low_price = self.current_mid
            
            knife_duration = current_time - self.knife_start_time
            
            # 1. Таймаут: нож идет слишком долго, это уже тренд вниз, а не сквиз
            if knife_duration > self.max_knife_duration_ms:
                self._reset_knife_state()
                return
            
            # 2. Ждем мин время и ищем успокоение
            if knife_duration >= self.min_knife_duration_ms:
                if self._detect_calm():
                    self.state = "WAITING_CALM"
        
        elif self.state == "WAITING_CALM":
            # 1. Таймаут ожидания (FIX: Защита от зависания)
            if current_time - self.knife_start_time > self.max_knife_duration_ms:
                self._reset_knife_state()
                return

            # 2. Если всё спокойно -> ВХОД
            if self._confirm_calm():
                self._enter_position(engine, current_time)
            
            # 3. Если снова полетели вниз -> Обратно в KNIFE_DETECTED
            elif self._detect_knife():
                self.state = "KNIFE_DETECTED"
                if self.current_mid < self.knife_low_price:
                    self.knife_low_price = self.current_mid

    # --- Предикаты (Условия) ---
    
    def _detect_knife(self) -> bool:
        """Всплеск волатильности + падение цены."""
        if self._avg_volatility <= 1e-8: return False
        
        vol_spike = (self._volatility / self._avg_volatility - 1) * 100
        if vol_spike < self.volatility_spike_pct: return False
        
        if not self._is_falling: return False
        if abs(self._price_change) < self.min_price_drop_bps: return False
        
        return True
    
    def _detect_calm(self) -> bool:
        """Предварительное успокоение: цена перестала обновлять лои."""
        return not self._is_falling
    
    def _confirm_calm(self) -> bool:
        """Подтверждение: изменение цены в узком диапазоне."""
        if abs(self._price_change) > self.calm_price_change_bps:
            return False
        return True

    # --- Исполнение ---

    def _enter_position(self, engine, current_time: int):
        if self.current_ask <= 0: return

        # 1. Market Buy
        market_size = self.market_order_size
        engine.place_order("market", price=0, size=market_size)
        
        # 2. Limit DCA (Усреднение)
        # Округляем цены согласно тику инструмента (в реальном боте), тут упрощенно
        limit_price_1 = self.current_ask * (1 - self.limit_order_1_offset_bps / 10000)
        limit_size_1 = self.limit_order_1_size
        o1 = engine.place_order("limit", price=limit_price_1, size=limit_size_1)
        if o1: self.limit_order_ids.append(o1)
        
        limit_price_2 = self.current_ask * (1 - self.limit_order_2_offset_bps / 10000)
        limit_size_2 = self.limit_order_2_size
        o2 = engine.place_order("limit", price=limit_price_2, size=limit_size_2)
        if o2: self.limit_order_ids.append(o2)
        
        self.state = "IN_POSITION"
        self._reset_knife_state()

    def _manage_position(self, engine, pos, current_time: int):
        avg_price = pos.price
        pos_size = pos.size
        
        # STOP LOSS
        sl_price = avg_price * (1 - self.stop_loss_bps / 10000)
        if self.current_bid <= sl_price:
            self._execute_stop_loss(engine, pos_size)
            return

        # TAKE PROFIT
        tp_price = avg_price * (1 + self.take_profit_bps / 10000)
        
        if self.tp_order_id is None:
            self._place_take_profit(engine, tp_price, pos_size)
        else:
            self._update_take_profit(engine, tp_price, pos_size)
            
            # Проверка исполнения TP (если движок не шлет trade event)
            tp_order = self._find_order_by_id(engine, self.tp_order_id)
            if tp_order and tp_order.status == "filled":
                 # TP исполнился, позиция должна стать 0 на след тике
                 pass

    def _place_take_profit(self, engine, tp_price, pos_size):
        self.tp_order_id = engine.place_order("limit", price=tp_price, size=-pos_size)
        if self.tp_order_id:
            self.last_tp_price = tp_price

    def _update_take_profit(self, engine, new_tp_price, pos_size):
        # Обновляем TP только если цена сильно ушла, чтобы не спамить API
        if self.last_tp_price > 0:
            diff_bps = abs(new_tp_price - self.last_tp_price) / self.last_tp_price * 10000
            if diff_bps < 2.0: return # < 2 bps разницы - игнор
            
        tp_order = self._find_order_by_id(engine, self.tp_order_id)
        if not tp_order or tp_order.status == "filled":
            self.tp_order_id = None
            return

        if tp_order.status in ["new", "partially_filled"]:
            engine.cancel_order(self.tp_order_id)
            self.tp_order_id = None 
            # Новый ордер выставится на следующем тике, когда tp_order_id будет None

    def _execute_stop_loss(self, engine, pos_size):
        engine.place_order("market", price=0, size=-pos_size)
        self.is_closing_position = True # Блокируем логику до закрытия

    def _on_position_closed(self, engine, current_time):
        self._cancel_all_orders(engine)
        self.state = "IDLE"
        self.is_closing_position = False
        self.last_close_time = current_time
        self.tp_order_id = None
        self.last_tp_price = 0.0
        self._reset_knife_state()

    def _cancel_all_orders(self, engine):
        for oid in self.limit_order_ids:
            o = self._find_order_by_id(engine, oid)
            if o and o.status in ["new", "partially_filled"]:
                engine.cancel_order(oid)
        self.limit_order_ids.clear()
        
        if self.tp_order_id:
            o = self._find_order_by_id(engine, self.tp_order_id)
            if o and o.status in ["new", "partially_filled"]:
                engine.cancel_order(self.tp_order_id)
            self.tp_order_id = None

    def _reset_knife_state(self):
        self.knife_start_time = 0
        self.knife_low_price = 0.0
        if self.state != "IN_POSITION":
            self.state = "IDLE"

    # --- Утилиты ---

    def _get_position(self, engine):
        for p in engine.positions:
            if p.size != 0: return p
        return None
    
    def _find_order_by_id(self, engine, order_id):
        # В реальном HFT тут нужен dict {id: order}, поиск по списку O(N) медленный.
        # Но для стандартного бэктестера норм.
        for order in engine.orders:
            if order.id == order_id:
                return order
        return None