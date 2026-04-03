from typing import Dict, Optional
from collections import deque

import numpy as np
from numba import njit

from visualization.visualization import PlotRecorder


@njit(cache=True, fastmath=True, nogil=True)
def calc_bollinger_bands(prices: np.ndarray, period: int, num_std: float):
    """
    Рассчитывает Bollinger Bands.
    Возвращает: (sma, upper_band, lower_band, std)
    """
    n = prices.shape[0]
    if n < period:
        return 0.0, 0.0, 0.0, 0.0
    
    # Берём последние period значений
    start_idx = n - period
    
    # SMA
    sma = 0.0
    for i in range(start_idx, n):
        sma += prices[i]
    sma /= period
    
    # Стандартное отклонение
    var = 0.0
    for i in range(start_idx, n):
        diff = prices[i] - sma
        var += diff * diff
    var /= period
    std = np.sqrt(var)
    
    upper = sma + num_std * std
    lower = sma - num_std * std
    
    return sma, upper, lower, std


@njit(cache=True, fastmath=True, nogil=True)
def calc_rsi(prices: np.ndarray, period: int) -> float:
    """Простой RSI для фильтрации сигналов."""
    n = prices.shape[0]
    if n < period + 1:
        return 50.0
    
    gains = 0.0
    losses = 0.0
    
    for i in range(n - period, n):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains += change
        else:
            losses -= change
    
    if losses == 0:
        return 100.0
    
    rs = gains / losses
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


class BollingerBandsStrategy:
    def __init__(
        self,
        initial_balance: float = 1000.0,
        
        # ══════════════════════════════════════════════════════
        # Параметры Bollinger Bands
        # ══════════════════════════════════════════════════════
        bb_period: int = 50000,
        bb_std_multiplier: float = 2.0,
        
        # ══════════════════════════════════════════════════════
        # Параметры торговли
        # ══════════════════════════════════════════════════════
        order_size: float = 100.0,
        max_position: float = 500.0,
        
        # Режим: True = mean reversion, False = breakout
        mean_reversion: bool = True,
        
        # Фильтры
        use_rsi_filter: bool = True,
        rsi_period: int = 40,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        
        # ══════════════════════════════════════════════════════
        # Настройки движка
        # ══════════════════════════════════════════════════════
        price_window_sec: float = 120.0,
        recalc_interval_ms: int = 100,
        min_data_points: int = 500,
    ):
        self.initial_balance = initial_balance
        
        # BB параметры
        self.bb_period = bb_period
        self.bb_std = bb_std_multiplier
        
        # Торговля
        self.order_size = order_size
        self.max_position = max_position
        self.mean_reversion = mean_reversion
        
        # RSI фильтр
        self.use_rsi_filter = use_rsi_filter
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        # Настройки
        self.price_window_ms = int(price_window_sec * 1000)
        self.recalc_interval_ms = recalc_interval_ms
        self.min_data_points = min_data_points
        
        # Окно цен
        self.price_times: deque = deque()
        self.price_values: deque = deque()
        
        # Состояние индикаторов
        self.last_recalc_time = 0
        self.mid_price = 0.0
        self.sma = 0.0
        self.upper_band = 0.0
        self.lower_band = 0.0
        self.bb_std_value = 0.0
        self.rsi = 50.0
        
        # Положение цены относительно полос
        # -1 = ниже lower, 0 = между, 1 = выше upper
        self.prev_price_zone = 0
        
        # Активные ордера
        self.active_orders: Dict[str, int] = {}
        
        # ══════════════════════════════════════════════════════
        # PlotRecorder для визуализации
        # ══════════════════════════════════════════════════════
        self.plot = PlotRecorder()
        
        # Прогрев Numba
        _warmup = np.zeros(50, dtype=np.float64)
        _ = calc_bollinger_bands(_warmup, 20, 2.0)
        _ = calc_rsi(_warmup, 14)

    # ==================================================================
    #  Основной хук из движка
    # ==================================================================
    def on_tick(self, event, engine):
        current_time = event["event_time"]
        
        # 1) Обновляем mid price и окно цен
        if event.get("event_type") == "bookticker":
            bid = event["bid_price"]
            ask = event["ask_price"]
            self.mid_price = 0.5 * (bid + ask)
            
            self.price_times.append(current_time)
            self.price_values.append(self.mid_price)
            
            # Очищаем старые данные
            cutoff = current_time - self.price_window_ms
            while self.price_times and self.price_times[0] < cutoff:
                self.price_times.popleft()
                self.price_values.popleft()
        
        # Ждём достаточного количества точек
        if len(self.price_values) < self.min_data_points:
            return
        
        # 2) Пересчёт по интервалу
        if current_time - self.last_recalc_time >= self.recalc_interval_ms:
            self._calculate_indicators_and_trade(engine, current_time)
            self.last_recalc_time = current_time

    # ==================================================================
    #  Расчёт индикаторов и торговая логика
    # ==================================================================
    def _calculate_indicators_and_trade(self, engine, current_time: int):
        prices_np = np.array(self.price_values, dtype=np.float64)
        
        # ══════════════════════════════════════════════════════
        # Расчёт Bollinger Bands
        # ══════════════════════════════════════════════════════
        self.sma, self.upper_band, self.lower_band, self.bb_std_value = \
            calc_bollinger_bands(prices_np, self.bb_period, self.bb_std)
        
        if self.sma == 0.0:
            return
        
        # RSI
        if self.use_rsi_filter:
            self.rsi = calc_rsi(prices_np, self.rsi_period)
        
        # ══════════════════════════════════════════════════════
        # ВИЗУАЛИЗАЦИЯ
        # ══════════════════════════════════════════════════════
        self._record_plots(current_time)
        
        # ══════════════════════════════════════════════════════
        # ТОРГОВАЯ ЛОГИКА
        # ══════════════════════════════════════════════════════
        self._check_signals_and_trade(engine, current_time)

    # ==================================================================
    #  Запись данных для визуализации
    # ==================================================================
    def _record_plots(self, current_time: int):
        # ──────────────────────────────────────────────────────
        # SMA (средняя линия Bollinger)
        # ──────────────────────────────────────────────────────
        self.plot.line(
            "BB SMA",
            self.sma,
            current_time,
            color="#FF9800",      # оранжевый
            linewidth=1.5,
            alpha=0.9
        )
        
        # ──────────────────────────────────────────────────────
        # Bollinger Bands как заполненная область
        # ──────────────────────────────────────────────────────
        self.plot.band(
            "Bollinger Bands",
            upper=self.upper_band,
            lower=self.lower_band,
            time=current_time,
            color="#2196F3",      # синий
            alpha=0.15
        )
        
        # ──────────────────────────────────────────────────────
        # Верхняя граница (пунктир)
        # ──────────────────────────────────────────────────────
        self.plot.line(
            "BB Upper",
            self.upper_band,
            current_time,
            color="#F44336",      # красный
            linewidth=1,
            linestyle="dashed",
            alpha=0.7
        )
        
        # ──────────────────────────────────────────────────────
        # Нижняя граница (пунктир)
        # ──────────────────────────────────────────────────────
        self.plot.line(
            "BB Lower",
            self.lower_band,
            current_time,
            color="#4CAF50",      # зелёный
            linewidth=1,
            linestyle="dashed",
            alpha=0.7
        )

    # ==================================================================
    #  Проверка сигналов и торговля
    # ==================================================================
    def _check_signals_and_trade(self, engine, current_time: int):
        # Определяем текущую зону цены
        if self.mid_price <= self.lower_band:
            current_zone = -1
        elif self.mid_price >= self.upper_band:
            current_zone = 1
        else:
            current_zone = 0
        
        # Текущая позиция
        pos = self._get_position(engine)
        inventory = pos.size if pos else 0.0
        
        signal: Optional[str] = None
        
        # ══════════════════════════════════════════════════════
        # Mean Reversion режим
        # ══════════════════════════════════════════════════════
        if self.mean_reversion:
            # Цена вернулась из-под нижней полосы -> покупаем
            if self.prev_price_zone == -1 and current_zone >= 0:
                rsi_ok = (not self.use_rsi_filter) or (self.rsi < self.rsi_oversold + 10)
                if inventory < self.max_position and rsi_ok:
                    signal = "buy"
            
            # Цена вернулась из-над верхней полосы -> продаём
            elif self.prev_price_zone == 1 and current_zone <= 0:
                rsi_ok = (not self.use_rsi_filter) or (self.rsi > self.rsi_overbought - 10)
                if inventory > -self.max_position and rsi_ok:
                    signal = "sell"
            
            # Цена вернулась к SMA -> закрываем позицию
            sma_zone = (self.upper_band - self.sma) * 0.3
            if abs(self.mid_price - self.sma) < sma_zone:
                if inventory > 0:
                    signal = "close_long"
                elif inventory < 0:
                    signal = "close_short"
        
        # ══════════════════════════════════════════════════════
        # Breakout режим
        # ══════════════════════════════════════════════════════
        else:
            # Пробой верхней полосы -> покупаем (тренд вверх)
            if self.prev_price_zone != 1 and current_zone == 1:
                rsi_ok = (not self.use_rsi_filter) or (self.rsi > 50)
                if inventory < self.max_position and rsi_ok:
                    signal = "buy"
            
            # Пробой нижней полосы -> продаём (тренд вниз)
            elif self.prev_price_zone != -1 and current_zone == -1:
                rsi_ok = (not self.use_rsi_filter) or (self.rsi < 50)
                if inventory > -self.max_position and rsi_ok:
                    signal = "sell"
            
            # Возврат в канал -> закрываем
            if current_zone == 0 and self.prev_price_zone != 0:
                if inventory > 0 and self.prev_price_zone == 1:
                    signal = "close_long"
                elif inventory < 0 and self.prev_price_zone == -1:
                    signal = "close_short"
        
        # Обновляем зону
        self.prev_price_zone = current_zone
        
        # Исполняем сигнал
        if signal:
            self._execute_signal(engine, signal, current_time)

    # ==================================================================
    #  Исполнение сигналов
    # ==================================================================
    def _execute_signal(self, engine, signal: str, current_time: int):
        
        if signal == "buy":
            oid = engine.place_order("market", size=self.order_size, price=100000.0)
            if oid:
                self.active_orders["last_buy"] = oid
                
                # Маркер покупки
                self.plot.marker(
                    "Buy Signal",
                    self.mid_price,
                    current_time,
                    marker="triangle",
                    color="#00E676",  # ярко-зелёный
                    size=12
                )
        
        elif signal == "sell":
            oid = engine.place_order("market", size=-self.order_size, price=0.0)
            if oid:
                self.active_orders["last_sell"] = oid
                
                # Маркер продажи
                self.plot.marker(
                    "Sell Signal",
                    self.mid_price,
                    current_time,
                    marker="inverted_triangle",
                    color="#FF5252",  # ярко-красный
                    size=12
                )
        
        elif signal == "close_long":
            pos = self._get_position(engine)
            if pos and pos.size > 0:
                oid = engine.place_order("market", size=-pos.size, price=0.0)
                if oid:
                    # Маркер закрытия лонга
                    self.plot.marker(
                        "Close Long",
                        self.mid_price,
                        current_time,
                        marker="x",
                        color="#FFC107",  # жёлтый
                        size=10
                    )
        
        elif signal == "close_short":
            pos = self._get_position(engine)
            if pos and pos.size < 0:
                oid = engine.place_order("market", size=-pos.size, price=100000.0)
                if oid:
                    # Маркер закрытия шорта
                    self.plot.marker(
                        "Close Short",
                        self.mid_price,
                        current_time,
                        marker="x",
                        color="#FFC107",
                        size=10
                    )

    # ==================================================================
    #  Вспомогательные методы
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