from typing import Dict, Optional
from collections import deque
import numpy as np
from numba import njit

from visualization.visualization import PlotRecorder


@njit(cache=True, fastmath=True, nogil=True)
def calc_sma(prices: np.ndarray) -> float:
    """Простая скользящая средняя."""
    n = prices.shape[0]
    if n == 0:
        return 0.0
    
    total = 0.0
    for i in range(n):
        total += prices[i]
    
    return total / n


@njit(cache=True, fastmath=True, nogil=True)
def calc_std(prices: np.ndarray, mean: float) -> float:
    """Стандартное отклонение."""
    n = prices.shape[0]
    if n < 2:
        return 0.0
    
    var = 0.0
    for i in range(n):
        diff = prices[i] - mean
        var += diff * diff
    
    return np.sqrt(var / (n - 1))


class ChannelStrategy:
    """
    Канальная стратегия с защитой позиции и стоп-лоссом.
    
    Логика:
    - Покупаем на нижней границе канала
    - Продаём на верхней границе канала
    
    ЗАЩИТА:
    - Лонг закрываем ТОЛЬКО если цена >= цены входа (в плюс)
    - Шорт закрываем ТОЛЬКО если цена <= цены входа (в плюс)
    
    СТОП-ЛОСС:
    - При максимальном инвентаре, если убыток >= stop_loss_pct
    - Закрываем маркет-ордером
    """
    
    def __init__(
        self,
        # Параметры канала
        sma_window_sec: float = 180.0,
        channel_multiplier: float = 1.0,
        
        # Параметры торговли
        order_size: float = 100.0,
        max_position: float = 500.0,
        
        # Параметры защиты
        min_profit_bps: float = 0.0,
        
        # СТОП-ЛОСС
        stop_loss_pct: float = 1.0,  # Стоп при -1% от позиции
        
        # Технические параметры
        min_data_points: int = 50,
        recalc_interval_ms: int = 500,
    ):
        
        self.initial_balance = 1000.0
        self.sma_window_ms = int(sma_window_sec * 1000)
        self.channel_mult = channel_multiplier
        self.order_size = order_size
        self.max_position = max_position
        self.min_profit_bps = min_profit_bps
        self.stop_loss_pct = stop_loss_pct
        self.min_data_points = min_data_points
        self.recalc_interval_ms = recalc_interval_ms
        
        # Буферы данных
        self.price_times = deque()
        self.price_values = deque()
        
        # Состояние
        self.mid_price = 0.0
        self.sma = 0.0
        self.upper_channel = 0.0
        self.lower_channel = 0.0
        self.last_recalc_time = 0
        
        # Трекинг позиции
        self.entry_price = 0.0
        self.last_inventory = 0.0
        
        # Стоп-лосс состояние
        self.stop_triggered = False  # Флаг что стоп уже сработал
        self.stop_order_id: Optional[int] = None  # ID маркет-ордера стопа
        
        # Активные ордера
        self.active_orders: Dict[str, int] = {}
        
        # Визуализация
        self.plot = PlotRecorder()
        
        # Прогрев Numba
        _dummy = np.zeros(2, dtype=np.float64)
        _ = calc_sma(_dummy)
        _ = calc_std(_dummy, 0.0)
    
    def on_tick(self, event, engine):
        """Главный обработчик."""
        current_time = event["event_time"]
        
        # ═══════════════════════════════════════════
        # ШАГ 1: СОБИРАЕМ ДАННЫЕ
        # ═══════════════════════════════════════════
        if event.get("event_type") == "bookticker":
            bid = event["bid_price"]
            ask = event["ask_price"]
            self.mid_price = (bid + ask) / 2
            
            self.price_times.append(current_time)
            self.price_values.append(self.mid_price)
            
            cutoff = current_time - self.sma_window_ms
            while self.price_times and self.price_times[0] < cutoff:
                self.price_times.popleft()
                self.price_values.popleft()
        
        # ═══════════════════════════════════════════
        # ШАГ 2: ПРОВЕРЯЕМ ГОТОВНОСТЬ
        # ═══════════════════════════════════════════
        if len(self.price_values) < self.min_data_points:
            return
        
        if current_time - self.last_recalc_time < self.recalc_interval_ms:
            return
        
        # ═══════════════════════════════════════════
        # ШАГ 3: РАСЧЁТ КАНАЛА
        # ═══════════════════════════════════════════
        prices_np = np.array(self.price_values, dtype=np.float64)
        
        self.sma = calc_sma(prices_np)
        std = calc_std(prices_np, self.sma)
        
        self.upper_channel = self.sma + std * self.channel_mult
        self.lower_channel = self.sma - std * self.channel_mult
        
        # ═══════════════════════════════════════════
        # ШАГ 4: ПОЛУЧАЕМ ПОЗИЦИЮ И ЦЕНУ ВХОДА
        # ═══════════════════════════════════════════
        position = self._get_position(engine)
        inventory = position.size if position else 0.0
        
        # Обновляем цену входа
        entry_price = self._get_entry_price(position, inventory)
        
        # Сбрасываем флаг стопа если позиция закрылась
        if inventory == 0.0:
            self.stop_triggered = False
            self.stop_order_id = None
        
        # ═══════════════════════════════════════════
        # ШАГ 5: ПРОВЕРКА СТОП-ЛОССА (ПРИОРИТЕТ!)
        # ═══════════════════════════════════════════
        stop_hit = self._check_stop_loss(engine, inventory, entry_price, current_time)
        
        if stop_hit:
            # Стоп сработал - не торгуем дальше
            self._draw_chart(current_time, inventory, entry_price)
            self.last_recalc_time = current_time
            return
        
        # ═══════════════════════════════════════════
        # ШАГ 6: ТОРГОВАЯ ЛОГИКА С ЗАЩИТОЙ
        # ═══════════════════════════════════════════
        self._execute_strategy(engine, inventory, entry_price, current_time)
        
        # ═══════════════════════════════════════════
        # ШАГ 7: ВИЗУАЛИЗАЦИЯ
        # ═══════════════════════════════════════════
        self._draw_chart(current_time, inventory, entry_price)
        
        self.last_recalc_time = current_time
    
    def _check_stop_loss(self, engine, inventory: float, entry_price: float, current_time: int) -> bool:
        """
        Проверка и исполнение стоп-лосса.
        
        Условие: при максимальном инвентаре убыток >= stop_loss_pct
        
        Returns:
            True если стоп сработал (нужно остановить торговлю)
        """
        # Если уже сработал стоп - ждём закрытия
        if self.stop_triggered:
            return True
        
        # Нет позиции - нечего стопать
        if inventory == 0.0 or entry_price <= 0:
            return False
        
        # Проверяем только при максимальном инвентаре
        if abs(inventory) < self.max_position * 0.99:  # 99% от максимума
            return False
        
        # ─────────────────────────────────────────
        # Расчёт unrealized PnL в процентах
        # ─────────────────────────────────────────
        if inventory > 0:  # ЛОНГ
            # PnL% = (текущая - вход) / вход * 100
            pnl_pct = (self.mid_price - entry_price) / entry_price * 100
        else:  # ШОРТ
            # PnL% = (вход - текущая) / вход * 100
            pnl_pct = (entry_price - self.mid_price) / entry_price * 100
        
        # ─────────────────────────────────────────
        # Проверка условия стопа
        # ─────────────────────────────────────────
        if pnl_pct <= -self.stop_loss_pct:
            # СТОП СРАБОТАЛ!
            self._execute_stop_loss(engine, inventory, entry_price, pnl_pct, current_time)
            return True
        
        # ─────────────────────────────────────────
        # Визуализация уровня стопа
        # ─────────────────────────────────────────
        if abs(inventory) >= self.max_position * 0.99:
            if inventory > 0:  # Лонг
                stop_price = entry_price * (1 - self.stop_loss_pct / 100)
            else:  # Шорт
                stop_price = entry_price * (1 + self.stop_loss_pct / 100)
            
            self.plot.line(
                "Stop Loss Level",
                stop_price,
                current_time,
                color="#FF0000",
                linewidth=2,
                linestyle="dotted",
                alpha=0.8
            )
        
        return False
    
    def _execute_stop_loss(self, engine, inventory: float, entry_price: float, 
                           pnl_pct: float, current_time: int):
        """
        Исполнение стоп-лосса через маркет-ордер.
        """
        self.stop_triggered = True
        
        # ─────────────────────────────────────────
        # 1. Отменяем все активные лимитные ордера
        # ─────────────────────────────────────────
        for side in list(self.active_orders.keys()):
            self._cancel_order(engine, side)
        
        # ─────────────────────────────────────────
        # 2. Закрываем позицию маркет-ордером
        # ─────────────────────────────────────────
        # Для закрытия лонга - продаём (size < 0)
        # Для закрытия шорта - покупаем (size > 0)
        close_size = -inventory  # Противоположный знак
        
        self.stop_order_id = engine.place_order(
            "market",
            price=self.mid_price,  # Для маркета цена индикативная
            size=close_size
        )
        
        # ─────────────────────────────────────────
        # 3. Визуализация
        # ─────────────────────────────────────────
        # Расчёт убытка в деньгах
        notional = abs(inventory) * entry_price
        loss_amount = notional * abs(pnl_pct) / 100
        
        self.plot.marker(
            f"STOP LOSS ({pnl_pct:.2f}%)",
            self.mid_price,
            current_time,
            marker="x",
            color="#FF0000",
            size=15
        )
        
        # Логируем для отладки
        print(f"🛑 STOP LOSS TRIGGERED!")
        print(f"   Inventory: {inventory:.2f}")
        print(f"   Entry: {entry_price:.6f}")
        print(f"   Current: {self.mid_price:.6f}")
        print(f"   PnL: {pnl_pct:.2f}%")
        print(f"   Loss: ${loss_amount:.4f}")
    
    def _get_entry_price(self, position, inventory: float) -> float:
        """Получаем цену входа в позицию."""
        if position is not None:
            if hasattr(position, 'price') and position.price:
                return position.price
            if hasattr(position, 'entry_price') and position.entry_price:
                return position.entry_price
            if hasattr(position, 'average_price') and position.average_price:
                return position.average_price
        
        if inventory == 0.0:
            self.entry_price = 0.0
            self.last_inventory = 0.0
            return 0.0
        
        if self.last_inventory == 0.0 and inventory != 0.0:
            self.entry_price = self.mid_price
        
        elif abs(inventory) > abs(self.last_inventory) and self.entry_price > 0:
            added = abs(inventory) - abs(self.last_inventory)
            total = abs(inventory)
            self.entry_price = (
                self.entry_price * abs(self.last_inventory) + 
                self.mid_price * added
            ) / total
        
        self.last_inventory = inventory
        return self.entry_price
    
    def _execute_strategy(self, engine, inventory: float, entry_price: float, current_time: int):
        """Торговая логика с защитой позиции."""
        
        # Сигналы канала
        buy_signal = self.mid_price <= self.lower_channel
        sell_signal = self.mid_price >= self.upper_channel
        
        # Базовые лимиты
        can_buy = inventory < self.max_position
        can_sell = inventory > -self.max_position
        
        # ═══════════════════════════════════════════
        # ЗАЩИТА ПОЗИЦИИ
        # ═══════════════════════════════════════════
        min_profit_mult = 1 + (self.min_profit_bps / 10000)
        
        # ЛОНГ позиция
        if inventory > 0 and entry_price > 0:
            min_sell_price = entry_price * min_profit_mult
            
            if self.mid_price < min_sell_price:
                can_sell = False
                self.plot.marker(
                    "Holding Long",
                    self.mid_price,
                    current_time,
                    marker="diamond",
                    color="#FFC107",
                    size=5
                )
        
        # ШОРТ позиция
        if inventory < 0 and entry_price > 0:
            max_buy_price = entry_price / min_profit_mult
            
            if self.mid_price > max_buy_price:
                can_buy = False
                self.plot.marker(
                    "Holding Short",
                    self.mid_price,
                    current_time,
                    marker="diamond",
                    color="#FFC107",
                    size=5
                )
        
        # ─────────────────────────────────────────
        # ПОКУПКА
        # ─────────────────────────────────────────
        if buy_signal and can_buy:
            if "buy" not in self.active_orders:
                self._place_order(engine, "buy", self.lower_channel, current_time)
        else:
            if "buy" in self.active_orders:
                self._cancel_order(engine, "buy")
        
        # ─────────────────────────────────────────
        # ПРОДАЖА
        # ─────────────────────────────────────────
        if sell_signal and can_sell:
            if "sell" not in self.active_orders:
                self._place_order(engine, "sell", self.upper_channel, current_time)
        else:
            if "sell" in self.active_orders:
                self._cancel_order(engine, "sell")
    
    def _place_order(self, engine, side: str, price: float, current_time: int):
        """Размещение ордера."""
        size = self.order_size
        if side == "sell":
            size = -size
        
        order_id = engine.place_order("limit", price=price, size=size)
        
        if order_id:
            self.active_orders[side] = order_id
            
            if side == "buy":
                self.plot.marker(
                    "Buy Order",
                    price,
                    current_time,
                    marker="triangle",
                    color="#00E676",
                    size=8
                )
            else:
                self.plot.marker(
                    "Sell Order",
                    price,
                    current_time,
                    marker="inverted_triangle",
                    color="#FF5252",
                    size=8
                )
    
    def _cancel_order(self, engine, side: str):
        """Отмена ордера."""
        order_id = self.active_orders.get(side)
        if order_id is None:
            return
        
        order = self._find_order(engine, order_id)
        if order and order.status in ("new", "partially_filled"):
            engine.cancel_order(order_id)
        
        self.active_orders.pop(side, None)
    
    def _draw_chart(self, current_time: int, inventory: float, entry_price: float):
        """Визуализация."""
        # SMA
        self.plot.line(
            "SMA",
            self.sma,
            current_time,
            color="#2196F3",
            linewidth=2,
            alpha=0.9
        )
        
        # Каналы
        self.plot.line(
            "Upper Channel",
            self.upper_channel,
            current_time,
            color="#F44336",
            linewidth=1.5,
            linestyle="dashed",
            alpha=0.7
        )
        
        self.plot.line(
            "Lower Channel",
            self.lower_channel,
            current_time,
            color="#4CAF50",
            linewidth=1.5,
            linestyle="dashed",
            alpha=0.7
        )
        
        self.plot.band(
            "Channel Band",
            upper=self.upper_channel,
            lower=self.lower_channel,
            time=current_time,
            color="#9C27B0",
            alpha=0.1
        )
        
        # Цена входа
        if entry_price > 0 and inventory != 0:
            self.plot.line(
                "Entry Price",
                entry_price,
                current_time,
                color="#FF9800",
                linewidth=2,
                linestyle="dotted",
                alpha=0.9
            )
    
    def _get_position(self, engine) -> Optional[object]:
        """Получить текущую позицию."""
        for p in engine.positions:
            if p.status == "open" and p.size != 0:
                return p
        return None
    
    def _find_order(self, engine, order_id: int) -> Optional[object]:
        """Найти ордер по ID."""
        for o in engine.orders:
            if o.id == order_id:
                return o
        return None