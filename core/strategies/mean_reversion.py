# strategies/mean_reversion.py
from typing import Optional
from collections import deque


class MeanReversionStrategy:
    """
    Mean Reversion стратегия:
    - Считаем скользящее среднее mid price
    - Покупаем когда цена ниже MA на N%
    - Продаём когда цена выше MA на N%
    - Тейк-профит при возврате к MA
    - Стоп-лосс если цена ушла ещё дальше
    """

    def __init__(
            self,
            initial_balance: float = 10000.0,
            order_size_usd: float = 100.0,
            ma_period: int = 2000,  # Период MA в тиках
            entry_threshold_bps: float = 30.0,  # Вход при отклонении 0.3%
            take_profit_bps: float = 25.0,  # Тейк 0.25%
            stop_loss_bps: float = 50.0,  # Стоп 0.5%
            max_position_usd: float = 300.0,  # Макс позиция
            cooldown_ms: int = 5000,  # Пауза между сделками
    ):
        self.initial_balance = initial_balance
        self.order_size_usd = order_size_usd
        self.ma_period = ma_period
        self.entry_threshold_bps = entry_threshold_bps
        self.take_profit_bps = take_profit_bps
        self.stop_loss_bps = stop_loss_bps
        self.max_position_usd = max_position_usd
        self.cooldown_ms = cooldown_ms

        # State
        self.prices: deque = deque(maxlen=ma_period)
        self.last_trade_time: int = 0
        self.entry_price: Optional[float] = None

    def on_tick(self, event, engine):
        if event["event_type"] != "bookticker":
            return

        current_time = event["event_time"]
        bid = event["bid_price"]
        ask = event["ask_price"]
        mid = (bid + ask) / 2

        # Собираем данные для MA
        self.prices.append(mid)

        # Ждём пока накопится достаточно данных
        if len(self.prices) < self.ma_period:
            return

        ma = sum(self.prices) / len(self.prices)
        deviation_bps = (mid - ma) / ma * 10000

        # ═══════════════════════════════════════════════════════
        # Получаем текущую позицию из движка
        # ═══════════════════════════════════════════════════════
        pos = self._get_open_position(engine)
        pos_size = pos.size if pos else 0.0
        pos_value = abs(pos_size * mid)

        # ═══════════════════════════════════════════════════════
        # Если есть позиция — проверяем TP/SL
        # ═══════════════════════════════════════════════════════
        if pos_size != 0 and self.entry_price:
            pnl_bps = self._get_pnl_bps(pos_size, mid)

            # Take Profit
            if pnl_bps >= self.take_profit_bps:
                self._close_position(engine, pos_size)
                return

            # Stop Loss
            if pnl_bps <= -self.stop_loss_bps:
                self._close_position(engine, pos_size)
                return

        # ═══════════════════════════════════════════════════════
        # Cooldown
        # ═══════════════════════════════════════════════════════
        if current_time - self.last_trade_time < self.cooldown_ms:
            return

        # ═══════════════════════════════════════════════════════
        # Вход в позицию
        # ═══════════════════════════════════════════════════════

        # Проверяем лимит позиции
        if pos_value >= self.max_position_usd:
            return

        size_in_coins = self.order_size_usd / mid

        # Цена ниже MA — покупаем (ждём возврат вверх)
        if deviation_bps <= -self.entry_threshold_bps:
            # Можем покупать только если нет позиции или она short
            if pos_size <= 0:
                engine.place_order("market", price=0, size=size_in_coins)
                self.entry_price = ask
                self.last_trade_time = current_time
            return

        # Цена выше MA — продаём (ждём возврат вниз)
        if deviation_bps >= self.entry_threshold_bps:
            # Можем продавать только если нет позиции или она long
            if pos_size >= 0:
                engine.place_order("market", price=0, size=-size_in_coins)
                self.entry_price = bid
                self.last_trade_time = current_time
            return

    def _get_open_position(self, engine):
        """Получает открытую позицию из движка."""
        for p in engine.positions:
            if p.status == "open":
                return p
        return None

    def _get_pnl_bps(self, pos_size: float, mid: float) -> float:
        """PnL в базисных пунктах."""
        if not self.entry_price or self.entry_price == 0:
            return 0.0

        if pos_size > 0:  # Long
            return (mid - self.entry_price) / self.entry_price * 10000
        else:  # Short
            return (self.entry_price - mid) / self.entry_price * 10000

    def _close_position(self, engine, pos_size: float):
        """Закрываем позицию маркет ордером."""
        engine.place_order("market", price=0, size=-pos_size)
        self.entry_price = None
        self.last_trade_time = engine.last_event_time