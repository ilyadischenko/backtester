# strategies/grid_averaging.py
from typing import Optional


class GridAveragingStrategy:
    """
    Стратегия усреднения позиции по сетке:
    
    1. Открываем начальную позицию по рынку
    2. Если цена идёт против нас на N% — добавляем к позиции (усредняем)
    3. Максимум M добавлений
    4. Закрываем всю позицию при достижении тейк-профита от средней цены
    5. Стоп-лосс на всю позицию
    
    Это создаст несколько горизонтальных линий средней цены!
    """

    def __init__(
            self,
            initial_balance: float = 10000.0,
            initial_order_usd: float = 100.0,
            add_order_usd: float = 150.0,      # Добавление больше чем первый вход
            grid_step_bps: float = 50.0,       # Шаг сетки 0.5%
            max_additions: int = 3,            # Максимум 3 добавления
            take_profit_bps: float = 100.0,    # TP 1% от средней
            stop_loss_bps: float = 300.0,      # SL 3% от средней
            direction: str = "long",           # "long" или "short"
    ):
        self.initial_balance = initial_balance
        self.initial_order_usd = initial_order_usd
        self.add_order_usd = add_order_usd
        self.grid_step_bps = grid_step_bps
        self.max_additions = max_additions
        self.take_profit_bps = take_profit_bps
        self.stop_loss_bps = stop_loss_bps
        self.direction = direction

        # State
        self.position_opened = False
        self.additions_count = 0
        self.last_add_price: Optional[float] = None
        self.warmup_ticks = 0

    def on_tick(self, event, engine):
        if event["event_type"] != "bookticker":
            return

        bid = event["bid_price"]
        ask = event["ask_price"]
        mid = (bid + ask) / 2

        # Небольшой прогрев — ждём 100 тиков
        if self.warmup_ticks < 100:
            self.warmup_ticks += 1
            return

        # Получаем текущую позицию
        pos = self._get_open_position(engine)
        pos_size = pos.size if pos else 0.0

        # ═══════════════════════════════════════════════════════
        # Если позиции нет — открываем первую
        # ═══════════════════════════════════════════════════════
        if not self.position_opened and pos_size == 0:
            self._open_initial_position(engine, mid)
            return

        # ═══════════════════════════════════════════════════════
        # Если есть позиция — проверяем TP/SL и добавления
        # ═══════════════════════════════════════════════════════
        if pos and pos_size != 0:
            avg_price = pos.price

            # Проверяем тейк-профит от средней цены
            if self._check_take_profit(pos_size, mid, avg_price):
                self._close_position(engine, pos_size)
                self._reset_state()
                return

            # Проверяем стоп-лосс от средней цены
            if self._check_stop_loss(pos_size, mid, avg_price):
                self._close_position(engine, pos_size)
                self._reset_state()
                return

            # Проверяем условие для добавления в позицию
            if self.additions_count < self.max_additions:
                if self._should_add_to_position(pos_size, mid):
                    self._add_to_position(engine, mid)
                    return

    def _open_initial_position(self, engine, mid: float):
        """Открываем первую позицию"""
        size_in_coins = self.initial_order_usd / mid

        if self.direction == "long":
            engine.place_order("market", price=0, size=size_in_coins)
            self.last_add_price = mid
        else:  # short
            engine.place_order("market", price=0, size=-size_in_coins)
            self.last_add_price = mid

        self.position_opened = True
        print(f"[OPEN] Initial {self.direction} position at {mid:.5f}")

    def _should_add_to_position(self, pos_size: float, mid: float) -> bool:
        """
        Проверяет, нужно ли добавить к позиции.
        Добавляем если цена ушла против нас на grid_step_bps от последнего добавления.
        """
        if self.last_add_price is None:
            return False

        price_change_bps = (mid - self.last_add_price) / self.last_add_price * 10000

        if self.direction == "long":
            # Long: добавляем если цена упала
            return price_change_bps <= -self.grid_step_bps
        else:
            # Short: добавляем если цена выросла
            return price_change_bps >= self.grid_step_bps

    def _add_to_position(self, engine, mid: float):
        """Добавляем к позиции (усредняем)"""
        size_in_coins = self.add_order_usd / mid

        if self.direction == "long":
            engine.place_order("market", price=0, size=size_in_coins)
        else:
            engine.place_order("market", price=0, size=-size_in_coins)

        self.additions_count += 1
        self.last_add_price = mid
        print(f"[ADD {self.additions_count}] Adding to {self.direction} at {mid:.5f}")

    def _check_take_profit(self, pos_size: float, mid: float, avg_price: float) -> bool:
        """Проверяет достижение тейк-профита от средней цены"""
        if self.direction == "long":
            pnl_bps = (mid - avg_price) / avg_price * 10000
        else:
            pnl_bps = (avg_price - mid) / avg_price * 10000

        if pnl_bps >= self.take_profit_bps:
            print(f"[TP] Take profit hit at {mid:.5f} (avg: {avg_price:.5f}, pnl: {pnl_bps:.1f}bps)")
            return True
        return False

    def _check_stop_loss(self, pos_size: float, mid: float, avg_price: float) -> bool:
        """Проверяет достижение стоп-лосса от средней цены"""
        if self.direction == "long":
            pnl_bps = (mid - avg_price) / avg_price * 10000
        else:
            pnl_bps = (avg_price - mid) / avg_price * 10000

        if pnl_bps <= -self.stop_loss_bps:
            print(f"[SL] Stop loss hit at {mid:.5f} (avg: {avg_price:.5f}, pnl: {pnl_bps:.1f}bps)")
            return True
        return False

    def _close_position(self, engine, pos_size: float):
        """Закрываем всю позицию"""
        engine.place_order("market", price=0, size=-pos_size)
        print(f"[CLOSE] Closing entire position (size: {pos_size:.8f})")

    def _reset_state(self):
        """Сбрасываем состояние для новой позиции"""
        self.position_opened = False
        self.additions_count = 0
        self.last_add_price = None

    def _get_open_position(self, engine):
        """Получает открытую позицию из движка"""
        for p in engine.positions:
            if p.status == "open":
                return p
        return None


# ═══════════════════════════════════════════════════════════════
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ═══════════════════════════════════════════════════════════════

