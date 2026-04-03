from collections import deque
from dataclasses import dataclass
from typing import Optional

from visualization.visualization import PlotRecorder


@dataclass(slots=True)
class Trade:
    time: int
    price: float
    volume: float
    is_buy: bool


class OrderflowImbalanceStrategy:
    """
    HFT‑логика:
    - Смотрим дисбаланс стакана (bookticker) и агрессивных сделок за короткое окно.
    - Входим по направлению, где стакан и сделки согласованы + есть импульс цены.
    - RR задаётся через stop_pct / take_pct (по умолчанию ~2:1).
    - Все расчёты по окну O(1) на тик за счёт инкрементальных сумм.
    """

    def __init__(
        # окна
        self,
        trades_window_ms: int = 1_000,       # окно по сделкам
        momentum_window_ms: int = 1_000,     # окно по mid для импульса
        recalc_ms: int = 0,                  # можно 0 => решение на каждом тике

        # фильтры
        min_trades: int = 10,
        min_window_volume: float = 5.0,
        max_spread_pct: float = 0.02,        # 0.02 => 0.02%

        # пороги сигналов
        min_book_imbalance: float = 0.6,
        min_trade_imbalance: float = 0.4,
        min_momentum_pct: float = 0.01,
        max_momentum_pct: float = 0.10,

        # мани‑менеджмент
        order_size: float = 50.0,
        max_position: float = 200.0,
        stop_pct: float = 0.05,              # 0.05% стоп
        take_pct: float = 0.10,              # 0.10% тейк → RR ~ 2:1
        max_holding_ms: int = 3_000,
        cooldown_ms: int = 500,
    ):
        self.initial_balance = 1000.0

        self.trades_window_ms = trades_window_ms
        self.momentum_window_ms = momentum_window_ms
        self.recalc_ms = recalc_ms

        self.min_trades = min_trades
        self.min_window_volume = min_window_volume
        self.max_spread_pct = max_spread_pct

        self.min_book_imbalance = min_book_imbalance
        self.min_trade_imbalance = min_trade_imbalance
        self.min_momentum_pct = min_momentum_pct
        self.max_momentum_pct = max_momentum_pct

        self.order_size = order_size
        self.max_position = max_position
        self.stop_pct = stop_pct
        self.take_pct = take_pct
        self.max_holding_ms = max_holding_ms
        self.cooldown_ms = cooldown_ms

        # данные
        self.trades: deque[Trade] = deque()
        self.mid_prices: deque[tuple[int, float]] = deque()  # (time, mid)

        self.best_bid = 0.0
        self.best_ask = 0.0
        self.bid_qty = 0.0
        self.ask_qty = 0.0
        self.mid_price = 0.0

        # инкрементальные суммы по окну сделок
        self.window_buy_vol: float = 0.0
        self.window_sell_vol: float = 0.0

        # состояние позиции
        self.in_position: bool = False
        self.position_side: int = 0
        self.entry_price: float = 0.0
        self.entry_time: int = 0
        self.stop_price: float = 0.0
        self.take_price: float = 0.0
        self.take_order_id: Optional[int] = None

        # тайминги
        self.last_recalc: int = 0
        self.last_exit_time: int = 0

        self.plot = PlotRecorder()

    # === Основной вход ===

    def on_tick(self, event, engine):
        t = event["event_time"]
        etype = event.get("event_type")

        # обновляем стакан и mid
        if etype == "bookticker":
            self.best_bid = event.get("bid_price", 0.0)
            self.best_ask = event.get("ask_price", 0.0)
            self.bid_qty = float(event.get("bid_qty", event.get("bid_size", 0.0)))
            self.ask_qty = float(event.get("ask_qty", event.get("ask_size", 0.0)))

            if self.best_bid > 0 and self.best_ask > 0:
                self.mid_price = (self.best_bid + self.best_ask) / 2.0
                self.mid_prices.append((t, self.mid_price))

        # обновляем сделки (сразу инкрементируем суммы)
        elif etype == "trade":
            p = event.get("price", 0.0)
            v = event.get("quantity", 0.0)
            if p > 0 and v > 0:
                is_buy = not event.get("is_maker", False)
                self.trades.append(Trade(t, p, v, is_buy))
                if is_buy:
                    self.window_buy_vol += v
                else:
                    self.window_sell_vol += v

        # чистим старое и поддерживаем агрегаты
        self._cleanup(t)

        if self.mid_price <= 0:
            return

        # ограничение частоты логики (если нужно)
        if t - self.last_recalc < self.recalc_ms:
            return
        self.last_recalc = t

        # актуальная позиция из движка
        pos = self._get_position(engine)
        inv = pos.size if pos else 0.0
        if inv == 0.0 and self.in_position:
            self._reset()

        if self.in_position:
            self._manage(engine, inv, t)
        else:
            if t - self.last_exit_time < self.cooldown_ms:
                return
            self._look_for_entry(engine, t)

        self._draw(t)

    # === Поддержка окон (O(1)) ===

    def _cleanup(self, t: int):
        # сделки
        cutoff_trades = t - self.trades_window_ms
        while self.trades and self.trades[0].time < cutoff_trades:
            old = self.trades.popleft()
            if old.is_buy:
                self.window_buy_vol -= old.volume
            else:
                self.window_sell_vol -= old.volume

        # mid‑цены
        cutoff_prices = t - self.momentum_window_ms
        while self.mid_prices and self.mid_prices[0][0] < cutoff_prices:
            self.mid_prices.popleft()

    # === Быстрые расчёты по окну ===

    def _calc_orderflow(self):
        """Без прохода по deque: используем инкрементальные суммы."""
        total_vol = self.window_buy_vol + self.window_sell_vol
        trade_imb = (
            (self.window_buy_vol - self.window_sell_vol) / total_vol
            if total_vol > 0
            else 0.0
        )

        book_denom = self.bid_qty + self.ask_qty
        book_imb = (
            (self.bid_qty - self.ask_qty) / book_denom
            if book_denom > 0
            else 0.0
        )

        return trade_imb, book_imb, total_vol

    def _calc_momentum(self):
        """Импульс = (последняя mid – первая mid) / первая mid, без цикла."""
        if len(self.mid_prices) < 2:
            return 0.0
        p0 = self.mid_prices[0][1]
        p1 = self.mid_prices[-1][1]
        if p0 <= 0:
            return 0.0
        return (p1 - p0) / p0 * 100.0

    # === Вход ===

    def _look_for_entry(self, engine, t: int):
        if len(self.trades) < self.min_trades:
            return
        if not self.mid_prices:
            return

        # фильтр по спреду
        spread = self.best_ask - self.best_bid
        if spread <= 0:
            return
        spread_pct = (spread / self.mid_price) * 100.0
        if spread_pct > self.max_spread_pct:
            return

        trade_imb, book_imb, total_vol = self._calc_orderflow()
        if total_vol < self.min_window_volume:
            return

        momentum = self._calc_momentum()
        side = 0

        # long
        if (
            book_imb >= self.min_book_imbalance
            and trade_imb >= self.min_trade_imbalance
            and self.min_momentum_pct <= momentum <= self.max_momentum_pct
        ):
            side = 1
        # short
        elif (
            book_imb <= -self.min_book_imbalance
            and trade_imb <= -self.min_trade_imbalance
            and -self.max_momentum_pct <= momentum <= -self.min_momentum_pct
        ):
            side = -1

        if side == 0:
            return

        pos = self._get_position(engine)
        inv = pos.size if pos else 0.0
        if abs(inv) + self.order_size > self.max_position:
            return

        self._enter(engine, side, t)

    def _enter(self, engine, side: int, t: int):
        size = self.order_size * side
        oid = engine.place_order("market", price=self.mid_price, size=size)
        if not oid:
            return

        self.in_position = True
        self.position_side = side
        self.entry_price = self.mid_price
        self.entry_time = t

        if side == 1:
            self.stop_price = self.entry_price * (1 - self.stop_pct / 100.0)
            self.take_price = self.entry_price * (1 + self.take_pct / 100.0)
        else:
            self.stop_price = self.entry_price * (1 + self.stop_pct / 100.0)
            self.take_price = self.entry_price * (1 - self.take_pct / 100.0)

        tp_size = -self.order_size * side
        self.take_order_id = engine.place_order("limit", price=self.take_price, size=tp_size)

        label = "LONG" if side == 1 else "SHORT"
        color = "#00E676" if side == 1 else "#FF5252"
        self.plot.marker(f"ENTRY {label}", self.mid_price, t,
                         marker="star", color=color, size=15)

    # === Управление и выход ===

    def _manage(self, engine, inv: float, t: int):
        if inv == 0.0:
            self._reset()
            return

        # максимальное время удержания
        if t - self.entry_time >= self.max_holding_ms:
            self._close(engine, inv, "TIMEOUT", t)
            return

        # стоп
        if self.position_side == 1 and self.mid_price <= self.stop_price:
            self._close(engine, inv, "STOP", t)
            return
        if self.position_side == -1 and self.mid_price >= self.stop_price:
            self._close(engine, inv, "STOP", t)
            return

        # тейк по лимитке
        if self.take_order_id:
            order = self._find_order(engine, self.take_order_id)
            if order and order.status == "filled":
                self.plot.marker("TAKE", self.take_price, t,
                                 marker="star", color="#00E676", size=15)
                self.last_exit_time = t
                self._reset()
                return

        # визуализация
        self.plot.line("Stop", self.stop_price, t,
                       color="#FF0000", linewidth=2, linestyle="dotted", alpha=0.7)
        self.plot.line("Take", self.take_price, t,
                       color="#00E676", linewidth=2, linestyle="dotted", alpha=0.7)

    def _close(self, engine, inv: float, reason: str, t: int):
        if self.take_order_id:
            order = self._find_order(engine, self.take_order_id)
            if order and order.status in ("new", "partially_filled"):
                engine.cancel_order(self.take_order_id)

        engine.place_order("market", price=self.mid_price, size=-inv)

        pnl_pct = (self.mid_price - self.entry_price) / self.entry_price * 100.0
        if self.position_side == -1:
            pnl_pct = -pnl_pct

        color = "#00E676" if pnl_pct >= 0 else "#FF5252"
        self.plot.marker(f"{reason} ({pnl_pct:+.2f}%)", self.mid_price, t,
                         marker="x", color=color, size=15)

        self.last_exit_time = t
        self._reset()

    def _reset(self):
        self.in_position = False
        self.position_side = 0
        self.entry_price = 0.0
        self.entry_time = 0
        self.stop_price = 0.0
        self.take_price = 0.0
        self.take_order_id = None

    # === Служебные ===

    def _draw(self, t: int):
        if self.mid_price > 0:
            self.plot.line("Mid", self.mid_price, t,
                           color="#2196F3", linewidth=1, alpha=0.4)
        if self.in_position:
            self.plot.line("Entry", self.entry_price, t,
                           color="#FFC107", linewidth=2, linestyle="dotted", alpha=0.9)

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