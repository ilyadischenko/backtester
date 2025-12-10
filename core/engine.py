from dataclasses import dataclass
from typing import List, Optional, Dict
import random
import math
import polars as pl


@dataclass
class Order:
    id: int
    price: float
    size: float              # >0 buy, <0 sell
    type: str                # "market" | "limit"
    created_time: int
    exchange_time: int = 0
    filled: float = 0.0
    fill_price: float = 0.0
    status: str = "pending"  # "pending" | "new" | "filled" | "canceled"


@dataclass
class Position:
    id: int
    open_time: int
    size: float = 0.0
    price: float = 0.0
    close_time: Optional[int] = None
    fees: float = 0.0
    status: str = "open"
    realized_pnl: float = 0.0


class ExchangeEngine:
    EPSILON = 1e-12

    def __init__(
        self,
        data_bookticker: pl.DataFrame = None,
        data_trades: pl.DataFrame = None,
        strategy=None,
        taker_fee: float = 0.0004,   # 0.04%
        maker_fee: float = 0.0002,   # 0.02%
        network_delay: int = 2,
    ):
        self.bookticker = self._parse_bookticker(data_bookticker)
        self.trades = self._parse_trades(data_trades)
        
        self.events = (
            pl.concat([self.bookticker, self.trades], how="diagonal_relaxed")
            .sort("event_time")
        )

        self.strategy = strategy
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.network_delay = network_delay

        self.requests: List[dict] = []
        self.orders: List[Order] = []
        self.positions: List[Position] = []

        self.last_event_time: int = 0
        self.last_bookticker: Dict = {}
        self.last_trade: Dict = {}

    def _parse_bookticker(self, df):
        if df is None:
            return pl.DataFrame()
        return (
            df.rename({
                "column_1": "event_time",
                "column_2": "event_id",
                "column_3": "bid_price",
                "column_4": "bid_size",
                "column_5": "ask_price",
                "column_6": "ask_size",
            })
            .with_columns(pl.lit("bookticker").alias("event_type"))
        )

    def _parse_trades(self, df):
        if df is None:
            return pl.DataFrame()
        return (
            df.rename({
                "column_1": "event_time",
                "column_2": "event_id",
                "column_3": "price",
                "column_4": "quantity",
                "column_5": "time",
                "column_6": "is_maker",
            })
            .with_columns(pl.lit("trade").alias("event_type"))
        )

    # ═══════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════

    def run(self):
        for event in self.events.to_dicts():
            self.last_event_time = event["event_time"]

            if event["event_type"] == "bookticker":
                self.last_bookticker = event
            else:
                self.last_trade = event

            self._process_requests()
            self._process_orders()

            if self.strategy:
                self.strategy.on_tick(event, self)

    # ═══════════════════════════════════════════════════════════
    # ORDER MANAGEMENT
    # ═══════════════════════════════════════════════════════════

    def place_order(self, order_type: str, price: float, size: float) -> int:
        order_id = random.randint(0, 10_000_000)
        self.requests.append({
            "action": "place",
            "id": order_id,
            "type": order_type,
            "price": price,
            "size": size,
            "send_time": self.last_event_time,
        })
        return order_id

    def cancel_order(self, order_id: int):
        self.requests.append({
            "action": "cancel",
            "id": order_id,
            "send_time": self.last_event_time,
        })

    def _process_requests(self):
        pending = self.requests[:]
        self.requests.clear()

        for req in pending:
            arrival = req["send_time"] + self.network_delay

            if arrival > self.last_event_time:
                self.requests.append(req)
                continue

            if req["action"] == "place":
                self.orders.append(Order(
                    id=req["id"],
                    price=req["price"],
                    size=req["size"],
                    type=req["type"],
                    created_time=req["send_time"],
                    exchange_time=arrival,
                    status="new",
                ))

            elif req["action"] == "cancel":
                for o in self.orders:
                    if o.id == req["id"] and o.status == "new":
                        o.status = "canceled"
                        break

    # ═══════════════════════════════════════════════════════════
    # ORDER EXECUTION
    # ═══════════════════════════════════════════════════════════

    def _process_orders(self):
        if not self.last_bookticker:
            return

        bid = self.last_bookticker["bid_price"]
        ask = self.last_bookticker["ask_price"]

        for order in self.orders:
            if order.status != "new":
                continue

            if order.exchange_time > self.last_event_time:
                continue

            # ─────────────────────────────────────────────────
            # MARKET ORDER — исполняется сразу
            # ─────────────────────────────────────────────────
            if order.type == "market":
                exec_price = ask if order.size > 0 else bid
                self._fill_order(order, exec_price, is_maker=False)
                continue

            # ─────────────────────────────────────────────────
            # LIMIT ORDER
            # ─────────────────────────────────────────────────
            if order.type == "limit":
                if order.size > 0:  # Buy limit
                    # Агрессивный: цена >= ask → исполняем как taker по ask
                    if order.price >= ask:
                        self._fill_order(order, ask, is_maker=False)
                    # Пассивный: ждём пока ask <= нашей цены
                    elif ask <= order.price:
                        self._fill_order(order, order.price, is_maker=True)

                else:  # Sell limit (order.size < 0)
                    # Агрессивный: цена <= bid → исполняем как taker по bid
                    if order.price <= bid:
                        self._fill_order(order, bid, is_maker=False)
                    # Пассивный: ждём пока bid >= нашей цены
                    elif bid >= order.price:
                        self._fill_order(order, order.price, is_maker=True)

    def _fill_order(self, order: Order, price: float, is_maker: bool):
        """Полностью исполняет ордер."""
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        
        order.filled = order.size
        order.fill_price = price
        order.status = "filled"

        # Обновляем позицию
        self._update_position(order.size, price, fee_rate)

    # ═══════════════════════════════════════════════════════════
    # POSITION LOGIC
    # ═══════════════════════════════════════════════════════════

    def _update_position(self, trade_size: float, price: float, fee_rate: float):
        pos = self._get_or_create_position()

        trade_abs = abs(trade_size)
        fee = trade_abs * price * fee_rate
        pos.fees += fee

        pos_abs = abs(pos.size)

        # ─────────────────────────────────────────────────
        # Случай 0: Пустая позиция — открываем
        # ─────────────────────────────────────────────────
        if self._is_zero(pos.size):
            pos.size = trade_size
            pos.price = price
            return

        # ─────────────────────────────────────────────────
        # Случай 1: Та же сторона — усреднение
        # ─────────────────────────────────────────────────
        if pos.size * trade_size > 0:
            old_notional = pos_abs * pos.price
            new_notional = trade_abs * price
            total = pos_abs + trade_abs

            pos.price = (old_notional + new_notional) / total
            pos.size += trade_size
            return

        # ─────────────────────────────────────────────────
        # Случай 2: Противоположная сторона
        # ─────────────────────────────────────────────────
        closed_qty = min(pos_abs, trade_abs)

        # PnL по закрываемой части
        if pos.size > 0:
            pnl = (price - pos.price) * closed_qty
        else:
            pnl = (pos.price - price) * closed_qty

        pos.realized_pnl += pnl

        # 2.1: Частичное закрытие
        if trade_abs < pos_abs - self.EPSILON:
            pos.size += trade_size
            return

        # 2.2: Полное закрытие
        if self._is_close(trade_abs, pos_abs):
            self._close_position(pos)
            return

        # 2.3: Переворот
        remaining = trade_abs - pos_abs
        self._close_position(pos)

        new_pos = self._create_position()
        new_pos.size = math.copysign(remaining, trade_size)
        new_pos.price = price

    # ═══════════════════════════════════════════════════════════
    # POSITION HELPERS
    # ═══════════════════════════════════════════════════════════

    def _get_or_create_position(self) -> Position:
        for p in self.positions:
            if p.status == "open":
                return p
        return self._create_position()

    def _create_position(self) -> Position:
        pos = Position(
            id=random.randint(0, 10_000_000),
            open_time=self.last_event_time,
        )
        self.positions.append(pos)
        return pos

    def _close_position(self, pos: Position):
        pos.size = 0.0
        pos.close_time = self.last_event_time
        pos.status = "closed"

    # ═══════════════════════════════════════════════════════════
    # FLOAT HELPERS
    # ═══════════════════════════════════════════════════════════

    def _is_zero(self, v: float) -> bool:
        return abs(v) < self.EPSILON

    def _is_close(self, a: float, b: float) -> bool:
        return abs(a - b) < self.EPSILON

    # ═══════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════

    def get_position(self) -> Optional[Position]:
        for p in self.positions:
            if p.status == "open" and not self._is_zero(p.size):
                return p
        return None

    def get_position_size(self) -> float:
        pos = self.get_position()
        return pos.size if pos else 0.0

    def get_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions)

    def get_total_fees(self) -> float:
        return sum(p.fees for p in self.positions)

    def get_net_pnl(self) -> float:
        return self.get_realized_pnl() - self.get_total_fees()

    def get_unrealized_pnl(self) -> float:
        pos = self.get_position()
        if pos is None:
            return 0.0

        if not self.last_bookticker:
            return 0.0

        if pos.size > 0:
            mark = self.last_bookticker["bid_price"]
            return (mark - pos.price) * pos.size
        else:
            mark = self.last_bookticker["ask_price"]
            return (pos.price - mark) * abs(pos.size)
        
def test_limit_orders():
    engine = ExchangeEngine()
    engine.last_event_time = 1000
    engine.network_delay = 0  # Для теста убираем задержку
    
    # Симулируем bookticker
    engine.last_bookticker = {"bid_price": 99.0, "ask_price": 101.0}
    
    # ═══════════════════════════════════════════════════════════
    # Тест 1: Пассивный buy limit (ждёт)
    # ═══════════════════════════════════════════════════════════
    engine.place_order("limit", price=98.0, size=10)
    engine._process_requests()
    engine._process_orders()
    
    assert engine.orders[0].status == "new", "Должен ждать"
    print("✅ Тест 1: Пассивный buy limit ждёт")
    
    # Цена дошла
    engine.last_bookticker = {"bid_price": 97.0, "ask_price": 98.0}
    engine._process_orders()
    
    assert engine.orders[0].status == "filled"
    assert engine.orders[0].fill_price == 98.0  # По лимитной цене
    print("✅ Тест 1: Исполнился по лимитной цене")
    
    # ═══════════════════════════════════════════════════════════
    # Тест 2: Агрессивный buy limit (сразу)
    # ═══════════════════════════════════════════════════════════
    engine.orders.clear()
    engine.positions.clear()
    engine.last_bookticker = {"bid_price": 99.0, "ask_price": 100.0}
    
    engine.place_order("limit", price=101.0, size=10)  # Выше ask!
    engine._process_requests()
    engine._process_orders()
    
    assert engine.orders[0].status == "filled"
    assert engine.orders[0].fill_price == 100.0  # По ask (taker)
    print("✅ Тест 2: Агрессивный buy limit исполнился как taker")
    
    print("\n🎉 Все тесты пройдены!")

# test_limit_orders()