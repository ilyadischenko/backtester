from dataclasses import dataclass, field
from typing import List, Optional, Dict
import random
import math
import polars as pl


@dataclass(slots=True)
class Order:
    id: int
    price: float
    size: float
    type: str
    created_time: int
    exchange_time: int = 0
    filled: float = 0.0
    fill_price: float = 0.0
    fill_time: int = 0                           # НОВОЕ
    status: str = "pending"
    cancel_requested_at: Optional[int] = None    


@dataclass(slots=True)
class Position:
    id: int
    open_time: int
    size: float = 0.0
    price: float = 0.0                    # Средняя цена входа
    close_time: Optional[int] = None
    status: str = "open"
    
    # PnL разделён на компоненты
    realized_pnl: float = 0.0             # Чистый P&L (без комиссий)
    fees: float = 0.0                     # Сумма всех комиссий
    net_pnl: float = 0.0                  # realized_pnl - fees (заполняется при закрытии)
    
    # История ордеров
    order_ids: List[int] = field(default_factory=list)          # Все ордера, влиявшие на позицию
    
    def __post_init__(self):
        if self.order_ids is None:
            self.order_ids = []

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
        self.bookticker = self._prepare_bookticker(data_bookticker)
        self.trades = self._prepare_trades(data_trades)
        
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

    # def _parse_bookticker(self, df):
    #     if df is None:
    #         return pl.DataFrame()
    #     return (
    #         df.rename({
    #             "column_1": "event_time",
    #             "column_2": "event_id",
    #             "column_3": "bid_price",
    #             "column_4": "bid_size",
    #             "column_5": "ask_price",
    #             "column_6": "ask_size",
    #         })
    #         .with_columns(pl.lit("bookticker").alias("event_type"))
    #     )

    # def _parse_trades(self, df):
    #     if df is None:
    #         return pl.DataFrame()
    #     return (
    #         df.rename({
    #             "column_1": "event_time",
    #             "column_2": "event_id",
    #             "column_3": "price",
    #             "column_4": "quantity",
    #             "column_5": "time",
    #             "column_6": "is_maker",
    #         })
    #         .with_columns(pl.lit("trade").alias("event_type"))
    #     )

    def _prepare_bookticker(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Подготовка orderbook данных.
        Колонки уже именованы в DataManager:
        event_time, update_id, bid_price, bid_qty, ask_price, ask_qty
        """
        if df is None:
            return pl.DataFrame()
        
        return (
            df
            .rename({
                "update_id": "event_id",
                "bid_qty": "bid_size",
                "ask_qty": "ask_size",
            })
            .with_columns(pl.lit("bookticker").alias("event_type"))
        )

    def _prepare_trades(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Подготовка trades данных.
        Колонки уже именованы в DataManager:
        event_time, trade_id, price, qty, trade_time, is_maker
        """
        if df is None:
            return pl.DataFrame()
        
        return (
            df
            .rename({
                "trade_id": "event_id",
                "qty": "quantity",
                "trade_time": "time",
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
                    if o.id == req["id"] and o.cancel_requested_at is None:
                        o.cancel_requested_at = arrival  # ✅ Записываем время!
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
            # Race condition: cancel дошёл ДО этого тика — отменяем
            # ─────────────────────────────────────────────────
            if (order.cancel_requested_at is not None 
                and order.cancel_requested_at < self.last_event_time):
                order.status = "canceled"
                continue

            just_placed = order.exchange_time == self.last_event_time

            # ─────────────────────────────────────────────────
            # MARKET ORDER
            # ─────────────────────────────────────────────────
            if order.type == "market":
                exec_price = ask if order.size > 0 else bid
                self._fill_order(order, exec_price, is_maker=False)
                continue

            # ─────────────────────────────────────────────────
            # LIMIT ORDER
            # ─────────────────────────────────────────────────
            if order.type == "limit":
                filled = self._try_fill_limit_order(order, bid, ask, just_placed)

                # Race condition: cancel в ЭТОМ тике, но fill не случился
                if (not filled 
                    and order.cancel_requested_at is not None
                    and order.cancel_requested_at <= self.last_event_time):
                    order.status = "canceled"

    def _try_fill_limit_order(
        self, order: Order, bid: float, ask: float, just_placed: bool
    ) -> bool:
        """Пытается исполнить лимитный ордер. Возвращает True если исполнен."""
        is_buy = order.size > 0

        if is_buy:
            # Агрессивный: цена >= ask при размещении → taker
            if just_placed and order.price >= ask:
                self._fill_order(order, ask, is_maker=False)
                return True

            # Пассивный вариант 1: сделка прошла по уровню
            if self._trade_hit_level(order):
                self._fill_order(order, order.price, is_maker=True)
                return True
            
            # Пассивный вариант 2: ask провалился ниже нашего ордера
            # (в реальности наш ордер собрали бы)
            if ask <= order.price:
                self._fill_order(order, order.price, is_maker=True)
                return True

        else:  # Sell
            if just_placed and order.price <= bid:
                self._fill_order(order, bid, is_maker=False)
                return True

            if self._trade_hit_level(order):
                self._fill_order(order, order.price, is_maker=True)
                return True
            
            # Пассивный вариант 2: bid поднялся выше нашего ордера
            if bid >= order.price:
                self._fill_order(order, order.price, is_maker=True)
                return True

        return False

    def _trade_hit_level(self, order: Order) -> bool:
        """Проверяет, прошла ли сделка по уровню ордера"""
        if not self.last_trade:
            return False
        
        if self.last_trade.get("event_type") != "trade":
            return False

        trade_price = self.last_trade.get("price", 0)
        is_buyer_maker = self.last_trade.get("is_maker", False)

        if order.size > 0:  # Buy limit — ждём агрессивную продажу
            # is_buyer_maker=True → продавец был taker → sell прошёл вниз
            return is_buyer_maker and trade_price <= order.price
        
        else:  # Sell limit — ждём агрессивную покупку
            # is_buyer_maker=False → покупатель был taker → buy прошёл вверх
            return (not is_buyer_maker) and trade_price >= order.price
        

    def _fill_order(self, order: Order, price: float, is_maker: bool):
        """Полностью исполняет ордер."""
        fee_rate = self.maker_fee if is_maker else self.taker_fee

        order.filled = order.size
        order.fill_price = price
        order.fill_time = self.last_event_time
        order.status = "filled"

        # Передаём order.id в позицию
        self._update_position(order.size, price, fee_rate, order.id)

    # ═══════════════════════════════════════════════════════════
    # POSITION LOGIC
    # ═══════════════════════════════════════════════════════════

    def _update_position(self, trade_size: float, price: float, fee_rate: float, order_id: int):
        pos = self._get_or_create_position()
        pos.order_ids.append(order_id)

        trade_abs = abs(trade_size)
        fee = trade_abs * price * fee_rate
        # НЕ добавляем fee сразу - распределим позже

        pos_abs = abs(pos.size)

        # Случай 0: открытие
        if self._is_zero(pos.size):
            pos.size = trade_size
            pos.price = price
            pos.fees += fee  # ✅ Добавляем здесь
            return

        # Случай 1: усреднение
        if pos.size * trade_size > 0:
            old_notional = pos_abs * pos.price
            new_notional = trade_abs * price
            total = pos_abs + trade_abs
            pos.price = (old_notional + new_notional) / total
            pos.size += trade_size
            pos.fees += fee  # ✅ Добавляем здесь
            return

        # Случай 2: закрытие
        closed_qty = min(pos_abs, trade_abs)
        
        if pos.size > 0:
            pnl = (price - pos.price) * closed_qty
        else:
            pnl = (pos.price - price) * closed_qty
        
        pos.realized_pnl += pnl

        # 2.1: частичное
        if trade_abs < pos_abs - self.EPSILON:
            pos.size += trade_size
            pos.fees += fee  # ✅ Вся комиссия за закрытие
            return

        # 2.2: полное
        if self._is_close(trade_abs, pos_abs):
            pos.fees += fee  # ✅ Вся комиссия за закрытие
            self._close_position(pos)
            return

        # 2.3: переворот
        remaining = trade_abs - pos_abs
        
        # Распределяем комиссию пропорционально
        close_fee = fee * (pos_abs / trade_abs)
        open_fee = fee * (remaining / trade_abs)
        
        pos.fees += close_fee  # ✅ Комиссия за закрытие
        self._close_position(pos)
        
        # Новая позиция
        new_pos = self._create_position()
        new_pos.size = math.copysign(remaining, trade_size)
        new_pos.price = price
        new_pos.order_ids.append(order_id)
        new_pos.fees = open_fee  # ✅ Комиссия за открытие
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
        """Закрывает позицию и рассчитывает финальный net PnL"""
        pos.size = 0.0
        pos.close_time = self.last_event_time
        pos.status = "closed"
        
        # Финальный расчёт: прибыль минус комиссии
        pos.net_pnl = pos.realized_pnl - pos.fees

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

    def get_realized_pnl(self) -> float:
        """Суммарный realized PnL (без комиссий)"""
        return sum(p.realized_pnl for p in self.positions)

    def get_total_fees(self) -> float:
        """Суммарные комиссии по всем позициям"""
        return sum(p.fees for p in self.positions)

    def get_net_pnl(self) -> float:
        """Чистая прибыль = PnL - комиссии"""
        return sum(p.net_pnl for p in self.positions if p.status == "closed")

    def get_unrealized_pnl(self) -> float:
        """Нереализованный PnL открытой позиции"""
        pos = self.get_position()
        if pos is None:
            return 0.0

        if not self.last_bookticker:
            return 0.0

        if pos.size > 0:
            mark = self.last_bookticker["bid_price"]
            unrealized = (mark - pos.price) * pos.size
        else:
            mark = self.last_bookticker["ask_price"]
            unrealized = (pos.price - mark) * abs(pos.size)
        
        return unrealized

    def get_unrealized_net_pnl(self) -> float:
        """Нереализованный чистый PnL (с учётом комиссий)"""
        pos = self.get_position()
        if pos is None:
            return 0.0
        
        unrealized_gross = self.get_unrealized_pnl()
        return unrealized_gross - pos.fees

    def get_position_summary(self, position_id: int) -> dict:
        """Детальная информация о позиции"""
        pos = next((p for p in self.positions if p.id == position_id), None)
        if not pos:
            return {}
        
        orders = [o for o in self.orders if o.id in pos.order_ids]
        
        return {
            "id": pos.id,
            "status": pos.status,
            "open_time": pos.open_time,
            "close_time": pos.close_time,
            "entry_price": pos.price,
            "size": pos.size,
            "realized_pnl": pos.realized_pnl,
            "fees": pos.fees,
            "net_pnl": pos.net_pnl,
            "order_count": len(pos.order_ids),
            "orders": [
                {
                    "id": o.id,
                    "type": o.type,
                    "size": o.size,
                    "price": o.fill_price,
                    "time": o.fill_time,
                }
                for o in orders
            ],
        }
        
    def get_position(self):
        """Возвращает текущую открытую позицию (если есть)"""
        for p in self.positions:
            if p.status == "open":
                return p
        return None