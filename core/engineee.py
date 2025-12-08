from dataclasses import dataclass
import random
from typing import List, Optional
import polars as pl
from strategies.base_strategy import BaseStrategy
from data_manager import dataManager


# ---------- Модели ----------

@dataclass
class Order:
    id: int
    price: float
    size: float
    type: str      # "limit" / "market"
    side: str      # "buy" / "sell"
    filled: float = 0.0
    status: str = "new"     # "new" / "filled" / "partial" / "canceled"
    added_time: int = 0


@dataclass
class Position:
    """
    Отдельный «отрезок» позиции (открытие–закрытие).
    Для отслеживания истории входов/выходов/переворотов.
    """
    id: int
    size: float            # +long / -short
    price: float           # средняя цена входа по этому отрезку
    open_time: int
    close_time: Optional[int] = None
    fees: float = 0.0
    status: str = "open"   # "open" / "closed"


@dataclass
class NetPosition:
    """
    Агрегированная текущая нетто-позиция.
    """
    size: float = 0.0      # >0 long, <0 short, 0 flat
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    fees: float = 0.0


# ---------- Движок ----------

class ExchangeEngine:
    def __init__(
        self,
        data_bookticker: pl.DataFrame = None,
        data_trades: pl.DataFrame = None,
        strategy: BaseStrategy = None,
        market_fee: float = 0.04,   # в %
        limit_fee: float = 0.02,    # в %
    ):
        # --- подготовка данных ---
        if data_bookticker is not None:
            self.bookticker = (
                data_bookticker
                .rename({
                    "column_1": "event_time",
                    "column_2": "event_id",
                    "column_3": "bid_price",
                    "column_4": "bid_size",
                    "column_5": "ask_price",
                    "column_6": "ask_size",
                })
                .with_columns(pl.lit("bookticker").alias("event_type"))
            )
        else:
            self.bookticker = None

        if data_trades is not None:
            self.trades = (
                data_trades
                .rename({
                    "column_1": "event_time",
                    "column_2": "event_id",
                    "column_3": "price",
                    "column_4": "quantity",
                    "column_5": "time",
                    "column_6": "is_maker",
                })
                .with_columns(pl.lit("trade").alias("event_type"))
            )
        else:
            self.trades = None

        frames = [df for df in [self.bookticker, self.trades] if df is not None]
        if not frames:
            raise ValueError("Нет данных bookticker/trades")

        self.events = (
            pl.concat(frames, how="diagonal_relaxed")
            .sort("event_time")
        )

        # --- стратегия ---
        if strategy is None:
            print("В движок не загружена стратегия")
        self.strategy = strategy

        # --- параметры ---
        self.market_fee = market_fee
        self.limit_fee = limit_fee
        self.network_delay = 2  # мс/тик условная задержка сети

        # --- состояние ---
        self.requests: List[dict] = []
        self.orders: List[Order] = []
        self.positions: List[Position] = []     # история отрезков позиции
        self.position_history: List[dict] = []  # история net_position

        self.net_position = NetPosition()

        self.last_event_time: int = 0
        self.last_bookticker_event: dict = {}
        self.last_trade_event: dict = {}

    # ---------- Основной цикл ----------

    def run(self):
            for event in self.events.to_dicts():
                self.last_event_time = event["event_time"]
                etype = event["event_type"]

                if etype == "bookticker":
                    self.last_bookticker_event = event
                elif etype == "trade":
                    self.last_trade_event = event

                # 1. Обработка старых запросов -> создаём/отменяем ордера
                self.process_requests()

                # 2. Исполняем ордера, обновляем позиции
                self.process_orders()

                # 3. Вызываем стратегию
                if self.strategy is not None:
                    context = {
                        "net_position": self.net_position,
                        "open_positions": [p for p in self.positions if p.status == "open"],
                        "open_orders": [o for o in self.orders if o.status == "new"],
                    }

                    commands = self.strategy.on_tick(event, context) or []
                    self._enqueue_strategy_commands(commands)

    def _enqueue_strategy_commands(self, commands: List[dict]):
        """
        Переводит команды стратегии в requests движка.
        Стратегия не знает ни про last_event_time, ни про network_delay.
        """
        for cmd in commands:
            ctype = cmd.get("cmd")

            if ctype == "place_order":
                self.requests.append({
                    "request_type": "place_order",
                    "id": random.randint(0, 10_000_000),
                    "order_type": cmd["type"],    # "market"/"limit"
                    "side": cmd["side"],
                    "price": cmd.get("price", 0.0) or 0.0,
                    "size": float(cmd["size"]),
                    "send_time": self.last_event_time,
                })

            elif ctype == "cancel_order":
                self.requests.append({
                    "request_type": "cancel_order",
                    "id": int(cmd["order_id"]),
                })

            else:
                # можно логировать неизвестную команду
                pass

    # ---------- API для стратегии ----------

    def place_order(self, type: str, side: str, price: float, size: float):
        self.requests.append({
            "request_type": "place_order",
            "id": random.randint(0, 10_000_000),
            "order_type": type,    # "market" / "limit"
            "side": side,          # "buy" / "sell"
            "price": price,
            "size": size,
            "send_time": self.last_event_time,
        })

    def cancel_order(self, id: int):
        self.requests.append({
            "request_type": "cancel_order",
            "id": id,
        })

    # ---------- Обработка запросов ----------

    def process_requests(self):
        # итерируемся по копии и удаляем обработанные
        for request in list(self.requests):
            rtype = request["request_type"]

            if rtype == "place_order":
                order = Order(
                    id=request["id"],
                    price=request["price"],
                    size=request["size"],
                    type=request["order_type"],
                    side=request["side"],
                    added_time=request["send_time"] + self.network_delay,
                )
                self.orders.append(order)
                self.requests.remove(request)

            elif rtype == "cancel_order":
                for o in self.orders:
                    if o.id == request["id"] and o.status == "new":
                        o.status = "canceled"
                self.requests.remove(request)

            else:
                print("Неизвестный запрос:", rtype)
                self.requests.remove(request)

    # ---------- Исполнение ордеров + позиции ----------

    def process_orders(self):
        """
        Упрощённое исполнение:
        - как только текущее время >= added_time, считаем, что ордер исполнился по лучшей цене.
        - market: buy по ask, sell по bid
        - limit: buy по min(limit_price, ask), sell по max(limit_price, bid)
        """
        if not self.last_bookticker_event:
            return

        bid = self.last_bookticker_event["bid_price"]
        ask = self.last_bookticker_event["ask_price"]

        for o in self.orders:
            if o.status != "new":
                continue

            if o.added_time > self.last_event_time:
                continue  # еще не «долетел»

            # определяем цену исполнения
            if o.type == "market":
                exec_price = ask if o.side == "buy" else bid
            else:  # limit
                if o.side == "buy":
                    # если лимит выше ask — исполнился по ask (best ask)
                    if o.price >= ask:
                        exec_price = ask
                    else:
                        # можно оставить неисполненным, но для простоты считаем
                        # что исполнился по своей цене
                        exec_price = o.price
                else:  # sell
                    if o.price <= bid:
                        exec_price = bid
                    else:
                        exec_price = o.price

            exec_size = o.size  # пока без частичных исполнений

            # применяем исполнение к позиции
            self._apply_fill(order=o, fill_size=exec_size, fill_price=exec_price)

            o.filled = exec_size
            o.status = "filled"

    def _apply_fill(self, order: Order, fill_size: float, fill_price: float):
        """
        Обновление net_position и списка Position с учётом:
        - открытия
        - частичного закрытия
        - полного закрытия
        - переворота (реверса)
        """

        side_sign = 1 if order.side == "buy" else -1
        trade_size = side_sign * fill_size  # +long / -short
        old_size = self.net_position.size
        new_size = old_size + trade_size

        # комиссия
        fee_rate = self.market_fee if order.type == "market" else self.limit_fee
        fee = abs(fill_size * fill_price) * fee_rate / 100.0
        self.net_position.fees += fee

        # ---- кейсы ----
        # 1) не было позиции, открываем новую
        if old_size == 0:
            self.net_position.size = trade_size
            self.net_position.avg_price = fill_price

            pos = Position(
                id=random.randint(0, 10_000_000),
                size=trade_size,
                price=fill_price,
                open_time=self.last_event_time,
                fees=fee,
                status="open",
            )
            self.positions.append(pos)

        # 2) та же сторона (увеличение long или short)
        elif (old_size > 0 and trade_size > 0) or (old_size < 0 and trade_size < 0):
            # усреднение
            total_notional = abs(old_size) * self.net_position.avg_price + abs(trade_size) * fill_price
            total_size = abs(old_size) + abs(trade_size)
            self.net_position.size = new_size
            self.net_position.avg_price = total_notional / total_size

            # добавляем новый отрезок позиции
            pos = Position(
                id=random.randint(0, 10_000_000),
                size=trade_size,
                price=fill_price,
                open_time=self.last_event_time,
                fees=fee,
                status="open",
            )
            self.positions.append(pos)

        # 3) противоположная сторона: закрытие/переворот
        else:
            # модульные размеры
            remaining_to_close = abs(trade_size)
            direction = 1 if old_size > 0 else -1  # 1=long был, -1=short был

            # PnL при закрытии части позиции:
            # для long: (fill_price - avg_price) * closed_qty
            # для short: (avg_price - fill_price) * closed_qty
            close_qty = min(abs(old_size), remaining_to_close)
            if direction > 0:  # закрываем long
                pnl = (fill_price - self.net_position.avg_price) * close_qty
            else:  # закрываем short
                pnl = (self.net_position.avg_price - fill_price) * close_qty

            self.net_position.realized_pnl += pnl
            # уменьшаем/обнуляем размер
            closed_part = direction * close_qty
            self.net_position.size = old_size + closed_part * (-1)  # т.к. trade_size имеет противоположный знак

            remaining_to_close -= close_qty

            # если полностью закрылись и ещё остался остаток => переворот
            if abs(self.net_position.size) < 1e-12 and remaining_to_close > 0:
                # остаток открывает новую позицию в сторону trade_size
                new_side_size = remaining_to_close * side_sign
                self.net_position.size = new_side_size
                self.net_position.avg_price = fill_price

                pos = Position(
                    id=random.randint(0, 10_000_000),
                    size=new_side_size,
                    price=fill_price,
                    open_time=self.last_event_time,
                    fees=fee,
                    status="open",
                )
                self.positions.append(pos)

            # если не полностью закрылись (частичное закрытие)
            # то new_size уже отражён в net_position.size выше
            # avg_price старой части не меняем

        # ---- сохраняем снимок net_position ----
        self.position_history.append({
            "time": self.last_event_time,
            "size": self.net_position.size,
            "avg_price": self.net_position.avg_price,
            "realized_pnl": self.net_position.realized_pnl,
            "fees": self.net_position.fees,
        })


# ---------- пример использования ----------

df = dataManager.load_timerange(
    exchange="binance",
    symbol="cvcusdt",
    start_time="2025-12-05 13:00:00",
    end_time="2025-12-05 14:00:00",
    data_type="all",
    market_type="futures"
)
trades = df.trades
book = df.orderbook

engine = ExchangeEngine(data_bookticker=book, data_trades=trades, strategy=None)

print(engine.trades)
print(engine.bookticker)