from dataclasses import dataclass
import random
from typing import List
import polars as pl
from strategies.base_strategy import BaseStrategy
from data_manager import dataManager





@dataclass
class Order:
    id: int 
    price: float 
    size: float
    type: str 
    side: str
    filled: float = 0.0 
    status: str = "new"
    added_time: int


@dataclass
class Position:
    id: int
    size: float = 0.0
    price: float = 0.0
    open_time: int
    close_time: int = None
    fees: float = 0.0
    status: str = "open"


                                 

class ExchangeEngine:
    def __init__(
            self,
            data_bookticker: pl.DataFrame = None,
            data_trades: pl.DataFrame = None,

            strategy: BaseStrategy = None,   

            market_fee: float = 0.04,
            limit_fee: float = 0.02,
        ):
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

        self.events = (
            pl.concat([self.bookticker, self.trades], how="diagonal_relaxed")
            .sort("event_time")
        )


        if strategy is None:
            print("В движок не загружена стратегия")
        else:
            self.strategy = strategy



        self.market_fee = market_fee
        self.limit_fee = limit_fee
        self.network_delay = 2
        
        self.requests = []
        self.orders: List[Order] = []
        self.positions: List[Position] = []
        self.position_history = []

        self.last_event_time = 0
        self.last_bookticker_event = {}
        self.last_trade_event = {}

    def run(self):
        for event in self.events.to_dicts():
            self.last_event_time = event["event_time"]
            etype = event["event_type"]
            if etype == "bookticker":
                self.last_bookticker_event = event
            elif etype == "trade":
                self.last_trade_event = event

            self.process_requests()

            self.strategy.on_tick(event, self.orders)


    def place_order(self, type, side, price, size):
        self.requests.append({
            "request_type": "place_order",


            "id": random.randint(0, 10000000),
            "order_type": type,
            "side": side,
            "price": price,
            "size": size,

            "send_time": self.last_event_time,
        })
    
    def cancel_order(self, id: int):
        self.requests.append({
            "request_type": "cancel_order",

            "id": id
        })


    def process_requests(self):
        for request in self.requests:
            if request["request_type"] == "place_order":
                if request["order_type"] == "limit":
                    self.orders.append(Order(
                        id=request["id"],
                        price=request["price"],
                        size=request["size"],
                        type=request["order_type"],
                        side=request["side"],
                        added_time=self.last_event_time + self.network_delay
                    ))
                elif request["order_type"] == "market":
                    self.orders.append(Order(
                        id=request["id"],                            
                        price=request["price"],
                        size=request["size"],
                        type=request["order_type"],
                        side=request["side"],
                        added_time=self.last_event_time + self.network_delay
                    ))
            elif request["request_type"] == "cancel_order":
                for i in self.orders:
                    if i.id == request["id"]:
                        i.status = "canceled"
            else:
                print("Неизвестный запрос")
            
            self.requests.remove(request)


    def process_orders(self):
        if not self.last_bookticker_event:
            return
        for o in self.orders:
            if o.status != "new":
                continue

            if o.added_time > self.last_event_time:
                continue  # еще не «долетел»


            if o.type == "market":
                position = self.get_actual_position()
                if not position:
                    position = self.create_position()
                if o.side == "buy":
                    position.price = self.last_bookticker_event["ask_price"]
                    position.size += i.size
    


    def get_actual_position(self) -> Position | None:
        for i in self.positions:
            if i.status == open:
                return i
            
        return None
                    
    def create_position(self):
        self.positions.push(Position(id=random.randint(0, 10_000_000), open_time=self.last_event_time))









df = dataManager.load_timerange(
        exchange="binance",
        symbol="luna2usdt",
        start_time="2025-12-07 00:00:00",
        end_time="2025-12-07 23:00:00",
        data_type="all",
        market_type="futures"
    )
trades = df.trades
book = df.orderbook

engine = ExchangeEngine(data_bookticker=book, data_trades=trades)

print(engine.trades)
print(engine.bookticker)




