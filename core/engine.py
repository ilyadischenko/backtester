import random
import polars as pl
from strategies.base_strategy import BaseStrategy
from data_manager import dataManager


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
            self.bookticker = data_bookticker.with_columns(
                pl.lit("bookticker").alias("event_type")
            )
            self.bookticker = data_bookticker.rename({
                "column_1": "event_time", 
                "column_2": "event_id",
                "column_3": "bid_price",
                "column_4": "bid_size",
                "column_5": "ask_price",
                "column_6": "ask_size"
                })
        if data_trades is not None:
            self.trades = data_trades.with_columns(
                pl.lit("trade").alias("event_type")
            )
            self.trades = data_trades.rename({
                "column_1": "event_time", 
                "column_2": "event_id",
                "column_3": "price",
                "column_4": "quantity",
                "column_5": "time",
                "column_6": "is_maker"
            })
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
        self.delay = 2
        
        self.orders = []
        self.position = {}
        self.position_history = []

    def run(self):
        for event in self.events:

            # process orders

            self.strategy.on_tick(event, self.orders)


    def place_order(self, type, price, size, send_time):
        self.orders.append({
            "id": random.randint(0, 100000),
            "type": type,
            "price": price,
            "size": size,
            "fill": False,
            "fill_count": 0,
            "is_active": True,

            "send_time": send_time,
            "place_time": 0,
        })
    
    def cancel_order(self, id: int):
        for i in self.orders:
            if i.id == id and i.is_active == True:
                pass


    def process_orders(self):
        # если время отправленного ордера в будущем (от текущего события), то не исполняем пока
        pass
















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

# trades и book — это твои polars.DataFrame

# # 1. Переименуем колонки в осмысленные
# trades = trades.rename({
#     "column_1": "event_time_ms",
#     "column_2": "agg_id",
#     "column_3": "price",
#     "column_4": "qty",
#     "column_5": "trade_time_ms",
#     "column_6": "is_buyer_maker",  # 1/0
# })
engine = ExchangeEngine(data_bookticker=book, data_trades=trades)

print(engine.trades)
print(engine.bookticker)




