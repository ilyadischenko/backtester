# strategies/market_maker.py
from typing import Set


class MarketMakerStrategy:

    def __init__(
            self,
            initial_balance: float = 10000.0,
            order_size_usd: float = 100.0,
            spread_bps: float = 5.0,
            max_position_usd: float = 500.0,
            refresh_interval_ms: int = 1000,
    ):
        self.initial_balance = initial_balance
        self.order_size_usd = order_size_usd
        self.spread_bps = spread_bps
        self.max_position_usd = max_position_usd
        self.refresh_interval_ms = refresh_interval_ms

        self.last_refresh_time = 0
        self.active_order_ids: Set[int] = set()

    def on_tick(self, event, engine):
        if event["event_type"] != "bookticker":
            return

        current_time = event["event_time"]

        if current_time - self.last_refresh_time < self.refresh_interval_ms:
            return

        self.last_refresh_time = current_time

        # Отменяем старые
        for order_id in list(self.active_order_ids):
            engine.cancel_order(order_id)
        self.active_order_ids.clear()

        # Цены
        bid = event["bid_price"]
        ask = event["ask_price"]
        mid = (bid + ask) / 2

        half_spread = mid * (self.spread_bps / 10000) / 2
        our_bid = mid - half_spread
        our_ask = mid + half_spread

        size_in_coins = self.order_size_usd / mid

        pos_size = engine.get_position_size()
        pos_value = abs(pos_size * mid)

        # Контроль позиции
        if pos_value > self.max_position_usd:
            if pos_size > 0:
                oid = engine.place_order("limit", our_ask, -size_in_coins)
                self.active_order_ids.add(oid)
            else:
                oid = engine.place_order("limit", our_bid, size_in_coins)
                self.active_order_ids.add(oid)
            return

        # Skew
        skew = (pos_size * mid / self.max_position_usd) * half_spread * 0.5 if self.max_position_usd > 0 else 0
        our_bid -= skew
        our_ask -= skew

        # Ордера
        if our_bid < bid:
            oid = engine.place_order("limit", our_bid, size_in_coins)
            self.active_order_ids.add(oid)

        if our_ask > ask:
            oid = engine.place_order("limit", our_ask, -size_in_coins)
            self.active_order_ids.add(oid)

    def get_equity(self, engine) -> float:
        return self.initial_balance + engine.get_net_pnl() + engine.get_unrealized_pnl()

    def get_return_pct(self, engine) -> float:
        return (self.get_equity(engine) - self.initial_balance) / self.initial_balance * 100

    def get_total_fees(self, engine) -> float:
        return engine.get_total_fees()