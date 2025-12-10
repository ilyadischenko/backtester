# strategies/spread_capture.py
from typing import Set


class SpreadCaptureStrategy:
    """
    Самая консервативная: просто ловим спред.
    - Ставим лимитку на покупку чуть ниже bid
    - Ставим лимитку на продажу чуть выше ask
    - Если обе исполнились — заработали спред
    - Строгий контроль позиции
    """

    def __init__(
            self,
            initial_balance: float = 10000.0,
            order_size_usd: float = 50.0,  # Маленький размер
            offset_bps: float = 1.0,  # Отступ от лучшей цены
            max_position_usd: float = 100.0,  # Очень маленькая позиция
            refresh_interval_ms: int = 500,
    ):
        self.initial_balance = initial_balance
        self.order_size_usd = order_size_usd
        self.offset_bps = offset_bps
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

        bid = event["bid_price"]
        ask = event["ask_price"]
        mid = (bid + ask) / 2
        spread = ask - bid
        spread_bps = spread / mid * 10000

        # Не торгуем если спред слишком маленький (не покроет комиссию)
        min_spread_bps = 2.0  # 0.02%
        if spread_bps < min_spread_bps:
            return

        offset = mid * (self.offset_bps / 10000)
        our_bid = bid - offset
        our_ask = ask + offset

        size_in_coins = self.order_size_usd / mid

        pos_size = engine.get_position_size()
        pos_value = abs(pos_size * mid)

        # ═══════════════════════════════════════════════════════
        # Контроль позиции — главный приоритет
        # ═══════════════════════════════════════════════════════

        if pos_value > self.max_position_usd:
            # Только закрываем
            if pos_size > 0:
                oid = engine.place_order("limit", our_ask, -size_in_coins)
                self.active_order_ids.add(oid)
            else:
                oid = engine.place_order("limit", our_bid, size_in_coins)
                self.active_order_ids.add(oid)
            return

        # ═══════════════════════════════════════════════════════
        # Нормальный режим — котируем обе стороны
        # ═══════════════════════════════════════════════════════

        # Skew в зависимости от позиции
        skew = 0.0
        if self.max_position_usd > 0 and pos_size != 0:
            # Если лонг — хотим продать → сдвигаем ask ближе
            # Если шорт — хотим купить → сдвигаем bid ближе
            skew_factor = pos_value / self.max_position_usd
            skew = offset * skew_factor * 0.5

            if pos_size > 0:
                our_ask -= skew  # Делаем ask привлекательнее
            else:
                our_bid += skew  # Делаем bid привлекательнее

        # Ставим ордера только если они пассивные
        if our_bid < bid:
            oid = engine.place_order("limit", our_bid, size_in_coins)
            self.active_order_ids.add(oid)

        if our_ask > ask:
            oid = engine.place_order("limit", our_ask, -size_in_coins)
            self.active_order_ids.add(oid)