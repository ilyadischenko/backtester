from dataclasses import dataclass, field
from typing import List, Optional, Dict
import random
import math
import polars as pl
from sortedcontainers import SortedDict


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
    fill_time: int = 0
    status: str = "pending"
    cancel_requested_at: Optional[int] = None
    is_maker: bool = False          # FIX: сохраняем для корректного расчёта комиссий


@dataclass(slots=True)
class Position:
    id: int
    open_time: int
    size: float = 0.0
    price: float = 0.0
    close_time: Optional[int] = None
    status: str = "open"

    realized_pnl: float = 0.0
    fees: float = 0.0
    net_pnl: float = 0.0

    order_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        if self.order_ids is None:
            self.order_ids = []


class OrderBook:
    __slots__ = ("bids", "asks", "best_bid", "best_ask")

    def __init__(self):
        self.bids: SortedDict = SortedDict()
        self.asks: SortedDict = SortedDict()
        self.best_bid: float = 0.0
        self.best_ask: float = float("inf")

    def apply_snapshot(self, b_p, b_q, a_p, a_q):
        self.bids = SortedDict({p: q for p, q in zip(b_p, b_q) if q > 0})
        self.asks = SortedDict({p: q for p, q in zip(a_p, a_q) if q > 0})
        self._update_best()

    def apply_depth(self, b_p, b_q, a_p, a_q):
        for p, q in zip(b_p, b_q):
            if q == 0: self.bids.pop(p, None)
            else:      self.bids[p] = q

        for p, q in zip(a_p, a_q):
            if q == 0: self.asks.pop(p, None)
            else:      self.asks[p] = q

        self._update_best()

    def _update_best(self):
        self.best_bid = self.bids.keys()[-1] if self.bids else 0.0
        self.best_ask = self.asks.keys()[0]  if self.asks else float("inf")

    @property
    def is_ready(self) -> bool:
        return self.best_bid > 0 and self.best_ask < float("inf")


class ExchangeEngine:
    EPSILON = 1e-12

    def __init__(
        self,
        data_trades: pl.DataFrame = None,
        data_depth: pl.DataFrame = None,
        data_ob_snapshot: pl.DataFrame = None,
        strategy=None,
        taker_fee: float = 0.0004,
        maker_fee: float = 0.0002,
        network_delay: int = 2,
    ):
        self.strategy = strategy
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.network_delay = network_delay

        self.ob = OrderBook()

        self.requests: List[dict] = []
        self.orders: List[Order] = []
        self.positions: List[Position] = []
        self.active_orders: Dict[int, Order] = {}
        self.orders_by_id: Dict[int, Order] = {}

        self.last_event_time: int = 0
        self.last_trade: dict = {}

        self.events = self._build_events(data_trades, data_depth, data_ob_snapshot)

    def _build_events(
        self,
        data_trades: pl.DataFrame,
        data_depth: pl.DataFrame,
        data_ob_snapshot: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Собирает единый поток событий из трёх источников.
        Колонки на входе:
          trades:      E, t, p, q
          depth:       E, U, u, b_p, b_q, a_p, a_q
          ob_snapshot: ts, lastUpdateId, b_p, b_q, a_p, a_q

        При одинаковом timestamp ob_snapshot идёт раньше depth (sort_key=0),
        depth раньше trades (sort_key=1), trades последними (sort_key=2).
        """
        frames = []

        if data_ob_snapshot is not None and len(data_ob_snapshot) > 0:
            frames.append(
                data_ob_snapshot
                .rename({"ts": "event_time"})
                .with_columns([
                    pl.lit("ob_snapshot").alias("event_type"),
                    pl.lit(0).alias("_sort_key"),
                ])
            )

        if data_depth is not None and len(data_depth) > 0:
            frames.append(
                data_depth
                .rename({"E": "event_time"})
                .with_columns([
                    pl.lit("depth").alias("event_type"),
                    pl.lit(1).alias("_sort_key"),
                ])
            )

        if data_trades is not None and len(data_trades) > 0:
            frames.append(
                data_trades
                .rename({"E": "event_time"})
                .with_columns([
                    pl.lit("trade").alias("event_type"),
                    pl.lit(2).alias("_sort_key"),
                ])
            )

        if not frames:
            return pl.DataFrame()

        return (
            pl.concat(frames, how="diagonal_relaxed")
            .sort(["event_time", "_sort_key"])
        )

    # ═══════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════

    def run(self):
        total = len(self.events)
        for i, event in enumerate(self.events.iter_rows(named=True)):
            if i % 10_000 == 0:
                pct = i / total * 100
                print(f"\r⏳ {pct:5.1f}% | {i:,} / {total:,} events", end="", flush=True)

            self.last_event_time = event["event_time"]
            etype = event["event_type"]

            if etype == "ob_snapshot":
                self.ob.apply_snapshot(
                    event["b_p"], event["b_q"],
                    event["a_p"], event["a_q"],
                )

            elif etype == "depth":
                self.ob.apply_depth(
                    event["b_p"], event["b_q"],
                    event["a_p"], event["a_q"],
                )

            elif etype == "trade":
                self.last_trade = event

            self._process_requests()

            if self.ob.is_ready:
                self._process_orders()

            if self.strategy:
                self.strategy.on_tick(event, self)

        print(f"\r✅ 100.0% | {total:,} / {total:,} events")

    # ═══════════════════════════════════════════════════════════
    # ORDER MANAGEMENT
    # ═══════════════════════════════════════════════════════════

    def place_order(self, order_type: str, price: float, size: float) -> int:
        """
        Выставить ордер.

        order_type:
          "market" — исполняется немедленно по best ask/bid
          "limit"  — исполняется когда цена достигает уровня
          "stop"   — триггерится по цене последней сделки,
                     исполняется маркетом по best ask/bid.
                     size > 0 → стоп на покупку (триггер: trade_price >= price)
                     size < 0 → стоп на продажу (триггер: trade_price <= price)
        """
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
                order = Order(
                    id=req["id"],
                    price=req["price"],
                    size=req["size"],
                    type=req["type"],
                    created_time=req["send_time"],
                    exchange_time=arrival,
                    status="new",
                )
                self.orders.append(order)
                self.orders_by_id[order.id] = order
                self.active_orders[order.id] = order

            elif req["action"] == "cancel":
                order = self.orders_by_id.get(req["id"])
                if order is not None and order.cancel_requested_at is None:
                    order.cancel_requested_at = arrival

    # ═══════════════════════════════════════════════════════════
    # ORDER EXECUTION
    # ═══════════════════════════════════════════════════════════

    def _process_orders(self):
        bid = self.ob.best_bid
        ask = self.ob.best_ask

        for order in list(self.active_orders.values()):
            if order.exchange_time > self.last_event_time:
                continue

            if (order.cancel_requested_at is not None
                    and order.cancel_requested_at <= self.last_event_time):
                order.status = "canceled"
                self.active_orders.pop(order.id, None)
                continue

            just_placed = order.exchange_time == self.last_event_time

            if order.type == "market":
                exec_price = ask if order.size > 0 else bid
                self._fill_order(order, exec_price, is_maker=False)
                self.active_orders.pop(order.id, None)

            elif order.type == "limit":
                filled = self._try_fill_limit_order(order, bid, ask, just_placed)
                if filled:
                    self.active_orders.pop(order.id, None)

            elif order.type == "stop":
                triggered = self._try_trigger_stop_order(order, bid, ask, just_placed)
                if triggered:
                    self.active_orders.pop(order.id, None)

    def _try_fill_limit_order(
        self, order: Order, bid: float, ask: float, just_placed: bool
    ) -> bool:
        is_buy = order.size > 0

        if is_buy:
            if just_placed and order.price >= ask:
                self._fill_order(order, ask, is_maker=False)
                return True
            if self._trade_hit_level(order):
                self._fill_order(order, order.price, is_maker=True)
                return True
            if ask <= order.price:
                self._fill_order(order, order.price, is_maker=True)
                return True
        else:
            if just_placed and order.price <= bid:
                self._fill_order(order, bid, is_maker=False)
                return True
            if self._trade_hit_level(order):
                self._fill_order(order, order.price, is_maker=True)
                return True
            if bid >= order.price:
                self._fill_order(order, order.price, is_maker=True)
                return True

        return False

    def _try_trigger_stop_order(
        self, order: Order, bid: float, ask: float, just_placed: bool
    ) -> bool:
        """
        Стоп-ордер триггерится по цене последней сделки:
          size > 0 (buy stop):  триггер когда trade_price >= stop_price
          size < 0 (sell stop): триггер когда trade_price <= stop_price

        После триггера исполняется маркетом по текущему best ask/bid.
        На тике размещения (just_placed) не триггерим — даём осесть.
        """
        if just_placed:
            return False

        if not self.last_trade or self.last_trade.get("event_type") != "trade":
            return False

        trade_price = self.last_trade.get("p", 0.0)
        stop_price  = order.price

        if order.size > 0:   # buy stop — цена пробила уровень снизу вверх
            if trade_price < stop_price:
                return False
            exec_price = ask  # маркет покупки — по аску
        else:                 # sell stop — цена пробила уровень сверху вниз
            if trade_price > stop_price:
                return False
            exec_price = bid  # маркет продажи — по биду

        self._fill_order(order, exec_price, is_maker=False)
        return True

    def _trade_hit_level(self, order: Order) -> bool:
        """
        q > 0 — покупатель агрессор (buy taker), цена идёт вверх.
        q < 0 — продавец агрессор (sell taker), цена идёт вниз.

        Buy limit исполняется когда sell taker (q < 0) бьёт по нашей цене или ниже.
        Sell limit исполняется когда buy taker (q > 0) бьёт по нашей цене или выше.
        """
        if not self.last_trade:
            return False
        if self.last_trade.get("event_type") != "trade":
            return False

        trade_price = self.last_trade.get("p", 0.0)
        trade_qty   = self.last_trade.get("q", 0.0)

        if order.size > 0:  # Buy limit
            return trade_qty < 0 and trade_price <= order.price
        else:               # Sell limit
            return trade_qty > 0 and trade_price >= order.price

    def _fill_order(self, order: Order, price: float, is_maker: bool):
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        order.filled = order.size
        order.fill_price = price
        order.fill_time = self.last_event_time
        order.status = "filled"
        order.is_maker = is_maker       # FIX: сохраняем для trades_to_dataframe / validate
        self._update_position(order.size, price, fee_rate, order.id)

    # ═══════════════════════════════════════════════════════════
    # POSITION LOGIC
    # ═══════════════════════════════════════════════════════════

    def _update_position(self, trade_size: float, price: float, fee_rate: float, order_id: int):
        pos = self._get_or_create_position()
        pos.order_ids.append(order_id)

        trade_abs = abs(trade_size)
        fee = trade_abs * price * fee_rate
        pos_abs = abs(pos.size)

        if self._is_zero(pos.size):
            pos.size = trade_size
            pos.price = price
            pos.fees += fee
            return

        if pos.size * trade_size > 0:
            old_notional = pos_abs * pos.price
            new_notional = trade_abs * price
            total = pos_abs + trade_abs
            pos.price = (old_notional + new_notional) / total
            pos.size += trade_size
            pos.fees += fee
            return

        closed_qty = min(pos_abs, trade_abs)

        if pos.size > 0:
            pnl = (price - pos.price) * closed_qty
        else:
            pnl = (pos.price - price) * closed_qty

        pos.realized_pnl += pnl

        if trade_abs < pos_abs - self.EPSILON:
            pos.size += trade_size
            pos.fees += fee
            return

        if self._is_close(trade_abs, pos_abs):
            pos.fees += fee
            self._close_position(pos)
            return

        remaining = trade_abs - pos_abs
        close_fee = fee * (pos_abs / trade_abs)
        open_fee  = fee * (remaining / trade_abs)

        pos.fees += close_fee
        self._close_position(pos)

        new_pos = self._create_position()
        new_pos.size  = math.copysign(remaining, trade_size)
        new_pos.price = price
        new_pos.order_ids.append(order_id)
        new_pos.fees  = open_fee

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

    def get_position(self) -> Optional[Position]:
        for p in self.positions:
            if p.status == "open":
                return p
        return None

    def get_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions)

    def get_total_fees(self) -> float:
        return sum(p.fees for p in self.positions)

    def get_net_pnl(self) -> float:
        return sum(p.net_pnl for p in self.positions if p.status == "closed")

    def get_unrealized_pnl(self) -> float:
        pos = self.get_position()
        if pos is None or not self.ob.is_ready:
            return 0.0
        if pos.size > 0:
            return (self.ob.best_bid - pos.price) * pos.size
        else:
            return (pos.price - self.ob.best_ask) * abs(pos.size)

    def get_unrealized_net_pnl(self) -> float:
        pos = self.get_position()
        if pos is None:
            return 0.0
        return self.get_unrealized_pnl() - pos.fees

    def get_position_summary(self, position_id: int) -> dict:
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

    # ═══════════════════════════════════════════════════════════
    # REPORTING
    # ═══════════════════════════════════════════════════════════

    def get_all_positions_summary(self) -> List[dict]:
        result = []
        for pos in self.positions:
            orders = [self.orders_by_id.get(oid) for oid in pos.order_ids]
            orders = [o for o in orders if o is not None]
            result.append({
                "id": pos.id,
                "status": pos.status,
                "open_time": pos.open_time,
                "close_time": pos.close_time,
                "entry_price": pos.price,
                "size": pos.size,
                "realized_pnl": pos.realized_pnl,
                "fees": pos.fees,
                "net_pnl": pos.net_pnl,
                "orders": [
                    {
                        "id": o.id,
                        "type": o.type,
                        "side": "BUY" if o.size > 0 else "SELL",
                        "size": o.size,
                        "order_price": o.price,
                        "fill_price": o.fill_price,
                        "filled": o.filled,
                        "status": o.status,
                        "created_time": o.created_time,
                        "fill_time": o.fill_time,
                    }
                    for o in orders
                ],
            })
        return result

    def print_positions_report(self, detailed: bool = True):
        positions = self.get_all_positions_summary()

        print("\n" + "═" * 80)
        print("                         POSITIONS REPORT")
        print("═" * 80)

        total_realized = 0.0
        total_fees = 0.0
        total_net = 0.0

        for i, pos in enumerate(positions, 1):
            status_emoji = "✅" if pos["status"] == "closed" else "🔄"
            pnl_emoji = "📈" if pos["net_pnl"] >= 0 else "📉"

            print(f"\n{'─' * 80}")
            print(f"Position #{i} | ID: {pos['id']} | {status_emoji} {pos['status'].upper()}")
            print(f"{'─' * 80}")
            print(f"  Open Time:    {pos['open_time']}")
            if pos["close_time"]:
                print(f"  Close Time:   {pos['close_time']}")
                duration = pos["close_time"] - pos["open_time"]
                print(f"  Duration:     {duration} ms ({duration/1000:.1f} sec)")
            print(f"  Entry Price:  {pos['entry_price']:.6f}")
            print(f"  Size:         {pos['size']:.4f}")
            print(f"\n  {pnl_emoji} P&L Breakdown:")
            print(f"     Realized PnL:  {pos['realized_pnl']:+.6f}")
            print(f"     Fees:          {pos['fees']:.6f}")
            print(f"     Net PnL:       {pos['net_pnl']:+.6f}")

            if detailed and pos["orders"]:
                print(f"\n  📋 Orders ({len(pos['orders'])}):")
                print(f"  {'─' * 70}")
                print(f"  {'ID':<12} {'Side':<6} {'Type':<8} {'Size':<12} {'Order Px':<12} {'Fill Px':<12} {'Status':<10}")
                print(f"  {'─' * 70}")
                for o in pos["orders"]:
                    print(
                        f"  {o['id']:<12} "
                        f"{o['side']:<6} "
                        f"{o['type']:<8} "
                        f"{o['size']:<12.4f} "
                        f"{o['order_price']:<12.6f} "
                        f"{o['fill_price']:<12.6f} "
                        f"{o['status']:<10}"
                    )

            if pos["status"] == "closed":
                total_realized += pos["realized_pnl"]
                total_fees += pos["fees"]
                total_net += pos["net_pnl"]

        print(f"\n{'═' * 80}")
        print("                            SUMMARY")
        print(f"{'═' * 80}")
        print(f"  Total Positions:    {len(positions)}")
        print(f"  Closed:             {sum(1 for p in positions if p['status'] == 'closed')}")
        print(f"  Open:               {sum(1 for p in positions if p['status'] == 'open')}")
        print(f"\n  Total Realized PnL: {total_realized:+.6f}")
        print(f"  Total Fees:         {total_fees:.6f}")
        print(f"  Total Net PnL:      {total_net:+.6f}")

        open_pos = self.get_position()
        if open_pos:
            unrealized = self.get_unrealized_pnl()
            print(f"\n  ⚠️  Open Position:")
            print(f"      Size: {open_pos.size:.4f} @ {open_pos.price:.6f}")
            print(f"      Unrealized PnL: {unrealized:+.6f}")
            print(f"      Accumulated Fees: {open_pos.fees:.6f}")

        print(f"{'═' * 80}\n")

    def positions_to_dataframe(self) -> pl.DataFrame:
        rows = []
        for pos in self.positions:
            rows.append({
                "position_id": pos.id,
                "status": pos.status,
                "open_time": pos.open_time,
                "close_time": pos.close_time,
                "entry_price": pos.price,
                "size": pos.size,
                "realized_pnl": pos.realized_pnl,
                "fees": pos.fees,
                "net_pnl": pos.net_pnl,
                "order_count": len(pos.order_ids),
            })
        return pl.DataFrame(rows)

    def orders_to_dataframe(self) -> pl.DataFrame:
        order_to_position = {}
        for pos in self.positions:
            for oid in pos.order_ids:
                order_to_position[oid] = pos.id

        rows = []
        for o in self.orders:
            rows.append({
                "order_id": o.id,
                "position_id": order_to_position.get(o.id),
                "type": o.type,
                "side": "BUY" if o.size > 0 else "SELL",
                "size": o.size,
                "order_price": o.price,
                "fill_price": o.fill_price,
                "filled": o.filled,
                "status": o.status,
                "created_time": o.created_time,
                "exchange_time": o.exchange_time,
                "fill_time": o.fill_time,
                "is_maker": o.is_maker,
            })
        return pl.DataFrame(rows)

    def trades_to_dataframe(self) -> pl.DataFrame:
        order_to_position = {}
        for pos in self.positions:
            for oid in pos.order_ids:
                order_to_position[oid] = pos

        rows = []
        for o in self.orders:
            if o.status != "filled":
                continue
            pos = order_to_position.get(o.id)
            fee_rate = self.maker_fee if o.is_maker else self.taker_fee  # FIX: используем сохранённый флаг
            notional = abs(o.size) * o.fill_price
            fee = notional * fee_rate
            rows.append({
                "order_id": o.id,
                "position_id": pos.id if pos else None,
                "time": o.fill_time,
                "side": "BUY" if o.size > 0 else "SELL",
                "size": abs(o.size),
                "price": o.fill_price,
                "notional": notional,
                "fee_rate": fee_rate,
                "fee": fee,
                "is_maker": o.is_maker,
            })
        return pl.DataFrame(rows)

    def validate_pnl(self) -> dict:
        results = {"positions": [], "totals": {}, "errors": []}

        for pos in self.positions:
            if pos.status != "closed":
                continue

            orders = [self.orders_by_id.get(oid) for oid in pos.order_ids]
            orders = [o for o in orders if o and o.status == "filled"]

            running_size = 0.0
            running_cost = 0.0
            calculated_pnl = 0.0
            calculated_fees = 0.0
            trades_detail = []

            for o in orders:
                trade_size = o.size
                trade_price = o.fill_price
                trade_abs = abs(trade_size)
                fee_rate = self.maker_fee if o.is_maker else self.taker_fee  # FIX: точный fee_rate
                fee = trade_abs * trade_price * fee_rate
                calculated_fees += fee

                if running_size == 0:
                    running_size = trade_size
                    running_cost = trade_abs * trade_price
                    action = "OPEN"
                    pnl_change = 0.0
                elif running_size * trade_size > 0:
                    running_size += trade_size
                    running_cost += trade_abs * trade_price
                    action = "ADD"
                    pnl_change = 0.0
                else:
                    avg_price = running_cost / abs(running_size) if running_size != 0 else 0
                    closed_qty = min(abs(running_size), trade_abs)
                    if running_size > 0:
                        pnl_change = (trade_price - avg_price) * closed_qty
                    else:
                        pnl_change = (avg_price - trade_price) * closed_qty
                    calculated_pnl += pnl_change
                    if trade_abs >= abs(running_size):
                        action = "CLOSE"
                        remaining = trade_abs - abs(running_size)
                        running_size = 0 if remaining < 1e-9 else math.copysign(remaining, trade_size)
                        running_cost = remaining * trade_price if remaining > 0 else 0
                    else:
                        action = "PARTIAL_CLOSE"
                        ratio = closed_qty / abs(running_size)
                        running_cost *= (1 - ratio)
                        running_size += trade_size

                trades_detail.append({
                    "order_id": o.id,
                    "action": action,
                    "size": trade_size,
                    "price": trade_price,
                    "fee": fee,
                    "pnl_change": pnl_change,
                    "running_size": running_size,
                })

            calculated_net = calculated_pnl - calculated_fees
            pnl_match  = abs(calculated_pnl  - pos.realized_pnl) < 1e-6
            fees_match = abs(calculated_fees - pos.fees)         < 1e-6
            net_match  = abs(calculated_net  - pos.net_pnl)      < 1e-6

            pos_result = {
                "position_id": pos.id,
                "stored":     {"realized_pnl": pos.realized_pnl, "fees": pos.fees, "net_pnl": pos.net_pnl},
                "calculated": {"realized_pnl": calculated_pnl,   "fees": calculated_fees, "net_pnl": calculated_net},
                "match":      {"pnl": pnl_match, "fees": fees_match, "net": net_match},
                "trades":     trades_detail,
            }
            results["positions"].append(pos_result)

            if not (pnl_match and fees_match and net_match):
                results["errors"].append({
                    "position_id": pos.id,
                    "pnl_diff":  calculated_pnl  - pos.realized_pnl,
                    "fees_diff": calculated_fees - pos.fees,
                    "net_diff":  calculated_net  - pos.net_pnl,
                })

        results["totals"] = {
            "positions_checked": len(results["positions"]),
            "errors_found":      len(results["errors"]),
            "all_valid":         len(results["errors"]) == 0,
        }
        return results

    def print_validation_report(self):
        validation = self.validate_pnl()

        print("\n" + "═" * 80)
        print("                      P&L VALIDATION REPORT")
        print("═" * 80)

        for pos in validation["positions"]:
            status = "✅" if all(pos["match"].values()) else "❌"
            print(f"\nPosition {pos['position_id']}: {status}")
            print(f"  {'Metric':<15} {'Stored':<15} {'Calculated':<15} {'Match':<10}")
            print(f"  {'-'*55}")
            for key in ["realized_pnl", "fees", "net_pnl"]:
                stored = pos["stored"][key]
                calc   = pos["calculated"][key]
                mk = "pnl" if "realized" in key else ("fees" if "fees" in key else "net")
                match  = "✅" if pos["match"].get(mk, False) else "❌"
                print(f"  {key:<15} {stored:<15.6f} {calc:<15.6f} {match}")

        print(f"\n{'═' * 80}")
        print(f"Summary: {validation['totals']['positions_checked']} positions checked")
        if validation["totals"]["all_valid"]:
            print("✅ All calculations are correct!")
        else:
            print(f"❌ Found {validation['totals']['errors_found']} errors:")
            for err in validation["errors"]:
                print(f"   Position {err['position_id']}: "
                      f"PnL diff={err['pnl_diff']:.6f}, "
                      f"Fees diff={err['fees_diff']:.6f}")
        print(f"{'═' * 80}\n")