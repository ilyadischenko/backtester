from typing import Optional
from dataclasses import dataclass

from visualization.plot_recorder import PlotRecorder


@dataclass
class LargeOrder:
    price: float
    side: int
    qty_initial: float
    qty_current: float
    seen_at: int


_OB_SCAN_DEPTH = 50


class FrontrunStrategy:
    """
    Фронтраннинг крупных заявок с OCO (тейк + стоп).

    Логика:
      1. Сканируем стакан, ищем крупную заявку (>= large_qty_threshold)
         в диапазоне min/max_wall_distance_pct от mid.
      2. Ждём wall_survive_ms мс — убеждаемся что стена не фейковая.
      3. Встаём лимиткой на tick_size перед стеной:
           bid-стена → покупаем на wall.price + tick_size
           ask-стена → продаём  на wall.price - tick_size
      4. После входа выставляем OCO:
           Тейк = лимит на take_pct % от цены входа
           Стоп = стоп-ордер на уровне wall.price
             лонг → стоп продажа на wall.price (триггер: trade <= wall.price)
             шорт → стоп покупка на wall.price (триггер: trade >= wall.price)
      5. Что сработало первым (тейк или стоп) — второй отменяем.
      6. Если стена исчезла до закрытия позиции — выходим маркетом,
         отменяем оба ордера OCO.
    """

    def __init__(
        self,
        large_qty_threshold: float = 10.0,
        tick_size: float = 0.1,
        order_size: float = 1.0,
        take_pct: float = 1.0,
        min_wall_distance_pct: float = 0.05,
        max_wall_distance_pct: float = 2.0,
        wall_survive_ms: int = 500,
        disappear_pct: float = 50.0,
        recalc_ms: int = 100,
    ):
        self.initial_balance = 1000.0

        self.large_qty_threshold   = large_qty_threshold
        self.tick_size             = tick_size
        self.order_size            = order_size
        self.take_pct              = take_pct
        self.min_wall_distance_pct = min_wall_distance_pct
        self.max_wall_distance_pct = max_wall_distance_pct
        self.wall_survive_ms       = wall_survive_ms
        self.disappear_pct         = disappear_pct
        self.recalc_ms             = recalc_ms

        self.mid_price = 0.0
        self.wall: Optional[LargeOrder] = None

        self.entry_order_id: Optional[int] = None
        self.entry_price: float = 0.0

        self.in_position   = False
        self.position_side = 0

        self.take_order_id: Optional[int] = None
        self.take_price: float = 0.0
        self.stop_order_id: Optional[int] = None
        self.stop_price: float = 0.0

        self._oco_placed = False

        self.last_recalc = 0
        self.plot = PlotRecorder()

    # ═══════════════════════════════════════════════════════════
    # ГЛАВНЫЙ ОБРАБОТЧИК
    # ═══════════════════════════════════════════════════════════

    def on_tick(self, event: dict, engine):
        t     = event["event_time"]
        etype = event.get("event_type")

        if etype in ("depth", "ob_snapshot") and engine.ob.is_ready:
            self.mid_price = (engine.ob.best_bid + engine.ob.best_ask) / 2

        if self.mid_price <= 0:
            return
        if t - self.last_recalc < self.recalc_ms:
            return
        self.last_recalc = t

        pos = self._get_position(engine)
        inv = pos.size if pos else 0.0

        if inv != 0.0 and not self.in_position:
            self.in_position = True
            if self.entry_order_id is not None:
                order = self._find_order(engine, self.entry_order_id)
                if order and order.status == "filled":
                    self.entry_price = order.fill_price

        if inv == 0.0 and self.in_position:
            self._reset()

        if self.in_position:
            self._manage_position(engine, inv, t)
        else:
            self._manage_entry(engine, t)

        self._draw(t)

    # ═══════════════════════════════════════════════════════════
    # УПРАВЛЕНИЕ ВХОДОМ
    # ═══════════════════════════════════════════════════════════

    def _manage_entry(self, engine, t: int):
        if self.entry_order_id is not None:
            order = self._find_order(engine, self.entry_order_id)
            if order and order.status == "filled":
                return

        if self.wall is not None:
            self._check_wall(engine, t)
        else:
            wall = self._find_best_wall(engine.ob, t)
            if wall is None:
                return

            self.wall = wall

            color  = "#00BCD4" if wall.side == 1 else "#FF9800"
            marker = "triangle" if wall.side == 1 else "inverted_triangle"
            self.plot.marker(
                f"Wall {'BID' if wall.side == 1 else 'ASK'}",
                wall.price, t, marker=marker, color=color, size=12,
            )

            if self.wall_survive_ms == 0:
                self._place_entry(engine, t)

    def _check_wall(self, engine, t: int):
        wall = self.wall
        ob   = engine.ob

        qty_now = ob.bids.get(wall.price, 0.0) if wall.side == 1 \
                  else ob.asks.get(wall.price, 0.0)

        if self._is_disappeared(wall, qty_now):
            if self.entry_order_id is not None:
                order = self._find_order(engine, self.entry_order_id)
                if order and order.status in ("new", "pending"):
                    engine.cancel_order(self.entry_order_id)
                    self.plot.marker(
                        "Cancel (wall gone)", self.entry_price, t,
                        marker="x", color="#FF5252", size=10,
                    )
            self.wall           = None
            self.entry_order_id = None
            return

        if wall.side == 1 and self.mid_price < wall.price - self.tick_size:
            self.wall = None
            self.entry_order_id = None
            return
        if wall.side == -1 and self.mid_price > wall.price + self.tick_size:
            self.wall = None
            self.entry_order_id = None
            return

        wall.qty_current = qty_now

        if self.entry_order_id is None:
            if t - wall.seen_at >= self.wall_survive_ms:
                self._place_entry(engine, t)

    def _find_best_wall(self, ob, t: int) -> Optional[LargeOrder]:
        mid     = self.mid_price
        min_d   = mid * self.min_wall_distance_pct / 100
        max_d   = mid * self.max_wall_distance_pct / 100
        best: Optional[LargeOrder] = None
        best_qty = 0.0

        bid_keys  = ob.bids.keys()
        bid_slice = bid_keys[max(0, len(bid_keys) - _OB_SCAN_DEPTH):]
        for price in bid_slice:
            qty = ob.bids[price]
            if qty < self.large_qty_threshold:
                continue
            dist = mid - price
            if dist < min_d or dist > max_d:
                continue
            if qty > best_qty:
                best_qty = qty
                best = LargeOrder(price=price, side=1,
                                  qty_initial=qty, qty_current=qty, seen_at=t)

        ask_keys  = ob.asks.keys()
        ask_slice = ask_keys[:_OB_SCAN_DEPTH]
        for price in ask_slice:
            qty = ob.asks[price]
            if qty < self.large_qty_threshold:
                continue
            dist = price - mid
            if dist < min_d or dist > max_d:
                continue
            if qty > best_qty:
                best_qty = qty
                best = LargeOrder(price=price, side=-1,
                                  qty_initial=qty, qty_current=qty, seen_at=t)

        return best

    def _place_entry(self, engine, t: int):
        wall = self.wall
        if wall is None:
            return

        if wall.side == 1:
            entry_price = wall.price + self.tick_size
            size        = self.order_size
        else:
            entry_price = wall.price - self.tick_size
            size        = -self.order_size

        self.entry_order_id = engine.place_order("limit", price=entry_price, size=size)
        self.entry_price    = entry_price  # временно; перезапишется реальным fill в on_tick

        color = "#00E676" if size > 0 else "#FF5252"
        label = "Entry BUY" if size > 0 else "Entry SELL"
        self.plot.marker(label, entry_price, t, marker="star", color=color, size=14)

    # ═══════════════════════════════════════════════════════════
    # УПРАВЛЕНИЕ ПОЗИЦИЕЙ
    # ═══════════════════════════════════════════════════════════

    def _manage_position(self, engine, inv: float, t: int):
        # Выставляем OCO один раз после открытия позиции
        if not self._oco_placed:
            self._place_oco(engine, inv)

        take_filled = self._check_order_filled(engine, self.take_order_id)
        stop_filled = self._check_order_filled(engine, self.stop_order_id)

        if take_filled:
            self._cancel_order_safe(engine, self.stop_order_id)
            self.plot.marker("TAKE ✓", self.take_price, t,
                             marker="star", color="#00E676", size=15)
            self._reset()
            return

        if stop_filled:
            self._cancel_order_safe(engine, self.take_order_id)
            self.plot.marker("STOP ✓", self.stop_price, t,
                             marker="x", color="#FF5252", size=15)
            self._reset()
            return

        # Стена исчезла — выходим маркетом, снимаем оба ордера
        if self.wall is not None:
            ob      = engine.ob
            qty_now = ob.bids.get(self.wall.price, 0.0) if self.wall.side == 1 \
                      else ob.asks.get(self.wall.price, 0.0)

            if self._is_disappeared(self.wall, qty_now):
                self._exit_market(engine, inv, "Wall Gone", t)
                return

        # Линии на графике
        self.plot.line("Take", self.take_price, t,
                       color="#00E676", linewidth=2, linestyle="dotted", alpha=0.9)
        self.plot.line("Stop", self.stop_price, t,
                       color="#FF5252", linewidth=2, linestyle="dotted", alpha=0.9)
        if self.wall:
            wall_color = "#00BCD4" if self.wall.side == 1 else "#FF9800"
            self.plot.line("Wall", self.wall.price, t,
                           color=wall_color, linewidth=1, linestyle="dashed", alpha=0.6)

    def _place_oco(self, engine, inv: float):
        """
        Выставляем тейк + стоп одновременно (OCO).

        Тейк: лимит на take_pct % от цены входа.
        Стоп: стоп-ордер на уровне wall.price.
          лонг (inv > 0) → стоп на продажу (size < 0),
                           триггер: trade_price <= wall.price
          шорт (inv < 0) → стоп на покупку (size > 0),
                           триггер: trade_price >= wall.price
        """
        if inv == 0.0 or self.entry_price == 0.0:
            return
        if self.wall is None:
            return

        self.position_side = 1 if inv > 0 else -1

        # Тейк
        if self.position_side == 1:
            self.take_price = self.entry_price * (1 + self.take_pct / 100)
            take_size       = -abs(inv)
        else:
            self.take_price = self.entry_price * (1 - self.take_pct / 100)
            take_size       = abs(inv)

        self.take_order_id = engine.place_order("limit", price=self.take_price, size=take_size)

        # Стоп на уровне стены
        self.stop_price    = self.wall.price
        stop_size          = -abs(inv) if self.position_side == 1 else abs(inv)
        self.stop_order_id = engine.place_order("stop", price=self.stop_price, size=stop_size)

        self._oco_placed = True

        self.plot.marker(
            f"Stop @ {self.stop_price:.4f}", self.stop_price, engine.last_event_time,
            marker="inverted_triangle" if self.position_side == 1 else "triangle",
            color="#FF5252", size=10,
        )

    def _exit_market(self, engine, inv: float, reason: str, t: int):
        """Выход маркетом — отменяем оба ордера OCO."""
        self._cancel_order_safe(engine, self.take_order_id)
        self._cancel_order_safe(engine, self.stop_order_id)

        engine.place_order("market", price=self.mid_price, size=-inv)

        pnl = 0.0
        if self.entry_price > 0 and self.mid_price > 0:
            pnl = (self.mid_price - self.entry_price) / self.entry_price * 100
            if self.position_side == -1:
                pnl = -pnl

        color = "#FF5252" if pnl < 0 else "#00E676"
        self.plot.marker(f"{reason} ({pnl:+.2f}%)", self.mid_price, t,
                         marker="x", color=color, size=14)
        self._reset()

    # ═══════════════════════════════════════════════════════════
    # ВИЗУАЛИЗАЦИЯ
    # ═══════════════════════════════════════════════════════════

    def _draw(self, t: int):
        if self.in_position and self.entry_price > 0:
            self.plot.line("Entry", self.entry_price, t,
                           color="#FF9800", linewidth=1, linestyle="dotted", alpha=0.8)

    # ═══════════════════════════════════════════════════════════
    # ХЕЛПЕРЫ
    # ═══════════════════════════════════════════════════════════

    def _check_order_filled(self, engine, order_id: Optional[int]) -> bool:
        if order_id is None:
            return False
        order = self._find_order(engine, order_id)
        return order is not None and order.status == "filled"

    def _cancel_order_safe(self, engine, order_id: Optional[int]):
        if order_id is None:
            return
        order = self._find_order(engine, order_id)
        if order and order.status in ("new", "pending"):
            engine.cancel_order(order_id)

    def _is_disappeared(self, wall: LargeOrder, qty_now: float) -> bool:
        if qty_now == 0.0:
            return True
        if wall.qty_initial > 0:
            eaten_pct = (1 - qty_now / wall.qty_initial) * 100
            return eaten_pct >= self.disappear_pct
        return False

    def _reset(self):
        self.in_position    = False
        self.position_side  = 0
        self.entry_price    = 0.0
        self.take_order_id  = None
        self.take_price     = 0.0
        self.stop_order_id  = None
        self.stop_price     = 0.0
        self.wall           = None
        self.entry_order_id = None
        self._oco_placed    = False

    def _get_position(self, engine):
        for p in engine.positions:
            if p.status == "open" and p.size != 0:
                return p
        return None

    def _find_order(self, engine, oid: int):
        return engine.orders_by_id.get(oid)