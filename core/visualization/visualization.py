# visualization.py
import bisect
import math
from typing import Dict, List, Tuple

from bokeh.plotting import figure, show, output_file
from bokeh.models import (
    ColumnDataSource, HoverTool, Span, LabelSet, CustomJS,
    Button, Div, CrosshairTool, GlyphRenderer,
)
from bokeh.layouts import column, row
from bokeh.events import Tap
import pandas as pd
import polars as pl

from visualization.plot_recorder import PlotRecorder, PlotSeries   # ← fix: was cyrillic «с»
from engine import ExchangeEngine


class BacktestVisualizer:

    def __init__(self, engine: ExchangeEngine, strategy=None):
        self.engine = engine
        self.strategy = strategy
        self._price_cache: Dict[int, Tuple[float, float]] = {}
        self._sorted_cache_keys: List[int] = []

    # ═══════════════════════════════════════════════════════════
    # ГЛАВНЫЙ МЕТОД
    # ═══════════════════════════════════════════════════════════

    def show(self, title="Backtest Result", output_path="backtest_chart.html"):
        price_df = self._build_price_df()

        if price_df.empty:
            print("⚠️  Нет depth данных для отрисовки графика цены")
            return

        if len(price_df) > 50_000:
            price_df = self._downsample_minmax(price_df, 50_000)

        source = ColumnDataSource(price_df)

        # ── 1. Основной график цены ────────────────────────────
        p = figure(
            title=title,
            x_axis_type="datetime",
            height=600,
            sizing_mode="stretch_width",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.add_tools(CrosshairTool())
        self._apply_dark_theme(p)

        p.line(x="time", y="bid_price", source=source,
               legend_label="Bid", line_width=2, color="#006132", alpha=0.9)
        p.line(x="time", y="ask_price", source=source,
               legend_label="Ask", line_width=2, color="#6b1f1f", alpha=0.9)

        hover_price = HoverTool(tooltips=[
            ("Time",   "@time{%F %T}"),
            ("Bid",    "@bid_price{0.0000}"),
            ("Ask",    "@ask_price{0.0000}"),
            ("Spread", "@spread{0.0000}"),
        ], formatters={"@time": "datetime"})
        p.add_tools(hover_price)

        # ── 2. Все выставленные ордера ─────────────────────────
        orders_renderers = self._draw_all_orders(p)

        # ── 3. Исполненные сделки ──────────────────────────────
        buys, sells = self._get_trades()
        if not buys.empty:
            p.scatter(x="time", y="price", source=ColumnDataSource(buys),
                      marker="triangle", size=10, color="#00ff00",
                      alpha=0.8, legend_label="Buy Fill")
        if not sells.empty:
            p.scatter(x="time", y="price", source=ColumnDataSource(sells),
                      marker="inverted_triangle", size=10, color="#ff0000",
                      alpha=0.8, legend_label="Sell Fill")

        # ── 4. Позиции ─────────────────────────────────────────
        self._draw_positions(p)

        # ── 5. Индикаторы стратегии ────────────────────────────
        strategy_renderers = self._draw_strategy_indicators(p)

        p.legend.location = "top_left"
        p.legend.background_fill_alpha = 0.5
        p.legend.label_text_color = "white"
        p.legend.click_policy = "hide"

        # ── 6. Панель управления ───────────────────────────────
        ruler_btn, clear_btn, ruler_info = self._setup_ruler_components(p)
        toggle_orders_btn   = self._create_orders_toggle_button(orders_renderers)
        toggle_strategy_btn = self._create_strategy_toggle_button(strategy_renderers)

        sep = Div(text="|", styles={
            "color": "#555", "font-size": "20px",
            "margin": "0 10px", "margin-top": "-2px",
        })
        controls = row(
            ruler_btn, clear_btn, sep,
            toggle_orders_btn, toggle_strategy_btn, ruler_info,
            sizing_mode="stretch_width",
            styles={"background-color": "#1e1e1e", "padding": "10px",
                    "border-bottom": "1px solid #333"},
        )

        # ── 7. График Equity ───────────────────────────────────
        equity_df = self._build_equity_curve()
        initial_balance = getattr(
            getattr(self.engine, "strategy", None), "initial_balance", 0
        )

        p_equity = figure(
            title="Equity",
            x_axis_type="datetime",
            height=300,
            sizing_mode="stretch_width",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            x_range=p.x_range,
        )
        p_equity.add_tools(CrosshairTool())
        self._apply_dark_theme(p_equity)

        if not equity_df.empty:
            p_equity.line(
                x="time", y="equity",
                source=ColumnDataSource(equity_df),
                line_width=2, color="#2196F3", legend_label="Equity",
            )
            if initial_balance > 0:
                p_equity.add_layout(Span(
                    location=initial_balance, dimension="width",
                    line_color="#888888", line_width=1, line_dash="dashed",
                ))
            p_equity.add_tools(HoverTool(
                tooltips=[("Time", "@time{%F %T}"),
                          ("Equity", "$@equity{0,0.00}")],
                formatters={"@time": "datetime"},
            ))
            p_equity.legend.location = "top_left"
            p_equity.legend.background_fill_alpha = 0.5
            p_equity.legend.label_text_color = "white"

        self._print_stats()
        output_file(output_path)
        show(column(controls, p, p_equity, sizing_mode="stretch_width"))

    # ═══════════════════════════════════════════════════════════
    # ПОСТРОЕНИЕ ЦЕНОВОГО DataFrame  (HOT PATH)
    # ═══════════════════════════════════════════════════════════

    def _build_price_df(self) -> pd.DataFrame:
        """
        Фильтруем события в Polars (SIMD), затем извлекаем колонки
        как Python-списки и обходим в одном цикле.
        best bid/ask обновляются инкрементально — O(1) в среднем
        вместо O(levels) на каждый depth-апдейт.
        """
        events = self.engine.events

        # ── быстрая фильтрация в Polars ────────────────────────
        depth_events = events.filter(
            pl.col("event_type").is_in(["ob_snapshot", "depth"])
        )
        if depth_events.height == 0:
            return pd.DataFrame()

        # ── колонки → Python-списки (без dict-оверхеда) ────────
        col_etype = depth_events["event_type"].to_list()
        col_time  = depth_events["event_time"].to_list()
        col_bp    = depth_events["b_p"].to_list()
        col_bq    = depth_events["b_q"].to_list()
        col_ap    = depth_events["a_p"].to_list()
        col_aq    = depth_events["a_q"].to_list()

        n = len(col_etype)

        out_times:   List[int]   = []
        out_bids:    List[float] = []
        out_asks:    List[float] = []
        out_spreads: List[float] = []

        bids: Dict[float, float] = {}
        asks: Dict[float, float] = {}
        best_bid = 0.0
        best_ask = float("inf")

        price_cache = self._price_cache          # локальная ссылка — быстрее

        for i in range(n):
            etype = col_etype[i]
            t     = col_time[i]
            b_p   = col_bp[i] or []
            b_q   = col_bq[i] or []
            a_p   = col_ap[i] or []
            a_q   = col_aq[i] or []

            if etype == "ob_snapshot":
                bids = {p: q for p, q in zip(b_p, b_q) if q > 0}
                asks = {p: q for p, q in zip(a_p, a_q) if q > 0}
                best_bid = max(bids) if bids else 0.0
                best_ask = min(asks) if asks else float("inf")
                continue                         # снапшот — точку не пишем

            # etype == "depth"
            bid_dirty = False
            for p, q in zip(b_p, b_q):
                if q == 0:
                    if bids.pop(p, None) is not None and p >= best_bid:
                        bid_dirty = True
                else:
                    bids[p] = q
                    if p > best_bid:
                        best_bid = p
            if bid_dirty:
                best_bid = max(bids) if bids else 0.0

            ask_dirty = False
            for p, q in zip(a_p, a_q):
                if q == 0:
                    if asks.pop(p, None) is not None and p <= best_ask:
                        ask_dirty = True
                else:
                    asks[p] = q
                    if p < best_ask:
                        best_ask = p
            if ask_dirty:
                best_ask = min(asks) if asks else float("inf")

            if best_bid > 0 and best_ask < float("inf"):
                spread = best_ask - best_bid
                out_times.append(t)
                out_bids.append(best_bid)
                out_asks.append(best_ask)
                out_spreads.append(spread)
                price_cache[t] = (best_bid, best_ask)

        if not out_times:
            return pd.DataFrame()

        df = pd.DataFrame({
            "event_time": out_times,
            "bid_price":  out_bids,
            "ask_price":  out_asks,
            "spread":     out_spreads,
        })
        df["time"] = pd.to_datetime(df["event_time"], unit="ms")

        # отсортированные ключи — для bisect в _get_price_at
        self._sorted_cache_keys = sorted(price_cache)
        return df

    # ───────────────────────────────────────────────────────────

    def _get_price_at(self, timestamp: int):
        """Ближайший bid/ask ≤ timestamp.  O(log n) через bisect."""
        keys = self._sorted_cache_keys
        if not keys:
            return None, None
        idx = bisect.bisect_right(keys, timestamp) - 1
        if idx < 0:
            return None, None
        return self._price_cache[keys[idx]]

    def _ensure_price_cache(self):
        if not self._sorted_cache_keys:
            self._build_price_df()

    # ═══════════════════════════════════════════════════════════
    # ДАУНСЭМПЛИНГ  (min-max: сохраняет экстремумы)
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _downsample_minmax(df: pd.DataFrame, target_points: int) -> pd.DataFrame:
        n = len(df)
        if n <= target_points:
            return df
        n_buckets   = target_points // 2
        bucket_size = max(1, n // n_buckets)
        bid_arr     = df["bid_price"].values            # numpy — быстрый argmin/argmax

        keep = {0, n - 1}
        for start in range(0, n, bucket_size):
            seg = bid_arr[start : start + bucket_size]
            keep.add(start + int(seg.argmin()))
            keep.add(start + int(seg.argmax()))

        indices = sorted(keep)
        return df.iloc[indices].reset_index(drop=True)

    # ═══════════════════════════════════════════════════════════
    # ОРДЕРА
    # ═══════════════════════════════════════════════════════════

    def _draw_all_orders(self, fig) -> List[GlyphRenderer]:
        limit_orders = [o for o in self.engine.orders if o.type == "limit"]
        buys, sells = [], []
        for o in limit_orders:
            item = {
                "time":   pd.to_datetime(o.created_time, unit="ms"),
                "price":  o.price,
                "size":   o.size,
                "id":     o.id,
                "status": o.status,
            }
            (buys if o.size > 0 else sells).append(item)

        renderers = []
        if buys:
            r = fig.scatter(
                x="time", y="price",
                source=ColumnDataSource(pd.DataFrame(buys)),
                marker="circle", size=4, color="#00e676",
                line_alpha=0.6, fill_alpha=0.3,
                legend_label="Placed Buy (All)", visible=False,
            )
            renderers.append(r)
        if sells:
            r = fig.scatter(
                x="time", y="price",
                source=ColumnDataSource(pd.DataFrame(sells)),
                marker="circle", size=4, color="#ff5252",
                line_alpha=0.6, fill_alpha=0.3,
                legend_label="Placed Sell (All)", visible=False,
            )
            renderers.append(r)
        return renderers

    def _create_orders_toggle_button(self, renderers: List[GlyphRenderer]) -> Button:
        btn = Button(label="Show All Orders", button_type="primary", width=130)
        if not renderers:
            btn.label = "No Orders"
            btn.disabled = True
            return btn
        btn.js_on_click(CustomJS(args=dict(renderers=renderers, btn=btn), code="""
            var new_state = !renderers[0].visible;
            for (var i = 0; i < renderers.length; i++) renderers[i].visible = new_state;
            btn.label = new_state ? "Hide All Orders" : "Show All Orders";
            btn.button_type = new_state ? "success" : "primary";
        """))
        return btn

    def _get_trades(self):
        filled = [o for o in self.engine.orders if o.status == "filled"]
        buys, sells = [], []
        for o in filled:
            item = {
                "time":  pd.to_datetime(o.fill_time, unit="ms"),
                "price": o.fill_price,
                "size":  abs(o.size),
            }
            (buys if o.size > 0 else sells).append(item)
        return pd.DataFrame(buys), pd.DataFrame(sells)

    # ═══════════════════════════════════════════════════════════
    # ПОЗИЦИИ
    # ═══════════════════════════════════════════════════════════

    def _draw_positions(self, fig):
        # индекс filled-ордеров по id — O(1) lookup вместо O(N) на позицию
        filled_by_id = {
            o.id: o for o in self.engine.orders if o.status == "filled"
        }

        segments_all: list = []
        by_pos: Dict[int, list] = {}

        for position in self.engine.positions:
            segs = self._build_position_segments(position, filled_by_id)
            if not segs:
                continue
            for seg in segs:
                color = "#00e676" if seg["side"] == "long" else "#ff9800"
                seg_dict = {
                    "x0": pd.to_datetime(seg["start_time"], unit="ms"),
                    "y0": seg["price"],
                    "x1": pd.to_datetime(seg["end_time"], unit="ms"),
                    "y1": seg["price"],
                    "color": color,
                    "pos_id": position.id,
                    "status": position.status,
                    "side": seg["side"],
                    "seg_size": seg["size"],
                    "avg_price_seg": seg["price"],
                    "entry_price_pos": position.price,
                    "open_time_pos": pd.to_datetime(position.open_time, unit="ms"),
                    "close_time_pos": (
                        pd.to_datetime(position.close_time, unit="ms")
                        if position.close_time else pd.NaT
                    ),
                    "realized_pnl": position.realized_pnl,
                    "fees": position.fees,
                    "net_pnl": position.net_pnl,
                    "is_close": False,
                }
                segments_all.append(seg_dict)
                by_pos.setdefault(position.id, []).append(seg_dict)

        if not segments_all:
            return

        connectors = []
        for pos_id, segs in by_pos.items():
            segs_sorted = sorted(segs, key=lambda s: s["x0"])
            segs_sorted[-1]["is_close"] = True
            for i in range(1, len(segs_sorted)):
                prev, curr = segs_sorted[i - 1], segs_sorted[i]
                connectors.append({
                    "x": curr["x0"], "y0": prev["y0"],
                    "y1": curr["y0"], "color": curr["color"],
                    "pos_id": pos_id,
                })

        seg_df     = pd.DataFrame(segments_all)
        seg_source = ColumnDataSource(seg_df)

        seg_renderer = fig.segment(
            x0="x0", y0="y0", x1="x1", y1="y1",
            source=seg_source, line_width=3, color="color", alpha=0.9,
        )
        fig.circle(
            x="x0", y="y0", source=seg_source,
            size=6, color="color", alpha=0.9,
        )

        close_df = seg_df[seg_df["is_close"]]
        if not close_df.empty:
            fig.scatter(
                x="x1", y="y1",
                source=ColumnDataSource(close_df),
                marker="x", size=10, line_color="color",
                fill_color=None, line_width=2, alpha=0.9,
            )

        if connectors:
            fig.segment(
                x0="x", y0="y0", x1="x", y1="y1",
                source=ColumnDataSource(pd.DataFrame(connectors)),
                line_width=2, color="color", alpha=0.9,
            )

        fig.add_tools(HoverTool(
            renderers=[seg_renderer],
            tooltips=[
                ("Position ID", "@pos_id"),
                ("Status",      "@status"),
                ("Side",        "@side"),
                ("Size",        "@seg_size{0.0000}"),
                ("Price",       "@avg_price_seg{0.0000}"),
                ("Net PnL",     "@net_pnl{0,0.00}"),
            ],
            mode="mouse",
        ))

    # ───────────────────────────────────────────────────────────

    def _build_position_segments(self, position, filled_by_id: dict) -> list:
        orders = [
            filled_by_id[oid]
            for oid in position.order_ids
            if oid in filled_by_id
        ]
        orders.sort(key=lambda o: o.fill_time)
        if not orders:
            return []

        segments = []
        current_size  = 0.0
        current_price = 0.0
        seg_start     = orders[0].fill_time

        for order in orders:
            old_abs    = abs(current_size)
            trade_size = order.size
            trade_abs  = abs(trade_size)

            if old_abs < 1e-12:
                current_size  = trade_size
                current_price = order.fill_price
                seg_start     = order.fill_time
                continue

            side_label = "long" if current_size > 0 else "short"

            if current_size * trade_size > 0:
                # усреднение
                segments.append({
                    "start_time": seg_start, "end_time": order.fill_time,
                    "price": current_price, "side": side_label, "size": old_abs,
                })
                current_price = (
                    (old_abs * current_price + trade_abs * order.fill_price)
                    / (old_abs + trade_abs)
                )
                current_size += trade_size
                seg_start = order.fill_time

            elif trade_abs < old_abs - 1e-12:
                # частичное закрытие
                segments.append({
                    "start_time": seg_start, "end_time": order.fill_time,
                    "price": current_price, "side": side_label, "size": old_abs,
                })
                current_size += trade_size
                seg_start = order.fill_time

            elif abs(trade_abs - old_abs) < 1e-12:
                # полное закрытие
                segments.append({
                    "start_time": seg_start, "end_time": order.fill_time,
                    "price": current_price, "side": side_label, "size": old_abs,
                })
                current_size  = 0.0
                current_price = 0.0

            else:
                # переворот
                segments.append({
                    "start_time": seg_start, "end_time": order.fill_time,
                    "price": current_price, "side": side_label, "size": old_abs,
                })
                remaining     = trade_abs - old_abs
                current_size  = math.copysign(remaining, trade_size)
                current_price = order.fill_price
                seg_start     = order.fill_time

        if abs(current_size) > 1e-12:
            end_time = (
                position.close_time
                if position.status == "closed" and position.close_time
                else getattr(self.engine, "last_event_time", None)
                     or int(pd.Timestamp.now().timestamp() * 1000)
            )
            side_label = "long" if current_size > 0 else "short"
            segments.append({
                "start_time": seg_start, "end_time": end_time,
                "price": current_price, "side": side_label,
                "size": abs(current_size),
            })

        return segments

    # ═══════════════════════════════════════════════════════════
    # EQUITY CURVE
    # ═══════════════════════════════════════════════════════════

    def _build_equity_curve(self) -> pd.DataFrame:
        self._ensure_price_cache()

        initial_balance = getattr(
            getattr(self.engine, "strategy", None), "initial_balance", 0
        )
        filled_orders = sorted(
            (o for o in self.engine.orders if o.status == "filled"),
            key=lambda o: o.fill_time,
        )
        if not filled_orders:
            return pd.DataFrame()

        # ── предсортированные закрытые позиции для two-pointer ──
        closed_positions = sorted(
            (p for p in self.engine.positions
             if p.status == "closed" and p.close_time is not None),
            key=lambda p: p.close_time,
        )
        all_positions = self.engine.positions

        points_time:   list = []
        points_equity: list = []

        closed_pnl = 0.0
        cp_idx     = 0

        for order in filled_orders:
            ts = order.fill_time

            # два указателя: накапливаем PnL закрытых позиций → O(N+M)
            while (cp_idx < len(closed_positions)
                   and closed_positions[cp_idx].close_time <= ts):
                closed_pnl += closed_positions[cp_idx].net_pnl
                cp_idx += 1

            # активные позиции на момент ts
            unrealized = 0.0
            bid = ask = None
            for pos in all_positions:
                if pos.open_time > ts:
                    continue
                if pos.status == "closed" and (
                    pos.close_time is None or pos.close_time <= ts
                ):
                    continue
                if pos.size == 0:
                    continue
                # лениво берём цену — один вызов bisect на все позиции
                if bid is None:
                    bid, ask = self._get_price_at(ts)
                    if bid is None or ask is None:
                        break
                if pos.size > 0:
                    unrealized += (bid - pos.price) * pos.size - pos.fees
                else:
                    unrealized += (pos.price - ask) * abs(pos.size) - pos.fees

            points_time.append(ts)
            points_equity.append(initial_balance + closed_pnl + unrealized)

        if not points_time:
            return pd.DataFrame()

        df = pd.DataFrame({
            "time":   pd.to_datetime(points_time, unit="ms"),
            "equity": points_equity,
        })
        return df

    # ═══════════════════════════════════════════════════════════
    # ИНДИКАТОРЫ СТРАТЕГИИ
    # ═══════════════════════════════════════════════════════════

    def _draw_strategy_indicators(self, fig) -> List[GlyphRenderer]:
        if not self.strategy or not hasattr(self.strategy, "plot"):
            return []
        plot_recorder = self.strategy.plot
        if not plot_recorder.has_data():
            return []

        renderers = []
        for series in plot_recorder.series.values():
            if not series.data:
                continue
            r = self._draw_series(fig, series)
            if r:
                renderers.append(r)
        return renderers

    def _draw_series(self, fig, series):
        dash_map = {"solid": "solid", "dashed": "dashed", "dotted": "dotted"}
        line_dash = dash_map.get(series.linestyle, "solid")

        if series.plot_type == "line":
            df = pd.DataFrame(series.data, columns=["time_ms", "value"])
            df["time"] = pd.to_datetime(df["time_ms"], unit="ms")
            return fig.line(
                x="time", y="value", source=ColumnDataSource(df),
                color=series.color, line_width=series.linewidth,
                line_dash=line_dash, alpha=series.alpha,
                legend_label=series.label, visible=False,
            )
        elif series.plot_type == "band":
            df = pd.DataFrame(series.data, columns=["time_ms", "upper", "lower"])
            df["time"] = pd.to_datetime(df["time_ms"], unit="ms")
            return fig.varea(
                x="time", y1="lower", y2="upper", source=ColumnDataSource(df),
                fill_color=series.color, fill_alpha=series.alpha,
                legend_label=series.label, visible=False,
            )
        elif series.plot_type == "marker":
            df = pd.DataFrame(series.data, columns=["time_ms", "price"])
            df["time"] = pd.to_datetime(df["time_ms"], unit="ms")
            return fig.scatter(
                x="time", y="price", source=ColumnDataSource(df),
                marker=series.marker, size=series.size,
                color=series.color, alpha=series.alpha,
                legend_label=series.label, visible=False,
            )
        elif series.plot_type == "hline":
            price = series.data[0][0]
            fig.add_layout(Span(
                location=price, dimension="width",
                line_color=series.color, line_width=series.linewidth,
                line_dash=line_dash,
            ))
        return None

    def _create_strategy_toggle_button(
        self, renderers: List[GlyphRenderer]
    ) -> Button:
        btn = Button(label="📊 Strategy", button_type="default", width=100)
        valid = [r for r in renderers if r is not None]
        if not valid:
            btn.label = "📊 No Data"
            btn.disabled = True
            return btn
        btn.js_on_click(CustomJS(args=dict(renderers=valid, btn=btn), code="""
            var visible = !renderers[0].visible;
            for (var i = 0; i < renderers.length; i++) renderers[i].visible = visible;
            btn.label = visible ? "📊 Hide" : "📊 Strategy";
            btn.button_type = visible ? "success" : "default";
        """))
        return btn

    # ═══════════════════════════════════════════════════════════
    # ЛИНЕЙКА
    # ═══════════════════════════════════════════════════════════

    def _setup_ruler_components(self, fig):
        ruler_points  = ColumnDataSource(data={"x": [], "y": [], "color": []})
        ruler_lines   = ColumnDataSource(
            data={"x0": [], "y0": [], "x1": [], "y1": [], "color": []}
        )
        ruler_labels  = ColumnDataSource(data={"x": [], "y": [], "text": []})
        ruler_helpers = ColumnDataSource(
            data={"x0": [], "y0": [], "x1": [], "y1": []}
        )

        fig.scatter(
            x="x", y="y", source=ruler_points,
            size=8, color="color", alpha=0.9,
        )
        fig.segment(
            x0="x0", y0="y0", x1="x1", y1="y1", source=ruler_lines,
            line_width=2, color="color", alpha=0.9,
        )
        fig.segment(
            x0="x0", y0="y0", x1="x1", y1="y1", source=ruler_helpers,
            line_width=1, color="#888888", alpha=0.6, line_dash="dashed",
        )
        fig.add_layout(LabelSet(
            x="x", y="y", text="text", source=ruler_labels,
            text_font_size="11pt", text_color="white",
            background_fill_color="#1e1e1e", background_fill_alpha=0.85,
            border_line_color="#ff9800", border_line_width=1,
            x_offset=10, y_offset=5,
        ))

        state = ColumnDataSource(data={
            "active": [False], "first_click": [False],
            "start_x": [0], "start_y": [0],
        })
        info_div = Div(
            text=(
                '<span style="color:#888;font-size:12px;'
                'line-height:28px;">📏 Линейка: выкл</span>'
            ),
            width=300, height=30,
            styles={"margin-left": "10px"},
        )
        ruler_btn = Button(
            label="📏 Линейка", button_type="default", width=100,
        )
        clear_btn = Button(
            label="🗑 Очистить", button_type="warning", width=100,
        )

        ruler_btn.js_on_click(CustomJS(
            args=dict(state=state, btn=ruler_btn, info=info_div),
            code="""
            const active = !state.data['active'][0];
            state.data['active'][0] = active;
            state.data['first_click'][0] = false;
            if (active) {
                btn.button_type = 'success';
                btn.label = '📏 Линейка (ВКЛ)';
                info.text = '<span style="color:#ff9800;font-size:12px;'
                          + 'line-height:28px;">📏 Кликните первую точку</span>';
            } else {
                btn.button_type = 'default';
                btn.label = '📏 Линейка';
                info.text = '<span style="color:#888;font-size:12px;'
                          + 'line-height:28px;">📏 Линейка: выкл</span>';
            }
            state.change.emit();
        """))

        clear_btn.js_on_click(CustomJS(
            args=dict(
                points=ruler_points, lines=ruler_lines,
                labels=ruler_labels, helpers=ruler_helpers,
                state=state, btn=ruler_btn, info=info_div,
            ),
            code="""
            points.data  = {x:[],y:[],color:[]};
            lines.data   = {x0:[],y0:[],x1:[],y1:[],color:[]};
            labels.data  = {x:[],y:[],text:[]};
            helpers.data = {x0:[],y0:[],x1:[],y1:[]};
            state.data['active'][0] = false;
            state.data['first_click'][0] = false;
            btn.button_type = 'default';
            btn.label = '📏 Линейка';
            info.text = '<span style="color:#4caf50;font-size:12px;'
                      + 'line-height:28px;">✓ Очищено</span>';
            points.change.emit(); lines.change.emit();
            labels.change.emit(); helpers.change.emit();
            state.change.emit();
        """))

        fig.js_on_event(Tap, CustomJS(
            args=dict(
                state=state, points=ruler_points, lines=ruler_lines,
                labels=ruler_labels, helpers=ruler_helpers,
                info=info_div, btn=ruler_btn,
            ),
            code="""
            if (!state.data['active'][0]) return;
            const x = cb_obj.x, y = cb_obj.y;
            if (!state.data['first_click'][0]) {
                state.data['first_click'][0] = true;
                state.data['start_x'][0] = x;
                state.data['start_y'][0] = y;
                points.data['x'].push(x);
                points.data['y'].push(y);
                points.data['color'].push('#ff9800');
                info.text = '<span style="color:#ff9800;font-size:12px;'
                          + 'line-height:28px;">📏 Кликните вторую точку</span>';
                points.change.emit();
                state.change.emit();
            } else {
                const sx = state.data['start_x'][0];
                const sy = state.data['start_y'][0];
                const diff = y - sy;
                const pct = ((y - sy) / sy) * 100;
                const is_pos = diff >= 0;
                const color = is_pos ? '#00e676' : '#ff5252';
                const ms = Math.abs(x - sx);
                const hrs = ms / 3600000;
                const t_str = (hrs < 1)
                    ? Math.round(hrs * 60) + ' мин'
                    : hrs.toFixed(1) + ' ч';
                const sign = is_pos ? '+' : '';
                const txt = sign + pct.toFixed(2) + '% | '
                          + sign + diff.toFixed(2) + ' | ' + t_str;

                points.data['x'].push(x);
                points.data['y'].push(y);
                points.data['color'].push(color);

                lines.data['x0'].push(sx);
                lines.data['y0'].push(sy);
                lines.data['x1'].push(x);
                lines.data['y1'].push(y);
                lines.data['color'].push(color);

                helpers.data['x0'].push(sx);
                helpers.data['y0'].push(sy);
                helpers.data['x1'].push(x);
                helpers.data['y1'].push(sy);
                helpers.data['x0'].push(x);
                helpers.data['y0'].push(sy);
                helpers.data['x1'].push(x);
                helpers.data['y1'].push(y);

                labels.data['x'].push((sx + x) / 2);
                labels.data['y'].push((sy + y) / 2);
                labels.data['text'].push(txt);

                state.data['first_click'][0] = false;
                state.data['active'][0] = false;
                btn.button_type = 'default';
                btn.label = '📏 Линейка';
                info.text = '<span style="color:'
                          + (is_pos ? '#00e676' : '#ff5252')
                          + ';font-size:12px;line-height:28px;">📏 '
                          + txt + '</span>';

                points.change.emit(); lines.change.emit();
                labels.change.emit(); helpers.change.emit();
                state.change.emit();
            }
        """))

        return ruler_btn, clear_btn, info_div

    # ═══════════════════════════════════════════════════════════
    # УТИЛИТЫ
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _apply_dark_theme(fig):
        fig.background_fill_color = "#1e1e1e"
        fig.border_fill_color     = "#1e1e1e"
        fig.title.text_color      = "white"
        fig.xaxis.axis_label_text_color  = "white"
        fig.yaxis.axis_label_text_color  = "white"
        fig.xaxis.major_label_text_color = "#d5d5d5"
        fig.yaxis.major_label_text_color = "#d5d5d5"
        fig.grid.grid_line_alpha         = 0.3

    def _print_stats(self):
        filled = [o for o in self.engine.orders if o.status == "filled"]
        buys   = [o for o in filled if o.size > 0]
        sells  = [o for o in filled if o.size < 0]

        buy_vol  = sum(abs(o.size) * o.fill_price for o in buys)
        sell_vol = sum(abs(o.size) * o.fill_price for o in sells)

        realized   = self.engine.get_realized_pnl()
        fees       = self.engine.get_total_fees()
        net_closed = self.engine.get_net_pnl()

        open_pos_list = [p for p in self.engine.positions if p.status == "open"]
        if open_pos_list:
            unrealized     = self.engine.get_unrealized_pnl()
            unrealized_net = unrealized - open_pos_list[0].fees
        else:
            unrealized = unrealized_net = 0.0

        init_bal  = getattr(
            getattr(self.engine, "strategy", None), "initial_balance", 0
        )
        total_net = net_closed + unrealized_net
        final_eq  = init_bal + total_net
        ret_pct   = (total_net / init_bal * 100) if init_bal > 0 else 0.0

        W = 63                              # ширина содержимого между ║…║

        def _line(label: str, val: str) -> str:
            content = f"  {label:<24s}{val:>18s}"
            return f"║{content:<{W}s}║"

        hdr = f"║{'BACKTEST RESULTS':^{W}s}║"
        sep = f"╠{'═' * W}╣"
        top = f"╔{'═' * W}╗"
        bot = f"╚{'═' * W}╝"

        print(f"""
{top}
{hdr}
{sep}
{_line("Initial Balance:",      f"${init_bal:>,.2f}")}
{_line("Final Equity:",         f"${final_eq:>,.2f}")}
{_line("Realized PnL (gross):", f"${realized:>+,.2f}")}
{_line("Unrealized PnL:",       f"${unrealized:>+,.2f}")}
{_line("Net PnL (closed):",     f"${net_closed:>+,.2f}")}
{_line("Net PnL (total):",      f"${total_net:>+,.2f}")}
{_line("Return:",               f"{ret_pct:>+.2f}%")}
{sep}
{_line("Buy fills:",            f"{len(buys):>d}")}
{_line("Sell fills:",           f"{len(sells):>d}")}
{_line("Total fills:",          f"{len(filled):>d}")}
{sep}
{_line("Buy Volume:",           f"${buy_vol:>,.2f}")}
{_line("Sell Volume:",          f"${sell_vol:>,.2f}")}
{_line("Total Volume:",         f"${buy_vol + sell_vol:>,.2f}")}
{sep}
{_line("Total Fees:",           f"${fees:>,.2f}")}
{bot}
        """)