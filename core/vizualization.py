# visualizer.py
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Span
from bokeh.layouts import column
import pandas as pd


class BacktestVisualizer:

    def __init__(self, engine, strategy=None):
        self.engine = engine
        self.strategy = strategy

    def show(self, title="Backtest Result"):
        df = self.engine.bookticker
        pdf = df.to_pandas()
        pdf['time'] = pd.to_datetime(pdf['event_time'], unit='ms')

        if len(pdf) > 50000:
            step = len(pdf) // 50000
            pdf = pdf.iloc[::step]

        source = ColumnDataSource(pdf)

        # ══════════════════════════════════════════════════════
        # График цены
        # ══════════════════════════════════════════════════════
        p = figure(
            title=title,
            x_axis_type='datetime',
            height=800,
            sizing_mode="stretch_width",
            tools="pan,wheel_zoom,box_zoom,reset,save,crosshair"
        )

        self._apply_dark_theme(p)

        p.line(x='time', y='bid_price', source=source,
               legend_label="Bid", line_width=2, color="#00e676", alpha=0.9)
        p.line(x='time', y='ask_price', source=source,
               legend_label="Ask", line_width=2, color="#ff5252", alpha=0.9)

        # Сделки из движка
        buys, sells = self._get_trades()

        if not buys.empty:
            buys_source = ColumnDataSource(buys)
            p.scatter(
                x='time', y='price', source=buys_source,
                marker='triangle', size=10, color="#00ff00",
                alpha=0.8, legend_label="Buy"
            )

        if not sells.empty:
            sells_source = ColumnDataSource(sells)
            p.scatter(
                x='time', y='price', source=sells_source,
                marker='inverted_triangle', size=10, color="#ff0000",
                alpha=0.8, legend_label="Sell"
            )

        hover_price = HoverTool(tooltips=[
            ("Time", "@time{%F %T}"),
            ("Bid", "@bid_price{0.00000}"),
            ("Ask", "@ask_price{0.00000}"),
        ], formatters={'@time': 'datetime'}, mode='vline')
        p.add_tools(hover_price)

        p.legend.location = "top_left"
        p.legend.background_fill_alpha = 0.5
        p.legend.label_text_color = "white"
        p.legend.click_policy = "hide"

        # ══════════════════════════════════════════════════════
        # График Equity
        # ══════════════════════════════════════════════════════
        equity_df = self._build_equity_curve()
        initial_balance = self.engine.strategy.initial_balance if self.engine.strategy else 0

        p_equity = figure(
            title=f"Equity",
            x_axis_type='datetime',
            height=400,
            sizing_mode="stretch_width",
            tools="pan,wheel_zoom,box_zoom,reset,save,crosshair",
            x_range=p.x_range
        )

        self._apply_dark_theme(p_equity)

        if not equity_df.empty:
            equity_source = ColumnDataSource(equity_df)

            p_equity.line(
                x='time', y='equity', source=equity_source,
                line_width=2, color="#2196F3", legend_label="Equity"
            )

            # Линия начального баланса
            if initial_balance > 0:
                initial_line = Span(
                    location=initial_balance, dimension='width',
                    line_color='#888888', line_width=1, line_dash='dashed'
                )
                p_equity.add_layout(initial_line)

            hover_equity = HoverTool(tooltips=[
                ("Time", "@time{%F %T}"),
                ("Equity", "$@equity{0,0.00}"),
            ], formatters={'@time': 'datetime'})
            p_equity.add_tools(hover_equity)

            p_equity.legend.location = "top_left"
            p_equity.legend.background_fill_alpha = 0.5
            p_equity.legend.label_text_color = "white"

        self._print_stats()

        output_file("backtest_chart.html")
        show(column(p, p_equity, sizing_mode="stretch_width"))

    def _get_trades(self):
        """Сделки из движка."""
        filled_orders = [o for o in self.engine.orders if o.status == "filled"]

        buys = []
        sells = []

        for order in filled_orders:
            trade = {
                'time': pd.to_datetime(order.exchange_time, unit='ms'),
                'price': order.fill_price,
                'size': abs(order.size),
            }

            if order.size > 0:
                buys.append(trade)
            else:
                sells.append(trade)

        return pd.DataFrame(buys), pd.DataFrame(sells)

    def _build_equity_curve(self):
        """
        Строим equity curve из ордеров движка.
        """
        initial_balance = self.engine.strategy.initial_balance if self.engine.strategy else 0

        filled_orders = [o for o in self.engine.orders if o.status == "filled"]
        filled_orders.sort(key=lambda o: o.exchange_time)

        if not filled_orders:
            return pd.DataFrame()

        points = []
        cumulative_pnl = 0.0
        cumulative_fees = 0.0
        position_size = 0.0
        position_price = 0.0

        for order in filled_orders:
            trade_size = order.size
            trade_price = order.fill_price
            notional = abs(trade_size) * trade_price

            # Комиссия из движка
            fee = notional * self.engine.maker_fee
            cumulative_fees += fee

            # PnL при закрытии
            if position_size != 0 and (position_size * trade_size < 0):
                closed_size = min(abs(position_size), abs(trade_size))

                if position_size > 0:
                    pnl = (trade_price - position_price) * closed_size
                else:
                    pnl = (position_price - trade_price) * closed_size

                cumulative_pnl += pnl

            # Обновляем позицию
            if position_size == 0:
                position_size = trade_size
                position_price = trade_price
            elif position_size * trade_size > 0:
                total = abs(position_size) + abs(trade_size)
                position_price = (abs(position_size) * position_price + abs(trade_size) * trade_price) / total
                position_size += trade_size
            else:
                if abs(trade_size) >= abs(position_size):
                    remaining = abs(trade_size) - abs(position_size)
                    if remaining > 0:
                        position_size = remaining if trade_size > 0 else -remaining
                        position_price = trade_price
                    else:
                        position_size = 0
                        position_price = 0
                else:
                    position_size += trade_size

            equity = initial_balance + cumulative_pnl - cumulative_fees

            points.append({
                'time': pd.to_datetime(order.exchange_time, unit='ms'),
                'equity': equity,
            })

        return pd.DataFrame(points)

    def _apply_dark_theme(self, fig):
        fig.background_fill_color = "#1e1e1e"
        fig.border_fill_color = "#1e1e1e"
        fig.title.text_color = "white"
        fig.xaxis.axis_label_text_color = "white"
        fig.yaxis.axis_label_text_color = "white"
        fig.xaxis.major_label_text_color = "#d5d5d5"
        fig.yaxis.major_label_text_color = "#d5d5d5"
        fig.grid.grid_line_alpha = 0.3

    def _print_stats(self):
        """Статистика — всё из движка."""
        filled = [o for o in self.engine.orders if o.status == "filled"]
        buys = [o for o in filled if o.size > 0]
        sells = [o for o in filled if o.size < 0]

        # Объёмы
        buy_volume_usd = sum(abs(o.size) * o.fill_price for o in buys)
        sell_volume_usd = sum(abs(o.size) * o.fill_price for o in sells)
        total_volume_usd = buy_volume_usd + sell_volume_usd

        # Из движка
        realized_pnl = self.engine.get_realized_pnl()
        total_fees = self.engine.get_total_fees()
        net_pnl = self.engine.get_net_pnl()
        unrealized_pnl = self.engine.get_unrealized_pnl()

        # Из стратегии только initial_balance
        initial_balance = self.engine.strategy.initial_balance if self.engine.strategy else 0
        final_equity = initial_balance + net_pnl + unrealized_pnl
        return_pct = (net_pnl / initial_balance * 100) if initial_balance > 0 else 0

        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                      BACKTEST RESULTS                         ║
╠═══════════════════════════════════════════════════════════════╣
║  Initial Balance:       ${initial_balance:>14,.2f}            ║
║  Final Equity:          ${final_equity:>14,.2f}               ║
║  Realized PnL:          ${realized_pnl:>+14,.2f}              ║
║  Unrealized PnL:        ${unrealized_pnl:>+14,.2f}            ║
║  Net PnL:               ${net_pnl:>+14,.2f}                   ║
║  Return:                {return_pct:>+14.2f}%                 ║
╠═══════════════════════════════════════════════════════════════╣
║  Buy fills:             {len(buys):>14}                       ║
║  Sell fills:            {len(sells):>14}                      ║
║  Total fills:           {len(filled):>14}                     ║
╠═══════════════════════════════════════════════════════════════╣
║  Buy Volume:            ${buy_volume_usd:>14,.2f}             ║
║  Sell Volume:           ${sell_volume_usd:>14,.2f}            ║
║  Total Volume:          ${total_volume_usd:>14,.2f}           ║
╠═══════════════════════════════════════════════════════════════╣
║  Total Fees:            ${total_fees:>14,.2f}                 ║
╚═══════════════════════════════════════════════════════════════╝
        """)