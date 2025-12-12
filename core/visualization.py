# visualizer.py
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Span, Segment
from bokeh.layouts import column
import pandas as pd

from engine import ExchangeEngine


class BacktestVisualizer:

    def __init__(self, engine, strategy=None):
        self.engine: ExchangeEngine = engine
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
            height=600,
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

        # ══════════════════════════════════════════════════════
        # НОВОЕ: Отображение позиций
        # ══════════════════════════════════════════════════════
        self._draw_positions(p)

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
                'time': pd.to_datetime(order.fill_time, unit='ms'),
                'price': order.fill_price,
                'size': abs(order.size),
            }

            if order.size > 0:
                buys.append(trade)
            else:
                sells.append(trade)

        return pd.DataFrame(buys), pd.DataFrame(sells)

    def _draw_positions(self, fig):
        """
        Рисует горизонтальные линии для каждой позиции.
        Линия показывает среднюю цену входа и тянется от открытия до закрытия.
        При усреднении линия меняет уровень.
        """
        for position in self.engine.positions:
            segments = self._build_position_segments(position)
            
            if not segments:
                continue
            
            # Определяем цвет по направлению позиции
            # Long = зелёный, Short = красный
            color = "#00bcd4" if position.size >= 0 else "#ff9800"
            
            for segment in segments:
                x0 = pd.to_datetime(segment['start_time'], unit='ms')
                x1 = pd.to_datetime(segment['end_time'], unit='ms')
                y = segment['price']
                
                # Рисуем сегмент линии
                fig.segment(
                    x0=x0, y0=y, x1=x1, y1=y,
                    line_width=2, color=color, alpha=0.7,
                    line_dash='dashed'
                )

    def _build_position_segments(self, position):
        """
        Строит сегменты горизонтальных линий для позиции.
        Каждый сегмент = период с одной средней ценой входа.
        
        Returns: list of {'start_time', 'end_time', 'price'}
        """
        # Получаем все ордера позиции, отсортированные по времени
        position_orders = [
            o for o in self.engine.orders 
            if o.id in position.order_ids and o.status == "filled"
        ]
        position_orders.sort(key=lambda o: o.fill_time)
        
        if not position_orders:
            return []
        
        segments = []
        
        # Симулируем изменение позиции по каждому ордеру
        current_size = 0.0
        current_price = 0.0
        segment_start_time = position_orders[0].fill_time
        
        for i, order in enumerate(position_orders):
            old_size = abs(current_size)
            trade_size = order.size
            trade_abs = abs(trade_size)
            
            # Если позиция пустая — это первый вход
            if abs(current_size) < 1e-12:
                current_size = trade_size
                current_price = order.fill_price
                segment_start_time = order.fill_time
                continue
            
            # Та же сторона — усреднение
            if current_size * trade_size > 0:
                # Сохраняем старый сегмент
                segments.append({
                    'start_time': segment_start_time,
                    'end_time': order.fill_time,
                    'price': current_price
                })
                
                # Пересчитываем среднюю
                old_notional = old_size * current_price
                new_notional = trade_abs * order.fill_price
                total_size = old_size + trade_abs
                
                current_price = (old_notional + new_notional) / total_size
                current_size += trade_size
                segment_start_time = order.fill_time
                continue
            
            # Противоположная сторона — закрытие или переворот
            closed_qty = min(old_size, trade_abs)
            
            # Частичное закрытие
            if trade_abs < old_size - 1e-12:
                current_size += trade_size
                # Цена не меняется
                continue
            
            # Полное закрытие
            if abs(trade_abs - old_size) < 1e-12:
                # Завершаем последний сегмент
                segments.append({
                    'start_time': segment_start_time,
                    'end_time': order.fill_time,
                    'price': current_price
                })
                current_size = 0.0
                current_price = 0.0
                continue
            
            # Переворот — закрываем старую, начинаем новую
            remaining = trade_abs - old_size
            
            # Завершаем сегмент закрываемой позиции
            segments.append({
                'start_time': segment_start_time,
                'end_time': order.fill_time,
                'price': current_price
            })
            
            # Открываем новую позицию
            current_size = trade_size / abs(trade_size) * remaining
            current_price = order.fill_price
            segment_start_time = order.fill_time
        
        # Если позиция всё ещё открыта — добавляем последний сегмент до конца
        if abs(current_size) > 1e-12:
            end_time = position.close_time if position.status == "closed" else self.engine.last_event_time
            segments.append({
                'start_time': segment_start_time,
                'end_time': end_time,
                'price': current_price
            })
        
        return segments

    def _build_equity_curve(self):
        """
        Строим equity curve: отслеживаем баланс после каждой сделки.
        
        Логика:
        1. Каждый fill меняет позицию
        2. После fill считаем текущий PnL всех позиций
        3. Equity = initial_balance + sum(закрытые позиции) + unrealized(открытой)
        """
        initial_balance = self.engine.strategy.initial_balance if self.engine.strategy else 0

        filled_orders = [o for o in self.engine.orders if o.status == "filled"]
        filled_orders.sort(key=lambda o: o.fill_time)

        if not filled_orders:
            return pd.DataFrame()

        points = []
        
        # Кэш цен для расчёта unrealized PnL
        price_cache = self._build_price_cache()
        
        for order in filled_orders:
            timestamp = order.fill_time
            
            # 1. Суммируем PnL всех ЗАКРЫТЫХ позиций на момент timestamp
            closed_pnl = sum(
                p.net_pnl for p in self.engine.positions 
                if p.status == "closed" and p.close_time <= timestamp
            )
            
            # 2. Находим ОТКРЫТУЮ позицию на момент timestamp
            open_position = None
            for pos in self.engine.positions:
                if pos.status == "open" or (pos.status == "closed" and pos.close_time > timestamp):
                    # Позиция открыта на момент timestamp
                    if pos.open_time <= timestamp:
                        open_position = pos
                        break
            
            # 3. Считаем unrealized PnL открытой позиции
            unrealized_pnl = 0.0
            if open_position and open_position.size != 0:
                # Получаем bid/ask на момент timestamp
                bid, ask = price_cache.get(timestamp, (None, None))
                
                if bid and ask:
                    if open_position.size > 0:  # Long
                        mark_price = bid
                        unrealized_gross = (mark_price - open_position.price) * open_position.size
                    else:  # Short
                        mark_price = ask
                        unrealized_gross = (open_position.price - mark_price) * abs(open_position.size)
                    
                    # Вычитаем накопленные комиссии
                    unrealized_pnl = unrealized_gross - open_position.fees
            
            # 4. Итоговый equity
            equity = initial_balance + closed_pnl + unrealized_pnl

            points.append({
                'time': pd.to_datetime(timestamp, unit='ms'),
                'equity': equity,
            })

        return pd.DataFrame(points)

    def _build_price_cache(self):
        """
        Строим кэш bid/ask цен для каждого timestamp.
        Возвращает dict: {timestamp: (bid, ask)}
        """
        cache = {}
        
        # Проходим по всем bookticker событиям
        for event in self.engine.events.to_dicts():
            if event.get("event_type") == "bookticker":
                timestamp = event["event_time"]
                bid = event.get("bid_price")
                ask = event.get("ask_price")
                cache[timestamp] = (bid, ask)
        
        return cache

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
        net_pnl_closed = self.engine.get_net_pnl()
        
        # Для открытой позиции (если есть)
        open_positions = [p for p in self.engine.positions if p.status == "open"]
        
        if open_positions:
            open_pos = open_positions[0]  # Должна быть только одна
            unrealized_pnl = self.engine.get_unrealized_pnl()
            unrealized_net = unrealized_pnl - open_pos.fees
        else:
            unrealized_pnl = 0.0
            unrealized_net = 0.0

        # Итоговый баланс
        initial_balance = self.engine.strategy.initial_balance if self.engine.strategy else 0
        final_equity = initial_balance + net_pnl_closed + unrealized_net
        
        # Общий net PnL (закрытые + открытая)
        total_net_pnl = net_pnl_closed + unrealized_net
        return_pct = (total_net_pnl / initial_balance * 100) if initial_balance > 0 else 0

        print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                      BACKTEST RESULTS                         ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Initial Balance:       ${initial_balance:>14,.2f}            ║
    ║  Final Equity:          ${final_equity:>14,.2f}               ║
    ║  Realized PnL (gross):  ${realized_pnl:>+14,.2f}              ║
    ║  Unrealized PnL:        ${unrealized_pnl:>+14,.2f}            ║
    ║  Net PnL (closed):      ${net_pnl_closed:>+14,.2f}            ║
    ║  Net PnL (total):       ${total_net_pnl:>+14,.2f}             ║
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
    ║  Open Position:         {"YES" if open_positions else "NO":>14}     ║
    ╚═══════════════════════════════════════════════════════════════╝
        """)