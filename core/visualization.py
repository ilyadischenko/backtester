# visualizer.py
import math
from typing import Dict
from bokeh.plotting import figure, show, output_file
from bokeh.models import (
    ColumnDataSource, HoverTool, Span, LabelSet, CustomJS, Button, Div
)
from bokeh.layouts import column, row
from bokeh.events import Tap
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
               legend_label="Bid", line_width=2, color="#006132", alpha=0.9)
        p.line(x='time', y='ask_price', source=source,
               legend_label="Ask", line_width=2, color="#6b1f1f", alpha=0.9)

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
        # Отображение позиций
        # ══════════════════════════════════════════════════════
        self._draw_positions(p)

        # ══════════════════════════════════════════════════════
        # НОВОЕ: Инструмент "Линейка"
        # ══════════════════════════════════════════════════════
        ruler_controls = self._add_ruler_tool(p, pdf)

        # hover_price = HoverTool(tooltips=[
        #     ("Time", "@time{%F %T}"),
        #     ("Bid", "@bid_price{0.00000}"),
        #     ("Ask", "@ask_price{0.00000}"),
        # ], formatters={'@time': 'datetime'}, mode='vline')
        # p.add_tools(hover_price)

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

        # Компонуем с панелью управления линейкой
        show(column(ruler_controls, p, p_equity, sizing_mode="stretch_width"))

    def _add_ruler_tool(self, fig, price_df):
        """
        Добавляет инструмент "Линейка" для измерения процентного движения.

        Использование:
        1. Нажмите кнопку "📏 Линейка"
        2. Кликните на начальную точку
        3. Кликните на конечную точку
        4. Увидите измерение (%, цена, время)
        5. "Очистить" удаляет все линейки
        """

        # Источники данных для линейки
        # Точки линейки
        ruler_points_source = ColumnDataSource(data={
            'x': [], 'y': [], 'color': []
        })

        # Линии линейки (сегменты)
        ruler_lines_source = ColumnDataSource(data={
            'x0': [], 'y0': [], 'x1': [], 'y1': [], 'color': []
        })

        # Лейблы с информацией
        ruler_labels_source = ColumnDataSource(data={
            'x': [], 'y': [], 'text': []
        })

        # Вспомогательные линии (вертикальная и горизонтальная)
        ruler_helper_source = ColumnDataSource(data={
            'x0': [], 'y0': [], 'x1': [], 'y1': []
        })

        # Рисуем элементы линейки
        # Точки
        fig.scatter(
            x='x', y='y', source=ruler_points_source,
            size=10, color='color', alpha=0.9,
            marker='circle'
        )

        # Основная линия
        fig.segment(
            x0='x0', y0='y0', x1='x1', y1='y1',
            source=ruler_lines_source,
            line_width=2, color='color', alpha=0.9
        )

        # Вспомогательные пунктирные линии
        fig.segment(
            x0='x0', y0='y0', x1='x1', y1='y1',
            source=ruler_helper_source,
            line_width=1, color='#888888', alpha=0.6,
            line_dash='dashed'
        )

        # Лейблы
        labels = LabelSet(
            x='x', y='y', text='text',
            source=ruler_labels_source,
            text_font_size='11pt',
            text_color='white',
            background_fill_color='#1e1e1e',
            background_fill_alpha=0.85,
            border_line_color='#ff9800',
            border_line_width=1,
            x_offset=10, y_offset=5
        )
        fig.add_layout(labels)

        # Состояние линейки (хранится в скрытом источнике)
        state_source = ColumnDataSource(data={
            'active': [False],
            'first_click': [False],
            'start_x': [0],
            'start_y': [0]
        })

        # Информационный div
        info_div = Div(
            text='<span style="color: #888; font-size: 12px;">📏 Линейка: выключена</span>',
            width=400,
            height=30,
            styles={'margin-left': '10px'}
        )

        # Кнопки управления
        ruler_button = Button(
            label="📏 Линейка",
            button_type="default",
            width=120
        )

        clear_button = Button(
            label="🗑 Очистить",
            button_type="warning",
            width=100
        )

        # JavaScript для переключения режима линейки
        toggle_ruler_js = CustomJS(args=dict(
            state=state_source,
            btn=ruler_button,
            info=info_div
        ), code="""
            const active = !state.data['active'][0];
            state.data['active'][0] = active;
            state.data['first_click'][0] = false;

            if (active) {
                btn.button_type = 'success';
                btn.label = '📏 Линейка (ВКЛ)';
                info.text = '<span style="color: #ff9800; font-size: 12px;">📏 Кликните на начальную точку</span>';
            } else {
                btn.button_type = 'default';
                btn.label = '📏 Линейка';
                info.text = '<span style="color: #888; font-size: 12px;">📏 Линейка: выключена</span>';
            }
            state.change.emit();
        """)
        ruler_button.js_on_click(toggle_ruler_js)

        # JavaScript для очистки линеек
        clear_ruler_js = CustomJS(args=dict(
            points=ruler_points_source,
            lines=ruler_lines_source,
            labels=ruler_labels_source,
            helpers=ruler_helper_source,
            state=state_source,
            btn=ruler_button,
            info=info_div
        ), code="""
            points.data = {'x': [], 'y': [], 'color': []};
            lines.data = {'x0': [], 'y0': [], 'x1': [], 'y1': [], 'color': []};
            labels.data = {'x': [], 'y': [], 'text': []};
            helpers.data = {'x0': [], 'y0': [], 'x1': [], 'y1': []};

            state.data['active'][0] = false;
            state.data['first_click'][0] = false;

            btn.button_type = 'default';
            btn.label = '📏 Линейка';
            info.text = '<span style="color: #4caf50; font-size: 12px;">✓ Линейки очищены</span>';

            points.change.emit();
            lines.change.emit();
            labels.change.emit();
            helpers.change.emit();
            state.change.emit();
        """)
        clear_button.js_on_click(clear_ruler_js)

        # JavaScript для обработки кликов на графике
        tap_callback = CustomJS(args=dict(
            state=state_source,
            points=ruler_points_source,
            lines=ruler_lines_source,
            labels=ruler_labels_source,
            helpers=ruler_helper_source,
            info=info_div,
            btn=ruler_button
        ), code="""
            if (!state.data['active'][0]) return;

            const x = cb_obj.x;
            const y = cb_obj.y;

            if (!state.data['first_click'][0]) {
                // Первый клик - начальная точка
                state.data['first_click'][0] = true;
                state.data['start_x'][0] = x;
                state.data['start_y'][0] = y;

                // Добавляем начальную точку
                points.data['x'].push(x);
                points.data['y'].push(y);
                points.data['color'].push('#ff9800');

                info.text = '<span style="color: #ff9800; font-size: 12px;">📏 Кликните на конечную точку</span>';

                points.change.emit();
                state.change.emit();
            } else {
                // Второй клик - конечная точка
                const start_x = state.data['start_x'][0];
                const start_y = state.data['start_y'][0];

                // Расчёты
                const price_diff = y - start_y;
                const percent_change = ((y - start_y) / start_y) * 100;
                const is_positive = price_diff >= 0;
                const color = is_positive ? '#00e676' : '#ff5252';

                // Время
                const start_date = new Date(start_x);
                const end_date = new Date(x);
                const time_diff_ms = Math.abs(x - start_x);
                const time_diff_hours = time_diff_ms / (1000 * 60 * 60);

                let time_str;
                if (time_diff_hours < 1) {
                    time_str = Math.round(time_diff_hours * 60) + ' мин';
                } else if (time_diff_hours < 24) {
                    time_str = time_diff_hours.toFixed(1) + ' ч';
                } else {
                    time_str = (time_diff_hours / 24).toFixed(1) + ' дн';
                }

                // Добавляем конечную точку
                points.data['x'].push(x);
                points.data['y'].push(y);
                points.data['color'].push(color);

                // Добавляем основную линию
                lines.data['x0'].push(start_x);
                lines.data['y0'].push(start_y);
                lines.data['x1'].push(x);
                lines.data['y1'].push(y);
                lines.data['color'].push(color);

                // Вспомогательные линии (горизонтальная и вертикальная)
                // Горизонтальная от start до x на уровне start_y
                helpers.data['x0'].push(start_x);
                helpers.data['y0'].push(start_y);
                helpers.data['x1'].push(x);
                helpers.data['y1'].push(start_y);

                // Вертикальная от start_y до y на x
                helpers.data['x0'].push(x);
                helpers.data['y0'].push(start_y);
                helpers.data['x1'].push(x);
                helpers.data['y1'].push(y);

                // Форматируем цену
                const format_price = (p) => {
                    if (Math.abs(p) >= 1000) return p.toFixed(2);
                    if (Math.abs(p) >= 1) return p.toFixed(4);
                    return p.toFixed(6);
                };

                // Лейбл с информацией
                const sign = is_positive ? '+' : '';
                const label_text = sign + percent_change.toFixed(2) + '% | ' + 
                                   sign + format_price(price_diff) + ' | ' + 
                                   time_str;

                labels.data['x'].push((start_x + x) / 2);
                labels.data['y'].push((start_y + y) / 2);
                labels.data['text'].push(label_text);

                // Сброс состояния
                state.data['first_click'][0] = false;
                state.data['active'][0] = false;

                btn.button_type = 'default';
                btn.label = '📏 Линейка';

                const result_color = is_positive ? '#00e676' : '#ff5252';
                info.text = '<span style="color: ' + result_color + '; font-size: 12px;">📏 ' + label_text + '</span>';

                points.change.emit();
                lines.change.emit();
                labels.change.emit();
                helpers.change.emit();
                state.change.emit();
            }
        """)

        fig.js_on_event(Tap, tap_callback)

        # Панель управления
        controls = row(
            ruler_button,
            clear_button,
            info_div,
            sizing_mode="stretch_width",
            styles={'background-color': '#1e1e1e', 'padding': '10px'}
        )

        return controls

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
        Рисует позиции:
        - горизонтальные сегменты среднего входа;
        - вертикальные соединения между сегментами одной позиции (смена средней / переворот);
        - X‑маркер в конце последнего сегмента (закрытие позиции).
        """
        segments_all = []
        by_pos: Dict[int, list] = {}

        # Собираем все сегменты всех позиций
        for position in self.engine.positions:
            segs = self._build_position_segments(position)
            if not segs:
                continue

            for seg in segs:
                start_dt = pd.to_datetime(seg['start_time'], unit='ms')
                end_dt = pd.to_datetime(seg['end_time'], unit='ms')

                color = "#00e676" if seg['side'] == "long" else "#ff9800"

                seg_dict = {
                    # геометрия
                    'x0': start_dt,
                    'y0': seg['price'],
                    'x1': end_dt,
                    'y1': seg['price'],
                    'color': color,

                    # данные о позиции
                    'pos_id': position.id,
                    'status': position.status,
                    'side': seg['side'],
                    'seg_size': seg['size'],
                    'avg_price_seg': seg['price'],          # средняя в этом сегменте
                    'entry_price_pos': position.price,      # общая средняя позиции
                    'open_time_pos': pd.to_datetime(position.open_time, unit='ms'),
                    'close_time_pos': (
                        pd.to_datetime(position.close_time, unit='ms')
                        if position.close_time is not None else pd.NaT
                    ),
                    'realized_pnl': position.realized_pnl,
                    'fees': position.fees,
                    'net_pnl': position.net_pnl,

                    # флаг "это последний сегмент позиции"
                    'is_close': False,
                }

                segments_all.append(seg_dict)
                by_pos.setdefault(position.id, []).append(seg_dict)

        if not segments_all:
            return

        # Помечаем последний сегмент каждой позиции как "закрытие"
        connectors = []
        for pos_id, segs in by_pos.items():
            segs_sorted = sorted(segs, key=lambda s: s['x0'])
            # последний сегмент — там позиция заканчивается
            segs_sorted[-1]['is_close'] = True

            # вертикальные соединения между сегментами этой позиции
            for i in range(1, len(segs_sorted)):
                prev_seg = segs_sorted[i - 1]
                curr_seg = segs_sorted[i]

                connectors.append({
                    'x': curr_seg['x0'],
                    'y0': prev_seg['y0'],
                    'y1': curr_seg['y0'],
                    'color': curr_seg['color'],
                    'pos_id': pos_id,
                })

        # --- рисуем сегменты ---
        seg_df = pd.DataFrame(segments_all)
        seg_source = ColumnDataSource(seg_df)

        seg_renderer = fig.segment(
            x0='x0', y0='y0', x1='x1', y1='y1',
            source=seg_source,
            line_width=3,
            color='color',
            alpha=0.9,
        )

        # точки на старте каждого сегмента (открытие/изменение средней)
        fig.circle(
            x='x0', y='y0',
            source=seg_source,
            size=6,
            color='color',
            alpha=0.9,
        )

        # X‑маркер в конце последнего сегмента (где позиция закрылась)
        close_df = seg_df[seg_df['is_close']]
        if not close_df.empty:
            close_source = ColumnDataSource(close_df)
            fig.scatter(
                x='x1', y='y1',
                source=close_source,
                marker='x',
                size=10,
                line_color='color',
                fill_color=None,
                line_width=2,
                alpha=0.9,
            )

        # --- вертикальные соединения между сегментами ---
        if connectors:
            conn_df = pd.DataFrame(connectors)
            conn_source = ColumnDataSource(conn_df)
            fig.segment(
                x0='x', y0='y0', x1='x', y1='y1',
                source=conn_source,
                line_width=2,
                color='color',
                alpha=0.9,
            )

        # Hover по сегментам (позиционная информация)
        hover_pos = HoverTool(
            renderers=[seg_renderer],
            tooltips=[
                ("Position ID", "@pos_id"),
                ("Status", "@status"),
                ("Side", "@side"),
                ("Segment size", "@seg_size{0.0000}"),
                ("Segment price", "@avg_price_seg{0.0000}"),
                ("Pos avg price", "@entry_price_pos{0.0000}"),
                ("Pos open", "@open_time_pos{%F %T}"),
                ("Pos close", "@close_time_pos{%F %T}"),
                ("Realized PnL", "@realized_pnl{0,0.00}"),
                ("Fees", "@fees{0,0.00}"),
                ("Net PnL", "@net_pnl{0,0.00}"),
            ],
            formatters={
                '@open_time_pos': 'datetime',
                '@close_time_pos': 'datetime',
            },
            mode='mouse',
        )
        fig.add_tools(hover_pos)  
    def _build_position_segments(self, position):
        """
        Строит сегменты горизонтальных линий для позиции.
        Каждый сегмент: период времени, когда средняя цена и знак позиции постоянны.
        """
        position_orders = [
            o for o in self.engine.orders
            if o.id in position.order_ids and o.status == "filled"
        ]
        position_orders.sort(key=lambda o: o.fill_time)

        if not position_orders:
            return []

        segments = []
        current_size = 0.0
        current_price = 0.0
        segment_start_time = position_orders[0].fill_time

        for order in position_orders:
            old_size = abs(current_size)
            trade_size = order.size
            trade_abs = abs(trade_size)

            # Открытие новой позиции (до этого size = 0)
            if abs(current_size) < 1e-12:
                current_size = trade_size
                current_price = order.fill_price
                segment_start_time = order.fill_time
                continue

            # Усреднение (увеличение в ту же сторону)
            if current_size * trade_size > 0:
                side_label = "long" if current_size > 0 else "short"
                segments.append({
                    'start_time': segment_start_time,
                    'end_time': order.fill_time,
                    'price': current_price,
                    'side': side_label,
                    'size': abs(current_size),
                })

                old_notional = old_size * current_price
                new_notional = trade_abs * order.fill_price
                total_size = old_size + trade_abs

                current_price = (old_notional + new_notional) / total_size
                current_size += trade_size
                segment_start_time = order.fill_time
                continue

            # Противоположная сделка: частичное/полное закрытие или переворот
            closed_qty = min(old_size, trade_abs)

            # Частичное закрытие — размер позиции уменьшается, цена не меняется
            if trade_abs < old_size - 1e-12:
                current_size += trade_size
                # сегмент продолжается, цена та же, просто размер меньше
                continue

            # Полное закрытие
            if abs(trade_abs - old_size) < 1e-12:
                side_label = "long" if current_size > 0 else "short"
                segments.append({
                    'start_time': segment_start_time,
                    'end_time': order.fill_time,
                    'price': current_price,
                    'side': side_label,
                    'size': abs(current_size),
                })
                current_size = 0.0
                current_price = 0.0
                # новый сегмент ещё не начинается, ждём следующую сделку
                continue

            # Переворот: сначала закрываем старую, потом открываем новую в другую сторону
            remaining = trade_abs - old_size

            # сегмент старой позиции
            side_label = "long" if current_size > 0 else "short"
            segments.append({
                'start_time': segment_start_time,
                'end_time': order.fill_time,
                'price': current_price,
                'side': side_label,
                'size': abs(current_size),
            })

            # новая позиция в обратную сторону
            current_size = math.copysign(remaining, trade_size)
            current_price = order.fill_price
            segment_start_time = order.fill_time

        # Хвост: позиция осталась открытой до close_time или конца теста
        if abs(current_size) > 1e-12:
            end_time = position.close_time if position.status == "closed" else self.engine.last_event_time
            side_label = "long" if current_size > 0 else "short"
            segments.append({
                'start_time': segment_start_time,
                'end_time': end_time,
                'price': current_price,
                'side': side_label,
                'size': abs(current_size),
            })

        return segments
    
    def _build_equity_curve(self):
        """
        Строим equity curve.
        """
        initial_balance = self.engine.strategy.initial_balance if self.engine.strategy else 0

        filled_orders = [o for o in self.engine.orders if o.status == "filled"]
        filled_orders.sort(key=lambda o: o.fill_time)

        if not filled_orders:
            return pd.DataFrame()

        points = []
        price_cache = self._build_price_cache()

        for order in filled_orders:
            timestamp = order.fill_time

            closed_pnl = sum(
                p.net_pnl for p in self.engine.positions
                if p.status == "closed" and p.close_time <= timestamp
            )

            open_position = None
            for pos in self.engine.positions:
                if pos.status == "open" or (pos.status == "closed" and pos.close_time > timestamp):
                    if pos.open_time <= timestamp:
                        open_position = pos
                        break

            unrealized_pnl = 0.0
            if open_position and open_position.size != 0:
                bid, ask = price_cache.get(timestamp, (None, None))

                if bid and ask:
                    if open_position.size > 0:
                        mark_price = bid
                        unrealized_gross = (mark_price - open_position.price) * open_position.size
                    else:
                        mark_price = ask
                        unrealized_gross = (open_position.price - mark_price) * abs(open_position.size)

                    unrealized_pnl = unrealized_gross - open_position.fees

            equity = initial_balance + closed_pnl + unrealized_pnl

            points.append({
                'time': pd.to_datetime(timestamp, unit='ms'),
                'equity': equity,
            })

        return pd.DataFrame(points)

    def _build_price_cache(self):
        """
        Строим кэш bid/ask цен.
        """
        cache = {}

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
        """Статистика."""
        filled = [o for o in self.engine.orders if o.status == "filled"]
        buys = [o for o in filled if o.size > 0]
        sells = [o for o in filled if o.size < 0]

        buy_volume_usd = sum(abs(o.size) * o.fill_price for o in buys)
        sell_volume_usd = sum(abs(o.size) * o.fill_price for o in sells)
        total_volume_usd = buy_volume_usd + sell_volume_usd

        realized_pnl = self.engine.get_realized_pnl()
        total_fees = self.engine.get_total_fees()
        net_pnl_closed = self.engine.get_net_pnl()

        open_positions = [p for p in self.engine.positions if p.status == "open"]

        if open_positions:
            open_pos = open_positions[0]
            unrealized_pnl = self.engine.get_unrealized_pnl()
            unrealized_net = unrealized_pnl - open_pos.fees
        else:
            unrealized_pnl = 0.0
            unrealized_net = 0.0

        initial_balance = self.engine.strategy.initial_balance if self.engine.strategy else 0
        final_equity = initial_balance + net_pnl_closed + unrealized_net

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