import polars as pl
import plotly.graph_objects as go
import numpy as np
from data_manager import dataManager

# 1. Допустим, df_pl - это твой Polars DataFrame
# (Этот блок для примера, удали его и используй свой df)

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

# 1. Переименуем колонки в осмысленные
trades = trades.rename({
    "column_1": "event_time_ms",
    "column_2": "agg_id",
    "column_3": "price",
    "column_4": "qty",
    "column_5": "trade_time_ms",
    "column_6": "is_buyer_maker",  # 1/0
})

book = book.rename({
    "column_1": "event_time_ms",
    "column_2": "update_id",
    "column_3": "best_bid",
    "column_4": "best_bid_qty",
    "column_5": "best_ask",
    "column_6": "best_ask_qty",
})

# 2. Переводим время в datetime (для нормальной оси X)
trades = trades.with_columns(
    pl.from_epoch("event_time_ms", time_unit="ms").alias("time")
).sort("time")

book = book.with_columns(
    pl.from_epoch("event_time_ms", time_unit="ms").alias("time")
).sort("time")


def show_time_range(df: pl.DataFrame, name: str, ts_col: str = "column_1"):
    out = df.select([
        pl.col(ts_col).min().alias("first_ms"),
        pl.col(ts_col).max().alias("last_ms"),
        pl.col(ts_col).cast(pl.Datetime("ms")).min().alias("first_dt"),
        pl.col(ts_col).cast(pl.Datetime("ms")).max().alias("last_dt"),
    ])
    print(f"{name}:\n{out}\n")

# если уже делал rename:
ts_col_trades = "event_time_ms"
ts_col_book   = "event_time_ms"
# ts_col_trades = "column_1"
# ts_col_book   = "column_1"

show_time_range(trades,    "TRADES",    ts_col_trades)
show_time_range(book, "ORDERBOOK", ts_col_book)

# (опционально) ограничиваемся куском по времени, чтобы не грузить браузер
# start = trades["time"][0]
# end = start + pl.duration(minutes=5)
# trades = trades.filter(pl.col("time").is_between(start, end))
# book = book.filter(pl.col("time").is_between(start, end))

# 3. Вытаскиваем данные в numpy (Plotly ест list/ndarray)
t_time = trades["time"].to_numpy()
t_price = trades["price"].to_numpy()
t_qty = trades["qty"].to_numpy()
t_side = trades["is_buyer_maker"].to_numpy().astype(int)

b_time  = book["time"].to_numpy()
b_bid   = book["best_bid"].to_numpy()
b_ask   = book["best_ask"].to_numpy()

# Маски для покупок/продаж
# В Binance m = true → buyer is maker → сделка по bid (то есть агрессор = продавец).
# Если хочешь «покупки» зелёным, можешь развернуть логику под себя.
sell_mask = (t_side == 1)   # maker = buyer → агрессивный продавец
buy_mask  = (t_side == 0)   # maker = seller → агрессивный покупатель

# Размер маркера от объёма (чуть-чуть масштабирую и ограничиваю)
vol_scale = max(t_qty.max(), 1) / 10
marker_size = np.clip(t_qty / vol_scale, 4, 18)

# 4. Строим фигуру
fig = go.Figure()

# Линии best bid / best ask (буктикер)
fig.add_trace(go.Scattergl(
    x=b_time,
    y=b_bid,
    mode="lines",
    name="Best bid",
    line=dict(color="green", width=1)
))

fig.add_trace(go.Scattergl(
    x=b_time,
    y=b_ask,
    mode="lines",
    name="Best ask",
    line=dict(color="red", width=1)
))

# Трейды — покупатель агрессор (buy) треугольник вверх
fig.add_trace(go.Scattergl(
    x=t_time[buy_mask],
    y=t_price[buy_mask],
    mode="markers",
    name="Aggressive buys",
    marker=dict(
        symbol="triangle-up",
        color="lime",
        size=marker_size[buy_mask],
        line=dict(width=1, color="darkgreen"),
        opacity=0.9,
    ),
    hovertemplate=(
        "time=%{x}<br>"
        "price=%{y}<br>"
        "qty=%{customdata}<extra>buy</extra>"
    ),
    customdata=t_qty[buy_mask],
))

# Трейды — продавец агрессор (sell) треугольник вниз
fig.add_trace(go.Scattergl(
    x=t_time[sell_mask],
    y=t_price[sell_mask],
    mode="markers",
    name="Aggressive sells",
    marker=dict(
        symbol="triangle-down",
        color="orange",
        size=marker_size[sell_mask],
        line=dict(width=1, color="darkred"),
        opacity=0.9,
    ),
    hovertemplate=(
        "time=%{x}<br>"
        "price=%{y}<br>"
        "qty=%{customdata}<extra>sell</extra>"
    ),
    customdata=t_qty[sell_mask],
))

fig.update_layout(
    template="plotly_dark",
    xaxis=dict(title="Time"),
    yaxis=dict(title="Price"),
    legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
    margin=dict(l=60, r=20, t=40, b=40),
)

fig.show()
# или так, если запускаешь как скрипт:
# fig.write_html("ticks.html", auto_open=True)