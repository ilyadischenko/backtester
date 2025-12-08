import polars as pl
import matplotlib.pyplot as plt


def plot_results(engine: "ExchangeEngine"):
    # 1. PnL история
    if not engine.position_history:
        print("Нет истории позиции")
        return

    pnl_df = pl.DataFrame(engine.position_history)

    # если нет realized_pnl/fees в history — нужно добавить их в _apply_fill;
    # ниже предполагаем, что ты их туда уже пишешь, как в предыдущем коде:
    # {"time": ..., "size": ..., "avg_price": ..., "realized_pnl": ..., "fees": ...}

    pnl_df = pnl_df.sort("time")
    pnl_df = pnl_df.with_columns(
        (pl.col("realized_pnl") - pl.col("fees")).alias("equity")
    )

    # 2. mid‑price из bookticker
    bt = engine.bookticker.select(
        pl.col("event_time"),
        ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid"),
    ).sort("event_time")

    # 3. ордера для входов/выходов
    orders = [o for o in engine.orders if o.status == "filled"]
    if orders:
        orders_df = pl.DataFrame([{
            "time": o.added_time,
            "side": o.side,
            "price": o.price,   # у нас price=0 для market, поэтому лучше взять из близкого mid
            "size": o.size,
        } for o in orders]).sort("time")
        # подставим рыночную цену из mid
        orders_df = orders_df.join(
            bt.rename({"event_time": "time"}),
            on="time",
            how="left",
        ).with_columns(
            pl.when(pl.col("price") == 0)
              .then(pl.col("mid"))
              .otherwise(pl.col("price"))
              .alias("exec_price")
        )
    else:
        orders_df = None

    # ---------- Рисуем ----------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # График equity
    ax1 = axes[0]
    ax1.plot(pnl_df["time"], pnl_df["equity"], label="Equity")
    ax1.set_ylabel("Equity")
    ax1.grid(True)
    ax1.legend()

    # График цены + ордера
    ax2 = axes[1]
    ax2.plot(bt["event_time"], bt["mid"], label="Mid price", color="black", linewidth=1)

    if orders_df is not None and len(orders_df) > 0:
        buys = orders_df.filter(pl.col("side") == "buy")
        sells = orders_df.filter(pl.col("side") == "sell")

        if len(buys) > 0:
            ax2.scatter(
                buys["time"],
                buys["exec_price"],
                color="green",
                marker="^",
                label="Buys",
                alpha=0.8,
            )
        if len(sells) > 0:
            ax2.scatter(
                sells["time"],
                sells["exec_price"],
                color="red",
                marker="v",
                label="Sells",
                alpha=0.8,
            )

    ax2.set_ylabel("Price")
    ax2.set_xlabel("Time")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


from data_manager import dataManager
from engineee import ExchangeEngine  # твой файл с классом
from strategies.test_strategy import TradeKnifeStrategy


# 1. загружаем данные
df = dataManager.load_timerange(
    exchange="binance",
    symbol="luna2usdt",
    start_time="2025-12-07 00:00:00",
    end_time="2025-12-07 23:00:00",
    data_type="all",
    market_type="futures",
)
trades = df.trades
book = df.orderbook

print(f"orderbook={len(book)} rows, trades={len(trades)} rows")

# 2. создаём стратегию
strategy = TradeKnifeStrategy(
    window_sec=2,
    min_sell_ratio=2.0,    # продажи > 2x покупок
    min_drop_pct=-0.005,   # падение цены > 0.5% за окно
    usd_step=100.0,        # заходим по 100$
    max_usd_position=1000.0,
    stop_loss_pct=0.01,   
)

# 3. создаём и запускаем движок
engine = ExchangeEngine(
    data_bookticker=book,
    data_trades=trades,
    strategy=strategy,
    market_fee=0.04,
    limit_fee=0.02,
)

engine.run()

# 4. выводим итоговые метрики
print("Net position size:", engine.net_position.size)
print("Net position avg_price:", engine.net_position.avg_price)
print("Realized PnL:", engine.net_position.realized_pnl)
print("Fees:", engine.net_position.fees)
print("Всего ордеров:", len(engine.orders))

# 5. рисуем графики
plot_results(engine)