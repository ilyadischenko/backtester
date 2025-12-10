import polars as pl
from engine import ExchangeEngine
from strategies.test_strategy import  MarketMakerStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.spread_capture import SpreadCaptureStrategy
from vizualization import BacktestVisualizer

# Твой data manager
from data.data_manager import dataManager




# ═══════════════════════════════════════════════════════════
# Загрузка данных
# ═══════════════════════════════════════════════════════════
df = dataManager.load_timerange(
    exchange="binance",
    symbol="fheusdt",
    start_time="2025-12-10 10:00",
    end_time="2025-12-10 13:00",  # 2 часа для теста
    data_type="all",
    market_type="futures"
)


def main():
    # Загружаем данные
    bookticker = df.orderbook
    trades = df.trades

    # strategy = MarketMakerStrategy(
    #     initial_balance=1000.0,
    #     order_size_usd=10.0,
    #     spread_bps=5.0,
    #     max_position_usd=500.0,
    #     refresh_interval_ms=1000,
    # )

    strategy = MeanReversionStrategy(
    )


    engine = ExchangeEngine(
        data_bookticker=bookticker,
        data_trades=trades,
        strategy=strategy,
        taker_fee=0.0004,
        maker_fee=0.0002,
        network_delay=2

    )

    # Запускаем бэктест
    print("Running backtest...")
    engine.run()


    # Визуализируем
    viz = BacktestVisualizer(engine)
    viz.show("My Backtest")




    # print(df.orderbook)
    # bookticker = df.orderbook
    #
    # # Создаём движок (без стратегии - просто для визуализации)
    # engine = ExchangeEngine(data_bookticker=bookticker)
    #
    # # Показываем график
    # viz = BacktestVisualizer(engine)
    # viz.show()


if __name__ == "__main__":
    main()

