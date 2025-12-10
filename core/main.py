# main.py

import polars as pl
from engine import ExchangeEngine
from strategies.test_strategy import  SimpleStrategy
from vizualization import BacktestVisualizer
import matplotlib.pyplot as plt

# Твой data manager
from data.data_manager import dataManager


# ═══════════════════════════════════════════════════════════
# Загрузка данных
# ═══════════════════════════════════════════════════════════
df = dataManager.load_timerange(
    exchange="binance",
    symbol="luna2usdt",
    start_time="2025-12-07 10:00",
    end_time="2025-12-07 11:00",  # 2 часа для теста
    data_type="all",
    market_type="futures"
)
def main():
    # Загружаем данные
    # bookticker = df.orderbook
    # trades = df.trades
    
    # # Создаём стратегию и движок
    # strategy = SimpleStrategy()
    # engine = ExchangeEngine(
    #     data_bookticker=bookticker,
    #     data_trades=trades,
    #     strategy=strategy,
    # )
    
    # # Запускаем бэктест
    # print("Running backtest...")
    # engine.run()
    
    # # Визуализируем
    # viz = BacktestVisualizer(engine)
    # viz.show("My Backtest")
        # Загружаем данные
    print(df.orderbook)
    bookticker = df.orderbook
    
    # Создаём движок (без стратегии - просто для визуализации)
    engine = ExchangeEngine(data_bookticker=bookticker)
    
    # Показываем график
    viz = BacktestVisualizer(engine)
    viz.show()


if __name__ == "__main__":
    main()