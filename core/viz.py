import polars as pl
import pandas as pd
import mplfinance as mpf
from data_manager import dataManager

# 1. Допустим, df_pl - это твой Polars DataFrame
# (Этот блок для примера, удали его и используй свой df)

df_trades = dataManager.load_timerange(
        exchange="binance",
        symbol="cvcusdt",
        start_time="2025-12-05 12:00:00",
        end_time="2025-12-05 12:00:00",
        data_type="all",
        market_type="futures"
    )
print(f"\nTrades only: {df_trades.trades}")
df_pl = pl.DataFrame(df_trades.trades)

# -------------------------------------------------------------
# ЛОГИКА ОБРАБОТКИ (POLARS)
# -------------------------------------------------------------

# 2. Подготовка данных
# Преобразуем int64 (мс) в datetime и сразу сортируем
q = (
    df_pl
    .lazy()
    .with_columns(
        pl.from_epoch(pl.col("column_1"), time_unit="ms").alias("datetime")
    )
    .sort("datetime")
)

# 3. Агрегация в свечи (OHLCV) с помощью group_by_dynamic
# '1m' означает интервал 1 минута.
candles_pl = (
    q.group_by_dynamic("datetime", every="1s")
    .agg([
        pl.col("column_3").first().alias("Open"),
        pl.col("column_3").max().alias("High"),
        pl.col("column_3").min().alias("Low"),
        pl.col("column_3").last().alias("Close"),
        pl.col("column_4").sum().alias("Volume"),
    ])
    .collect() # Выполняем вычисления
)

# -------------------------------------------------------------
# ВИЗУАЛИЗАЦИЯ (MPLFINANCE)
# -------------------------------------------------------------

# 4. Конвертация в Pandas (только итоговых свечей, это быстро)
df_pandas = candles_pl.to_pandas()

# Устанавливаем дату индексом (требование mplfinance)
df_pandas.set_index("datetime", inplace=True)

print("Сформированные свечи:")
print(df_pandas.head())

# 5. Рисуем график
mpf.plot(
    df_pandas,
    type='candle',   # тип графика: свечи
    volume=True,     # показать объемы
    style='binance', # стиль
    title='1-Minute Chart from Polars',
    warn_too_much_data=10000 # отключить предупреждение, если свечей много
)