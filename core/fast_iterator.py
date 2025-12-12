# fast_runner.py
"""
Быстрый runner для ExchangeEngine.
Заменяет медленный to_dicts() на numpy arrays.
"""
import numpy as np
import polars as pl
from engine import ExchangeEngine


class FastEventIterator:
    """
    Быстрый итератор событий.
    Извлекаем данные как numpy arrays (почти zero-copy).
    """

    def __init__(self, events_df: pl.DataFrame):
        self.n_events = len(events_df)

        # ═══════════════════════════════════════════════════════
        # Извлекаем колонки как numpy arrays
        # ═══════════════════════════════════════════════════════

        # Общие поля
        self.event_times = events_df["event_time"].to_numpy()

        # Тип события: True = bookticker, False = trade
        event_types = events_df["event_type"].to_numpy()
        self.is_bookticker = np.array([t == "bookticker" for t in event_types])

        # Bookticker поля
        self.bid_prices = self._get_column(events_df, "bid_price")
        self.ask_prices = self._get_column(events_df, "ask_price")
        self.bid_sizes = self._get_column(events_df, "bid_size")
        self.ask_sizes = self._get_column(events_df, "ask_size")

        # Trade поля
        self.trade_prices = self._get_column(events_df, "price")
        self.trade_quantities = self._get_column(events_df, "quantity")
        self.is_makers = self._get_column_bool(events_df, "is_maker")

    def _get_column(self, df: pl.DataFrame, col: str) -> np.ndarray:
        """Безопасное извлечение float колонки."""
        if col not in df.columns:
            return np.zeros(len(df), dtype=np.float64)
        return df[col].fill_null(0).to_numpy().astype(np.float64)

    def _get_column_bool(self, df: pl.DataFrame, col: str) -> np.ndarray:
        """Безопасное извлечение bool колонки."""
        if col not in df.columns:
            return np.zeros(len(df), dtype=bool)
        return df[col].fill_null(False).to_numpy()


def run_fast(engine: ExchangeEngine, show_progress: bool = True):
    """
    Быстрая версия engine.run().

    Использование:
        from fast_runner import run_fast

        engine = ExchangeEngine(...)
        run_fast(engine)  # Вместо engine.run()
    """
    # Создаём быстрый итератор
    fast = FastEventIterator(engine.events)
    n = fast.n_events

    if show_progress:
        print(f"⚡ Fast runner: {n:,} events")

    # ═══════════════════════════════════════════════════════════
    # Локальные ссылки для скорости (избегаем self.xxx lookups)
    # ═══════════════════════════════════════════════════════════
    event_times = fast.event_times
    is_bookticker = fast.is_bookticker
    bid_prices = fast.bid_prices
    ask_prices = fast.ask_prices
    bid_sizes = fast.bid_sizes
    ask_sizes = fast.ask_sizes
    trade_prices = fast.trade_prices
    trade_quantities = fast.trade_quantities
    is_makers = fast.is_makers

    strategy = engine.strategy
    process_requests = engine._process_requests
    process_orders = engine._process_orders

    # Прогресс
    progress_step = max(1, n // 100)

    # ═══════════════════════════════════════════════════════════
    # Главный цикл
    # ═══════════════════════════════════════════════════════════
    for i in range(n):
        # Обновляем время
        current_time = int(event_times[i])
        engine.last_event_time = current_time

        # Формируем событие
        if is_bookticker[i]:
            event = {
                "event_type": "bookticker",
                "event_time": current_time,
                "bid_price": bid_prices[i],
                "ask_price": ask_prices[i],
                "bid_size": bid_sizes[i],
                "ask_size": ask_sizes[i],
            }
            engine.last_bookticker = event
        else:
            event = {
                "event_type": "trade",
                "event_time": current_time,
                "price": trade_prices[i],
                "quantity": trade_quantities[i],
                "is_maker": bool(is_makers[i]),
            }
            engine.last_trade = event

        # Обрабатываем
        process_requests()
        process_orders()

        # Стратегия
        if strategy:
            strategy.on_tick(event, engine)

        # Прогресс
        if show_progress and i % progress_step == 0:
            pct = (i + 1) / n * 100
            print(f"\r   Processing: {pct:5.1f}%", end="", flush=True)

    if show_progress:
        print(f"\r✅ Completed: {n:,} events")