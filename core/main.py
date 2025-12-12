
import cProfile
import pstats
import time

from core.fast_iterator import run_fast

profiler = cProfile.Profile()
profiler.enable()


from core.engine import ExchangeEngine
# from strategies.test_strategy import  MarketMakerStrategy
from strategies.knife_catcher import KnifeCatcherUltraFast
from core.visualization import BacktestVisualizer

# Твой data manager
from data.data_manager import dataManager




# ═══════════════════════════════════════════════════════════
# Загрузка данных
# ═══════════════════════════════════════════════════════════
df = dataManager.load_timerange(
    exchange="binance",
    symbol="pippinusdt",
    start_time="2025-12-10 10:00",
    end_time="2025-12-10 15:00",  # 2 часа для теста
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

    strategy = KnifeCatcherUltraFast(
        # initial_balance  = 10000.0,

        market_order_size_usd  = 100.0,  # Маркет при входе
        limit_order_1_size_usd  = 100.0,  # Первая лимитка
        limit_order_2_size_usd = 100.0,  # Вторая лимитка

        volatility_window_sec = 30.0,  # Окно расчёта текущей волатильности
        volatility_spike_pct = 150.0,  # Порог: волатильность > средней на X%
        volatility_lookback_sec = 300.0,  # Период для средней волатильности

        imbalance_window_sec = 5.0,  # Окно расчёта дисбаланса
        imbalance_sell_threshold = 0.65,  # Порог: sells > 65% = нож
        imbalance_neutral_threshold = 0.55,  # Порог: sells < 55% = затихание


        calm_window_sec = 2.0,  # Окно для определения затихания
        calm_price_change_bps = 5.0,  # Макс изменение цены для "спокойствия"
        min_knife_duration_sec = 1.0,  # Мин длительность ножа
        max_knife_duration_sec = 60.0,  # Макс ожидание затихания

        limit_order_1_offset_bps = 20.0,  # Первая лимитка ниже на X bps
        limit_order_2_offset_bps = 40.0,  # Вторая лимитка ниже на X bps
        stop_loss_bps = 60.0,  # Стоп ниже средней на X bps
        take_profit_bps = 30.0,  # Тейк выше средней на X bps

        cooldown_after_close_sec = 10.0,  # Пауза после закрытия
        min_data_points = 100,  # Мин точек для старта
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
    # СТАЛО (быстро):
    t0 = time.time()
    run_fast(engine)
    print(f"\nTime: {time.time() - t0:.2f}s")
    # ═══════════════════════════════════════════════════════════════

    # Результаты
    print(f"Net PnL: ${engine.get_net_pnl():,.2f}")
    # Визуализируем
    viz = BacktestVisualizer(engine)
    viz.show("My Backtest")


    # print("\n" + "="*70)
    # print("DETAILED POSITIONS:")
    # print("="*70)
    #
    # for i, pos in enumerate(engine.positions, 1):
    #     print(f"\n--- Position {i} (ID: {pos.id}) ---")
    #     print(f"Status: {pos.status}")
    #     print(f"Open time: {pos.open_time}")
    #     print(f"Close time: {pos.close_time}")
    #     print(f"Entry price: {pos.price:.5f}")
    #     print(f"Size: {pos.size:.8f}")
    #     print(f"Realized PnL (gross): ${pos.realized_pnl:.2f}")
    #     print(f"Fees: ${pos.fees:.2f}")
    #     print(f"Net PnL: ${pos.net_pnl:.2f}")
    #     print(f"Orders in position: {len(pos.order_ids)}")
    #
    #     # Детали ордеров
    #     print("\nOrders:")
    #     for order_id in pos.order_ids:
    #         order = next((o for o in engine.orders if o.id == order_id), None)
    #         if order:
    #             print(f"  - {order.type} | size: {order.size:+.8f} | "
    #                 f"price: {order.fill_price:.5f} | "
    #                 f"time: {order.fill_time}")


if __name__ == "__main__":
    main()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Топ-20 медленных функций

