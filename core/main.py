
# import cProfile
import pstats
import time

# profiler = cProfile.Profile()
# profiler.enable()


from engine import ExchangeEngine
# from strategies.test_strategy import  MarketMakerStrategy
# from strategies.knife_catcher import KnifeCatcherUltraFast
# from strategies.mean_reversion import BollingerBandsStrategy
# from strategies.test_strategy import AvellanedaStoikovMMSlope
# from strategies.stoikov import HFTMarketMaker
# from strategies.nn import MLStrategy
# from strategies.channel import ChannelStrategy
# from strategies.clusters import ClusterMeanReversionStrategy
# from strategies.obImbalance import OrderflowImbalanceStrategy
from strategies.frontrunning import FrontrunStrategy



from visualization.visualization import BacktestVisualizer

# Твой data manager
from data.data_manager import dataManager




# ═══════════════════════════════════════════════════════════
# Загрузка данных
# ═══════════════════════════════════════════════════════════
df = dataManager.load_timerange(
    exchange="binance",
    symbol="sirenusdt",
    start_time="2026-03-29 00:00:00",
    end_time="2026-03-29 18:00:00",
    data_type="all",
    market_type="futures",
)


def main():
    # Загружаем данные
    # bookticker = df.orderbook
    # trades = df.trades

    strategy = FrontrunStrategy(
        tick_size=0.001,
        take_pct=0.5,
        order_size=10.0
    )

    # strategy = MLStrategy(
    #     confidence_threshold=0.7,
    #     model_path="models/pippinusdt_100ms_20251221_201102",
    # )

    # strategy = KnifeCatcherUltraFast(
    #     # initial_balance  = 10000.0,

    #     market_order_size_usd  = 100.0,  # Маркет при входе
    #     limit_order_1_size_usd  = 100.0,  # Первая лимитка
    #     limit_order_2_size_usd = 100.0,  # Вторая лимитка

    #     volatility_window_sec = 30.0,  # Окно расчёта текущей волатильности
    #     volatility_spike_pct = 150.0,  # Порог: волатильность > средней на X%
    #     volatility_lookback_sec = 300.0,  # Период для средней волатильности

    #     imbalance_window_sec = 5.0,  # Окно расчёта дисбаланса
    #     imbalance_sell_threshold = 0.65,  # Порог: sells > 65% = нож
    #     imbalance_neutral_threshold = 0.55,  # Порог: sells < 55% = затихание


    #     calm_window_sec = 2.0,  # Окно для определения затихания
    #     calm_price_change_bps = 5.0,  # Макс изменение цены для "спокойствия"
    #     min_knife_duration_sec = 1.0,  # Мин длительность ножа
    #     max_knife_duration_sec = 60.0,  # Макс ожидание затихания

    #     limit_order_1_offset_bps = 20.0,  # Первая лимитка ниже на X bps
    #     limit_order_2_offset_bps = 40.0,  # Вторая лимитка ниже на X bps
    #     stop_loss_bps = 60.0,  # Стоп ниже средней на X bps
    #     take_profit_bps = 30.0,  # Тейк выше средней на X bps

    #     cooldown_after_close_sec = 10.0,  # Пауза после закрытия
    #     min_data_points = 100,  # Мин точек для старта
    # )


    engine = ExchangeEngine(
        data_trades=df.trades,
        data_depth=df.depth,
        data_ob_snapshot=df.ob_snapshot,
        strategy=strategy,
        taker_fee=0.0004,
        maker_fee=0.0002,
        network_delay=2,
    )

    # Запускаем бэктест
    print("Running backtest...")
    # СТАЛО (быстро):
    t0 = time.time()

    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        engine.run()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # топ 20 функций
    print(f"\nTime: {time.time() - t0:.2f}s")
    # ═══════════════════════════════════════════════════════════════

    # Результаты
    print(f"Net PnL: ${engine.get_net_pnl():,.2f}")
    # Визуализируем
    viz = BacktestVisualizer(engine, strategy=strategy)
    viz.show("My Backtest")
    print(len(engine.orders))
    
    
    # val_pnl = engine.validate_pnl()
    # print(f"Validation PnL: ${val_pnl}")


    # positions_df = engine.positions_to_dataframe()
    # print(positions_df)



if __name__ == "__main__":
    main()

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative')
    # stats.print_stats(20)  # Топ-20 медленных функций

