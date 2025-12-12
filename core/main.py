import polars as pl
from engine import ExchangeEngine
# from strategies.test_strategy import  MarketMakerStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.spread_capture import SpreadCaptureStrategy
from strategies.test_strategy import GridAveragingStrategy
from core.visualization import BacktestVisualizer

# Твой data manager
from data.data_manager import dataManager




# ═══════════════════════════════════════════════════════════
# Загрузка данных
# ═══════════════════════════════════════════════════════════
df = dataManager.load_timerange(
    exchange="binance",
    symbol="fheusdt",
    start_time="2025-12-10 10:00",
    end_time="2025-12-10 10:59",  # 2 часа для теста
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

    strategy = GridAveragingStrategy(
        initial_balance=10000.0,
        # order_size_usd=1000.0,
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


    # После engine.run()

    print("\n" + "="*70)
    print("DETAILED POSITIONS:")
    print("="*70)

    for i, pos in enumerate(engine.positions, 1):
        print(f"\n--- Position {i} (ID: {pos.id}) ---")
        print(f"Status: {pos.status}")
        print(f"Open time: {pos.open_time}")
        print(f"Close time: {pos.close_time}")
        print(f"Entry price: {pos.price:.5f}")
        print(f"Size: {pos.size:.8f}")
        print(f"Realized PnL (gross): ${pos.realized_pnl:.2f}")
        print(f"Fees: ${pos.fees:.2f}")
        print(f"Net PnL: ${pos.net_pnl:.2f}")
        print(f"Orders in position: {len(pos.order_ids)}")
        
        # Детали ордеров
        print("\nOrders:")
        for order_id in pos.order_ids:
            order = next((o for o in engine.orders if o.id == order_id), None)
            if order:
                print(f"  - {order.type} | size: {order.size:+.8f} | "
                    f"price: {order.fill_price:.5f} | "
                    f"time: {order.fill_time}")


if __name__ == "__main__":
    main()

