# data_integrity_checker.py

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Patch
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
import numpy as np

from data_manager import dataManager, MarketData


@dataclass
class IntegrityReport:
    """Отчёт о целостности данных"""
    symbol: str
    start_time: datetime
    end_time: datetime
    
    # Orderbook
    ob_total_rows: int = 0
    ob_expected_hours: int = 0
    ob_actual_hours: int = 0
    ob_gaps: list = None
    ob_max_gap_ms: int = 0
    ob_avg_update_ms: float = 0
    
    # Trades
    tr_total_rows: int = 0
    tr_expected_hours: int = 0
    tr_actual_hours: int = 0
    tr_gaps: list = None
    tr_max_gap_ms: int = 0
    
    # Аномалии
    ob_price_anomalies: int = 0
    tr_price_anomalies: int = 0
    
    def __post_init__(self):
        if self.ob_gaps is None:
            self.ob_gaps = []
        if self.tr_gaps is None:
            self.tr_gaps = []
    
    def print_report(self):
        print("\n" + "=" * 70)
        print(f"📊 ОТЧЁТ О ЦЕЛОСТНОСТИ ДАННЫХ: {self.symbol.upper()}")
        print(f"📅 Период: {self.start_time} → {self.end_time}")
        print("=" * 70)
        
        # Orderbook
        print("\n📗 ORDERBOOK (Best Bid/Ask)")
        print("-" * 40)
        print(f"   Строк: {self.ob_total_rows:,}")
        print(f"   Часов: {self.ob_actual_hours}/{self.ob_expected_hours}")
        print(f"   Средний интервал: {self.ob_avg_update_ms:.1f} ms")
        print(f"   Макс. пропуск: {self.ob_max_gap_ms:,} ms ({self.ob_max_gap_ms/1000:.1f} сек)")
        print(f"   Пропусков >1сек: {len([g for g in self.ob_gaps if g['duration_ms'] > 1000])}")
        print(f"   Аномалий цены: {self.ob_price_anomalies}")
        
        # Trades
        print("\n📕 TRADES")
        print("-" * 40)
        print(f"   Строк: {self.tr_total_rows:,}")
        print(f"   Часов: {self.tr_actual_hours}/{self.tr_expected_hours}")
        print(f"   Макс. пропуск: {self.tr_max_gap_ms:,} ms ({self.tr_max_gap_ms/1000:.1f} сек)")
        print(f"   Пропусков >5сек: {len([g for g in self.tr_gaps if g['duration_ms'] > 5000])}")
        print(f"   Аномалий цены: {self.tr_price_anomalies}")
        
        # Итог
        print("\n" + "=" * 70)
        is_ok = (
            self.ob_actual_hours == self.ob_expected_hours and
            self.tr_actual_hours == self.tr_expected_hours and
            self.ob_max_gap_ms < 60000 and
            self.ob_price_anomalies == 0
        )
        
        if is_ok:
            print("✅ ДАННЫЕ В ПОРЯДКЕ")
        else:
            print("⚠️  ОБНАРУЖЕНЫ ПРОБЛЕМЫ:")
            if self.ob_actual_hours < self.ob_expected_hours:
                print(f"   - Не хватает {self.ob_expected_hours - self.ob_actual_hours} часов orderbook")
            if self.tr_actual_hours < self.tr_expected_hours:
                print(f"   - Не хватает {self.tr_expected_hours - self.tr_actual_hours} часов trades")
            if self.ob_max_gap_ms >= 60000:
                print(f"   - Пропуск в orderbook: {self.ob_max_gap_ms/1000:.0f} сек")
            if self.ob_price_anomalies > 0:
                print(f"   - Аномалии в ценах: {self.ob_price_anomalies}")
        
        print("=" * 70 + "\n")


class DataIntegrityChecker:
    """Проверка целостности рыночных данных"""
    
    def __init__(
        self,
        gap_threshold_ob_ms: int = 1000,
        gap_threshold_tr_ms: int = 5000,
        max_spread_pct: float = 0.05,
    ):
        self.gap_threshold_ob_ms = gap_threshold_ob_ms
        self.gap_threshold_tr_ms = gap_threshold_tr_ms
        self.max_spread_pct = max_spread_pct
    
    def check(
        self,
        exchange: str,
        symbol: str,
        start_time: datetime | str,
        end_time: datetime | str,
        market_type: Literal["futures", "spot"] = "futures",
    ) -> IntegrityReport:
        """Проверить целостность данных за период"""
        
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time)
        
        print(f"\n🔍 Проверка данных: {symbol.upper()} @ {exchange.upper()}")
        print(f"📅 {start_time} → {end_time}")
        
        # Загружаем данные
        try:
            data = dataManager.load_timerange(
                exchange=exchange,
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                data_type="all",
                market_type=market_type,
            )
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            raise
        
        # Создаём отчёт
        expected_hours = int((end_time - start_time).total_seconds() / 3600) + 1
        
        report = IntegrityReport(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            ob_expected_hours=expected_hours,
            tr_expected_hours=expected_hours,
        )
        
        # Проверяем orderbook
        if data.orderbook is not None and len(data.orderbook) > 0:
            self._check_orderbook(data.orderbook, report)
        else:
            print("⚠️  Orderbook данные отсутствуют!")
        
        # Проверяем trades
        if data.trades is not None and len(data.trades) > 0:
            self._check_trades(data.trades, report, data.orderbook)
        else:
            print("⚠️  Trades данные отсутствуют!")
        
        return report, data
    
    def _check_orderbook(self, df: pl.DataFrame, report: IntegrityReport):
        """Проверка orderbook данных"""
        # Колонки уже именованы в DataManager:
        # event_time, update_id, bid_price, bid_qty, ask_price, ask_qty
        
        report.ob_total_rows = len(df)
        
        # Сортируем по времени
        df = df.sort("event_time")
        
        # Уникальные часы
        df_with_hour = df.with_columns(
            (pl.col("event_time") // 3600000).alias("hour")
        )
        report.ob_actual_hours = df_with_hour.select("hour").n_unique()
        
        # Вычисляем gaps
        times = df["event_time"].to_numpy()
        if len(times) > 1:
            diffs = np.diff(times)
            report.ob_avg_update_ms = float(np.mean(diffs))
            report.ob_max_gap_ms = int(np.max(diffs))
            
            # Находим большие пропуски
            gap_indices = np.where(diffs > self.gap_threshold_ob_ms)[0]
            for idx in gap_indices:
                report.ob_gaps.append({
                    "start_time": int(times[idx]),
                    "end_time": int(times[idx + 1]),
                    "duration_ms": int(diffs[idx]),
                })
        
        # Проверка аномалий цен
        anomalies = df.filter(
            (pl.col("bid_price") >= pl.col("ask_price")) |
            (pl.col("bid_price") <= 0) |
            (pl.col("ask_price") <= 0) |
            ((pl.col("ask_price") - pl.col("bid_price")) / pl.col("bid_price") > self.max_spread_pct)
        )
        report.ob_price_anomalies = len(anomalies)
    
    def _check_trades(self, df: pl.DataFrame, report: IntegrityReport, ob_df: pl.DataFrame = None):
        """Проверка trades данных"""
        # Колонки уже именованы в DataManager:
        # event_time, trade_id, price, qty, trade_time, is_maker
        
        report.tr_total_rows = len(df)
        
        df = df.sort("event_time")
        
        # Уникальные часы
        df_with_hour = df.with_columns(
            (pl.col("event_time") // 3600000).alias("hour")
        )
        report.tr_actual_hours = df_with_hour.select("hour").n_unique()
        
        # Вычисляем gaps
        times = df["event_time"].to_numpy()
        if len(times) > 1:
            diffs = np.diff(times)
            report.tr_max_gap_ms = int(np.max(diffs))
            
            gap_indices = np.where(diffs > self.gap_threshold_tr_ms)[0]
            for idx in gap_indices:
                report.tr_gaps.append({
                    "start_time": int(times[idx]),
                    "end_time": int(times[idx + 1]),
                    "duration_ms": int(diffs[idx]),
                })
    
    def print_data_sample(
        self,
        data: MarketData,
        n_rows: int = 10,
        show_head: bool = True,
        show_tail: bool = True,
    ):
        """Вывод примеров данных в консоль"""
        
        print("\n" + "=" * 80)
        print("📋 ПРИМЕРЫ ДАННЫХ")
        print("=" * 80)
        
        # Orderbook
        if data.orderbook is not None and len(data.orderbook) > 0:
            print("\n📗 ORDERBOOK")
            print("-" * 80)
            print(f"Колонки: {data.orderbook.columns}")
            print(f"Типы: {data.orderbook.dtypes}")
            print()
            
            if show_head:
                print(f"Первые {n_rows} строк:")
                print(data.orderbook.head(n_rows))
            
            if show_tail:
                print(f"\nПоследние {n_rows} строк:")
                print(data.orderbook.tail(n_rows))
            
            # Статистика
            print("\nСтатистика:")
            print(data.orderbook.select([
                pl.col("bid_price").min().alias("bid_min"),
                pl.col("bid_price").max().alias("bid_max"),
                pl.col("ask_price").min().alias("ask_min"),
                pl.col("ask_price").max().alias("ask_max"),
            ]))
        else:
            print("\n⚠️ Orderbook: нет данных")
        
        # Trades
        if data.trades is not None and len(data.trades) > 0:
            print("\n📕 TRADES")
            print("-" * 80)
            print(f"Колонки: {data.trades.columns}")
            print(f"Типы: {data.trades.dtypes}")
            print()
            
            if show_head:
                print(f"Первые {n_rows} строк:")
                print(data.trades.head(n_rows))
            
            if show_tail:
                print(f"\nПоследние {n_rows} строк:")
                print(data.trades.tail(n_rows))
            
            # Статистика
            print("\nСтатистика:")
            print(data.trades.select([
                pl.col("price").min().alias("price_min"),
                pl.col("price").max().alias("price_max"),
                pl.col("price").mean().alias("price_avg"),
                pl.col("qty").sum().alias("total_qty"),
                pl.col("is_maker").sum().alias("maker_trades"),
                (pl.len() - pl.col("is_maker").sum()).alias("taker_trades"),
            ]))
        else:
            print("\n⚠️ Trades: нет данных")
        
        print("\n" + "=" * 80)
    
    def plot(
        self,
        data: MarketData,
        report: IntegrityReport,
        resample_ms: int = 1000,
        max_trades: int = 5000,
        figsize: tuple = (16, 10),
    ):
        """Визуализация данных"""
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, 
                                  gridspec_kw={'height_ratios': [3, 1, 1]})
        
        fig.suptitle(
            f"📊 Проверка данных: {report.symbol.upper()}\n"
            f"{report.start_time} → {report.end_time}",
            fontsize=12, fontweight='bold'
        )
        
        # ═══════════════════════════════════════════════════════════
        # График 1: Bid/Ask + Trades
        # ═══════════════════════════════════════════════════════════
        ax1 = axes[0]
        
        if data.orderbook is not None:
            ob = data.orderbook.sort("event_time")
            
            # Конвертируем в datetime
            times = [datetime.fromtimestamp(t / 1000) for t in ob["event_time"].to_list()]
            bids = ob["bid_price"].to_numpy()
            asks = ob["ask_price"].to_numpy()
            
            # Ресемплинг для скорости
            step = max(1, len(times) // 10000)
            times = times[::step]
            bids = bids[::step]
            asks = asks[::step]
            
            ax1.plot(times, bids, color='#4CAF50', linewidth=0.8, label='Bid', alpha=0.8)
            ax1.plot(times, asks, color='#F44336', linewidth=0.8, label='Ask', alpha=0.8)
            ax1.fill_between(times, bids, asks, color='gray', alpha=0.1)
        
        # Trades как точки
        if data.trades is not None:
            tr = data.trades.sort("event_time")
            
            # Ограничиваем количество
            if len(tr) > max_trades:
                step = len(tr) // max_trades
                tr = tr.gather_every(step)
            
            tr_times = [datetime.fromtimestamp(t / 1000) for t in tr["event_time"].to_list()]
            tr_prices = tr["price"].to_numpy()
            tr_is_maker = tr["is_maker"].to_numpy()
            
            # Buy = maker был продавец (is_maker=1 = buyer is maker)
            buy_mask = tr_is_maker.astype(bool)
            sell_mask = ~buy_mask
            
            ax1.scatter(
                [t for t, m in zip(tr_times, buy_mask) if m],
                tr_prices[buy_mask],
                color='#4CAF50', marker='^', s=15, alpha=0.6, label='Buy trades'
            )
            ax1.scatter(
                [t for t, m in zip(tr_times, sell_mask) if m],
                tr_prices[sell_mask],
                color='#F44336', marker='v', s=15, alpha=0.6, label='Sell trades'
            )
        
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Bid/Ask + Trades', fontsize=10)
        
        # ═══════════════════════════════════════════════════════════
        # График 2: Spread
        # ═══════════════════════════════════════════════════════════
        ax2 = axes[1]
        
        if data.orderbook is not None:
            ob = data.orderbook.sort("event_time")
            
            times = [datetime.fromtimestamp(t / 1000) for t in ob["event_time"].to_list()]
            spread_pct = ((ob["ask_price"] - ob["bid_price"]) / ob["bid_price"] * 100).to_numpy()
            
            step = max(1, len(times) // 10000)
            times = times[::step]
            spread_pct = spread_pct[::step]
            
            ax2.fill_between(times, 0, spread_pct, color='#2196F3', alpha=0.4)
            ax2.plot(times, spread_pct, color='#1976D2', linewidth=0.5)
            ax2.axhline(y=np.median(spread_pct), color='red', linestyle='--', 
                       linewidth=1, label=f'Median: {np.median(spread_pct):.4f}%')
        
        ax2.set_ylabel('Spread %')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Spread', fontsize=10)
        
        # ═══════════════════════════════════════════════════════════
        # График 3: Gaps (пропуски)
        # ═══════════════════════════════════════════════════════════
        ax3 = axes[2]
        
        # Orderbook gaps
        for gap in report.ob_gaps:
            if gap['duration_ms'] > self.gap_threshold_ob_ms:
                start = datetime.fromtimestamp(gap['start_time'] / 1000)
                end = datetime.fromtimestamp(gap['end_time'] / 1000)
                ax3.axvspan(start, end, color='#F44336', alpha=0.3)
        
        # Trades gaps
        for gap in report.tr_gaps:
            if gap['duration_ms'] > self.gap_threshold_tr_ms:
                start = datetime.fromtimestamp(gap['start_time'] / 1000)
                end = datetime.fromtimestamp(gap['end_time'] / 1000)
                ax3.axvspan(start, end, color='#FF9800', alpha=0.3)
        
        # Update frequency
        if data.orderbook is not None:
            ob = data.orderbook.sort("event_time")
            times_arr = ob["event_time"].to_numpy()
            
            if len(times_arr) > 1:
                # Группируем по секундам
                bucket_size = 1000
                buckets = times_arr[:-1] // bucket_size
                
                unique_buckets, counts = np.unique(buckets, return_counts=True)
                bucket_times = [datetime.fromtimestamp(b * bucket_size / 1000) for b in unique_buckets]
                
                ax3.bar(bucket_times, counts, width=0.0001, color='#2196F3', alpha=0.6)
        
        ax3.set_ylabel('Updates/sec')
        ax3.set_xlabel('Time')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Data Frequency (Red = OB gaps, Orange = Trade gaps)', fontsize=10)
        
        # Легенда для gaps
        legend_elements = [
            Patch(facecolor='#F44336', alpha=0.3, label=f'OB gap >{self.gap_threshold_ob_ms}ms'),
            Patch(facecolor='#FF9800', alpha=0.3, label=f'Trade gap >{self.gap_threshold_tr_ms}ms'),
        ]
        ax3.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # Форматирование осей
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig


# ═══════════════════════════════════════════════════════════════════
# CLI Interface
# ═══════════════════════════════════════════════════════════════════

def check_data(
    exchange: str,
    symbol: str,
    start_time: str,
    end_time: str,
    market_type: str = "futures",
    save_plot: str = None,
    show_plot: bool = True,
    show_table: bool = True,
    table_rows: int = 10,
):
    """
    Проверить целостность данных и показать визуализацию.
    
    Args:
        exchange: "binance", "bybit", "gate"
        symbol: "btcusdt", "ethusdt"
        start_time: "2025-01-07 00:00:00"
        end_time: "2025-01-07 02:00:00"
        market_type: "futures" или "spot"
        save_plot: путь для сохранения графика
        show_plot: показать график
        show_table: показать таблицу с данными
        table_rows: количество строк в таблице
    """
    checker = DataIntegrityChecker(
        gap_threshold_ob_ms=1000,
        gap_threshold_tr_ms=5000,
        max_spread_pct=0.05,
    )
    
    report, data = checker.check(
        exchange=exchange,
        symbol=symbol,
        start_time=start_time,
        end_time=end_time,
        market_type=market_type,
    )
    
    report.print_report()
    
    # Показать таблицу с данными
    if show_table:
        checker.print_data_sample(data, n_rows=table_rows)
    
    fig = checker.plot(data, report)
    
    if save_plot:
        fig.savefig(save_plot, dpi=150, bbox_inches='tight')
        print(f"📁 График сохранён: {save_plot}")
    
    if show_plot:
        plt.show()
    
    return report, data, fig


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    report, data, fig = check_data(
        exchange="gate",
        symbol="btcusdt",
        start_time="2025-12-17 10:00:00",
        end_time="2025-12-17 15:00:00",
        market_type="futures",
        save_plot="data_integrity_check.png",
        show_plot=True,
        show_table=True,
        table_rows=10,
    )