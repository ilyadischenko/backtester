# visualizer_finplot.py
import finplot as fplt
import pandas as pd


class BacktestVisualizer:
    
    def __init__(self, engine):
        self.engine = engine
        
    def show(self):
        df = self.engine.bookticker
        pdf = df.to_pandas()
        pdf['time'] = pd.to_datetime(pdf['event_time'], unit='ms')
        
        # Прореживаем
        if len(pdf) > 10000:
            step = len(pdf) // 10000
            pdf = pdf.iloc[::step].reset_index(drop=True)
        
        pdf = pdf.set_index('time')
        
        # Создаём окно
        ax = fplt.create_plot('Backtest')
        
        # Рисуем bid/ask
        fplt.plot(pdf['bid_price'], ax=ax, color='#26a69a', width=2, legend='Bid')
        fplt.plot(pdf['ask_price'], ax=ax, color='#ef5350', width=2, legend='Ask')
        
        fplt.show()