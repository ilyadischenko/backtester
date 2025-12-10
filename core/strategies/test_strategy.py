# main.py
from engine import ExchangeEngine
import polars as pl


class SimpleStrategy:
    """Простая стратегия для теста."""
    
    def __init__(self):
        self.last_trade_time = 0
        self.trade_interval = 10_000  # мс между сделками
        
    def on_tick(self, event, engine: ExchangeEngine):
        if event["event_type"] != "bookticker":
            return
        
        current_time = event["event_time"]
        
        # Торгуем каждые N миллисекунд
        if current_time - self.last_trade_time < self.trade_interval:
            return
        
        pos_size = engine.get_position_size()
        
        if pos_size == 0:
            # Открываем long
            engine.place_order("market", price=0, size=0.01)
            self.last_trade_time = current_time
            
        elif pos_size > 0:
            # Закрываем
            engine.place_order("market", price=0, size=-pos_size)
            self.last_trade_time = current_time


