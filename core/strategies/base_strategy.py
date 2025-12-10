



class BaseStrategy:

    def __init__(self, initial_balance, order_size_usd):
        self.initial_balance: float = 1000.0
        self.order_size_usd: float = 100.0

    def on_tick(self):
        pass