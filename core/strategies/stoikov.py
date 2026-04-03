from typing import Dict, Optional, List
from collections import deque
from dataclasses import dataclass
import numpy as np
from numba import njit
from visualization.visualization import PlotRecorder

# --- Numba High-Performance Calc ---

@njit(cache=True, fastmath=True, nogil=True)
def calc_volatility(prices: np.ndarray, window: int) -> float:
    """
    Считает волатильность как стандартное отклонение доходностей.
    Очень быстро на массивах numpy.
    """
    n = len(prices)
    if n < 2:
        return 0.0
    
    # Берем последние N цен
    if n > window:
        subset = prices[-window:]
    else:
        subset = prices
        
    # Log returns: ln(p_t / p_{t-1}) ~ (p_t - p_{t-1}) / p_{t-1}
    # Для скорости используем простую процентную разницу
    diffs = np.diff(subset)
    means = subset[:-1]
    returns = diffs / means
    
    std_dev = np.std(returns)
    return std_dev

@njit(cache=True, fastmath=True, nogil=True)
def calc_reservation_price(mid_price: float, inventory: float, 
                           risk_aversion: float, vol: float) -> float:
    """
    Формула Avellaneda-Stoikov для Reservation Price.
    r = s - q * gamma * sigma^2
    Где:
    s = mid price
    q = inventory (inventory quantity)
    gamma = risk aversion parameter
    sigma = volatility
    """
    # Упрощенная линейная модель перекоса, более стабильная для крипты
    # Сдвиг цены против позы, чтобы разгрузиться
    skew = inventory * risk_aversion * (vol * mid_price) 
    return mid_price - skew

# --- Strategy Class ---

class HFTMarketMaker:
    def __init__(
        self,
        # Настройки риска
        risk_aversion: float = 0.5,    # Как сильно бояться инвентаря (Gamma)
        order_amount: float = 100.0,   # Размер ордера в $ (или монетах)
        max_position: float = 1000.0,  # Макс позиция
        
        # Настройки спреда
        min_spread_pct: float = 0.04,  # Минимальный спред (половинка, % от цены)
        vol_multiplier: float = 2.0,   # Множитель расширения спреда от волатильности
        
        # Настройки скорости
        vol_window: int = 100,         # Окно тиков для расчета волатильности
        requote_threshold_pct: float = 0.01, # Минимальное изменение цены для перестановки ордера (фильтр шума)
        
        # Имбаланс стакана
        imbalance_weight: float = 0.2, # Вес влияния дисбаланса стакана на цену
    ):
        self.risk_aversion = risk_aversion
        self.order_amount = order_amount
        self.max_position = max_position
        self.min_spread_pct = min_spread_pct / 100.0
        self.vol_multiplier = vol_multiplier
        self.vol_window = vol_window
        self.requote_threshold_pct = requote_threshold_pct / 100.0
        self.imbalance_weight = imbalance_weight
        
        # Данные (используем deque с maxlen для автоматического удаления старого)
        self.mid_prices = deque(maxlen=2000) 
        
        # Текущее состояние
        self.mid_price = 0.0
        self.volatility = 0.0
        self.inventory = 0.0
        self.book_imbalance = 0.0 # От -1 (продавцы) до 1 (покупатели)
        
        # Активные ордера
        self.bid_oid: Optional[int] = None
        self.ask_oid: Optional[int] = None
        self.active_bid_price = 0.0
        self.active_ask_price = 0.0
        
        self.plot = PlotRecorder()
        
        # Прогрев Numba
        _p = np.array([100.0, 101.0, 100.5], dtype=np.float64)
        calc_volatility(_p, 10)
        calc_reservation_price(100.0, 1.0, 0.1, 0.01)

    def on_tick(self, event, engine):
        t = event["event_time"]
        etype = event.get("event_type")
        
        # 1. Обработка BookTicker (самое важное для MM)
        if etype == "bookticker":
            bid = event["bid_price"]
            ask = event["ask_price"]
            bid_qty = event["bid_size"]
            ask_qty = event["ask_size"]
            
            self.mid_price = (bid + ask) / 2
            self.mid_prices.append(self.mid_price)
            
            # Расчет дисбаланса (Order Book Imbalance)
            # Если бидов больше -> imbalance > 0 -> цена скорее пойдет вверх
            total_qty = bid_qty + ask_qty
            if total_qty > 0:
                self.book_imbalance = (bid_qty - ask_qty) / total_qty
            else:
                self.book_imbalance = 0.0
                
            self._logic_cycle(engine, t)
            
        # 2. Обработка сделок (для обновления волатильности)
        elif etype == "trade":
            # Можно обновлять волатильность тут, но для скорости HFT 
            # достаточно обновлять на bookticker, так как цена там тоже меняется
            pass

    def _logic_cycle(self, engine, t: int):
        """Основной цикл принятия решений"""
        
        # 0. Проверка готовности данных
        if len(self.mid_prices) < 10:
            return

        # 1. Получаем текущую позицию
        pos = self._get_position(engine)
        self.inventory = pos.size if pos else 0.0
        
        # 2. Считаем волатильность (быстро через Numba)
        # Превращаем deque в numpy array только когда нужно
        # (в реальном проде лучше держать numpy буфер, но для бэктеста deque ок)
        price_arr = np.array(self.mid_prices, dtype=np.float64)
        self.volatility = calc_volatility(price_arr, self.vol_window)
        
        # 3. Reservation Price (Сдвиг цены на основе инвентаря)
        # Если у нас много лонга, res_price будет НИЖЕ mid_price
        res_price = calc_reservation_price(
            self.mid_price, 
            self.inventory / self.order_amount, # Нормализуем инвентарь к размеру ордера
            self.risk_aversion, 
            self.volatility
        )
        
        # 4. Учет микроструктуры (Imbalance)
        # Если imbalance > 0 (давление покупателей), сдвигаем цену вверх
        res_price += (self.book_imbalance * self.imbalance_weight * (self.mid_price * self.min_spread_pct))
        
        # 5. Расчет ширины спреда
        # Базовый спред + расширение от волатильности
        half_spread = (self.mid_price * self.min_spread_pct) * (1 + self.volatility * self.vol_multiplier * 100)
        
        target_bid = res_price - half_spread
        target_ask = res_price + half_spread
        
        # 6. Исполнение ордеров (Quotations)
        self._manage_quotes(engine, target_bid, target_ask, t)
        
        # Визуализация для отладки
        self.plot.line("Mid", self.mid_price, t, color="gray", alpha=0.5)
        self.plot.line("ResPrice", res_price, t, color="blue", alpha=0.8)
        self.plot.line("TargetBid", target_bid, t, color="green", linestyle="dotted")
        self.plot.line("TargetAsk", target_ask, t, color="red", linestyle="dotted")

    def _manage_quotes(self, engine, target_bid, target_ask, t):
        """Умная перестановка ордеров"""
        
        # --- BID LOGIC ---
        # Если инвентарь переполнен лонгом, можем не ставить бид вообще (Risk Management)
        should_buy = self.inventory < self.max_position
        
        if should_buy:
            # Проверяем, нужно ли двигать ордер
            if self.bid_oid is None:
                self._place_bid(engine, target_bid)
            else:
                # Двигаем, только если цена ушла существенно (фильтр спама ордерами)
                dist = abs(self.active_bid_price - target_bid) / target_bid
                if dist > self.requote_threshold_pct:
                    engine.cancel_order(self.bid_oid)
                    self._place_bid(engine, target_bid)
        elif self.bid_oid:
             engine.cancel_order(self.bid_oid)
             self.bid_oid = None

        # --- ASK LOGIC ---
        should_sell = self.inventory > -self.max_position
        
        if should_sell:
            if self.ask_oid is None:
                self._place_ask(engine, target_ask)
            else:
                dist = abs(self.active_ask_price - target_ask) / target_ask
                if dist > self.requote_threshold_pct:
                    engine.cancel_order(self.ask_oid)
                    self._place_ask(engine, target_ask)
        elif self.ask_oid:
            engine.cancel_order(self.ask_oid)
            self.ask_oid = None

    def _place_bid(self, engine, price):
        # Округляем цену (важно для биржи)
        # В реальном коде тут нужен round_to_tick_size
        qty = self.order_amount / price
        oid = engine.place_order("limit", price=price, size=qty)
        if oid:
            self.bid_oid = oid
            self.active_bid_price = price

    def _place_ask(self, engine, price):
        qty = self.order_amount / price
        oid = engine.place_order("limit", price=price, size=-qty)
        if oid:
            self.ask_oid = oid
            self.active_ask_price = price

    def _get_position(self, engine):
        # Кешируем позицию или ищем её
        for p in engine.positions:
            if p.status == "open":
                return p
        return None
