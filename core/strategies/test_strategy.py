# strategies/trade_knife_strategy.py
from typing import List, Deque
from collections import deque
from .base_strategy import BaseStrategy


class TradeKnifeStrategy(BaseStrategy):
    """
    Ловля ножей по трейдам:
    - смотрим окно последних N секунд (по trade.time)
    - если за это окно сильно доминируют продажи и цена упала -> покупаем на usd_step долларов
    - суммарный долларовый риск позиции ограничиваем max_usd_position
    - используем стоп: если цена падает на stop_loss_pct от средней входной цены -> маркет-выход
    """

    def __init__(
        self,
        window_sec: int = 30,         # окно по времени в секундах
        min_sell_ratio: float = 2.0,  # во сколько раз продажи > покупок
        min_drop_pct: float = -0.005, # минимальное падение цены за окно, например -0.5%
        usd_step: float = 100.0,      # шаг входа в долларах
        max_usd_position: float = 1000.0,  # максимум по модулю в долларах
        stop_loss_pct: float = 0.01,  # 1% стоп от средней входной цены
    ):
        self.window_sec = window_sec
        self.min_sell_ratio = min_sell_ratio
        self.min_drop_pct = min_drop_pct
        self.usd_step = usd_step
        self.max_usd_position = max_usd_position
        self.stop_loss_pct = stop_loss_pct

        # окно трейдов: (time, price, qty, is_maker_side)
        # is_maker=True на Binance => трейдер агрессор — продавец? (покупатель является мейкером? наоборот).
        # Для простоты считаем: is_maker == 1 => SELL агрессор, is_maker == 0 => BUY агрессор.
        self.trades_window: Deque[dict] = deque(maxlen=5000)

        # для стопов будем хранить среднюю цену входа (можно брать из net_position.avg_price)
        self.last_trade_price: float | None = None

    def _update_trades_window(self, event: dict):
        """
        Оставляем в окне только трейды за последние window_sec секунд.
        event["time"] — это trade time в мс, судя по твоему rename.
        """
        t_ms = event["time"]  # колонка из твоих trades (column_5 -> "time")
        price = event["price"]
        qty = event["quantity"]
        is_maker = event["is_maker"]  # 0/1

        self.last_trade_price = price

        self.trades_window.append({
            "time_ms": t_ms,
            "price": price,
            "qty": qty,
            "is_maker": is_maker,
        })

        # чистим старые
        cutoff = t_ms - self.window_sec * 1000
        while self.trades_window and self.trades_window[0]["time_ms"] < cutoff:
            self.trades_window.popleft()

    def _calc_signals(self) -> tuple[float, float, float]:
        """
        Возвращает:
        - sell_volume
        - buy_volume
        - price_return за окно (последняя / первая - 1)
        """
        if len(self.trades_window) < 2:
            return 0.0, 0.0, 0.0

        first_price = self.trades_window[0]["price"]
        last_price = self.trades_window[-1]["price"]
        price_ret = (last_price / first_price) - 1.0

        buy_vol = 0.0
        sell_vol = 0.0
        for tr in self.trades_window:
            if tr["is_maker"] == 1:  # считаем, что это SELL агрессор
                sell_vol += tr["qty"]
            else:
                buy_vol += tr["qty"]

        return sell_vol, buy_vol, price_ret

    def on_tick(self, event: dict, context: dict) -> List[dict]:
        cmds: List[dict] = []

        # 1. обновляем окно, только для trade-событий
        if event["event_type"] == "trade":
            self._update_trades_window(event)

        # если нет трейдов, нечего делать
        if self.last_trade_price is None or not self.trades_window:
            return cmds

        # 2. стопы: если есть лонг и цена опустилась ниже стопа, выходим
        net_pos = context["net_position"]
        pos_size = net_pos.size
        avg_price = net_pos.avg_price

        current_price = self.last_trade_price
        current_notional = abs(pos_size) * current_price

        # ограничение по максимальному номиналу позиции
        if current_notional > self.max_usd_position:
            # если уже превысили лимит, можно принудительно сокращать,
            # но для начала просто перестаём добавлять.
            over_limit = True
        else:
            over_limit = False

        # стоп только для лонга (для шорта можно добавить аналогично)
        if pos_size > 0 and avg_price > 0:
            stop_price = avg_price * (1.0 - self.stop_loss_pct)
            if current_price <= stop_price:
                # маркет-продажа на весь объём
                cmds.append({
                    "cmd": "place_order",
                    "type": "market",
                    "side": "sell",
                    "price": None,
                    "size": float(abs(pos_size)),
                })
                # после стопа дальше в этом тике ничего не делаем
                return cmds

        # 3. сигналы на вход по окну трейдов (knife catching)
        sell_vol, buy_vol, price_ret = self._calc_signals()

        if sell_vol == 0 and buy_vol == 0:
            return cmds

        # сильно доминируют продажи и цена упала
        # условие можно играть: и по объёму, и по падению
        cond_sell_dominance = sell_vol > self.min_sell_ratio * buy_vol
        cond_price_drop = price_ret <= self.min_drop_pct

        if cond_sell_dominance and cond_price_drop and not over_limit:
            # считаем размер лота на usd_step долларов
            if current_price <= 0:
                return cmds

            step_size = self.usd_step / current_price

            # проверяем, что новый размер не превысит лимит
            future_notional = current_notional + self.usd_step
            if future_notional <= self.max_usd_position:
                cmds.append({
                    "cmd": "place_order",
                    "type": "market",
                    "side": "buy",
                    "price": None,
                    "size": float(step_size),
                })

        return cmds