"""
train_model.py

Обучение нейросети для предсказания движения цены.
Использует данные из S3 через DataManager.
"""

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from collections import deque
import joblib
import json

# Твой DataManager
from data.data_manager import dataManager, MarketData


# ══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    
    # Данные
    exchange: str = "binance"
    symbol: str = "pippinusdt"
    market_type: str = "futures"
    
    # Временной диапазон для обучения
    start_time: str = "2025-12-18 10:00"
    end_time: str = "2025-12-18 12:00"
    
    # Параметры предсказания
    horizon_ms: int = 500          # Горизонт предсказания в мс
    threshold_bps: float = 2.0     # Порог для классификации (basis points)
    
    # Параметры фич
    lookback_ticks: int = 500       # Сколько тиков смотрим назад
    
    # Обучение
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    
    # Валидация
    val_ratio: float = 0.15        # Доля для валидации (от train)
    test_ratio: float = 0.15       # Доля для теста (от всего)
    
    # Пути
    model_dir: Path = Path("./models")
    
    def __post_init__(self):
        self.model_dir = Path(self.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# ИЗВЛЕЧЕНИЕ ФИЧ
# ══════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """
    Извлекает фичи из потока orderbook и trades данных.
    Оптимизировано для работы с polars DataFrame.
    """
    
    def __init__(self, lookback_ticks: int = 50):
        self.lookback_ticks = lookback_ticks
        
        # Буферы для накопления данных
        self.times: deque = deque(maxlen=lookback_ticks + 100)
        self.mids: deque = deque(maxlen=lookback_ticks + 100)
        self.spreads: deque = deque(maxlen=lookback_ticks + 100)
        self.bid_qtys: deque = deque(maxlen=lookback_ticks + 100)
        self.ask_qtys: deque = deque(maxlen=lookback_ticks + 100)
        
        # Trade буферы
        self.trade_times: deque = deque(maxlen=lookback_ticks + 100)
        self.trade_qtys: deque = deque(maxlen=lookback_ticks + 100)
        self.trade_sides: deque = deque(maxlen=lookback_ticks + 100)
        
        # Названия фич для документации
        self.feature_names: List[str] = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Строит список названий фич"""
        self.feature_names = [
            # Order book imbalance
            "imbalance_current",
            "imbalance_vs_avg",
            "imbalance_vs_neutral",
            
            # Price momentum
            "return_last",
            "return_sum_5",
            "return_sum_20",
            "return_mean",
            
            # Volatility
            "volatility",
            "volatility_scaled",
            
            # Spread
            "spread_relative",
            "spread_vs_avg",
            
            # Trade flow
            "net_flow",
            "buy_ratio",
            
            # Micro features
            "time_since_trade_sec",
            "tick_direction_net",
            "tick_direction_up",
            "tick_direction_down",
        ]
    
    def reset(self):
        """Сбрасывает буферы"""
        self.times.clear()
        self.mids.clear()
        self.spreads.clear()
        self.bid_qtys.clear()
        self.ask_qtys.clear()
        self.trade_times.clear()
        self.trade_qtys.clear()
        self.trade_sides.clear()
    
    def add_orderbook(self, event_time: int, bid_price: float, ask_price: float,
                      bid_qty: float, ask_qty: float):
        """Добавляет orderbook событие"""
        mid = (bid_price + ask_price) / 2
        spread = ask_price - bid_price
        
        self.times.append(event_time)
        self.mids.append(mid)
        self.spreads.append(spread)
        self.bid_qtys.append(bid_qty)
        self.ask_qtys.append(ask_qty)
    
    def add_trade(self, event_time: int, qty: float, is_maker: int):
        """Добавляет trade событие"""
        # is_maker=1 означает buyer is maker, т.е. taker продаёт (side=-1)
        side = -1 if is_maker == 1 else 1
        
        self.trade_times.append(event_time)
        self.trade_qtys.append(qty)
        self.trade_sides.append(side)
    
    def get_features(self) -> Optional[np.ndarray]:
        """Извлекает фичи из текущего состояния буферов"""
        
        if len(self.mids) < self.lookback_ticks:
            return None
        
        # Конвертируем в numpy
        mids = np.array(list(self.mids)[-self.lookback_ticks:])
        spreads = np.array(list(self.spreads)[-self.lookback_ticks:])
        bid_qtys = np.array(list(self.bid_qtys)[-self.lookback_ticks:])
        ask_qtys = np.array(list(self.ask_qtys)[-self.lookback_ticks:])
        
        features = []
        
        # ══════════════════════════════════════════════════════
        # 1. ORDER BOOK IMBALANCE
        # ══════════════════════════════════════════════════════
        total_qty = bid_qtys + ask_qtys + 1e-10
        imbalances = bid_qtys / total_qty
        
        current_imbalance = imbalances[-1]
        avg_imbalance = np.mean(imbalances)
        
        features.extend([
            current_imbalance,                    # текущий imbalance
            current_imbalance - avg_imbalance,    # отклонение от среднего
            current_imbalance - 0.5,              # отклонение от нейтрального
        ])
        
        # ══════════════════════════════════════════════════════
        # 2. PRICE MOMENTUM
        # ══════════════════════════════════════════════════════
        returns = np.diff(mids) / (mids[:-1] + 1e-10)
        
        features.extend([
            returns[-1] if len(returns) > 0 else 0,
            np.sum(returns[-5:]) if len(returns) >= 5 else 0,
            np.sum(returns[-20:]) if len(returns) >= 20 else 0,
            np.mean(returns) if len(returns) > 0 else 0,
        ])
        
        # ══════════════════════════════════════════════════════
        # 3. VOLATILITY
        # ══════════════════════════════════════════════════════
        volatility = np.std(returns) if len(returns) > 1 else 0
        
        features.extend([
            volatility,
            volatility * np.sqrt(self.lookback_ticks),
        ])
        
        # ══════════════════════════════════════════════════════
        # 4. SPREAD
        # ══════════════════════════════════════════════════════
        current_spread = spreads[-1]
        avg_spread = np.mean(spreads)
        current_mid = mids[-1]
        
        features.extend([
            current_spread / (current_mid + 1e-10),
            (current_spread - avg_spread) / (avg_spread + 1e-10),
        ])
        
        # ══════════════════════════════════════════════════════
        # 5. TRADE FLOW
        # ══════════════════════════════════════════════════════
        if len(self.trade_sides) >= 10:
            recent_sides = list(self.trade_sides)[-20:]
            recent_qtys = list(self.trade_qtys)[-20:]
            
            buy_vol = sum(q for s, q in zip(recent_sides, recent_qtys) if s == 1)
            sell_vol = sum(q for s, q in zip(recent_sides, recent_qtys) if s == -1)
            total_vol = buy_vol + sell_vol + 1e-10
            
            features.extend([
                (buy_vol - sell_vol) / total_vol,
                buy_vol / total_vol,
            ])
        else:
            features.extend([0.0, 0.5])
        
        # ══════════════════════════════════════════════════════
        # 6. MICRO FEATURES
        # ══════════════════════════════════════════════════════
        current_time = self.times[-1]
        
        # Время с последнего трейда
        if self.trade_times:
            time_since_trade = (current_time - self.trade_times[-1]) / 1000.0
        else:
            time_since_trade = 0
        features.append(min(time_since_trade, 10.0))  # ограничиваем 10 сек
        
        # Tick direction
        tick_dirs = []
        for i in range(1, min(20, len(mids))):
            if mids[-i] > mids[-i-1]:
                tick_dirs.append(1)
            elif mids[-i] < mids[-i-1]:
                tick_dirs.append(-1)
            else:
                tick_dirs.append(0)
        
        if tick_dirs:
            features.extend([
                sum(tick_dirs),
                sum(1 for d in tick_dirs if d > 0),
                sum(1 for d in tick_dirs if d < 0),
            ])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def get_current_mid(self) -> float:
        return self.mids[-1] if self.mids else 0.0
    
    def get_current_time(self) -> int:
        return self.times[-1] if self.times else 0


# ══════════════════════════════════════════════════════════════════════════════
# ПОДГОТОВКА ДАТАСЕТА
# ══════════════════════════════════════════════════════════════════════════════

def prepare_dataset(
    orderbook_df: pl.DataFrame,
    trades_df: pl.DataFrame,
    config: TrainingConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Готовит датасет из DataFrame.
    
    Returns:
        X: фичи (N, num_features)
        y: классы (N,) — 0=down, 1=neutral, 2=up
        times: временные метки (N,)
    """
    
    print("\n🔧 Подготовка датасета...")
    print(f"   Orderbook записей: {len(orderbook_df):,}")
    print(f"   Trade записей: {len(trades_df):,}")
    print(f"   Горизонт: {config.horizon_ms} мс")
    print(f"   Порог: {config.threshold_bps} bps")
    
    extractor = FeatureExtractor(lookback_ticks=config.lookback_ticks)
    
    # ──────────────────────────────────────────────────────
    # 1. Объединяем события по времени
    # ──────────────────────────────────────────────────────
    
    # Добавляем тип события
    ob_events = orderbook_df.select([
        pl.col("event_time"),
        pl.col("bid_price"),
        pl.col("ask_price"),
        pl.col("bid_qty"),
        pl.col("ask_qty"),
        pl.lit("ob").alias("event_type"),
    ])
    
    tr_events = trades_df.select([
        pl.col("event_time"),
        pl.col("qty"),
        pl.col("is_maker"),
        pl.lit("tr").alias("event_type"),
    ])
    
    # ──────────────────────────────────────────────────────
    # 2. Строим индекс mid price для быстрого поиска future price
    # ──────────────────────────────────────────────────────
    
    print("   📊 Строим индекс цен...")
    
    ob_sorted = orderbook_df.sort("event_time")
    times_arr = ob_sorted["event_time"].to_numpy()
    mids_arr = ((ob_sorted["bid_price"] + ob_sorted["ask_price"]) / 2).to_numpy()
    
    # ──────────────────────────────────────────────────────
    # 3. Итерируемся по событиям
    # ──────────────────────────────────────────────────────
    
    print("   🔄 Извлечение фич...")
    
    X_list = []
    y_list = []
    t_list = []
    
    # Конвертируем в numpy для скорости
    ob_times = orderbook_df["event_time"].to_numpy()
    ob_bid_prices = orderbook_df["bid_price"].to_numpy()
    ob_ask_prices = orderbook_df["ask_price"].to_numpy()
    ob_bid_qtys = orderbook_df["bid_qty"].to_numpy()
    ob_ask_qtys = orderbook_df["ask_qty"].to_numpy()
    
    tr_times = trades_df["event_time"].to_numpy()
    tr_qtys = trades_df["qty"].to_numpy()
    tr_is_maker = trades_df["is_maker"].to_numpy()
    
    # Индексы для итерации
    ob_idx = 0
    tr_idx = 0
    ob_len = len(ob_times)
    tr_len = len(tr_times)
    
    # Прогресс
    total_events = ob_len + tr_len
    last_progress = 0
    processed = 0
    
    while ob_idx < ob_len or tr_idx < tr_len:
        # Определяем какое событие раньше
        ob_time = ob_times[ob_idx] if ob_idx < ob_len else float('inf')
        tr_time = tr_times[tr_idx] if tr_idx < tr_len else float('inf')
        
        if ob_time <= tr_time and ob_idx < ob_len:
            # Обрабатываем orderbook
            extractor.add_orderbook(
                ob_times[ob_idx],
                ob_bid_prices[ob_idx],
                ob_ask_prices[ob_idx],
                ob_bid_qtys[ob_idx],
                ob_ask_qtys[ob_idx],
            )
            
            # Извлекаем фичи
            features = extractor.get_features()
            
            if features is not None:
                current_time = ob_times[ob_idx]
                current_mid = (ob_bid_prices[ob_idx] + ob_ask_prices[ob_idx]) / 2
                
                # Ищем future price через binary search
                target_time = current_time + config.horizon_ms
                future_idx = np.searchsorted(times_arr, target_time)
                
                if future_idx < len(times_arr):
                    future_mid = mids_arr[future_idx]
                    
                    # Считаем return в bps
                    ret_bps = (future_mid - current_mid) / current_mid * 10000
                    
                    # Классификация
                    if ret_bps > config.threshold_bps:
                        label = 2  # UP
                    elif ret_bps < -config.threshold_bps:
                        label = 0  # DOWN
                    else:
                        label = 1  # NEUTRAL
                    
                    X_list.append(features)
                    y_list.append(label)
                    t_list.append(current_time)
            
            ob_idx += 1
        else:
            # Обрабатываем trade
            if tr_idx < tr_len:
                extractor.add_trade(
                    tr_times[tr_idx],
                    tr_qtys[tr_idx],
                    tr_is_maker[tr_idx],
                )
                tr_idx += 1
        
        # Прогресс
        processed += 1
        progress = int(processed / total_events * 100)
        if progress >= last_progress + 10:
            print(f"      {progress}% обработано...")
            last_progress = progress
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    times = np.array(t_list, dtype=np.int64)
    
    # Статистика
    print(f"\n📈 Датасет готов:")
    print(f"   Сэмплов: {len(y):,}")
    print(f"   Фич: {X.shape[1]}")
    print(f"   Распределение классов:")
    print(f"      DOWN:    {(y == 0).sum():,} ({(y == 0).mean()*100:.1f}%)")
    print(f"      NEUTRAL: {(y == 1).sum():,} ({(y == 1).mean()*100:.1f}%)")
    print(f"      UP:      {(y == 2).sum():,} ({(y == 2).mean()*100:.1f}%)")
    
    return X, y, times


# ══════════════════════════════════════════════════════════════════════════════
# МОДЕЛЬ
# ══════════════════════════════════════════════════════════════════════════════

class PricePredictor(nn.Module):
    """
    Нейросеть для предсказания направления цены.
    Простая архитектура для начала.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


# ══════════════════════════════════════════════════════════════════════════════
# ОБУЧЕНИЕ
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    config: TrainingConfig,
) -> Tuple[PricePredictor, StandardScaler, dict]:
    """
    Обучает модель с walk-forward валидацией.
    
    Returns:
        model: обученная модель
        scaler: StandardScaler для фич
        metrics: метрики обучения
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Обучение на {device.upper()}")
    
    # ──────────────────────────────────────────────────────
    # 1. Split данных (временной, НЕ случайный!)
    # ──────────────────────────────────────────────────────
    
    n_samples = len(y)
    test_size = int(n_samples * config.test_ratio)
    train_val_size = n_samples - test_size
    val_size = int(train_val_size * config.val_ratio)
    train_size = train_val_size - val_size
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"\n📊 Split данных (временной):")
    print(f"   Train: {len(y_train):,} ({len(y_train)/n_samples*100:.1f}%)")
    print(f"   Val:   {len(y_val):,} ({len(y_val)/n_samples*100:.1f}%)")
    print(f"   Test:  {len(y_test):,} ({len(y_test)/n_samples*100:.1f}%)")
    
    # ──────────────────────────────────────────────────────
    # 2. Нормализация (fit только на train!)
    # ──────────────────────────────────────────────────────
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Убираем NaN/Inf
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ──────────────────────────────────────────────────────
    # 3. DataLoaders
    # ──────────────────────────────────────────────────────
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    
    # ──────────────────────────────────────────────────────
    # 4. Модель, Loss, Optimizer
    # ──────────────────────────────────────────────────────
    
    input_dim = X_train.shape[1]
    num_classes = 3
    
    model = PricePredictor(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        num_classes=num_classes,
        dropout=config.dropout,
    ).to(device)
    
    print(f"\n🧠 Модель:")
    print(f"   Input dim: {input_dim}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Веса классов для несбалансированных данных
    class_counts = np.bincount(y_train, minlength=3)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * 3
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"   Class weights: {class_weights.cpu().numpy().round(2)}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=5,
        factor=0.5,
        # verbose=True,
    )
    
    # ──────────────────────────────────────────────────────
    # 5. Training Loop
    # ──────────────────────────────────────────────────────
    
    print(f"\n📚 Обучение {config.epochs} эпох...")
    
    best_val_acc = 0
    best_model_state = None
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        train_loss /= train_total
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        
        # Сохраняем историю
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Scheduler step
        scheduler.step(val_acc)
        
        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Логирование
        if epoch % 5 == 0 or epoch == config.epochs - 1:
            print(f"   Epoch {epoch:3d}: "
                  f"Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
                  f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
    
    # Загружаем лучшую модель
    model.load_state_dict(best_model_state)
    
    # ──────────────────────────────────────────────────────
    # 6. Тестирование
    # ──────────────────────────────────────────────────────
    
    print(f"\n🧪 Тестирование...")
    
    model.eval()
    test_X = torch.FloatTensor(X_test_scaled).to(device)
    test_y = torch.LongTensor(y_test).to(device)
    
    with torch.no_grad():
        outputs = model(test_X)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        test_acc = (predicted == test_y).float().mean().item() * 100
    
    # Confusion matrix
    predicted_np = predicted.cpu().numpy()
    test_y_np = y_test
    
    print(f"\n📊 Результаты на тесте:")
    print(f"   Accuracy: {test_acc:.2f}%")
    print(f"   Best Val Accuracy: {best_val_acc:.2f}%")
    
    # Подробная статистика по классам
    for i, class_name in enumerate(['DOWN', 'NEUTRAL', 'UP']):
        mask = test_y_np == i
        if mask.sum() > 0:
            class_acc = (predicted_np[mask] == i).mean() * 100
            print(f"   {class_name}: {class_acc:.1f}% ({mask.sum()} samples)")
    
    # Собираем метрики
    metrics = {
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
        'history': history,
        'config': {
            'horizon_ms': config.horizon_ms,
            'threshold_bps': config.threshold_bps,
            'lookback_ticks': config.lookback_ticks,
            'input_dim': input_dim,
        }
    }
    
    return model, scaler, metrics


# ══════════════════════════════════════════════════════════════════════════════
# СОХРАНЕНИЕ/ЗАГРУЗКА
# ══════════════════════════════════════════════════════════════════════════════

def save_model(
    model: PricePredictor,
    scaler: StandardScaler,
    metrics: dict,
    config: TrainingConfig,
    suffix: str = "",
):
    """Сохраняет модель и все артефакты"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{config.symbol}_{config.horizon_ms}ms{suffix}_{timestamp}"
    model_path = config.model_dir / model_name
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Модель
    torch.save(model.state_dict(), model_path / "model.pt")
    
    # Scaler
    joblib.dump(scaler, model_path / "scaler.joblib")
    
    # Метрики и конфиг
    with open(model_path / "metrics.json", "w") as f:
        # Конвертируем numpy в python types
        metrics_clean = {
            'test_accuracy': float(metrics['test_accuracy']),
            'best_val_accuracy': float(metrics['best_val_accuracy']),
            'config': metrics['config'],
        }
        json.dump(metrics_clean, f, indent=2)
    
    # Конфиг для воспроизводимости
    config_dict = {
        'exchange': config.exchange,
        'symbol': config.symbol,
        'horizon_ms': config.horizon_ms,
        'threshold_bps': config.threshold_bps,
        'lookback_ticks': config.lookback_ticks,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
    }
    with open(model_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n💾 Модель сохранена: {model_path}")
    
    return model_path


def load_model(model_path: Path | str) -> Tuple[PricePredictor, StandardScaler, dict]:
    """Загружает модель и артефакты"""
    
    model_path = Path(model_path)
    
    # Конфиг
    with open(model_path / "config.json", "r") as f:
        config_dict = json.load(f)
    
    # Метрики
    with open(model_path / "metrics.json", "r") as f:
        metrics = json.load(f)
    
    # Scaler
    scaler = joblib.load(model_path / "scaler.joblib")
    
    # Модель
    input_dim = metrics['config']['input_dim']
    model = PricePredictor(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path / "model.pt", map_location='cpu'))
    model.eval()
    
    print(f"✅ Модель загружена: {model_path}")
    
    return model, scaler, metrics


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Главная функция обучения"""
    
    # ──────────────────────────────────────────────────────
    # 1. Конфигурация
    # ──────────────────────────────────────────────────────
    
    config = TrainingConfig(
        # Данные
        exchange="binance",
        # symbol="btcusdt",
        market_type="futures",
        
        # Период обучения (подстрой под свои данные!)
        # start_time="2025-01-10 00:00:00",
        # end_time="2025-01-15 00:00:00",
        
        # Параметры предсказания
        horizon_ms=100,
        threshold_bps=2.0,
        
        # Обучение
        epochs=50,
        batch_size=512,
        learning_rate=1e-3,
    )
    
    print("=" * 60)
    print("🎯 ОБУЧЕНИЕ МОДЕЛИ ПРЕДСКАЗАНИЯ ЦЕНЫ")
    print("=" * 60)
    print(f"   Exchange: {config.exchange}")
    print(f"   Symbol: {config.symbol}")
    print(f"   Period: {config.start_time} → {config.end_time}")
    print(f"   Horizon: {config.horizon_ms} ms")
    print(f"   Threshold: {config.threshold_bps} bps")
    
    # ──────────────────────────────────────────────────────
    # 2. Загрузка данных
    # ──────────────────────────────────────────────────────
    
    print("\n" + "=" * 60)
    print("📥 ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)
    
    data: MarketData = dataManager.load_timerange(
        exchange=config.exchange,
        symbol=config.symbol,
        start_time=config.start_time,
        end_time=config.end_time,
        data_type="all",
        market_type=config.market_type,
    )
    
    if data.orderbook is None or data.trades is None:
        raise ValueError("❌ Не удалось загрузить данные!")
    
    # ──────────────────────────────────────────────────────
    # 3. Подготовка датасета
    # ──────────────────────────────────────────────────────
    
    print("\n" + "=" * 60)
    print("🔧 ПОДГОТОВКА ДАТАСЕТА")
    print("=" * 60)
    
    X, y, times = prepare_dataset(
        orderbook_df=data.orderbook,
        trades_df=data.trades,
        config=config,
    )
    
    # ──────────────────────────────────────────────────────
    # 4. Обучение
    # ──────────────────────────────────────────────────────
    
    print("\n" + "=" * 60)
    print("🚀 ОБУЧЕНИЕ")
    print("=" * 60)
    
    model, scaler, metrics = train_model(X, y, times, config)
    
    # ──────────────────────────────────────────────────────
    # 5. Сохранение
    # ──────────────────────────────────────────────────────
    
    print("\n" + "=" * 60)
    print("💾 СОХРАНЕНИЕ")
    print("=" * 60)
    
    model_path = save_model(model, scaler, metrics, config)
    
    print("\n" + "=" * 60)
    print("✅ ГОТОВО!")
    print("=" * 60)
    print(f"\nДля использования в бэктесте:")
    print(f"   model, scaler, metrics = load_model('{model_path}')")
    
    return model, scaler, metrics, model_path


if __name__ == "__main__":
    model, scaler, metrics, model_path = main()