# strategy_ml.py

import torch
import numpy as np
from train_model import load_model, FeatureExtractor


class MLStrategy:
    """Стратегия на основе обученной модели"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.initial_balance = 1000.0
        self.model, self.scaler, self.metrics = load_model(model_path)
        self.model.eval()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        config = self.metrics['config']
        self.extractor = FeatureExtractor(lookback_ticks=config['lookback_ticks'])
        self.confidence_threshold = confidence_threshold
        
        # Для визуализации
        from visualization.visualization import PlotRecorder
        self.plot = PlotRecorder()
    
    def on_tick(self, event, engine):
        current_time = event["event_time"]
        
        if event.get("event_type") == "bookticker":
            self.extractor.add_orderbook(
                event["event_time"],
                event["bid_price"],
                event["ask_price"],
                event["bid_size"],
                event["ask_size"],
            )
        elif event.get("event_type") == "trade":
            self.extractor.add_trade(
                event["event_time"],
                event["quantity"],
                event.get("is_maker", 0),
            )
            return
        
        # Получаем фичи
        features = self.extractor.get_features()
        if features is None:
            return
        
        # Предсказание
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        features_scaled = np.nan_to_num(features_scaled)
        
        with torch.no_grad():
            x = torch.FloatTensor(features_scaled).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
        
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()
        
        mid = self.extractor.get_current_mid()
        
        # Визуализация
        self.plot.line("Confidence", confidence, current_time, color="#9C27B0")
        
        # Торгуем только при высокой уверенности
        if confidence < self.confidence_threshold:
            return
        
        if pred_class == 2:  # UP
            engine.place_order("market", size=100, price=1000000.0)
            self.plot.marker("Buy", mid, current_time, marker="triangle", color="#4CAF50", size=10)
        
        elif pred_class == 0:  # DOWN
            engine.place_order("market", size=-100, price=0.0)
            self.plot.marker("Sell", mid, current_time, marker="inverted_triangle", color="#F44336", size=10)