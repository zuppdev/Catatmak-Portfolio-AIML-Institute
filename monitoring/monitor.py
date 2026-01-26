"""
Model monitoring and drift detection
"""

import numpy as np
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from typing import Dict, List
import json
from datetime import datetime
from collections import deque
import threading
import time


# Prometheus metrics
REQUEST_COUNT = Counter(
    'expense_requests_total',
    'Total number of expense extraction requests',
    ['modality', 'category']
)

REQUEST_DURATION = Histogram(
    'expense_request_duration_seconds',
    'Time spent processing expense requests',
    ['modality']
)

MODEL_CONFIDENCE = Histogram(
    'model_confidence',
    'Model confidence scores',
    ['model', 'category']
)

AMOUNT_DISTRIBUTION = Histogram(
    'expense_amount_idr',
    'Distribution of expense amounts in IDR',
    buckets=[1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
)

PREDICTION_ERRORS = Counter(
    'prediction_errors_total',
    'Total number of prediction errors',
    ['model', 'error_type']
)

DRIFT_SCORE = Gauge(
    'model_drift_score',
    'Model drift detection score',
    ['model']
)


class ModelMonitor:
    """
    Monitor model performance and detect drift
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
        # Store recent predictions
        self.text_predictions = deque(maxlen=window_size)
        self.image_predictions = deque(maxlen=window_size)
        self.audio_predictions = deque(maxlen=window_size)
        
        # Store baseline statistics
        self.baseline_stats = {
            'text': {'mean_conf': 0.0, 'std_conf': 0.0},
            'image': {'mean_conf': 0.0, 'std_conf': 0.0},
            'audio': {'mean_conf': 0.0, 'std_conf': 0.0}
        }
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Start background drift detection
        self.drift_check_interval = 300  # 5 minutes
        self.start_drift_monitoring()
    
    def log_prediction(
        self,
        modality: str,
        category: str,
        confidence: float,
        amount: float,
        duration: float,
        error: bool = False
    ):
        """
        Log a prediction for monitoring
        
        Args:
            modality: 'text', 'image', or 'audio'
            category: Predicted category
            confidence: Model confidence
            amount: Predicted amount
            duration: Processing time
            error: Whether there was an error
        """
        with self.lock:
            # Update Prometheus metrics
            REQUEST_COUNT.labels(modality=modality, category=category).inc()
            REQUEST_DURATION.labels(modality=modality).observe(duration)
            MODEL_CONFIDENCE.labels(model=modality, category=category).observe(confidence)
            AMOUNT_DISTRIBUTION.observe(amount)
            
            if error:
                PREDICTION_ERRORS.labels(model=modality, error_type='prediction').inc()
            
            # Store prediction
            prediction_data = {
                'timestamp': datetime.now().isoformat(),
                'category': category,
                'confidence': confidence,
                'amount': amount,
                'duration': duration
            }
            
            if modality == 'text':
                self.text_predictions.append(prediction_data)
            elif modality == 'image':
                self.image_predictions.append(prediction_data)
            elif modality == 'audio':
                self.audio_predictions.append(prediction_data)
    
    def calculate_drift_score(self, modality: str) -> float:
        """
        Calculate drift score for a model
        Uses PSI (Population Stability Index)
        
        Args:
            modality: 'text', 'image', or 'audio'
            
        Returns:
            Drift score (0 = no drift, higher = more drift)
        """
        with self.lock:
            if modality == 'text':
                predictions = list(self.text_predictions)
            elif modality == 'image':
                predictions = list(self.image_predictions)
            elif modality == 'audio':
                predictions = list(self.audio_predictions)
            else:
                return 0.0
            
            if len(predictions) < 100:
                return 0.0
            
            # Split into baseline and current
            split_idx = len(predictions) // 2
            baseline = predictions[:split_idx]
            current = predictions[split_idx:]
            
            # Extract confidence scores
            baseline_conf = [p['confidence'] for p in baseline]
            current_conf = [p['confidence'] for p in current]
            
            # Calculate PSI
            bins = np.linspace(0, 1, 11)
            
            baseline_hist, _ = np.histogram(baseline_conf, bins=bins)
            current_hist, _ = np.histogram(current_conf, bins=bins)
            
            # Add small value to avoid division by zero
            baseline_hist = baseline_hist + 1e-6
            current_hist = current_hist + 1e-6
            
            # Normalize
            baseline_pct = baseline_hist / baseline_hist.sum()
            current_pct = current_hist / current_hist.sum()
            
            # Calculate PSI
            psi = np.sum(
                (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
            )
            
            return float(psi)
    
    def check_drift(self):
        """Check for model drift across all modalities"""
        for modality in ['text', 'image', 'audio']:
            drift_score = self.calculate_drift_score(modality)
            DRIFT_SCORE.labels(model=modality).set(drift_score)
            
            # Alert if drift is significant
            if drift_score > 0.25:  # PSI > 0.25 indicates significant drift
                print(f"⚠️  WARNING: Significant drift detected in {modality} model!")
                print(f"   Drift score: {drift_score:.4f}")
    
    def start_drift_monitoring(self):
        """Start background thread for drift monitoring"""
        def monitor_loop():
            while True:
                time.sleep(self.drift_check_interval)
                self.check_drift()
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
    
    def get_statistics(self) -> Dict:
        """Get current monitoring statistics"""
        with self.lock:
            return {
                'text': {
                    'total_predictions': len(self.text_predictions),
                    'avg_confidence': np.mean([p['confidence'] for p in self.text_predictions]) if self.text_predictions else 0,
                    'drift_score': self.calculate_drift_score('text')
                },
                'image': {
                    'total_predictions': len(self.image_predictions),
                    'avg_confidence': np.mean([p['confidence'] for p in self.image_predictions]) if self.image_predictions else 0,
                    'drift_score': self.calculate_drift_score('image')
                },
                'audio': {
                    'total_predictions': len(self.audio_predictions),
                    'avg_confidence': np.mean([p['confidence'] for p in self.audio_predictions]) if self.audio_predictions else 0,
                    'drift_score': self.calculate_drift_score('audio')
                }
            }
    
    def export_metrics(self, filepath: str):
        """Export metrics to file"""
        stats = self.get_statistics()
        
        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'statistics': stats
            }, f, indent=2)
        
        print(f"Metrics exported to {filepath}")


# Global monitor instance
monitor = ModelMonitor()


def start_monitoring_server(port: int = 8001):
    """Start Prometheus metrics server"""
    start_http_server(port)
    print(f"Monitoring server started on port {port}")
    print(f"Metrics available at http://localhost:{port}/metrics")


if __name__ == "__main__":
    # Start monitoring server
    start_monitoring_server(port=8001)
    
    # Simulate some predictions for demo
    import random
    
    print("\nSimulating predictions for demonstration...")
    
    categories = ['makanan', 'transportasi', 'belanja', 'hiburan']
    
    for i in range(100):
        modality = random.choice(['text', 'image', 'audio'])
        category = random.choice(categories)
        confidence = random.uniform(0.6, 0.95)
        amount = random.uniform(10000, 100000)
        duration = random.uniform(0.1, 2.0)
        
        monitor.log_prediction(
            modality=modality,
            category=category,
            confidence=confidence,
            amount=amount,
            duration=duration
        )
        
        if i % 20 == 0:
            print(f"Logged {i} predictions...")
    
    print("\nMonitoring Statistics:")
    stats = monitor.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # Export metrics
    monitor.export_metrics('monitoring_metrics.json')
    
    print("\n✓ Monitoring server is running")
    print("  Visit http://localhost:8001/metrics to see Prometheus metrics")
    print("  Press Ctrl+C to stop")
    
    # Keep running
    try:
        while True:
            time.sleep(60)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Monitoring active...")
    except KeyboardInterrupt:
        print("\nShutting down monitoring server...")
