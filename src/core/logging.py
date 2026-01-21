"""
Structured logging utilities for the anomaly detection system.
"""
import logging
import json
from datetime import datetime
from typing import Any


class StructuredLogger:
    """Logger that outputs structured JSON logs."""
    
    def __init__(self, name: str, level: str = "INFO", log_file: str | None = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Formatter for structured output
        formatter = logging.Formatter('%(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _format_log(self, level: str, event: str, **kwargs) -> str:
        """Format log entry as JSON."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "event": event,
            **kwargs
        }
        return json.dumps(log_entry)
    
    def info(self, event: str, **kwargs):
        """Log info level event."""
        self.logger.info(self._format_log("INFO", event, **kwargs))
    
    def warning(self, event: str, **kwargs):
        """Log warning level event."""
        self.logger.warning(self._format_log("WARNING", event, **kwargs))
    
    def error(self, event: str, **kwargs):
        """Log error level event."""
        self.logger.error(self._format_log("ERROR", event, **kwargs))
    
    def debug(self, event: str, **kwargs):
        """Log debug level event."""
        self.logger.debug(self._format_log("DEBUG", event, **kwargs))


class AnomalyAlertChecker:
    """
    Monitors anomaly rates and triggers alerts when thresholds are exceeded.
    """
    
    def __init__(self, alert_threshold: float = 0.3, window_size: int = 100):
        """
        Args:
            alert_threshold: Fraction of anomalies that triggers an alert (0.3 = 30%)
            window_size: Number of recent predictions to consider
        """
        self.alert_threshold = alert_threshold
        self.window_size = window_size
        self.recent_predictions: list[bool] = []
    
    def add_predictions(self, predictions: list[bool]) -> dict[str, Any]:
        """
        Add predictions and check if alert should be triggered.
        
        Returns:
            Alert status dictionary
        """
        self.recent_predictions.extend(predictions)
        
        # Keep only the most recent predictions
        if len(self.recent_predictions) > self.window_size:
            self.recent_predictions = self.recent_predictions[-self.window_size:]
        
        # Calculate anomaly rate
        if len(self.recent_predictions) < 10:
            return {
                "alert": False,
                "reason": "insufficient_data",
                "sample_count": len(self.recent_predictions)
            }
        
        anomaly_rate = sum(self.recent_predictions) / len(self.recent_predictions)
        
        return {
            "alert": anomaly_rate > self.alert_threshold,
            "anomaly_rate": round(anomaly_rate, 4),
            "threshold": self.alert_threshold,
            "sample_count": len(self.recent_predictions),
            "anomaly_count": sum(self.recent_predictions)
        }
    
    def reset(self):
        """Clear the prediction window."""
        self.recent_predictions.clear()
