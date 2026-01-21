"""Comprehensive tests for the anomaly detection models."""
import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil

from src.ml.trainer import ModelTrainer
from src.ml.predictor import AnomalyDetector
from src.ml.transformer import DataPreprocessor


@pytest.fixture
def sample_sensor_data():
    """Generate sample sensor data for testing."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame({
        "Battery_Voltage": np.random.uniform(300, 400, n_samples),
        "Battery_Current": np.random.uniform(-100, 100, n_samples),
        "Battery_Temperature": np.random.uniform(20, 40, n_samples),
        "Motor_Temperature": np.random.uniform(30, 80, n_samples),
        "Motor_Vibration": np.random.uniform(0.1, 2.0, n_samples),
        "Motor_Torque": np.random.uniform(50, 200, n_samples),
        "Motor_RPM": np.random.uniform(1000, 5000, n_samples),
        "Power_Consumption": np.random.uniform(10, 50, n_samples),
        "Brake_Pressure": np.random.uniform(100, 500, n_samples),
        "Tire_Pressure": np.random.uniform(30, 40, n_samples),
        "Tire_Temperature": np.random.uniform(20, 50, n_samples),
        "Suspension_Load": np.random.uniform(500, 2000, n_samples),
        "Ambient_Temperature": np.random.uniform(15, 35, n_samples),
        "Ambient_Humidity": np.random.uniform(30, 70, n_samples),
        "Driving_Speed": np.random.uniform(0, 120, n_samples),
        "Vehicle_ID": ["test_vehicle"] * n_samples,
    })


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestDataPreprocessor:
    """Tests for the DataPreprocessor class."""
    
    def test_fit_transform(self, sample_sensor_data):
        """Test that fit_transform returns correct shape."""
        preprocessor = DataPreprocessor()
        X = preprocessor.fit_transform(sample_sensor_data)
        
        assert X is not None
        assert len(X) == len(sample_sensor_data)
        assert X.shape[1] > 15  # Should have more features after rolling
    
    def test_transform_consistency(self, sample_sensor_data):
        """Test that transform produces consistent results."""
        preprocessor = DataPreprocessor()
        X1 = preprocessor.fit_transform(sample_sensor_data)
        X2 = preprocessor.transform(sample_sensor_data)
        
        np.testing.assert_array_almost_equal(X1, X2)
    
    def test_save_and_load(self, sample_sensor_data, temp_model_dir):
        """Test scaler persistence."""
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(sample_sensor_data)
        
        save_path = os.path.join(temp_model_dir, "scaler.pkl")
        preprocessor.save(save_path)
        
        new_preprocessor = DataPreprocessor()
        new_preprocessor.load(save_path)
        
        assert new_preprocessor.feature_columns == preprocessor.feature_columns


class TestModelTrainer:
    """Tests for the ModelTrainer class."""
    
    def test_train_isolation_forest(self, sample_sensor_data, temp_model_dir):
        """Test Isolation Forest training."""
        preprocessor = DataPreprocessor()
        X = preprocessor.fit_transform(sample_sensor_data)
        
        trainer = ModelTrainer(model_dir=temp_model_dir, version="test")
        threshold = trainer.train_isolation_forest(X)
        
        assert threshold > 0
        assert os.path.exists(os.path.join(trainer.model_dir, "isolation_forest_model.pkl"))
    
    def test_train_returns_thresholds(self, sample_sensor_data, temp_model_dir):
        """Test that train() returns threshold dictionary."""
        preprocessor = DataPreprocessor()
        X = preprocessor.fit_transform(sample_sensor_data)
        
        trainer = ModelTrainer(model_dir=temp_model_dir, version="test")
        thresholds = trainer.train(X)
        
        assert "isolation_forest" in thresholds
        assert thresholds["isolation_forest"] > 0


class TestAnomalyDetector:
    """Tests for the AnomalyDetector class."""
    
    @pytest.fixture
    def trained_detector(self, sample_sensor_data, temp_model_dir):
        """Create a trained detector for testing."""
        # Train model
        preprocessor = DataPreprocessor()
        X = preprocessor.fit_transform(sample_sensor_data)
        
        trainer = ModelTrainer(model_dir=temp_model_dir, version="test")
        thresholds = trainer.train(X)
        
        # Save scaler and thresholds
        preprocessor.save(os.path.join(trainer.model_dir, "scaler.pkl"))
        
        import json
        with open(os.path.join(trainer.model_dir, "thresholds.json"), "w") as f:
            json.dump(thresholds, f)
        
        # Load detector
        detector = AnomalyDetector(model_dir=temp_model_dir, version="test")
        return detector
    
    def test_predict_returns_correct_structure(self, trained_detector, sample_sensor_data):
        """Test prediction output structure."""
        result = trained_detector.predict(sample_sensor_data)
        
        assert "is_anomaly" in result
        assert "scores" in result
        assert "threshold" in result
        assert "version" in result
        assert len(result["is_anomaly"]) == len(sample_sensor_data)
        assert len(result["scores"]) == len(sample_sensor_data)
    
    def test_predict_with_threshold_override(self, trained_detector, sample_sensor_data):
        """Test prediction with custom threshold."""
        # Very high threshold - should detect no anomalies
        result = trained_detector.predict(sample_sensor_data, threshold_override=1000.0)
        assert not any(result["is_anomaly"])
        
        # Very low threshold - should detect all as anomalies (after warm-up)
        trained_detector.clear_history()
        result = trained_detector.predict(sample_sensor_data, threshold_override=0.0)
        # First 10 have grace period, rest should be anomalies
        assert sum(result["is_anomaly"]) > 0
    
    def test_history_buffer(self, trained_detector, sample_sensor_data):
        """Test that history buffer is maintained."""
        # First prediction
        trained_detector.predict(sample_sensor_data.head(10))
        assert len(trained_detector.history) > 0
        
        # Clear history
        trained_detector.clear_history()
        assert len(trained_detector.history) == 0
    
    def test_update_threshold(self, trained_detector):
        """Test threshold update functionality."""
        new_threshold = 0.5
        trained_detector.update_threshold(new_threshold)
        assert trained_detector.threshold == new_threshold


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_record_prediction(self, sample_sensor_data, temp_model_dir):
        """Test prediction on a single record."""
        # Setup trained detector
        preprocessor = DataPreprocessor()
        X = preprocessor.fit_transform(sample_sensor_data)
        
        trainer = ModelTrainer(model_dir=temp_model_dir, version="test")
        thresholds = trainer.train(X)
        preprocessor.save(os.path.join(trainer.model_dir, "scaler.pkl"))
        
        import json
        with open(os.path.join(trainer.model_dir, "thresholds.json"), "w") as f:
            json.dump(thresholds, f)
        
        detector = AnomalyDetector(model_dir=temp_model_dir, version="test")
        
        # Predict on single record
        single = sample_sensor_data.head(1)
        result = detector.predict(single)
        
        assert len(result["is_anomaly"]) == 1
        assert len(result["scores"]) == 1
    
    def test_missing_model_raises_error(self, temp_model_dir):
        """Test that missing model raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            AnomalyDetector(model_dir=temp_model_dir, version="nonexistent")
