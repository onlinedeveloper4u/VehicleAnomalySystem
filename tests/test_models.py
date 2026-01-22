"""Comprehensive tests for the anomaly detection models."""
import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
import json

from src.ml.trainer import ModelTrainer
from src.ml.predictor import AnomalyDetector
from src.ml.transformer import DataPreprocessor


@pytest.fixture
def sample_sensor_data():
    """Generate sample sensor data for testing (CMAPSS schema)."""
    np.random.seed(42)
    n_samples = 100
    
    data_dict = {
        "engine_id": [1] * n_samples,
        "cycle": range(1, n_samples + 1),
        "setting1": np.random.uniform(0, 1, n_samples),
        "setting2": np.random.uniform(0, 1, n_samples),
        "setting3": [100.0] * n_samples
    }
    # Create valid s1-s21
    for i in range(1, 22):
        data_dict[f"s{i}"] = np.random.uniform(100, 1000, n_samples)
        
    return pd.DataFrame(data_dict)


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
        assert X.shape[1] == 14  # 14 useful sensors
    
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
        
        save_path = os.path.join(temp_model_dir, "preprocessor.pkl")
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
        thresholds = trainer.train_isolation_forest(X)
        
        assert isinstance(thresholds, dict)
        assert thresholds["p99"] > 0
        assert os.path.exists(os.path.join(trainer.model_dir, "isolation_forest.pkl"))
    
    def test_train_returns_thresholds(self, sample_sensor_data, temp_model_dir):
        """Test that train() returns threshold dictionary."""
        preprocessor = DataPreprocessor()
        X = preprocessor.fit_transform(sample_sensor_data)
        
        trainer = ModelTrainer(model_dir=temp_model_dir, version="test")
        thresholds = trainer.train(X)
        
        assert "p95" in thresholds
        assert "mean" in thresholds


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
        
        # Save components
        preprocessor.save(os.path.join(trainer.model_dir, "preprocessor.pkl"))
        
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
        assert "thresholds" in result
        assert "version" in result
        assert len(result["is_anomaly"]) == len(sample_sensor_data)
        assert len(result["scores"]) == len(sample_sensor_data)
    
    def test_predict_logic(self, trained_detector, sample_sensor_data):
        """Test prediction logic."""
        # Since random data is used, just check valid output types
        result = trained_detector.predict(sample_sensor_data)
        assert isinstance(result["is_anomaly"][0], bool)
        assert isinstance(result["scores"][0], float)
    
    def test_history_buffer(self, trained_detector, sample_sensor_data):
        """Test that history buffer calculates."""
        # Split data into chunks
        chunk1 = sample_sensor_data.iloc[:50]
        chunk2 = sample_sensor_data.iloc[50:]
        
        trained_detector.predict(chunk1)
        # Check history populated for engine 1
        assert 1 in trained_detector.history
        assert len(trained_detector.history[1]) > 0
        
        trained_detector.clear_history()
        assert len(trained_detector.history) == 0

    def test_single_record_prediction(self, trained_detector, sample_sensor_data):
        """Test prediction on a single record."""
        single = sample_sensor_data.head(1)
        result = trained_detector.predict(single)
        
        assert len(result["is_anomaly"]) == 1
        assert len(result["scores"]) == 1
