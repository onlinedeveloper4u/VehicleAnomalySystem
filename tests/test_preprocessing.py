"""Tests for the DataPreprocessor class."""
import pytest
import pandas as pd
import numpy as np
import os
from src.ml.transformer import DataPreprocessor


def test_preprocessor_init():
    """Test preprocessor initialization with default columns."""
    prep = DataPreprocessor()
    # 21 total - 7 constant = 14 useful sensors
    assert len(prep.sensor_columns) == 14


def test_preprocessor_fit_transform():
    """Test fit_transform produces correct output shape."""
    # Create dummy data with realistic ranges
    np.random.seed(42)
    n_samples = 10
    
    # Generate data for s1-s21 + settings + engine_id + cycle
    data_dict = {
        "engine_id": [1] * n_samples,
        "cycle": range(1, n_samples + 1),
        "setting1": np.random.uniform(0, 1, n_samples),
        "setting2": np.random.uniform(0, 1, n_samples),
        "setting3": [100.0] * n_samples
    }
    for i in range(1, 22):
        data_dict[f"s{i}"] = np.random.uniform(100, 1000, n_samples)
        
    data = pd.DataFrame(data_dict)
    
    prep = DataPreprocessor()
    X = prep.fit_transform(data)
    
    # Output should have 14 feature columns (just scaled values, no rolling features in transformer currently)
    # Wait, the new transformer DOES NOT do rolling features inside it anymore?
    # Let me check transformer.py content in memory or verify.
    # The new transformer.py I wrote (id: 62) does regime normalization but no rolling features.
    
    assert X.shape == (10, 14)
    # Check no NaN values
    assert not np.isnan(X).any()


def test_preprocessor_regime_normalization():
    """Test fit_transform with regime normalization."""
    np.random.seed(42)
    n_samples = 20
    
    data_dict = {
        "engine_id": [1] * n_samples,
        "cycle": range(1, n_samples + 1),
        "setting1": np.random.choice([0, 10, 20, 35, 42, 49], n_samples), # 6 clusters
        "setting2": np.random.uniform(0, 1, n_samples),
        "setting3": [100.0] * n_samples
    }
    for i in range(1, 22):
        data_dict[f"s{i}"] = np.random.uniform(100, 1000, n_samples)
        
    data = pd.DataFrame(data_dict)
    
    prep = DataPreprocessor(use_regime_normalization=True, n_regimes=6)
    X = prep.fit_transform(data)
    
    # Still 14 features, just normalized differently
    assert X.shape == (20, 14) 
    assert prep.kmeans is not None


def test_preprocessor_missing_cols():
    """Test that missing logic uses defaults (0.0) instead of crashing, or crash if intended."""
    # The new transformer fills missing cols with 0.0 in transform()
    data = pd.DataFrame({"s2": [1, 2, 3]}) # Missing s3, s4...
    prep = DataPreprocessor()
    
    # Fit needs all columns? fit() calls _fit_regime_normalizer() which needs SETTINGS.
    # But basic fit just fits scaler on available columns or fail?
    # New prep logic:
    # fit: X = data[self.sensor_columns].values. If cols missing in training data, it fails naturally on df indexing
    
    full_data = pd.DataFrame({col: [0]*10 for col in prep.sensor_columns})
    prep.fit(full_data)
    
    # transform: fills with 0.0
    result = prep.transform(data)
    assert result.shape == (3, 14)


def test_preprocessor_save_load(tmp_path):
    """Test preprocessor persistence."""
    np.random.seed(42)
    n_samples = 10
    data_dict = {f"s{i}": np.random.rand(n_samples) for i in range(1, 22)}
    # Add settings for potential regime norm usage (though defaults to False)
    data_dict["setting1"] = np.random.rand(n_samples)
    data_dict["setting2"] = np.random.rand(n_samples)
    data_dict["setting3"] = np.random.rand(n_samples)
    
    data = pd.DataFrame(data_dict)
    
    prep = DataPreprocessor()
    prep.fit_transform(data)
    
    save_path = tmp_path / "preprocessor.pkl"
    prep.save(str(save_path))
    
    new_prep = DataPreprocessor()
    new_prep.load(str(save_path))
    
    assert new_prep.sensor_columns == prep.sensor_columns
    assert new_prep.fitted == True
