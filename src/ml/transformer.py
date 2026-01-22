"""
Data Preprocessor for NASA CMAPSS Turbofan Engine Data

Features:
- Handles 21 sensor measurements + 3 operational settings
- Removes near-constant sensors (s1, s5, s6, s10, s16, s18, s19)
- Optional operating regime normalization for multi-condition datasets
- Compatible with single-record and batch predictions
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
from typing import List, Optional


class DataPreprocessor:
    """Preprocessor for CMAPSS turbofan sensor data."""
    
    # Near-constant sensors to exclude (low variance, no degradation signal)
    CONSTANT_SENSORS = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
    
    # All sensor columns
    ALL_SENSORS = [f's{i}' for i in range(1, 22)]
    
    # Useful sensors (after removing near-constant)
    # Useful sensors (after removing near-constant)
    USEFUL_SENSORS = [s for s in ALL_SENSORS if s not in ['s1', 's5', 's6', 's10', 's16', 's18', 's19']]
    
    # Operational settings
    SETTING_COLS = ['setting1', 'setting2', 'setting3']
    
    def __init__(self, 
                 sensor_columns: Optional[List[str]] = None,
                 use_regime_normalization: bool = False,
                 n_regimes: int = 6):
        """
        Initialize preprocessor.
        
        Args:
            sensor_columns: List of sensor columns to use. Defaults to USEFUL_SENSORS.
            use_regime_normalization: If True, normalize sensors per operating regime.
                                     Recommended for FD002/FD004 (multi-condition).
            n_regimes: Number of operating regimes for KMeans clustering.
        """
        self.sensor_columns = sensor_columns or self.USEFUL_SENSORS
        self.use_regime_normalization = use_regime_normalization
        self.n_regimes = n_regimes
        
        self.scaler = StandardScaler()
        self.kmeans = None  # For regime clustering
        self.regime_stats = {}  # Mean/std per regime per sensor
        self.feature_columns = None
        self.fitted = False

    def fit(self, data: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            data: Training DataFrame with sensor columns
        """
        if self.use_regime_normalization:
            self._fit_regime_normalizer(data)
            self.feature_columns = [f'{c}_norm' for c in self.sensor_columns]
        else:
            self.feature_columns = self.sensor_columns
        
        # Fit scaler on sensor values
        X = data[self.sensor_columns].values
        X = np.nan_to_num(X, nan=0.0)
        self.scaler.fit(X)
        
        self.fitted = True
        return self

    def _fit_regime_normalizer(self, data: pd.DataFrame):
        """Fit KMeans on operational settings to identify operating regimes."""
        settings = data[self.SETTING_COLS].values
        
        self.kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        regimes = self.kmeans.fit_predict(settings)
        
        # Compute per-regime statistics
        data_temp = data.copy()
        data_temp['regime'] = regimes
        
        for regime in range(self.n_regimes):
            regime_data = data_temp[data_temp['regime'] == regime]
            self.regime_stats[regime] = {}
            for col in self.sensor_columns:
                self.regime_stats[regime][col] = {
                    'mean': regime_data[col].mean(),
                    'std': regime_data[col].std() + 1e-8  # Avoid division by zero
                }

    def transform(self, data: pd.DataFrame, return_df: bool = False) -> np.ndarray:
        """
        Transform sensor data to feature matrix.
        
        Args:
            data: DataFrame with sensor columns
            return_df: If True, return DataFrame instead of numpy array
            
        Returns:
            Scaled feature matrix (numpy array or DataFrame)
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        df = data.copy()
        
        # Ensure all sensor columns exist (handle missing with defaults)
        for col in self.sensor_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        if self.use_regime_normalization and self.kmeans is not None:
            # Assign operating regimes
            settings = df[self.SETTING_COLS].values
            regimes = self.kmeans.predict(settings)
            df['regime'] = regimes
            
            # Normalize each sensor by regime statistics
            for col in self.sensor_columns:
                normalized = np.zeros(len(df))
                for regime in range(self.n_regimes):
                    mask = df['regime'] == regime
                    if mask.sum() > 0:
                        stats = self.regime_stats[regime][col]
                        normalized[mask] = (df.loc[mask, col] - stats['mean']) / stats['std']
                df[f'{col}_norm'] = normalized
            
            X = df[self.feature_columns].values
        else:
            X = df[self.sensor_columns].values
            X = self.scaler.transform(X)
        
        X = np.nan_to_num(X, nan=0.0)
        
        if return_df:
            return pd.DataFrame(X, columns=self.feature_columns, index=df.index)
        
        return X

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)

    def save(self, filepath: str):
        """Save preprocessor state to file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        state = {
            'scaler': self.scaler,
            'sensor_columns': self.sensor_columns,
            'feature_columns': self.feature_columns,
            'use_regime_normalization': self.use_regime_normalization,
            'n_regimes': self.n_regimes,
            'kmeans': self.kmeans,
            'regime_stats': self.regime_stats,
            'fitted': self.fitted
        }
        joblib.dump(state, filepath)

    def load(self, filepath: str) -> 'DataPreprocessor':
        """Load preprocessor state from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        
        state = joblib.load(filepath)
        
        # Handle both old (scaler only) and new (full state) formats
        if isinstance(state, dict):
            self.scaler = state.get('scaler', StandardScaler())
            self.sensor_columns = state.get('sensor_columns', self.USEFUL_SENSORS)
            self.feature_columns = state.get('feature_columns', self.sensor_columns)
            self.use_regime_normalization = state.get('use_regime_normalization', False)
            self.n_regimes = state.get('n_regimes', 6)
            self.kmeans = state.get('kmeans')
            self.regime_stats = state.get('regime_stats', {})
            self.fitted = state.get('fitted', True)
        else:
            # Legacy format: just the scaler
            self.scaler = state
            self.feature_columns = self.sensor_columns
            self.fitted = True
        
        return self


# Backward compatibility alias
CMAPSSPreprocessor = DataPreprocessor
