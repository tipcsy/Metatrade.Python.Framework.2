"""
Data Preprocessing for LSTM Model Training
Handles normalization, sequence generation, and feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging
import pickle

logger = logging.getLogger(__name__)


class ForexDataPreprocessor:
    """Preprocesses forex OHLC data for LSTM training"""

    def __init__(self, sequence_length: int = 60):
        """
        Initialize preprocessor

        Args:
            sequence_length: Number of time steps for sequence (default 60)
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume']

    def prepare_data(self, df: pd.DataFrame, add_indicators: bool = True):
        """
        Prepare data for LSTM training

        Args:
            df: DataFrame with OHLC data
            add_indicators: Whether to add technical indicators (default True)

        Returns:
            Tuple of (scaled_data, scaler)
        """
        try:
            # Make a copy to avoid modifying original
            data = df.copy()

            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in data.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Add technical indicators if requested
            if add_indicators:
                data = self._add_technical_indicators(data)
                # Update feature columns
                self.feature_columns = [col for col in data.columns if col not in ['time', 'timestamp']]

            # Select only numeric features
            feature_data = data[self.feature_columns].values

            # Check for NaN values
            if np.isnan(feature_data).any():
                logger.warning("NaN values detected, filling with forward fill")
                data = data.fillna(method='ffill').fillna(method='bfill')
                feature_data = data[self.feature_columns].values

            # Normalize data to [0, 1]
            scaled_data = self.scaler.fit_transform(feature_data)

            logger.info(f"Data prepared: shape={scaled_data.shape}, features={len(self.feature_columns)}")

            return scaled_data, self.scaler

        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise

    def create_sequences(self, data: np.ndarray, target_column_index: int = 3):
        """
        Create sequences for LSTM training using sliding window

        Args:
            data: Scaled data array
            target_column_index: Index of target column (default 3 for 'close')

        Returns:
            Tuple of (X, y) where:
                X: shape [samples, sequence_length, n_features]
                y: shape [samples, 1]
        """
        try:
            X, y = [], []

            for i in range(self.sequence_length, len(data)):
                # Input: last 'sequence_length' bars
                X.append(data[i - self.sequence_length:i])
                # Output: next close price
                y.append(data[i, target_column_index])

            X = np.array(X)
            y = np.array(y)

            logger.info(f"Sequences created: X shape={X.shape}, y shape={y.shape}")

            return X, y

        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            raise

    def split_data(self, X, y, train_ratio: float = 0.8):
        """
        Split data into train and validation sets

        Args:
            X: Input sequences
            y: Target values
            train_ratio: Ratio of training data (default 0.8)

        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        try:
            split_idx = int(len(X) * train_ratio)

            X_train = X[:split_idx]
            y_train = y[:split_idx]
            X_val = X[split_idx:]
            y_val = y[split_idx:]

            logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}")

            return X_train, y_train, X_val, y_val

        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise

    def _add_technical_indicators(self, df: pd.DataFrame):
        """
        Add technical indicators to dataframe

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with added indicators
        """
        try:
            data = df.copy()

            # Exponential Moving Averages
            data['ema_9'] = data['close'].ewm(span=9, adjust=False).mean()
            data['ema_21'] = data['close'].ewm(span=21, adjust=False).mean()
            data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()

            # RSI (Relative Strength Index)
            data['rsi'] = self._calculate_rsi(data['close'], period=14)

            # MACD (Moving Average Convergence Divergence)
            ema_12 = data['close'].ewm(span=12, adjust=False).mean()
            ema_26 = data['close'].ewm(span=26, adjust=False).mean()
            data['macd'] = ema_12 - ema_26
            data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

            # ATR (Average True Range)
            data['atr'] = self._calculate_atr(data, period=14)

            # Bollinger Bands
            data['bb_middle'] = data['close'].rolling(window=20).mean()
            bb_std = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)

            logger.info("Technical indicators added successfully")

            return data

        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            raise

    def _calculate_rsi(self, prices: pd.Series, period: int = 14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14):
        """Calculate ATR indicator"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        atr = true_range.rolling(window=period).mean()

        return atr

    def inverse_transform(self, scaled_data: np.ndarray, feature_index: int = 3):
        """
        Convert normalized data back to original scale

        Args:
            scaled_data: Normalized data
            feature_index: Index of feature to inverse transform (default 3 for 'close')

        Returns:
            Data in original scale
        """
        try:
            # Create dummy array with same shape as original features
            dummy = np.zeros((len(scaled_data), len(self.feature_columns)))
            dummy[:, feature_index] = scaled_data.flatten()

            # Inverse transform
            inverse_scaled = self.scaler.inverse_transform(dummy)

            # Return only the target column
            return inverse_scaled[:, feature_index]

        except Exception as e:
            logger.error(f"Error in inverse transform: {e}")
            raise

    def save_scaler(self, filepath: str):
        """Save scaler to file"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Scaler saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving scaler: {e}")
            raise

    def load_scaler(self, filepath: str):
        """Load scaler from file"""
        try:
            with open(filepath, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Scaler loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            raise

    def validate_data(self, df: pd.DataFrame, min_rows: int = 10000):
        """
        Validate data quality

        Args:
            df: DataFrame to validate
            min_rows: Minimum required rows (default 10000)

        Returns:
            Boolean indicating if data is valid
        """
        try:
            # Check minimum rows
            if len(df) < min_rows:
                logger.error(f"Insufficient data: {len(df)} rows (minimum {min_rows} required)")
                return False

            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    return False

            # Check for excessive NaN values
            nan_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if nan_ratio > 0.1:  # More than 10% NaN
                logger.error(f"Too many NaN values: {nan_ratio*100:.2f}%")
                return False

            logger.info("Data validation passed")
            return True

        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False
