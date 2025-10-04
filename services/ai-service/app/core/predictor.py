"""
Predictor Module
Handles model loading and inference for price predictions
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import requests

from .lstm_model import LSTMForexModel
from .data_preprocessor import ForexDataPreprocessor

logger = logging.getLogger(__name__)


class ForexPredictor:
    """Handles LSTM model inference for forex price prediction"""

    def __init__(self, data_service_url: str = "http://localhost:5002"):
        """
        Initialize predictor

        Args:
            data_service_url: URL of the data service
        """
        self.data_service_url = data_service_url
        self.loaded_models = {}  # model_id -> (model, preprocessor, metadata)

    def load_model(self, model_id: str) -> bool:
        """
        Load model into memory

        Args:
            model_id: Model identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if already loaded
            if model_id in self.loaded_models:
                logger.info(f"Model {model_id} already loaded")
                return True

            # Find model files
            model_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models')
            model_path = os.path.join(model_dir, f"{model_id}.h5")
            scaler_path = os.path.join(model_dir, f"{model_id}_scaler.pkl")
            metadata_path = os.path.join(model_dir, f"{model_id}_metadata.json")

            # Check if files exist
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False

            if not os.path.exists(scaler_path):
                logger.error(f"Scaler file not found: {scaler_path}")
                return False

            if not os.path.exists(metadata_path):
                logger.error(f"Metadata file not found: {metadata_path}")
                return False

            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Initialize model
            sequence_length = metadata['parameters']['sequence_length']
            n_features = metadata['n_features']

            lstm_model = LSTMForexModel(sequence_length=sequence_length, n_features=n_features)
            lstm_model.load_model(model_path)

            # Initialize preprocessor
            preprocessor = ForexDataPreprocessor(sequence_length=sequence_length)
            preprocessor.load_scaler(scaler_path)
            preprocessor.feature_columns = metadata['feature_columns']

            # Store in cache
            self.loaded_models[model_id] = {
                'model': lstm_model,
                'preprocessor': preprocessor,
                'metadata': metadata
            }

            logger.info(f"Model loaded successfully: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return False

    async def predict(self, model_id: str, symbol: str, timeframe: str,
                     steps_ahead: int = 1, use_confidence: bool = True) -> Dict[str, Any]:
        """
        Make price prediction

        Args:
            model_id: Model identifier
            symbol: Trading symbol
            timeframe: Timeframe
            steps_ahead: Number of steps to predict (default 1)
            use_confidence: Whether to calculate confidence score (default True)

        Returns:
            Dictionary with prediction results
        """
        try:
            # Load model if not already loaded
            if model_id not in self.loaded_models:
                if not self.load_model(model_id):
                    return {
                        'success': False,
                        'error': f'Failed to load model: {model_id}'
                    }

            model_data = self.loaded_models[model_id]
            lstm_model = model_data['model']
            preprocessor = model_data['preprocessor']
            metadata = model_data['metadata']

            # Verify symbol and timeframe match
            if metadata['symbol'] != symbol or metadata['timeframe'] != timeframe:
                logger.warning(f"Model trained on {metadata['symbol']} {metadata['timeframe']}, "
                             f"but prediction requested for {symbol} {timeframe}")

            # Fetch recent data
            sequence_length = metadata['parameters']['sequence_length']
            logger.info(f"Fetching last {sequence_length} bars for prediction")

            recent_data = await self._fetch_recent_data(symbol, timeframe, sequence_length + 50)

            if recent_data is None or len(recent_data) < sequence_length:
                return {
                    'success': False,
                    'error': 'Insufficient recent data for prediction'
                }

            # Convert to DataFrame
            df = pd.DataFrame(recent_data)

            # Prepare data
            add_indicators = metadata['parameters'].get('add_indicators', True)

            # Prepare data using same method as training
            if add_indicators:
                df = preprocessor._add_technical_indicators(df)

            # Select features
            feature_data = df[preprocessor.feature_columns].values

            # Handle NaN from indicators
            if np.isnan(feature_data).any():
                df = df.fillna(method='ffill').fillna(method='bfill')
                feature_data = df[preprocessor.feature_columns].values

            # Normalize
            scaled_data = preprocessor.scaler.transform(feature_data)

            # Get last sequence
            last_sequence = scaled_data[-sequence_length:]
            last_sequence = last_sequence.reshape(1, sequence_length, len(preprocessor.feature_columns))

            # Make prediction
            if use_confidence:
                prediction_scaled, confidence = lstm_model.predict_with_confidence(last_sequence)
                confidence_score = float(confidence[0][0])
            else:
                prediction_scaled = lstm_model.predict(last_sequence)
                confidence_score = None

            # Inverse transform prediction
            predicted_price = preprocessor.inverse_transform(prediction_scaled, feature_index=3)

            # Get current price
            current_price = float(df['close'].iloc[-1])

            # Determine direction
            direction = 'UP' if predicted_price[0] > current_price else 'DOWN'

            # Calculate price change
            price_change = predicted_price[0] - current_price
            price_change_pct = (price_change / current_price) * 100

            result = {
                'success': True,
                'model_id': model_id,
                'prediction': {
                    'price': float(predicted_price[0]),
                    'current_price': current_price,
                    'price_change': float(price_change),
                    'price_change_pct': float(price_change_pct),
                    'direction': direction,
                    'confidence': confidence_score,
                    'steps_ahead': steps_ahead,
                    'timestamp': pd.Timestamp.now().isoformat()
                },
                'input_data': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'last_close': current_price,
                    'sequence_length': sequence_length
                }
            }

            logger.info(f"Prediction completed: {symbol} {timeframe} -> {predicted_price[0]:.5f} "
                       f"({direction}, confidence: {confidence_score})")

            return result

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _fetch_recent_data(self, symbol: str, timeframe: str, count: int):
        """
        Fetch recent OHLC data from Data Service

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            count: Number of bars to fetch

        Returns:
            List of OHLC data dictionaries
        """
        try:
            # Call Data Service API
            url = f"{self.data_service_url}/data/ohlc"
            params = {
                'symbol': symbol,
                'timeframe': timeframe,
                'count': count
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return result.get('data', [])
                else:
                    logger.error(f"Data Service returned error: {result.get('error')}")
                    return None
            else:
                logger.error(f"Data Service request failed: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error fetching recent data: {e}")
            return None

    def unload_model(self, model_id: str) -> bool:
        """
        Unload model from memory

        Args:
            model_id: Model identifier

        Returns:
            True if successful
        """
        try:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
                logger.info(f"Model unloaded: {model_id}")
                return True
            else:
                logger.warning(f"Model not loaded: {model_id}")
                return False

        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {e}")
            return False

    def get_loaded_models(self):
        """Get list of loaded models"""
        return list(self.loaded_models.keys())

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a loaded model

        Args:
            model_id: Model identifier

        Returns:
            Model metadata or None
        """
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]['metadata']
        return None
