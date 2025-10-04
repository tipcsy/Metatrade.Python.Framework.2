"""
Model Training Pipeline
Handles complete training workflow from data loading to model saving
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import requests

from .lstm_model import LSTMForexModel
from .data_preprocessor import ForexDataPreprocessor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles LSTM model training workflow"""

    def __init__(self, data_service_url: str = "http://localhost:5002"):
        """
        Initialize model trainer

        Args:
            data_service_url: URL of the data service
        """
        self.data_service_url = data_service_url
        self.preprocessor = None
        self.model = None

    async def train_model(self, config: Dict[str, Any]):
        """
        Complete training pipeline

        Args:
            config: Training configuration dictionary containing:
                - symbol: Trading symbol (e.g., "EURUSD")
                - timeframe: Timeframe (e.g., "M15")
                - start_date: Start date for training data
                - end_date: End date for training data
                - parameters: Model hyperparameters

        Returns:
            Dictionary with training results and model metadata
        """
        try:
            logger.info(f"Starting model training for {config['symbol']} {config['timeframe']}")

            # Extract configuration
            symbol = config['symbol']
            timeframe = config['timeframe']
            start_date = config['start_date']
            end_date = config['end_date']
            params = config.get('parameters', {})

            # Training parameters
            sequence_length = params.get('sequence_length', 60)
            lstm_units = params.get('lstm_units', 50)
            num_lstm_layers = params.get('num_lstm_layers', 3)
            dropout_rate = params.get('dropout_rate', 0.2)
            epochs = params.get('epochs', 50)
            batch_size = params.get('batch_size', 32)
            train_ratio = params.get('train_ratio', 0.8)
            add_indicators = params.get('add_indicators', True)

            # Step 1: Load data from Data Service
            logger.info("Fetching historical data from Data Service")
            data = await self._fetch_historical_data(symbol, timeframe, start_date, end_date)

            if data is None or len(data) == 0:
                raise ValueError("No data received from Data Service")

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Step 2: Initialize preprocessor
            self.preprocessor = ForexDataPreprocessor(sequence_length=sequence_length)

            # Validate data
            if not self.preprocessor.validate_data(df, min_rows=sequence_length + 1000):
                raise ValueError("Data validation failed")

            # Step 3: Prepare data
            logger.info("Preprocessing data")
            scaled_data, scaler = self.preprocessor.prepare_data(df, add_indicators=add_indicators)

            # Step 4: Create sequences
            X, y = self.preprocessor.create_sequences(scaled_data, target_column_index=3)

            # Step 5: Split data
            X_train, y_train, X_val, y_val = self.preprocessor.split_data(X, y, train_ratio=train_ratio)

            # Step 6: Build model
            logger.info("Building LSTM model")
            n_features = X_train.shape[2]
            self.model = LSTMForexModel(sequence_length=sequence_length, n_features=n_features)
            self.model.build_model(
                lstm_units=lstm_units,
                num_lstm_layers=num_lstm_layers,
                dropout_rate=dropout_rate
            )

            # Step 7: Train model
            logger.info(f"Training model: epochs={epochs}, batch_size={batch_size}")

            # Create model save path
            model_id = self._generate_model_id(symbol, timeframe)
            model_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models')
            os.makedirs(model_dir, exist_ok=True)

            model_path = os.path.join(model_dir, f"{model_id}.h5")
            scaler_path = os.path.join(model_dir, f"{model_id}_scaler.pkl")

            # Train
            history = self.model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=epochs,
                batch_size=batch_size,
                model_save_path=model_path
            )

            # Step 8: Evaluate model
            logger.info("Evaluating model")
            eval_metrics = self.model.evaluate(X_val, y_val)

            # Step 9: Save scaler
            self.preprocessor.save_scaler(scaler_path)

            # Step 10: Create metadata
            metadata = {
                'model_id': model_id,
                'symbol': symbol,
                'timeframe': timeframe,
                'model_type': 'LSTM',
                'training_date': datetime.now().isoformat(),
                'training_data_period': {
                    'start': start_date,
                    'end': end_date
                },
                'data_points': len(df),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'parameters': {
                    'sequence_length': sequence_length,
                    'lstm_units': lstm_units,
                    'num_lstm_layers': num_lstm_layers,
                    'dropout_rate': dropout_rate,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'add_indicators': add_indicators
                },
                'metrics': {
                    'val_loss': float(eval_metrics['loss']),
                    'val_mae': float(eval_metrics['mae']),
                    'train_loss': float(history.history['loss'][-1]),
                    'train_mae': float(history.history['mae'][-1])
                },
                'n_features': n_features,
                'feature_columns': self.preprocessor.feature_columns,
                'status': 'active',
                'model_path': model_path,
                'scaler_path': scaler_path
            }

            # Save metadata
            metadata_path = os.path.join(model_dir, f"{model_id}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model training completed successfully: {model_id}")

            return {
                'success': True,
                'model_id': model_id,
                'metadata': metadata,
                'training_history': {
                    'epochs': len(history.history['loss']),
                    'final_train_loss': float(history.history['loss'][-1]),
                    'final_val_loss': float(history.history['val_loss'][-1]),
                    'best_val_loss': float(min(history.history['val_loss']))
                }
            }

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _fetch_historical_data(self, symbol: str, timeframe: str,
                                     start_date: str, end_date: str):
        """
        Fetch historical data from Data Service

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of OHLC data dictionaries
        """
        try:
            # Call Data Service API
            url = f"{self.data_service_url}/data/historical"
            params = {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date
            }

            response = requests.get(url, params=params, timeout=30)

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
            logger.error(f"Error fetching historical data: {e}")
            return None

    def _generate_model_id(self, symbol: str, timeframe: str):
        """
        Generate unique model ID

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Model ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{symbol.lower()}_{timeframe.lower()}_lstm_{timestamp}"
        return model_id


class TrainingJobManager:
    """Manages async training jobs"""

    def __init__(self):
        self.jobs = {}  # job_id -> job_info
        self.job_counter = 0

    def create_job(self, config: Dict[str, Any]):
        """
        Create a new training job

        Args:
            config: Training configuration

        Returns:
            Job ID
        """
        self.job_counter += 1
        job_id = f"train-{self.job_counter}"

        self.jobs[job_id] = {
            'job_id': job_id,
            'status': 'queued',
            'progress': 0,
            'config': config,
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'completed_at': None,
            'result': None,
            'error': None
        }

        logger.info(f"Training job created: {job_id}")

        return job_id

    def update_job_status(self, job_id: str, status: str, **kwargs):
        """
        Update job status

        Args:
            job_id: Job ID
            status: New status ('queued', 'running', 'completed', 'failed')
            **kwargs: Additional fields to update
        """
        if job_id in self.jobs:
            self.jobs[job_id]['status'] = status

            if status == 'running' and self.jobs[job_id]['started_at'] is None:
                self.jobs[job_id]['started_at'] = datetime.now().isoformat()

            if status in ['completed', 'failed']:
                self.jobs[job_id]['completed_at'] = datetime.now().isoformat()

            # Update additional fields
            for key, value in kwargs.items():
                self.jobs[job_id][key] = value

            logger.info(f"Job {job_id} status updated: {status}")

    def get_job_status(self, job_id: str):
        """Get job status"""
        return self.jobs.get(job_id, None)

    def get_all_jobs(self):
        """Get all jobs"""
        return list(self.jobs.values())
