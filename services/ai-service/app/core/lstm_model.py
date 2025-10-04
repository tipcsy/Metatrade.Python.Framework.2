"""
LSTM Model Architecture for Forex Time Series Forecasting
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging

logger = logging.getLogger(__name__)


class LSTMForexModel:
    """LSTM model for forex price prediction"""

    def __init__(self, sequence_length: int = 60, n_features: int = 5):
        """
        Initialize LSTM model

        Args:
            sequence_length: Number of time steps to look back (default 60)
            n_features: Number of input features (OHLC + Volume = 5)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None

    def build_model(self, lstm_units: int = 50, num_lstm_layers: int = 3,
                   dropout_rate: float = 0.2, learning_rate: float = 0.001):
        """
        Build LSTM model architecture

        Args:
            lstm_units: Number of LSTM units per layer (default 50)
            num_lstm_layers: Number of LSTM layers (default 3)
            dropout_rate: Dropout rate between layers (default 0.2)
            learning_rate: Learning rate for optimizer (default 0.001)
        """
        try:
            model = Sequential(name='LSTM_Forex_Predictor')

            # First LSTM layer
            model.add(LSTM(
                units=lstm_units,
                return_sequences=True if num_lstm_layers > 1 else False,
                input_shape=(self.sequence_length, self.n_features),
                name='lstm_1'
            ))
            model.add(Dropout(dropout_rate, name='dropout_1'))

            # Additional LSTM layers
            for i in range(2, num_lstm_layers + 1):
                return_seq = i < num_lstm_layers
                model.add(LSTM(
                    units=lstm_units,
                    return_sequences=return_seq,
                    name=f'lstm_{i}'
                ))
                model.add(Dropout(dropout_rate, name=f'dropout_{i}'))

            # Output layer (predicts next close price)
            model.add(Dense(units=1, name='output'))

            # Compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mae']
            )

            self.model = model
            logger.info(f"LSTM model built successfully with {num_lstm_layers} layers and {lstm_units} units")

            return model

        except Exception as e:
            logger.error(f"Error building LSTM model: {e}")
            raise

    def train(self, X_train, y_train, X_val, y_val, epochs: int = 50,
             batch_size: int = 32, model_save_path: str = None):
        """
        Train the LSTM model

        Args:
            X_train: Training data (shape: [samples, sequence_length, n_features])
            y_train: Training labels (shape: [samples, 1])
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            model_save_path: Path to save the best model

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        try:
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                )
            ]

            # Add model checkpoint if save path provided
            if model_save_path:
                callbacks.append(
                    ModelCheckpoint(
                        filepath=model_save_path,
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=1
                    )
                )

            logger.info(f"Starting model training: epochs={epochs}, batch_size={batch_size}")

            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

            logger.info("Model training completed successfully")

            return history

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def predict(self, X, batch_size: int = 32):
        """
        Make predictions

        Args:
            X: Input data (shape: [samples, sequence_length, n_features])
            batch_size: Batch size for prediction

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")

        try:
            predictions = self.model.predict(X, batch_size=batch_size, verbose=0)
            return predictions

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def predict_with_confidence(self, X, n_iterations: int = 10):
        """
        Make predictions with confidence score
        Uses Monte Carlo Dropout for uncertainty estimation

        Args:
            X: Input data
            n_iterations: Number of forward passes with dropout

        Returns:
            Tuple of (mean_prediction, confidence_score)
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")

        try:
            import numpy as np

            # Enable dropout during inference
            predictions = []
            for _ in range(n_iterations):
                pred = self.model(X, training=True)  # training=True keeps dropout active
                predictions.append(pred.numpy())

            predictions = np.array(predictions)

            # Calculate mean and std
            mean_prediction = np.mean(predictions, axis=0)
            std_prediction = np.std(predictions, axis=0)

            # Confidence score (inverse of normalized std)
            # Lower variance = higher confidence
            confidence = 1.0 - np.tanh(std_prediction)  # Maps to [0, 1]

            return mean_prediction, confidence

        except Exception as e:
            logger.error(f"Error during confidence prediction: {e}")
            raise

    def save_model(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save.")

        try:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, filepath: str):
        """Load model from file"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_summary(self):
        """Get model summary"""
        if self.model is None:
            return "Model not built yet."

        return self.model.summary()

    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data

        Args:
            X_test: Test data
            y_test: Test labels

        Returns:
            Dictionary with loss and metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")

        try:
            results = self.model.evaluate(X_test, y_test, verbose=0)

            metrics = {
                'loss': results[0],
                'mae': results[1]
            }

            logger.info(f"Model evaluation: {metrics}")

            return metrics

        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise
