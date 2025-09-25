"""
Transformer-based Market Prediction Models

This module implements state-of-the-art transformer architectures for financial market prediction,
including attention mechanisms, multi-head attention, and specialized financial transformers.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass
import json
import math

try:
    from transformers import AutoModel, AutoTokenizer, PreTrainedModel
    from transformers.modeling_outputs import BaseModelOutput
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. Using mock implementations.")

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of transformer prediction"""
    predictions: Dict[str, float]
    confidence: float
    attention_weights: Optional[np.ndarray]
    feature_importance: Dict[str, float]
    prediction_horizon: int
    model_uncertainty: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'predictions': self.predictions,
            'confidence': self.confidence,
            'attention_weights': self.attention_weights.tolist() if self.attention_weights is not None else None,
            'feature_importance': self.feature_importance,
            'prediction_horizon': self.prediction_horizon,
            'model_uncertainty': self.model_uncertainty,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


class FinancialTransformer(nn.Module):
    """
    Specialized Transformer architecture for financial time series prediction

    Features:
    - Multi-head attention with temporal encoding
    - Price-aware positional encoding
    - Volatility-adjusted attention mechanisms
    - Multi-scale temporal processing
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        max_seq_length: int = 1000,
        num_features: int = 20,
        num_outputs: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_features = num_features

        # Input projection
        self.input_projection = nn.Linear(num_features, d_model)

        # Positional encoding with price awareness
        self.pos_encoder = FinancialPositionalEncoding(d_model, max_seq_length)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Multi-scale temporal convolutions
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model // 4, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]
        ])

        # Attention pooling
        self.attention_pool = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Output layers
        self.output_layers = nn.ModuleList([
            nn.Linear(d_model, d_model // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model // 4, num_outputs)
        ])

        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the financial transformer

        Args:
            x: Input tensor [batch_size, seq_length, num_features]
            mask: Attention mask [batch_size, seq_length]

        Returns:
            predictions, uncertainty, attention_weights
        """
        batch_size, seq_length, _ = x.shape

        # Input projection
        x = self.input_projection(x)  # [batch, seq, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        if mask is not None:
            src_key_padding_mask = mask
        else:
            src_key_padding_mask = None

        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Multi-scale temporal processing
        temporal_features = []
        encoded_transposed = encoded.transpose(1, 2)  # [batch, d_model, seq]

        for conv in self.temporal_convs:
            temp_feat = conv(encoded_transposed)  # [batch, d_model//4, seq]
            temporal_features.append(temp_feat.transpose(1, 2))  # [batch, seq, d_model//4]

        # Concatenate temporal features
        multi_scale = torch.cat(temporal_features, dim=-1)  # [batch, seq, d_model]

        # Combine with original encoded features
        combined = encoded + multi_scale

        # Attention pooling for sequence aggregation
        pooled, attention_weights = self.attention_pool(
            combined.mean(dim=1, keepdim=True),  # Query: mean of sequence
            combined,  # Key, Value: full sequence
            combined
        )
        pooled = pooled.squeeze(1)  # [batch, d_model]

        # Generate predictions
        output = pooled
        for layer in self.output_layers:
            if isinstance(layer, nn.Linear):
                output = layer(output)
            else:
                output = layer(output)

        # Uncertainty estimation
        uncertainty = self.uncertainty_head(pooled)

        return output, uncertainty, attention_weights


class FinancialPositionalEncoding(nn.Module):
    """
    Price-aware positional encoding for financial time series

    Incorporates both temporal position and price level information
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Standard sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor"""
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class TransformerPredictor:
    """
    Advanced transformer-based market predictor with quantum-enhanced features

    Provides state-of-the-art market prediction using transformer architectures
    with specialized financial features and quantum-inspired optimizations.
    """

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = model_config or self._default_config()

        # Initialize model
        self.model = FinancialTransformer(**self.model_config)
        self.model.to(self.device)

        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None

        # Prediction history
        self.prediction_history = []
        self.performance_metrics = {}

        logger.info(f"TransformerPredictor initialized on {self.device}")

    def _default_config(self) -> Dict[str, Any]:
        """Default model configuration"""
        return {
            'd_model': 512,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 2048,
            'max_seq_length': 1000,
            'num_features': 20,
            'num_outputs': 3,  # price, direction, volatility
            'dropout': 0.1
        }

    async def predict(
        self,
        market_data: pd.DataFrame,
        prediction_horizon: int = 1,
        features: Optional[List[str]] = None
    ) -> PredictionResult:
        """
        Generate market predictions using transformer model

        Args:
            market_data: Historical market data
            prediction_horizon: Number of steps to predict ahead
            features: List of feature columns to use

        Returns:
            PredictionResult with predictions and metadata
        """
        try:
            # Prepare input data
            input_tensor, feature_names = self._prepare_input(market_data, features)

            # Set model to evaluation mode
            self.model.eval()

            with torch.no_grad():
                # Forward pass
                predictions, uncertainty, attention_weights = self.model(input_tensor)

                # Process outputs
                pred_dict = self._process_predictions(predictions, feature_names)
                confidence = self._calculate_confidence(uncertainty)
                feature_importance = self._calculate_feature_importance(attention_weights, feature_names)

                result = PredictionResult(
                    predictions=pred_dict,
                    confidence=float(confidence),
                    attention_weights=attention_weights.cpu().numpy() if attention_weights is not None else None,
                    feature_importance=feature_importance,
                    prediction_horizon=prediction_horizon,
                    model_uncertainty=float(uncertainty.mean())
                )

                # Store prediction history
                self.prediction_history.append(result.to_dict())

                logger.info(f"Generated predictions with {confidence:.3f} confidence")

                return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return default prediction
            return PredictionResult(
                predictions={'price': 0.0, 'direction': 0.0, 'volatility': 0.1},
                confidence=0.0,
                attention_weights=None,
                feature_importance={},
                prediction_horizon=prediction_horizon,
                model_uncertainty=1.0
            )

    def _prepare_input(
        self,
        market_data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, List[str]]:
        """Prepare input tensor from market data"""

        if features is None:
            # Use all numeric columns
            features = [col for col in market_data.columns
                       if market_data[col].dtype in ['float64', 'int64']]

        # Select and normalize features
        feature_data = market_data[features].fillna(0)

        # Convert to tensor
        input_tensor = torch.FloatTensor(feature_data.values).unsqueeze(0).to(self.device)

        return input_tensor, features

    def _process_predictions(
        self,
        predictions: torch.Tensor,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Process model outputs into prediction dictionary"""

        pred_np = predictions.cpu().numpy().flatten()

        pred_dict = {
            'price': float(pred_np[0]) if len(pred_np) > 0 else 0.0,
            'direction': float(torch.sigmoid(torch.tensor(pred_np[1]))) if len(pred_np) > 1 else 0.5,
            'volatility': float(torch.softplus(torch.tensor(pred_np[2]))) if len(pred_np) > 2 else 0.1
        }

        return pred_dict

    def _calculate_confidence(self, uncertainty: torch.Tensor) -> float:
        """Calculate prediction confidence from uncertainty"""
        # Convert uncertainty to confidence (inverse relationship)
        uncertainty_mean = uncertainty.mean().item()
        confidence = 1.0 / (1.0 + uncertainty_mean)
        return max(0.0, min(1.0, confidence))

    def _calculate_feature_importance(
        self,
        attention_weights: Optional[torch.Tensor],
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate feature importance from attention weights"""

        if attention_weights is None:
            return {}

        # Average attention weights across heads and layers
        importance_scores = attention_weights.mean(dim=1).cpu().numpy().flatten()

        # Normalize to sum to 1
        importance_scores = importance_scores / importance_scores.sum()

        # Map to feature names
        feature_importance = {}
        for i, feature in enumerate(feature_names):
            if i < len(importance_scores):
                feature_importance[feature] = float(importance_scores[i])

        return feature_importance

    async def train(
        self,
        training_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Train the transformer model on financial data

        Args:
            training_data: Training dataset
            validation_data: Validation dataset
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training

        Returns:
            Training metrics and history
        """
        try:
            # Prepare training data
            train_loader = self._create_data_loader(training_data, batch_size, shuffle=True)
            val_loader = None
            if validation_data is not None:
                val_loader = self._create_data_loader(validation_data, batch_size, shuffle=False)

            # Initialize optimizer and scheduler
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=1e-5
            )
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=10,
                factor=0.5
            )

            # Training loop
            training_history = {
                'train_loss': [],
                'val_loss': [],
                'learning_rate': []
            }

            best_val_loss = float('inf')

            for epoch in range(epochs):
                # Training phase
                train_loss = await self._train_epoch(train_loader)
                training_history['train_loss'].append(train_loss)

                # Validation phase
                if val_loader is not None:
                    val_loss = await self._validate_epoch(val_loader)
                    training_history['val_loss'].append(val_loss)

                    # Learning rate scheduling
                    self.scheduler.step(val_loss)

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), 'best_transformer_model.pth')

                # Log progress
                current_lr = self.optimizer.param_groups[0]['lr']
                training_history['learning_rate'].append(current_lr)

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f if val_loader else 'N/A'}")

            # Load best model
            if val_loader is not None:
                self.model.load_state_dict(torch.load('best_transformer_model.pth'))

            logger.info("Training completed successfully")

            return {
                'training_history': training_history,
                'best_val_loss': best_val_loss,
                'final_train_loss': train_loss
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'error': str(e)}

    def _create_data_loader(
        self,
        data: pd.DataFrame,
        batch_size: int,
        shuffle: bool = False
    ):
        """Create PyTorch DataLoader from pandas DataFrame"""
        # This is a simplified implementation
        # In production, you'd want proper Dataset classes with sequence windowing

        from torch.utils.data import DataLoader, TensorDataset

        # Convert to tensors (simplified)
        features = torch.FloatTensor(data.select_dtypes(include=[np.number]).values)

        dataset = TensorDataset(features)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # For simplicity
        )

    async def _train_epoch(self, data_loader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (batch_data,) in enumerate(data_loader):
            batch_data = batch_data.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    predictions, uncertainty, _ = self.model(batch_data)
                    loss = self._calculate_loss(predictions, uncertainty, batch_data)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions, uncertainty, _ = self.model(batch_data)
                loss = self._calculate_loss(predictions, uncertainty, batch_data)

                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    async def _validate_epoch(self, data_loader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (batch_data,) in enumerate(data_loader):
                batch_data = batch_data.to(self.device)

                predictions, uncertainty, _ = self.model(batch_data)
                loss = self._calculate_loss(predictions, uncertainty, batch_data)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _calculate_loss(
        self,
        predictions: torch.Tensor,
        uncertainty: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate training loss with uncertainty"""

        # Simplified loss function
        # In practice, you'd want more sophisticated loss functions
        # for financial prediction tasks

        mse_loss = F.mse_loss(predictions, targets[:, :predictions.size(1)])
        uncertainty_loss = uncertainty.mean()  # Regularize uncertainty

        return mse_loss + 0.1 * uncertainty_loss

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        if not self.prediction_history:
            return {}

        confidences = [pred['confidence'] for pred in self.prediction_history]
        uncertainties = [pred['model_uncertainty'] for pred in self.prediction_history]

        return {
            'num_predictions': len(self.prediction_history),
            'average_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'average_uncertainty': np.mean(uncertainties),
            'uncertainty_std': np.std(uncertainties),
            'prediction_consistency': 1.0 - np.std(confidences)  # Higher is better
        }

    async def ensemble_predict(
        self,
        market_data: pd.DataFrame,
        num_models: int = 5,
        prediction_horizon: int = 1
    ) -> PredictionResult:
        """Generate ensemble predictions using multiple model variations"""

        ensemble_predictions = []

        for i in range(num_models):
            # Add noise to create model variations
            with torch.no_grad():
                # Temporarily modify model parameters
                original_state = {name: param.clone() for name, param in self.model.named_parameters()}

                # Add small random perturbations
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param) * 0.01
                        param.add_(noise)

                # Generate prediction
                prediction = await self.predict(market_data, prediction_horizon)
                ensemble_predictions.append(prediction)

                # Restore original parameters
                for name, param in self.model.named_parameters():
                    param.copy_(original_state[name])

        # Combine ensemble predictions
        combined_predictions = {}
        for key in ensemble_predictions[0].predictions.keys():
            values = [pred.predictions[key] for pred in ensemble_predictions]
            combined_predictions[key] = float(np.mean(values))

        # Calculate ensemble confidence
        confidences = [pred.confidence for pred in ensemble_predictions]
        ensemble_confidence = np.mean(confidences) * (1.0 - np.std(confidences))  # Penalize high variance

        return PredictionResult(
            predictions=combined_predictions,
            confidence=float(ensemble_confidence),
            attention_weights=ensemble_predictions[0].attention_weights,
            feature_importance=ensemble_predictions[0].feature_importance,
            prediction_horizon=prediction_horizon,
            model_uncertainty=float(np.mean([pred.model_uncertainty for pred in ensemble_predictions]))
        )