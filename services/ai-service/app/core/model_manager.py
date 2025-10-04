"""
Model Manager
Handles model lifecycle, versioning, and registry
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages AI model lifecycle and registry"""

    def __init__(self, models_dir: str = None):
        """
        Initialize model manager

        Args:
            models_dir: Directory where models are stored
        """
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models')

        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

        logger.info(f"Model Manager initialized: {self.models_dir}")

    def list_models(self, symbol: str = None, timeframe: str = None,
                   status: str = None) -> List[Dict[str, Any]]:
        """
        List all available models

        Args:
            symbol: Filter by symbol (optional)
            timeframe: Filter by timeframe (optional)
            status: Filter by status (optional)

        Returns:
            List of model metadata dictionaries
        """
        try:
            models = []

            # Find all metadata files
            for filename in os.listdir(self.models_dir):
                if filename.endswith('_metadata.json'):
                    metadata_path = os.path.join(self.models_dir, filename)

                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    # Apply filters
                    if symbol and metadata.get('symbol') != symbol:
                        continue

                    if timeframe and metadata.get('timeframe') != timeframe:
                        continue

                    if status and metadata.get('status') != status:
                        continue

                    models.append(metadata)

            # Sort by training date (most recent first)
            models.sort(key=lambda x: x.get('training_date', ''), reverse=True)

            logger.info(f"Found {len(models)} models")

            return models

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model metadata by ID

        Args:
            model_id: Model identifier

        Returns:
            Model metadata dictionary or None
        """
        try:
            metadata_path = os.path.join(self.models_dir, f"{model_id}_metadata.json")

            if not os.path.exists(metadata_path):
                logger.warning(f"Model metadata not found: {model_id}")
                return None

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            return metadata

        except Exception as e:
            logger.error(f"Error getting model {model_id}: {e}")
            return None

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model and all associated files

        Args:
            model_id: Model identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            # Find all files for this model
            files_to_delete = [
                f"{model_id}.h5",
                f"{model_id}_scaler.pkl",
                f"{model_id}_metadata.json"
            ]

            deleted_count = 0

            for filename in files_to_delete:
                filepath = os.path.join(self.models_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
                    deleted_count += 1
                    logger.info(f"Deleted: {filename}")

            if deleted_count == 0:
                logger.warning(f"No files found for model: {model_id}")
                return False

            logger.info(f"Model deleted successfully: {model_id} ({deleted_count} files)")
            return True

        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            return False

    def update_model_status(self, model_id: str, status: str) -> bool:
        """
        Update model status

        Args:
            model_id: Model identifier
            status: New status ('active', 'inactive', 'deprecated')

        Returns:
            True if successful
        """
        try:
            metadata = self.get_model(model_id)

            if metadata is None:
                return False

            # Update status
            metadata['status'] = status
            metadata['updated_at'] = datetime.now().isoformat()

            # Save metadata
            metadata_path = os.path.join(self.models_dir, f"{model_id}_metadata.json")

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model status updated: {model_id} -> {status}")
            return True

        except Exception as e:
            logger.error(f"Error updating model status: {e}")
            return False

    def get_active_model(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Get the active model for a symbol/timeframe pair

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Active model metadata or None
        """
        try:
            models = self.list_models(symbol=symbol, timeframe=timeframe, status='active')

            if len(models) == 0:
                logger.warning(f"No active model found for {symbol} {timeframe}")
                return None

            # Return the most recent active model
            return models[0]

        except Exception as e:
            logger.error(f"Error getting active model: {e}")
            return None

    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all models

        Returns:
            Dictionary with model statistics
        """
        try:
            all_models = self.list_models()

            stats = {
                'total_models': len(all_models),
                'active_models': len([m for m in all_models if m.get('status') == 'active']),
                'inactive_models': len([m for m in all_models if m.get('status') == 'inactive']),
                'models_by_symbol': {},
                'models_by_timeframe': {},
                'models_by_type': {}
            }

            for model in all_models:
                # Count by symbol
                symbol = model.get('symbol', 'unknown')
                stats['models_by_symbol'][symbol] = stats['models_by_symbol'].get(symbol, 0) + 1

                # Count by timeframe
                timeframe = model.get('timeframe', 'unknown')
                stats['models_by_timeframe'][timeframe] = stats['models_by_timeframe'].get(timeframe, 0) + 1

                # Count by type
                model_type = model.get('model_type', 'unknown')
                stats['models_by_type'][model_type] = stats['models_by_type'].get(model_type, 0) + 1

            return stats

        except Exception as e:
            logger.error(f"Error getting model statistics: {e}")
            return {}

    def activate_model(self, model_id: str, deactivate_others: bool = True) -> bool:
        """
        Activate a model (and optionally deactivate others for the same symbol/timeframe)

        Args:
            model_id: Model identifier to activate
            deactivate_others: Whether to deactivate other models for same symbol/timeframe

        Returns:
            True if successful
        """
        try:
            metadata = self.get_model(model_id)

            if metadata is None:
                logger.error(f"Model not found: {model_id}")
                return False

            # Deactivate other models if requested
            if deactivate_others:
                symbol = metadata['symbol']
                timeframe = metadata['timeframe']

                other_models = self.list_models(symbol=symbol, timeframe=timeframe, status='active')

                for other_model in other_models:
                    if other_model['model_id'] != model_id:
                        self.update_model_status(other_model['model_id'], 'inactive')
                        logger.info(f"Deactivated model: {other_model['model_id']}")

            # Activate this model
            return self.update_model_status(model_id, 'active')

        except Exception as e:
            logger.error(f"Error activating model: {e}")
            return False

    def get_model_performance_history(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model performance history (if tracked)

        Args:
            model_id: Model identifier

        Returns:
            Performance history or None
        """
        try:
            metadata = self.get_model(model_id)

            if metadata is None:
                return None

            # Return metrics and training history
            return {
                'model_id': model_id,
                'training_metrics': metadata.get('metrics', {}),
                'training_date': metadata.get('training_date'),
                'data_period': metadata.get('training_data_period', {}),
                'parameters': metadata.get('parameters', {})
            }

        except Exception as e:
            logger.error(f"Error getting model performance history: {e}")
            return None

    def cleanup_old_models(self, keep_last_n: int = 5, per_symbol_timeframe: bool = True):
        """
        Cleanup old inactive models, keeping only the most recent N

        Args:
            keep_last_n: Number of models to keep per symbol/timeframe
            per_symbol_timeframe: If True, keep N per symbol/timeframe; if False, keep N total

        Returns:
            Number of models deleted
        """
        try:
            deleted_count = 0

            if per_symbol_timeframe:
                # Group models by symbol/timeframe
                model_groups = {}

                all_models = self.list_models(status='inactive')

                for model in all_models:
                    key = f"{model['symbol']}_{model['timeframe']}"
                    if key not in model_groups:
                        model_groups[key] = []
                    model_groups[key].append(model)

                # For each group, delete old models
                for key, models in model_groups.items():
                    # Sort by training date (most recent first)
                    models.sort(key=lambda x: x.get('training_date', ''), reverse=True)

                    # Delete models beyond keep_last_n
                    for model in models[keep_last_n:]:
                        if self.delete_model(model['model_id']):
                            deleted_count += 1

            else:
                # Keep N most recent models overall
                all_models = self.list_models(status='inactive')
                all_models.sort(key=lambda x: x.get('training_date', ''), reverse=True)

                for model in all_models[keep_last_n:]:
                    if self.delete_model(model['model_id']):
                        deleted_count += 1

            logger.info(f"Cleanup completed: {deleted_count} models deleted")

            return deleted_count

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
