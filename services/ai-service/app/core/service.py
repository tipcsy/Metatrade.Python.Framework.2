"""
Core service logic for AI Service
Orchestrates model training, inference, and management
"""

import logging
import asyncio
from typing import Dict, Any

from .model_trainer import ModelTrainer, TrainingJobManager
from .predictor import ForexPredictor
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class AIService:
    """Main AI Service orchestrator"""

    def __init__(self, data_service_url: str = "http://localhost:5002"):
        """
        Initialize AI Service

        Args:
            data_service_url: URL of the data service
        """
        self.data_service_url = data_service_url

        # Initialize components
        self.model_trainer = ModelTrainer(data_service_url=data_service_url)
        self.predictor = ForexPredictor(data_service_url=data_service_url)
        self.model_manager = ModelManager()
        self.job_manager = TrainingJobManager()

        logger.info("AI Service initialized successfully")

    async def train_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a new LSTM model (async)

        Args:
            config: Training configuration

        Returns:
            Job information
        """
        try:
            # Create training job
            job_id = self.job_manager.create_job(config)

            # Start training in background
            asyncio.create_task(self._train_model_background(job_id, config))

            return {
                'success': True,
                'job_id': job_id,
                'status': 'queued',
                'estimated_duration': 3600
            }

        except Exception as e:
            logger.error(f"Error creating training job: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _train_model_background(self, job_id: str, config: Dict[str, Any]):
        """
        Background task for model training

        Args:
            job_id: Job identifier
            config: Training configuration
        """
        try:
            # Update status to running
            self.job_manager.update_job_status(job_id, 'running', progress=0)

            # Train model
            result = await self.model_trainer.train_model(config)

            # Update job status
            if result['success']:
                self.job_manager.update_job_status(
                    job_id,
                    'completed',
                    progress=100,
                    result=result
                )
                logger.info(f"Training job completed: {job_id}")
            else:
                self.job_manager.update_job_status(
                    job_id,
                    'failed',
                    error=result.get('error', 'Unknown error')
                )
                logger.error(f"Training job failed: {job_id}")

        except Exception as e:
            logger.error(f"Error in background training: {e}")
            self.job_manager.update_job_status(
                job_id,
                'failed',
                error=str(e)
            )

    def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get training job status

        Args:
            job_id: Job identifier

        Returns:
            Job status information
        """
        job = self.job_manager.get_job_status(job_id)

        if job is None:
            return {
                'success': False,
                'error': f'Job not found: {job_id}'
            }

        return {
            'success': True,
            'job': job
        }

    async def predict(self, model_id: str, symbol: str, timeframe: str,
                     steps_ahead: int = 1, use_confidence: bool = True) -> Dict[str, Any]:
        """
        Make price prediction

        Args:
            model_id: Model identifier
            symbol: Trading symbol
            timeframe: Timeframe
            steps_ahead: Number of steps to predict
            use_confidence: Whether to calculate confidence score

        Returns:
            Prediction result
        """
        return await self.predictor.predict(
            model_id=model_id,
            symbol=symbol,
            timeframe=timeframe,
            steps_ahead=steps_ahead,
            use_confidence=use_confidence
        )

    def list_models(self, symbol: str = None, timeframe: str = None,
                   status: str = None) -> Dict[str, Any]:
        """
        List available models

        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            status: Filter by status

        Returns:
            List of models
        """
        models = self.model_manager.list_models(
            symbol=symbol,
            timeframe=timeframe,
            status=status
        )

        return {
            'success': True,
            'models': models,
            'total': len(models)
        }

    def get_model(self, model_id: str) -> Dict[str, Any]:
        """
        Get model details

        Args:
            model_id: Model identifier

        Returns:
            Model information
        """
        model = self.model_manager.get_model(model_id)

        if model is None:
            return {
                'success': False,
                'error': f'Model not found: {model_id}'
            }

        return {
            'success': True,
            'model': model
        }

    def delete_model(self, model_id: str) -> Dict[str, Any]:
        """
        Delete a model

        Args:
            model_id: Model identifier

        Returns:
            Deletion result
        """
        # Unload from memory if loaded
        self.predictor.unload_model(model_id)

        # Delete files
        success = self.model_manager.delete_model(model_id)

        if success:
            return {
                'success': True,
                'message': f'Model deleted: {model_id}'
            }
        else:
            return {
                'success': False,
                'error': f'Failed to delete model: {model_id}'
            }

    def activate_model(self, model_id: str) -> Dict[str, Any]:
        """
        Activate a model

        Args:
            model_id: Model identifier

        Returns:
            Activation result
        """
        success = self.model_manager.activate_model(model_id, deactivate_others=True)

        if success:
            return {
                'success': True,
                'model_id': model_id,
                'message': 'Model activated successfully'
            }
        else:
            return {
                'success': False,
                'error': f'Failed to activate model: {model_id}'
            }

    def get_service_stats(self) -> Dict[str, Any]:
        """
        Get service statistics

        Returns:
            Service statistics
        """
        import tensorflow as tf

        model_stats = self.model_manager.get_model_statistics()
        loaded_models = self.predictor.get_loaded_models()

        # Check GPU availability
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0

        return {
            'loaded_models': len(loaded_models),
            'gpu_available': gpu_available,
            'tensorflow_version': tf.__version__,
            'model_statistics': model_stats
        }
