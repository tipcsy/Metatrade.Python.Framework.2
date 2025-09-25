"""
Advanced AI Systems for MetaTrader Python Framework Phase 6

This module provides next-generation AI capabilities including transformer models,
reinforcement learning, federated learning, and explainable AI for trading optimization.

Key Components:
- TransformerModels: Transformer-based market prediction
- ReinforcementLearning: Advanced RL trading agents
- FederatedLearning: Distributed learning systems
- ExplainableAI: Model interpretation and compliance
- NeuromorphicComputing: Brain-inspired computing
"""

from .transformer_models import TransformerPredictor
from .reinforcement_learning import QuantumRLAgent
from .federated_learning import FederatedTradingNetwork
from .explainable_ai import ExplainableTrading
from .neuromorphic_computing import NeuromorphicProcessor

__all__ = [
    'TransformerPredictor',
    'QuantumRLAgent',
    'FederatedTradingNetwork',
    'ExplainableTrading',
    'NeuromorphicProcessor'
]

__version__ = '6.0.0'