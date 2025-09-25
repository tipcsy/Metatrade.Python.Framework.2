"""
Phase 6 Quantum Computing Components for MetaTrader Python Framework.

This module provides quantum computing integration for advanced portfolio optimization,
machine learning acceleration, and hybrid classical-quantum algorithms.

Key Features:
- Quantum portfolio optimization using VQE and QAOA algorithms
- Quantum machine learning for enhanced pattern recognition
- Quantum simulation for risk assessment
- Hybrid classical-quantum algorithms for trading strategies
- Integration with quantum cloud services (IBM Quantum, AWS Braket)
- Quantum-resistant cryptographic implementations
"""

from .quantum_optimizer import QuantumOptimizer
from .quantum_ml import QuantumMLPipeline
from .quantum_simulator import QuantumSimulator
from .quantum_gateway import QuantumGateway
from .hybrid_algorithms import HybridAlgorithms

__all__ = [
    'QuantumOptimizer',
    'QuantumMLPipeline',
    'QuantumSimulator',
    'QuantumGateway',
    'HybridAlgorithms'
]

__version__ = '6.0.0'