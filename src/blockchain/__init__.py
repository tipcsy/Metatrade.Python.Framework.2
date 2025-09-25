"""
Blockchain and DeFi Integration for MetaTrader Python Framework Phase 6

This module provides comprehensive blockchain integration including DeFi protocols,
smart contracts, cross-chain bridges, and tokenized asset management.

Key Components:
- DeFiProtocols: Integration with major DeFi platforms
- SmartContracts: Smart contract management and deployment
- CrossChain: Cross-chain bridge implementations
- Tokenization: Asset tokenization systems
- YieldOptimizer: DeFi yield optimization strategies
"""

from .defi_protocols import DeFiIntegrator
from .smart_contracts import SmartContractManager
from .cross_chain import CrossChainBridge
from .tokenization import AssetTokenizer
from .yield_optimizer import YieldOptimizer

__all__ = [
    'DeFiIntegrator',
    'SmartContractManager',
    'CrossChainBridge',
    'AssetTokenizer',
    'YieldOptimizer'
]

__version__ = '6.0.0'