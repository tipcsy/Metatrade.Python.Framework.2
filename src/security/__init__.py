"""
Next-Generation Security Module for MetaTrader Python Framework Phase 6

This module provides advanced security capabilities including post-quantum cryptography,
zero-knowledge proofs, homomorphic encryption, and quantum-resistant security protocols.

Key Components:
- PostQuantumCrypto: Post-quantum cryptographic algorithms
- ZeroKnowledge: Zero-knowledge proof systems
- HomomorphicEncryption: Privacy-preserving computations
- QuantumKeyDistribution: Quantum-secured key exchange
- AIThreatDetection: AI-powered security monitoring
"""

from .post_quantum_crypto import PostQuantumCrypto
from .zero_knowledge import ZeroKnowledgeProver
from .homomorphic_encryption import HomomorphicProcessor
from .quantum_key_distribution import QuantumKeyDistributor
from .ai_threat_detection import AISecurityMonitor

__all__ = [
    'PostQuantumCrypto',
    'ZeroKnowledgeProver',
    'HomomorphicProcessor',
    'QuantumKeyDistributor',
    'AISecurityMonitor'
]

__version__ = '6.0.0'