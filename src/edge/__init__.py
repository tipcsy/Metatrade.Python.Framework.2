"""
Edge Computing and IoT Integration for MetaTrader Python Framework Phase 6

This module provides edge computing capabilities including IoT sensor integration,
5G/6G network optimization, distributed mesh computing, and real-time edge processing.

Key Components:
- EdgeProcessor: Real-time edge device processing
- IoTIntegration: IoT sensor data integration
- NetworkOptimizer: 5G/6G network optimization
- DistributedMesh: Distributed computing mesh
"""

from .edge_processor import EdgeProcessor
from .iot_integration import IoTSensorManager
from .network_optimizer import NetworkOptimizer
from .distributed_mesh import DistributedMesh

__all__ = [
    'EdgeProcessor',
    'IoTSensorManager',
    'NetworkOptimizer',
    'DistributedMesh'
]

__version__ = '6.0.0'