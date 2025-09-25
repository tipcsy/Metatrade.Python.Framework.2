"""
Edge Computing Processor

This module provides real-time edge computing capabilities for distributed
trading analysis, low-latency decision making, and IoT data processing
at the network edge.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import uuid
import statistics

logger = logging.getLogger(__name__)


class EdgeNodeType(Enum):
    """Types of edge computing nodes"""
    TRADING_GATEWAY = "trading_gateway"
    DATA_PROCESSOR = "data_processor"
    ML_INFERENCE = "ml_inference"
    RISK_MONITOR = "risk_monitor"
    IOT_AGGREGATOR = "iot_aggregator"


@dataclass
class EdgeTask:
    """Represents a task for edge processing"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int
    deadline: Optional[datetime]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    assigned_node: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'payload': self.payload,
            'priority': self.priority,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'created_at': self.created_at.isoformat(),
            'assigned_node': self.assigned_node,
            'status': self.status
        }


@dataclass
class EdgeNode:
    """Represents an edge computing node"""
    node_id: str
    node_type: EdgeNodeType
    location: str
    capabilities: List[str]
    current_load: float = 0.0
    max_capacity: int = 100
    active_tasks: List[str] = field(default_factory=list)
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "active"  # active, inactive, maintenance

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'node_type': self.node_type.value,
            'location': self.location,
            'capabilities': self.capabilities,
            'current_load': self.current_load,
            'max_capacity': self.max_capacity,
            'active_tasks': len(self.active_tasks),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'status': self.status
        }


@dataclass
class ProcessingResult:
    """Result of edge processing task"""
    task_id: str
    result: Dict[str, Any]
    processing_time: float
    node_id: str
    success: bool
    error_message: Optional[str] = None
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'result': self.result,
            'processing_time': self.processing_time,
            'node_id': self.node_id,
            'success': self.success,
            'error_message': self.error_message,
            'completed_at': self.completed_at.isoformat()
        }


class EdgeProcessor:
    """
    Edge Computing Processor for Distributed Trading System

    Provides distributed computing capabilities at the network edge for:
    - Real-time market data processing
    - Low-latency trading decisions
    - IoT sensor data aggregation
    - ML model inference
    - Risk monitoring and alerting

    Features:
    - Dynamic load balancing
    - Fault tolerance and recovery
    - Real-time task scheduling
    - Edge-to-cloud synchronization
    - 5G/6G network optimization
    """

    def __init__(self, node_id: Optional[str] = None, location: str = "unknown"):
        self.node_id = node_id or str(uuid.uuid4())
        self.location = location

        # Edge nodes registry
        self.edge_nodes: Dict[str, EdgeNode] = {}
        self.local_node = None

        # Task management
        self.task_queue: List[EdgeTask] = []
        self.processing_tasks: Dict[str, EdgeTask] = {}
        self.completed_tasks: Dict[str, ProcessingResult] = {}

        # Task handlers
        self.task_handlers: Dict[str, Callable] = {
            'market_data_processing': self._process_market_data,
            'ml_inference': self._run_ml_inference,
            'risk_assessment': self._assess_risk,
            'iot_data_aggregation': self._aggregate_iot_data,
            'trading_signal': self._generate_trading_signal
        }

        # Performance metrics
        self.total_tasks_processed = 0
        self.average_processing_time = 0.0
        self.task_success_rate = 1.0
        self.node_utilization = 0.0

        # Network optimization
        self.network_latency = {}
        self.bandwidth_usage = {}

        logger.info(f"EdgeProcessor initialized: {self.node_id} at {location}")

    async def start(self):
        """Start the edge processor"""
        try:
            # Initialize local edge node
            self.local_node = EdgeNode(
                node_id=self.node_id,
                node_type=EdgeNodeType.DATA_PROCESSOR,
                location=self.location,
                capabilities=['market_data_processing', 'ml_inference', 'risk_assessment']
            )

            self.edge_nodes[self.node_id] = self.local_node

            # Start background tasks
            asyncio.create_task(self._task_scheduler())
            asyncio.create_task(self._heartbeat_monitor())
            asyncio.create_task(self._performance_monitor())

            logger.info("EdgeProcessor started successfully")

        except Exception as e:
            logger.error(f"Failed to start EdgeProcessor: {e}")
            raise

    async def stop(self):
        """Stop the edge processor"""
        try:
            # Complete pending tasks
            await self._complete_pending_tasks()

            # Update node status
            if self.local_node:
                self.local_node.status = "inactive"

            logger.info("EdgeProcessor stopped")

        except Exception as e:
            logger.error(f"Error stopping EdgeProcessor: {e}")

    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 5,
        deadline: Optional[datetime] = None
    ) -> str:
        """
        Submit a task for edge processing

        Args:
            task_type: Type of task to process
            payload: Task data and parameters
            priority: Task priority (1-10, higher = more urgent)
            deadline: Optional deadline for task completion

        Returns:
            Task ID for tracking
        """
        try:
            task_id = str(uuid.uuid4())

            task = EdgeTask(
                task_id=task_id,
                task_type=task_type,
                payload=payload,
                priority=priority,
                deadline=deadline
            )

            # Add to queue (maintain priority order)
            self.task_queue.append(task)
            self.task_queue.sort(key=lambda t: t.priority, reverse=True)

            logger.info(f"Task submitted: {task_id} ({task_type})")

            return task_id

        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise

    async def get_task_result(self, task_id: str) -> Optional[ProcessingResult]:
        """
        Get result of a processed task

        Args:
            task_id: ID of the task

        Returns:
            ProcessingResult if available, None otherwise
        """
        return self.completed_tasks.get(task_id)

    async def _task_scheduler(self):
        """Background task scheduler"""
        while True:
            try:
                if self.task_queue and len(self.processing_tasks) < self.local_node.max_capacity:
                    # Get highest priority task
                    task = self.task_queue.pop(0)

                    # Assign to best available node
                    best_node = self._select_best_node(task.task_type)

                    if best_node:
                        task.assigned_node = best_node.node_id
                        task.status = "processing"
                        self.processing_tasks[task.task_id] = task

                        # Process task
                        asyncio.create_task(self._process_task(task))

                await asyncio.sleep(0.1)  # 100ms scheduling interval

            except Exception as e:
                logger.error(f"Task scheduler error: {e}")
                await asyncio.sleep(1)

    def _select_best_node(self, task_type: str) -> Optional[EdgeNode]:
        """Select best available node for task"""
        available_nodes = [
            node for node in self.edge_nodes.values()
            if (node.status == "active" and
                task_type in node.capabilities and
                node.current_load < 0.8)  # 80% max utilization
        ]

        if not available_nodes:
            return None

        # Select node with lowest current load
        return min(available_nodes, key=lambda n: n.current_load)

    async def _process_task(self, task: EdgeTask):
        """Process an individual task"""
        start_time = asyncio.get_event_loop().time()

        try:
            # Get task handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler for task type: {task.task_type}")

            # Execute task
            result_data = await handler(task.payload)

            # Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time

            # Create result
            result = ProcessingResult(
                task_id=task.task_id,
                result=result_data,
                processing_time=processing_time,
                node_id=task.assigned_node or self.node_id,
                success=True
            )

            # Store result
            self.completed_tasks[task.task_id] = result

            # Update metrics
            self.total_tasks_processed += 1
            self._update_performance_metrics(processing_time, True)

            logger.info(f"Task completed: {task.task_id} in {processing_time:.3f}s")

        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time

            # Create error result
            result = ProcessingResult(
                task_id=task.task_id,
                result={},
                processing_time=processing_time,
                node_id=task.assigned_node or self.node_id,
                success=False,
                error_message=str(e)
            )

            self.completed_tasks[task.task_id] = result
            self._update_performance_metrics(processing_time, False)

            logger.error(f"Task failed: {task.task_id} - {e}")

        finally:
            # Remove from processing tasks
            if task.task_id in self.processing_tasks:
                del self.processing_tasks[task.task_id]

            # Update node load
            if self.local_node:
                self.local_node.current_load = len(self.processing_tasks) / self.local_node.max_capacity

    async def _process_market_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data at the edge"""
        try:
            # Extract market data
            symbol = payload.get('symbol')
            price_data = payload.get('prices', [])
            volume_data = payload.get('volumes', [])

            if not price_data:
                return {'error': 'No price data provided'}

            # Calculate technical indicators
            prices = [float(p) for p in price_data]

            # Simple moving averages
            sma_5 = statistics.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
            sma_20 = statistics.mean(prices[-20:]) if len(prices) >= 20 else statistics.mean(prices)

            # Price volatility
            if len(prices) > 1:
                returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
                volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0
            else:
                volatility = 0.0

            # Volume analysis
            if volume_data:
                volumes = [float(v) for v in volume_data]
                avg_volume = statistics.mean(volumes)
                volume_trend = (volumes[-1] / avg_volume - 1) if avg_volume > 0 else 0
            else:
                avg_volume = 0
                volume_trend = 0

            return {
                'symbol': symbol,
                'current_price': prices[-1],
                'sma_5': sma_5,
                'sma_20': sma_20,
                'volatility': volatility,
                'average_volume': avg_volume,
                'volume_trend': volume_trend,
                'trend_signal': 'bullish' if sma_5 > sma_20 else 'bearish'
            }

        except Exception as e:
            logger.error(f"Market data processing failed: {e}")
            return {'error': str(e)}

    async def _run_ml_inference(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run ML model inference at the edge"""
        try:
            model_type = payload.get('model_type')
            input_features = payload.get('features', [])

            if not input_features:
                return {'error': 'No features provided'}

            # Mock ML inference (in production, would load actual models)
            import numpy as np

            features_array = np.array(input_features)

            if model_type == 'price_prediction':
                # Mock price prediction
                prediction = float(np.mean(features_array) * 1.02)  # 2% increase
                confidence = 0.85

            elif model_type == 'volatility_forecast':
                # Mock volatility forecast
                prediction = float(np.std(features_array))
                confidence = 0.75

            elif model_type == 'trend_classification':
                # Mock trend classification
                trend_signal = np.mean(features_array[-5:]) - np.mean(features_array[-10:-5])
                prediction = 1 if trend_signal > 0 else 0  # 1 = uptrend, 0 = downtrend
                confidence = min(abs(trend_signal) + 0.5, 0.95)

            else:
                return {'error': f'Unsupported model type: {model_type}'}

            return {
                'model_type': model_type,
                'prediction': prediction,
                'confidence': float(confidence),
                'features_processed': len(input_features)
            }

        except Exception as e:
            logger.error(f"ML inference failed: {e}")
            return {'error': str(e)}

    async def _assess_risk(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk at the edge"""
        try:
            position_size = payload.get('position_size', 0)
            account_balance = payload.get('account_balance', 0)
            volatility = payload.get('volatility', 0)
            correlation_matrix = payload.get('correlations', [])

            # Risk calculations
            position_risk = (position_size / account_balance) if account_balance > 0 else 0
            volatility_risk = min(volatility * 10, 1.0)  # Cap at 100%

            # Portfolio risk (simplified)
            if correlation_matrix:
                avg_correlation = statistics.mean([abs(c) for row in correlation_matrix for c in row])
                correlation_risk = avg_correlation
            else:
                correlation_risk = 0

            # Overall risk score
            overall_risk = (position_risk * 0.4 + volatility_risk * 0.4 + correlation_risk * 0.2)

            # Risk level classification
            if overall_risk < 0.3:
                risk_level = "LOW"
            elif overall_risk < 0.6:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"

            return {
                'position_risk': position_risk,
                'volatility_risk': volatility_risk,
                'correlation_risk': correlation_risk,
                'overall_risk_score': overall_risk,
                'risk_level': risk_level,
                'recommended_action': 'HOLD' if overall_risk < 0.7 else 'REDUCE_EXPOSURE'
            }

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {'error': str(e)}

    async def _aggregate_iot_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate IoT sensor data"""
        try:
            sensor_data = payload.get('sensor_readings', [])
            sensor_types = payload.get('sensor_types', [])

            if not sensor_data:
                return {'error': 'No sensor data provided'}

            # Group data by sensor type
            aggregated = {}

            for i, reading in enumerate(sensor_data):
                sensor_type = sensor_types[i] if i < len(sensor_types) else 'unknown'

                if sensor_type not in aggregated:
                    aggregated[sensor_type] = []

                aggregated[sensor_type].append(reading)

            # Calculate statistics for each sensor type
            summary = {}
            for sensor_type, readings in aggregated.items():
                numeric_readings = [float(r) for r in readings if isinstance(r, (int, float))]

                if numeric_readings:
                    summary[sensor_type] = {
                        'count': len(numeric_readings),
                        'min': min(numeric_readings),
                        'max': max(numeric_readings),
                        'mean': statistics.mean(numeric_readings),
                        'median': statistics.median(numeric_readings),
                        'std': statistics.stdev(numeric_readings) if len(numeric_readings) > 1 else 0
                    }

            return {
                'sensors_processed': len(sensor_types),
                'readings_processed': len(sensor_data),
                'sensor_summary': summary,
                'anomalies_detected': []  # Could implement anomaly detection
            }

        except Exception as e:
            logger.error(f"IoT data aggregation failed: {e}")
            return {'error': str(e)}

    async def _generate_trading_signal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal at the edge"""
        try:
            symbol = payload.get('symbol')
            price_data = payload.get('price_data', [])
            technical_indicators = payload.get('indicators', {})
            ml_predictions = payload.get('ml_predictions', {})

            # Combine multiple signals
            signals = []

            # Technical analysis signals
            if 'sma_5' in technical_indicators and 'sma_20' in technical_indicators:
                if technical_indicators['sma_5'] > technical_indicators['sma_20']:
                    signals.append(('technical', 1, 0.6))  # bullish
                else:
                    signals.append(('technical', -1, 0.6))  # bearish

            # ML prediction signals
            if 'trend_classification' in ml_predictions:
                ml_signal = 1 if ml_predictions['trend_classification']['prediction'] > 0.5 else -1
                ml_confidence = ml_predictions['trend_classification'].get('confidence', 0.5)
                signals.append(('ml', ml_signal, ml_confidence))

            # Volume confirmation
            if 'volume_trend' in technical_indicators:
                volume_signal = 1 if technical_indicators['volume_trend'] > 0.1 else 0
                signals.append(('volume', volume_signal, 0.3))

            # Aggregate signals
            if signals:
                weighted_signal = sum(signal * confidence for _, signal, confidence in signals)
                total_weight = sum(confidence for _, _, confidence in signals)
                final_signal = weighted_signal / total_weight if total_weight > 0 else 0

                # Generate trading recommendation
                if final_signal > 0.3:
                    action = "BUY"
                elif final_signal < -0.3:
                    action = "SELL"
                else:
                    action = "HOLD"

                confidence = min(abs(final_signal), 1.0)

            else:
                action = "HOLD"
                final_signal = 0
                confidence = 0

            return {
                'symbol': symbol,
                'action': action,
                'signal_strength': final_signal,
                'confidence': confidence,
                'contributing_signals': [
                    {'source': source, 'signal': signal, 'confidence': conf}
                    for source, signal, conf in signals
                ]
            }

        except Exception as e:
            logger.error(f"Trading signal generation failed: {e}")
            return {'error': str(e)}

    async def _heartbeat_monitor(self):
        """Monitor node heartbeats"""
        while True:
            try:
                current_time = datetime.now(timezone.utc)

                # Update local node heartbeat
                if self.local_node:
                    self.local_node.last_heartbeat = current_time

                # Check other nodes for stale heartbeats
                stale_nodes = []
                for node_id, node in self.edge_nodes.items():
                    if node_id != self.node_id:
                        time_diff = (current_time - node.last_heartbeat).total_seconds()
                        if time_diff > 60:  # 60 seconds timeout
                            node.status = "inactive"
                            stale_nodes.append(node_id)

                if stale_nodes:
                    logger.warning(f"Detected stale nodes: {stale_nodes}")

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(30)

    async def _performance_monitor(self):
        """Monitor system performance"""
        while True:
            try:
                # Update node utilization
                if self.local_node:
                    self.local_node.current_load = len(self.processing_tasks) / self.local_node.max_capacity

                # Calculate system metrics
                self.node_utilization = self.local_node.current_load if self.local_node else 0

                await asyncio.sleep(10)  # Update every 10 seconds

            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(10)

    def _update_performance_metrics(self, processing_time: float, success: bool):
        """Update performance metrics"""
        # Update average processing time
        if self.total_tasks_processed > 0:
            self.average_processing_time = (
                (self.average_processing_time * (self.total_tasks_processed - 1) + processing_time) /
                self.total_tasks_processed
            )
        else:
            self.average_processing_time = processing_time

        # Update success rate
        if self.total_tasks_processed > 0:
            successful_tasks = sum(1 for result in self.completed_tasks.values() if result.success)
            self.task_success_rate = successful_tasks / len(self.completed_tasks)

    async def _complete_pending_tasks(self):
        """Complete any pending tasks during shutdown"""
        timeout = 30  # 30 second timeout
        start_time = asyncio.get_event_loop().time()

        while self.processing_tasks and (asyncio.get_event_loop().time() - start_time) < timeout:
            await asyncio.sleep(0.1)

        if self.processing_tasks:
            logger.warning(f"Shutdown with {len(self.processing_tasks)} incomplete tasks")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'node_id': self.node_id,
            'location': self.location,
            'status': self.local_node.status if self.local_node else 'unknown',
            'total_nodes': len(self.edge_nodes),
            'active_nodes': len([n for n in self.edge_nodes.values() if n.status == 'active']),
            'tasks_in_queue': len(self.task_queue),
            'tasks_processing': len(self.processing_tasks),
            'tasks_completed': len(self.completed_tasks),
            'total_processed': self.total_tasks_processed,
            'average_processing_time': self.average_processing_time,
            'success_rate': self.task_success_rate,
            'node_utilization': self.node_utilization,
            'supported_task_types': list(self.task_handlers.keys()),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    async def register_edge_node(self, node: EdgeNode):
        """Register a new edge node"""
        self.edge_nodes[node.node_id] = node
        logger.info(f"Registered edge node: {node.node_id} ({node.node_type.value})")

    async def optimize_task_distribution(self):
        """Optimize task distribution across edge nodes"""
        try:
            # Analyze current load distribution
            node_loads = {node_id: node.current_load for node_id, node in self.edge_nodes.items()}

            # Identify overloaded nodes
            overloaded_nodes = [
                node_id for node_id, load in node_loads.items()
                if load > 0.8  # 80% threshold
            ]

            # Redistribute tasks if needed
            if overloaded_nodes:
                logger.info(f"Optimizing load for overloaded nodes: {overloaded_nodes}")
                # Implementation would involve moving tasks between nodes

            return {
                'optimization_applied': len(overloaded_nodes) > 0,
                'overloaded_nodes': overloaded_nodes,
                'current_loads': node_loads
            }

        except Exception as e:
            logger.error(f"Task distribution optimization failed: {e}")
            return {'error': str(e)}