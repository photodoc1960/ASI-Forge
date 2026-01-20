"""
Emergence Detection and Monitoring Agent
Continuously monitors for unexpected behaviors and novel capabilities
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmergentCapability:
    """Detected emergent capability"""
    name: str
    description: str
    detection_time: datetime
    confidence: float
    evidence: Dict[str, Any]
    baseline_performance: float
    new_performance: float
    task_domain: str
    requires_human_review: bool


@dataclass
class AnomalyEvent:
    """Detected anomalous behavior"""
    event_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    timestamp: datetime
    metrics: Dict[str, float]
    affected_components: List[str]
    auto_frozen: bool


class CapabilityTracker:
    """Tracks model capabilities across different domains"""

    def __init__(self):
        self.capability_history: Dict[str, List[float]] = {}
        self.baseline_capabilities: Dict[str, float] = {}

        # Define capability domains to track
        self.domains = [
            'logical_reasoning',
            'pattern_recognition',
            'abstraction',
            'planning',
            'generalization',
            'transfer_learning',
            'compositional_reasoning',
            'meta_learning'
        ]

        for domain in self.domains:
            self.capability_history[domain] = []

    def record_capability(self, domain: str, score: float):
        """Record a capability measurement"""

        if domain not in self.capability_history:
            self.capability_history[domain] = []

        self.capability_history[domain].append(score)

        # Set baseline if not yet established (first 10 measurements)
        if domain not in self.baseline_capabilities:
            if len(self.capability_history[domain]) >= 10:
                self.baseline_capabilities[domain] = np.mean(
                    self.capability_history[domain][:10]
                )

    def detect_capability_jump(
        self,
        domain: str,
        threshold: float = 0.15
    ) -> Optional[float]:
        """
        Detect significant capability improvements

        Returns improvement ratio if jump detected, None otherwise
        """

        if domain not in self.baseline_capabilities:
            return None

        if len(self.capability_history[domain]) < 2:
            return None

        baseline = self.baseline_capabilities[domain]
        recent = np.mean(self.capability_history[domain][-5:])

        if baseline < 1e-6:  # Avoid division by zero
            return None

        improvement = (recent - baseline) / baseline

        if improvement > threshold:
            return improvement

        return None

    def get_capability_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked capabilities"""

        summary = {}
        for domain in self.domains:
            if domain in self.baseline_capabilities:
                history = self.capability_history[domain]
                summary[domain] = {
                    'baseline': self.baseline_capabilities[domain],
                    'current': np.mean(history[-5:]) if len(history) >= 5 else None,
                    'trend': self._compute_trend(history),
                    'measurements': len(history)
                }

        return summary

    def _compute_trend(self, history: List[float]) -> str:
        """Compute trend direction"""

        if len(history) < 10:
            return 'insufficient_data'

        recent_mean = np.mean(history[-5:])
        baseline_mean = np.mean(history[:5])

        diff = (recent_mean - baseline_mean) / (baseline_mean + 1e-6)

        if diff > 0.05:
            return 'improving'
        elif diff < -0.05:
            return 'degrading'
        else:
            return 'stable'


class AnomalyDetector:
    """Detects anomalous behaviors and patterns"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        # Track various metrics
        self.loss_history = deque(maxlen=window_size)
        self.gradient_norms = deque(maxlen=window_size)
        self.output_distributions = deque(maxlen=window_size)
        self.runtime_history = deque(maxlen=window_size)

    def record_training_step(
        self,
        loss: float,
        gradient_norm: float,
        runtime: float
    ):
        """Record metrics from a training step"""

        self.loss_history.append(loss)
        self.gradient_norms.append(gradient_norm)
        self.runtime_history.append(runtime)

    def detect_anomalies(self) -> List[AnomalyEvent]:
        """Detect various types of anomalies"""

        anomalies = []

        # Check for loss explosion
        if len(self.loss_history) >= 10:
            recent_loss = list(self.loss_history)[-5:]
            baseline_loss = list(self.loss_history)[:50]

            if len(baseline_loss) >= 10:
                recent_mean = np.mean(recent_loss)
                baseline_mean = np.mean(baseline_loss)
                baseline_std = np.std(baseline_loss)

                # Loss > 3 standard deviations above baseline
                if recent_mean > baseline_mean + 3 * baseline_std:
                    anomalies.append(AnomalyEvent(
                        event_type='loss_explosion',
                        severity='high',
                        description=f'Loss exploded to {recent_mean:.4f} from baseline {baseline_mean:.4f}',
                        timestamp=datetime.now(),
                        metrics={
                            'recent_loss': recent_mean,
                            'baseline_loss': baseline_mean,
                            'std_deviations': (recent_mean - baseline_mean) / (baseline_std + 1e-8)
                        },
                        affected_components=['training'],
                        auto_frozen=True
                    ))

        # Check for gradient explosion
        if len(self.gradient_norms) >= 10:
            recent_grad = list(self.gradient_norms)[-5:]
            max_grad = max(recent_grad)

            if max_grad > 100.0:  # Arbitrary threshold
                anomalies.append(AnomalyEvent(
                    event_type='gradient_explosion',
                    severity='high',
                    description=f'Gradient norm {max_grad:.2f} exceeds safe threshold',
                    timestamp=datetime.now(),
                    metrics={'max_gradient_norm': max_grad},
                    affected_components=['training', 'optimization'],
                    auto_frozen=True
                ))

        # Check for runtime anomalies
        if len(self.runtime_history) >= 20:
            recent_runtime = np.mean(list(self.runtime_history)[-5:])
            baseline_runtime = np.mean(list(self.runtime_history)[:10])

            # Runtime increased by more than 3x
            if recent_runtime > 3 * baseline_runtime:
                anomalies.append(AnomalyEvent(
                    event_type='runtime_anomaly',
                    severity='medium',
                    description=f'Runtime increased to {recent_runtime:.2f}s from {baseline_runtime:.2f}s',
                    timestamp=datetime.now(),
                    metrics={
                        'recent_runtime': recent_runtime,
                        'baseline_runtime': baseline_runtime
                    },
                    affected_components=['execution'],
                    auto_frozen=False
                ))

        return anomalies


class EmergenceDetectionAgent:
    """
    Main agent for detecting emergent capabilities and anomalies
    Continuously monitors system and freezes when unexpected behavior detected
    """

    def __init__(self, alert_callback=None):
        self.capability_tracker = CapabilityTracker()
        self.anomaly_detector = AnomalyDetector()

        # Callback for human notification
        self.alert_callback = alert_callback

        # State
        self.is_frozen = False
        self.freeze_reason: Optional[str] = None

        # History
        self.detected_capabilities: List[EmergentCapability] = []
        self.detected_anomalies: List[AnomalyEvent] = []

        # Known capabilities at initialization
        self.known_capabilities: Set[str] = set()

    def monitor_training_step(
        self,
        loss: float,
        gradient_norm: float,
        runtime: float
    ) -> bool:
        """
        Monitor a training step

        Returns:
            True if safe to continue, False if system should freeze
        """

        if self.is_frozen:
            return False

        # Record metrics
        self.anomaly_detector.record_training_step(loss, gradient_norm, runtime)

        # Check for anomalies
        anomalies = self.anomaly_detector.detect_anomalies()

        if anomalies:
            self.detected_anomalies.extend(anomalies)

            # Auto-freeze on critical anomalies
            critical_anomalies = [a for a in anomalies if a.auto_frozen]
            if critical_anomalies:
                self.freeze_system(
                    f"Critical anomalies detected: {[a.event_type for a in critical_anomalies]}"
                )
                return False

            # Alert on non-critical anomalies
            for anomaly in anomalies:
                self._send_alert(
                    f"Anomaly detected: {anomaly.description}",
                    severity=anomaly.severity
                )

        return True

    def evaluate_capabilities(
        self,
        model: nn.Module,
        evaluation_suite: Dict[str, callable]
    ) -> List[EmergentCapability]:
        """
        Evaluate model on capability benchmarks

        Args:
            model: Model to evaluate
            evaluation_suite: Dict of {domain: evaluation_function}

        Returns:
            List of newly detected emergent capabilities
        """

        if self.is_frozen:
            return []

        new_capabilities = []

        for domain, eval_fn in evaluation_suite.items():
            try:
                score = eval_fn(model)
                self.capability_tracker.record_capability(domain, score)

                # Check for capability jump
                improvement = self.capability_tracker.detect_capability_jump(domain)

                if improvement is not None:
                    # Potential emergent capability
                    baseline = self.capability_tracker.baseline_capabilities[domain]
                    current = score

                    capability = EmergentCapability(
                        name=f"{domain}_enhancement",
                        description=f"Significant improvement in {domain}",
                        detection_time=datetime.now(),
                        confidence=min(improvement, 1.0),
                        evidence={
                            'improvement_ratio': improvement,
                            'measurements': len(self.capability_tracker.capability_history[domain])
                        },
                        baseline_performance=baseline,
                        new_performance=current,
                        task_domain=domain,
                        requires_human_review=True
                    )

                    new_capabilities.append(capability)
                    self.detected_capabilities.append(capability)

                    # Freeze and alert
                    self.freeze_system(
                        f"Emergent capability detected: {capability.name} "
                        f"({improvement:.1%} improvement)"
                    )

                    self._send_alert(
                        f"🚨 EMERGENT CAPABILITY DETECTED: {capability.description}\n"
                        f"Domain: {domain}\n"
                        f"Improvement: {improvement:.1%}\n"
                        f"Baseline: {baseline:.4f} → Current: {current:.4f}\n"
                        f"System FROZEN - Requires human review",
                        severity='critical'
                    )

            except Exception as e:
                logger.error(f"Error evaluating {domain}: {e}")

        return new_capabilities

    def freeze_system(self, reason: str):
        """Freeze the system and alert human operator"""

        self.is_frozen = True
        self.freeze_reason = reason

        logger.critical(f"SYSTEM FROZEN: {reason}")
        self._send_alert(
            f"🛑 SYSTEM FROZEN\nReason: {reason}\nTime: {datetime.now()}\n"
            f"Human approval required to resume.",
            severity='critical'
        )

    def unfreeze_system(self, approval_code: str):
        """Unfreeze system with human approval"""

        # In production, verify approval_code
        self.is_frozen = False
        self.freeze_reason = None

        logger.info(f"System unfrozen with approval code: {approval_code[:8]}...")
        self._send_alert("System unfrozen by human operator", severity='info')

    def _send_alert(self, message: str, severity: str = 'info'):
        """Send alert to human operator"""

        logger.log(
            {
                'info': logging.INFO,
                'medium': logging.WARNING,
                'high': logging.ERROR,
                'critical': logging.CRITICAL
            }.get(severity, logging.INFO),
            message
        )

        if self.alert_callback:
            self.alert_callback(message, severity)

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report"""

        return {
            'is_frozen': self.is_frozen,
            'freeze_reason': self.freeze_reason,
            'emergent_capabilities_detected': len(self.detected_capabilities),
            'anomalies_detected': len(self.detected_anomalies),
            'capability_summary': self.capability_tracker.get_capability_summary(),
            'recent_emergent_capabilities': [
                {
                    'name': c.name,
                    'domain': c.task_domain,
                    'improvement': f"{((c.new_performance - c.baseline_performance) / c.baseline_performance):.1%}",
                    'time': c.detection_time.isoformat()
                }
                for c in self.detected_capabilities[-5:]
            ],
            'recent_anomalies': [
                {
                    'type': a.event_type,
                    'severity': a.severity,
                    'description': a.description,
                    'time': a.timestamp.isoformat()
                }
                for a in self.detected_anomalies[-5:]
            ]
        }
