"""
Core Safety Validation System for AEGIS
Provides multi-layer safety checks for all system operations
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum, IntEnum
import json
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RiskLevel(IntEnum):
    """Risk levels for operations (ordered by severity)"""
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ValidationResult(Enum):
    """Validation outcomes"""
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_HUMAN = "requires_human"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SafetyCheck:
    """Result of a safety check"""
    passed: bool
    risk_level: RiskLevel
    validation_result: ValidationResult
    reason: str
    details: Dict[str, Any]
    timestamp: datetime
    validator: str


class SafetyBounds:
    """Defines safety bounds for system operations"""

    def __init__(self):
        # Resource limits
        self.max_parameters = 1_000_000_000  # 1B parameters max
        self.max_memory_gb = 32
        self.max_compute_hours = 24
        self.max_layers = 200  # Increased to accommodate HRM with all submodules

        # Architecture constraints
        self.max_recursion_depth = 16
        self.max_attention_heads = 128
        self.min_interpretability_score = 0.3

        # Behavioral constraints
        self.max_performance_degradation = 0.1  # 10% max degradation
        self.required_test_coverage = 0.9  # 90% test coverage
        self.max_capability_jump = 0.15  # 15% max improvement per step

        # Evolution constraints
        self.max_mutation_size = 0.05  # 5% of architecture can change
        self.min_population_diversity = 0.3
        self.max_evolution_rate = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Export bounds as dictionary"""
        return {k: v for k, v in self.__dict__.items()}


class CodeSafetyValidator:
    """Validates generated code for safety issues"""

    FORBIDDEN_IMPORTS = [
        'os.system', 'subprocess', 'eval', 'exec',
        '__import__', 'compile', 'open'  # Restricted, not forbidden
    ]

    FORBIDDEN_OPERATIONS = [
        'rmtree', 'unlink', 'remove', 'kill',
        'network', 'socket', 'urllib', 'requests'  # Network access
    ]

    def __init__(self):
        self.max_code_lines = 1000
        self.max_complexity = 20  # Cyclomatic complexity

    def validate_code(self, code: str) -> SafetyCheck:
        """Validate code for safety issues"""

        issues = []
        risk = RiskLevel.SAFE

        # Check for forbidden imports
        for forbidden in self.FORBIDDEN_IMPORTS:
            if forbidden in code:
                issues.append(f"Forbidden import/operation: {forbidden}")
                risk = RiskLevel.CRITICAL

        # Check for forbidden operations
        for forbidden in self.FORBIDDEN_OPERATIONS:
            if forbidden in code:
                issues.append(f"Forbidden operation: {forbidden}")
                risk = RiskLevel.HIGH

        # Check code length
        lines = code.split('\n')
        if len(lines) > self.max_code_lines:
            issues.append(f"Code too long: {len(lines)} lines (max {self.max_code_lines})")
            risk = max(risk, RiskLevel.MEDIUM)

        # Check for infinite loops (basic heuristic)
        if 'while True' in code and 'break' not in code:
            issues.append("Potential infinite loop detected")
            risk = RiskLevel.HIGH

        passed = len(issues) == 0
        validation = ValidationResult.APPROVED if passed else ValidationResult.REQUIRES_HUMAN

        if risk == RiskLevel.CRITICAL:
            validation = ValidationResult.REJECTED

        return SafetyCheck(
            passed=passed,
            risk_level=risk,
            validation_result=validation,
            reason="; ".join(issues) if issues else "Code passed safety checks",
            details={"issues": issues, "line_count": len(lines)},
            timestamp=datetime.now(),
            validator="CodeSafetyValidator"
        )


class ArchitectureSafetyValidator:
    """Validates neural architecture changes for safety"""

    def __init__(self, bounds: SafetyBounds):
        self.bounds = bounds

    def validate_architecture(
        self,
        new_arch: nn.Module,
        baseline_arch: Optional[nn.Module] = None
    ) -> SafetyCheck:
        """Validate architecture against safety bounds"""

        issues = []
        risk = RiskLevel.SAFE

        # Count parameters
        param_count = sum(p.numel() for p in new_arch.parameters())
        if param_count > self.bounds.max_parameters:
            issues.append(
                f"Parameter count {param_count:,} exceeds limit {self.bounds.max_parameters:,}"
            )
            risk = RiskLevel.HIGH

        # Check layer count
        layer_count = sum(1 for _ in new_arch.modules() if isinstance(_, nn.Module))
        if layer_count > self.bounds.max_layers:
            issues.append(f"Layer count {layer_count} exceeds limit {self.bounds.max_layers}")
            risk = max(risk, RiskLevel.MEDIUM)

        # If we have a baseline, check for excessive changes
        if baseline_arch is not None:
            baseline_params = sum(p.numel() for p in baseline_arch.parameters())
            param_change = abs(param_count - baseline_params) / baseline_params

            if param_change > self.bounds.max_mutation_size:
                issues.append(
                    f"Architecture change {param_change:.2%} exceeds maximum {self.bounds.max_mutation_size:.2%}"
                )
                risk = max(risk, RiskLevel.MEDIUM)

        passed = len(issues) == 0
        validation = ValidationResult.APPROVED if passed else ValidationResult.REQUIRES_HUMAN

        return SafetyCheck(
            passed=passed,
            risk_level=risk,
            validation_result=validation,
            reason="; ".join(issues) if issues else "Architecture passed safety checks",
            details={
                "param_count": param_count,
                "layer_count": layer_count,
                "issues": issues
            },
            timestamp=datetime.now(),
            validator="ArchitectureSafetyValidator"
        )


class BehaviorSafetyValidator:
    """Validates model behavior for unexpected patterns"""

    def __init__(self, bounds: SafetyBounds):
        self.bounds = bounds

    def validate_behavior(
        self,
        model: nn.Module,
        test_inputs: torch.Tensor,
        baseline_outputs: Optional[torch.Tensor] = None
    ) -> SafetyCheck:
        """Validate model behavior on test inputs"""

        issues = []
        risk = RiskLevel.SAFE

        model.eval()
        with torch.no_grad():
            try:
                outputs = model(test_inputs)
            except Exception as e:
                return SafetyCheck(
                    passed=False,
                    risk_level=RiskLevel.CRITICAL,
                    validation_result=ValidationResult.REJECTED,
                    reason=f"Model execution failed: {str(e)}",
                    details={"error": str(e)},
                    timestamp=datetime.now(),
                    validator="BehaviorSafetyValidator"
                )

        # Handle dict outputs (e.g., HRM returns dict with 'logits' key)
        if isinstance(outputs, dict):
            if 'logits' in outputs:
                outputs = outputs['logits']
            else:
                return SafetyCheck(
                    passed=False,
                    risk_level=RiskLevel.HIGH,
                    validation_result=ValidationResult.REJECTED,
                    reason="Model returned dict without 'logits' key",
                    details={"keys": list(outputs.keys())},
                    timestamp=datetime.now(),
                    validator="BehaviorSafetyValidator"
                )

        # Check for NaN or Inf
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            issues.append("Model produced NaN or Inf values")
            risk = RiskLevel.CRITICAL

        # Check output range (should be bounded)
        if outputs.abs().max() > 1e6:
            issues.append(f"Outputs have extreme values: {outputs.abs().max():.2e}")
            risk = max(risk, RiskLevel.HIGH)

        # Compare to baseline if provided
        if baseline_outputs is not None:
            mse = torch.nn.functional.mse_loss(outputs, baseline_outputs)
            relative_diff = mse / (baseline_outputs.pow(2).mean() + 1e-8)

            if relative_diff > self.bounds.max_performance_degradation:
                issues.append(
                    f"Behavior differs from baseline by {relative_diff:.2%}"
                )
                risk = max(risk, RiskLevel.MEDIUM)

        passed = len(issues) == 0
        validation = ValidationResult.APPROVED if passed else ValidationResult.REQUIRES_HUMAN

        if risk == RiskLevel.CRITICAL:
            validation = ValidationResult.REJECTED

        return SafetyCheck(
            passed=passed,
            risk_level=risk,
            validation_result=validation,
            reason="; ".join(issues) if issues else "Behavior passed safety checks",
            details={
                "output_stats": {
                    "mean": outputs.mean().item(),
                    "std": outputs.std().item(),
                    "min": outputs.min().item(),
                    "max": outputs.max().item()
                },
                "issues": issues
            },
            timestamp=datetime.now(),
            validator="BehaviorSafetyValidator"
        )


class ComprehensiveSafetyValidator:
    """Main safety validation system coordinating all validators"""

    def __init__(self, bounds: Optional[SafetyBounds] = None):
        self.bounds = bounds or SafetyBounds()
        self.code_validator = CodeSafetyValidator()
        self.arch_validator = ArchitectureSafetyValidator(self.bounds)
        self.behavior_validator = BehaviorSafetyValidator(self.bounds)

        # Audit trail
        self.validation_history: List[SafetyCheck] = []

    def validate_all(
        self,
        code: Optional[str] = None,
        architecture: Optional[nn.Module] = None,
        baseline_architecture: Optional[nn.Module] = None,
        test_inputs: Optional[torch.Tensor] = None,
        baseline_outputs: Optional[torch.Tensor] = None
    ) -> Tuple[bool, List[SafetyCheck]]:
        """
        Run all applicable safety checks

        Returns:
            (all_passed, list_of_checks)
        """

        checks = []

        # Validate code if provided
        if code is not None:
            check = self.code_validator.validate_code(code)
            checks.append(check)
            self.validation_history.append(check)

        # Validate architecture if provided
        if architecture is not None:
            check = self.arch_validator.validate_architecture(
                architecture, baseline_architecture
            )
            checks.append(check)
            self.validation_history.append(check)

        # Validate behavior if we have both architecture and test inputs
        if architecture is not None and test_inputs is not None:
            check = self.behavior_validator.validate_behavior(
                architecture, test_inputs, baseline_outputs
            )
            checks.append(check)
            self.validation_history.append(check)

        # Determine overall result
        all_passed = all(check.passed for check in checks)
        requires_human = any(
            check.validation_result == ValidationResult.REQUIRES_HUMAN
            for check in checks
        )
        has_rejection = any(
            check.validation_result == ValidationResult.REJECTED
            for check in checks
        )

        # Log results
        self._log_validation_results(checks, all_passed)

        return all_passed and not requires_human and not has_rejection, checks

    def _log_validation_results(self, checks: List[SafetyCheck], all_passed: bool):
        """Log validation results"""

        status = "PASSED" if all_passed else "FAILED"
        logger.info(f"Safety Validation {status}: {len(checks)} checks performed")

        for check in checks:
            level = logging.INFO if check.passed else logging.WARNING
            logger.log(
                level,
                f"{check.validator}: {check.validation_result.value} - {check.reason}"
            )

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations performed"""

        total = len(self.validation_history)
        passed = sum(1 for c in self.validation_history if c.passed)

        return {
            "total_validations": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "recent_checks": [
                {
                    "validator": c.validator,
                    "passed": c.passed,
                    "risk": c.risk_level.value,
                    "timestamp": c.timestamp.isoformat()
                }
                for c in self.validation_history[-10:]
            ]
        }

    def emergency_stop(self, reason: str):
        """Emergency stop - log and raise exception"""

        logger.critical(f"EMERGENCY STOP: {reason}")

        check = SafetyCheck(
            passed=False,
            risk_level=RiskLevel.CRITICAL,
            validation_result=ValidationResult.EMERGENCY_STOP,
            reason=reason,
            details={"emergency": True},
            timestamp=datetime.now(),
            validator="ComprehensiveSafetyValidator"
        )

        self.validation_history.append(check)

        raise RuntimeError(f"EMERGENCY STOP: {reason}")
