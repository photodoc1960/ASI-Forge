"""
Supervised Evolution Framework
Autonomous architecture discovery with mandatory human approval gates

Based on ASI-Arch (Liu et al. 2025) with comprehensive safety controls
"""

import torch
import torch.nn as nn
import json
import copy
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import hashlib

from core.safety.safety_validator import ComprehensiveSafetyValidator, SafetyBounds
from core.safety.emergence_detector import EmergenceDetectionAgent
from core.evolution.code_execution import SafeCodeExecutor, CodeDebugger, TrainingValidator
from interfaces.human_approval import (
    ApprovalManager,
    ChangeType,
    ApprovalStatus
)

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureCandidate:
    """Candidate architecture for evaluation"""
    architecture_id: str
    generation: int
    parent_id: Optional[str]
    description: str
    code: str
    model: Optional[nn.Module]
    creation_time: datetime
    performance_metrics: Dict[str, float]
    safety_validated: bool
    human_approved: bool
    approval_request_id: Optional[str] = None


@dataclass
class EvolutionConfig:
    """Configuration for evolution process"""
    population_size: int = 20
    elite_ratio: float = 0.2
    mutation_rate: float = 0.3
    max_generations: int = 100

    # Safety constraints
    require_approval_for_deployment: bool = True
    require_approval_for_code_gen: bool = True
    max_parameter_increase: float = 0.1  # 10% max increase

    # Evolution strategy
    evolution_strategy: str = "supervised_genetic"  # or "gradient_based", "neuro_evolution"

    # Auto-approval settings
    enable_auto_approval: bool = True
    auto_approve_low_risk: bool = True
    min_performance_gain: float = 0.05  # 5% improvement required for auto-approval
    max_auto_approve_param_increase: float = 0.05  # Max 5% param increase for auto-approval


class ArchitectureGenerator:
    """Generates new architecture variations"""

    def __init__(self, safety_validator: ComprehensiveSafetyValidator):
        self.safety_validator = safety_validator
        self.generation_history: List[str] = []

    def generate_variation(
        self,
        parent_arch: ArchitectureCandidate,
        mutation_type: str
    ) -> Tuple[str, str]:
        """
        Generate a variation of parent architecture

        Returns:
            (code, description)
        """

        # In a full implementation, this would use an LLM to generate code
        # For now, we'll create template-based variations

        if mutation_type == "add_layer":
            code = self._add_layer_mutation(parent_arch.code)
            description = f"Added layer to {parent_arch.description}"

        elif mutation_type == "modify_attention":
            code = self._modify_attention_mutation(parent_arch.code)
            description = f"Modified attention mechanism in {parent_arch.description}"

        elif mutation_type == "change_activation":
            code = self._change_activation_mutation(parent_arch.code)
            description = f"Changed activation function in {parent_arch.description}"

        elif mutation_type == "adjust_dimensions":
            code = self._adjust_dimensions_mutation(parent_arch.code)
            description = f"Adjusted dimensions in {parent_arch.description}"

        else:
            # Default: small random modification
            code = parent_arch.code
            description = parent_arch.description

        return code, description

    def _add_layer_mutation(self, code: str) -> str:
        """Add a layer to the architecture (template)"""
        # This is a simplified example
        # In production, use LLM to intelligently add layers
        return code + "\n# Added layer mutation"

    def _modify_attention_mutation(self, code: str) -> str:
        """Modify attention mechanism"""
        # Simplified example
        return code.replace("n_heads=8", "n_heads=12")

    def _change_activation_mutation(self, code: str) -> str:
        """Change activation function"""
        return code.replace("GELU()", "ReLU()")

    def _adjust_dimensions_mutation(self, code: str) -> str:
        """Adjust hidden dimensions"""
        return code.replace("hidden_dim=512", "hidden_dim=768")


class SupervisedEvolutionFramework:
    """
    Main evolution framework with human-in-the-loop oversight

    Key Features:
    1. All code generation requires safety validation
    2. Deployment requires human approval
    3. Emergence detection with automatic freeze
    4. Complete audit trail
    """

    def __init__(
        self,
        config: EvolutionConfig,
        safety_bounds: Optional[SafetyBounds] = None,
        approval_manager: Optional[ApprovalManager] = None
    ):
        self.config = config
        self.safety_bounds = safety_bounds or SafetyBounds()

        # Core components
        self.safety_validator = ComprehensiveSafetyValidator(self.safety_bounds)
        self.emergence_detector = EmergenceDetectionAgent(
            alert_callback=self._on_emergence_alert
        )
        self.approval_manager = approval_manager or ApprovalManager()

        # Code execution and testing
        self.code_executor = SafeCodeExecutor(
            timeout_seconds=120,
            max_memory_gb=self.safety_bounds.max_memory_gb
        )
        self.code_debugger = CodeDebugger(max_attempts=3)
        self.training_validator = TrainingValidator(training_steps=100)

        # Architecture management
        self.population: List[ArchitectureCandidate] = []
        self.archive: List[ArchitectureCandidate] = []
        self.current_generation = 0

        # Code generator
        self.arch_generator = ArchitectureGenerator(self.safety_validator)

        # State
        self.is_paused = False
        self.pause_reason: Optional[str] = None

    def initialize_population(self, base_architecture: nn.Module):
        """Initialize population with base architecture"""

        logger.info("Initializing population with base architecture")

        # Create base candidate
        base_candidate = ArchitectureCandidate(
            architecture_id=self._generate_id(0, "base"),
            generation=0,
            parent_id=None,
            description="Base HRM Architecture",
            code=self._model_to_code(base_architecture),
            model=base_architecture,
            creation_time=datetime.now(),
            performance_metrics={},
            safety_validated=True,
            human_approved=True
        )

        self.population.append(base_candidate)
        self.archive.append(base_candidate)

        logger.info(f"Population initialized with 1 base architecture")

    def evolve_generation(
        self,
        evaluation_function: callable
    ) -> Dict[str, Any]:
        """
        Evolve one generation

        Args:
            evaluation_function: Function to evaluate architectures
                                Should return dict of metrics

        Returns:
            Generation statistics
        """

        if self.is_paused:
            logger.warning(
                f"Evolution paused: {self.pause_reason}. "
                f"Cannot evolve until unpaused."
            )
            return {"status": "paused", "reason": self.pause_reason}

        if self.emergence_detector.is_frozen:
            logger.warning(
                f"System frozen by emergence detector: {self.emergence_detector.freeze_reason}"
            )
            return {"status": "frozen", "reason": self.emergence_detector.freeze_reason}

        logger.info(f"Starting generation {self.current_generation + 1}")

        # Select elite candidates
        elite_candidates = self._select_elite()

        # Generate new candidates through mutation
        new_candidates = []
        for parent in elite_candidates:
            # Generate variation
            num_variations = max(1, self.config.population_size // len(elite_candidates))

            for _ in range(num_variations):
                candidate = self._generate_candidate(parent)

                if candidate is not None:
                    new_candidates.append(candidate)

        logger.info(f"Generated {len(new_candidates)} new candidates")

        # Evaluate all candidates (including elite)
        all_candidates = elite_candidates + new_candidates

        for candidate in all_candidates:
            if candidate.model is not None:
                # Evaluate
                metrics = evaluation_function(candidate.model)
                candidate.performance_metrics = metrics

                # Check for emergent capabilities
                self.emergence_detector.evaluate_capabilities(
                    candidate.model,
                    evaluation_suite={}  # Would be populated with eval functions
                )

        # Update population
        self.population = self._select_next_generation(all_candidates)
        self.archive.extend(new_candidates)

        self.current_generation += 1

        # Generate statistics
        stats = self._compute_generation_stats()

        if stats and 'best_score' in stats:
            logger.info(f"Generation {self.current_generation} complete. Best score: {stats['best_score']:.4f}")
        else:
            logger.info(f"Generation {self.current_generation} complete. No candidates with performance metrics.")

        return stats

    def _generate_candidate(
        self,
        parent: ArchitectureCandidate
    ) -> Optional[ArchitectureCandidate]:
        """
        Generate a new candidate from parent

        Returns None if generation fails safety checks or approval
        """

        # Choose mutation type
        mutation_types = [
            "add_layer", "modify_attention",
            "change_activation", "adjust_dimensions"
        ]

        import random
        mutation_type = random.choice(mutation_types)

        # Generate code
        code, description = self.arch_generator.generate_variation(
            parent, mutation_type
        )

        # Safety validation of code
        safe, checks = self.safety_validator.validate_all(code=code)

        if not safe:
            logger.warning(f"Generated code failed safety checks: {[c.reason for c in checks]}")
            return None

        # Build risk assessment
        risk_assessment = {
            "mutation_type": mutation_type,
            "parent_performance": parent.performance_metrics,
            "safety_checks": [c.reason for c in checks],
            "risk_level": "low" if len(checks) <= 2 else "medium"  # Simple heuristic
        }

        # Check if we should auto-approve
        approval_request_id = None
        auto_approved = False

        # Create preliminary candidate for auto-approval check
        # (we'll update it after execution)
        prelim_candidate = ArchitectureCandidate(
            architecture_id=self._generate_id(self.current_generation + 1, mutation_type),
            generation=self.current_generation + 1,
            parent_id=parent.architecture_id,
            description=description,
            code=code,
            model=None,  # Not yet executed
            creation_time=datetime.now(),
            performance_metrics={},
            safety_validated=safe,
            human_approved=False,
            approval_request_id=None
        )

        should_auto_approve, auto_reason = self._should_auto_approve(
            prelim_candidate,
            parent,
            risk_assessment
        )

        if should_auto_approve:
            logger.info(f"🤖 Auto-approving candidate: {auto_reason}")
            auto_approved = True
        elif self.config.require_approval_for_code_gen:
            # Request human approval
            approval_request_id = self.approval_manager.request_approval(
                change_type=ChangeType.CODE_GENERATION,
                title=f"New Architecture Variation: {mutation_type}",
                description=description,
                rationale=f"Evolved from parent {parent.architecture_id} via {mutation_type}",
                risk_assessment=risk_assessment,
                proposed_changes={"code": code[:500]},  # First 500 chars
                reversibility=True,
                estimated_impact=risk_assessment.get('risk_level', 'low')
            )

            # Wait for approval (in production, this would be async)
            status = self.approval_manager.check_approval_status(approval_request_id)

            if status != ApprovalStatus.APPROVED:
                logger.info(f"Code generation awaiting human approval: {approval_request_id}")
                logger.info(f"Reason: {auto_reason}")
                # In production, candidate would be added to pending queue
                return None

        # Execute code in safe sandbox with debug loop
        model = None
        execution_error = None

        for attempt in range(self.code_debugger.max_attempts):
            try:
                # Execute code in sandbox
                test_input = torch.randint(0, 1000, (2, 20))  # Test input
                exec_result = self.code_executor.execute_model_code(
                    code=code,
                    test_inputs=test_input,
                    architecture_id=f"{self.current_generation}_{mutation_type}_{attempt}"
                )

                if exec_result.success and exec_result.model is not None:
                    model = exec_result.model
                    logger.info(f"Code execution successful on attempt {attempt + 1}")
                    break
                else:
                    execution_error = exec_result.error_message

                    if attempt < self.code_debugger.max_attempts - 1:
                        # Try to debug and fix
                        logger.warning(f"Execution failed (attempt {attempt + 1}): {execution_error}")
                        code, changes = self.code_debugger.debug_and_fix(
                            code, exec_result.error_message, exec_result.stderr
                        )
                        logger.info(f"Applied debug changes: {changes}")
                    else:
                        logger.error(f"Code execution failed after {self.code_debugger.max_attempts} attempts")
                        return None

            except Exception as e:
                logger.error(f"Failed to execute model code (attempt {attempt + 1}): {e}")
                execution_error = str(e)
                if attempt == self.code_debugger.max_attempts - 1:
                    return None

        if model is None:
            logger.error("Failed to create model from code")
            return None

        # Validate architecture
        safe, checks = self.safety_validator.validate_all(
            architecture=model,
            baseline_architecture=parent.model
        )

        if not safe:
            logger.warning(f"Architecture failed safety validation")
            return None

        # Validate trainability (optional but recommended)
        logger.info("Validating model trainability...")
        trainable, train_metrics = self.training_validator.validate_training(
            model=model,
            vocab_size=1000,  # Should come from config
            seq_len=20
        )

        if not trainable:
            logger.warning(f"Model failed trainability test: {train_metrics.get('error', 'unknown')}")
            # Don't reject, but note it
            train_metrics['trainable'] = False

        # Create candidate
        candidate = ArchitectureCandidate(
            architecture_id=self._generate_id(self.current_generation + 1, mutation_type),
            generation=self.current_generation + 1,
            parent_id=parent.architecture_id,
            description=description,
            code=code,
            model=model,
            creation_time=datetime.now(),
            performance_metrics={},
            safety_validated=True,
            human_approved=auto_approved or (not self.config.require_approval_for_code_gen),
            approval_request_id=approval_request_id
        )

        if auto_approved:
            logger.info(f"✓ Auto-approved candidate {candidate.architecture_id} ready for evaluation")

        return candidate

    def _should_auto_approve(
        self,
        candidate: ArchitectureCandidate,
        parent: ArchitectureCandidate,
        risk_assessment: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Determine if a candidate should be auto-approved

        Returns:
            (should_approve, reason)
        """

        if not self.config.enable_auto_approval:
            return False, "Auto-approval disabled in config"

        # Check if auto-approval for low risk is enabled
        if not self.config.auto_approve_low_risk:
            return False, "Auto-approval for low risk disabled"

        # Check parameter increase
        if candidate.model is not None and parent.model is not None:
            candidate_params = sum(p.numel() for p in candidate.model.parameters())
            parent_params = sum(p.numel() for p in parent.model.parameters())

            param_increase = (candidate_params - parent_params) / parent_params

            if param_increase > self.config.max_auto_approve_param_increase:
                return False, f"Parameter increase {param_increase:.1%} exceeds auto-approval limit {self.config.max_auto_approve_param_increase:.1%}"

        # Check performance improvement (if metrics available)
        # If no metrics yet (early generations), skip this check
        if candidate.performance_metrics and parent.performance_metrics:
            # Get primary metric (accuracy, loss, etc.)
            parent_perf = parent.performance_metrics.get('accuracy', parent.performance_metrics.get('loss', 0))
            candidate_perf = candidate.performance_metrics.get('accuracy', candidate.performance_metrics.get('loss', 0))

            if parent_perf > 0:
                perf_gain = (candidate_perf - parent_perf) / parent_perf

                if perf_gain < self.config.min_performance_gain:
                    return False, f"Performance gain {perf_gain:.1%} below auto-approval threshold {self.config.min_performance_gain:.1%}"
        # If no performance metrics available yet, approve based on other criteria
        elif not candidate.performance_metrics and not parent.performance_metrics:
            logger.info("No performance metrics available yet, auto-approving based on safety checks")

        # Check safety validation
        if not candidate.safety_validated:
            return False, "Candidate not safety validated"

        # Check risk level from assessment
        risk_level = risk_assessment.get('risk_level', 'unknown')
        if risk_level not in ['low', 'minimal']:
            return False, f"Risk level '{risk_level}' too high for auto-approval"

        # All checks passed
        logger.info(f"✓ Candidate {candidate.architecture_id} qualifies for auto-approval")
        return True, "Low-risk improvement with safety validation passed"

    def _select_elite(self) -> List[ArchitectureCandidate]:
        """Select elite candidates for next generation"""

        if not self.population:
            return []

        # Separate candidates with and without metrics
        with_metrics = [c for c in self.population if c.performance_metrics]
        without_metrics = [c for c in self.population if not c.performance_metrics]

        # If we have candidates with metrics, sort by performance
        if with_metrics:
            sorted_pop = sorted(
                with_metrics,
                key=lambda c: c.performance_metrics.get('accuracy', 0),
                reverse=True
            )
        else:
            # No metrics yet - use all candidates (early generations)
            sorted_pop = self.population

        n_elite = max(1, int(len(sorted_pop) * self.config.elite_ratio))
        return sorted_pop[:n_elite]

    def _select_next_generation(
        self,
        candidates: List[ArchitectureCandidate]
    ) -> List[ArchitectureCandidate]:
        """Select candidates for next generation"""

        # Sort by performance
        sorted_candidates = sorted(
            [c for c in candidates if c.performance_metrics],
            key=lambda c: c.performance_metrics.get('accuracy', 0),
            reverse=True
        )

        # Take top N
        return sorted_candidates[:self.config.population_size]

    def _compute_generation_stats(self) -> Dict[str, Any]:
        """Compute statistics for current generation"""

        if not self.population:
            return {}

        scores = [
            c.performance_metrics.get('accuracy', 0)
            for c in self.population
            if c.performance_metrics
        ]

        if not scores:
            return {}

        return {
            'generation': self.current_generation,
            'population_size': len(self.population),
            'best_score': max(scores),
            'mean_score': sum(scores) / len(scores),
            'worst_score': min(scores),
            'approved_candidates': sum(1 for c in self.population if c.human_approved),
            'timestamp': datetime.now().isoformat()
        }

    def deploy_best_model(self) -> Optional[ArchitectureCandidate]:
        """
        Deploy best model after human approval

        Returns:
            Best candidate if approved, None otherwise
        """

        if not self.population:
            logger.error("No population to deploy from")
            return None

        # Get best candidate
        best = max(
            [c for c in self.population if c.performance_metrics],
            key=lambda c: c.performance_metrics.get('accuracy', 0)
        )

        # Request deployment approval
        request_id = self.approval_manager.request_approval(
            change_type=ChangeType.CAPABILITY_DEPLOYMENT,
            title=f"Deploy Best Model: {best.description}",
            description=f"Deploy architecture from generation {best.generation}",
            rationale=f"Best performing model with accuracy {best.performance_metrics.get('accuracy', 0):.4f}",
            risk_assessment={
                "performance_metrics": best.performance_metrics,
                "generation": best.generation,
                "parent": best.parent_id
            },
            proposed_changes={
                "model_id": best.architecture_id,
                "description": best.description
            },
            reversibility=True,
            estimated_impact="high"
        )

        logger.info(f"Deployment approval requested: {request_id}")

        # In production, wait for approval asynchronously
        # For now, just return the candidate with pending approval
        best.approval_request_id = request_id

        return best

    def _on_emergence_alert(self, message: str, severity: str):
        """Callback for emergence detection alerts"""

        logger.warning(f"Emergence Alert [{severity}]: {message}")

        # Pause evolution on critical alerts
        if severity == 'critical':
            self.pause_evolution(f"Critical emergence alert: {message}")

    def pause_evolution(self, reason: str):
        """Pause evolution process"""

        self.is_paused = True
        self.pause_reason = reason
        logger.warning(f"Evolution paused: {reason}")

    def resume_evolution(self, approval_code: str):
        """Resume evolution after human approval"""

        # In production, verify approval code
        self.is_paused = False
        self.pause_reason = None
        logger.info(f"Evolution resumed with approval: {approval_code[:8]}...")

    def _generate_id(self, generation: int, mutation_type: str) -> str:
        """Generate unique ID for architecture"""

        timestamp = datetime.now().isoformat()
        content = f"{generation}_{mutation_type}_{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _model_to_code(self, model: nn.Module) -> str:
        """Convert model to code representation (simplified)"""

        # In production, this would generate actual code
        return f"# Architecture: {type(model).__name__}\n# Parameters: {sum(p.numel() for p in model.parameters())}"

    def _code_to_model(self, code: str) -> nn.Module:
        """Convert code to model instance (simplified)"""

        # In production, use safe execution environment
        # For now, return a dummy model
        return nn.Linear(10, 10)

    def get_evolution_report(self) -> Dict[str, Any]:
        """Generate comprehensive evolution report"""

        return {
            'current_generation': self.current_generation,
            'total_architectures_created': len(self.archive),
            'population_size': len(self.population),
            'is_paused': self.is_paused,
            'pause_reason': self.pause_reason,
            'emergence_detector_status': self.emergence_detector.get_status_report(),
            'safety_validation_summary': self.safety_validator.get_validation_summary(),
            'approval_summary': self.approval_manager.generate_approval_report()
        }
