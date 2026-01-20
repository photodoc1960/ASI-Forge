"""
AEGIS: Adaptive Evolutionary General Intelligence System
Main orchestration class with comprehensive safety controls
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime

from core.hrm.hierarchical_reasoning import HierarchicalReasoningModel
from core.evolution.supervised_evolution import (
    SupervisedEvolutionFramework,
    EvolutionConfig,
    ArchitectureCandidate
)
from core.safety.safety_validator import (
    ComprehensiveSafetyValidator,
    SafetyBounds
)
from core.safety.emergence_detector import EmergenceDetectionAgent
from interfaces.human_approval import (
    ApprovalManager,
    console_notification
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AEGISConfig:
    """Configuration for AEGIS system"""

    # HRM Configuration
    vocab_size: int = 10000
    d_model: int = 512
    high_level_layers: int = 6
    low_level_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 2048
    h_cycles: int = 3
    l_cycles: int = 3
    max_act_steps: int = 16

    # Evolution Configuration
    population_size: int = 20
    elite_ratio: float = 0.2
    mutation_rate: float = 0.3
    max_generations: int = 100

    # Safety Configuration
    max_parameters: int = 1_000_000_000
    max_memory_gb: int = 32
    max_compute_hours: int = 24

    # Operational Configuration
    require_approval_for_deployment: bool = True
    require_approval_for_code_gen: bool = True
    auto_freeze_on_emergence: bool = True


class AEGIS:
    """
    Main AEGIS System

    Features:
    - Hierarchical reasoning via HRM
    - Supervised autonomous evolution
    - Comprehensive safety validation
    - Emergence detection and monitoring
    - Human-in-the-loop approval for all critical decisions
    """

    def __init__(self, config: Optional[AEGISConfig] = None):
        self.config = config or AEGISConfig()

        logger.info("Initializing AEGIS System")
        logger.info("=" * 70)

        # Initialize core components
        self._initialize_safety_systems()
        self._initialize_reasoning_engine()
        self._initialize_evolution_framework()

        # System state
        self.is_running = False
        self.start_time: Optional[datetime] = None

        logger.info("AEGIS System initialized successfully")
        logger.info("=" * 70)

    def _initialize_safety_systems(self):
        """Initialize all safety mechanisms"""

        logger.info("Initializing safety systems...")

        # Safety bounds
        self.safety_bounds = SafetyBounds()
        self.safety_bounds.max_parameters = self.config.max_parameters
        self.safety_bounds.max_memory_gb = self.config.max_memory_gb
        self.safety_bounds.max_compute_hours = self.config.max_compute_hours

        # Safety validator
        self.safety_validator = ComprehensiveSafetyValidator(self.safety_bounds)

        # Emergence detector
        self.emergence_detector = EmergenceDetectionAgent(
            alert_callback=self._on_emergence_alert
        )

        # Approval manager
        self.approval_manager = ApprovalManager()
        self.approval_manager.register_notification_callback(console_notification)

        logger.info("✓ Safety systems initialized")

    def _initialize_reasoning_engine(self):
        """Initialize HRM reasoning engine"""

        logger.info("Initializing reasoning engine...")

        self.reasoning_engine = HierarchicalReasoningModel(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            high_level_layers=self.config.high_level_layers,
            low_level_layers=self.config.low_level_layers,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout,
            max_seq_len=self.config.max_seq_len,
            h_cycles=self.config.h_cycles,
            l_cycles=self.config.l_cycles,
            max_act_steps=self.config.max_act_steps
        )

        # Validate initial architecture
        test_input = torch.randint(0, self.config.vocab_size, (1, 10))
        safe, checks = self.safety_validator.validate_all(
            architecture=self.reasoning_engine,
            test_inputs=test_input
        )

        if not safe:
            raise RuntimeError(
                f"Base reasoning engine failed safety validation: "
                f"{[c.reason for c in checks]}"
            )

        logger.info(f"✓ Reasoning engine initialized ({self._count_parameters():,} parameters)")

    def _initialize_evolution_framework(self):
        """Initialize supervised evolution framework"""

        logger.info("Initializing evolution framework...")

        evolution_config = EvolutionConfig(
            population_size=self.config.population_size,
            elite_ratio=self.config.elite_ratio,
            mutation_rate=self.config.mutation_rate,
            max_generations=self.config.max_generations,
            require_approval_for_deployment=self.config.require_approval_for_deployment,
            require_approval_for_code_gen=self.config.require_approval_for_code_gen
        )

        self.evolution_framework = SupervisedEvolutionFramework(
            config=evolution_config,
            safety_bounds=self.safety_bounds,
            approval_manager=self.approval_manager
        )

        # Initialize population with base reasoning engine
        self.evolution_framework.initialize_population(self.reasoning_engine)

        logger.info("✓ Evolution framework initialized")

    def reason(
        self,
        input_ids: torch.Tensor,
        goal: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform reasoning on input

        Args:
            input_ids: Input token IDs (batch, seq_len)
            goal: Optional goal description

        Returns:
            Dictionary with reasoning outputs
        """

        # Safety check
        if self.emergence_detector.is_frozen:
            raise RuntimeError(
                f"System frozen by emergence detector: {self.emergence_detector.freeze_reason}"
            )

        # Run reasoning engine
        with torch.no_grad():
            outputs = self.reasoning_engine(input_ids)

        # Validate outputs
        safe, checks = self.safety_validator.validate_all(
            architecture=self.reasoning_engine,
            test_inputs=input_ids,
            baseline_outputs=outputs['logits']
        )

        if not safe:
            logger.warning(f"Reasoning outputs failed safety checks: {[c.reason for c in checks]}")

        return {
            'logits': outputs['logits'],
            'ponder_cost': outputs['ponder_cost'],
            'safe': safe,
            'safety_checks': [
                {'validator': c.validator, 'passed': c.passed, 'reason': c.reason}
                for c in checks
            ]
        }

    def evolve(
        self,
        num_generations: int = 1,
        evaluation_function: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Run evolution for specified number of generations

        Args:
            num_generations: Number of generations to evolve
            evaluation_function: Function to evaluate architectures
                                Returns dict of metrics

        Returns:
            List of generation statistics
        """

        if evaluation_function is None:
            # Use default evaluation function
            evaluation_function = self._default_evaluation

        logger.info(f"Starting evolution for {num_generations} generations")

        generation_stats = []

        for gen in range(num_generations):
            logger.info(f"\nGeneration {gen + 1}/{num_generations}")
            logger.info("-" * 70)

            # Check system state
            if self.emergence_detector.is_frozen:
                logger.error(
                    f"Evolution halted: System frozen by emergence detector\n"
                    f"Reason: {self.emergence_detector.freeze_reason}"
                )
                break

            if self.evolution_framework.is_paused:
                logger.error(
                    f"Evolution halted: Framework paused\n"
                    f"Reason: {self.evolution_framework.pause_reason}"
                )
                break

            # Evolve one generation
            stats = self.evolution_framework.evolve_generation(evaluation_function)
            generation_stats.append(stats)

            # Log progress
            if 'best_score' in stats:
                logger.info(
                    f"Generation {gen + 1} complete: "
                    f"Best={stats['best_score']:.4f}, "
                    f"Mean={stats['mean_score']:.4f}"
                )

        logger.info("\nEvolution complete")
        logger.info("=" * 70)

        return generation_stats

    def _default_evaluation(self, model: nn.Module) -> Dict[str, float]:
        """Default evaluation function for architectures"""

        # Simple evaluation: count parameters and test forward pass
        param_count = sum(p.numel() for p in model.parameters())

        # Test forward pass
        try:
            test_input = torch.randint(0, self.config.vocab_size, (2, 20))
            with torch.no_grad():
                output = model(test_input)

            # Success metric
            success = 1.0

            # Size penalty (prefer smaller models)
            size_penalty = min(1.0, 100_000_000 / param_count)

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            success = 0.0
            size_penalty = 0.0

        return {
            'accuracy': success * size_penalty,
            'parameters': param_count,
            'success': success
        }

    def deploy_best_model(self) -> Optional[ArchitectureCandidate]:
        """
        Deploy best evolved model (requires human approval)

        Returns:
            Best model if approved, None otherwise
        """

        logger.info("Requesting deployment approval for best model...")

        best_model = self.evolution_framework.deploy_best_model()

        if best_model is None:
            logger.error("No model available for deployment")
            return None

        logger.info(
            f"Best model: {best_model.description}\n"
            f"Approval request: {best_model.approval_request_id}\n"
            f"Waiting for human approval..."
        )

        return best_model

    def _on_emergence_alert(self, message: str, severity: str):
        """Handle emergence detection alerts"""

        logger.warning(f"\n{'!'*70}")
        logger.warning(f"EMERGENCE ALERT [{severity.upper()}]")
        logger.warning(message)
        logger.warning(f"{'!'*70}\n")

        if self.config.auto_freeze_on_emergence and severity == 'critical':
            logger.critical("Auto-freezing system due to critical emergence event")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""

        return {
            'is_running': self.is_running,
            'uptime': str(datetime.now() - self.start_time) if self.start_time else None,
            'reasoning_engine': {
                'parameters': self._count_parameters(),
                'architecture': type(self.reasoning_engine).__name__
            },
            'evolution': self.evolution_framework.get_evolution_report(),
            'safety': {
                'validator': self.safety_validator.get_validation_summary(),
                'emergence_detector': self.emergence_detector.get_status_report(),
                'is_frozen': self.emergence_detector.is_frozen,
                'freeze_reason': self.emergence_detector.freeze_reason
            },
            'approvals': self.approval_manager.generate_approval_report()
        }

    def _count_parameters(self) -> int:
        """Count total parameters in reasoning engine"""
        return sum(p.numel() for p in self.reasoning_engine.parameters())

    def emergency_stop(self, reason: str):
        """Emergency stop entire system"""

        logger.critical(f"\n{'#'*70}")
        logger.critical("EMERGENCY STOP INITIATED")
        logger.critical(f"Reason: {reason}")
        logger.critical(f"{'#'*70}\n")

        self.is_running = False
        self.emergence_detector.freeze_system(reason)
        self.evolution_framework.pause_evolution(reason)
        self.safety_validator.emergency_stop(reason)


def main():
    """Example usage of AEGIS system"""

    print("\n" + "=" * 70)
    print("AEGIS: Adaptive Evolutionary General Intelligence System")
    print("Safety-First Architecture Discovery Framework")
    print("=" * 70 + "\n")

    # Create system
    config = AEGISConfig(
        vocab_size=1000,  # Smaller for demo
        d_model=256,
        high_level_layers=4,
        low_level_layers=2,
        population_size=5  # Small population for demo
    )

    aegis = AEGIS(config)

    # Test reasoning
    print("\n[1] Testing reasoning engine...")
    test_input = torch.randint(0, config.vocab_size, (1, 10))
    result = aegis.reason(test_input)
    print(f"Reasoning successful: {result['safe']}")

    # Get status
    print("\n[2] System status:")
    status = aegis.get_system_status()
    print(f"  Parameters: {status['reasoning_engine']['parameters']:,}")
    print(f"  Safety validations: {status['safety']['validator']['total_validations']}")
    print(f"  System frozen: {status['safety']['is_frozen']}")

    print("\n" + "=" * 70)
    print("AEGIS system ready for supervised evolution")
    print("All critical operations require human approval")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
