"""
ASI-Forge Specialization Manager

Core management class for domain specializations.
Handles loading, creating, switching, and lifecycle of specializations.

Part of the ASI-Forge meta-research framework, extending ASI-Arch to support any research domain.
"""

import asyncio
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try multiple possible locations for .env
    possible_paths = [
        Path(__file__).parent.parent.parent / ".env",  # /ASI-Arch/.env from manager.py
        Path(__file__).parent.parent / ".env",          # /ASI-Arch/pipeline/.env
        Path.cwd() / ".env",                            # Current working directory
        Path.cwd().parent / ".env",                     # Parent of cwd
    ]
    for env_path in possible_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass  # dotenv not installed, rely on system environment
from typing import List, Optional, Callable, Awaitable
from datetime import datetime

from .schema import (
    DomainSpecialization,
    SpecializationSummary,
    InitMode
)
from .storage import SpecializationStorage


class SpecializationError(Exception):
    """Base exception for specialization errors."""
    pass


class SpecializationNotFoundError(SpecializationError):
    """Raised when a specialization is not found."""
    pass


class SpecializationNotValidatedError(SpecializationError):
    """Raised when trying to use an unvalidated specialization."""
    pass


class SpecializationManager:
    """
    Manages domain specializations for the ASI-Arch framework.

    This class provides the interface for:
    - Listing available specializations
    - Loading and activating specializations
    - Creating new specializations
    - Switching between specializations
    - Managing specialization lifecycle
    """

    def __init__(
        self,
        base_path: str = None,
        allow_unvalidated: bool = False
    ):
        """
        Initialize the specialization manager.

        Args:
            base_path: Root directory for specialization storage
            allow_unvalidated: If True, allow using unvalidated specializations
        """
        # Default to specializations directory at project root
        if base_path is None:
            base_path = str(Path(__file__).parent.parent.parent / "specializations")
        self.storage = SpecializationStorage(base_path)
        self.allow_unvalidated = allow_unvalidated
        self._current: Optional[DomainSpecialization] = None
        self._on_switch_callbacks: List[Callable[[DomainSpecialization], Awaitable[None]]] = []

    @property
    def current(self) -> Optional[DomainSpecialization]:
        """Get the currently active specialization."""
        return self._current

    @property
    def current_name(self) -> Optional[str]:
        """Get the name of the currently active specialization."""
        return self._current.name if self._current else None

    def register_switch_callback(
        self,
        callback: Callable[[DomainSpecialization], Awaitable[None]]
    ):
        """
        Register a callback to be called when specialization is switched.

        This allows other components to update their state when the
        active specialization changes.

        Args:
            callback: Async function that takes the new specialization
        """
        self._on_switch_callbacks.append(callback)

    async def _notify_switch(self, spec: DomainSpecialization):
        """Notify all registered callbacks about a specialization switch."""
        for callback in self._on_switch_callbacks:
            await callback(spec)

    def list_specializations(self) -> List[SpecializationSummary]:
        """
        List all available specializations.

        Returns:
            List of SpecializationSummary objects
        """
        return self.storage.list_summaries()

    def list_names(self) -> List[str]:
        """
        List names of all available specializations.

        Returns:
            List of specialization names
        """
        return self.storage.list_all()

    def exists(self, name: str) -> bool:
        """
        Check if a specialization exists.

        Args:
            name: Specialization name

        Returns:
            True if exists, False otherwise
        """
        return self.storage.exists(name)

    def load(self, name: str) -> DomainSpecialization:
        """
        Load a specialization without activating it.

        Args:
            name: Specialization name

        Returns:
            The loaded DomainSpecialization

        Raises:
            SpecializationNotFoundError: If not found
        """
        try:
            return self.storage.load(name)
        except FileNotFoundError as e:
            raise SpecializationNotFoundError(str(e))

    async def activate(self, name: str) -> DomainSpecialization:
        """
        Load and activate a specialization.

        This sets the specialization as the current one and notifies
        all registered callbacks.

        Args:
            name: Specialization name

        Returns:
            The activated DomainSpecialization

        Raises:
            SpecializationNotFoundError: If not found
            SpecializationNotValidatedError: If not validated and allow_unvalidated is False
        """
        spec = self.load(name)

        if not spec.is_validated and not self.allow_unvalidated:
            raise SpecializationNotValidatedError(
                f"Specialization '{name}' has not been validated. "
                f"Run validation first or set allow_unvalidated=True."
            )

        self._current = spec
        await self._notify_switch(spec)
        return spec

    def get_current(self) -> DomainSpecialization:
        """
        Get the currently active specialization.

        Returns:
            The current DomainSpecialization

        Raises:
            SpecializationError: If no specialization is active
        """
        if self._current is None:
            raise SpecializationError("No specialization is currently active")
        return self._current

    async def switch(self, name: str) -> DomainSpecialization:
        """
        Switch to a different specialization.

        Saves any changes to the current specialization before switching.

        Args:
            name: Name of the specialization to switch to

        Returns:
            The newly activated DomainSpecialization
        """
        # Save current if exists
        if self._current:
            self.storage.save(self._current)

        return await self.activate(name)

    def save_current(self):
        """Save the current specialization to disk."""
        if self._current:
            self.storage.save(self._current)

    def update_current_statistics(self, score: float):
        """
        Update statistics for the current specialization.

        Args:
            score: Score from the latest experiment
        """
        if self._current:
            self._current.update_statistics(score)
            self.storage.save(self._current)

    async def create(
        self,
        name: str,
        display_name: str,
        description: str,
        init_mode: InitMode,
        seed_path: Optional[str] = None,
        reference_docs: Optional[List[str]] = None,
        reference_folder: Optional[str] = None,
        auto_bootstrap: bool = True
    ) -> DomainSpecialization:
        """
        Create a new specialization.

        This is the main entry point for creating new domain specializations.
        It can either bootstrap the domain automatically using LLM agents,
        or create a minimal shell for manual configuration.

        Args:
            name: Unique identifier for the specialization
            display_name: Human-readable name
            description: Full description of the research domain
            init_mode: Whether to use seeded or genesis mode
            seed_path: Path to seed codebase (required if init_mode is SEEDED)
            reference_docs: Optional list of reference document paths
            reference_folder: Optional path to folder containing reference documents
            auto_bootstrap: If True, use LLM agents to generate configuration

        Returns:
            The created DomainSpecialization

        Raises:
            ValueError: If name already exists or invalid parameters
        """
        # Validate inputs
        if self.exists(name):
            raise ValueError(f"Specialization '{name}' already exists")

        if init_mode == InitMode.SEEDED and not seed_path:
            raise ValueError("seed_path is required for SEEDED init_mode")

        if seed_path and not Path(seed_path).exists():
            raise ValueError(f"Seed path does not exist: {seed_path}")

        # Create directory structure
        self.storage.create_directory_structure(name)

        # Copy seed codebase if provided
        if seed_path:
            self.storage.copy_seed_codebase(name, seed_path)

        if auto_bootstrap:
            # Use Domain Bootstrapper to generate configuration
            # Import here to avoid circular imports
            from .bootstrapper import DomainBootstrapper

            bootstrapper = DomainBootstrapper(self.storage)
            spec = await bootstrapper.bootstrap(
                name=name,
                display_name=display_name,
                description=description,
                init_mode=init_mode,
                seed_path=seed_path,
                reference_docs=reference_docs,
                reference_folder=reference_folder
            )
        else:
            # Create minimal shell for manual configuration
            spec = self._create_minimal_spec(
                name=name,
                display_name=display_name,
                description=description,
                init_mode=init_mode,
                seed_path=seed_path
            )

        # Save the specialization
        self.storage.save(spec)
        return spec

    def _create_minimal_spec(
        self,
        name: str,
        display_name: str,
        description: str,
        init_mode: InitMode,
        seed_path: Optional[str]
    ) -> DomainSpecialization:
        """Create a minimal specialization shell for manual configuration."""
        from .schema import (
            ArchitectureConfig,
            EvaluationConfig,
            ConstraintConfig,
            PromptTemplates,
            KnowledgeConfig,
            InfrastructureConfig
        )

        spec_path = self.storage._get_spec_path(name)

        return DomainSpecialization.create_new(
            name=name,
            display_name=display_name,
            description=description,
            init_mode=init_mode,
            seed_codebase_path=seed_path,
            architecture=ArchitectureConfig(
                base_class_name="Model",
                artifact_type="code",
                standard_parameters=[],
                interface_signature="def forward(self, x)",
                required_decorators=[],
                file_extension=".py",
                code_style_guidelines=""
            ),
            evaluation=EvaluationConfig(
                baseline_models=[],
                benchmarks=[],
                primary_metric="score",
                scoring_weights={"performance": 1.0},
                result_format="csv",
                loss_column="loss",
                metric_columns=[],
                higher_is_better=True,
                normalization_baseline=0.0
            ),
            constraints=ConstraintConfig(
                complexity_requirement=None,
                strict_constraints=[],
                critical_constraints=[],
                flexible_constraints=[],
                preservation_rules=[]
            ),
            prompts=PromptTemplates(
                planner="[PLACEHOLDER - Configure planner prompt]",
                checker="[PLACEHOLDER - Configure checker prompt]",
                motivation_checker="[PLACEHOLDER - Configure motivation checker prompt]",
                deduplication="[PLACEHOLDER - Configure deduplication prompt]",
                debugger="[PLACEHOLDER - Configure debugger prompt]",
                analyzer="[PLACEHOLDER - Configure analyzer prompt]",
                summarizer="[PLACEHOLDER - Configure summarizer prompt]",
                model_judger="[PLACEHOLDER - Configure model judger prompt]"
            ),
            knowledge=KnowledgeConfig(
                corpus_path=str(spec_path / "knowledge"),
                index_name=f"{name}_knowledge",
                embedding_model="intfloat/e5-base-v2",
                default_search_queries=[]
            ),
            infrastructure=InfrastructureConfig(
                source_file=str(spec_path / "seed" / "model.py"),
                training_script=str(spec_path / "infrastructure" / "train.sh"),
                result_file=str(spec_path / "results" / "loss.csv"),
                test_result_file=str(spec_path / "results" / "benchmark.csv"),
                debug_file=str(spec_path / "debug" / "error.txt"),
                code_pool=str(spec_path / "pool"),
                timeout_seconds=7200,
                max_debug_attempts=3,
                max_retry_attempts=10
            )
        )

    def delete(self, name: str, confirm: bool = False):
        """
        Delete a specialization.

        Args:
            name: Specialization name
            confirm: Must be True to actually delete
        """
        if self._current and self._current.name == name:
            raise SpecializationError(
                "Cannot delete the currently active specialization. "
                "Switch to another specialization first."
            )
        self.storage.delete(name, confirm=confirm)

    def backup(self, name: str) -> str:
        """
        Create a backup of a specialization.

        Args:
            name: Specialization name

        Returns:
            Name of the backup
        """
        return self.storage.backup(name)

    def export(self, name: str, output_path: str):
        """
        Export a specialization as a zip archive.

        Args:
            name: Specialization name
            output_path: Path for the output file
        """
        self.storage.export_specialization(name, output_path)

    def import_specialization(self, zip_path: str, name: Optional[str] = None):
        """
        Import a specialization from a zip archive.

        Args:
            zip_path: Path to the zip file
            name: Optional name override
        """
        self.storage.import_specialization(zip_path, name)

    def get_summary(self, name: str) -> SpecializationSummary:
        """
        Get a summary of a specialization.

        Args:
            name: Specialization name

        Returns:
            SpecializationSummary
        """
        spec = self.load(name)
        return SpecializationSummary.from_specialization(spec)

    def format_list(self) -> str:
        """
        Format the list of specializations for display.

        Returns:
            Formatted string showing all specializations
        """
        summaries = self.list_specializations()

        if not summaries:
            return "No specializations available."

        lines = ["Available Specializations:", ""]

        for i, summary in enumerate(summaries, 1):
            status = "✓" if summary.is_validated else "○"
            current = " (active)" if self._current and self._current.name == summary.name else ""

            lines.append(f"  [{i}] {status} {summary.display_name}{current}")
            lines.append(f"      {summary.experiment_count} experiments | Best score: {summary.best_score:.3f}")
            if summary.description:
                desc = summary.description[:60] + "..." if len(summary.description) > 60 else summary.description
                lines.append(f"      {desc}")
            lines.append("")

        return "\n".join(lines)

    async def validate_specialization(self, name: str) -> List[str]:
        """
        Run full validation on a specialization.

        Args:
            name: Specialization name

        Returns:
            List of validation errors (empty if passed)
        """
        from .bootstrapper import DomainBootstrapper

        spec = self.load(name)
        bootstrapper = DomainBootstrapper(self.storage)

        errors = await bootstrapper.validate(spec)

        # Update validation status
        spec.mark_validated(errors if errors else None)
        self.storage.save(spec)

        return errors


# Global singleton instance
_manager_instance: Optional[SpecializationManager] = None


def get_manager(base_path: str = "./specializations") -> SpecializationManager:
    """
    Get the global SpecializationManager instance.

    Args:
        base_path: Root directory for specializations

    Returns:
        The global SpecializationManager
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = SpecializationManager(base_path)
    return _manager_instance


def reset_manager():
    """Reset the global manager instance (mainly for testing)."""
    global _manager_instance
    _manager_instance = None
