"""
ASI-Forge Domain Bootstrapper

Orchestrates the multi-agent pipeline for creating new domain specializations.
Uses LLM agents to automatically generate all domain-specific components.

This is the core innovation of ASI-Forge: the ability to bootstrap ASI-Arch into
any research domain through automated configuration generation.

Based on ASI-Arch by Liu et al. (2025) - "AlphaGo Moment for Model Architecture Discovery"
"""

import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from .schema import (
    DomainSpecialization,
    InitMode,
    ArchitectureConfig,
    EvaluationConfig,
    ConstraintConfig,
    PromptTemplates,
    KnowledgeConfig,
    InfrastructureConfig,
    SourceFileConfig,
    FileSelectionStrategy,
    BaselineModel,
    Constraint,
    ConstraintSeverity
)
from .storage import SpecializationStorage
from .agents import (
    DomainUnderstandingAgent,
    KnowledgeAcquisitionAgent,
    ConstraintDefinitionAgent,
    PromptGenerationAgent,
    InfrastructureSetupAgent,
    ValidationAgent
)


class BootstrapProgress:
    """Tracks progress through the bootstrapping process."""

    def __init__(self, callback=None):
        self.stages = [
            "domain_understanding",
            "knowledge_acquisition",
            "constraint_definition",
            "prompt_generation",
            "infrastructure_setup",
            "validation"
        ]
        self.current_stage = 0
        self.callback = callback

    def advance(self, stage_name: str, message: str = ""):
        """Advance to the next stage."""
        self.current_stage += 1
        # Always print progress to terminal
        stage_display = stage_name.replace("_", " ").title()
        print(f"  [{self.current_stage}/{len(self.stages)}] {stage_display}: {message}", flush=True)
        if self.callback:
            self.callback(stage_name, self.current_stage, len(self.stages), message)

    def complete(self):
        """Mark bootstrapping as complete."""
        print(f"  [Complete] Bootstrapping finished successfully!", flush=True)
        if self.callback:
            self.callback("complete", len(self.stages), len(self.stages), "Bootstrapping complete")


class DomainBootstrapper:
    """
    Orchestrates the creation of new domain specializations.

    This class coordinates the multi-agent pipeline that generates
    all components needed for a new research domain:
    1. Domain Understanding
    2. Knowledge Acquisition
    3. Constraint Definition
    4. Prompt Generation
    5. Infrastructure Setup
    6. Validation
    """

    def __init__(
        self,
        storage: SpecializationStorage,
        model: str = "gpt-4o",
        progress_callback=None
    ):
        """
        Initialize the bootstrapper.

        Args:
            storage: Storage instance for saving specialization
            model: LLM model to use for agents
            progress_callback: Optional callback for progress updates
        """
        self.storage = storage
        self.model = model
        self.progress = BootstrapProgress(progress_callback)

        # Initialize agents
        self.domain_agent = DomainUnderstandingAgent(model)
        self.knowledge_agent = KnowledgeAcquisitionAgent(model)
        self.constraint_agent = ConstraintDefinitionAgent(model)
        self.prompt_agent = PromptGenerationAgent(model)
        self.infrastructure_agent = InfrastructureSetupAgent(model)
        self.validation_agent = ValidationAgent(model)

    async def bootstrap(
        self,
        name: str,
        display_name: str,
        description: str,
        init_mode: InitMode,
        seed_path: Optional[str] = None,
        reference_docs: Optional[List[str]] = None,
        reference_folder: Optional[str] = None
    ) -> DomainSpecialization:
        """
        Bootstrap a complete domain specialization.

        This is the main entry point that runs the full multi-agent pipeline
        to generate all components for a new domain.

        Args:
            name: Unique identifier for the specialization
            display_name: Human-readable name
            description: Full description of the research domain
            init_mode: SEEDED or GENESIS
            seed_path: Path to seed codebase (for SEEDED mode)
            reference_docs: Optional reference documents (individual files)
            reference_folder: Optional path to folder containing reference documents

        Returns:
            Complete DomainSpecialization ready for use
        """
        spec_path = self.storage._get_spec_path(name)

        # Stage 1: Domain Understanding
        self.progress.advance("domain_understanding", "Analyzing domain...")
        domain_understanding = await self.domain_agent.analyze(
            description=description,
            seed_path=seed_path
        )
        understanding_dict = domain_understanding.model_dump()

        # Stage 2: Knowledge Acquisition
        self.progress.advance("knowledge_acquisition", "Building knowledge corpus...")
        knowledge_corpus = await self.knowledge_agent.acquire(
            domain_name=display_name,
            domain_description=description,
            domain_understanding=understanding_dict,
            reference_docs=reference_docs,
            reference_folder=reference_folder
        )

        # Save knowledge corpus
        await self.knowledge_agent.save_corpus(
            knowledge_corpus,
            str(spec_path / "knowledge")
        )

        # Stage 3: Constraint Definition
        self.progress.advance("constraint_definition", "Defining constraints...")
        constraint_def = await self.constraint_agent.define(
            domain_name=display_name,
            domain_understanding=understanding_dict,
            seed_patterns=understanding_dict.get("code_patterns")
        )

        # Convert to schema constraints
        strict, critical, flexible = self.constraint_agent.to_schema_constraints(constraint_def)

        # Stage 4: Prompt Generation
        self.progress.advance("prompt_generation", "Generating prompts...")
        prompts = await self.prompt_agent.generate(
            domain_name=display_name,
            domain_description=description,
            domain_understanding=understanding_dict,
            constraints=constraint_def.model_dump(),
            knowledge_corpus=knowledge_corpus.model_dump() if knowledge_corpus else None
        )

        # Stage 5: Infrastructure Setup
        self.progress.advance("infrastructure_setup", "Setting up infrastructure...")
        infrastructure = await self.infrastructure_agent.setup(
            domain_name=display_name,
            domain_description=description,
            domain_understanding=understanding_dict,
            specialization_path=str(spec_path),
            seed_path=seed_path,
            generate_seed=(init_mode == InitMode.GENESIS)
        )

        # Save infrastructure files
        await self.infrastructure_agent.save_infrastructure(
            infrastructure,
            str(spec_path)
        )

        # Build the specialization
        spec = DomainSpecialization.create_new(
            name=name,
            display_name=display_name,
            description=description,
            init_mode=init_mode,
            seed_codebase_path=seed_path,
            architecture=ArchitectureConfig(
                base_class_name=understanding_dict.get("base_class_name", "Model"),
                artifact_type=understanding_dict.get("artifact_type", "code"),
                standard_parameters=understanding_dict.get("standard_parameters", []),
                interface_signature=understanding_dict.get("interface_signature", "def forward(self, x)"),
                required_decorators=[],
                file_extension=".py",
                code_style_guidelines=""
            ),
            evaluation=EvaluationConfig(
                baseline_models=[
                    BaselineModel(**b) for b in infrastructure.baseline_models
                ] if infrastructure.baseline_models else [],
                benchmarks=infrastructure.benchmarks,
                primary_metric="score",
                scoring_weights={"performance": 0.4, "innovation": 0.3, "efficiency": 0.3},
                result_format="csv",
                loss_column="loss",
                metric_columns=infrastructure.benchmarks,
                higher_is_better=True,
                normalization_baseline=0.0
            ),
            constraints=ConstraintConfig(
                complexity_requirement=constraint_def.complexity_requirement,
                strict_constraints=strict,
                critical_constraints=critical,
                flexible_constraints=flexible,
                preservation_rules=constraint_def.preservation_rules
            ),
            prompts=prompts,
            knowledge=KnowledgeConfig(
                corpus_path=str(spec_path / "knowledge"),
                index_name=f"{name}_knowledge",
                embedding_model="intfloat/e5-base-v2",
                default_search_queries=knowledge_corpus.search_queries if knowledge_corpus else [],
                document_count=len(knowledge_corpus.documents) if knowledge_corpus else 0
            ),
            infrastructure=self._build_infrastructure_config(
                infrastructure, spec_path, seed_path, domain_understanding
            )
        )

        # Stage 6: Validation
        self.progress.advance("validation", "Running validation...")
        validation_result = await self.validation_agent.validate(spec, run_e2e=True)

        if validation_result.passed:
            spec.mark_validated()
        else:
            spec.mark_validated(validation_result.errors)

        # Save the specialization
        self.storage.save(spec)

        self.progress.complete()
        return spec

    async def validate(self, spec: DomainSpecialization) -> List[str]:
        """
        Run validation on an existing specialization.

        Args:
            spec: The specialization to validate

        Returns:
            List of error messages (empty if passed)
        """
        result = await self.validation_agent.validate(spec, run_e2e=True)
        return result.errors

    async def regenerate_prompts(
        self,
        spec: DomainSpecialization,
        feedback: Optional[str] = None
    ) -> PromptTemplates:
        """
        Regenerate prompts for a specialization.

        Args:
            spec: The specialization
            feedback: Optional feedback for improvement

        Returns:
            New PromptTemplates
        """
        # Get domain understanding from spec
        understanding = {
            "domain_concepts": [],
            "artifact_type": spec.architecture.artifact_type,
            "base_class_name": spec.architecture.base_class_name,
            "standard_parameters": spec.architecture.standard_parameters,
            "interface_signature": spec.architecture.interface_signature,
            "evaluation_approach": spec.evaluation.primary_metric,
            "complexity_constraints": spec.constraints.complexity_requirement
        }

        # Get constraint info
        constraints = {
            "strict_constraints": [c.to_dict() for c in spec.constraints.strict_constraints],
            "critical_constraints": [c.to_dict() for c in spec.constraints.critical_constraints],
            "preservation_rules": spec.constraints.preservation_rules
        }

        return await self.prompt_agent.generate(
            domain_name=spec.display_name,
            domain_description=spec.description,
            domain_understanding=understanding,
            constraints=constraints
        )

    async def update_knowledge(
        self,
        spec: DomainSpecialization,
        additional_docs: List[str]
    ):
        """
        Update the knowledge corpus with additional documents.

        Args:
            spec: The specialization to update
            additional_docs: Paths to additional documents
        """
        understanding = {
            "artifact_type": spec.architecture.artifact_type,
            "domain_concepts": []
        }

        new_corpus = await self.knowledge_agent.acquire(
            domain_name=spec.display_name,
            domain_description=spec.description,
            domain_understanding=understanding,
            reference_docs=additional_docs,
            max_web_results=0  # Don't search web, just process provided docs
        )

        # Add to existing knowledge
        await self.knowledge_agent.save_corpus(
            new_corpus,
            spec.knowledge.corpus_path
        )

        # Update document count
        spec.knowledge.document_count += len(new_corpus.documents)

    def _build_infrastructure_config(
        self,
        infrastructure,
        spec_path: Path,
        seed_path: Optional[str],
        domain_understanding: Any
    ) -> InfrastructureConfig:
        """
        Build InfrastructureConfig with proper multi-file support.

        Args:
            infrastructure: Output from InfrastructureSetupAgent
            spec_path: Path to specialization directory
            seed_path: Path to seed codebase (for seeded mode)
            domain_understanding: Domain analysis output (Pydantic model or dict)

        Returns:
            Properly configured InfrastructureConfig
        """
        # Get primary source file
        primary_source = infrastructure.config_paths.get(
            "source_file",
            str(spec_path / "seed" / "model.py")
        )

        # Build source_files list from evolution targets
        source_files = []

        # Handle both Pydantic model and dict formats
        if hasattr(domain_understanding, 'evolution_targets'):
            evolution_targets = domain_understanding.evolution_targets
        elif isinstance(domain_understanding, dict):
            evolution_targets = domain_understanding.get("evolution_targets", [])
        else:
            evolution_targets = []

        if evolution_targets:
            # Create SourceFileConfig for each evolution target
            for target in evolution_targets:
                # Handle both Pydantic model and dict
                if hasattr(target, 'file_path'):
                    file_path = target.file_path
                    description = target.current_behavior or target.component_name or "Source file"
                else:
                    file_path = target.get("file_path", "")
                    description = target.get("current_behavior", target.get("component_name", "Source file"))

                if not file_path:
                    continue

                # Make path absolute if relative
                if not file_path.startswith("/"):
                    if seed_path:
                        file_path = str(Path(seed_path) / file_path)
                    else:
                        file_path = str(spec_path / "seed" / file_path)

                source_files.append(SourceFileConfig(
                    path=file_path,
                    description=description,
                    is_entry_point=(file_path == primary_source),
                    dependencies=[]
                ))

        # If no evolution targets, use primary source file
        if not source_files:
            source_files.append(SourceFileConfig(
                path=primary_source,
                description="Primary source file",
                is_entry_point=True,
                dependencies=[]
            ))

        # Determine file selection strategy
        if len(source_files) > 1:
            strategy = FileSelectionStrategy.ROUND_ROBIN
        else:
            strategy = FileSelectionStrategy.SINGLE

        return InfrastructureConfig(
            source_files=source_files,
            file_selection_strategy=strategy,
            source_file=primary_source,
            training_script=infrastructure.config_paths.get(
                "training_script",
                str(spec_path / "infrastructure" / "train.sh")
            ),
            result_file=infrastructure.config_paths.get(
                "result_file",
                str(spec_path / "results" / "loss.csv")
            ),
            test_result_file=infrastructure.config_paths.get(
                "test_result_file",
                str(spec_path / "results" / "benchmark.csv")
            ),
            debug_file=infrastructure.config_paths.get(
                "debug_file",
                str(spec_path / "debug" / "error.txt")
            ),
            code_pool=infrastructure.config_paths.get(
                "code_pool",
                str(spec_path / "pool")
            ),
            timeout_seconds=7200,
            max_debug_attempts=3,
            max_retry_attempts=10
        )


async def bootstrap_domain(
    name: str,
    display_name: str,
    description: str,
    init_mode: str = "genesis",
    seed_path: Optional[str] = None,
    reference_docs: Optional[List[str]] = None,
    reference_folder: Optional[str] = None,
    base_path: str = "./specializations",
    model: str = "gpt-4o",
    progress_callback=None
) -> DomainSpecialization:
    """
    Convenience function to bootstrap a new domain specialization.

    Args:
        name: Unique identifier
        display_name: Human-readable name
        description: Domain description
        init_mode: "seeded" or "genesis"
        seed_path: Path to seed codebase (for seeded mode)
        reference_docs: Optional reference documents (individual files)
        reference_folder: Optional path to folder containing reference documents
        base_path: Path to specializations directory
        model: LLM model to use
        progress_callback: Optional progress callback

    Returns:
        Complete DomainSpecialization
    """
    storage = SpecializationStorage(base_path)
    storage.create_directory_structure(name)

    bootstrapper = DomainBootstrapper(storage, model, progress_callback)

    mode = InitMode.SEEDED if init_mode.lower() == "seeded" else InitMode.GENESIS

    return await bootstrapper.bootstrap(
        name=name,
        display_name=display_name,
        description=description,
        init_mode=mode,
        seed_path=seed_path,
        reference_docs=reference_docs,
        reference_folder=reference_folder
    )
