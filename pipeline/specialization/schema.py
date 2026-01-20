"""
ASI-Forge Domain Specialization Schema

Defines the data models for domain specializations in the ASI-Forge meta-research framework.
Each specialization represents a complete configuration for autonomous research in a specific domain.

Based on ASI-Arch by Liu et al. (2025) - "AlphaGo Moment for Model Architecture Discovery"
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional, Any
from enum import Enum
import uuid
import json


class InitMode(str, Enum):
    """Initialization mode for a specialization."""
    SEEDED = "seeded"    # User provides existing codebase
    GENESIS = "genesis"  # System generates initial codebase


class ConstraintSeverity(str, Enum):
    """Severity levels for constraints."""
    STRICT = "strict"       # Must-fix, blocks pipeline
    CRITICAL = "critical"   # Should-fix, generates warnings
    FLEXIBLE = "flexible"   # Nice-to-have, optional


@dataclass
class Constraint:
    """A single validation constraint for the domain."""
    name: str
    description: str
    severity: ConstraintSeverity
    validation_prompt: str  # Prompt snippet for checking this constraint
    fix_guidance: str       # Guidance for fixing violations
    examples: Optional[List[Dict[str, str]]] = None  # Before/after examples

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "validation_prompt": self.validation_prompt,
            "fix_guidance": self.fix_guidance,
            "examples": self.examples
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Constraint":
        return cls(
            name=data["name"],
            description=data["description"],
            severity=ConstraintSeverity(data["severity"]),
            validation_prompt=data["validation_prompt"],
            fix_guidance=data["fix_guidance"],
            examples=data.get("examples")
        )


@dataclass
class BaselineModel:
    """A baseline model for comparison in evaluation."""
    name: str
    description: str
    score: float
    metrics: Dict[str, float]  # Metric name -> value
    training_curve: Optional[List[float]] = None  # Loss values over training

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "score": self.score,
            "metrics": self.metrics,
            "training_curve": self.training_curve
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineModel":
        return cls(
            name=data["name"],
            description=data["description"],
            score=data["score"],
            metrics=data["metrics"],
            training_curve=data.get("training_curve")
        )


@dataclass
class ArchitectureConfig:
    """Configuration for the architecture/artifact structure in this domain."""
    base_class_name: str              # e.g., "DeltaNet", "MoleculePredictor"
    artifact_type: str                # e.g., "neural_network", "molecule", "algorithm"
    standard_parameters: List[str]    # e.g., ["d_model", "num_heads", ...]
    interface_signature: str          # e.g., "def forward(self, x, **kwargs)"
    required_decorators: List[str]    # e.g., ["@torch.compile"]
    file_extension: str               # e.g., ".py", ".smiles", ".json"
    code_style_guidelines: str        # Style requirements for generated code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_class_name": self.base_class_name,
            "artifact_type": self.artifact_type,
            "standard_parameters": self.standard_parameters,
            "interface_signature": self.interface_signature,
            "required_decorators": self.required_decorators,
            "file_extension": self.file_extension,
            "code_style_guidelines": self.code_style_guidelines
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchitectureConfig":
        return cls(
            base_class_name=data["base_class_name"],
            artifact_type=data["artifact_type"],
            standard_parameters=data["standard_parameters"],
            interface_signature=data["interface_signature"],
            required_decorators=data["required_decorators"],
            file_extension=data["file_extension"],
            code_style_guidelines=data["code_style_guidelines"]
        )


@dataclass
class EvaluationConfig:
    """Configuration for how experiments are evaluated in this domain."""
    baseline_models: List[BaselineModel]
    benchmarks: List[str]              # e.g., ["ARC Challenge", "BoolQ", ...]
    primary_metric: str                # Main metric to optimize
    scoring_weights: Dict[str, float]  # e.g., {"performance": 0.3, "innovation": 0.25}
    result_format: str                 # "csv", "json", "custom"
    loss_column: Optional[str]         # Column name for loss in results
    metric_columns: List[str]          # Column names for metrics
    higher_is_better: bool             # True if higher metric values are better
    normalization_baseline: float      # Baseline value for score normalization

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_models": [b.to_dict() for b in self.baseline_models],
            "benchmarks": self.benchmarks,
            "primary_metric": self.primary_metric,
            "scoring_weights": self.scoring_weights,
            "result_format": self.result_format,
            "loss_column": self.loss_column,
            "metric_columns": self.metric_columns,
            "higher_is_better": self.higher_is_better,
            "normalization_baseline": self.normalization_baseline
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationConfig":
        return cls(
            baseline_models=[BaselineModel.from_dict(b) for b in data["baseline_models"]],
            benchmarks=data["benchmarks"],
            primary_metric=data["primary_metric"],
            scoring_weights=data["scoring_weights"],
            result_format=data["result_format"],
            loss_column=data.get("loss_column"),
            metric_columns=data["metric_columns"],
            higher_is_better=data["higher_is_better"],
            normalization_baseline=data["normalization_baseline"]
        )


@dataclass
class ConstraintConfig:
    """Configuration for domain-specific constraints."""
    complexity_requirement: Optional[str]     # e.g., "O(n)", "O(n log n)", None
    strict_constraints: List[Constraint]      # Must-fix
    critical_constraints: List[Constraint]    # Should-fix
    flexible_constraints: List[Constraint]    # Nice-to-have
    preservation_rules: List[str]             # What must not change during evolution

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity_requirement": self.complexity_requirement,
            "strict_constraints": [c.to_dict() for c in self.strict_constraints],
            "critical_constraints": [c.to_dict() for c in self.critical_constraints],
            "flexible_constraints": [c.to_dict() for c in self.flexible_constraints],
            "preservation_rules": self.preservation_rules
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstraintConfig":
        return cls(
            complexity_requirement=data.get("complexity_requirement"),
            strict_constraints=[Constraint.from_dict(c) for c in data["strict_constraints"]],
            critical_constraints=[Constraint.from_dict(c) for c in data["critical_constraints"]],
            flexible_constraints=[Constraint.from_dict(c) for c in data["flexible_constraints"]],
            preservation_rules=data["preservation_rules"]
        )

    def get_all_constraints(self) -> List[Constraint]:
        """Get all constraints in priority order (strict first)."""
        return self.strict_constraints + self.critical_constraints + self.flexible_constraints


@dataclass
class PromptTemplates:
    """All prompt templates for this domain."""
    planner: str              # Architecture/artifact design
    checker: str              # Code/artifact validation
    motivation_checker: str   # Duplication detection
    deduplication: str        # Innovation diversification
    debugger: str             # Error fixing
    analyzer: str             # Results analysis
    summarizer: str           # Experience synthesis
    model_judger: str         # Scoring and ranking

    def to_dict(self) -> Dict[str, str]:
        return {
            "planner": self.planner,
            "checker": self.checker,
            "motivation_checker": self.motivation_checker,
            "deduplication": self.deduplication,
            "debugger": self.debugger,
            "analyzer": self.analyzer,
            "summarizer": self.summarizer,
            "model_judger": self.model_judger
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "PromptTemplates":
        return cls(
            planner=data["planner"],
            checker=data["checker"],
            motivation_checker=data["motivation_checker"],
            deduplication=data["deduplication"],
            debugger=data["debugger"],
            analyzer=data["analyzer"],
            summarizer=data["summarizer"],
            model_judger=data["model_judger"]
        )


@dataclass
class KnowledgeConfig:
    """Configuration for the domain knowledge base (RAG)."""
    corpus_path: str                  # Path to domain papers/knowledge
    index_name: str                   # OpenSearch index name
    embedding_model: str              # e.g., "intfloat/e5-base-v2"
    default_search_queries: List[str] # Default queries for knowledge retrieval
    rag_service_url: str = ""         # RAG service URL (empty = disabled)
    document_count: int = 0           # Number of documents indexed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corpus_path": self.corpus_path,
            "index_name": self.index_name,
            "embedding_model": self.embedding_model,
            "default_search_queries": self.default_search_queries,
            "rag_service_url": self.rag_service_url,
            "document_count": self.document_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeConfig":
        return cls(
            corpus_path=data["corpus_path"],
            index_name=data["index_name"],
            embedding_model=data["embedding_model"],
            default_search_queries=data["default_search_queries"],
            rag_service_url=data.get("rag_service_url", ""),
            document_count=data.get("document_count", 0)
        )


@dataclass
class SourceFileConfig:
    """Configuration for a single evolvable source file."""
    path: str                   # Absolute path to the file
    description: str            # What this file does (for LLM context)
    is_entry_point: bool        # Is this the main entry point?
    dependencies: List[str]     # Other source files this depends on (by path)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "description": self.description,
            "is_entry_point": self.is_entry_point,
            "dependencies": self.dependencies
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceFileConfig":
        return cls(
            path=data["path"],
            description=data["description"],
            is_entry_point=data.get("is_entry_point", False),
            dependencies=data.get("dependencies", [])
        )


class FileSelectionStrategy(str, Enum):
    """Strategy for selecting which file to evolve."""
    ROUND_ROBIN = "round_robin"       # Cycle through files sequentially
    RANDOM = "random"                  # Random selection each iteration
    WEIGHTED = "weighted"              # Weight by recent improvement potential
    LLM_SELECTED = "llm_selected"     # Let LLM decide based on context
    SINGLE = "single"                  # Always use the same file (legacy mode)


@dataclass
class InfrastructureConfig:
    """Configuration for training/evaluation infrastructure."""
    # Multi-file support
    source_files: List[SourceFileConfig]  # All evolvable source files
    file_selection_strategy: FileSelectionStrategy  # How to pick which file to evolve

    # Legacy single-file support (computed from source_files)
    source_file: str           # Primary source file (for backwards compatibility)

    # Infrastructure paths
    training_script: str       # Path to training script
    result_file: str           # Path to training results (loss)
    test_result_file: str      # Path to test/benchmark results
    debug_file: str            # Path to debug output
    code_pool: str             # Directory for saving successful artifacts

    # Limits
    timeout_seconds: int       # Training timeout
    max_debug_attempts: int    # Max debugging retries
    max_retry_attempts: int    # Max evolution retries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_files": [sf.to_dict() for sf in self.source_files],
            "file_selection_strategy": self.file_selection_strategy.value,
            "source_file": self.source_file,
            "training_script": self.training_script,
            "result_file": self.result_file,
            "test_result_file": self.test_result_file,
            "debug_file": self.debug_file,
            "code_pool": self.code_pool,
            "timeout_seconds": self.timeout_seconds,
            "max_debug_attempts": self.max_debug_attempts,
            "max_retry_attempts": self.max_retry_attempts
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InfrastructureConfig":
        # Handle legacy single-file configs
        if "source_files" in data:
            source_files = [SourceFileConfig.from_dict(sf) for sf in data["source_files"]]
            strategy = FileSelectionStrategy(data.get("file_selection_strategy", "single"))
        else:
            # Legacy: convert single source_file to source_files list
            source_files = [SourceFileConfig(
                path=data["source_file"],
                description="Primary source file",
                is_entry_point=True,
                dependencies=[]
            )]
            strategy = FileSelectionStrategy.SINGLE

        return cls(
            source_files=source_files,
            file_selection_strategy=strategy,
            source_file=data["source_file"],
            training_script=data["training_script"],
            result_file=data["result_file"],
            test_result_file=data["test_result_file"],
            debug_file=data["debug_file"],
            code_pool=data["code_pool"],
            timeout_seconds=data["timeout_seconds"],
            max_debug_attempts=data["max_debug_attempts"],
            max_retry_attempts=data["max_retry_attempts"]
        )

    def get_all_file_paths(self) -> List[str]:
        """Get all source file paths."""
        return [sf.path for sf in self.source_files]

    def get_entry_point(self) -> Optional[SourceFileConfig]:
        """Get the entry point file config."""
        for sf in self.source_files:
            if sf.is_entry_point:
                return sf
        return self.source_files[0] if self.source_files else None

    def get_file_by_path(self, path: str) -> Optional[SourceFileConfig]:
        """Get file config by path."""
        for sf in self.source_files:
            if sf.path == path:
                return sf
        return None


@dataclass
class DomainSpecialization:
    """
    Complete configuration for a research domain specialization.

    This is the main data structure that defines everything needed to run
    the ASI-Arch pipeline for a specific research domain.
    """
    # Identity
    id: str
    name: str                    # Unique identifier, e.g., "linear_attention"
    display_name: str            # Human-readable, e.g., "Linear Attention Architectures"
    description: str             # Full description of the domain
    created_at: str              # ISO format timestamp
    updated_at: str              # ISO format timestamp

    # Initialization Mode
    init_mode: InitMode
    seed_codebase_path: Optional[str]  # Path to user-provided codebase (if seeded)

    # Sub-configurations
    architecture: ArchitectureConfig
    evaluation: EvaluationConfig
    constraints: ConstraintConfig
    prompts: PromptTemplates
    knowledge: KnowledgeConfig
    infrastructure: InfrastructureConfig

    # Database configuration
    database_collection: str     # MongoDB collection name

    # Statistics (updated during runtime)
    experiment_count: int = 0
    best_score: float = 0.0
    last_experiment_at: Optional[str] = None

    # Status
    is_validated: bool = False   # Has passed full validation?
    validation_errors: List[str] = field(default_factory=list)

    @classmethod
    def create_new(
        cls,
        name: str,
        display_name: str,
        description: str,
        init_mode: InitMode,
        architecture: ArchitectureConfig,
        evaluation: EvaluationConfig,
        constraints: ConstraintConfig,
        prompts: PromptTemplates,
        knowledge: KnowledgeConfig,
        infrastructure: InfrastructureConfig,
        seed_codebase_path: Optional[str] = None
    ) -> "DomainSpecialization":
        """Create a new specialization with generated ID and timestamps."""
        now = datetime.utcnow().isoformat()
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            display_name=display_name,
            description=description,
            created_at=now,
            updated_at=now,
            init_mode=init_mode,
            seed_codebase_path=seed_codebase_path,
            architecture=architecture,
            evaluation=evaluation,
            constraints=constraints,
            prompts=prompts,
            knowledge=knowledge,
            infrastructure=infrastructure,
            database_collection=f"{name}_elements"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "init_mode": self.init_mode.value,
            "seed_codebase_path": self.seed_codebase_path,
            "architecture": self.architecture.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "constraints": self.constraints.to_dict(),
            "prompts": self.prompts.to_dict(),
            "knowledge": self.knowledge.to_dict(),
            "infrastructure": self.infrastructure.to_dict(),
            "database_collection": self.database_collection,
            "experiment_count": self.experiment_count,
            "best_score": self.best_score,
            "last_experiment_at": self.last_experiment_at,
            "is_validated": self.is_validated,
            "validation_errors": self.validation_errors
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainSpecialization":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            display_name=data["display_name"],
            description=data["description"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            init_mode=InitMode(data["init_mode"]),
            seed_codebase_path=data.get("seed_codebase_path"),
            architecture=ArchitectureConfig.from_dict(data["architecture"]),
            evaluation=EvaluationConfig.from_dict(data["evaluation"]),
            constraints=ConstraintConfig.from_dict(data["constraints"]),
            prompts=PromptTemplates.from_dict(data["prompts"]),
            knowledge=KnowledgeConfig.from_dict(data["knowledge"]),
            infrastructure=InfrastructureConfig.from_dict(data["infrastructure"]),
            database_collection=data["database_collection"],
            experiment_count=data.get("experiment_count", 0),
            best_score=data.get("best_score", 0.0),
            last_experiment_at=data.get("last_experiment_at"),
            is_validated=data.get("is_validated", False),
            validation_errors=data.get("validation_errors", [])
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "DomainSpecialization":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def update_statistics(self, score: float):
        """Update statistics after an experiment."""
        self.experiment_count += 1
        if score > self.best_score:
            self.best_score = score
        self.last_experiment_at = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow().isoformat()

    def mark_validated(self, errors: Optional[List[str]] = None):
        """Mark specialization as validated (or not)."""
        if errors:
            self.is_validated = False
            self.validation_errors = errors
        else:
            self.is_validated = True
            self.validation_errors = []
        self.updated_at = datetime.utcnow().isoformat()


@dataclass
class SpecializationSummary:
    """Lightweight summary of a specialization for listing."""
    name: str
    display_name: str
    description: str
    experiment_count: int
    best_score: float
    is_validated: bool
    created_at: str
    last_experiment_at: Optional[str]

    @classmethod
    def from_specialization(cls, spec: DomainSpecialization) -> "SpecializationSummary":
        return cls(
            name=spec.name,
            display_name=spec.display_name,
            description=spec.description[:200] + "..." if len(spec.description) > 200 else spec.description,
            experiment_count=spec.experiment_count,
            best_score=spec.best_score,
            is_validated=spec.is_validated,
            created_at=spec.created_at,
            last_experiment_at=spec.last_experiment_at
        )
