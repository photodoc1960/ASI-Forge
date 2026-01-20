"""
ASI-Forge Specialization System

This module provides the domain specialization framework for ASI-Forge,
enabling the system to bootstrap into any research domain.

Built on ASI-Arch's autonomous research capabilities.
Based on ASI-Arch by Liu et al. (2025) - "AlphaGo Moment for Model Architecture Discovery"
"""

from .schema import (
    DomainSpecialization,
    SpecializationSummary,
    InitMode,
    ConstraintSeverity,
    Constraint,
    BaselineModel,
    ArchitectureConfig,
    EvaluationConfig,
    ConstraintConfig,
    PromptTemplates,
    KnowledgeConfig,
    InfrastructureConfig,
    SourceFileConfig,
    FileSelectionStrategy
)

from .storage import SpecializationStorage
from .manager import SpecializationManager, get_manager
from .file_selector import FileSelector, FileSelectionContext, create_file_selector

__all__ = [
    # Main classes
    "DomainSpecialization",
    "SpecializationSummary",
    "SpecializationStorage",
    "SpecializationManager",
    "get_manager",

    # Multi-file support
    "FileSelector",
    "FileSelectionContext",
    "create_file_selector",
    "SourceFileConfig",
    "FileSelectionStrategy",

    # Enums
    "InitMode",
    "ConstraintSeverity",

    # Config classes
    "Constraint",
    "BaselineModel",
    "ArchitectureConfig",
    "EvaluationConfig",
    "ConstraintConfig",
    "PromptTemplates",
    "KnowledgeConfig",
    "InfrastructureConfig"
]
