"""
ASI-Forge Pipeline Configuration

Configuration for the autonomous research pipeline. This module supports:
1. Static configuration (legacy mode for linear attention research)
2. Dynamic configuration from domain specializations
3. Multi-file evolution with dynamic file selection

Based on ASI-Arch by Liu et al. (2025) - "AlphaGo Moment for Model Architecture Discovery"
"""

from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from specialization.schema import DomainSpecialization, SourceFileConfig
    from specialization.file_selector import FileSelector


class Config:
    """
    Configuration settings for the experiment.

    This class can operate in two modes:
    1. Static mode (default): Uses hardcoded values below
    2. Dynamic mode: Loads values from an active DomainSpecialization

    To use dynamic mode, call Config.load_specialization(spec).
    """

    # The active specialization (None for static mode)
    _specialization: Optional["DomainSpecialization"] = None

    # Multi-file support
    _file_selector: Optional["FileSelector"] = None
    _current_source_file: Optional[str] = None  # Currently selected file for this iteration

    # =========================================================================
    # Static Configuration (used when no specialization is loaded)
    # =========================================================================

    # Target file for evolution
    _SOURCE_FILE: str = "evolve file"

    # Training script
    _BASH_SCRIPT: str = "your training script"

    # Experiment results
    _RESULT_FILE: str = "./files/analysis/loss.csv"
    _RESULT_FILE_TEST: str = "./files/analysis/benchmark.csv"

    # Debug file
    _DEBUG_FILE: str = "./files/debug/training_error.txt"

    # Code pool directory
    _CODE_POOL: str = "./pool"

    # Maximum number of debug attempts
    _MAX_DEBUG_ATTEMPT: int = 3

    # Maximum number of retry attempts
    _MAX_RETRY_ATTEMPTS: int = 10

    # RAG service URL
    _RAG: str = "your rag url"

    # Database URL
    _DATABASE: str = "http://localhost:8001"

    # =========================================================================
    # Specialization Management
    # =========================================================================

    @classmethod
    def load_specialization(cls, spec: "DomainSpecialization"):
        """Load configuration from a domain specialization."""
        cls._specialization = spec
        # Initialize file selector for multi-file support
        cls._init_file_selector()

    @classmethod
    def clear_specialization(cls):
        """Clear the loaded specialization and revert to static config."""
        cls._specialization = None
        cls._file_selector = None
        cls._current_source_file = None

    @classmethod
    def is_specialization_loaded(cls) -> bool:
        """Check if a specialization is currently loaded."""
        return cls._specialization is not None

    @classmethod
    def get_specialization(cls) -> Optional["DomainSpecialization"]:
        """Get the currently loaded specialization."""
        return cls._specialization

    @classmethod
    def get_specialization_name(cls) -> Optional[str]:
        """Get the name of the currently loaded specialization."""
        return cls._specialization.name if cls._specialization else None

    # =========================================================================
    # Multi-File Support
    # =========================================================================

    @classmethod
    def _init_file_selector(cls):
        """Initialize the file selector for the current specialization."""
        if cls._specialization:
            from specialization.file_selector import FileSelector
            cls._file_selector = FileSelector(cls._specialization.infrastructure)
            # Set initial file to the entry point
            entry = cls._specialization.infrastructure.get_entry_point()
            if entry:
                cls._current_source_file = entry.path
            else:
                cls._current_source_file = cls._specialization.infrastructure.source_file

    @classmethod
    def get_file_selector(cls) -> Optional["FileSelector"]:
        """Get the file selector for multi-file evolution."""
        return cls._file_selector

    @classmethod
    def select_next_file(cls) -> str:
        """
        Select the next file to evolve based on the configured strategy.

        Returns the path of the selected file.
        """
        if cls._file_selector:
            selected = cls._file_selector.select_file()
            cls._current_source_file = selected.path
            return selected.path
        elif cls._specialization:
            return cls._specialization.infrastructure.source_file
        return cls._SOURCE_FILE

    @classmethod
    def get_current_source_file(cls) -> str:
        """
        Get the currently selected source file for this iteration.

        If no file has been explicitly selected, returns the default source file.
        """
        if cls._current_source_file:
            return cls._current_source_file
        if cls._specialization:
            return cls._specialization.infrastructure.source_file
        return cls._SOURCE_FILE

    @classmethod
    def set_current_source_file(cls, path: str):
        """Explicitly set the current source file."""
        cls._current_source_file = path

    @classmethod
    def get_all_source_files(cls) -> List[str]:
        """Get all source file paths configured for evolution."""
        if cls._specialization:
            return cls._specialization.infrastructure.get_all_file_paths()
        return [cls._SOURCE_FILE]

    @classmethod
    def get_source_files_context(cls) -> str:
        """Get context string describing all source files (for prompts)."""
        if cls._file_selector:
            return cls._file_selector.get_all_files_context()
        return f"Source file: {cls.get_current_source_file()}"

    @classmethod
    def record_evolution_result(cls, score: float, previous_score: Optional[float] = None):
        """Record the result of an evolution for the current file."""
        if cls._file_selector and cls._current_source_file:
            cls._file_selector.record_result(
                cls._current_source_file,
                score,
                previous_score
            )

    @classmethod
    def is_multi_file(cls) -> bool:
        """Check if this specialization uses multi-file evolution."""
        if cls._specialization:
            return len(cls._specialization.infrastructure.source_files) > 1
        return False


# =========================================================================
# Module-level accessors (for Config.SOURCE_FILE style access)
# =========================================================================

def _get_source_file() -> str:
    # Use current source file (selected by file selector) if available
    if Config._current_source_file:
        return Config._current_source_file
    if Config._specialization:
        return Config._specialization.infrastructure.source_file
    return Config._SOURCE_FILE

def _get_bash_script() -> str:
    if Config._specialization:
        return Config._specialization.infrastructure.training_script
    return Config._BASH_SCRIPT

def _get_result_file() -> str:
    if Config._specialization:
        return Config._specialization.infrastructure.result_file
    return Config._RESULT_FILE

def _get_result_file_test() -> str:
    if Config._specialization:
        return Config._specialization.infrastructure.test_result_file
    return Config._RESULT_FILE_TEST

def _get_debug_file() -> str:
    if Config._specialization:
        return Config._specialization.infrastructure.debug_file
    return Config._DEBUG_FILE

def _get_code_pool() -> str:
    if Config._specialization:
        return Config._specialization.infrastructure.code_pool
    return Config._CODE_POOL

def _get_max_debug_attempt() -> int:
    if Config._specialization:
        return Config._specialization.infrastructure.max_debug_attempts
    return Config._MAX_DEBUG_ATTEMPT

def _get_max_retry_attempts() -> int:
    if Config._specialization:
        return Config._specialization.infrastructure.max_retry_attempts
    return Config._MAX_RETRY_ATTEMPTS

def _get_rag() -> str:
    if Config._specialization and Config._specialization.knowledge.rag_service_url:
        return Config._specialization.knowledge.rag_service_url
    return Config._RAG

def _get_database() -> str:
    return Config._DATABASE

def _get_database_collection() -> str:
    if Config._specialization:
        return Config._specialization.database_collection
    return "data_elements"

def _get_knowledge_index() -> str:
    if Config._specialization:
        return Config._specialization.knowledge.index_name
    return "knowledge"

def _get_base_class_name() -> str:
    if Config._specialization:
        return Config._specialization.architecture.base_class_name
    return "DeltaNet"

def _get_complexity_requirement() -> Optional[str]:
    if Config._specialization:
        return Config._specialization.constraints.complexity_requirement
    return "O(N log N)"


# Mapping of attribute names to getter functions
_GETTERS = {
    'SOURCE_FILE': _get_source_file,
    'BASH_SCRIPT': _get_bash_script,
    'RESULT_FILE': _get_result_file,
    'RESULT_FILE_TEST': _get_result_file_test,
    'DEBUG_FILE': _get_debug_file,
    'CODE_POOL': _get_code_pool,
    'MAX_DEBUG_ATTEMPT': _get_max_debug_attempt,
    'MAX_RETRY_ATTEMPTS': _get_max_retry_attempts,
    'RAG': _get_rag,
    'DATABASE': _get_database,
    'DATABASE_COLLECTION': _get_database_collection,
    'KNOWLEDGE_INDEX': _get_knowledge_index,
    'BASE_CLASS_NAME': _get_base_class_name,
    'COMPLEXITY_REQUIREMENT': _get_complexity_requirement,
}


class _ConfigMeta(type):
    """Metaclass to allow Config.SOURCE_FILE style access on the class."""
    def __getattr__(cls, name):
        if name in _GETTERS:
            return _GETTERS[name]()
        raise AttributeError(f"type object 'Config' has no attribute '{name}'")


# Apply metaclass to Config
Config = _ConfigMeta('Config', (Config,), dict(Config.__dict__))


# For direct imports: from config import SOURCE_FILE
def __getattr__(name: str):
    """Allow direct attribute access for backward compatibility."""
    if name in _GETTERS:
        return _GETTERS[name]()
    if name == 'Config':
        return Config
    raise AttributeError(f"module 'config' has no attribute '{name}'")
