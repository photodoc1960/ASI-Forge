"""
File Selector Module

Handles selection of which source file to evolve in multi-file specializations.
Supports multiple strategies: round-robin, random, weighted, LLM-selected, and single.
"""

import random
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from .schema import (
    InfrastructureConfig,
    SourceFileConfig,
    FileSelectionStrategy
)


@dataclass
class FileSelectionContext:
    """Context for file selection decisions."""
    iteration: int                      # Current pipeline iteration
    recent_scores: Dict[str, List[float]]  # File path -> recent scores
    last_selected: Optional[str]        # Last selected file path
    experiment_context: str             # Context about current experiment goals


class FileSelector:
    """
    Selects which source file to evolve based on configured strategy.

    Maintains state across iterations to implement strategies like
    round-robin and weighted selection.
    """

    def __init__(self, infrastructure: InfrastructureConfig):
        self.infrastructure = infrastructure
        self.source_files = infrastructure.source_files
        self.strategy = infrastructure.file_selection_strategy

        # State for round-robin
        self._round_robin_index = 0

        # State for weighted selection
        self._file_scores: Dict[str, List[float]] = {
            sf.path: [] for sf in self.source_files
        }
        self._file_improvements: Dict[str, List[float]] = {
            sf.path: [] for sf in self.source_files
        }

    def select_file(self, context: Optional[FileSelectionContext] = None) -> SourceFileConfig:
        """
        Select the next file to evolve based on the configured strategy.

        Args:
            context: Optional context for smarter selection (used by weighted/LLM strategies)

        Returns:
            The selected SourceFileConfig
        """
        if not self.source_files:
            raise ValueError("No source files configured")

        if len(self.source_files) == 1 or self.strategy == FileSelectionStrategy.SINGLE:
            return self.source_files[0]

        if self.strategy == FileSelectionStrategy.ROUND_ROBIN:
            return self._select_round_robin()
        elif self.strategy == FileSelectionStrategy.RANDOM:
            return self._select_random()
        elif self.strategy == FileSelectionStrategy.WEIGHTED:
            return self._select_weighted(context)
        elif self.strategy == FileSelectionStrategy.LLM_SELECTED:
            return self._select_llm(context)
        else:
            # Default to round-robin
            return self._select_round_robin()

    def _select_round_robin(self) -> SourceFileConfig:
        """Cycle through files sequentially."""
        selected = self.source_files[self._round_robin_index]
        self._round_robin_index = (self._round_robin_index + 1) % len(self.source_files)
        return selected

    def _select_random(self) -> SourceFileConfig:
        """Random selection with equal probability."""
        return random.choice(self.source_files)

    def _select_weighted(self, context: Optional[FileSelectionContext]) -> SourceFileConfig:
        """
        Weighted selection based on improvement potential.

        Files that haven't been touched recently or showed good improvement
        get higher weights.
        """
        weights = []
        for sf in self.source_files:
            # Base weight
            weight = 1.0

            # Boost for files with recent improvements
            improvements = self._file_improvements.get(sf.path, [])
            if improvements:
                avg_improvement = sum(improvements[-5:]) / len(improvements[-5:])
                if avg_improvement > 0:
                    weight += avg_improvement * 2

            # Boost for files not recently selected
            if context and context.last_selected != sf.path:
                weight += 0.5

            # Small boost for entry points (they often have cascading effects)
            if sf.is_entry_point:
                weight += 0.3

            weights.append(max(weight, 0.1))  # Ensure positive weight

        # Normalize weights
        total = sum(weights)
        probabilities = [w / total for w in weights]

        # Weighted random selection
        return random.choices(self.source_files, weights=probabilities, k=1)[0]

    def _select_llm(self, context: Optional[FileSelectionContext]) -> SourceFileConfig:
        """
        Let the LLM decide which file to evolve.

        For now, falls back to weighted selection. The actual LLM selection
        happens in the planner prompt by providing file options.
        """
        # Mark all files as candidates for LLM selection
        # The actual selection happens in the planner by including
        # all file descriptions in the prompt
        return self._select_weighted(context)

    def record_result(self, file_path: str, score: float, previous_score: Optional[float] = None):
        """
        Record the result of an evolution for weighted selection.

        Args:
            file_path: Path of the evolved file
            score: Score achieved
            previous_score: Previous score (to calculate improvement)
        """
        if file_path not in self._file_scores:
            self._file_scores[file_path] = []
            self._file_improvements[file_path] = []

        self._file_scores[file_path].append(score)

        if previous_score is not None:
            improvement = score - previous_score
            self._file_improvements[file_path].append(improvement)

        # Keep only last 20 entries to avoid memory growth
        if len(self._file_scores[file_path]) > 20:
            self._file_scores[file_path] = self._file_scores[file_path][-20:]
        if len(self._file_improvements[file_path]) > 20:
            self._file_improvements[file_path] = self._file_improvements[file_path][-20:]

    def get_all_files_context(self) -> str:
        """
        Generate context string describing all source files.

        Used by LLM-selected strategy and multi-file planners.
        """
        lines = ["## Evolvable Source Files\n"]

        for i, sf in enumerate(self.source_files, 1):
            entry_marker = " (ENTRY POINT)" if sf.is_entry_point else ""
            lines.append(f"### File {i}: {sf.path}{entry_marker}")
            lines.append(f"**Description**: {sf.description}")

            if sf.dependencies:
                lines.append(f"**Dependencies**: {', '.join(sf.dependencies)}")

            # Add recent performance if available
            scores = self._file_scores.get(sf.path, [])
            if scores:
                recent = scores[-5:]
                avg = sum(recent) / len(recent)
                lines.append(f"**Recent Avg Score**: {avg:.4f} (last {len(recent)} runs)")

            lines.append("")

        return "\n".join(lines)

    def get_file_content_context(self, include_content: bool = True) -> Dict[str, str]:
        """
        Get content of all source files for context.

        Returns:
            Dict mapping file path to file content
        """
        contents = {}
        for sf in self.source_files:
            try:
                with open(sf.path, 'r', encoding='utf-8') as f:
                    contents[sf.path] = f.read()
            except Exception as e:
                contents[sf.path] = f"# Error reading file: {e}"
        return contents


def create_file_selector(infrastructure: InfrastructureConfig) -> FileSelector:
    """Factory function to create a FileSelector."""
    return FileSelector(infrastructure)
