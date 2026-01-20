"""
Specialization Storage

File-based storage for domain specializations.
Each specialization is stored as a directory with config.json and supporting files.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from .schema import (
    DomainSpecialization,
    SpecializationSummary,
    PromptTemplates
)


class SpecializationStorage:
    """
    Manages file-based storage of domain specializations.

    Directory structure:
    specializations/
    ├── linear_attention/
    │   ├── config.json           # Main configuration
    │   ├── prompts/              # Prompt template files
    │   │   ├── planner.txt
    │   │   ├── checker.txt
    │   │   └── ...
    │   ├── knowledge/            # Domain knowledge corpus
    │   │   └── *.json
    │   ├── infrastructure/       # Training scripts, etc.
    │   │   └── train.sh
    │   └── seed/                 # Initial codebase
    │       └── *.py
    └── drug_molecules/
        └── ...
    """

    def __init__(self, base_path: str = "./specializations"):
        """
        Initialize storage with base path for specializations.

        Args:
            base_path: Root directory for all specializations
        """
        self.base_path = Path(base_path)
        self._ensure_base_exists()

    def _ensure_base_exists(self):
        """Ensure the base specializations directory exists."""
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_spec_path(self, name: str) -> Path:
        """Get the path for a specialization directory."""
        return self.base_path / name

    def _get_config_path(self, name: str) -> Path:
        """Get the path to config.json for a specialization."""
        return self._get_spec_path(name) / "config.json"

    def _get_prompts_path(self, name: str) -> Path:
        """Get the prompts directory for a specialization."""
        return self._get_spec_path(name) / "prompts"

    def _get_knowledge_path(self, name: str) -> Path:
        """Get the knowledge directory for a specialization."""
        return self._get_spec_path(name) / "knowledge"

    def _get_infrastructure_path(self, name: str) -> Path:
        """Get the infrastructure directory for a specialization."""
        return self._get_spec_path(name) / "infrastructure"

    def _get_seed_path(self, name: str) -> Path:
        """Get the seed codebase directory for a specialization."""
        return self._get_spec_path(name) / "seed"

    def exists(self, name: str) -> bool:
        """Check if a specialization exists."""
        return self._get_config_path(name).exists()

    def list_all(self) -> List[str]:
        """List all specialization names."""
        if not self.base_path.exists():
            return []
        return [
            d.name for d in self.base_path.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        ]

    def list_summaries(self) -> List[SpecializationSummary]:
        """List summaries of all specializations."""
        summaries = []
        for name in self.list_all():
            try:
                spec = self.load(name)
                summaries.append(SpecializationSummary.from_specialization(spec))
            except Exception as e:
                print(f"Warning: Could not load specialization '{name}': {e}")
        return summaries

    def create_directory_structure(self, name: str):
        """Create the directory structure for a new specialization."""
        spec_path = self._get_spec_path(name)
        spec_path.mkdir(parents=True, exist_ok=True)
        self._get_prompts_path(name).mkdir(exist_ok=True)
        self._get_knowledge_path(name).mkdir(exist_ok=True)
        self._get_infrastructure_path(name).mkdir(exist_ok=True)
        self._get_seed_path(name).mkdir(exist_ok=True)

    def save(self, spec: DomainSpecialization, save_prompts_separately: bool = True):
        """
        Save a specialization to disk.

        Args:
            spec: The specialization to save
            save_prompts_separately: If True, save prompts as separate files
        """
        # Ensure directory structure exists
        self.create_directory_structure(spec.name)

        # Update timestamp
        spec.updated_at = datetime.utcnow().isoformat()

        # Save main config
        config_path = self._get_config_path(spec.name)
        config_data = spec.to_dict()

        if save_prompts_separately:
            # Save prompts as separate files and remove from config
            self._save_prompts_separately(spec.name, spec.prompts)
            config_data["prompts"] = {"_stored_separately": True}

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

    def _save_prompts_separately(self, name: str, prompts: PromptTemplates):
        """Save prompt templates as separate text files."""
        prompts_path = self._get_prompts_path(name)
        prompt_files = {
            "planner.txt": prompts.planner,
            "checker.txt": prompts.checker,
            "motivation_checker.txt": prompts.motivation_checker,
            "deduplication.txt": prompts.deduplication,
            "debugger.txt": prompts.debugger,
            "analyzer.txt": prompts.analyzer,
            "summarizer.txt": prompts.summarizer,
            "model_judger.txt": prompts.model_judger
        }
        for filename, content in prompt_files.items():
            with open(prompts_path / filename, "w", encoding="utf-8") as f:
                f.write(content)

    def _load_prompts_separately(self, name: str) -> PromptTemplates:
        """Load prompt templates from separate text files."""
        prompts_path = self._get_prompts_path(name)
        prompt_files = {
            "planner": "planner.txt",
            "checker": "checker.txt",
            "motivation_checker": "motivation_checker.txt",
            "deduplication": "deduplication.txt",
            "debugger": "debugger.txt",
            "analyzer": "analyzer.txt",
            "summarizer": "summarizer.txt",
            "model_judger": "model_judger.txt"
        }
        prompts = {}
        for key, filename in prompt_files.items():
            filepath = prompts_path / filename
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    prompts[key] = f.read()
            else:
                prompts[key] = ""  # Default empty if file missing
        return PromptTemplates(**prompts)

    def load(self, name: str) -> DomainSpecialization:
        """
        Load a specialization from disk.

        Args:
            name: The specialization name

        Returns:
            The loaded DomainSpecialization

        Raises:
            FileNotFoundError: If specialization doesn't exist
        """
        config_path = self._get_config_path(name)
        if not config_path.exists():
            raise FileNotFoundError(f"Specialization '{name}' not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        # Load prompts from separate files if needed
        if config_data.get("prompts", {}).get("_stored_separately"):
            prompts = self._load_prompts_separately(name)
            config_data["prompts"] = prompts.to_dict()

        return DomainSpecialization.from_dict(config_data)

    def delete(self, name: str, confirm: bool = False):
        """
        Delete a specialization.

        Args:
            name: The specialization name
            confirm: Must be True to actually delete

        Raises:
            ValueError: If confirm is not True
            FileNotFoundError: If specialization doesn't exist
        """
        if not confirm:
            raise ValueError("Must pass confirm=True to delete a specialization")

        spec_path = self._get_spec_path(name)
        if not spec_path.exists():
            raise FileNotFoundError(f"Specialization '{name}' not found")

        shutil.rmtree(spec_path)

    def backup(self, name: str, backup_suffix: Optional[str] = None) -> str:
        """
        Create a backup of a specialization.

        Args:
            name: The specialization name
            backup_suffix: Optional suffix for backup name (default: timestamp)

        Returns:
            The backup name
        """
        if not self.exists(name):
            raise FileNotFoundError(f"Specialization '{name}' not found")

        if backup_suffix is None:
            backup_suffix = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        backup_name = f"{name}_backup_{backup_suffix}"
        src_path = self._get_spec_path(name)
        dst_path = self._get_spec_path(backup_name)

        shutil.copytree(src_path, dst_path)
        return backup_name

    def copy_seed_codebase(self, name: str, source_path: str):
        """
        Copy a seed codebase into the specialization's seed directory.

        Args:
            name: The specialization name
            source_path: Path to the source codebase
        """
        seed_path = self._get_seed_path(name)
        source = Path(source_path)

        if source.is_file():
            # Single file
            shutil.copy2(source, seed_path / source.name)
        elif source.is_dir():
            # Directory - copy contents
            for item in source.iterdir():
                if item.is_file():
                    shutil.copy2(item, seed_path / item.name)
                elif item.is_dir():
                    shutil.copytree(item, seed_path / item.name)

    def add_knowledge_document(self, name: str, doc_name: str, content: Dict):
        """
        Add a knowledge document to the specialization.

        Args:
            name: The specialization name
            doc_name: Name for the document file (without .json)
            content: Document content as dictionary
        """
        knowledge_path = self._get_knowledge_path(name)
        doc_path = knowledge_path / f"{doc_name}.json"
        with open(doc_path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)

    def list_knowledge_documents(self, name: str) -> List[str]:
        """List all knowledge documents for a specialization."""
        knowledge_path = self._get_knowledge_path(name)
        if not knowledge_path.exists():
            return []
        return [f.stem for f in knowledge_path.glob("*.json")]

    def load_knowledge_document(self, name: str, doc_name: str) -> Dict:
        """Load a knowledge document."""
        knowledge_path = self._get_knowledge_path(name)
        doc_path = knowledge_path / f"{doc_name}.json"
        with open(doc_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_infrastructure_file(self, name: str, filename: str, content: str):
        """
        Save an infrastructure file (training script, etc.).

        Args:
            name: The specialization name
            filename: Name for the file
            content: File content
        """
        infra_path = self._get_infrastructure_path(name)
        file_path = infra_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        # Make shell scripts executable
        if filename.endswith(".sh"):
            os.chmod(file_path, 0o755)

    def get_infrastructure_file_path(self, name: str, filename: str) -> Path:
        """Get the full path to an infrastructure file."""
        return self._get_infrastructure_path(name) / filename

    def get_seed_file_path(self, name: str, filename: str) -> Path:
        """Get the full path to a seed file."""
        return self._get_seed_path(name) / filename

    def export_specialization(self, name: str, output_path: str):
        """
        Export a specialization as a zip archive.

        Args:
            name: The specialization name
            output_path: Path for the output zip file
        """
        spec_path = self._get_spec_path(name)
        if not spec_path.exists():
            raise FileNotFoundError(f"Specialization '{name}' not found")

        # Remove .zip extension if present (shutil adds it)
        if output_path.endswith(".zip"):
            output_path = output_path[:-4]

        shutil.make_archive(output_path, "zip", spec_path)

    def import_specialization(self, zip_path: str, name: Optional[str] = None):
        """
        Import a specialization from a zip archive.

        Args:
            zip_path: Path to the zip archive
            name: Optional name override (default: use archive name)
        """
        if name is None:
            name = Path(zip_path).stem
            # Remove _backup_ suffix if present
            if "_backup_" in name:
                name = name.split("_backup_")[0]

        spec_path = self._get_spec_path(name)
        if spec_path.exists():
            raise ValueError(f"Specialization '{name}' already exists")

        shutil.unpack_archive(zip_path, spec_path)
