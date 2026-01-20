"""
ASI-Forge Pipeline Tools

Utility functions and tools for the autonomous research pipeline.
Includes file operations, compression, and critical safeguards.

Based on ASI-Arch by Liu et al. (2025) - "AlphaGo Moment for Model Architecture Discovery"
"""

import subprocess
import gzip
import shutil
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from agents import function_tool
from config import Config


# ============================================================================
# CRITICAL SAFEGUARDS - File Corruption Prevention
# ============================================================================

class FileIntegrityValidator:
    """
    CRITICAL SAFETY SYSTEM: Validates file content before writes.

    This prevents the catastrophic corruption that occurred when LLM-generated
    code was written to files without validation, destroying the original codebase.

    NEVER DISABLE THESE SAFEGUARDS.
    """

    # Expected class/function patterns for each file type
    # If a file is configured, its write MUST contain at least one of these patterns
    REQUIRED_PATTERNS: Dict[str, List[str]] = {}

    # Global write protection - when True, ALL writes are blocked
    _write_protection_enabled: bool = False

    # Files that have been validated this session
    _validated_files: Set[str] = set()

    @classmethod
    def register_file_requirements(cls, file_path: str, required_patterns: List[str]):
        """
        Register required patterns for a file.

        Args:
            file_path: Absolute path to the file
            required_patterns: List of regex patterns that MUST appear in valid content
        """
        cls.REQUIRED_PATTERNS[file_path] = required_patterns

    @classmethod
    def enable_write_protection(cls):
        """Enable global write protection - blocks ALL file writes."""
        cls._write_protection_enabled = True

    @classmethod
    def disable_write_protection(cls):
        """Disable global write protection."""
        cls._write_protection_enabled = False

    @classmethod
    def is_write_protected(cls) -> bool:
        """Check if write protection is enabled."""
        return cls._write_protection_enabled

    @classmethod
    def validate_content(cls, file_path: str, content: str) -> tuple[bool, str]:
        """
        Validate that content is appropriate for the target file.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check global write protection
        if cls._write_protection_enabled:
            return False, "WRITE PROTECTION ENABLED - All file writes are blocked"

        # Check if file has registered requirements
        if file_path in cls.REQUIRED_PATTERNS:
            patterns = cls.REQUIRED_PATTERNS[file_path]
            found_patterns = []
            missing_patterns = []

            for pattern in patterns:
                if re.search(pattern, content):
                    found_patterns.append(pattern)
                else:
                    missing_patterns.append(pattern)

            if missing_patterns and not found_patterns:
                return False, (
                    f"CONTENT VALIDATION FAILED for {file_path}\n"
                    f"Expected at least one of: {patterns}\n"
                    f"Found none of the required patterns.\n"
                    f"This write has been BLOCKED to prevent file corruption."
                )

        return True, ""

    @classmethod
    def extract_file_signature(cls, content: str) -> List[str]:
        """
        Extract identifying signatures from file content.

        Returns class names, function definitions, and key identifiers.
        """
        signatures = []

        # Find class definitions
        classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
        signatures.extend([f"class:{c}" for c in classes])

        # Find top-level function definitions
        functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
        signatures.extend([f"def:{f}" for f in functions])

        # Find module docstring topic
        docstring_match = re.match(r'^["\']""(.+?)"""', content, re.DOTALL)
        if docstring_match:
            first_line = docstring_match.group(1).split('\n')[0].strip()
            signatures.append(f"docstring:{first_line[:50]}")

        return signatures

    @classmethod
    def auto_register_from_file(cls, file_path: str) -> bool:
        """
        Automatically register validation patterns from existing file content.

        Call this for each source file BEFORE any evolution to lock in expected patterns.

        Returns True if patterns were registered.
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            signatures = cls.extract_file_signature(content)
            if signatures:
                # Use class names as required patterns (most reliable)
                class_patterns = [s.replace("class:", r"class\s+") for s in signatures if s.startswith("class:")]
                if class_patterns:
                    cls.REQUIRED_PATTERNS[file_path] = class_patterns
                    cls._validated_files.add(file_path)
                    return True
        except Exception:
            pass

        return False


def initialize_file_safeguards():
    """
    Initialize file integrity safeguards for all configured source files.

    MUST be called at pipeline startup before any evolution.
    """
    if not Config._specialization:
        return

    for sf in Config._specialization.infrastructure.source_files:
        FileIntegrityValidator.auto_register_from_file(sf.path)


# ============================================================================
# Version Control / Backup System
# ============================================================================

class CodeVersionManager:
    """Manages versioned backups of code files before modification."""

    def __init__(self, backup_dir: str = None):
        """
        Initialize the version manager.

        Args:
            backup_dir: Directory to store backups. Defaults to .backups in specialization.
        """
        self._backup_dir = backup_dir
        self._history: Dict[str, List[dict]] = {}  # file_path -> list of versions

    def _get_backup_dir(self) -> Path:
        """Get the backup directory, creating if needed."""
        if self._backup_dir:
            backup_path = Path(self._backup_dir)
        elif Config._specialization:
            # Use specialization's directory
            spec_path = Path(Config._specialization.infrastructure.code_pool).parent
            backup_path = spec_path / ".backups"
        else:
            backup_path = Path("./.backups")

        backup_path.mkdir(parents=True, exist_ok=True)
        return backup_path

    def backup_before_write(self, file_path: str, label: str = None) -> Optional[str]:
        """
        Create a backup of a file before modifying it.

        Args:
            file_path: Path to the file to backup
            label: Optional label for this version (e.g., model name)

        Returns:
            Path to the backup file, or None if file doesn't exist
        """
        source = Path(file_path)
        if not source.exists():
            return None

        backup_dir = self._get_backup_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Create backup filename: original_name.timestamp.label.py.gz
        label_part = f".{label}" if label else ""
        backup_name = f"{source.stem}.{timestamp}{label_part}{source.suffix}.gz"
        backup_path = backup_dir / backup_name

        # Compress and save
        with open(source, 'rb') as f_in:
            with gzip.open(backup_path, 'wb') as f_out:
                f_out.write(f_in.read())

        # Track in history
        if file_path not in self._history:
            self._history[file_path] = []

        self._history[file_path].append({
            'backup_path': str(backup_path),
            'timestamp': timestamp,
            'label': label,
            'original_path': file_path
        })

        return str(backup_path)

    def rollback(self, file_path: str, version_index: int = -1) -> bool:
        """
        Rollback a file to a previous version.

        Args:
            file_path: Path to the file to rollback
            version_index: Which version to restore (-1 = most recent backup)

        Returns:
            True if successful, False otherwise
        """
        if file_path not in self._history or not self._history[file_path]:
            # Try to find backups on disk
            self._scan_for_backups(file_path)

        if file_path not in self._history or not self._history[file_path]:
            return False

        try:
            version = self._history[file_path][version_index]
            backup_path = Path(version['backup_path'])

            if not backup_path.exists():
                return False

            # Decompress and restore
            with gzip.open(backup_path, 'rb') as f_in:
                with open(file_path, 'wb') as f_out:
                    f_out.write(f_in.read())

            return True
        except (IndexError, Exception):
            return False

    def _scan_for_backups(self, file_path: str):
        """Scan backup directory for existing backups of a file."""
        source = Path(file_path)
        backup_dir = self._get_backup_dir()

        if not backup_dir.exists():
            return

        # Find all backups matching this file
        pattern = f"{source.stem}.*{source.suffix}.gz"
        backups = sorted(backup_dir.glob(pattern))

        if backups:
            self._history[file_path] = []
            for backup in backups:
                # Parse timestamp from filename
                parts = backup.stem.replace(source.suffix, '').split('.')
                timestamp = parts[1] if len(parts) > 1 else ''
                label = parts[2] if len(parts) > 2 else None

                self._history[file_path].append({
                    'backup_path': str(backup),
                    'timestamp': timestamp,
                    'label': label,
                    'original_path': file_path
                })

    def list_versions(self, file_path: str) -> List[dict]:
        """List all available versions of a file."""
        if file_path not in self._history:
            self._scan_for_backups(file_path)
        return self._history.get(file_path, [])

    def get_version_content(self, file_path: str, version_index: int = -1) -> Optional[str]:
        """Get the content of a specific version without restoring it."""
        if file_path not in self._history:
            self._scan_for_backups(file_path)

        if file_path not in self._history or not self._history[file_path]:
            return None

        try:
            version = self._history[file_path][version_index]
            backup_path = Path(version['backup_path'])

            with gzip.open(backup_path, 'rt') as f:
                return f.read()
        except (IndexError, Exception):
            return None

    def cleanup_old_backups(self, file_path: str, keep_count: int = 10):
        """Remove old backups, keeping only the most recent ones."""
        if file_path not in self._history:
            self._scan_for_backups(file_path)

        versions = self._history.get(file_path, [])
        if len(versions) <= keep_count:
            return

        # Remove oldest backups
        to_remove = versions[:-keep_count]
        for version in to_remove:
            try:
                Path(version['backup_path']).unlink()
            except Exception:
                pass

        self._history[file_path] = versions[-keep_count:]


# Global version manager instance
_version_manager = CodeVersionManager()


def get_version_manager() -> CodeVersionManager:
    """Get the global version manager instance."""
    return _version_manager


# ============================================================================
# Tool Functions
# ============================================================================

@function_tool
def read_code_file() -> Dict[str, Any]:
    """Read the currently selected code file and return its contents."""
    source_file = Config.get_current_source_file()
    try:
        with open(source_file, 'r') as f:
            content = f.read()
        return {
            'success': True,
            'content': content,
            'file_path': source_file
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'file_path': source_file
        }


@function_tool
def read_specific_file(file_path: str) -> Dict[str, Any]:
    """Read a specific file by path and return its contents."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return {
            'success': True,
            'content': content,
            'file_path': file_path
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'file_path': file_path
        }


@function_tool
def read_csv_file(file_path: str) -> Dict[str, Any]:
    """Read a CSV file and return its contents."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return {
            'success': True,
            'content': content
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@function_tool
def write_code_file(content: str) -> Dict[str, Any]:
    """
    Write content to the currently selected code file.

    SAFEGUARDS:
    - Validates content against registered patterns before writing
    - Creates compressed backup of existing file before writing
    - Blocks writes that don't match expected file patterns

    Use rollback_code_file() to restore previous versions if needed.
    """
    source_file = Config.get_current_source_file()

    try:
        # CRITICAL: Validate content before writing
        is_valid, error_msg = FileIntegrityValidator.validate_content(source_file, content)
        if not is_valid:
            return {
                'success': False,
                'error': f"WRITE BLOCKED: {error_msg}",
                'file_path': source_file,
                'validation_failed': True
            }

        # Create backup before writing
        backup_path = _version_manager.backup_before_write(source_file)

        # Write new content
        with open(source_file, 'w') as f:
            f.write(content)

        return {
            'success': True,
            'message': 'Successfully written',
            'file_path': source_file,
            'backup_created': backup_path is not None,
            'backup_path': backup_path
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'file_path': source_file
        }


@function_tool
def write_specific_file(file_path: str, content: str) -> Dict[str, Any]:
    """
    Write content to a specific file by path.

    SAFEGUARDS:
    - Validates content against registered patterns before writing
    - Creates compressed backup of existing file before writing
    - Blocks writes that don't match expected file patterns
    """
    try:
        # CRITICAL: Validate content before writing
        is_valid, error_msg = FileIntegrityValidator.validate_content(file_path, content)
        if not is_valid:
            return {
                'success': False,
                'error': f"WRITE BLOCKED: {error_msg}",
                'file_path': file_path,
                'validation_failed': True
            }

        # Create backup before writing
        backup_path = _version_manager.backup_before_write(file_path)

        # Write new content
        with open(file_path, 'w') as f:
            f.write(content)

        return {
            'success': True,
            'message': 'Successfully written',
            'file_path': file_path,
            'backup_created': backup_path is not None,
            'backup_path': backup_path
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'file_path': file_path
        }


@function_tool
def rollback_code_file(version_index: int = -1) -> Dict[str, Any]:
    """
    Rollback the currently selected source code file to a previous version.

    Args:
        version_index: Which version to restore. -1 = most recent backup,
                      -2 = second most recent, etc.

    Returns:
        Success status and information about the restored version.
    """
    source_file = Config.get_current_source_file()

    versions = _version_manager.list_versions(source_file)
    if not versions:
        return {
            'success': False,
            'error': 'No backups found for this file',
            'file_path': source_file
        }

    success = _version_manager.rollback(source_file, version_index)

    if success:
        restored = versions[version_index]
        return {
            'success': True,
            'message': f'Restored to version from {restored["timestamp"]}',
            'file_path': source_file,
            'restored_version': restored
        }
    else:
        return {
            'success': False,
            'error': 'Failed to restore backup',
            'file_path': source_file
        }


@function_tool
def list_code_versions() -> Dict[str, Any]:
    """
    List all available backup versions of the currently selected source code file.

    Returns:
        List of available versions with timestamps and labels.
    """
    source_file = Config.get_current_source_file()

    versions = _version_manager.list_versions(source_file)

    return {
        'success': True,
        'file': source_file,
        'version_count': len(versions),
        'versions': [
            {
                'index': i,
                'timestamp': v['timestamp'],
                'label': v['label'],
                'backup_path': v['backup_path']
            }
            for i, v in enumerate(versions)
        ]
    }


@function_tool
def get_all_source_files() -> Dict[str, Any]:
    """
    Get information about all configured source files for multi-file evolution.

    Returns:
        List of source files with their descriptions and current selection status.
    """
    all_files = Config.get_all_source_files()
    current_file = Config.get_current_source_file()

    files_info = []
    if Config._specialization:
        for sf in Config._specialization.infrastructure.source_files:
            files_info.append({
                'path': sf.path,
                'description': sf.description,
                'is_entry_point': sf.is_entry_point,
                'is_selected': sf.path == current_file,
                'dependencies': sf.dependencies
            })
    else:
        files_info.append({
            'path': current_file,
            'description': 'Primary source file',
            'is_entry_point': True,
            'is_selected': True,
            'dependencies': []
        })

    return {
        'success': True,
        'file_count': len(all_files),
        'current_file': current_file,
        'is_multi_file': Config.is_multi_file(),
        'files': files_info
    }


@function_tool
def select_source_file(file_path: str) -> Dict[str, Any]:
    """
    Explicitly select a specific source file as the current evolution target.

    Args:
        file_path: Path to the file to select

    Returns:
        Success status and the selected file info.
    """
    all_files = Config.get_all_source_files()

    if file_path not in all_files:
        return {
            'success': False,
            'error': f'File {file_path} is not in the configured source files',
            'available_files': all_files
        }

    Config.set_current_source_file(file_path)

    return {
        'success': True,
        'message': f'Selected {file_path} as current source file',
        'file_path': file_path
    }


@function_tool
def run_training_script(name: str, script_path: str) -> Dict[str, Any]:
    """Run the training script and return its output.

    The script receives the actual source file path (not the program name)
    as its first argument, since evaluation scripts need to know where
    the evolved code is located.
    """
    # Get the actual source file path - this is where the evolved code lives
    source_file_path = Config.SOURCE_FILE

    try:
        subprocess.run(['bash', script_path, source_file_path],
                      capture_output=True,
                      text=True,
                      check=True)
        return {
            'success': True,
            'error': 'Training script executed successfully'
        }
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'error': e.stderr
        }


@function_tool
def run_plot_script(script_path: str) -> Dict[str, Any]:
    """Run the plotting script."""
    try:
        result = subprocess.run(['python', script_path],
                              capture_output=True,
                              text=True,
                              check=True)
        return {
            'success': True,
            'output': result.stdout,
            'error': result.stderr
        }
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'output': e.stdout,
            'error': e.stderr
        }


def run_rag(query: str) -> Dict[str, Any]:
    """Run RAG and return the results."""
    try:
        import requests

        response = requests.post(
            f'{Config.RAG}/search',
            headers={'Content-Type': 'application/json'},
            json={
                'query': query,
                'k': 3,
                'similarity_threshold': 0.5
            }
        )

        response.raise_for_status()
        results = response.json()

        return {
            'success': True,
            'results': results
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
