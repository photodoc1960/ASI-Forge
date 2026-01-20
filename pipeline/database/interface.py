"""
ASI-Forge Database Interface

Interface for the database module that manages experimental data.
Handles sampling, updating, and storing research artifacts and results.

Based on ASI-Arch by Liu et al. (2025) - "AlphaGo Moment for Model Architecture Discovery"
"""

from typing import Tuple
from pathlib import Path

from config import Config
from .element import DataElement
from .mongo_database import create_client


# Create database instance lazily to avoid connection at import time
_db = None

def get_db():
    global _db
    if _db is None:
        _db = create_client()
    return _db


def _seed_database_if_empty():
    """
    Check if database is empty and seed it with initial code if so.

    For seeded specializations, reads the source file and creates an initial
    entry in the database. This allows the pipeline to start evolving.

    For multi-file specializations, seeds with the entry point file's content.
    """
    db = get_db()

    # Check if we have any candidates
    try:
        candidates = db.candidate_sample_from_range(1, 1, 1)
        if candidates:
            return  # Database has data, no seeding needed
    except Exception:
        pass  # Empty database, need to seed

    # Get source file path - use the entry point for multi-file specializations
    source_file = Config.get_current_source_file()

    source_path = Path(source_file)

    if not source_path.exists():
        print(f"[DATABASE] Warning: Source file not found: {source_file}")
        print(f"[DATABASE] Cannot seed database without initial code.")
        return

    # Read the initial code
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            initial_code = f.read()
    except Exception as e:
        print(f"[DATABASE] Error reading source file: {e}")
        return

    # Create initial element
    from datetime import datetime
    initial_element = DataElement(
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        name="seed_v0",
        result={"train": "", "test": ""},
        program=initial_code,
        motivation="Initial seed codebase for evolution",
        analysis="Base implementation to be improved through autonomous evolution",
        cognition="",
        log="",
        parent=None,  # No parent, this is the root
        index=None,   # Let the database assign index
        summary="",
        motivation_embedding=None,
        score=0.0
    )

    # Add to database
    try:
        db.add_element_from_dict(initial_element.to_dict())
        print(f"[DATABASE] Seeded database with initial code from: {source_file}")
    except Exception as e:
        print(f"[DATABASE] Error seeding database: {e}")


async def program_sample() -> Tuple[str, int]:
    """
    Sample program using UCT algorithm and generate context.

    Process:
    1. Check if database is empty and seed if needed
    2. Use UCT algorithm to select a node as parent node
    3. Get top 2 best results
    4. Get 2-50 random results
    5. Concatenate results into context
    6. The modified file is the program of the node selected by UCT

    Returns:
        Tuple containing context string and parent index
    """
    db = get_db()

    # Ensure database has at least one entry
    _seed_database_if_empty()

    context = ""

    # Get parent element - try candidate set first, fall back to main database
    candidates = db.candidate_sample_from_range(1, 10, 1)
    if not candidates:
        # Candidate set empty, try main database
        candidates = db.sample_from_range(1, 10, 1)
    if not candidates:
        raise RuntimeError("Database is empty and could not be seeded. Check source file path.")
    parent_element = candidates[0]

    # Get reference elements - try candidate set first, fall back to main database
    ref_elements = db.candidate_sample_from_range(11, 50, 4)
    if not ref_elements:
        ref_elements = db.sample_from_range(2, 50, 4)  # Start from 2 to avoid parent

    # Build context from parent and reference elements
    context += await parent_element.get_context()
    for element in ref_elements:
        context += await element.get_context()

    parent = parent_element.index

    # Write the program of the UCT selected node to the currently selected file
    # Uses Config.get_current_source_file() for multi-file support
    current_file = Config.get_current_source_file()
    with open(current_file, 'w', encoding='utf-8') as f:
        f.write(parent_element.program)
        print(f"[DATABASE] Implement Changes selected node (index: {parent}) to: {current_file}")

    return context, parent


def update(result: DataElement) -> bool:
    """
    Update database with new experimental result.

    Args:
        result: DataElement containing experimental results

    Returns:
        True if update successful
    """
    get_db().add_element_from_dict(result.to_dict())
    return True