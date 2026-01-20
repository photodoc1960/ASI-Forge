"""
ASI-Forge Evolution Interface

Interface for the evolution module that generates novel architectures/artifacts.
Coordinates the Planner, Checker, Motivation, and Deduplication agents.

Based on ASI-Arch by Liu et al. (2025) - "AlphaGo Moment for Model Architecture Discovery"
"""

from .prompt import Planner_input, Motivation_checker_input, Deduplication_input, CodeChecker_input
from .model import planner, motivation_checker, deduplication, code_checker
from .model import create_planner_for_specialization
from agents import exceptions, set_tracing_disabled
from typing import List, Tuple
from config import Config
from database.mongo_database import create_admin_client
from utils.agent_logger import log_agent_run


def get_planner_prompt(context: str) -> str:
    """Get planner prompt - uses PromptRenderer if specialization is loaded."""
    if Config.is_specialization_loaded():
        try:
            # Import from the pipeline module - check both 'pipeline' and '__main__'
            # (when run as `python pipeline.py`, it's loaded as __main__)
            import sys
            pipeline_module = sys.modules.get('pipeline') or sys.modules.get('__main__')
            if pipeline_module and hasattr(pipeline_module, 'get_prompt_renderer'):
                renderer = pipeline_module.get_prompt_renderer()
                if renderer:
                    rendered = renderer.render_planner(context)
                    print("[EVOLVE] Using specialization prompt")
                    return rendered
        except Exception as e:
            print(f"[EVOLVE] Warning: Could not load specialization prompt: {e}")
    # Fall back to static prompt
    print("[EVOLVE] Using static/legacy prompt")
    return Planner_input(context)


def get_planner_agent():
    """Get planner agent - creates dynamic agent if specialization is loaded."""
    if Config.is_specialization_loaded():
        try:
            agent = create_planner_for_specialization(Config._specialization)
            print(f"[EVOLVE] Using specialization agent: {agent.name}")
            return agent
        except Exception as e:
            print(f"[EVOLVE] Warning: Could not create specialization agent: {e}")
    print("[EVOLVE] Using static/legacy agent")
    return planner


def _build_file_evolution_context(selected_file: str, original_source: str) -> str:
    """Build context describing the file to be evolved with its actual content."""
    # Get file description from specialization if available
    file_description = "Source file"
    if Config._specialization:
        file_config = Config._specialization.infrastructure.get_file_by_path(selected_file)
        if file_config:
            file_description = file_config.description

    # Extract filename for cleaner display
    import os
    filename = os.path.basename(selected_file)

    context = f"""
## TARGET FILE FOR EVOLUTION

**File**: `{filename}`
**Full Path**: `{selected_file}`
**Description**: {file_description}

### CURRENT SOURCE CODE TO EVOLVE

You MUST evolve the code below. Your improvements should modify THIS specific file.
Do NOT generate code for a different module - focus on improving this exact file.

```python
{original_source}
```

### EVOLUTION INSTRUCTIONS

1. Analyze the current implementation above
2. Identify areas for improvement (performance, robustness, features, clarity)
3. Generate an improved version of THIS file
4. Preserve the existing public API and class names
5. Ensure backwards compatibility with code that imports from this file

"""
    return context


async def evolve(context: str) -> Tuple[str, str]:
    # Get the current file (already selected by pipeline.py)
    selected_file = Config.get_current_source_file()

    for attempt in range(Config.MAX_RETRY_ATTEMPTS):
        with open(selected_file, 'r') as f:
            original_source = f.read()

        # Build enhanced context with actual file content
        if Config.is_multi_file():
            # Include file listing and the actual source code
            file_listing = Config.get_source_files_context()
            file_evolution_context = _build_file_evolution_context(selected_file, original_source)
            enhanced_context = f"{context}\n\n{file_listing}\n\n{file_evolution_context}"
        else:
            # Single file mode - still include the source
            file_evolution_context = _build_file_evolution_context(selected_file, original_source)
            enhanced_context = f"{context}\n\n{file_evolution_context}"

        name, motivation = await gen(enhanced_context)

        if await check_code_correctness(motivation):
            return name, motivation

        with open(selected_file, 'w') as f:
            f.write(original_source)
        print("Try new motivations")
    return "Failed", "evolve error"
    
async def gen(context: str) -> Tuple[str, str]:
    # Save original file content (uses currently selected file)
    current_file = Config.get_current_source_file()
    with open(current_file, 'r') as f:
        original_source = f.read()

    repeated_result = None
    motivation = None

    for attempt in range(Config.MAX_RETRY_ATTEMPTS):
        try:
            # Restore original file
            with open(current_file, 'w') as f:
                f.write(original_source)
            
            # Use different prompt based on whether it's repeated
            plan = None
            if attempt == 0:
                input_prompt = get_planner_prompt(context)
                active_planner = get_planner_agent()
                plan = await log_agent_run("planner", active_planner, input_prompt)
            else:
                repeated_context = await get_repeated_context(repeated_result.repeated_index)
                input = Deduplication_input(context, repeated_context)
                plan = await log_agent_run("deduplication", deduplication, input)
                
            name, motivation = plan.final_output.name, plan.final_output.motivation
            
            repeated_result = await check_repeated_motivation(motivation)
            if repeated_result.is_repeated:
                print(f"Attempt {attempt + 1}: Motivation repeated, index is {repeated_result.repeated_index}")
                if attempt == Config.MAX_RETRY_ATTEMPTS - 1:
                    raise Exception("Maximum retry attempts reached, unable to generate non-repeated motivation")
                continue
            else:
                print(f"Attempt {attempt + 1}: Motivation not repeated, continue execution")
                print(motivation)
                return name, motivation
                
        except exceptions.MaxTurnsExceeded as e:
            print(f"Attempt {attempt + 1} exceeded maximum dialogue turns")
        except Exception as e:
            print(f"Attempt {attempt + 1} error: {e}")
            raise e

async def check_code_correctness(motivation) -> bool:
    """Check code correctness"""
    for attempt in range(Config.MAX_RETRY_ATTEMPTS):
        try:
            code_checker_result = await log_agent_run(
                "code_checker",
                code_checker,
                CodeChecker_input(motivation=motivation),
                max_turns=100
            )
            
            if code_checker_result.final_output.success:
                print("Code checker passed - code looks correct")
                return True
            else:
                error_msg = code_checker_result.final_output.error
                print(f"Code checker found issues: {error_msg}")
                if attempt == Config.MAX_RETRY_ATTEMPTS - 1:
                    print("Reaching checking limits")
                    return False
                continue
                
        except exceptions.MaxTurnsExceeded as e:
            print("Code checker exceeded maximum turns")
            return False
        except Exception as e:
            print(f"Code checker error: {e}")
            return False

async def check_repeated_motivation(motivation: str):
    client = create_admin_client()
    similar_elements = client.search_similar_motivations(motivation)
    context = similar_motivation_context(similar_elements)
    input = Motivation_checker_input(context, motivation)
    repeated_result = await log_agent_run("motivation_checker", motivation_checker, input)
    return repeated_result.final_output


def similar_motivation_context(similar_elements: list) -> str:
    """
    Generate structured context from similar motivation elements
    """
    if not similar_elements:
        return "No previous motivations found for comparison."
    
    context = "### PREVIOUS RESEARCH MOTIVATIONS\n\n"
    
    for i, element in enumerate(similar_elements, 1):
        context += f"**Reference #{i} (Index: {element.index})**\n"
        context += f"```\n{element.motivation}\n```\n\n"
    
    context += f"**Total Previous Motivations**: {len(similar_elements)}\n"
    context += "**Analysis Scope**: Compare target motivation against each reference above\n"
    
    return context

def get_repeated_context(repeated_index: list[int]) -> str:
    """
    Generate structured context from repeated motivation experiments
    """
    client = create_admin_client()
    repeated_elements = [client.get_elements_by_index(index) for index in repeated_index]
    
    if not repeated_elements:
        return "No repeated experimental context available."
    
    structured_context = "### REPEATED EXPERIMENTAL PATTERNS ANALYSIS\n\n"
    
    for i, element in enumerate(repeated_elements, 1):
        structured_context += f"**Experiment #{i} - Index {element.index}**\n"
        structured_context += f"```\n{element.motivation}\n```\n\n"
    
    structured_context += f"**Pattern Analysis Summary:**\n"
    structured_context += f"- **Total Repeated Experiments**: {len(repeated_elements)}\n"
    structured_context += f"- **Innovation Challenge**: Break free from these established pattern spaces\n"
    structured_context += f"- **Differentiation Requirement**: Implement orthogonal approaches that explore fundamentally different design principles\n\n"
    
    structured_context += f"**Key Insight**: The above experiments represent exhausted design spaces. Your task is to identify and implement approaches that operate on completely different mathematical, biological, or physical principles to achieve breakthrough innovation.\n"
    
    return structured_context