"""
ASI-Forge Research Pipeline

Main entry point for the autonomous research pipeline.
Built on ASI-Arch, supports both legacy mode (original linear attention research)
and specialization-aware mode for any research domain.

ASI-Forge extends ASI-Arch to enable autonomous scientific research in any domain.

Based on ASI-Arch by Liu et al. (2025) - "AlphaGo Moment for Model Architecture Discovery"
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from agents import set_default_openai_api, set_default_openai_client, set_tracing_disabled
from openai import AsyncOpenAI

from analyse import analyse
from database import program_sample, update
from eval import evaluation
from evolve import evolve
from config import Config
from utils.agent_logger import end_pipeline, log_error, log_info, log_step, log_warning, start_pipeline

# Initialize OpenAI client
client = AsyncOpenAI()
set_default_openai_client(client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# Global renderer for specialization-aware mode
_prompt_renderer = None


def get_prompt_renderer():
    """Get the current prompt renderer (or None in legacy mode)."""
    return _prompt_renderer


async def initialize_specialization(
    specialization_name: Optional[str] = None,
    interactive: bool = True
) -> bool:
    """
    Initialize the pipeline with a domain specialization.

    Args:
        specialization_name: Name of specialization to load (None for interactive)
        interactive: Whether to show interactive menu if no name provided

    Returns:
        True if specialization loaded successfully, False otherwise
    """
    global _prompt_renderer

    try:
        from specialization import SpecializationManager
        from specialization.templates import PromptRenderer

        manager = SpecializationManager(allow_unvalidated=True)

        # List available specializations
        available = manager.list_names()

        if not available:
            log_warning("No specializations found. Running in legacy mode.")
            return False

        # Select specialization
        if specialization_name:
            if specialization_name not in available:
                log_error(f"Specialization '{specialization_name}' not found.")
                log_info(f"Available: {', '.join(available)}")
                return False
            name = specialization_name
        elif interactive and sys.stdin.isatty():
            # Interactive selection
            print("\n" + "=" * 60)
            print("ASI-Forge: Universal Meta-Research Framework")
            print("=" * 60)
            print("\nAvailable Specializations:\n")

            for i, spec_name in enumerate(available, 1):
                summary = manager.get_summary(spec_name)
                status = "[validated]" if summary.is_validated else "[pending]"
                print(f"  [{i}] {summary.display_name} {status}")
                print(f"      {summary.experiment_count} experiments | Best: {summary.best_score:.3f}")
                print()

            print("  [0] Run in legacy mode (no specialization)")
            print()

            try:
                choice = input("Select specialization [1]: ").strip()
                if not choice:
                    choice = "1"

                if choice == "0":
                    log_info("Running in legacy mode.")
                    return False

                idx = int(choice) - 1
                if 0 <= idx < len(available):
                    name = available[idx]
                else:
                    log_error("Invalid selection.")
                    return False
            except (ValueError, EOFError):
                log_error("Invalid input.")
                return False
        else:
            # Non-interactive: use first available
            name = available[0]
            log_info(f"Auto-selecting specialization: {name}")

        # Load the specialization
        spec = await manager.activate(name)
        Config.load_specialization(spec)

        # CRITICAL: Initialize file integrity safeguards BEFORE any file operations
        # This prevents the catastrophic file corruption that occurred when
        # LLM-generated code was written to files without validation
        from tools.tools import initialize_file_safeguards, FileIntegrityValidator
        initialize_file_safeguards()
        log_info(f"File integrity safeguards initialized for {len(FileIntegrityValidator.REQUIRED_PATTERNS)} files")

        # Switch MongoDB collection for this specialization
        try:
            import requests
            collection_name = spec.database_collection
            response = requests.post(
                f"{Config._DATABASE}/switch-collection",
                json={"collection_name": collection_name},
                timeout=10
            )
            if response.status_code == 200:
                log_info(f"Switched to database collection: {collection_name}")
            else:
                log_warning(f"Failed to switch collection: {response.text}")
        except Exception as e:
            log_warning(f"Could not switch MongoDB collection: {e}")
            log_warning("Make sure the MongoDB API is updated and restarted")

        # Initialize prompt renderer
        _prompt_renderer = PromptRenderer(spec)

        log_info(f"Loaded specialization: {spec.display_name}")
        log_info(f"Domain: {spec.architecture.artifact_type}")
        log_info(f"Experiments: {spec.experiment_count}")

        if not spec.is_validated:
            log_warning("This specialization has not been fully validated.")

        return True

    except ImportError as e:
        log_warning(f"Specialization system not available: {e}")
        log_info("Running in legacy mode.")
        return False
    except Exception as e:
        log_error(f"Failed to initialize specialization: {e}")
        return False


async def run_single_experiment() -> bool:
    """Run single experiment loop - using pipeline categorized logging."""
    # Start a new pipeline process
    spec_name = Config.get_specialization_name() or "legacy"
    pipeline_id = start_pipeline(f"experiment_{spec_name}")

    try:
        # Step 0: Select file for multi-file evolution (before sampling)
        if Config.is_multi_file():
            selected_file = Config.select_next_file()
            log_info(f"Multi-file mode: Selected '{selected_file}' for this experiment")
        else:
            log_info(f"Single-file mode: Evolving '{Config.get_current_source_file()}'")

        # Step 1: Program sampling
        log_step("Program Sampling", "Start sampling program from database")
        context, parent = await program_sample()
        log_info(f"Program sampling completed, context length: {len(str(context))}")

        # Step 2: Evolution
        log_step("Program Evolution", "Start evolving new program")
        name, motivation = await evolve(context)
        if name == "Failed":
            log_error("Program evolution failed")
            end_pipeline(False, "Evolution failed")
            return False
        log_info(f"Program evolution successful, generated program: {name}")
        log_info(f"Evolution motivation: {motivation}")

        # Step 3: Evaluation
        log_step("Program Evaluation", f"Start evaluating program {name}")
        success = await evaluation(name, motivation)
        if not success:
            log_error(f"Program {name} evaluation failed")
            end_pipeline(False, "Evaluation failed")
            return False
        log_info(f"Program {name} evaluation successful")

        # Step 4: Analysis
        log_step("Result Analysis", f"Start analyzing program {name} results")
        result = await analyse(name, motivation, parent=parent)
        log_info(f"Analysis completed, result: {result}")

        # Step 5: Update database
        log_step("Database Update", "Update results to database")
        update(result)
        log_info("Database update completed")

        # Update specialization statistics if loaded
        if Config.is_specialization_loaded():
            try:
                from specialization import SpecializationManager
                manager = SpecializationManager()
                # Could extract score from result and update
                # manager.update_current_statistics(score)
            except Exception:
                pass

        # Successfully complete pipeline
        log_info("Experiment pipeline completed successfully")
        end_pipeline(True, f"Experiment completed successfully, program: {name}, result: {result}")
        return True

    except KeyboardInterrupt:
        log_warning("User interrupted experiment")
        end_pipeline(False, "User interrupted experiment")
        return False
    except Exception as e:
        log_error(f"Experiment pipeline unexpected error: {str(e)}")
        end_pipeline(False, f"Unexpected error: {str(e)}")
        return False


async def main():
    """Main function - continuous experiment execution."""
    set_tracing_disabled(True)

    log_info("Starting ASI-Arch pipeline...")

    # Parse command line arguments
    specialization_name = None
    interactive = True

    for arg in sys.argv[1:]:
        if arg.startswith("--spec="):
            specialization_name = arg.split("=", 1)[1]
            interactive = False
        elif arg == "--legacy":
            log_info("Running in legacy mode (--legacy flag)")
            specialization_name = None
            interactive = False
        elif arg == "--non-interactive":
            interactive = False

    # Initialize specialization
    spec_loaded = await initialize_specialization(
        specialization_name=specialization_name,
        interactive=interactive
    )

    if spec_loaded:
        log_info(f"Running with specialization: {Config.get_specialization_name()}")
    else:
        log_info("Running in legacy mode (no specialization)")

    # Parse experiment count
    max_experiments = 1  # Default to 1 experiment
    for arg in sys.argv[1:]:
        if arg.startswith("--experiments="):
            val = arg.split("=", 1)[1]
            if val.lower() in ("infinite", "inf", "0"):
                max_experiments = 0  # 0 means infinite
            else:
                max_experiments = int(val)
        elif arg == "--infinite":
            max_experiments = 0

    # Run experiments
    if max_experiments == 0:
        log_info("Starting continuous experiment pipeline (infinite mode, Ctrl+C to stop)...")
    else:
        log_info(f"Starting experiment pipeline ({max_experiments} experiment{'s' if max_experiments > 1 else ''})...")

    experiment_count = 0
    while max_experiments == 0 or experiment_count < max_experiments:
        try:
            experiment_count += 1
            log_info(f"Starting experiment {experiment_count}" + (f"/{max_experiments}" if max_experiments > 0 else ""))

            success = await run_single_experiment()
            if success:
                if max_experiments == 0 or experiment_count < max_experiments:
                    log_info(f"Experiment {experiment_count} completed successfully, starting next experiment...")
                else:
                    log_info(f"Experiment {experiment_count} completed successfully.")
            else:
                log_warning(f"Experiment {experiment_count} failed, retrying in 60 seconds...")
                await asyncio.sleep(60)

        except KeyboardInterrupt:
            log_warning("Experiment pipeline interrupted by user")
            break
        except Exception as e:
            log_error(f"Main loop unexpected error: {e}")
            log_info("Retrying in 60 seconds...")
            await asyncio.sleep(60)

    if max_experiments > 0 and experiment_count >= max_experiments:
        log_info(f"Completed {experiment_count} experiment{'s' if experiment_count > 1 else ''}. Pipeline finished.")


if __name__ == "__main__":
    asyncio.run(main())
