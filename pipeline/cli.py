"""
ASI-Forge Interactive CLI

Provides an interactive command-line interface for managing domain specializations
and running the autonomous research pipeline.

ASI-Forge extends ASI-Arch to support any research domain through specializations.

Based on ASI-Arch by Liu et al. (2025) - "AlphaGo Moment for Model Architecture Discovery"
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try project root first, then current directory
    for env_path in [Path(__file__).parent.parent / ".env", Path.cwd() / ".env", Path.cwd().parent / ".env"]:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass

from specialization import (
    SpecializationManager,
    DomainSpecialization,
    SpecializationSummary,
    InitMode
)


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    @classmethod
    def disable(cls):
        """Disable colors (for non-terminal output)."""
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.YELLOW = ''
        cls.RED = ''
        cls.ENDC = ''
        cls.BOLD = ''
        cls.DIM = ''


# Disable colors if not in a terminal
if not sys.stdout.isatty():
    Colors.disable()


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the ASI-Forge header."""
    print(f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     {Colors.BOLD}ASI-Forge: Autonomous Scientific Intelligence Forge{Colors.ENDC}{Colors.CYAN}                     ║
║           {Colors.DIM}Universal Meta-Research Framework{Colors.ENDC}{Colors.CYAN}  {Colors.DIM}(built on ASI-Arch){Colors.ENDC}{Colors.CYAN}       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
""")


def print_divider():
    """Print a divider line."""
    print(f"{Colors.DIM}{'─' * 78}{Colors.ENDC}")


def format_specialization_item(
    index: int,
    summary: SpecializationSummary,
    is_current: bool = False
) -> str:
    """Format a specialization for display in the list."""
    status_icon = f"{Colors.GREEN}●{Colors.ENDC}" if summary.is_validated else f"{Colors.YELLOW}○{Colors.ENDC}"
    current_tag = f" {Colors.CYAN}(active){Colors.ENDC}" if is_current else ""

    lines = [
        f"  {Colors.BOLD}[{index}]{Colors.ENDC} {status_icon} {summary.display_name}{current_tag}",
        f"      {Colors.DIM}├─{Colors.ENDC} {summary.experiment_count} experiments | Best: {summary.best_score:.3f}"
    ]

    if summary.description:
        desc = summary.description[:65] + "..." if len(summary.description) > 65 else summary.description
        lines.append(f"      {Colors.DIM}└─{Colors.ENDC} {desc}")

    return "\n".join(lines)


class ASIArchCLI:
    """Interactive CLI for ASI-Arch."""

    def __init__(self, specializations_path: str = None):
        """
        Initialize the CLI.

        Args:
            specializations_path: Path to specializations directory
        """
        if specializations_path is None:
            # Default to specializations directory relative to project root
            specializations_path = str(Path(__file__).parent.parent / "specializations")
        self.manager = SpecializationManager(specializations_path, allow_unvalidated=True)
        self.running = True

    async def run(self):
        """Run the interactive CLI loop."""
        clear_screen()
        print_header()

        while self.running:
            await self.main_menu()

    async def main_menu(self):
        """Display and handle the main menu."""
        summaries = self.manager.list_specializations()
        current_name = self.manager.current_name

        print(f"\n{Colors.BOLD}Available Specializations:{Colors.ENDC}\n")

        if not summaries:
            print(f"  {Colors.DIM}No specializations found.{Colors.ENDC}")
            print(f"  {Colors.DIM}Create a new one to get started.{Colors.ENDC}\n")
        else:
            for i, summary in enumerate(summaries, 1):
                is_current = summary.name == current_name
                print(format_specialization_item(i, summary, is_current))
                print()

        print_divider()
        print(f"\n{Colors.BOLD}Options:{Colors.ENDC}")
        print(f"  {Colors.CYAN}[N]{Colors.ENDC} Create New Specialization")
        print(f"  {Colors.CYAN}[R]{Colors.ENDC} Run Pipeline (requires active specialization)")
        print(f"  {Colors.CYAN}[V]{Colors.ENDC} Validate Specialization")
        print(f"  {Colors.CYAN}[E]{Colors.ENDC} Export/Import Specialization")
        print(f"  {Colors.CYAN}[D]{Colors.ENDC} Delete Specialization")
        print(f"  {Colors.CYAN}[Q]{Colors.ENDC} Quit")
        print()

        choice = input(f"{Colors.BOLD}Select option: {Colors.ENDC}").strip().upper()

        if choice == 'N':
            await self.create_specialization_wizard()
        elif choice == 'R':
            await self.run_pipeline()
        elif choice == 'V':
            await self.validate_specialization()
        elif choice == 'E':
            await self.export_import_menu()
        elif choice == 'D':
            await self.delete_specialization()
        elif choice == 'Q':
            self.running = False
            print(f"\n{Colors.GREEN}Goodbye!{Colors.ENDC}\n")
        elif choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(summaries):
                await self.select_specialization(summaries[index - 1].name)
            else:
                print(f"{Colors.RED}Invalid selection.{Colors.ENDC}")
        else:
            print(f"{Colors.RED}Invalid option.{Colors.ENDC}")

    async def select_specialization(self, name: str):
        """Select and activate a specialization."""
        try:
            spec = await self.manager.activate(name)
            print(f"\n{Colors.GREEN}Activated: {spec.display_name}{Colors.ENDC}")

            if not spec.is_validated:
                print(f"{Colors.YELLOW}Warning: This specialization has not been validated.{Colors.ENDC}")
                print(f"{Colors.YELLOW}Run validation before using in production.{Colors.ENDC}")

        except Exception as e:
            print(f"{Colors.RED}Error activating specialization: {e}{Colors.ENDC}")

    async def create_specialization_wizard(self):
        """Interactive wizard for creating a new specialization."""
        clear_screen()
        print_header()
        print(f"\n{Colors.BOLD}Create New Specialization{Colors.ENDC}\n")
        print_divider()

        # Get basic info
        print(f"\n{Colors.CYAN}Step 1: Basic Information{Colors.ENDC}\n")

        name = input("  Unique name (lowercase, no spaces): ").strip().lower().replace(" ", "_")
        if not name:
            print(f"{Colors.RED}Name is required.{Colors.ENDC}")
            return

        if self.manager.exists(name):
            print(f"{Colors.RED}Specialization '{name}' already exists.{Colors.ENDC}")
            return

        display_name = input("  Display name: ").strip()
        if not display_name:
            display_name = name.replace("_", " ").title()

        print("\n  Enter a description of the research domain.")
        print("  (This helps the system understand the domain for bootstrapping)")
        print("  Press Enter twice when done:\n")

        description_lines = []
        while True:
            line = input("  ")
            if line == "":
                if description_lines:
                    break
            else:
                description_lines.append(line)

        description = "\n".join(description_lines)

        # Get initialization mode
        print(f"\n{Colors.CYAN}Step 2: Initialization Mode{Colors.ENDC}\n")
        print("  How would you like to initialize this specialization?\n")
        print(f"  {Colors.BOLD}[1]{Colors.ENDC} Seeded - I have an existing codebase to evolve")
        print(f"  {Colors.BOLD}[2]{Colors.ENDC} Genesis - Generate initial codebase from scratch")
        print()

        mode_choice = input("  Select mode [1/2]: ").strip()

        if mode_choice == "1":
            init_mode = InitMode.SEEDED
            seed_path = input("\n  Path to seed codebase: ").strip()
            if not seed_path or not Path(seed_path).exists():
                print(f"{Colors.RED}Invalid path.{Colors.ENDC}")
                return
        else:
            init_mode = InitMode.GENESIS
            seed_path = None

        # Reference documents
        print(f"\n{Colors.CYAN}Step 3: Reference Documents (Optional){Colors.ENDC}\n")
        print("  You can provide reference documents to enhance the knowledge base.")
        print("  These can be papers, documentation, code examples, etc.\n")
        print(f"  {Colors.BOLD}[1]{Colors.ENDC} Specify a folder containing documents")
        print(f"  {Colors.BOLD}[2]{Colors.ENDC} Enter individual file paths")
        print(f"  {Colors.BOLD}[3]{Colors.ENDC} Skip (use web search only)")
        print()

        doc_choice = input("  Select option [1/2/3]: ").strip()

        reference_docs = []
        reference_folder = None

        if doc_choice == "1":
            # Folder option
            folder_path = input("\n  Path to folder containing documents: ").strip()
            if folder_path and Path(folder_path).exists() and Path(folder_path).is_dir():
                reference_folder = folder_path
                # Count documents in folder
                supported_extensions = {'.pdf', '.txt', '.md', '.json', '.py', '.tex', '.rst', '.html', '.xml'}
                doc_count = sum(
                    1 for f in Path(folder_path).rglob('*')
                    if f.is_file() and f.suffix.lower() in supported_extensions
                )
                print(f"  {Colors.GREEN}Found {doc_count} documents in folder.{Colors.ENDC}")
                if doc_count == 0:
                    print(f"  {Colors.YELLOW}Warning: No supported documents found.{Colors.ENDC}")
                    print(f"  {Colors.DIM}Supported: .pdf, .txt, .md, .json, .py, .tex, .rst, .html, .xml{Colors.ENDC}")
            else:
                print(f"  {Colors.YELLOW}Invalid folder path, skipping.{Colors.ENDC}")

        elif doc_choice == "2":
            # Individual files
            print("\n  Enter paths to reference documents")
            print("  One per line, empty line to finish:\n")
            while True:
                doc_path = input("  ").strip()
                if not doc_path:
                    break
                if Path(doc_path).exists():
                    reference_docs.append(doc_path)
                    print(f"  {Colors.GREEN}Added: {Path(doc_path).name}{Colors.ENDC}")
                else:
                    print(f"  {Colors.YELLOW}Warning: Path not found, skipping.{Colors.ENDC}")

        # Evolution goals (optional)
        print(f"\n{Colors.CYAN}Step 4: Evolution Goals (Optional){Colors.ENDC}\n")
        print("  You can specify what you want the system to focus on improving.")
        print("  Or leave blank to let the system analyze and propose strategies.\n")
        print(f"  {Colors.BOLD}[1]{Colors.ENDC} Let system analyze and propose evolution strategies (recommended)")
        print(f"  {Colors.BOLD}[2]{Colors.ENDC} Specify my own evolution goals")
        print()

        evolution_choice = input("  Select option [1/2]: ").strip()

        evolution_goals = None
        if evolution_choice == "2":
            print("\n  Describe how you want the code to evolve.")
            print("  Examples:")
            print("    - 'Improve performance and reduce memory usage'")
            print("    - 'Add support for new data formats'")
            print("    - 'Refactor for better modularity'")
            print("  Press Enter twice when done:\n")

            goal_lines = []
            while True:
                line = input("  ")
                if line == "":
                    if goal_lines:
                        break
                else:
                    goal_lines.append(line)

            if goal_lines:
                evolution_goals = "\n".join(goal_lines)
                print(f"\n  {Colors.GREEN}Evolution goals captured.{Colors.ENDC}")
        else:
            print(f"\n  {Colors.GREEN}System will analyze codebase and propose evolution strategies.{Colors.ENDC}")

        # Confirm
        print(f"\n{Colors.CYAN}Summary:{Colors.ENDC}\n")
        print(f"  Name: {name}")
        print(f"  Display Name: {display_name}")
        print(f"  Mode: {init_mode.value}")
        if seed_path:
            print(f"  Seed Path: {seed_path}")
        if reference_folder:
            print(f"  Reference Folder: {reference_folder}")
        elif reference_docs:
            print(f"  Reference Docs: {len(reference_docs)} files")
        if evolution_goals:
            print(f"  Evolution Goals: Specified")
        else:
            print(f"  Evolution Goals: Auto-propose")
        print()

        confirm = input(f"{Colors.BOLD}Create specialization? [y/N]: {Colors.ENDC}").strip().lower()

        if confirm != 'y':
            print(f"{Colors.YELLOW}Cancelled.{Colors.ENDC}")
            return

        # Create the specialization
        print(f"\n{Colors.CYAN}Creating specialization...{Colors.ENDC}")
        print("  This may take several minutes as the system:")
        print("  - Analyzes the domain")
        print("  - Searches for relevant papers")
        print("  - Generates prompts and constraints")
        print("  - Sets up infrastructure")
        if init_mode == InitMode.GENESIS:
            print("  - Generates initial codebase")
        print("  - Runs full validation")
        print()

        try:
            # Include evolution goals in description if specified
            full_description = description
            if evolution_goals:
                full_description = f"{description}\n\n## Evolution Goals\n{evolution_goals}"

            spec = await self.manager.create(
                name=name,
                display_name=display_name,
                description=full_description,
                init_mode=init_mode,
                seed_path=seed_path,
                reference_docs=reference_docs if reference_docs else None,
                reference_folder=reference_folder,
                auto_bootstrap=True
            )

            print(f"\n{Colors.GREEN}Specialization created successfully!{Colors.ENDC}")

            if spec.is_validated:
                print(f"{Colors.GREEN}Validation passed.{Colors.ENDC}")
            else:
                print(f"{Colors.YELLOW}Validation found issues:{Colors.ENDC}")
                for error in spec.validation_errors:
                    print(f"  - {error}")

        except Exception as e:
            print(f"\n{Colors.RED}Error creating specialization: {e}{Colors.ENDC}")

    async def run_pipeline(self):
        """Run the research pipeline with the active specialization."""
        if not self.manager.current:
            print(f"\n{Colors.YELLOW}No specialization is active.{Colors.ENDC}")
            print("Select a specialization first.")
            return

        spec = self.manager.current

        if not spec.is_validated:
            print(f"\n{Colors.YELLOW}Warning: '{spec.display_name}' has not been validated.{Colors.ENDC}")
            confirm = input("Continue anyway? [y/N]: ").strip().lower()
            if confirm != 'y':
                return

        # Ask for experiment count
        print(f"\n{Colors.BOLD}How many experiments to run?{Colors.ENDC}")
        print("  Enter a number (default: 1)")
        print("  Enter 0 or 'infinite' for continuous mode")
        exp_input = input("\nExperiments [1]: ").strip().lower()

        if not exp_input:
            experiment_count = 1
        elif exp_input in ('0', 'infinite', 'inf'):
            experiment_count = 0
            print(f"{Colors.CYAN}Running in infinite mode. Press Ctrl+C to stop.{Colors.ENDC}")
        else:
            try:
                experiment_count = int(exp_input)
            except ValueError:
                experiment_count = 1

        print(f"\n{Colors.CYAN}Starting pipeline with: {spec.display_name}{Colors.ENDC}")
        if experiment_count == 0:
            print("Mode: Infinite (Ctrl+C to stop)")
        else:
            print(f"Experiments: {experiment_count}")
        print()

        try:
            # Import and run the pipeline with experiment count
            import sys
            original_argv = sys.argv.copy()
            # CRITICAL: Pass the selected specialization name to the pipeline
            sys.argv = [
                sys.argv[0],
                f"--spec={spec.name}",
                f"--experiments={experiment_count}",
                "--non-interactive"
            ]

            from pipeline import main as pipeline_main
            await pipeline_main()

            sys.argv = original_argv
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Pipeline stopped.{Colors.ENDC}")
        except Exception as e:
            print(f"\n{Colors.RED}Pipeline error: {e}{Colors.ENDC}")

    async def validate_specialization(self):
        """Run validation on a specialization."""
        summaries = self.manager.list_specializations()

        if not summaries:
            print(f"\n{Colors.YELLOW}No specializations to validate.{Colors.ENDC}")
            return

        print(f"\n{Colors.BOLD}Select specialization to validate:{Colors.ENDC}\n")
        for i, summary in enumerate(summaries, 1):
            status = f"{Colors.GREEN}validated{Colors.ENDC}" if summary.is_validated else f"{Colors.YELLOW}not validated{Colors.ENDC}"
            print(f"  [{i}] {summary.display_name} ({status})")

        print()
        choice = input("Select [number]: ").strip()

        if not choice.isdigit():
            return

        index = int(choice)
        if not (1 <= index <= len(summaries)):
            print(f"{Colors.RED}Invalid selection.{Colors.ENDC}")
            return

        name = summaries[index - 1].name

        print(f"\n{Colors.CYAN}Running validation...{Colors.ENDC}")
        print("This will run a complete end-to-end test experiment.\n")

        try:
            errors = await self.manager.validate_specialization(name)

            if not errors:
                print(f"{Colors.GREEN}Validation passed!{Colors.ENDC}")
            else:
                print(f"{Colors.RED}Validation failed:{Colors.ENDC}")
                for error in errors:
                    print(f"  - {error}")

        except Exception as e:
            print(f"{Colors.RED}Validation error: {e}{Colors.ENDC}")

    async def export_import_menu(self):
        """Handle export/import operations."""
        print(f"\n{Colors.BOLD}Export/Import:{Colors.ENDC}\n")
        print(f"  {Colors.CYAN}[E]{Colors.ENDC} Export specialization")
        print(f"  {Colors.CYAN}[I]{Colors.ENDC} Import specialization")
        print(f"  {Colors.CYAN}[B]{Colors.ENDC} Back")
        print()

        choice = input("Select option: ").strip().upper()

        if choice == 'E':
            await self.export_specialization()
        elif choice == 'I':
            await self.import_specialization()

    async def export_specialization(self):
        """Export a specialization to a zip file."""
        summaries = self.manager.list_specializations()

        if not summaries:
            print(f"\n{Colors.YELLOW}No specializations to export.{Colors.ENDC}")
            return

        print(f"\n{Colors.BOLD}Select specialization to export:{Colors.ENDC}\n")
        for i, summary in enumerate(summaries, 1):
            print(f"  [{i}] {summary.display_name}")

        print()
        choice = input("Select [number]: ").strip()

        if not choice.isdigit():
            return

        index = int(choice)
        if not (1 <= index <= len(summaries)):
            print(f"{Colors.RED}Invalid selection.{Colors.ENDC}")
            return

        name = summaries[index - 1].name
        output_path = input(f"Output path [{name}.zip]: ").strip()
        if not output_path:
            output_path = f"{name}.zip"

        try:
            self.manager.export(name, output_path)
            print(f"{Colors.GREEN}Exported to: {output_path}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}Export error: {e}{Colors.ENDC}")

    async def import_specialization(self):
        """Import a specialization from a zip file."""
        zip_path = input("Path to zip file: ").strip()

        if not zip_path or not Path(zip_path).exists():
            print(f"{Colors.RED}File not found.{Colors.ENDC}")
            return

        name = input("Name for imported specialization (or Enter to auto-detect): ").strip()
        if not name:
            name = None

        try:
            self.manager.import_specialization(zip_path, name)
            print(f"{Colors.GREEN}Import successful!{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}Import error: {e}{Colors.ENDC}")

    async def delete_specialization(self):
        """Delete a specialization."""
        summaries = self.manager.list_specializations()

        if not summaries:
            print(f"\n{Colors.YELLOW}No specializations to delete.{Colors.ENDC}")
            return

        print(f"\n{Colors.BOLD}Select specialization to delete:{Colors.ENDC}\n")
        for i, summary in enumerate(summaries, 1):
            current = " (active)" if self.manager.current_name == summary.name else ""
            print(f"  [{i}] {summary.display_name}{current}")

        print()
        choice = input("Select [number]: ").strip()

        if not choice.isdigit():
            return

        index = int(choice)
        if not (1 <= index <= len(summaries)):
            print(f"{Colors.RED}Invalid selection.{Colors.ENDC}")
            return

        name = summaries[index - 1].name

        if self.manager.current_name == name:
            print(f"{Colors.RED}Cannot delete the active specialization.{Colors.ENDC}")
            print("Switch to another specialization first.")
            return

        print(f"\n{Colors.RED}WARNING: This will permanently delete '{name}'{Colors.ENDC}")
        print("All experiments and data will be lost.")
        confirm = input(f"Type '{name}' to confirm deletion: ").strip()

        if confirm != name:
            print(f"{Colors.YELLOW}Deletion cancelled.{Colors.ENDC}")
            return

        try:
            self.manager.delete(name, confirm=True)
            print(f"{Colors.GREEN}Deleted successfully.{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}Delete error: {e}{Colors.ENDC}")


async def main():
    """Main entry point for the CLI."""
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="ASI-Forge: Universal Meta-Research Framework (built on ASI-Arch)"
    )
    parser.add_argument(
        "--specialization", "-s",
        help="Directly activate a specialization by name"
    )
    parser.add_argument(
        "--run", "-r",
        action="store_true",
        help="Run the pipeline immediately (requires --specialization)"
    )
    parser.add_argument(
        "--path", "-p",
        default=None,
        help="Path to specializations directory (default: <project_root>/specializations)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available specializations and exit"
    )

    args = parser.parse_args()

    cli = ASIArchCLI(args.path)

    if args.list:
        print(cli.manager.format_list())
        return

    if args.specialization:
        await cli.select_specialization(args.specialization)

        if args.run:
            await cli.run_pipeline()
            return

    # Run interactive mode
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
