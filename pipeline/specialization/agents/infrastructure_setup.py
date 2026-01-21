"""
Infrastructure Setup Agent

Creates evaluation infrastructure for any codebase.
Does NOT assume neural networks or training - works with any type of code.

The infrastructure evaluates evolved code based on:
1. Whether it runs without errors
2. Whether tests pass (if tests exist)
3. Domain-specific quality metrics
4. Performance benchmarks (if applicable)
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from pydantic import BaseModel


class InfrastructureSetup(BaseModel):
    """Output from the Infrastructure Setup Agent."""
    training_script: str  # Called "training" for backwards compat, but really "evaluation"
    result_parser: str
    evaluation_harness: Optional[str]
    seed_artifact: Optional[str]  # For genesis mode
    config_paths: Dict[str, str]
    benchmarks: List[str]
    baseline_models: List[Dict[str, Any]]


EVALUATION_SCRIPT_PROMPT = """# Evaluation Script Generation Task

You need to generate an evaluation script for evolving a codebase.

## Codebase Overview
**Name:** {domain_name}
**Purpose:** {codebase_purpose}

## CRITICAL: Codebase Location
**Seed Codebase Path:** {seed_codebase_path}

The evolved code file will be passed as $1. This file EXISTS WITHIN or is related to the seed codebase at the path above. Your script MUST:
- Set a variable like `CODEBASE_DIR="{seed_codebase_path}"` to reference the actual codebase
- Run tests and validation FROM within the codebase directory
- NOT assume the evolved code is a standalone file - it's part of the larger codebase

## Codebase Structure
{codebase_structure}

## Evolution Targets
{evolution_targets}

## Evaluation Strategies
{evaluation_strategies}

## Your Task
Generate a script that evaluates evolved versions of files in this codebase.

The script MUST:
1. Accept the evolved code path as an argument ($1)
2. Set CODEBASE_DIR="{seed_codebase_path}" to reference the actual project
3. Run validation FROM the codebase directory (cd to it!)
4. Run the codebase's own tests if they exist (look in tests/, test/, etc.)
5. Measure success based on the evolution strategies' criteria
6. Save results to CSV files:
   - Main results to: {result_file}
   - Additional metrics to: {test_result_file}
7. Save any errors to: {debug_file}
8. Exit 0 on success, non-zero on failure

## What to Evaluate
Based on the evolution targets and strategies, evaluate:
{evaluation_focus}

## Important
- The seed codebase is at: {seed_codebase_path}
- Do NOT invent paths - use the actual codebase location
- Check if the codebase has pytest/unittest tests and run them
- Use the codebase's own test infrastructure if it exists
- Create practical metrics that measure real improvement

## Output
Provide a bash script (with Python calls if needed) that can be run as:
`bash evaluate.sh <evolved_code_path>`

The script should be practical and actually work with this codebase."""


SEED_GENERATION_PROMPT = """# Seed Code Generation Task

Generate initial code for a new project that will be automatically evolved.

## Project Description
{domain_name}: {domain_description}

## Domain Expertise
**Primary Domain:** {primary_domain}
**Key Concepts:** {key_concepts}

## Evolution Goals
The code will be evolved with these strategies:
{evolution_strategies}

## Requirements
Generate code that:
1. Implements the core functionality described
2. Follows best practices for the domain
3. Is well-structured for evolution (clear modules, functions)
4. Includes comments explaining key decisions
5. Has clear entry points and interfaces

## Quality Standards
- Production-quality code
- Well-documented
- Testable and extensible
- Follows domain conventions

Generate complete, working code that serves as a starting point for evolution."""


class InfrastructureSetupAgent:
    """
    Agent that creates evaluation infrastructure for any codebase.
    Works with any type of code, not just ML models.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def _extract_code_from_response(self, content: str, default_type: str = "python") -> str:
        """Extract code from LLM response, handling markdown code blocks."""
        content = content.strip()

        # Handle markdown code blocks
        if "```" in content:
            parts = content.split("```")
            if len(parts) >= 3:
                code_block = parts[1]
                lines = code_block.split("\n")
                if lines[0].strip().lower() in ["python", "py", "bash", "sh", "shell"]:
                    code_block = "\n".join(lines[1:])
                elif lines[0].strip() == "":
                    code_block = "\n".join(lines[1:])
                return code_block.strip()

        # Check for raw code
        first_lines = content.split("\n")[:5]
        first_content = "\n".join(first_lines).lower()

        code_indicators = [
            "import ", "from ", "#!/", "def ", "class ",
            "# ", '"""', "if __name__"
        ]

        for indicator in code_indicators:
            if indicator in first_content:
                lines = content.split("\n")
                code_lines = []
                in_docstring = False
                for line in lines:
                    stripped = line.strip()
                    if '"""' in line or "'''" in line:
                        count = line.count('"""') + line.count("'''")
                        if count == 1:
                            in_docstring = not in_docstring
                        code_lines.append(line)
                        continue
                    if in_docstring:
                        code_lines.append(line)
                        continue
                    if stripped and not stripped.startswith("#"):
                        explanation_starters = [
                            "this script", "this code", "note:", "note that",
                            "make sure", "to run", "usage:", "run the",
                            "you can", "you should", "remember to"
                        ]
                        if any(stripped.lower().startswith(s) for s in explanation_starters):
                            break
                    code_lines.append(line)
                return "\n".join(code_lines).strip()

        return content

    async def setup(
        self,
        domain_name: str,
        domain_description: str,
        domain_understanding: Dict[str, Any],
        specialization_path: str,
        seed_path: Optional[str] = None,
        generate_seed: bool = False
    ) -> InfrastructureSetup:
        """
        Set up evaluation infrastructure for a domain.

        Args:
            domain_name: Name of the domain
            domain_description: Full description
            domain_understanding: Output from DomainUnderstandingAgent (new format)
            specialization_path: Path to specialization directory
            seed_path: Path to existing seed codebase (for seeded mode)
            generate_seed: Whether to generate seed artifact (genesis mode)

        Returns:
            InfrastructureSetup with all components
        """
        spec_path = Path(specialization_path)

        # Extract new format understanding
        codebase_purpose = domain_understanding.get("codebase_purpose", domain_description)
        structure = domain_understanding.get("structure", {})
        evolution_targets = domain_understanding.get("evolution_targets", [])
        strategies = domain_understanding.get("proposed_strategies", [])
        expertise = domain_understanding.get("domain_expertise", {})

        # Generate evaluation script (replaces "training" script)
        eval_script = await self._generate_evaluation_script(
            domain_name,
            codebase_purpose,
            structure,
            evolution_targets,
            strategies,
            spec_path,
            seed_path
        )

        # Generate result parser
        result_parser = self._generate_result_parser(domain_name, strategies)

        # Generate seed artifact if in genesis mode
        seed_artifact = None
        if generate_seed:
            seed_artifact = await self._generate_seed_artifact(
                domain_name,
                domain_description,
                expertise,
                strategies
            )

        # Generate benchmarks and baselines from strategies
        benchmarks = self._extract_benchmarks(strategies)
        baselines = self._extract_baselines(strategies)

        # Configure paths
        source_file = self._find_source_file(spec_path, seed_path, domain_understanding)

        config_paths = {
            "source_file": source_file,
            "training_script": str(spec_path / "infrastructure" / "evaluate.sh"),
            "result_file": str(spec_path / "results" / "evaluation.csv"),
            "test_result_file": str(spec_path / "results" / "metrics.csv"),
            "debug_file": str(spec_path / "debug" / "error.txt"),
            "code_pool": str(spec_path / "pool")
        }

        return InfrastructureSetup(
            training_script=eval_script,
            result_parser=result_parser,
            evaluation_harness=None,
            seed_artifact=seed_artifact,
            config_paths=config_paths,
            benchmarks=benchmarks,
            baseline_models=baselines
        )

    def _find_source_file(
        self,
        spec_path: Path,
        seed_path: Optional[str],
        domain_understanding: Dict[str, Any]
    ) -> str:
        """Find the main source file to evolve based on evolution targets."""
        seed_dir = Path(seed_path) if seed_path else spec_path / "seed"

        # Try to get from evolution targets first
        evolution_targets = domain_understanding.get("evolution_targets", [])
        if evolution_targets:
            # Return the file path of the first evolution target
            first_target = evolution_targets[0]
            target_path = first_target.get("file_path", "")
            if target_path:
                # Make sure it's an absolute path
                if not target_path.startswith("/"):
                    # It's relative - make it absolute relative to seed
                    full_path = seed_dir / target_path
                    if full_path.exists():
                        return str(full_path)
                else:
                    return target_path

        # Try structure entry points
        structure = domain_understanding.get("structure", {})
        if structure.get("entry_points"):
            return structure["entry_points"][0]

        if structure.get("core_modules"):
            return structure["core_modules"][0]

        # Fall back to searching seed path
        seed_dir = Path(seed_path) if seed_path else spec_path / "seed"

        if not seed_dir.exists():
            return str(spec_path / "seed" / "main.py")

        # Priority patterns for finding main files
        patterns = [
            '**/main.py', '**/app.py', '**/core.py', '**/engine.py',
            '**/model.py', '**/agent.py', '**/__main__.py'
        ]

        for pattern in patterns:
            matches = list(seed_dir.glob(pattern))
            if matches:
                # Prefer files in root or src/
                for match in matches:
                    rel = str(match.relative_to(seed_dir))
                    if '/' not in rel or rel.startswith('src/'):
                        return str(match)
                return str(matches[0])

        # Find any Python file with meaningful content
        for py_file in seed_dir.rglob('*.py'):
            if '__pycache__' in str(py_file) or 'test' in py_file.name.lower():
                continue
            try:
                content = py_file.read_text(errors='ignore')
                if 'class ' in content or 'def main' in content:
                    return str(py_file)
            except Exception:
                continue

        return str(spec_path / "seed" / "main.py")

    async def _generate_evaluation_script(
        self,
        domain_name: str,
        codebase_purpose: str,
        structure: Dict,
        evolution_targets: List[Dict],
        strategies: List[Dict],
        spec_path: Path,
        seed_path: Optional[str] = None
    ) -> str:
        """Generate an evaluation script for the evolved code."""
        # Format inputs for the prompt
        structure_str = self._format_structure(structure)
        targets_str = self._format_targets(evolution_targets)
        strategies_str = self._format_strategies_for_eval(strategies)
        eval_focus = self._generate_evaluation_focus(evolution_targets, strategies)

        # Determine the actual codebase path
        codebase_path = seed_path if seed_path else str(spec_path / "seed")

        prompt = EVALUATION_SCRIPT_PROMPT.format(
            domain_name=domain_name,
            codebase_purpose=codebase_purpose,
            seed_codebase_path=codebase_path,
            codebase_structure=structure_str,
            evolution_targets=targets_str,
            evaluation_strategies=strategies_str,
            evaluation_focus=eval_focus,
            result_file=str(spec_path / "results" / "evaluation.csv"),
            test_result_file=str(spec_path / "results" / "metrics.csv"),
            debug_file=str(spec_path / "debug" / "error.txt")
        )

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI()

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate practical evaluation scripts. Output ONLY the script code. Do not include markdown code blocks."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000
            )

            content = response.choices[0].message.content
            return self._extract_code_from_response(content, "bash")

        except Exception as e:
            print(f"  Warning: Evaluation script generation failed: {e}", flush=True)
            return self._default_evaluation_script(domain_name, evolution_targets, strategies, seed_path, spec_path)

    def _format_structure(self, structure: Dict) -> str:
        """Format structure for prompt."""
        if not structure:
            return "Structure not analyzed"

        lines = []
        if structure.get("architecture_pattern"):
            lines.append(f"Architecture: {structure['architecture_pattern']}")
        if structure.get("entry_points"):
            lines.append(f"Entry points: {', '.join(structure['entry_points'][:3])}")
        if structure.get("core_modules"):
            lines.append(f"Core modules: {', '.join(structure['core_modules'][:5])}")
        if structure.get("test_modules"):
            lines.append(f"Test modules: {', '.join(structure['test_modules'][:3])}")

        return "\n".join(lines) if lines else "Standard structure"

    def _format_targets(self, targets: List[Dict]) -> str:
        """Format evolution targets for prompt."""
        if not targets:
            return "No specific targets - general improvement"

        lines = []
        for t in targets[:5]:
            lines.append(f"- {t.get('component_name', 'Component')} ({t.get('file_path', 'unknown')})")
            lines.append(f"  Current: {t.get('current_behavior', 'Not specified')}")
            lines.append(f"  Goal: {t.get('evolution_rationale', 'Improve')}")

        return "\n".join(lines)

    def _format_strategies_for_eval(self, strategies: List[Dict]) -> str:
        """Format strategies for evaluation prompt."""
        if not strategies:
            return "General code improvement"

        lines = []
        for s in strategies[:4]:
            lines.append(f"**{s.get('name', 'Strategy')}**")
            criteria = s.get("evaluation_criteria", [])
            if criteria:
                lines.append(f"  Success criteria: {', '.join(criteria[:3])}")

        return "\n".join(lines)

    def _generate_evaluation_focus(self, targets: List[Dict], strategies: List[Dict]) -> str:
        """Generate evaluation focus points."""
        focus = []

        # From targets
        for t in targets[:3]:
            constraints = t.get("constraints", [])
            if constraints:
                focus.append(f"- {t.get('component_name', 'Component')}: {', '.join(constraints[:2])}")

        # From strategies
        for s in strategies[:3]:
            criteria = s.get("evaluation_criteria", [])
            for c in criteria[:2]:
                focus.append(f"- {c}")

        if not focus:
            focus = [
                "- Code runs without errors",
                "- Core functionality is preserved",
                "- Improvement is measurable"
            ]

        return "\n".join(focus)

    def _default_evaluation_script(
        self,
        domain_name: str,
        targets: List[Dict],
        strategies: List[Dict],
        seed_path: Optional[str] = None,
        spec_path: Optional[Path] = None
    ) -> str:
        """Generate default evaluation script that works with the actual codebase."""
        # Extract evaluation criteria from strategies
        criteria = []
        for s in strategies:
            criteria.extend(s.get("evaluation_criteria", []))

        criteria_checks = ""
        if criteria:
            criteria_checks = "\n# Evaluation criteria from strategies:\n"
            for c in criteria[:5]:
                criteria_checks += f"# - {c}\n"

        # Determine paths
        codebase_dir = seed_path if seed_path else (str(spec_path / "seed") if spec_path else "./seed")

        return f'''#!/bin/bash
# Evaluation script for {domain_name}
#
# This script evaluates evolved code within the context of the actual codebase.
# The evolved code is a file that exists within or relates to the seed codebase.
#
# Usage: bash evaluate.sh <evolved_code_path>
{criteria_checks}
set -e

EVOLVED_CODE="$1"
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
SPEC_DIR="$SCRIPT_DIR/.."

# CRITICAL: The actual codebase location
CODEBASE_DIR="{codebase_dir}"

# Results paths
RESULT_FILE="$SPEC_DIR/results/evaluation.csv"
METRICS_FILE="$SPEC_DIR/results/metrics.csv"
DEBUG_FILE="$SPEC_DIR/debug/error.txt"

# Create directories
mkdir -p "$SPEC_DIR/results"
mkdir -p "$SPEC_DIR/debug"

# Clear previous results
> "$RESULT_FILE"
> "$METRICS_FILE"
> "$DEBUG_FILE"

if [ -z "$EVOLVED_CODE" ]; then
    echo "Usage: bash evaluate.sh <evolved_code_path>"
    exit 1
fi

echo "Evaluating: $EVOLVED_CODE"
echo "Codebase directory: $CODEBASE_DIR"

# Initialize results
echo "metric,value" > "$RESULT_FILE"
echo "metric,value" > "$METRICS_FILE"

# ============================================================================
# Check 1: Code file exists
# ============================================================================
if [ ! -f "$EVOLVED_CODE" ]; then
    echo "Error: Evolved code not found at $EVOLVED_CODE" | tee "$DEBUG_FILE"
    echo "code_exists,0" >> "$RESULT_FILE"
    exit 1
fi
echo "code_exists,1" >> "$RESULT_FILE"
echo "[OK] Code file exists"

# ============================================================================
# Check 2: Syntax validation (Python)
# ============================================================================
echo "Checking Python syntax..."
if [[ "$EVOLVED_CODE" == *.py ]]; then
    python3 -m py_compile "$EVOLVED_CODE" 2>>"$DEBUG_FILE"
    if [ $? -eq 0 ]; then
        echo "syntax_valid,1" >> "$RESULT_FILE"
        echo "[OK] Syntax is valid"
    else
        echo "syntax_valid,0" >> "$RESULT_FILE"
        echo "Syntax error in evolved code" >> "$DEBUG_FILE"
        exit 1
    fi
else
    echo "syntax_valid,N/A" >> "$RESULT_FILE"
fi

# ============================================================================
# Check 3: Module import test
# ============================================================================
echo "Testing module import..."
cd "$CODEBASE_DIR"

IMPORT_RESULT=$(python3 -c "
import sys
import importlib.util

# Add codebase directory to path so local modules can be imported
sys.path.insert(0, '$CODEBASE_DIR')

try:
    spec = importlib.util.spec_from_file_location('evolved_module', '$EVOLVED_CODE')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print('import_success,1')
except Exception as e:
    print('import_success,0')
    print(f'Import error: {{e}}', file=sys.stderr)
    sys.exit(1)
" 2>>"$DEBUG_FILE")

echo "$IMPORT_RESULT" >> "$RESULT_FILE"

if echo "$IMPORT_RESULT" | grep -q "import_success,0"; then
    echo "[FAIL] Module import failed"
    exit 1
else
    echo "[OK] Module imports successfully"
fi

# ============================================================================
# Check 4: Run pytest if tests exist
# ============================================================================
# Check for tests in the codebase
TEST_DIRS=("$CODEBASE_DIR/tests" "$CODEBASE_DIR/test" "$CODEBASE_DIR")

for TEST_DIR in "${{TEST_DIRS[@]}}"; do
    if [ -d "$TEST_DIR" ] && ls "$TEST_DIR"/test_*.py 1> /dev/null 2>&1; then
        echo "Running tests from $TEST_DIR..."
        cd "$CODEBASE_DIR"
        python3 -m pytest "$TEST_DIR" -v --tb=short 2>>"$DEBUG_FILE"
        if [ $? -eq 0 ]; then
            echo "tests_pass,1" >> "$METRICS_FILE"
            echo "[OK] Tests passed"
        else
            echo "tests_pass,0" >> "$METRICS_FILE"
            echo "[WARN] Some tests failed (see debug log)"
        fi
        break
    fi
done

# If no test directory found
if ! grep -q "tests_pass" "$METRICS_FILE" 2>/dev/null; then
    echo "tests_pass,N/A" >> "$METRICS_FILE"
    echo "[INFO] No tests directory found"
fi

# ============================================================================
# Calculate overall score
# ============================================================================
echo "Calculating score..."
python3 -c "
import csv

score = 0
max_score = 0

# Read main results
try:
    with open('$RESULT_FILE', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get('value', '0')
            if val not in ('N/A', ''):
                max_score += 1
                if val == '1':
                    score += 1
except Exception:
    pass

# Read metrics
try:
    with open('$METRICS_FILE', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get('value', '0')
            if val not in ('N/A', '') and row.get('metric') in ('tests_pass', 'model_instantiation'):
                max_score += 1
                if val == '1':
                    score += 1
except Exception:
    pass

final_score = score / max_score if max_score > 0 else 0.0
print(f'overall_score,{{final_score:.3f}}')
" >> "$RESULT_FILE"

echo ""
echo "============================================"
echo "Evaluation complete"
echo "Results: $RESULT_FILE"
echo "Metrics: $METRICS_FILE"
echo "Debug:   $DEBUG_FILE"
echo "============================================"

exit 0
'''

    def _generate_result_parser(self, domain_name: str, strategies: List[Dict]) -> str:
        """Generate result parsing code."""
        # Extract evaluation criteria for parser
        criteria = []
        for s in strategies:
            criteria.extend(s.get("evaluation_criteria", []))

        criteria_str = ", ".join(f'"{c}"' for c in criteria[:5]) if criteria else '"score"'

        return f'''"""
Result Parser for {domain_name}

Parses evaluation results from CSV files.
"""

import csv
from typing import Dict, List, Optional


# Expected evaluation criteria
EXPECTED_CRITERIA = [{criteria_str}]


def parse_evaluation_results(filepath: str) -> Dict[str, float]:
    """Parse evaluation results from CSV."""
    results = {{}}
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row.get('metric', '')
                value = row.get('value', '0')
                try:
                    results[metric] = float(value)
                except ValueError:
                    results[metric] = 0.0 if value != 'N/A' else None
    except FileNotFoundError:
        pass
    return results


def parse_metrics(filepath: str) -> Dict[str, float]:
    """Parse additional metrics from CSV."""
    return parse_evaluation_results(filepath)


def calculate_score(results: Dict[str, float], metrics: Dict[str, float]) -> float:
    """Calculate overall score from results and metrics."""
    score = 0.0
    count = 0

    # Score from main results
    for key, value in results.items():
        if value is not None and isinstance(value, (int, float)):
            score += value
            count += 1

    # Score from metrics
    for key, value in metrics.items():
        if value is not None and isinstance(value, (int, float)):
            score += value
            count += 1

    return score / count if count > 0 else 0.0


def format_results(results: Dict[str, float], metrics: Dict[str, float]) -> str:
    """Format results for display."""
    lines = ["Evaluation Results:"]
    for key, value in results.items():
        lines.append(f"  {{key}}: {{value}}")
    if metrics:
        lines.append("\\nMetrics:")
        for key, value in metrics.items():
            lines.append(f"  {{key}}: {{value}}")
    return "\\n".join(lines)
'''

    def _extract_benchmarks(self, strategies: List[Dict]) -> List[str]:
        """Extract benchmark names from strategies."""
        benchmarks = []
        for s in strategies:
            criteria = s.get("evaluation_criteria", [])
            benchmarks.extend(criteria[:2])

        if not benchmarks:
            benchmarks = ["functionality", "quality", "performance"]

        return list(set(benchmarks))[:5]

    def _extract_baselines(self, strategies: List[Dict]) -> List[Dict[str, Any]]:
        """Extract baseline definitions from strategies."""
        baselines = []
        for i, s in enumerate(strategies[:2]):
            baselines.append({
                "name": f"baseline_{s.get('name', f'strategy_{i}').lower().replace(' ', '_')}",
                "description": f"Baseline for {s.get('name', 'strategy')}",
                "score": 50.0,  # Default baseline score
                "metrics": {c: 0.5 for c in s.get("evaluation_criteria", ["score"])[:3]}
            })

        if not baselines:
            baselines = [{
                "name": "baseline_default",
                "description": "Default baseline",
                "score": 50.0,
                "metrics": {"score": 0.5}
            }]

        return baselines

    async def _generate_seed_artifact(
        self,
        domain_name: str,
        domain_description: str,
        expertise: Dict,
        strategies: List[Dict]
    ) -> str:
        """Generate initial seed code for genesis mode."""
        primary_domain = expertise.get("primary_domain", "software")
        key_concepts = expertise.get("key_concepts", [])
        strategies_str = "\n".join(
            f"- {s.get('name', 'Strategy')}: {s.get('description', '')}"
            for s in strategies[:3]
        )

        prompt = SEED_GENERATION_PROMPT.format(
            domain_name=domain_name,
            domain_description=domain_description,
            primary_domain=primary_domain,
            key_concepts=", ".join(key_concepts[:5]) if key_concepts else "general concepts",
            evolution_strategies=strategies_str if strategies_str else "General improvement"
        )

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI()

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate production-quality code. Output ONLY the code. No markdown code blocks."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=8000
            )

            content = response.choices[0].message.content
            return self._extract_code_from_response(content, "python")

        except Exception as e:
            print(f"  Warning: Seed generation failed: {e}", flush=True)
            return self._default_seed_artifact(domain_name, expertise)

    def _default_seed_artifact(self, domain_name: str, expertise: Dict) -> str:
        """Generate default seed artifact."""
        primary_domain = expertise.get("primary_domain", "software")

        return f'''"""
{domain_name} - Seed Implementation

This is the initial implementation that will be evolved.
Domain: {primary_domain}
"""


class {domain_name.replace(" ", "").replace("-", "")}:
    """
    Main class for {domain_name}.

    This serves as the seed for autonomous evolution.
    """

    def __init__(self, config=None):
        """Initialize with optional configuration."""
        self.config = config or {{}}
        self._setup()

    def _setup(self):
        """Set up initial state."""
        pass

    def run(self, *args, **kwargs):
        """
        Main entry point.

        Override this method with domain-specific logic.
        """
        raise NotImplementedError("Implement run() method")

    def evaluate(self):
        """
        Evaluate current state.

        Returns metrics for evolution scoring.
        """
        return {{"score": 0.0}}


def main():
    """Entry point for running the code."""
    instance = {domain_name.replace(" ", "").replace("-", "")}()
    result = instance.run()
    metrics = instance.evaluate()
    print(f"Result: {{result}}")
    print(f"Metrics: {{metrics}}")
    return metrics


if __name__ == "__main__":
    main()
'''

    def _is_python_script(self, content: str) -> bool:
        """Check if content is Python rather than bash."""
        first_lines = content.strip().split('\n')[:10]
        first_content = '\n'.join(first_lines)

        if content.strip().startswith('#!/bin/bash') or content.strip().startswith('#!/bin/sh'):
            return False

        python_indicators = [
            'import ', 'from ', 'def ', 'class ', 'if __name__',
            'argparse.ArgumentParser', 'import argparse'
        ]

        indicator_count = sum(1 for ind in python_indicators if ind in first_content)
        return indicator_count >= 2

    async def save_infrastructure(
        self,
        setup: InfrastructureSetup,
        specialization_path: str
    ):
        """Save infrastructure files to disk."""
        import os

        spec_path = Path(specialization_path)

        # Create directories
        (spec_path / "infrastructure").mkdir(parents=True, exist_ok=True)
        (spec_path / "results").mkdir(parents=True, exist_ok=True)
        (spec_path / "debug").mkdir(parents=True, exist_ok=True)
        (spec_path / "pool").mkdir(parents=True, exist_ok=True)
        (spec_path / "seed").mkdir(parents=True, exist_ok=True)

        # Save evaluation script
        script_content = setup.training_script

        if self._is_python_script(script_content):
            # Save as Python file
            py_path = spec_path / "infrastructure" / "evaluate.py"
            with open(py_path, 'w') as f:
                f.write(script_content)

            # Create bash wrapper
            wrapper = '''#!/bin/bash
# Wrapper script for evaluation
# Usage: bash evaluate.sh <evolved_code_path>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python "$SCRIPT_DIR/evaluate.py" "$@"
'''
            script_path = spec_path / "infrastructure" / "evaluate.sh"
            with open(script_path, 'w') as f:
                f.write(wrapper)
            os.chmod(script_path, 0o755)

            # Also create train.sh symlink for backwards compat
            train_path = spec_path / "infrastructure" / "train.sh"
            if not train_path.exists():
                with open(train_path, 'w') as f:
                    f.write(wrapper)
                os.chmod(train_path, 0o755)
        else:
            # Save as bash script
            script_path = spec_path / "infrastructure" / "evaluate.sh"
            with open(script_path, 'w') as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)

            # Also create train.sh for backwards compat
            train_path = spec_path / "infrastructure" / "train.sh"
            with open(train_path, 'w') as f:
                f.write(script_content)
            os.chmod(train_path, 0o755)

        # Save result parser
        parser_path = spec_path / "infrastructure" / "result_parser.py"
        with open(parser_path, 'w') as f:
            f.write(setup.result_parser)

        # Save seed artifact if present
        if setup.seed_artifact:
            seed_path = spec_path / "seed" / "main.py"
            with open(seed_path, 'w') as f:
                f.write(setup.seed_artifact)
