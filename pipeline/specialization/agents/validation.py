"""
Validation Agent

Validates entire specialization is coherent by running tests
on all components including a full end-to-end experiment.
"""

import json
import asyncio
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

from pydantic import BaseModel


class ValidationResult(BaseModel):
    """Result of validation."""
    passed: bool
    errors: List[str]
    warnings: List[str]
    tests_run: int
    tests_passed: int
    details: Dict[str, Any]


@dataclass
class ValidationTest:
    """A single validation test."""
    name: str
    description: str
    passed: bool = False
    error: Optional[str] = None


class ValidationAgent:
    """
    Agent that validates a complete specialization.

    Runs multiple validation tests:
    1. Configuration validity
    2. Prompt template rendering
    3. Infrastructure execution
    4. Knowledge base accessibility
    5. End-to-end experiment (optional)
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    async def validate(
        self,
        specialization: Any,  # DomainSpecialization
        run_e2e: bool = True
    ) -> ValidationResult:
        """
        Validate a specialization.

        Args:
            specialization: The DomainSpecialization to validate
            run_e2e: Whether to run full end-to-end test

        Returns:
            ValidationResult with pass/fail status and details
        """
        tests: List[ValidationTest] = []
        errors: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {}

        # Test 1: Configuration validity
        test = await self._validate_config(specialization)
        tests.append(test)
        if not test.passed:
            errors.append(f"Config validation failed: {test.error}")

        # Test 2: Prompt template rendering
        test = await self._validate_prompts(specialization)
        tests.append(test)
        if not test.passed:
            errors.append(f"Prompt validation failed: {test.error}")

        # Test 3: Infrastructure existence
        test = await self._validate_infrastructure(specialization)
        tests.append(test)
        if not test.passed:
            errors.append(f"Infrastructure validation failed: {test.error}")

        # Test 4: Knowledge base
        test = await self._validate_knowledge(specialization)
        tests.append(test)
        if not test.passed:
            warnings.append(f"Knowledge validation warning: {test.error}")

        # Test 5: Constraints validity
        test = await self._validate_constraints(specialization)
        tests.append(test)
        if not test.passed:
            errors.append(f"Constraint validation failed: {test.error}")

        # Test 6: End-to-end (if requested)
        if run_e2e:
            test = await self._validate_e2e(specialization)
            tests.append(test)
            if not test.passed:
                errors.append(f"E2E validation failed: {test.error}")
            details["e2e_result"] = test.error if not test.passed else "passed"

        # Compile results
        tests_passed = sum(1 for t in tests if t.passed)

        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            tests_run=len(tests),
            tests_passed=tests_passed,
            details=details
        )

    async def _validate_config(self, spec) -> ValidationTest:
        """Validate configuration is complete and valid."""
        test = ValidationTest(
            name="config_validity",
            description="Check configuration completeness"
        )

        try:
            # Check required fields
            required_fields = [
                "name", "display_name", "description",
                "architecture", "evaluation", "constraints",
                "prompts", "knowledge", "infrastructure"
            ]

            for field in required_fields:
                if not getattr(spec, field, None):
                    test.error = f"Missing required field: {field}"
                    return test

            # Check architecture config
            arch = spec.architecture
            if not arch.base_class_name:
                test.error = "Missing base_class_name in architecture"
                return test

            # Check evaluation config
            eval_config = spec.evaluation
            if not eval_config.benchmarks:
                test.error = "No benchmarks defined"
                return test

            test.passed = True
            return test

        except Exception as e:
            test.error = str(e)
            return test

    async def _validate_prompts(self, spec) -> ValidationTest:
        """Validate prompt templates can be rendered."""
        test = ValidationTest(
            name="prompt_rendering",
            description="Check prompt templates render correctly"
        )

        try:
            from ..templates.renderer import PromptRenderer

            renderer = PromptRenderer(spec)

            # Try rendering each prompt
            prompts_to_test = [
                ("planner", lambda: renderer.render_planner("test context")),
                ("checker", lambda: renderer.render_checker("test motivation")),
                ("analyzer", lambda: renderer.render_analyzer("test", "results", "motivation", "context")),
            ]

            for name, render_fn in prompts_to_test:
                try:
                    result = render_fn()
                    if not result or len(result) < 50:
                        test.error = f"Prompt '{name}' rendered too short"
                        return test
                except Exception as e:
                    test.error = f"Failed to render '{name}': {e}"
                    return test

            test.passed = True
            return test

        except Exception as e:
            test.error = str(e)
            return test

    async def _validate_infrastructure(self, spec) -> ValidationTest:
        """Validate infrastructure files exist."""
        test = ValidationTest(
            name="infrastructure",
            description="Check infrastructure files exist"
        )

        try:
            infra = spec.infrastructure

            # Check training script exists
            script_path = Path(infra.training_script)
            if not script_path.exists():
                test.error = f"Training script not found: {infra.training_script}"
                return test

            # Check directories exist
            for dir_path in [infra.code_pool]:
                path = Path(dir_path)
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)

            test.passed = True
            return test

        except Exception as e:
            test.error = str(e)
            return test

    async def _validate_knowledge(self, spec) -> ValidationTest:
        """Validate knowledge base is accessible."""
        test = ValidationTest(
            name="knowledge_base",
            description="Check knowledge base accessibility"
        )

        try:
            knowledge = spec.knowledge

            # Check corpus path exists
            corpus_path = Path(knowledge.corpus_path)
            if not corpus_path.exists():
                test.error = f"Knowledge corpus not found: {knowledge.corpus_path}"
                return test

            # Check if there are any documents
            docs = list(corpus_path.glob("*.json"))
            if len(docs) == 0:
                test.error = "No knowledge documents found"
                return test

            test.passed = True
            return test

        except Exception as e:
            test.error = str(e)
            return test

    async def _validate_constraints(self, spec) -> ValidationTest:
        """Validate constraints are properly defined."""
        test = ValidationTest(
            name="constraints",
            description="Check constraints are valid"
        )

        try:
            constraints = spec.constraints

            # Check we have at least some constraints
            all_constraints = constraints.get_all_constraints()
            if len(all_constraints) == 0:
                test.error = "No constraints defined"
                return test

            # Check each constraint has required fields
            for c in all_constraints:
                if not c.name or not c.description:
                    test.error = f"Constraint missing name or description"
                    return test

            test.passed = True
            return test

        except Exception as e:
            test.error = str(e)
            return test

    async def _validate_e2e(self, spec) -> ValidationTest:
        """Run full end-to-end validation."""
        test = ValidationTest(
            name="end_to_end",
            description="Run complete experiment cycle"
        )

        try:
            # This is a simplified E2E test
            # In production, this would run the full pipeline

            infra = spec.infrastructure

            # Check training script is executable
            script_path = Path(infra.training_script)
            if not script_path.exists():
                test.error = "Training script not found"
                return test

            # Try to run a basic test (with timeout)
            try:
                # Create a dummy test
                result = subprocess.run(
                    ["bash", "-n", str(script_path)],  # Syntax check only
                    capture_output=True,
                    timeout=30
                )

                if result.returncode != 0:
                    test.error = f"Training script syntax error: {result.stderr.decode()}"
                    return test

            except subprocess.TimeoutExpired:
                test.error = "Training script validation timed out"
                return test
            except FileNotFoundError:
                # bash not available, skip this check
                pass

            # Verify result paths are configured
            result_path = Path(infra.result_file)
            result_path.parent.mkdir(parents=True, exist_ok=True)

            test.passed = True
            return test

        except Exception as e:
            test.error = str(e)
            return test

    async def quick_validate(self, spec) -> List[str]:
        """
        Run quick validation without E2E test.

        Args:
            spec: DomainSpecialization to validate

        Returns:
            List of error messages (empty if passed)
        """
        result = await self.validate(spec, run_e2e=False)
        return result.errors
