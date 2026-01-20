"""
Prompt Renderer

Renders parameterized prompt templates with domain-specific content
from the active specialization.
"""

from typing import Dict, Any, Optional
from ..schema import (
    DomainSpecialization,
    Constraint,
    ConstraintSeverity,
    BaselineModel
)


class PromptRenderer:
    """
    Renders prompt templates with domain-specific content.

    This class takes the parameterized templates and fills them with
    values from the active domain specialization, producing ready-to-use
    prompts for each agent in the pipeline.
    """

    def __init__(self, specialization: DomainSpecialization):
        """
        Initialize the renderer with a domain specialization.

        Args:
            specialization: The domain specialization to use for rendering
        """
        self.spec = specialization
        self._cache: Dict[str, str] = {}

    def _title_case(self, text: str) -> str:
        """Convert artifact_type to Title Case."""
        return text.replace("_", " ").title()

    def _format_constraints(self, constraints: list[Constraint], prefix: str = "") -> str:
        """Format a list of constraints for inclusion in prompts."""
        if not constraints:
            return f"{prefix}No specific constraints defined."

        lines = []
        for i, c in enumerate(constraints, 1):
            severity_icon = {
                ConstraintSeverity.STRICT: "🔴",
                ConstraintSeverity.CRITICAL: "🟡",
                ConstraintSeverity.FLEXIBLE: "🟢"
            }.get(c.severity, "")

            lines.append(f"**{chr(64+i)}. {c.name}** {severity_icon}")
            lines.append(c.description)
            if c.validation_prompt:
                lines.append(f"Check: {c.validation_prompt}")
            lines.append("")

        return "\n".join(lines)

    def _format_baseline_models(self, baselines: list[BaselineModel]) -> str:
        """Format baseline models for the model judger prompt."""
        if not baselines:
            return "No baseline models defined."

        sections = []
        for i, baseline in enumerate(baselines, 1):
            section = f"""### Model {i}: {baseline.name} (Score: {baseline.score}/10)
**Description**: {baseline.description}

**Metrics**:
"""
            for metric, value in baseline.metrics.items():
                section += f"- {metric}: {value}\n"

            if baseline.training_curve:
                section += f"\n**Training Curve** (sample): {baseline.training_curve[:5]}...\n"

            sections.append(section)

        return "\n".join(sections)

    def _format_preservation_rules(self) -> str:
        """Format preservation rules for prompts."""
        rules = self.spec.constraints.preservation_rules
        if not rules:
            return "- Maintain interface compatibility"

        return "\n".join(f"- {rule}" for rule in rules)

    def _format_benchmarks(self) -> str:
        """Format benchmarks list."""
        benchmarks = self.spec.evaluation.benchmarks
        if not benchmarks:
            return "Standard evaluation metrics"

        return ", ".join(benchmarks)

    def _get_common_variables(self) -> Dict[str, Any]:
        """Get variables common across all prompts."""
        return {
            "domain_name": self.spec.display_name,
            "artifact_type": self.spec.architecture.artifact_type,
            "artifact_type_title": self._title_case(self.spec.architecture.artifact_type),
            "base_class_name": self.spec.architecture.base_class_name,
            "standard_parameters": ", ".join(self.spec.architecture.standard_parameters),
            "interface_signature": self.spec.architecture.interface_signature,
            "required_decorators": ", ".join(self.spec.architecture.required_decorators) or "None",
            "complexity_requirement": self.spec.constraints.complexity_requirement or "No specific requirement",
            "benchmarks": self._format_benchmarks(),
            "naming_prefix": self.spec.architecture.base_class_name.lower(),
        }

    def render_planner(self, context: str) -> str:
        """
        Render the planner prompt with domain-specific content.

        Args:
            context: Experimental context from previous experiments

        Returns:
            Rendered prompt string
        """
        # If custom prompt is provided in specialization, use it
        if not self.spec.prompts.planner.startswith("[PLACEHOLDER"):
            return self.spec.prompts.planner.format(context=context)

        # Otherwise, render from template
        vars = self._get_common_variables()
        vars["context"] = context

        # Build efficiency constraints section
        if self.spec.constraints.complexity_requirement:
            vars["efficiency_constraints"] = f"""- Design with {self.spec.constraints.complexity_requirement} complexity
- Optimize for computational efficiency
- Maintain performance gains within complexity bounds"""
        else:
            vars["efficiency_constraints"] = "- Optimize for computational efficiency"

        # Build domain-specific implementation rules
        rules = self.spec.architecture.code_style_guidelines or ""
        vars["domain_specific_implementation_rules"] = rules

        # Build quality assurance rules
        qa_rules = []
        if self.spec.architecture.required_decorators:
            qa_rules.append(f"- Maintain {', '.join(self.spec.architecture.required_decorators)} where appropriate")
        qa_rules.append("- Ensure code correctness and efficiency")
        vars["quality_assurance_rules"] = "\n".join(qa_rules)

        # Build preservation requirements
        vars["preservation_requirements"] = f"""- **Class Structure**: Maintain {self.spec.architecture.base_class_name} class name
- **Interface Stability**: Preserve {self.spec.architecture.interface_signature}
- **Parameter Compatibility**: Support standard parameters ({vars['standard_parameters']})"""

        # Build implementation quality standards
        standards = []
        if self.spec.constraints.complexity_requirement:
            standards.append(f"- **Complexity Bounds**: Ensure {self.spec.constraints.complexity_requirement}")
        for rule in self.spec.constraints.preservation_rules[:3]:  # First 3 rules
            standards.append(f"- {rule}")
        vars["implementation_quality_standards"] = "\n".join(standards) if standards else "- Follow best practices"

        # Build robustness standards
        vars["robustness_standards"] = """### Cross-Environment Robustness Standards
- **Universal Compatibility**: Identical performance across training/evaluation/inference
- **Resource Adaptation**: Graceful handling of varying constraints
- **Shape Tolerance**: Robust operation with varying input dimensions"""

        # Build enhancement areas
        vars["enhancement_areas"] = """- **Performance Optimization**: Superior capabilities within constraints
- **Efficiency Improvements**: Better resource utilization
- **Robustness Enhancements**: Consistent performance across contexts
- **Innovation Integration**: Novel approaches from research"""

        # Build technical standards summary
        vars["technical_standards_summary"] = f"complexity bounds, efficiency, correctness"

        # Build success criteria
        vars["success_criteria"] = """1. **Implementation Excellence**: Successfully create working code using write_code_file
2. **Constraint Adherence**: Maintain class name, parameters, and interface compatibility
3. **Technical Robustness**: Ensure required constraints are met
4. **Evidence-Based Innovation**: Embed research insights addressing identified limitations
5. **Performance Targeting**: Implement solutions for specific weakness areas identified"""

        from .base_templates import PLANNER_TEMPLATE
        return PLANNER_TEMPLATE.format(**vars)

    def render_checker(self, motivation: str) -> str:
        """
        Render the checker prompt with domain-specific constraints.

        Args:
            motivation: Design motivation for context

        Returns:
            Rendered prompt string
        """
        if not self.spec.prompts.checker.startswith("[PLACEHOLDER"):
            return self.spec.prompts.checker.format(motivation=motivation)

        vars = self._get_common_variables()
        vars["motivation"] = motivation

        # Format constraint sections
        vars["strict_checks"] = self._format_constraints(
            self.spec.constraints.strict_constraints
        )
        vars["critical_checks"] = self._format_constraints(
            self.spec.constraints.critical_constraints
        )
        vars["flexible_checks"] = self._format_constraints(
            self.spec.constraints.flexible_constraints
        )

        # Build common fixes section from constraint examples
        fixes = []
        for c in self.spec.constraints.get_all_constraints():
            if c.examples:
                fixes.append(f"**{c.name} Fix**:")
                for ex in c.examples[:2]:  # Max 2 examples per constraint
                    if "before" in ex and "after" in ex:
                        fixes.append(f"```python\n# Before (wrong)\n{ex['before']}\n# After (correct)\n{ex['after']}\n```")
                fixes.append("")

        vars["common_fixes"] = "\n".join(fixes) if fixes else "No common fixes documented."

        from .base_templates import CHECKER_TEMPLATE
        return CHECKER_TEMPLATE.format(**vars)

    def render_motivation_checker(self, motivation: str, historical_context: str) -> str:
        """
        Render the motivation checker prompt.

        Args:
            motivation: New motivation to check
            historical_context: Previous motivations for comparison

        Returns:
            Rendered prompt string
        """
        if not self.spec.prompts.motivation_checker.startswith("[PLACEHOLDER"):
            return self.spec.prompts.motivation_checker.format(
                motivation=motivation,
                historical_context=historical_context
            )

        vars = self._get_common_variables()
        vars["motivation"] = motivation
        vars["historical_context"] = historical_context

        # Domain-specific research context
        vars["domain_research_context"] = f"""- Diverse approaches to similar problems are valuable
- Different aspects of the same mechanism are distinct research directions
- Incremental improvements on promising approaches are valid
- Novel combinations of existing ideas represent genuine innovation"""

        from .base_templates import MOTIVATION_CHECKER_TEMPLATE
        return MOTIVATION_CHECKER_TEMPLATE.format(**vars)

    def render_deduplication(self, repeated_motivation: str, historical_context: str) -> str:
        """
        Render the deduplication prompt.

        Args:
            repeated_motivation: The motivation that was marked as repeated
            historical_context: Context from previous experiments

        Returns:
            Rendered prompt string
        """
        if not self.spec.prompts.deduplication.startswith("[PLACEHOLDER"):
            return self.spec.prompts.deduplication.format(
                repeated_motivation=repeated_motivation,
                historical_context=historical_context
            )

        vars = self._get_common_variables()
        vars["repeated_motivation"] = repeated_motivation
        vars["historical_context"] = historical_context

        # Cross-disciplinary exploration targets
        vars["cross_disciplinary_targets"] = """- Different mathematical foundations
- Cross-domain inspiration (biology, physics, engineering)
- Alternative computational paradigms
- Novel optimization approaches
- Different performance-efficiency trade-offs"""

        # Innovation guidelines
        vars["innovation_guidelines"] = """- Break away from the repeated pattern completely
- Explore fundamentally different approaches
- Consider unconventional solutions
- Maintain required technical standards"""

        # Preservation constraints
        vars["preservation_constraints"] = self._format_preservation_rules()

        # Robustness standards
        vars["robustness_standards"] = """- Code must work in all environments
- Handle edge cases gracefully
- Maintain interface compatibility"""

        # Success criteria
        vars["success_criteria"] = """- [ ] Approach is genuinely different from repeated pattern
- [ ] Implementation maintains required standards
- [ ] Innovation has theoretical justification
- [ ] Code is complete and functional"""

        from .base_templates import DEDUPLICATION_TEMPLATE
        return DEDUPLICATION_TEMPLATE.format(**vars)

    def render_debugger(self, motivation: str, error_log: str) -> str:
        """
        Render the debugger prompt.

        Args:
            motivation: Design motivation for context
            error_log: Error log from failed training

        Returns:
            Rendered prompt string
        """
        if not self.spec.prompts.debugger.startswith("[PLACEHOLDER"):
            return self.spec.prompts.debugger.format(
                motivation=motivation,
                error_log=error_log
            )

        vars = self._get_common_variables()
        vars["motivation"] = motivation
        vars["error_log"] = error_log

        # Debug constraints
        vars["debug_constraints"] = f"""- Preserve {self.spec.architecture.base_class_name} class name
- Maintain {self.spec.architecture.interface_signature}
- Keep {vars['required_decorators']} in place
- Don't change the core innovation"""

        # Fix strategies
        vars["timeout_fix_strategy"] = """- Look for complexity issues
- Optimize loops and operations
- Check for inefficient algorithms
- Verify chunking is used where appropriate"""

        vars["crash_fix_strategy"] = """- Check tensor shapes match expectations
- Verify device placement (CPU/GPU)
- Look for index out of bounds
- Check for type mismatches"""

        vars["numerical_fix_strategy"] = """- Add numerical stability (epsilon values)
- Check for division by zero
- Look for overflow/underflow conditions
- Verify gradients can flow"""

        from .base_templates import DEBUGGER_TEMPLATE
        return DEBUGGER_TEMPLATE.format(**vars)

    def render_analyzer(
        self,
        experiment_name: str,
        results: str,
        motivation: str,
        reference_context: str
    ) -> str:
        """
        Render the analyzer prompt.

        Args:
            experiment_name: Name of the experiment
            results: Experimental results
            motivation: Design motivation
            reference_context: Related experiments for ablation

        Returns:
            Rendered prompt string
        """
        if not self.spec.prompts.analyzer.startswith("[PLACEHOLDER"):
            return self.spec.prompts.analyzer.format(
                experiment_name=experiment_name,
                results=results,
                motivation=motivation,
                reference_context=reference_context
            )

        vars = self._get_common_variables()
        vars["experiment_name"] = experiment_name
        vars["results"] = results
        vars["motivation"] = motivation
        vars["reference_context"] = reference_context

        from .base_templates import ANALYZER_TEMPLATE
        return ANALYZER_TEMPLATE.format(**vars)

    def render_summarizer(
        self,
        motivation: str,
        analysis: str,
        cognition: str
    ) -> str:
        """
        Render the summarizer prompt.

        Args:
            motivation: Design motivation
            analysis: Analysis from analyzer
            cognition: Retrieved research papers

        Returns:
            Rendered prompt string
        """
        if not self.spec.prompts.summarizer.startswith("[PLACEHOLDER"):
            return self.spec.prompts.summarizer.format(
                motivation=motivation,
                analysis=analysis,
                cognition=cognition
            )

        vars = self._get_common_variables()
        vars["motivation"] = motivation
        vars["analysis"] = analysis
        vars["cognition"] = cognition

        from .base_templates import SUMMARIZER_TEMPLATE
        return SUMMARIZER_TEMPLATE.format(**vars)

    def render_model_judger(
        self,
        model_name: str,
        model_code: str,
        motivation: str,
        training_results: str,
        evaluation_results: str
    ) -> str:
        """
        Render the model judger prompt.

        Args:
            model_name: Name of the model to evaluate
            model_code: Model implementation code
            motivation: Design motivation
            training_results: Training performance data
            evaluation_results: Evaluation/benchmark results

        Returns:
            Rendered prompt string
        """
        if not self.spec.prompts.model_judger.startswith("[PLACEHOLDER"):
            return self.spec.prompts.model_judger.format(
                model_name=model_name,
                model_code=model_code,
                motivation=motivation,
                training_results=training_results,
                evaluation_results=evaluation_results
            )

        vars = self._get_common_variables()
        vars["model_name"] = model_name
        vars["model_code"] = model_code
        vars["motivation"] = motivation
        vars["training_results"] = training_results
        vars["evaluation_results"] = evaluation_results

        # Format baseline models section
        vars["baseline_models_section"] = self._format_baseline_models(
            self.spec.evaluation.baseline_models
        )

        # Format scoring criteria
        weights = self.spec.evaluation.scoring_weights
        criteria_lines = []
        for criterion, weight in weights.items():
            criteria_lines.append(f"### {criterion.replace('_', ' ').title()} ({weight*100:.0f}% weight)")
            criteria_lines.append(f"Evaluate the {criterion.lower()} of the implementation.")
            criteria_lines.append("")

        vars["scoring_criteria"] = "\n".join(criteria_lines)

        from .base_templates import MODEL_JUDGER_TEMPLATE
        return MODEL_JUDGER_TEMPLATE.format(**vars)

    def clear_cache(self):
        """Clear the rendered prompt cache."""
        self._cache.clear()
