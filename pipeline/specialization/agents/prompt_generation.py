"""
Prompt Generation Agent

Generates truly domain-specific prompts for the research pipeline.
Uses deep codebase understanding to create prompts that:
1. Know what the code DOES
2. Understand what should EVOLVE
3. Have domain-specific expertise
4. Guide meaningful improvements
"""

import json
from typing import Dict, List, Any, Optional

from pydantic import BaseModel

from ..schema import PromptTemplates


class PromptGenerationOutput(BaseModel):
    """Output from the Prompt Generation Agent."""
    planner: str
    checker: str
    motivation_checker: str
    deduplication: str
    debugger: str
    analyzer: str
    summarizer: str
    model_judger: str


PROMPT_GENERATION_SYSTEM = """You are an expert at creating prompts for AI agents that evolve and improve codebases.

Your prompts must be:
1. SPECIFIC to the actual codebase - reference real files, classes, functions
2. DOMAIN-EXPERT - use the correct terminology and concepts for this domain
3. EVOLUTION-FOCUSED - understand WHAT to improve and HOW
4. PRACTICAL - include concrete guidance, not abstract principles
5. COMPLETE - include all necessary context, constraints, and expected outputs

The prompts you generate will be used by AI agents to:
- Design improvements to specific code components
- Validate that changes work correctly
- Detect duplicate improvement approaches
- Generate innovative variations
- Debug failures in evolved code
- Analyze experimental results
- Synthesize insights for future improvements
- Score and rank evolved versions

CRITICAL: These prompts must understand the ACTUAL codebase - what it does, how it's structured, and what meaningful improvements look like for THIS specific code."""


PROMPT_GENERATION_TASK = """# Domain-Specific Prompt Generation Task

## Codebase Overview
**Name:** {domain_name}
**Purpose:** {codebase_purpose}
**Summary:** {codebase_summary}

## Architecture
{architecture_info}

## Evolution Targets
These are the specific components that should be evolved:
{evolution_targets}

## Evolution Strategies
These are the proposed approaches for improving the code:
{evolution_strategies}

## Domain Expertise Required
**Primary Domain:** {primary_domain}
**Key Concepts:** {key_concepts}
**Best Practices:** {best_practices}
**Common Pitfalls to Avoid:** {pitfalls}

## Constraints
{constraints}

## Your Task
Generate 8 specialized prompts for evolving this codebase. Each prompt MUST:
- Reference the ACTUAL code structure (real file names, class names, function names)
- Use the correct domain terminology
- Focus on the identified evolution targets
- Apply the proposed evolution strategies
- Respect the constraints

### 1. PLANNER Prompt
The planner designs improvements to the codebase. The prompt should:
- Guide analysis of what improvements would be valuable
- Reference specific evolution targets by name
- Suggest approaches from the evolution strategies
- Require implementation details, not vague ideas
- Include the context placeholder: {{context}}

Example focus areas for this codebase:
{planner_focus}

### 2. CHECKER Prompt
The checker validates that evolved code works correctly. The prompt should:
- List specific validation checks for THIS codebase
- Verify that evolution targets maintain their core functionality
- Check domain-specific correctness criteria
- Prioritize by severity
- Use the placeholder: {{motivation}}

Checks should verify:
{checker_focus}

### 3. MOTIVATION_CHECKER Prompt
Detects when an improvement approach duplicates previous attempts. The prompt should:
- Define what constitutes duplication for THIS codebase's evolution
- Consider the specific evolution strategies being used
- Be lenient - similar approaches with different implementations are valid
- Use placeholders: {{motivation}}, {{historical_context}}

### 4. DEDUPLICATION Prompt
When improvement patterns are over-used, generate orthogonal approaches. The prompt should:
- Identify alternative evolution strategies
- Suggest unexplored improvement directions for this codebase
- Maintain compatibility with existing architecture
- Use placeholders: {{repeated_motivation}}, {{historical_context}}

### 5. DEBUGGER Prompt
Fixes failures in evolved code. The prompt should:
- Categorize common error types for THIS codebase
- Reference specific files/modules where errors might occur
- Preserve the improvement intent while fixing
- Use placeholders: {{motivation}}, {{error_log}}

Common issues in this codebase:
{debugger_focus}

### 6. ANALYZER Prompt
Analyzes experimental results from evolved code. The prompt should:
- Structure analysis around the evolution targets
- Compare against the identified success criteria
- Understand WHY changes helped or hurt
- Use placeholders: {{experiment_name}}, {{results}}, {{motivation}}, {{reference_context}}

### 7. SUMMARIZER Prompt
Synthesizes insights for future improvements. The prompt should:
- Extract patterns about what works for THIS codebase
- Connect to the domain expertise
- Guide future evolution directions
- Use placeholders: {{motivation}}, {{analysis}}, {{cognition}}

### 8. MODEL_JUDGER Prompt
Scores and ranks evolved versions. The prompt should:
- Define scoring criteria specific to this codebase's goals
- Weight factors appropriately for the domain
- Use placeholders: {{model_name}}, {{model_code}}, {{motivation}}, {{training_results}}, {{evaluation_results}}

Scoring should consider:
{judger_focus}

## Output Format
Provide each prompt as a complete, ready-to-use template with the specified placeholders.
Format as JSON with keys: planner, checker, motivation_checker, deduplication, debugger, analyzer, summarizer, model_judger

IMPORTANT: Make prompts SPECIFIC to this codebase. Do NOT generate generic prompts that could apply to any code."""


class PromptGenerationAgent:
    """
    Agent that generates domain-specific prompts for all pipeline stages.
    Uses deep codebase understanding to create meaningful, specific prompts.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    async def generate(
        self,
        domain_name: str,
        domain_description: str,
        domain_understanding: Dict[str, Any],
        constraints: Dict[str, Any],
        knowledge_corpus: Optional[Dict[str, Any]] = None
    ) -> PromptTemplates:
        """
        Generate all 8 prompts for a domain based on deep codebase understanding.

        Args:
            domain_name: Name of the domain
            domain_description: Full description
            domain_understanding: Output from DomainUnderstandingAgent (new format)
            constraints: Output from ConstraintDefinitionAgent
            knowledge_corpus: Optional knowledge context

        Returns:
            PromptTemplates with all 8 domain-specific prompts
        """
        # Extract new-format understanding if available
        codebase_purpose = domain_understanding.get("codebase_purpose", domain_description)
        codebase_summary = domain_understanding.get("codebase_summary", "")

        # Get structure info
        structure = domain_understanding.get("structure", {})
        architecture_info = self._format_architecture(structure)

        # Get evolution targets
        evolution_targets = domain_understanding.get("evolution_targets", [])
        evolution_targets_str = self._format_evolution_targets(evolution_targets)

        # Get evolution strategies
        strategies = domain_understanding.get("proposed_strategies", [])
        strategies_str = self._format_strategies(strategies)

        # Get domain expertise
        expertise = domain_understanding.get("domain_expertise", {})
        primary_domain = expertise.get("primary_domain", "software engineering")
        key_concepts = expertise.get("key_concepts", [])
        best_practices = expertise.get("best_practices", [])
        pitfalls = expertise.get("common_pitfalls", [])

        # Generate focus areas for specific prompts
        planner_focus = self._generate_planner_focus(evolution_targets, strategies)
        checker_focus = self._generate_checker_focus(evolution_targets, expertise)
        debugger_focus = self._generate_debugger_focus(evolution_targets, structure)
        judger_focus = self._generate_judger_focus(strategies, expertise)

        prompt = PROMPT_GENERATION_TASK.format(
            domain_name=domain_name,
            codebase_purpose=codebase_purpose,
            codebase_summary=codebase_summary,
            architecture_info=architecture_info,
            evolution_targets=evolution_targets_str,
            evolution_strategies=strategies_str,
            primary_domain=primary_domain,
            key_concepts=", ".join(key_concepts[:10]) if key_concepts else "General software concepts",
            best_practices="\n".join(f"- {p}" for p in best_practices[:5]) if best_practices else "- Follow standard practices",
            pitfalls="\n".join(f"- {p}" for p in pitfalls[:5]) if pitfalls else "- Avoid common mistakes",
            constraints=json.dumps(constraints, indent=2) if constraints else "No specific constraints",
            planner_focus=planner_focus,
            checker_focus=checker_focus,
            debugger_focus=debugger_focus,
            judger_focus=judger_focus
        )

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PROMPT_GENERATION_SYSTEM + "\n\nYou MUST respond with valid JSON only - no markdown, no explanations."},
                    {"role": "user", "content": prompt + "\n\nIMPORTANT: Output ONLY the JSON object, no markdown code blocks."}
                ],
                max_tokens=8000
            )

            # Extract JSON from response (handle markdown blocks)
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                start_idx = 1 if lines[0].startswith("```") else 0
                end_idx = len(lines)
                for i, line in enumerate(lines[1:], 1):
                    if line.startswith("```"):
                        end_idx = i
                        break
                content = "\n".join(lines[start_idx:end_idx])

            data = json.loads(content)

            # Helper to extract string from potentially nested structure
            def extract_prompt(value, default):
                if value is None:
                    return default
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    # LLM might return {"prompt": "...", "description": "...", "context": "..."}
                    # Try common keys first
                    for key in ['prompt', 'content', 'template', 'text', 'value', 'description']:
                        if key in value and isinstance(value[key], str):
                            result = value[key]
                            # If there's also a context placeholder, append it
                            if 'context' in value and value['context'] and key != 'context':
                                ctx = value['context']
                                if isinstance(ctx, str) and '{' in ctx:
                                    result += f"\n\n## Context\n{ctx}"
                            return result
                    # If dict has description + context structure, build a prompt
                    if 'description' in value or 'instructions' in value:
                        parts = []
                        for k in ['instructions', 'description', 'task', 'goal']:
                            if k in value and value[k]:
                                parts.append(str(value[k]))
                        if 'context' in value:
                            parts.append(f"\n## Context\n{value['context']}")
                        if parts:
                            return "\n\n".join(parts)
                    # Last resort: convert dict to readable format
                    return "\n".join(f"{k}: {v}" for k, v in value.items() if v)
                return str(value)

            return PromptTemplates(
                planner=extract_prompt(data.get("planner"), self._default_planner(domain_name, evolution_targets)),
                checker=extract_prompt(data.get("checker"), self._default_checker(domain_name)),
                motivation_checker=extract_prompt(data.get("motivation_checker"), self._default_motivation_checker(domain_name)),
                deduplication=extract_prompt(data.get("deduplication"), self._default_deduplication(domain_name)),
                debugger=extract_prompt(data.get("debugger"), self._default_debugger(domain_name)),
                analyzer=extract_prompt(data.get("analyzer"), self._default_analyzer(domain_name)),
                summarizer=extract_prompt(data.get("summarizer"), self._default_summarizer(domain_name)),
                model_judger=extract_prompt(data.get("model_judger"), self._default_judger(domain_name))
            )

        except Exception as e:
            print(f"  Warning: Prompt generation failed: {e}", flush=True)
            print(f"  Using intelligent defaults based on domain understanding.", flush=True)
            return self._smart_default_prompts(domain_name, domain_understanding)

    def _format_architecture(self, structure: Dict) -> str:
        """Format architecture information from codebase structure."""
        if not structure:
            return "Architecture: Not analyzed"

        lines = []

        if structure.get("architecture_pattern"):
            lines.append(f"**Pattern:** {structure['architecture_pattern']}")

        if structure.get("entry_points"):
            lines.append(f"**Entry Points:** {', '.join(structure['entry_points'][:5])}")

        if structure.get("core_modules"):
            lines.append(f"**Core Modules:** {', '.join(structure['core_modules'][:5])}")

        if structure.get("dependency_graph"):
            deps = structure["dependency_graph"]
            if deps:
                sample = list(deps.items())[:3]
                deps_str = "; ".join(f"{k} -> {', '.join(v[:3])}" for k, v in sample)
                lines.append(f"**Key Dependencies:** {deps_str}")

        return "\n".join(lines) if lines else "Architecture: Standard structure"

    def _format_evolution_targets(self, targets: List[Dict]) -> str:
        """Format evolution targets for the prompt."""
        if not targets:
            return "No specific evolution targets identified - general improvement applies"

        lines = []
        for i, target in enumerate(targets[:5], 1):  # Top 5 targets
            lines.append(f"""
**Target {i}: {target.get('component_name', 'Unknown')}**
- File: {target.get('file_path', 'Unknown')}
- Type: {target.get('component_type', 'unknown')}
- Current: {target.get('current_behavior', 'Not specified')}
- Why evolve: {target.get('evolution_rationale', 'General improvement')}
- Constraints: {', '.join(target.get('constraints', ['None specified']))}
- Suggested approaches: {', '.join(target.get('suggested_approaches', ['General optimization']))}
""")
        return "\n".join(lines)

    def _format_strategies(self, strategies: List[Dict]) -> str:
        """Format evolution strategies for the prompt."""
        if not strategies:
            return "No specific strategies - use general improvement approaches"

        lines = []
        for strategy in strategies[:4]:  # Top 4 strategies
            lines.append(f"""
**{strategy.get('name', 'Strategy')}**
- Description: {strategy.get('description', 'Not specified')}
- Targets: {', '.join(strategy.get('target_improvements', ['General']))}
- Requires: {', '.join(strategy.get('required_expertise', ['Software engineering']))}
- Success criteria: {', '.join(strategy.get('evaluation_criteria', ['Improvement']))}
- Risks: {', '.join(strategy.get('risk_factors', ['None identified']))}
""")
        return "\n".join(lines)

    def _generate_planner_focus(self, targets: List[Dict], strategies: List[Dict]) -> str:
        """Generate focus areas for the planner prompt."""
        focus = []

        if targets:
            focus.append("Specific components to improve:")
            for t in targets[:3]:
                focus.append(f"  - {t.get('component_name', 'component')}: {t.get('evolution_rationale', 'general improvement')}")

        if strategies:
            focus.append("\nApproaches to consider:")
            for s in strategies[:3]:
                focus.append(f"  - {s.get('name', 'Strategy')}: {s.get('description', '')[:100]}")

        return "\n".join(focus) if focus else "General code improvement"

    def _generate_checker_focus(self, targets: List[Dict], expertise: Dict) -> str:
        """Generate focus areas for the checker prompt."""
        focus = []

        if targets:
            focus.append("- Core functionality of evolution targets is preserved")
            for t in targets[:3]:
                constraints = t.get('constraints', [])
                if constraints:
                    focus.append(f"- {t.get('component_name', 'Component')}: {', '.join(constraints[:2])}")

        if expertise.get("best_practices"):
            focus.append("- Domain best practices are followed")

        focus.append("- No regressions introduced")
        focus.append("- Code compiles/runs without errors")

        return "\n".join(focus)

    def _generate_debugger_focus(self, targets: List[Dict], structure: Dict) -> str:
        """Generate focus areas for the debugger prompt."""
        focus = []

        if structure.get("entry_points"):
            focus.append(f"- Entry point issues: {', '.join(structure['entry_points'][:2])}")

        if targets:
            focus.append("- Evolution target failures:")
            for t in targets[:2]:
                focus.append(f"  - {t.get('file_path', 'file')}: {t.get('component_name', 'component')}")

        focus.append("- Import/dependency errors")
        focus.append("- Type mismatches")
        focus.append("- Logic errors in evolved code")

        return "\n".join(focus)

    def _generate_judger_focus(self, strategies: List[Dict], expertise: Dict) -> str:
        """Generate focus areas for the judger prompt."""
        focus = []

        if strategies:
            for s in strategies[:2]:
                criteria = s.get("evaluation_criteria", [])
                if criteria:
                    focus.append(f"- {s.get('name', 'Strategy')} success: {', '.join(criteria[:2])}")

        focus.append("- Code quality improvement")
        focus.append("- Performance (if applicable)")
        focus.append("- Innovation value")
        focus.append("- Maintainability")

        return "\n".join(focus)

    def _smart_default_prompts(self, domain_name: str, understanding: Dict) -> PromptTemplates:
        """Generate intelligent default prompts using domain understanding."""
        codebase_purpose = understanding.get("codebase_purpose", f"{domain_name} codebase")
        targets = understanding.get("evolution_targets", [])
        strategies = understanding.get("proposed_strategies", [])
        expertise = understanding.get("domain_expertise", {})

        target_names = [t.get("component_name", "component") for t in targets[:3]]
        strategy_names = [s.get("name", "improvement") for s in strategies[:3]]

        return PromptTemplates(
            planner=self._default_planner(domain_name, targets),
            checker=self._default_checker(domain_name),
            motivation_checker=self._default_motivation_checker(domain_name),
            deduplication=self._default_deduplication(domain_name),
            debugger=self._default_debugger(domain_name),
            analyzer=self._default_analyzer(domain_name),
            summarizer=self._default_summarizer(domain_name),
            model_judger=self._default_judger(domain_name)
        )

    def _default_planner(self, domain_name: str, targets: List[Dict]) -> str:
        """Generate default planner prompt."""
        target_section = ""
        if targets:
            target_section = "\n\nEvolution Targets:\n"
            for t in targets[:3]:
                target_section += f"- {t.get('component_name', 'Component')}: {t.get('evolution_rationale', 'Improve')}\n"

        return f"""# {domain_name} Evolution Planner

You are an expert at improving codebases. Your task is to design a meaningful improvement to the code.

## Context
{{context}}
{target_section}

## Your Task
1. Analyze the current state of the codebase
2. Identify a specific, valuable improvement
3. Design the implementation in detail
4. Explain why this improvement matters

## Requirements
- Be SPECIFIC - reference actual files, functions, classes
- Be PRACTICAL - ensure the change can be implemented
- Be VALUABLE - focus on meaningful improvements, not trivial changes

## Output
Provide:
1. Name for this improvement
2. Motivation (why is this valuable?)
3. Implementation plan (what exactly changes?)
4. Expected impact (what improves?)"""

    def _default_checker(self, domain_name: str) -> str:
        """Generate default checker prompt."""
        return f"""# {domain_name} Code Validator

Validate that the evolved code works correctly.

## Proposed Change
{{motivation}}

## Validation Checklist
1. **Syntax**: Does the code parse without errors?
2. **Imports**: Are all dependencies available?
3. **Functionality**: Does the core behavior still work?
4. **No Regressions**: Are existing features preserved?
5. **Code Quality**: Is the code clean and maintainable?

## Output
For each check, report:
- PASS/FAIL
- Details if FAIL

If any critical checks fail, provide specific fix guidance."""

    def _default_motivation_checker(self, domain_name: str) -> str:
        """Generate default motivation checker prompt."""
        return f"""# {domain_name} Duplication Detector

Check if this improvement approach has already been tried.

## Proposed Improvement
{{motivation}}

## Historical Context
{{historical_context}}

## Task
Compare the proposed improvement to previous attempts:
1. Is this the SAME approach as something already tried?
2. Is it a VARIATION that's meaningfully different?
3. Is it completely NEW?

Be LENIENT - similar ideas with different implementations are valid.
Only flag TRUE duplicates where the approach AND implementation are the same."""

    def _default_deduplication(self, domain_name: str) -> str:
        """Generate default deduplication prompt."""
        return f"""# {domain_name} Innovation Generator

The following approach has been tried multiple times:
{{repeated_motivation}}

## Historical Context
{{historical_context}}

## Task
Generate an ORTHOGONAL improvement approach:
1. What different angle could be explored?
2. What hasn't been tried yet?
3. What cross-disciplinary ideas might apply?

Ensure the new approach:
- Is meaningfully different from previous attempts
- Maintains compatibility with the codebase
- Has potential for real improvement"""

    def _default_debugger(self, domain_name: str) -> str:
        """Generate default debugger prompt."""
        return f"""# {domain_name} Debug Assistant

Fix the error in the evolved code while preserving the improvement intent.

## Attempted Improvement
{{motivation}}

## Error Log
{{error_log}}

## Task
1. Identify the root cause of the error
2. Determine the minimal fix
3. Ensure the improvement intent is preserved
4. Provide the corrected code

Common issues to check:
- Import errors
- Type mismatches
- Missing dependencies
- Logic errors in the new code"""

    def _default_analyzer(self, domain_name: str) -> str:
        """Generate default analyzer prompt."""
        return f"""# {domain_name} Results Analyzer

Analyze the experimental results from this improvement.

## Experiment
Name: {{experiment_name}}
Motivation: {{motivation}}

## Results
{{results}}

## Reference Context
{{reference_context}}

## Analysis Tasks
1. **Performance**: How do the results compare?
2. **Impact**: What changed and why?
3. **Insights**: What can we learn from this?
4. **Recommendations**: What should be tried next?

Provide detailed analysis focusing on understanding WHY, not just WHAT."""

    def _default_summarizer(self, domain_name: str) -> str:
        """Generate default summarizer prompt."""
        return f"""# {domain_name} Insight Synthesizer

Synthesize insights from this experiment for future improvements.

## Improvement Attempted
{{motivation}}

## Analysis
{{analysis}}

## Cognition
{{cognition}}

## Task
Extract actionable insights:
1. What worked well?
2. What didn't work?
3. What patterns emerged?
4. What should future improvements consider?

Focus on practical, actionable takeaways."""

    def _default_judger(self, domain_name: str) -> str:
        """Generate default judger prompt."""
        return f"""# {domain_name} Improvement Scorer

Score this evolved version of the codebase.

## Model
Name: {{model_name}}
Motivation: {{motivation}}

## Code
{{model_code}}

## Results
Training: {{training_results}}
Evaluation: {{evaluation_results}}

## Scoring Criteria
Rate 0-100 on each:
1. **Impact** (40%): How significant is the improvement?
2. **Quality** (30%): How clean and maintainable is the code?
3. **Innovation** (20%): How novel is the approach?
4. **Reliability** (10%): Does it work consistently?

Provide:
- Individual scores with justification
- Overall weighted score
- Brief recommendation"""

    async def refine_prompt(
        self,
        prompt_name: str,
        current_prompt: str,
        feedback: str,
        domain_context: Dict[str, Any]
    ) -> str:
        """
        Refine a specific prompt based on feedback.

        Args:
            prompt_name: Which prompt to refine (e.g., "planner")
            current_prompt: Current version of the prompt
            feedback: What needs to be improved
            domain_context: Domain information

        Returns:
            Refined prompt string
        """
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You refine AI agent prompts based on feedback. Maintain all placeholders and make prompts more specific to the domain."
                    },
                    {
                        "role": "user",
                        "content": f"""Refine this {prompt_name} prompt based on the feedback.

Current prompt:
{current_prompt}

Feedback:
{feedback}

Domain context:
{json.dumps(domain_context, indent=2)}

Provide the improved prompt, keeping all {{placeholders}} intact."""
                    }
                ]
            )

            return response.choices[0].message.content

        except Exception:
            return current_prompt
