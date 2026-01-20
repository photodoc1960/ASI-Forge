"""
Constraint Definition Agent

Generates domain-specific constraints based on deep codebase analysis.
Uses evolution targets and strategies to define what must be preserved,
what should be checked, and what rules govern evolution.
"""

import json
from typing import Dict, List, Optional, Any

from pydantic import BaseModel

from ..schema import Constraint, ConstraintSeverity


class ConstraintDefinition(BaseModel):
    """Output from the Constraint Definition Agent."""
    complexity_requirement: Optional[str]
    strict_constraints: List[Dict[str, Any]]
    critical_constraints: List[Dict[str, Any]]
    flexible_constraints: List[Dict[str, Any]]
    preservation_rules: List[str]


CONSTRAINT_DEFINITION_PROMPT = """# Constraint Definition Task

You need to define validation constraints for evolving a codebase.

## Codebase Overview
**Name:** {domain_name}
**Purpose:** {codebase_purpose}

## Evolution Targets
These are the components being evolved:
{evolution_targets}

## Evolution Strategies
The approaches being used:
{evolution_strategies}

## Domain Expertise
**Domain:** {primary_domain}
**Best Practices:** {best_practices}
**Common Pitfalls:** {pitfalls}

## Your Task
Define validation constraints that ensure evolved code is correct and high-quality.

## Constraint Categories

### 1. STRICT Constraints (Must-Fix)
These are non-negotiable. If violated, the code MUST be fixed.
Base these on:
- Constraints from evolution targets (what must be preserved)
- Core functionality that cannot break
- Safety/correctness requirements

### 2. CRITICAL Constraints (Should-Fix)
These are important for quality. Violations should usually be fixed.
Base these on:
- Domain best practices
- Evaluation criteria from strategies
- Interface requirements

### 3. FLEXIBLE Constraints (Nice-to-Have)
These improve quality but shouldn't block innovation.
Base these on:
- Style preferences
- Optional optimizations
- Documentation requirements

### 4. Preservation Rules
What must NOT change during evolution?
- Extract from evolution target constraints
- Interface signatures to maintain
- Dependencies that must remain stable

## Output Format
For each constraint, provide:
- name: Short descriptive name
- description: Full explanation
- severity: strict/critical/flexible
- validation_prompt: How to check this constraint
- fix_guidance: How to fix violations

Provide JSON with: complexity_requirement, strict_constraints, critical_constraints, flexible_constraints, preservation_rules"""


class ConstraintDefinitionAgent:
    """
    Agent that generates domain-specific validation constraints
    based on deep codebase understanding.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    async def define(
        self,
        domain_name: str,
        domain_understanding: Dict[str, Any],
        seed_patterns: Optional[Dict[str, str]] = None
    ) -> ConstraintDefinition:
        """
        Define constraints for a domain based on codebase understanding.

        Args:
            domain_name: Name of the domain
            domain_understanding: Output from DomainUnderstandingAgent (new format)
            seed_patterns: Patterns extracted from seed codebase (legacy)

        Returns:
            ConstraintDefinition with all constraint categories
        """
        # Extract new format understanding
        codebase_purpose = domain_understanding.get("codebase_purpose", domain_name)
        evolution_targets = domain_understanding.get("evolution_targets", [])
        strategies = domain_understanding.get("proposed_strategies", [])
        expertise = domain_understanding.get("domain_expertise", {})

        # Format for prompt
        targets_str = self._format_evolution_targets(evolution_targets)
        strategies_str = self._format_strategies(strategies)

        primary_domain = expertise.get("primary_domain", "software engineering")
        best_practices = expertise.get("best_practices", [])
        pitfalls = expertise.get("common_pitfalls", [])

        prompt = CONSTRAINT_DEFINITION_PROMPT.format(
            domain_name=domain_name,
            codebase_purpose=codebase_purpose,
            evolution_targets=targets_str,
            evolution_strategies=strategies_str,
            primary_domain=primary_domain,
            best_practices="\n".join(f"- {p}" for p in best_practices[:5]) if best_practices else "- Standard practices",
            pitfalls="\n".join(f"- {p}" for p in pitfalls[:5]) if pitfalls else "- Common mistakes"
        )

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You define validation constraints for code evolution. You MUST respond with valid JSON only - no markdown, no explanations."
                    },
                    {"role": "user", "content": prompt + "\n\nIMPORTANT: Output ONLY the JSON object, no markdown code blocks."}
                ],
                max_tokens=4000
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

            # Handle preservation_rules being returned as dicts instead of strings
            if 'preservation_rules' in data and data['preservation_rules']:
                fixed_rules = []
                for rule in data['preservation_rules']:
                    if isinstance(rule, dict):
                        # Extract the name or description from the dict
                        fixed_rules.append(rule.get('name', rule.get('description', str(rule))))
                    else:
                        fixed_rules.append(str(rule))
                data['preservation_rules'] = fixed_rules

            # Handle complexity_requirement being returned as dict instead of string
            if 'complexity_requirement' in data and data['complexity_requirement']:
                cr = data['complexity_requirement']
                if isinstance(cr, dict):
                    # Extract meaningful text from the dict
                    if 'description' in cr:
                        data['complexity_requirement'] = cr['description']
                    elif 'desired_level' in cr:
                        data['complexity_requirement'] = f"{cr['desired_level']}: {cr.get('rationale', '')}"
                    else:
                        data['complexity_requirement'] = str(cr)
                elif not isinstance(cr, str):
                    data['complexity_requirement'] = str(cr)

            return ConstraintDefinition(**data)

        except Exception as e:
            print(f"  Warning: Constraint generation failed: {e}", flush=True)
            return self._smart_default_constraints(domain_understanding)

    def _format_evolution_targets(self, targets: List[Dict]) -> str:
        """Format evolution targets for the prompt."""
        if not targets:
            return "No specific targets - general improvement"

        lines = []
        for t in targets[:5]:
            lines.append(f"**{t.get('component_name', 'Component')}** ({t.get('file_path', 'unknown')})")
            lines.append(f"  - Current: {t.get('current_behavior', 'Not specified')}")
            constraints = t.get('constraints', [])
            if constraints:
                lines.append(f"  - Must preserve: {', '.join(constraints[:3])}")

        return "\n".join(lines)

    def _format_strategies(self, strategies: List[Dict]) -> str:
        """Format strategies for the prompt."""
        if not strategies:
            return "General code improvement"

        lines = []
        for s in strategies[:4]:
            lines.append(f"**{s.get('name', 'Strategy')}**")
            criteria = s.get("evaluation_criteria", [])
            if criteria:
                lines.append(f"  - Success criteria: {', '.join(criteria[:3])}")
            risks = s.get("risk_factors", [])
            if risks:
                lines.append(f"  - Risks to check: {', '.join(risks[:2])}")

        return "\n".join(lines)

    def _smart_default_constraints(self, understanding: Dict[str, Any]) -> ConstraintDefinition:
        """Generate intelligent default constraints from domain understanding."""
        evolution_targets = understanding.get("evolution_targets", [])
        strategies = understanding.get("proposed_strategies", [])
        expertise = understanding.get("domain_expertise", {})

        # Build strict constraints from evolution target constraints
        strict_constraints = []
        preservation_rules = []

        for target in evolution_targets[:3]:
            constraints = target.get("constraints", [])
            for c in constraints[:2]:
                strict_constraints.append({
                    "name": f"Preserve {target.get('component_name', 'Component')} {c[:20]}",
                    "description": f"The {target.get('component_name', 'component')} must maintain: {c}",
                    "severity": "strict",
                    "validation_prompt": f"Verify that {c} is still true",
                    "fix_guidance": f"Restore compliance with: {c}"
                })

            # Add to preservation rules
            component = target.get("component_name", "")
            if component:
                preservation_rules.append(f"Maintain {component} core functionality")

        # Add default strict constraint if none from targets
        if not strict_constraints:
            strict_constraints.append({
                "name": "Code Correctness",
                "description": "Code must produce correct outputs and not introduce bugs",
                "severity": "strict",
                "validation_prompt": "Verify the code logic is correct and handles edge cases",
                "fix_guidance": "Fix logical errors while preserving the improvement intent"
            })

        # Build critical constraints from strategies
        critical_constraints = []
        for strategy in strategies[:2]:
            criteria = strategy.get("evaluation_criteria", [])
            for c in criteria[:2]:
                critical_constraints.append({
                    "name": f"Achieve {c[:30]}",
                    "description": f"The evolved code should demonstrate: {c}",
                    "severity": "critical",
                    "validation_prompt": f"Check if the change achieves: {c}",
                    "fix_guidance": f"Adjust implementation to better achieve: {c}"
                })

        if not critical_constraints:
            critical_constraints.append({
                "name": "Code Quality",
                "description": "Code should be clean, readable, and maintainable",
                "severity": "critical",
                "validation_prompt": "Check code quality and readability",
                "fix_guidance": "Improve code organization and clarity"
            })

        # Build flexible constraints from best practices
        flexible_constraints = []
        best_practices = expertise.get("best_practices", [])
        for practice in best_practices[:2]:
            flexible_constraints.append({
                "name": f"Follow {practice[:25]}",
                "description": f"When possible, adhere to: {practice}",
                "severity": "flexible",
                "validation_prompt": f"Check if code follows: {practice}",
                "fix_guidance": f"Consider applying: {practice}"
            })

        if not flexible_constraints:
            flexible_constraints.append({
                "name": "Documentation",
                "description": "Important changes should be documented",
                "severity": "flexible",
                "validation_prompt": "Check if changes are adequately documented",
                "fix_guidance": "Add comments explaining non-obvious changes"
            })

        # Default preservation rules
        if not preservation_rules:
            preservation_rules = [
                "Maintain core functionality",
                "Preserve public interfaces",
                "Keep backward compatibility where possible"
            ]

        # Complexity requirement from strategies
        complexity_req = None
        for strategy in strategies:
            risks = strategy.get("risk_factors", [])
            for risk in risks:
                if "complex" in risk.lower() or "performance" in risk.lower():
                    complexity_req = f"Monitor for: {risk}"
                    break
            if complexity_req:
                break

        return ConstraintDefinition(
            complexity_requirement=complexity_req,
            strict_constraints=strict_constraints,
            critical_constraints=critical_constraints,
            flexible_constraints=flexible_constraints,
            preservation_rules=preservation_rules
        )

    def to_schema_constraints(
        self,
        definition: ConstraintDefinition
    ) -> tuple[list[Constraint], list[Constraint], list[Constraint]]:
        """Convert ConstraintDefinition to schema Constraint objects."""
        def convert(constraints: List[Dict], severity: ConstraintSeverity) -> List[Constraint]:
            result = []
            for c in constraints:
                result.append(Constraint(
                    name=c.get("name", "Unnamed"),
                    description=c.get("description", ""),
                    severity=severity,
                    validation_prompt=c.get("validation_prompt", ""),
                    fix_guidance=c.get("fix_guidance", ""),
                    examples=c.get("examples")
                ))
            return result

        strict = convert(definition.strict_constraints, ConstraintSeverity.STRICT)
        critical = convert(definition.critical_constraints, ConstraintSeverity.CRITICAL)
        flexible = convert(definition.flexible_constraints, ConstraintSeverity.FLEXIBLE)

        return strict, critical, flexible
