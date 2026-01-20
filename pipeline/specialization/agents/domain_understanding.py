"""
Domain Understanding Agent

Deeply analyzes any codebase to understand:
1. What the code DOES (purpose, functionality)
2. How it's structured (architecture, patterns)
3. What could be evolved (evolution targets)
4. How it should evolve (evolution goals/strategies)
5. What domain expertise is needed

This agent is domain-agnostic and works with ANY codebase.
"""

import json
import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from pydantic import BaseModel, Field


class FileAnalysis(BaseModel):
    """Analysis of a single file."""
    path: str
    purpose: str  # What this file does
    key_components: List[str]  # Classes, functions, etc.
    dependencies: List[str]  # Imports
    complexity: str  # low/medium/high
    evolution_potential: str  # Why this file could be evolved


class CodebaseStructure(BaseModel):
    """Deep analysis of codebase structure."""
    entry_points: List[str]  # Main execution entry points
    core_modules: List[str]  # Central/important modules
    utility_modules: List[str]  # Helper/utility modules
    test_modules: List[str]  # Testing infrastructure
    config_files: List[str]  # Configuration files
    data_files: List[str]  # Data/resource files
    architecture_pattern: str  # e.g., "MVC", "pipeline", "microservices", "monolithic"
    dependency_graph: Dict[str, List[str]]  # module -> [dependencies]


class EvolutionTarget(BaseModel):
    """A specific part of the code that can be evolved."""
    file_path: str
    component_name: str  # Class/function name
    component_type: str  # "class", "function", "module"
    current_behavior: str  # What it currently does
    evolution_rationale: str  # Why evolving this would be valuable
    constraints: List[str]  # What must be preserved
    suggested_approaches: List[str]  # Ideas for evolution


class EvolutionStrategy(BaseModel):
    """A proposed strategy for how the code should evolve."""
    name: str
    description: str
    target_improvements: List[str]  # What this strategy aims to improve
    required_expertise: List[str]  # Domain knowledge needed
    evaluation_criteria: List[str]  # How to measure success
    risk_factors: List[str]  # Potential issues to watch for


class DomainExpertise(BaseModel):
    """Domain-specific knowledge required for this codebase."""
    primary_domain: str  # Main domain (e.g., "machine learning", "web scraping", "game AI")
    sub_domains: List[str]  # Related areas
    key_concepts: List[str]  # Important concepts to understand
    terminology: Dict[str, str]  # Term -> definition
    best_practices: List[str]  # Domain best practices
    common_pitfalls: List[str]  # Things to avoid
    reference_resources: List[str]  # Types of resources that would help


class DomainUnderstanding(BaseModel):
    """Complete understanding of a codebase for evolution."""
    # Core understanding
    codebase_purpose: str  # High-level: what does this code DO?
    codebase_summary: str  # Detailed summary of functionality

    # Structure analysis
    structure: CodebaseStructure
    file_analyses: List[FileAnalysis]

    # Evolution planning
    evolution_targets: List[EvolutionTarget]
    proposed_strategies: List[EvolutionStrategy]

    # Domain knowledge
    domain_expertise: DomainExpertise

    # For backwards compatibility with existing pipeline
    domain_concepts: List[str] = Field(default_factory=list)
    terminology: Dict[str, str] = Field(default_factory=dict)
    artifact_type: str = "code"
    base_class_name: str = "Component"
    standard_parameters: List[str] = Field(default_factory=list)
    interface_signature: str = ""
    evaluation_approach: str = ""
    complexity_constraints: Optional[str] = None
    key_challenges: List[str] = Field(default_factory=list)
    research_areas: List[str] = Field(default_factory=list)
    code_patterns: Optional[Dict[str, str]] = None


# Simplified analysis prompt for smaller context models
CODEBASE_ANALYSIS_PROMPT = """Analyze this codebase and provide JSON output.

## Description
{description}

## Files
{file_structure}

## Code Samples
{file_contents}

## Required JSON Output
Analyze and return JSON with:
- codebase_purpose: one sentence describing what this code does
- codebase_summary: brief summary of functionality
- structure: {{entry_points: [], core_modules: [], architecture_pattern: ""}}
- evolution_targets: list of components to improve, each with file_path, component_name, component_type, current_behavior, evolution_rationale, constraints, suggested_approaches
- proposed_strategies: 2-3 improvement strategies, each with name, description, target_improvements, required_expertise, evaluation_criteria, risk_factors
- domain_expertise: {{primary_domain, sub_domains, key_concepts, terminology, best_practices, common_pitfalls}}

Focus on practical, actionable analysis."""


class DomainUnderstandingAgent:
    """
    Agent that deeply analyzes any codebase to understand its purpose,
    structure, and evolution opportunities.

    This agent is domain-agnostic and works with any type of code.
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the agent.

        Args:
            model: LLM model to use for analysis
        """
        self.model = model
        # Generous limits for gpt-4o (128k context)
        self.max_file_content = 10000  # Max chars per file
        self.max_total_content = 80000  # Max total content (~20k tokens)

    async def analyze(
        self,
        description: str,
        seed_path: Optional[str] = None
    ) -> DomainUnderstanding:
        """
        Deeply analyze a codebase to understand it for evolution.

        Args:
            description: User's description of the domain/project
            seed_path: Path to seed codebase

        Returns:
            DomainUnderstanding with comprehensive analysis
        """
        if not seed_path:
            # No seed - create minimal understanding from description only
            return await self._analyze_description_only(description)

        # Deep codebase analysis
        file_structure, file_contents = await self._deep_analyze_codebase(seed_path)

        # Build the prompt
        prompt = CODEBASE_ANALYSIS_PROMPT.format(
            description=description,
            file_structure=file_structure,
            file_contents=file_contents
        )

        # Call LLM for analysis
        return await self._get_llm_analysis(prompt, seed_path)

    async def _deep_analyze_codebase(self, seed_path: str) -> Tuple[str, str]:
        """
        Perform deep analysis of the codebase.

        Returns:
            Tuple of (file_structure, file_contents)
        """
        path = Path(seed_path)

        if not path.exists():
            return "Path does not exist", ""

        # Collect all files with their metadata
        all_files = []
        file_contents_parts = []
        total_content_size = 0

        # Prioritize important files
        priority_patterns = [
            # Entry points and main files
            r'main\.py$', r'app\.py$', r'run\.py$', r'__main__\.py$',
            r'setup\.py$', r'pyproject\.toml$',
            # Core modules (often at root or in src/)
            r'^[^/]+\.py$',  # Root-level Python files
            r'src/[^/]+\.py$',
            # Configuration
            r'config', r'settings',
            # Models/core logic (common naming)
            r'model', r'core', r'engine', r'agent',
        ]

        # Files to skip
        skip_patterns = [
            r'__pycache__', r'\.pyc$', r'\.git/', r'\.env',
            r'node_modules', r'\.egg-info', r'dist/', r'build/',
            r'\.pytest_cache', r'\.mypy_cache', r'\.tox',
        ]

        if path.is_file():
            all_files.append((path, self._get_file_priority(path.name, priority_patterns)))
        else:
            # Scan directory
            for file in path.rglob('*'):
                if file.is_file():
                    rel_path = str(file.relative_to(path))

                    # Skip unwanted files
                    if any(re.search(p, rel_path) for p in skip_patterns):
                        continue

                    priority = self._get_file_priority(rel_path, priority_patterns)
                    all_files.append((file, priority))

        # Sort by priority (higher first)
        all_files.sort(key=lambda x: x[1], reverse=True)

        # Build file structure
        structure_lines = ["```"]
        for file, priority in all_files:
            if path.is_file():
                structure_lines.append(f"- {file.name}")
            else:
                rel_path = str(file.relative_to(path))
                priority_marker = "★" if priority > 5 else ""
                structure_lines.append(f"- {rel_path} {priority_marker}")
        structure_lines.append("```")

        # Collect file contents (prioritized)
        for file, priority in all_files:
            if total_content_size >= self.max_total_content:
                break

            # Only include text files
            if not self._is_text_file(file):
                continue

            try:
                content = file.read_text(encoding='utf-8', errors='ignore')

                # Truncate very large files
                if len(content) > self.max_file_content:
                    content = content[:self.max_file_content] + "\n... [truncated]"

                if path.is_file():
                    rel_path = file.name
                else:
                    rel_path = str(file.relative_to(path))

                # Add analysis hints for Python files
                analysis_hints = ""
                if file.suffix == '.py':
                    analysis_hints = self._extract_python_hints(content)

                file_section = f"""
### File: {rel_path}
{analysis_hints}
```{self._get_language(file)}
{content}
```
"""
                file_contents_parts.append(file_section)
                total_content_size += len(file_section)

            except Exception as e:
                continue

        return "\n".join(structure_lines), "\n".join(file_contents_parts)

    def _get_file_priority(self, path: str, priority_patterns: List[str]) -> int:
        """Calculate file priority for analysis ordering."""
        priority = 0

        # Check priority patterns
        for i, pattern in enumerate(priority_patterns):
            if re.search(pattern, path, re.IGNORECASE):
                priority += 10 - i  # Earlier patterns = higher priority

        # Boost Python files
        if path.endswith('.py'):
            priority += 5

        # Boost files with certain names
        important_names = ['model', 'main', 'core', 'engine', 'agent', 'train', 'eval']
        for name in important_names:
            if name in path.lower():
                priority += 3

        # Reduce priority for test files (still include, but after core)
        if 'test' in path.lower():
            priority -= 2

        return priority

    def _is_text_file(self, path: Path) -> bool:
        """Check if a file is a text file worth analyzing."""
        text_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
            '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
            '.md', '.txt', '.rst', '.html', '.css', '.sql',
            '.sh', '.bash', '.zsh', '.ps1',
            '.dockerfile', '.makefile'
        }

        # Check extension
        if path.suffix.lower() in text_extensions:
            return True

        # Check common names without extensions
        if path.name.lower() in {'dockerfile', 'makefile', 'readme', 'license', 'requirements.txt'}:
            return True

        return False

    def _get_language(self, path: Path) -> str:
        """Get language identifier for code block."""
        ext_to_lang = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.go': 'go',
            '.rs': 'rust', '.rb': 'ruby', '.php': 'php',
            '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml',
            '.md': 'markdown', '.html': 'html', '.css': 'css',
            '.sh': 'bash', '.sql': 'sql', '.toml': 'toml',
        }
        return ext_to_lang.get(path.suffix.lower(), '')

    def _extract_python_hints(self, content: str) -> str:
        """Extract structural hints from Python code."""
        hints = []

        try:
            tree = ast.parse(content)

            # Extract classes
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            if classes:
                hints.append(f"**Classes:** {', '.join(classes)}")

            # Extract top-level functions
            functions = [node.name for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and not node.name.startswith('_')]
            if functions[:10]:  # Limit to first 10
                hints.append(f"**Functions:** {', '.join(functions[:10])}")

            # Extract imports (to understand dependencies)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            # Group imports by category
            stdlib = []
            third_party = []
            local = []

            for imp in set(imports):
                if imp.startswith('.'):
                    local.append(imp)
                elif self._is_stdlib(imp):
                    stdlib.append(imp)
                else:
                    third_party.append(imp)

            if third_party:
                hints.append(f"**Key dependencies:** {', '.join(sorted(set(third_party))[:8])}")

        except SyntaxError:
            pass

        if hints:
            return "\n".join(hints) + "\n"
        return ""

    def _is_stdlib(self, module: str) -> bool:
        """Check if a module is part of Python standard library."""
        stdlib_modules = {
            'os', 'sys', 'json', 'typing', 'pathlib', 'datetime', 'time',
            'collections', 'itertools', 'functools', 'dataclasses', 'abc',
            'asyncio', 're', 'math', 'random', 'copy', 'io', 'logging',
            'unittest', 'subprocess', 'threading', 'multiprocessing',
            'argparse', 'configparser', 'csv', 'pickle', 'hashlib',
            'urllib', 'http', 'socket', 'email', 'html', 'xml',
        }
        root_module = module.split('.')[0]
        return root_module in stdlib_modules

    async def _get_llm_analysis(self, prompt: str, seed_path: Optional[str]) -> DomainUnderstanding:
        """Get LLM analysis of the codebase."""
        try:
            from agents import Agent, Runner
            from agents.extensions.models.litellm_model import LitellmModel

            agent = Agent(
                name="Domain Understanding Agent",
                instructions=prompt,
                model=LitellmModel(model=self.model),
                output_type=DomainUnderstanding
            )

            result = await Runner.run(agent, "Analyze the codebase and provide comprehensive structured output.")
            understanding = result.final_output

            # Populate backwards-compatible fields
            self._populate_legacy_fields(understanding)

            return understanding

        except ImportError:
            return await self._analyze_with_openai(prompt, seed_path)

    async def _analyze_with_openai(
        self,
        prompt: str,
        seed_path: Optional[str]
    ) -> DomainUnderstanding:
        """Fallback analysis using OpenAI API directly."""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()

            schema_description = """{
  "codebase_purpose": "one sentence",
  "codebase_summary": "brief summary",
  "structure": {"entry_points": [], "core_modules": [], "architecture_pattern": ""},
  "evolution_targets": [{"file_path": "", "component_name": "", "component_type": "", "current_behavior": "", "evolution_rationale": "", "constraints": [], "suggested_approaches": []}],
  "proposed_strategies": [{"name": "", "description": "", "target_improvements": [], "required_expertise": [], "evaluation_criteria": [], "risk_factors": []}],
  "domain_expertise": {"primary_domain": "", "sub_domains": [], "key_concepts": [], "terminology": {}, "best_practices": [], "common_pitfalls": []}
}"""

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert software architect analyzing codebases. Provide thorough, practical analysis. Your response MUST be valid JSON only - no markdown, no explanations, just the JSON object."
                    },
                    {
                        "role": "user",
                        "content": prompt + f"\n\nProvide your response as a JSON object with this structure:\n{schema_description}\n\nIMPORTANT: Output ONLY the JSON object, no markdown code blocks, no explanations."
                    }
                ],
                max_tokens=8000  # gpt-4o can handle large responses
            )

            # Extract JSON from response (handle markdown blocks)
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                # Remove markdown code blocks
                lines = content.split("\n")
                # Find start and end of code block
                start_idx = 0
                end_idx = len(lines)
                for i, line in enumerate(lines):
                    if line.startswith("```") and i == 0:
                        start_idx = 1
                    elif line.startswith("```"):
                        end_idx = i
                        break
                content = "\n".join(lines[start_idx:end_idx])

            data = json.loads(content)

            # Build the DomainUnderstanding from response
            understanding = self._build_understanding_from_json(data)
            self._populate_legacy_fields(understanding)

            return understanding

        except Exception as e:
            print(f"Warning: LLM analysis failed: {e}")
            return self._create_fallback_understanding(seed_path)

    def _build_understanding_from_json(self, data: Dict) -> DomainUnderstanding:
        """Build DomainUnderstanding from JSON response."""
        # Build nested structures with safe defaults
        struct_data = data.get('structure', {})
        structure = CodebaseStructure(
            entry_points=struct_data.get('entry_points', []),
            core_modules=struct_data.get('core_modules', []),
            utility_modules=struct_data.get('utility_modules', []),
            test_modules=struct_data.get('test_modules', []),
            config_files=struct_data.get('config_files', []),
            data_files=struct_data.get('data_files', []),
            architecture_pattern=struct_data.get('architecture_pattern', 'unknown'),
            dependency_graph=struct_data.get('dependency_graph', {})
        )

        # Build file analyses with safe parsing
        file_analyses = []
        for fa in data.get('file_analyses', []):
            try:
                file_analyses.append(FileAnalysis(
                    path=fa.get('path', ''),
                    purpose=fa.get('purpose', ''),
                    key_components=fa.get('key_components', []),
                    dependencies=fa.get('dependencies', []),
                    complexity=fa.get('complexity', 'medium'),
                    evolution_potential=fa.get('evolution_potential', '')
                ))
            except Exception:
                pass

        # Build evolution targets with safe parsing
        evolution_targets = []
        for et in data.get('evolution_targets', []):
            try:
                evolution_targets.append(EvolutionTarget(
                    file_path=et.get('file_path', ''),
                    component_name=et.get('component_name', ''),
                    component_type=et.get('component_type', 'module'),
                    current_behavior=et.get('current_behavior', ''),
                    evolution_rationale=et.get('evolution_rationale', ''),
                    constraints=et.get('constraints', []),
                    suggested_approaches=et.get('suggested_approaches', [])
                ))
            except Exception:
                pass

        # Build strategies with safe parsing
        proposed_strategies = []
        for ps in data.get('proposed_strategies', []):
            try:
                proposed_strategies.append(EvolutionStrategy(
                    name=ps.get('name', 'Improvement'),
                    description=ps.get('description', ''),
                    target_improvements=ps.get('target_improvements', []),
                    required_expertise=ps.get('required_expertise', []),
                    evaluation_criteria=ps.get('evaluation_criteria', []),
                    risk_factors=ps.get('risk_factors', [])
                ))
            except Exception:
                pass

        # Build domain expertise
        domain_exp_data = data.get('domain_expertise', {})
        domain_expertise = DomainExpertise(
            primary_domain=domain_exp_data.get('primary_domain', 'software'),
            sub_domains=domain_exp_data.get('sub_domains', []),
            key_concepts=domain_exp_data.get('key_concepts', []),
            terminology=domain_exp_data.get('terminology', {}),
            best_practices=domain_exp_data.get('best_practices', []),
            common_pitfalls=domain_exp_data.get('common_pitfalls', []),
            reference_resources=domain_exp_data.get('reference_resources', [])
        )

        return DomainUnderstanding(
            codebase_purpose=data.get('codebase_purpose', ''),
            codebase_summary=data.get('codebase_summary', ''),
            structure=structure,
            file_analyses=file_analyses,
            evolution_targets=evolution_targets,
            proposed_strategies=proposed_strategies,
            domain_expertise=domain_expertise
        )

    def _populate_legacy_fields(self, understanding: DomainUnderstanding):
        """Populate backwards-compatible fields from new analysis."""
        # Map new fields to legacy fields
        understanding.domain_concepts = understanding.domain_expertise.key_concepts
        understanding.terminology = understanding.domain_expertise.terminology
        understanding.key_challenges = understanding.domain_expertise.common_pitfalls
        understanding.research_areas = [s.name for s in understanding.proposed_strategies]

        # Set evaluation approach from strategies
        if understanding.proposed_strategies:
            criteria = []
            for s in understanding.proposed_strategies:
                criteria.extend(s.evaluation_criteria)
            understanding.evaluation_approach = "; ".join(criteria[:3])

        # Set interface signature from evolution targets
        if understanding.evolution_targets:
            target = understanding.evolution_targets[0]
            understanding.base_class_name = target.component_name
            understanding.interface_signature = f"Component: {target.component_type}"

    async def _analyze_description_only(self, description: str) -> DomainUnderstanding:
        """Analyze when only description is provided (GENESIS mode)."""
        prompt = f"""# Domain Analysis (No Codebase)

The user wants to create a new research project from scratch.

## Domain Description
{description}

## Your Task
Based on this description, provide:

1. **Codebase Purpose**: What should this code do?
2. **Proposed Structure**: What files/modules would this project need?
3. **Evolution Targets**: What components would benefit from automated evolution?
4. **Evolution Strategies**: How should the code be evolved/improved?
5. **Domain Expertise**: What knowledge is needed for this domain?

Provide practical, actionable analysis that can guide code generation."""

        return await self._get_llm_analysis(prompt, None)

    def _create_fallback_understanding(self, seed_path: Optional[str]) -> DomainUnderstanding:
        """Create minimal understanding when analysis fails."""
        return DomainUnderstanding(
            codebase_purpose="Unable to analyze - using defaults",
            codebase_summary="Analysis failed, using minimal defaults",
            structure=CodebaseStructure(
                entry_points=[],
                core_modules=[],
                utility_modules=[],
                test_modules=[],
                config_files=[],
                data_files=[],
                architecture_pattern="unknown",
                dependency_graph={}
            ),
            file_analyses=[],
            evolution_targets=[],
            proposed_strategies=[
                EvolutionStrategy(
                    name="General Improvement",
                    description="Improve code quality and functionality",
                    target_improvements=["performance", "reliability"],
                    required_expertise=["software engineering"],
                    evaluation_criteria=["code quality", "test coverage"],
                    risk_factors=["may require manual review"]
                )
            ],
            domain_expertise=DomainExpertise(
                primary_domain="software",
                sub_domains=[],
                key_concepts=["code quality"],
                terminology={},
                best_practices=["write clean code"],
                common_pitfalls=["overcomplexity"],
                reference_resources=["documentation"]
            ),
            domain_concepts=["software engineering"],
            terminology={},
            artifact_type="code",
            base_class_name="Component",
            standard_parameters=[],
            interface_signature="",
            evaluation_approach="Standard metrics",
            complexity_constraints=None,
            key_challenges=["Analysis failed"],
            research_areas=["General improvement"],
            code_patterns=None
        )
