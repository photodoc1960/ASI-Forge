"""
Base Prompt Templates

Parameterized prompt templates for all agent prompts in the ASI-Arch pipeline.
These templates use variable placeholders that are filled in by the PromptRenderer
based on the active domain specialization.

Template Variables:
    {context} - Experimental context/history
    {domain_name} - Human-readable domain name
    {artifact_type} - Type of artifact (e.g., "neural network", "molecule")
    {base_class_name} - Name of the main class/artifact
    {standard_parameters} - List of standard parameters
    {complexity_requirement} - Computational complexity constraint
    {preservation_rules} - What must not change during evolution
    {strict_constraints} - Must-fix validation rules
    {critical_constraints} - Should-fix validation rules
    {benchmarks} - List of evaluation benchmarks
    {baseline_models} - Baseline model definitions
    {code_style_guidelines} - Code style requirements
    {interface_signature} - Required interface signature
    {required_decorators} - Required decorators/annotations
"""

# =============================================================================
# PLANNER TEMPLATE
# =============================================================================

PLANNER_TEMPLATE = """# {domain_name} Evolution Mission

## EXPERIMENTAL CONTEXT & HISTORICAL EVIDENCE
{context}

## EVOLUTION OBJECTIVE
Your mission is to create a breakthrough {artifact_type} that addresses critical performance limitations identified through experimental evidence while integrating cutting-edge research insights. Design and implement an innovative {artifact_type} that maintains efficiency while achieving superior capabilities.

## SYSTEMATIC EVOLUTION METHODOLOGY

### PHASE 1: Evidence-Based Analysis Framework

#### 1.1 {artifact_type_title} Forensics
**Current State Assessment:**
- Use `read_code_file` to examine existing implementations
- Map mechanisms, design patterns, and information flow
- Identify core approaches and their theoretical foundations
- Document interface constraints and compatibility requirements

#### 1.2 Performance Pattern Recognition
**Historical Evidence Analysis:**
- **Training Dynamics Diagnosis**: Extract optimization challenges from loss curves and convergence patterns
- **Task-Specific Performance Profiling**: Identify capability gaps across evaluation domains
- **Bottleneck Identification**: Pinpoint elements limiting performance vs. those enabling strengths
- **Cross-Experiment Comparison**: Analyze performance patterns across different experimental variants

#### 1.3 Research Integration Strategy
**Theoretical Foundation Building:**
- Map research insights to observed performance limitations
- Identify specific theoretical principles addressing weaknesses
- Synthesize multiple research findings for comprehensive enhancement opportunities
- Validate theoretical applicability through experimental evidence correlation

### PHASE 2: Innovation Design Framework

#### 2.1 Targeted Performance Engineering
**Gap-Specific Solutions:**
- Design modifications targeting the most critical performance bottlenecks
- Create mechanisms leveraging research insights for problematic capability domains
- Balance multiple improvement objectives while maintaining coherence
- Ensure modifications address root causes rather than symptoms

#### 2.2 Theoretical Grounding Protocol
**Research-Driven Design:**
- Ground all modifications in validated theoretical principles
- Ensure mathematical and computational justification for proposed changes
- Verify alignment with established research findings and best practices
- Create novel combinations of insights for breakthrough potential

#### 2.3 Efficiency Optimization Standards
**Computational Constraints:**
{efficiency_constraints}

### PHASE 3: Implementation Excellence Protocol

#### 3.1 Implementation Standards
**Code Development Requirements:**
- Use `write_code_file` to implement the complete evolved {artifact_type}
- Preserve interface compatibility ({interface_signature})
- Add new parameters with sensible defaults (enabled by default for new features)
- Remove or refactor existing features to prevent bloat
{domain_specific_implementation_rules}

#### 3.2 Quality Assurance Framework
**Technical Excellence Standards:**
{quality_assurance_rules}

#### 3.3 Documentation and Justification
**Innovation Communication:**
- Create comprehensive motivation explaining evolution rationale
- Connect experimental evidence to theoretical insights and implementation decisions
- Justify expected improvements based on research findings
- Provide clear reasoning for all design choices

## TECHNICAL IMPLEMENTATION SPECIFICATIONS

### Critical Preservation Requirements
{preservation_requirements}

### Implementation Quality Standards
{implementation_quality_standards}

{robustness_standards}

## INNOVATION TARGET DOMAINS

### Primary Capability Enhancement Areas
{enhancement_areas}

## DELIVERABLE SPECIFICATIONS

### PRIMARY DELIVERABLE: Complete Implementation
**{artifact_type_title} Code (MANDATORY):**
- **Implementation Tool**: Use `write_code_file` to create complete working {artifact_type}
- **Innovation Quality**: Embed revolutionary advances in functional code
- **Constraint Compliance**: Preserve class structure, parameters, and interface compatibility
- **Technical Standards**: {technical_standards_summary}

### SECONDARY DELIVERABLE: Design Documentation
**{artifact_type_title} Description:**
- **Naming Convention**: `{naming_prefix}_[innovation_identifier]` reflecting core innovations
- **Motivation Document**: Comprehensive explanation including:
  - Key innovations and their implementation
  - Research insights applied and expected performance improvements
  - Design choice justification based on experimental evidence
  - Connection between theory, evidence, and implementation

## SUCCESS CRITERIA FRAMEWORK

### Critical Success Factors (Ranked by Priority)
{success_criteria}

## MISSION EMPHASIS
Your **PRIMARY OBJECTIVE** is implementing breakthrough code that demonstrates robust performance across all execution environments and configurations. Create working innovations that directly address identified performance gaps through research-guided evolution. Documentation serves as secondary validation of implemented innovations.

Begin your evolution process by examining the experimental evidence and identifying the most critical improvement opportunities."""


# =============================================================================
# CHECKER TEMPLATE
# =============================================================================

CHECKER_TEMPLATE = """Check the implemented code for critical issues and fix them if found.

## Motivation (for context)
{motivation}

## YOUR CHECKING TASK

Perform these checks IN ORDER:

### 1. READ AND UNDERSTAND (MANDATORY)
Use read_code_file to examine the implementation. Understand what the code is trying to achieve based on the motivation.

### 2. STRICT CHECKS - MUST FIX IF FOUND

{strict_checks}

### 3. CRITICAL CHECKS - SHOULD FIX

{critical_checks}

### 4. FLEXIBLE CHECKS - PRESERVE INNOVATION

{flexible_checks}

### 5. DECISION AND ACTION

IF any issues found in STRICT or CRITICAL checks:
1. Use write_code_file to save the FIXED version
2. Preserve the original innovation while fixing issues
3. Set success=False
4. Explain what was fixed in error field

IF no issues or only minor concerns:
1. Set success=True
2. Leave error empty or note minor concerns

## Common Fixes

{common_fixes}

Remember: The goal is to ensure the code works correctly and efficiently. Fix issues while preserving the innovative ideas."""


# =============================================================================
# MOTIVATION CHECKER TEMPLATE
# =============================================================================

MOTIVATION_CHECKER_TEMPLATE = """# Research Motivation Duplication Check

## Your Task
Analyze whether the following motivation represents a genuinely novel research direction, or if it substantially duplicates previous experimental efforts.

## Target Motivation for Analysis
{motivation}

## Historical Research Context
The following motivations have already been explored in previous experiments:
{historical_context}

## Structured Analysis Framework

### Step 1: Core Component Extraction
For the target motivation, identify:
1. The central problem or limitation being addressed
2. The proposed solution approach or mechanism
3. The theoretical foundation or hypothesis
4. The expected improvement or outcome
5. The scope of application

### Step 2: Systematic Comparison
For each historical motivation, evaluate:
1. **Problem Overlap**: Does it address the same fundamental issue?
2. **Approach Similarity**: Does it propose essentially the same solution?
3. **Theoretical Basis**: Does it rely on the same principles?
4. **Expected Impact**: Does it target the same improvement type?

### Step 3: Duplication Decision
A motivation is DUPLICATE only if:
- It addresses the SAME core problem/limitation
- Using the SAME or very similar approach
- With the SAME theoretical justification
- Targeting the SAME type of improvement

A motivation is NOT duplicate if:
- It addresses the same problem but with a DIFFERENT approach
- It uses similar techniques for a DIFFERENT problem
- It explores DIFFERENT aspects of a similar mechanism
- It applies established concepts in a NOVEL way
- It combines multiple concepts in a NEW configuration

## Analysis Guidelines

### Research Context Awareness
Remember that in {domain_name} research:
{domain_research_context}

### Decision Principles
- Be LENIENT: Incremental improvements on a theme are valid research
- Focus on CORE innovation, not superficial wording differences
- Consider the COMBINATION of ideas, not just individual elements
- Different applications of the same technique are NOT duplicates
- Different scales or contexts make for valid new research

## Output Requirements
Provide:
1. `is_repeated`: Boolean - True ONLY if clearly duplicating previous work
2. `repeated_index`: List of integers - indices of similar experiments (for context)
3. `judgement_reason`: String - detailed explanation of your decision

Be conservative in marking as duplicate - when in doubt, allow the research to proceed."""


# =============================================================================
# DEDUPLICATION TEMPLATE
# =============================================================================

DEDUPLICATION_TEMPLATE = """# Innovation Diversification Task

## Your Mission
The previous motivation was identified as too similar to existing experiments. Your task is to design a genuinely novel {artifact_type} that explores an ORTHOGONAL design space.

## The Repeated Pattern
{repeated_motivation}

## Historical Context
{historical_context}

## Innovation Framework

### Phase 1: Pattern Breaking Analysis
Analyze the repeated pattern to identify:
1. The core design space that has been over-explored
2. The assumptions being made repeatedly
3. The techniques being recycled
4. The performance aspects being targeted repeatedly

### Phase 2: Orthogonal Innovation Design
Explore FUNDAMENTALLY DIFFERENT approaches:

**Cross-Disciplinary Exploration Targets:**
{cross_disciplinary_targets}

**Innovation Direction Guidelines:**
{innovation_guidelines}

### Phase 3: Implementation
Implement a revolutionary design that:

**Preservation Constraints:**
{preservation_constraints}

**Robustness Standards:**
{robustness_standards}

## Structured Execution Protocol

1. **Pattern Analysis**: Identify the exhausted design space
2. **Direction Selection**: Choose a fundamentally different approach
3. **Design**: Create detailed design using read_code_file for reference
4. **Implementation**: Use write_code_file to implement the complete solution

## Success Validation Criteria
{success_criteria}

## Critical Reminders
- Generate a UNIQUE approach that breaks from the repeated pattern
- Focus on ORTHOGONAL innovation, not incremental variation
- Preserve interface compatibility and technical standards
- Create working code that can be validated experimentally"""


# =============================================================================
# DEBUGGER TEMPLATE
# =============================================================================

DEBUGGER_TEMPLATE = """# Debug and Fix Task

## Design Motivation (for context)
{motivation}

## Training Error Log
{error_log}

## Your Task
Analyze the training error and fix the code while preserving the design intent.

## Error Analysis Guidelines
1. **Identify Error Type**:
   - Timeout/Performance Issues (likely complexity problem)
   - Runtime Crashes (shape mismatches, device errors)
   - Numerical Issues (NaN, overflow, underflow)
   - Interface Errors (signature mismatches)

2. **Root Cause Analysis**:
   - Filter noise from error logs to find the actual issue
   - Trace the error to specific code elements
   - Identify whether it's a fundamental design flaw or implementation bug

## Key Constraints
{debug_constraints}

## Fix Strategy Based on Error Type

**For Timeout/Performance Issues:**
{timeout_fix_strategy}

**For Runtime Crashes:**
{crash_fix_strategy}

**For Numerical Issues:**
{numerical_fix_strategy}

## Process Steps
1. Use read_code_file to examine the current implementation
2. Analyze the error in context of the design motivation
3. Identify the minimal fix that preserves the innovation
4. Use write_code_file to save the corrected version
5. Document what was changed and why

## Critical Reminders
- Preserve the core innovation and design intent
- Make minimal, targeted fixes
- Don't redesign - just fix the issue
- Maintain all interface requirements
- Keep {required_decorators} in place"""


# =============================================================================
# ANALYZER TEMPLATE
# =============================================================================

ANALYZER_TEMPLATE = """# Analysis Request: {experiment_name}

## Resources:
- Results: `{results}`
- Code implementation: Use read_code_file tool to examine the {artifact_type}
- Design motivation: {motivation}

## Related Experiments for Ablation Study:
{reference_context}

**IMPORTANT:** The above related experiments represent either parent nodes (previous iterations that led to this design) or sibling nodes (alternative approaches explored from the same parent). Use these for ablation study analysis to understand:
- What specific changes differentiate the current experiment from its relatives
- Which components are responsible for performance differences
- Whether the modifications represent genuine improvements or trade-offs

## Analysis Requirements:

Please read the results, examine the code implementation using read_code_file tool, and analyze the design motivation. Your analysis must include:

1. **MOTIVATION AND DESIGN EVALUATION**
   - Assess the theoretical soundness of the proposed changes
   - Evaluate whether the code implementation correctly reflects the design intention
   - Identify any gaps between motivation and actual implementation
   - Judge the plausibility of expected improvements based on the changes

2. **EXPERIMENTAL RESULTS ANALYSIS WITH ABLATION STUDY**
   - Summarize performance outcomes using task-descriptive language
   - Compare results with baseline models using clear improvement/degradation statements
   - **ABLATION ANALYSIS**: Compare with related experiments to identify:
     * Which specific changes caused performance differences
     * Whether improvements are due to the intended modifications or other factors
     * Trade-offs introduced by each component
   - Identify which capabilities were enhanced vs compromised
   - Provide an overall assessment of whether the modifications achieved their intended goals

3. **EXPECTATION VS REALITY COMPARISON**
   - Analyze whether experimental results align with the stated motivation and expected outcomes
   - Identify surprising results (both positive and negative) that weren't anticipated
   - Assess the accuracy of the design hypothesis based on empirical evidence
   - Determine if the changes produced the predicted effects
   - **CROSS-EXPERIMENT VALIDATION**: Check if similar modifications in related experiments produced consistent effects

4. **THEORETICAL EXPLANATION WITH EVIDENCE**
   - Provide mechanistic explanations for observed performance patterns, supported by:
     * Specific code elements that caused the effects
     * Mathematical reasoning linking changes to performance outcomes
     * Information-theoretic or computational arguments where applicable
   - **COMPARATIVE ANALYSIS**: Explain why this approach outperformed or underperformed relative experiments
   - For performance degradations: explain the precise mechanisms that undermined specific capabilities
   - For improvements: identify the features responsible for enhanced performance
   - Connect theoretical predictions with empirical observations

5. **SYNTHESIS AND INSIGHTS**
   - Summarize key lessons learned about this type of modification
   - **ABLATION INSIGHTS**: Based on comparison with related experiments, identify:
     * Essential vs. redundant components
     * Optimal combinations of modifications
     * Decisions that should be preserved or discarded in future iterations
   - Identify fundamental trade-offs revealed by the experiments
   - Provide actionable insights for future design decisions
   - Suggest specific directions for addressing identified limitations

**Critical Analysis Standards:**
- Support all claims with specific evidence from code, results, or theoretical reasoning
- Use ablation study methodology: isolate the impact of individual changes by comparing with related experiments
- Be honest about failures and unexpected outcomes
- Focus on understanding WHY results occurred, not just WHAT happened
- Maintain scientific rigor in explanations and avoid speculation without evidence
- When analyzing improvements/degradations, always reference related experiments to validate conclusions

Your analysis should be thorough, evidence-based, and provide actionable insights through systematic ablation study."""


# =============================================================================
# SUMMARIZER TEMPLATE
# =============================================================================

SUMMARIZER_TEMPLATE = """# Experience Synthesis Task

## Experimental Context

### Design Motivation
{motivation}

### Performance Analysis
{analysis}

### Available Research Cognition
{cognition}

## Synthesis Instructions

Your task is to synthesize these experimental results into a comprehensive experience summary that will guide future innovations. Focus on extracting maximum value for the Planner agent.

### Analysis Process:

1. **Performance Pattern Extraction**:
   - Identify specific strengths and weaknesses in the experimental results
   - Trace performance limitations to design choices
   - Highlight consistent patterns across different evaluation metrics
   - Assess whether results align with stated design motivations

2. **Theoretical Validation Assessment**:
   - Evaluate how well the experimental outcomes match theoretical expectations
   - Identify where design hypotheses were confirmed or refuted
   - Assess the effectiveness of specific innovations
   - Determine if complexity/performance trade-offs were optimal

3. **Root Cause Diagnosis**:
   - Pinpoint the fundamental elements limiting performance
   - Identify computational bottlenecks and efficiency issues
   - Assess information flow and processing integrity
   - Evaluate parameter utilization and representational capacity

4. **Research Integration Analysis**:
   - Map observed weaknesses to available research insights that could address them
   - Identify principles that align with experimental needs
   - Highlight implementation strategies from research that could be beneficial
   - Assess which research directions are most promising for addressing limitations

5. **Innovation Opportunity Identification**:
   - Specify concrete improvements based on the analysis
   - Provide clear guidance on what should be preserved vs. modified
   - Identify breakthrough opportunities that could significantly improve performance
   - Ensure recommendations maintain required constraints

### Output Requirements:

Generate a comprehensive experience summary that includes:

- **Multi-Element Performance Analysis**: Clear identification of consistent patterns, strengths, and weaknesses across experiments
- **Bottleneck Identification**: Specific pinpointing of design elements that limit performance with supporting evidence
- **Theoretical Consistency Evaluation**: Assessment of how well results align with design motivations and expectations
- **Research Integration Opportunities**: Clear connections between observed weaknesses and available research insights
- **Processing Verification**: Confirmation of integrity and identification of any potential issues
- **Innovation Direction Guidance**: Specific, actionable recommendations for evolution
- **Implementation Strategy**: Concrete suggestions for how to address identified limitations while preserving successful elements

Focus on providing the Planner with:
1. **Clear Understanding** of what specifically is limiting current performance
2. **Targeted Solutions** based on available research insights
3. **Preservation Guidance** for successful elements
4. **Innovation Opportunities** with theoretical justification
5. **Implementation Roadmap** for addressing identified issues

The experience should enable the Planner to make informed decisions about evolution while avoiding repeated failures and building on demonstrated successes."""


# =============================================================================
# MODEL JUDGER TEMPLATE
# =============================================================================

MODEL_JUDGER_TEMPLATE = """# {artifact_type_title} Evaluation Task

## Baseline Models Reference

{baseline_models_section}

## New {artifact_type_title} to Evaluate

### Model Name
{model_name}

### Implementation Details
{model_code}

### Motivation
{motivation}

### Training Performance
{training_results}

### Evaluation Results
{evaluation_results}

## Evaluation Criteria and Scoring Framework

{scoring_criteria}

## Scoring Instructions

1. Compare the new {artifact_type} against the baselines
2. Evaluate each criterion independently
3. Apply the weights to calculate final score
4. Provide detailed reasoning for each score

## Quantitative Analysis Required
- Calculate improvement percentages vs baselines
- Identify specific strengths and weaknesses
- Note any anomalies or unexpected results

## Expected Score Distribution
- Score 1-3: Below baseline, significant issues
- Score 4-5: Baseline level, minimal improvement
- Score 6-7: Moderate improvement, good potential
- Score 8-9: Strong improvement, recommended for further development
- Score 10: Exceptional, breakthrough-level performance

Provide your detailed evaluation with scores and reasoning for each criterion."""
