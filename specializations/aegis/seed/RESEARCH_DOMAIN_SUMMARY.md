# AEGIS Research Domain Summary

## For Research Team Testing and Improvement

**Document Version:** 1.0
**Date:** January 2026
**Project:** Adaptive Evolutionary General Intelligence System (AEGIS)

---

## Executive Summary

AEGIS is an autonomous AGI research framework implementing state-of-the-art techniques from recent AI safety and neural architecture research. The system combines **hierarchical reasoning**, **supervised evolutionary architecture discovery**, and **multi-layer safety controls** with mandatory human oversight. This document provides guidance for research teams working on testing and improving its functions.

---

## 1. Research Domain Overview

### 1.1 Core Research Areas

AEGIS intersects several critical AI research domains:

| Domain | Implementation | Key Research Questions |
|--------|----------------|----------------------|
| **Hierarchical Reasoning** | Dual-timescale processing (HRM) | How do abstract planning and detailed execution interact? |
| **Neural Architecture Search** | Supervised evolution with safety gates | Can architectures be safely evolved with human oversight? |
| **AI Safety** | Multi-layer validation, emergence detection | How to detect and prevent unexpected capability jumps? |
| **Autonomous Agents** | Curiosity-driven goal generation | How to build intrinsically motivated, controllable agents? |
| **Human-AI Collaboration** | Approval workflows, oversight gates | What approval granularity balances safety vs. autonomy? |

### 1.2 Theoretical Foundations

The system is grounded in published research:

1. **Wang et al. 2025** (arXiv:2506.21734) - Hierarchical Reasoning Model
   - Dual-timescale processing: slow abstract planning + fast detailed execution
   - Adaptive Computation Time (ACT) for variable-depth reasoning
   - Recurrent state passing between hierarchical modules

2. **Liu et al. 2025** - ASI-Arch (Autonomous Architecture Discovery)
   - Population-based genetic algorithm for neural architecture search
   - Multi-objective fitness evaluation
   - Controlled mutation operators

3. **Information Gap Theory** (Loewenstein 1994) - Curiosity system
   - Curiosity arises from knowledge gaps
   - Intrinsic motivation drives autonomous behavior

---

## 2. System Architecture

### 2.1 Component Hierarchy

```
AEGIS System
├── Core Intelligence Layer
│   ├── Hierarchical Reasoning Model (HRM)
│   │   ├── High-Level Planning Module (abstract reasoning)
│   │   ├── Low-Level Execution Module (detailed computation)
│   │   ├── Cross-Attention (inter-module communication)
│   │   └── Adaptive Computation Time (dynamic depth)
│   │
│   └── Autonomous Agent System
│       ├── Curiosity Engine (knowledge gap detection)
│       ├── Goal Generator (autonomous goal creation)
│       ├── Intrinsic Motivation System (drives)
│       └── Neural Integration (HRM/LLM decision-making)
│
├── Evolution Layer
│   ├── Supervised Evolution Framework
│   │   ├── Architecture Generator (mutations)
│   │   ├── Population Manager (selection, elite)
│   │   └── Fitness Evaluation (multi-objective)
│   │
│   └── Code Execution System
│       ├── Safe Code Executor (sandbox)
│       ├── Code Debugger (auto-fix)
│       └── Training Validator
│
├── Safety Layer
│   ├── Comprehensive Safety Validator
│   │   ├── Code Safety Validator
│   │   ├── Architecture Safety Validator
│   │   └── Behavior Safety Validator
│   │
│   └── Emergence Detection Agent
│       ├── Capability Tracker (8 domains)
│       ├── Anomaly Detector
│       └── Auto-Freeze Mechanism
│
└── Human Interface Layer
    ├── Approval Manager
    ├── Request Workflow
    └── Audit Trail
```

### 2.2 Key File Locations

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| HRM Core | `core/hrm/hierarchical_reasoning.py` | 500 | Dual-timescale reasoning architecture |
| Autonomous Agent | `core/agency/autonomous_agent.py` | 714 | Goal-driven autonomous behavior |
| Curiosity Engine | `core/agency/autonomous_agent.py:72` | ~85 | Knowledge gap detection & questions |
| Supervised Evolution | `core/evolution/supervised_evolution.py` | 684 | Architecture discovery with approval |
| Safety Validator | `core/safety/safety_validator.py` | 413 | Multi-layer safety checks |
| Emergence Detector | `core/safety/emergence_detector.py` | 445 | Capability jump detection |
| Human Approval | `interfaces/human_approval.py` | 13,290 | Approval workflow management |
| Main System | `aegis_autonomous.py` | 84,868 | Full autonomous AGI integration |

---

## 3. Research Testing Areas

### 3.1 Hierarchical Reasoning Model (HRM)

**Location:** `core/hrm/hierarchical_reasoning.py`

#### Key Components to Test

1. **Dual-Timescale Processing** (lines 266-371)
   - `RecurrentPlanningModule`: High-level abstract planning
   - `RecurrentExecutionModule`: Low-level detailed execution
   - Test: Does high-level guidance improve low-level performance?

2. **Adaptive Computation Time** (lines 223-263)
   - `AdaptiveComputationTime`: Dynamic reasoning depth
   - Test: Does ACT appropriately allocate computation to harder problems?

3. **Cross-Module Communication** (lines 169-220)
   - `CrossAttention`: Information flow between modules
   - Test: What information transfers between abstraction levels?

4. **Rotary Positional Embeddings** (lines 18-59)
   - `RotaryPositionalEmbedding`: RoPE implementation
   - Test: Position encoding effectiveness for sequence understanding

#### Suggested Experiments

```python
# Test HRM reasoning depth adaptation
from core.hrm.hierarchical_reasoning import HierarchicalReasoningModel

model = HierarchicalReasoningModel(vocab_size=10000, d_model=512, use_act=True)
# Compare ponder_cost on easy vs. hard reasoning tasks
# Expected: harder tasks should have higher ponder_cost
```

### 3.2 Autonomous Agent System

**Location:** `core/agency/autonomous_agent.py`

#### Key Components to Test

1. **Curiosity Engine** (lines 72-177)
   - Knowledge gap detection and question generation
   - Test: Does curiosity drive meaningful exploration?

2. **Goal Generator** (lines 179-264)
   - Autonomous goal creation and selection
   - Test: Are generated goals coherent and achievable?

3. **Intrinsic Motivation** (lines 267-329)
   - Curiosity, competence, autonomy, exploration drives
   - Test: Do motivation levels correlate with productive behavior?

4. **Neural Decision Making** (lines 491-557)
   - HRM/LLM-based action selection
   - Test: How does neural vs. rule-based decision quality compare?

#### Suggested Experiments

```python
# Test curiosity-driven goal generation
agent = AutonomousAgent("test_agent")
agent.curiosity.register_knowledge_gap("machine_learning", "transformer attention")

# Should generate KNOWLEDGE_ACQUISITION goal
goal = agent.goal_generator.select_next_goal()
assert goal.goal_type == GoalType.KNOWLEDGE_ACQUISITION
```

### 3.3 Supervised Evolution Framework

**Location:** `core/evolution/supervised_evolution.py`

#### Key Components to Test

1. **Architecture Mutations** (lines 69-132)
   - `add_layer`, `modify_attention`, `change_activation`, `adjust_dimensions`
   - Test: Do mutations produce valid, trainable architectures?

2. **Auto-Approval Logic** (lines 458-515)
   - Low-risk change detection and auto-approval
   - Test: Are approval thresholds appropriate for safety?

3. **Elite Selection** (lines 517-555)
   - Population-based genetic algorithm
   - Test: Does selection pressure improve fitness over generations?

4. **Emergence Freeze** (lines 627-649)
   - Auto-pause on critical alerts
   - Test: Does freeze trigger reliably on anomalies?

#### Suggested Experiments

```python
# Test auto-approval thresholds
config = EvolutionConfig(
    enable_auto_approval=True,
    max_auto_approve_param_increase=0.05  # 5%
)
framework = SupervisedEvolutionFramework(config)

# Candidate with 6% param increase should NOT auto-approve
# Candidate with 4% param increase SHOULD auto-approve (if other criteria met)
```

### 3.4 Safety Validation System

**Location:** `core/safety/safety_validator.py`

#### Key Components to Test

1. **Code Safety Validator** (lines 78-138)
   - Forbidden imports/operations detection
   - Test: Coverage of dangerous code patterns

2. **Architecture Safety Validator** (lines 141-197)
   - Parameter count, layer depth, mutation size checks
   - Test: Are bounds appropriate for different model scales?

3. **Behavior Safety Validator** (lines 200-290)
   - NaN/Inf detection, output range checking
   - Test: False positive/negative rates on edge cases

4. **Comprehensive Validator** (lines 293-412)
   - Multi-validator coordination
   - Test: Integration of all safety layers

#### Suggested Experiments

```python
# Test code safety detection
validator = CodeSafetyValidator()

# Should REJECT (forbidden import)
result = validator.validate_code("import subprocess; subprocess.run(['rm', '-rf', '/'])")
assert not result.passed
assert result.risk_level == RiskLevel.CRITICAL

# Should PASS
result = validator.validate_code("import torch; model = torch.nn.Linear(10, 10)")
assert result.passed
```

### 3.5 Emergence Detection System

**Location:** `core/safety/emergence_detector.py`

#### Key Components to Test

1. **Capability Tracker** (lines 44-143)
   - 8 capability domains: logical_reasoning, pattern_recognition, abstraction, planning, generalization, transfer_learning, compositional_reasoning, meta_learning
   - Test: Sensitivity and specificity of capability jump detection

2. **Anomaly Detector** (lines 146-237)
   - Loss explosion, gradient explosion, runtime anomaly detection
   - Test: Detection latency and false positive rates

3. **Auto-Freeze Mechanism** (lines 378-399)
   - System freeze on critical events
   - Test: Freeze triggers reliably without false positives

4. **Capability Jump Threshold** (line 84)
   - 15% improvement threshold for alerts
   - Test: Is this threshold appropriate for your models?

#### Suggested Experiments

```python
# Test capability jump detection
tracker = CapabilityTracker()

# Establish baseline (10 measurements)
for _ in range(10):
    tracker.record_capability('logical_reasoning', 0.50)

# Record significant improvement
tracker.record_capability('logical_reasoning', 0.70)  # 40% jump

improvement = tracker.detect_capability_jump('logical_reasoning', threshold=0.15)
assert improvement is not None  # Should detect jump
assert improvement > 0.15  # Above threshold
```

---

## 4. Training Systems

### 4.1 Available Training Methods

| Method | File | Purpose |
|--------|------|---------|
| HRM Supervised | `core/training/hrm_trainer.py` | Train HRM on reasoning tasks |
| GPT-2 LoRA | `core/training/gpt2_lora_trainer.py` | Fine-tune pretrained LLM |
| Reinforcement | `core/training/rl_trainer.py` | Policy gradient training |
| Data Collection | `core/training/data_collector.py` | Gather training examples |

### 4.2 Training Data Requirements

From `TRAINING_DATA_PLAN.md`:

1. **Reasoning Tasks**: Chain-of-thought, multi-step reasoning
2. **Planning Tasks**: Goal decomposition, action sequencing
3. **Knowledge Tasks**: Question-answering, fact retrieval
4. **Safety Tasks**: Harmful request rejection, boundary adherence

---

## 5. Configuration & Bounds

### 5.1 Safety Bounds (Default)

From `core/safety/safety_validator.py:48-70`:

| Bound | Default Value | Purpose |
|-------|---------------|---------|
| `max_parameters` | 1 billion | Prevent runaway model size |
| `max_memory_gb` | 32 GB | Resource protection |
| `max_layers` | 200 | Architecture complexity limit |
| `max_recursion_depth` | 16 | ACT step limit |
| `max_capability_jump` | 15% | Emergence detection threshold |
| `max_mutation_size` | 5% | Evolution step size limit |
| `max_performance_degradation` | 10% | Behavior change limit |

### 5.2 Evolution Configuration

From `core/evolution/supervised_evolution.py:47-66`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `population_size` | 20 | Genetic algorithm population |
| `elite_ratio` | 0.2 | Top 20% survive |
| `mutation_rate` | 0.3 | Mutation probability |
| `enable_auto_approval` | True | Auto-approve low-risk changes |
| `min_performance_gain` | 5% | Required improvement for auto-approval |

---

## 6. Testing Recommendations

### 6.1 Unit Tests

| Area | Recommended Tests |
|------|-------------------|
| HRM | Forward pass, gradient flow, ACT allocation, cross-attention |
| Agent | Goal generation, curiosity triggers, action selection |
| Evolution | Mutation validity, selection pressure, approval logic |
| Safety | Code patterns, bounds checking, false positive rates |
| Emergence | Capability tracking, anomaly detection, freeze triggers |

### 6.2 Integration Tests

1. **End-to-End Reasoning**: Input → HRM → Action → Execution → Result
2. **Evolution Cycle**: Generate → Validate → Approve → Evaluate → Select
3. **Safety Pipeline**: Code → Arch → Behavior → Approval → Execute
4. **Emergence Response**: Capability jump → Freeze → Human review → Resume

### 6.3 Stress Tests

1. **Scale Tests**: Vary model size from 1M to 1B parameters
2. **Long-Running**: Multi-day evolution experiments
3. **Adversarial**: Attempt to bypass safety validators
4. **Edge Cases**: Boundary conditions for all thresholds

---

## 7. Improvement Opportunities

### 7.1 High Priority

1. **LLM-Based Code Generation**: Replace template mutations with LLM-generated code
2. **Richer Emergence Detection**: Add more capability domains, improve baselines
3. **Async Approval Workflow**: Non-blocking human approval system
4. **Distributed Evolution**: Multi-machine population training

### 7.2 Research Extensions

1. **Meta-Learning Integration**: Add meta-learning capability to HRM
2. **Multi-Agent Systems**: Multiple autonomous agents with coordination
3. **Interpretability Tools**: Understand HRM reasoning paths
4. **Formal Verification**: Prove safety properties mathematically

### 7.3 Known Limitations

1. Template-based mutations (not LLM-generated) in current evolution
2. Simplified code-to-model conversion in `_code_to_model`
3. Basic curiosity question generation (templates, not learned)
4. Synchronous approval (blocks on human response)

---

## 8. Quick Start for Researchers

### 8.1 Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run auto-configuration
python setup_aegis.py

# Run tests
python test_aegis.py
```

### 8.2 Key Entry Points

```python
# 1. Test HRM directly
from core.hrm.hierarchical_reasoning import HierarchicalReasoningModel
model = HierarchicalReasoningModel(vocab_size=10000, d_model=256)

# 2. Test autonomous agent
from core.agency.autonomous_agent import AutonomousAgent
agent = AutonomousAgent("research_agent", reasoning_engine=model)

# 3. Test evolution framework
from core.evolution.supervised_evolution import SupervisedEvolutionFramework, EvolutionConfig
config = EvolutionConfig(population_size=10)
evolution = SupervisedEvolutionFramework(config)

# 4. Test safety validation
from core.safety.safety_validator import ComprehensiveSafetyValidator
validator = ComprehensiveSafetyValidator()
```

### 8.3 Demo Execution

```bash
# Run full demo
python demo.py

# Run with specific configuration
python aegis_autonomous.py --config aegis_config.json
```

---

## 9. Glossary

| Term | Definition |
|------|------------|
| **HRM** | Hierarchical Reasoning Model - dual-timescale neural architecture |
| **ACT** | Adaptive Computation Time - dynamic reasoning depth |
| **RoPE** | Rotary Positional Embeddings - position encoding method |
| **LoRA** | Low-Rank Adaptation - efficient fine-tuning technique |
| **Emergence** | Unexpected capability improvements in the system |
| **Approval Gate** | Human checkpoint before executing changes |
| **Capability Domain** | Category of cognitive ability being tracked |

---

## 10. References

1. Wang et al. 2025. "Hierarchical Reasoning Model." arXiv:2506.21734
2. Liu et al. 2025. "ASI-Arch: Autonomous System for Intelligent Architecture Discovery"
3. Loewenstein, G. 1994. "The Psychology of Curiosity: A Review and Reinterpretation"
4. Graves, A. 2016. "Adaptive Computation Time for Recurrent Neural Networks"
5. Su, J. et al. 2021. "RoFormer: Enhanced Transformer with Rotary Position Embedding"

---

**Document Maintained By:** AEGIS Research Team
**Last Updated:** January 2026
