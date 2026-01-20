# AEGIS: Adaptive Evolutionary General Intelligence System

## ✅ FULLY FUNCTIONAL - Ready to Use

A complete autonomous AGI framework that sets its own goals, asks questions, searches for knowledge, and proposes self-improvements with comprehensive human oversight.

## Quick Start

```bash
# Test the system
python test_aegis.py

# Or use the launcher
./run_aegis.sh
```

## Safety-First Architecture Discovery Framework

A supervised system for autonomous architecture discovery combining Hierarchical Reasoning Model (HRM) and evolutionary neural architecture search, with comprehensive human oversight and safety mechanisms.

## Key Safety Features

### 1. Human-in-the-Loop Architecture
- **Approval Gates**: All code modifications require explicit human approval
- **Notification System**: Real-time alerts for emergent behaviors
- **Emergency Stop**: Immediate system freeze capability
- **Audit Trail**: Complete logging of all decisions and changes

### 2. Safety Bounds
- **Sandboxed Execution**: All experiments run in isolated environments
- **Resource Limits**: CPU, memory, and time constraints on all operations
- **Capability Bounds**: Explicitly defined maximum complexity limits
- **Rollback System**: Ability to revert to any previous safe state

### 3. Emergence Detection & Monitoring
- **Dedicated Safety Agent**: Continuously monitors for unexpected behaviors
- **Capability Tracking**: Detects novel emergent capabilities
- **Anomaly Detection**: Flags unusual patterns for human review
- **Automatic Freeze**: Halts evolution when anomalies detected

### 4. Robust Safety System
- **Multi-layer Validation**: Code, behavior, and impact validation
- **Formal Verification**: Mathematical proofs where applicable
- **Test Coverage**: Comprehensive safety test suite
- **Impact Assessment**: Evaluates potential risks before execution

## Architecture Overview

```
aegis/
├── core/
│   ├── hrm/                    # Hierarchical reasoning implementation
│   ├── evolution/              # Supervised evolution framework
│   ├── memory/                 # Memory systems
│   └── safety/                 # Core safety mechanisms
├── knowledge/                  # Knowledge base and databases
├── reasoning/                  # Advanced reasoning modules
├── learning/                   # Continual learning with safeguards
├── evaluation/                 # Benchmark and testing suite
├── interfaces/                 # Human approval interface
├── config/                     # Configuration management
└── logs/                       # Audit logs and monitoring
```

## Core Principles

1. **Safety First**: No action without safety validation
2. **Human Control**: Humans have final authority on all changes
3. **Transparency**: Full interpretability and explainability
4. **Bounded Exploration**: Clear limits on architectural changes
5. **Reversibility**: All changes can be undone

## Getting Started

See [SETUP.md](SETUP.md) for installation and configuration.

## Research References

- HRM: Wang et al. 2025 "Hierarchical Reasoning Model" (arXiv:2506.21734)
- ASI-Arch: Liu et al. 2025 "AlphaGo Moment for Model Architecture Discovery" (arXiv:2507.18074)

## License

Apache 2.0 - See LICENSE file for details.
