## AEGIS Setup Guide

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for HRM training)
- 16GB RAM minimum (32GB recommended)
- MongoDB 4.4+ (for knowledge base storage)
- Docker (optional, for containerized databases)

### Installation

#### 1. Clone and Setup Environment

```bash
cd aegis
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Install PyTorch with CUDA (if available)

```bash
# For CUDA 12.1
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch==2.1.0
```

#### 4. Set Up Configuration

Create a `.env` file:

```bash
# API Keys (optional, for web search integration)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Database (optional)
MONGODB_URI=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
```

### Quick Start

#### Option 1: Interactive Mode (Recommended for first use)

```python
from aegis_autonomous import AutonomousAEGIS, AEGISConfig

# Create configuration
config = AEGISConfig(
    vocab_size=1000,
    d_model=256,
    high_level_layers=4,
    low_level_layers=2,
    population_size=5
)

# Create AEGIS system
aegis = AutonomousAEGIS(config)

# Start interactive session
aegis.interactive_session()
```

#### Option 2: Autonomous Mode

```python
from aegis_autonomous import AutonomousAEGIS, AEGISConfig

config = AEGISConfig()
aegis = AutonomousAEGIS(config)

# Start autonomous operation
# The agent will:
# - Generate its own goals
# - Ask questions
# - Search for knowledge
# - Propose improvements (requires your approval)
aegis.start_autonomous_operation(
    max_iterations=100,
    think_interval_seconds=5
)
```

#### Option 3: Manual Control

```python
from aegis_system import AEGIS, AEGISConfig
import torch

config = AEGISConfig()
aegis = AEGIS(config)

# Test reasoning
input_ids = torch.randint(0, config.vocab_size, (1, 20))
result = aegis.reason(input_ids)
print(f"Reasoning safe: {result['safe']}")

# Manual evolution
aegis.evolve(num_generations=10)

# Deploy best model (requires approval)
best_model = aegis.deploy_best_model()
```

### System Architecture

```
AEGIS/
├── Reasoning Engine (HRM)
│   ├── High-level planning module
│   ├── Low-level execution module
│   └── Adaptive computation time
│
├── Evolution Framework
│   ├── Population management
│   ├── Mutation operators
│   ├── Fitness evaluation
│   └── Human approval gates
│
├── Safety Systems
│   ├── Code validation
│   ├── Architecture validation
│   ├── Behavior validation
│   └── Emergence detection
│
├── Autonomous Agent
│   ├── Goal generation
│   ├── Curiosity engine
│   ├── Intrinsic motivation
│   └── Action execution
│
└── Knowledge System
    ├── Web search
    ├── Knowledge base
    └── Synthesis
```

### Safety Features

1. **Human Approval Gates**
   - All code modifications require approval
   - Architecture changes need review
   - Deployment requires explicit permission

2. **Emergence Detection**
   - Monitors for unexpected behaviors
   - Automatically freezes on anomalies
   - Notifies human operator

3. **Safety Bounds**
   - Parameter limits
   - Compute limits
   - Change rate limits

4. **Audit Trail**
   - Complete logging of all decisions
   - Reversibility of all changes
   - Approval history

### Approving Requests

When the system requests approval, you'll see:

```
╔══════════════════════════════════════════════════════════════╗
║              HUMAN APPROVAL REQUIRED                          ║
╚══════════════════════════════════════════════════════════════╝

Request ID: abc123...
Type: architecture_modification

Title: Agent-Proposed Improvement: reasoning efficiency
...
```

To approve:

```python
aegis.approval_manager.approve_request(
    request_id="abc123...",
    reviewer_name="Your Name",
    approval_code="unique_approval_code_123",
    notes="Approved for testing"
)
```

To reject:

```python
aegis.approval_manager.reject_request(
    request_id="abc123...",
    reviewer_name="Your Name",
    reason="Need more analysis first"
)
```

### Monitoring

Check system status:

```python
status = aegis.get_system_status()
print(status)
```

Monitor emergence detector:

```python
emergence_status = aegis.emergence_detector.get_status_report()
print(emergence_status)
```

View pending approvals:

```python
pending = aegis.approval_manager.get_pending_requests()
for request in pending:
    print(f"{request.title}: {request.description}")
```

### Emergency Stop

If you need to immediately stop the system:

```python
aegis.emergency_stop("User-initiated emergency stop")
```

This will:
- Freeze the emergence detector
- Pause evolution
- Trigger safety validator emergency stop
- Log the event

### Troubleshooting

**Issue: System frozen**
- Check: `aegis.emergence_detector.is_frozen`
- Reason: `aegis.emergence_detector.freeze_reason`
- Unfreeze: `aegis.emergence_detector.unfreeze_system(approval_code)`

**Issue: Evolution paused**
- Check: `aegis.evolution_framework.is_paused`
- Reason: `aegis.evolution_framework.pause_reason`
- Resume: `aegis.evolution_framework.resume_evolution(approval_code)`

**Issue: Out of memory**
- Reduce `d_model` in configuration
- Reduce `population_size`
- Use smaller `max_seq_len`

### Advanced Configuration

```python
config = AEGISConfig(
    # Model architecture
    vocab_size=10000,
    d_model=512,
    high_level_layers=6,
    low_level_layers=4,
    n_heads=8,
    dropout=0.1,
    max_seq_len=2048,

    # Evolution
    population_size=20,
    elite_ratio=0.2,
    mutation_rate=0.3,
    max_generations=100,

    # Safety
    max_parameters=1_000_000_000,
    max_memory_gb=32,
    max_compute_hours=24,

    # Control
    require_approval_for_deployment=True,
    require_approval_for_code_gen=True,
    auto_freeze_on_emergence=True
)
```

### Best Practices

1. **Start Small**: Use small model sizes for initial testing
2. **Monitor Closely**: Watch for emergence alerts
3. **Review Approvals**: Carefully review all approval requests
4. **Regular Checkpoints**: Save system state frequently
5. **Gradual Scaling**: Increase complexity incrementally

### Next Steps

- Read `ARCHITECTURE.md` for system design details
- See `EXAMPLES.md` for usage examples
- Check `SAFETY.md` for safety protocols
- Review `API.md` for API documentation
