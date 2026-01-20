# AEGIS Sandbox and Code Testing Infrastructure

## Overview

AEGIS now includes **proper code execution and testing infrastructure** based on ASI-Arch's approach, with additional safety layers.

---

## How Code is Tested

### 1. Safe Code Execution (Subprocess Sandbox)

**Location**: `core/evolution/code_execution.py` → `SafeCodeExecutor`

Generated code is executed in a **subprocess sandbox** with multiple safety layers:

```python
executor = SafeCodeExecutor(
    timeout_seconds=120,      # Kill after 2 minutes
    max_memory_gb=4.0,        # Memory limit
    work_dir="/tmp/aegis_sandbox_xxx"  # Isolated filesystem
)

result = executor.execute_model_code(
    code="<generated Python code>",
    test_inputs=torch.randn(2, 20),
    architecture_id="gen1_add_layer_0"
)
```

**Safety Features**:
- ✅ **Process Isolation**: Crashes don't kill main process
- ✅ **Timeout**: Kills runaway code after N seconds
- ✅ **Memory Limits**: Prevents memory exhaustion
- ✅ **Filesystem Isolation**: Sandboxed work directory
- ✅ **Safe Environment**: Restricted environment variables

**What Happens**:
1. Code written to temp file (`model_gen1_add_layer.py`)
2. Execution script created that:
   - Imports the code
   - Instantiates the model class
   - Runs forward pass with test inputs
   - Validates outputs (no NaN/Inf)
   - Counts parameters
   - Saves model to file
3. Script executed via `subprocess.run()` with timeout
4. Results parsed from saved files

**Result**:
```python
ExecutionResult(
    success=True,
    model=<loaded nn.Module>,
    error_message="",
    stdout="SUCCESS: Model validated with 1234567 parameters",
    stderr="",
    execution_time=2.3,
    metrics={'parameters': 1234567, 'output_shape': [2, 20, 1000], ...}
)
```

---

### 2. Debug Loop (ASI-Arch Style)

**Location**: `core/evolution/code_execution.py` → `CodeDebugger`

If code execution fails, AEGIS attempts to **automatically debug and fix** it:

```python
for attempt in range(3):  # Max 3 attempts
    exec_result = executor.execute_model_code(code, ...)

    if exec_result.success:
        break  # Success!
    else:
        # Automatic debugging
        code, changes = debugger.debug_and_fix(
            code,
            exec_result.error_message,
            exec_result.stderr
        )
        logger.info(f"Applied fix: {changes}")
```

**Error Types Handled**:
- `ImportError` → Adds missing imports (`torch`, `nn`, `F`)
- `Shape mismatches` → (Future: tensor reshaping)
- `NameError` → (Future: variable definition)
- `AttributeError` → (Future: method fixes)

**Example**:
```python
# Original code (missing imports)
class Model(nn.Module):
    def forward(self, x):
        return F.relu(x)  # Error: F not defined

# After debugging
import torch
import torch.nn as nn
import torch.nn.functional as F  # ← Added

class Model(nn.Module):
    def forward(self, x):
        return F.relu(x)  # ← Now works!
```

---

### 3. Trainability Validation

**Location**: `core/evolution/code_execution.py` → `TrainingValidator`

After code executes successfully, AEGIS validates the model **can actually be trained**:

```python
validator = TrainingValidator(training_steps=100)

trainable, metrics = validator.validate_training(
    model=model,
    vocab_size=1000,
    seq_len=20
)
```

**What It Does**:
1. Creates synthetic training data
2. Sets up optimizer (Adam) and loss function (CrossEntropy)
3. Runs 100 training steps
4. Checks for:
   - ✅ No NaN gradients
   - ✅ Loss is finite
   - ✅ Model doesn't explode
   - ✅ Forward/backward pass works

**Result**:
```python
(True, {
    'avg_loss': 2.3,
    'final_loss': 1.8,
    'training_steps': 100,
    'trainable': True
})
```

If training fails → Model is rejected (or flagged)

---

## Complete Validation Pipeline

When a new architecture is generated, it goes through:

```
Generated Code
     │
     ▼
┌─────────────────────────────────────────────┐
│ 1. Code Safety Validation                  │
│    • No dangerous imports (os, subprocess)  │
│    • No file operations                     │
│    • No network access                      │
└─────────────────────────────────────────────┘
     │ PASS
     ▼
┌─────────────────────────────────────────────┐
│ 2. Subprocess Execution (with debug loop)   │
│    Attempt 1: Execute code                  │
│    ↓ FAIL                                   │
│    Auto-debug: Add missing imports          │
│    Attempt 2: Execute code                  │
│    ↓ SUCCESS                                │
│    Model loaded ✓                           │
└─────────────────────────────────────────────┘
     │ SUCCESS
     ▼
┌─────────────────────────────────────────────┐
│ 3. Architecture Safety Validation           │
│    • Parameter count < limit                │
│    • Layer count < limit                    │
│    • Memory usage < limit                   │
│    • No NaN/Inf in forward pass             │
└─────────────────────────────────────────────┘
     │ PASS
     ▼
┌─────────────────────────────────────────────┐
│ 4. Trainability Validation                  │
│    • Run 100 training steps                 │
│    • Check gradients are finite             │
│    • Verify loss decreases (or stable)      │
│    • Test backward pass works               │
└─────────────────────────────────────────────┘
     │ PASS
     ▼
┌─────────────────────────────────────────────┐
│ 5. Human Approval Gate                      │
│    • Present to human                       │
│    • Include all metrics                    │
│    • Wait for approval                      │
└─────────────────────────────────────────────┘
     │ APPROVED
     ▼
  Deploy ✓
```

---

## Example: Complete Flow

```python
# 1. Code is generated
generated_code = '''
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(512, 768)
        self.layer2 = nn.Linear(768, 1000)

    def forward(self, x):
        # x shape: (batch, seq_len)
        x = self.layer1(x)  # Wrong! Expects (batch, 512)
        x = self.layer2(x)
        return x
'''

# 2. Execute in sandbox
exec_result = executor.execute_model_code(
    code=generated_code,
    test_inputs=torch.randint(0, 1000, (2, 20)),
    architecture_id="gen1_test"
)

# Result: FAILURE (shape mismatch)
# exec_result.stderr: "RuntimeError: mat1 and mat2 shapes cannot be multiplied"

# 3. Debug loop tries to fix
# (Currently would fail - shape errors need more sophisticated fixing)
# In future: Would detect shape mismatch and add embedding layer

# 4. After human fixes or debugger improves:
fixed_code = '''
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 512)  # ← Added
        self.layer1 = nn.Linear(512, 768)
        self.layer2 = nn.Linear(768, 1000)

    def forward(self, x):
        x = self.embed(x)   # (batch, seq_len, 512)
        x = x.mean(dim=1)   # (batch, 512)
        x = self.layer1(x)
        x = self.layer2(x)
        return x
'''

# 5. Execute again - SUCCESS!
exec_result = executor.execute_model_code(...)
# exec_result.success = True
# exec_result.model = <Model object>

# 6. Validate trainability
trainable, metrics = validator.validate_training(exec_result.model, ...)
# trainable = True
# metrics = {'avg_loss': 2.1, 'final_loss': 1.9, ...}

# 7. Request human approval
approval_id = approval_manager.request_approval(
    title="New Architecture: add_layer",
    description=f"Model with {metrics['parameters']} parameters",
    risk_assessment={'trainable': True, 'metrics': metrics},
    ...
)

# 8. Human approves
approval_manager.approve_request(approval_id, ...)

# 9. Model added to population
```

---

## Comparison: ASI-Arch vs AEGIS

| Feature | ASI-Arch | AEGIS (Now) |
|---------|----------|-------------|
| **Code Execution** | Bash script + training | Python subprocess |
| **Sandbox** | External process | External process ✓ |
| **Timeout** | ✓ | ✓ |
| **Debug Loop** | LLM-based (3 attempts) | Rule-based (3 attempts) |
| **Validation** | Actual model training | Synthetic training steps |
| **Error Handling** | Full stderr capture | Full stderr capture ✓ |
| **Code Pool** | Saves successful code | Can be added |
| **Human Approval** | ✗ | ✓ (Safety addition) |
| **Safety Layers** | Minimal | 5 layers ✓ |

---

## Safety Improvements Over ASI-Arch

AEGIS adds several safety layers that ASI-Arch doesn't have:

### 1. Pre-Execution Code Validation
```python
# Before executing, check code content
safe = code_validator.validate_code(generated_code)
if not safe:
    reject()  # Don't even try to execute
```

### 2. Resource Limits
```python
# Memory and CPU limits (OS-dependent)
executor = SafeCodeExecutor(
    timeout_seconds=120,
    max_memory_gb=4.0
)
```

### 3. Architecture Validation
```python
# After execution, validate model structure
safe = architecture_validator.validate_architecture(model)
# Checks: parameter count, layer count, memory
```

### 4. Trainability Test
```python
# Not just "does it run", but "can we train it"
trainable = training_validator.validate_training(model)
```

### 5. Human Approval Gate
```python
# Even if all tests pass, human must approve deployment
approved = approval_manager.request_approval(...)
if not approved:
    don't_deploy()
```

---

## Current Limitations & Future Work

### Limitations

1. **Debug Loop is Rule-Based**
   - ASI-Arch uses LLM to intelligently fix errors
   - We use simple pattern matching
   - **Future**: Integrate LLM-based debugging

2. **No Real Training**
   - We do 100 synthetic steps
   - ASI-Arch runs full training script
   - **Future**: Optional full training validation

3. **Limited Error Fixes**
   - Currently only fixes import errors well
   - Shape/logic errors need smarter fixes
   - **Future**: AST-based code transformation

4. **No Code Pool Yet**
   - ASI-Arch saves all successful codes
   - We could add this easily
   - **Future**: Versioned code repository

### Planned Improvements

**Phase 1: Better Debugging** (Next)
```python
class LLMCodeDebugger:
    """Use Claude/GPT to fix code errors intelligently"""

    def debug_with_llm(self, code, error, stderr):
        prompt = f"""
        This code failed with error:
        {error}

        Stderr:
        {stderr}

        Fix the code:
        {code}
        """
        fixed_code = call_llm(prompt)
        return fixed_code
```

**Phase 2: Full Training Validation**
```python
# Option to run actual training
if config.full_training_validation:
    metrics = run_full_training(
        model=model,
        dataset=dataset,
        epochs=10
    )
```

**Phase 3: Code Pool & Versioning**
```python
# Save all successful architectures
code_pool.save(
    architecture_id=arch_id,
    code=code,
    metrics=metrics,
    generation=gen
)

# Retrieve for analysis
best_codes = code_pool.get_top_k(k=10, metric='accuracy')
```

---

## Usage

### Basic Usage (Automatic)

Evolution framework now uses sandbox automatically:

```python
# Just run evolution - sandbox is used internally
aegis.evolution_framework.evolve_generation(evaluation_function)

# Sandbox handles:
# 1. Code execution
# 2. Debug loop
# 3. Trainability test
# 4. All automatically!
```

### Manual Testing

Test any code manually:

```python
from core.evolution.code_execution import SafeCodeExecutor

executor = SafeCodeExecutor()

code = '''
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)
'''

result = executor.execute_model_code(
    code=code,
    test_inputs=torch.randn(2, 100),
    architecture_id="test_1"
)

if result.success:
    print(f"Success! Model: {result.model}")
    print(f"Metrics: {result.metrics}")
else:
    print(f"Failed: {result.error_message}")
```

### Testing Trainability

```python
from core.evolution.code_execution import TrainingValidator

validator = TrainingValidator(training_steps=100)

trainable, metrics = validator.validate_training(
    model=your_model,
    vocab_size=1000,
    seq_len=20
)

print(f"Trainable: {trainable}")
print(f"Metrics: {metrics}")
```

---

## Summary

**AEGIS now has proper sandbox and testing!**

✅ **Subprocess isolation** (like ASI-Arch)
✅ **Debug loop** (like ASI-Arch)
✅ **Timeout protection**
✅ **Trainability validation**
✅ **5-layer safety system** (exceeds ASI-Arch)
✅ **Human approval gates** (safer than ASI-Arch)

**Key difference from ASI-Arch**:
- ASI-Arch: "Run training, if it works, keep it"
- AEGIS: "Validate code → Test execution → Validate architecture → Test training → Human approval → Deploy"

**More conservative, but much safer for AGI development!**
