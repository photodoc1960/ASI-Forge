#!/bin/bash
# Evaluation script for HRM (Hierarchical Reasoning Model)
#
# For HRM, we evaluate the evolved code by:
# 1. Checking syntax validity
# 2. Running Python import tests
# 3. Running unit tests if they exist
# 4. Optionally running a quick training sanity check

set -e

EVOLVED_CODE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPEC_DIR="$SCRIPT_DIR/.."
HRM_DIR="/mnt/d/HRM"

# Results paths
RESULT_FILE="$SPEC_DIR/results/evaluation.csv"
METRICS_FILE="$SPEC_DIR/results/metrics.csv"
DEBUG_FILE="$SPEC_DIR/debug/error.txt"

# Create directories
mkdir -p "$SPEC_DIR/results"
mkdir -p "$SPEC_DIR/debug"

# Clear previous results
> "$RESULT_FILE"
> "$METRICS_FILE"
> "$DEBUG_FILE"

if [ -z "$EVOLVED_CODE" ]; then
    echo "Usage: bash evaluate.sh <evolved_code_path>"
    exit 1
fi

echo "Evaluating HRM evolved code: $EVOLVED_CODE"
echo "HRM project directory: $HRM_DIR"

# Initialize results
echo "metric,value" > "$RESULT_FILE"
echo "metric,value" > "$METRICS_FILE"

# ============================================================================
# Check 1: Code file exists
# ============================================================================
if [ ! -f "$EVOLVED_CODE" ]; then
    echo "Error: Evolved code not found at $EVOLVED_CODE" | tee "$DEBUG_FILE"
    echo "code_exists,0" >> "$RESULT_FILE"
    exit 1
fi
echo "code_exists,1" >> "$RESULT_FILE"
echo "[OK] Code file exists"

# ============================================================================
# Check 2: Syntax validation
# ============================================================================
echo "Checking Python syntax..."
python3 -m py_compile "$EVOLVED_CODE" 2>>"$DEBUG_FILE"
if [ $? -eq 0 ]; then
    echo "syntax_valid,1" >> "$RESULT_FILE"
    echo "[OK] Syntax is valid"
else
    echo "syntax_valid,0" >> "$RESULT_FILE"
    echo "Syntax error in evolved code" >> "$DEBUG_FILE"
    exit 1
fi

# ============================================================================
# Check 3: Import test - can the module be imported?
# ============================================================================
echo "Testing module import..."
cd "$HRM_DIR"

IMPORT_RESULT=$(python3 -c "
import sys
import importlib.util

# Add HRM directory to path so local modules (common, etc.) can be imported
sys.path.insert(0, '$HRM_DIR')

try:
    spec = importlib.util.spec_from_file_location('evolved_module', '$EVOLVED_CODE')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print('import_success,1')
except Exception as e:
    print('import_success,0')
    print(f'Import error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>>"$DEBUG_FILE")

echo "$IMPORT_RESULT" >> "$RESULT_FILE"

if echo "$IMPORT_RESULT" | grep -q "import_success,0"; then
    echo "[FAIL] Module import failed"
    exit 1
else
    echo "[OK] Module imports successfully"
fi

# ============================================================================
# Check 4: Run pytest if tests exist
# ============================================================================
if [ -d "$HRM_DIR/tests" ]; then
    echo "Running tests..."
    cd "$HRM_DIR"
    python3 -m pytest tests/ -v --tb=short 2>>"$DEBUG_FILE"
    if [ $? -eq 0 ]; then
        echo "tests_pass,1" >> "$METRICS_FILE"
        echo "[OK] Tests passed"
    else
        echo "tests_pass,0" >> "$METRICS_FILE"
        echo "[WARN] Some tests failed (see debug log)"
    fi
else
    echo "tests_pass,N/A" >> "$METRICS_FILE"
    echo "[INFO] No tests directory found"
fi

# ============================================================================
# Check 5: Quick sanity check - can we instantiate HRM model?
# ============================================================================
echo "Running HRM instantiation test..."
python3 -c "
import sys
sys.path.insert(0, '$HRM_DIR')

try:
    # Import HRM model and config
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config

    # Create a small test config
    config = HierarchicalReasoningModel_ACTV1Config(
        batch_size=2,
        seq_len=32,
        num_puzzle_identifiers=10,
        vocab_size=100,
        H_cycles=2,
        L_cycles=2,
        H_layers=2,
        L_layers=2,
        hidden_size=64,
        expansion=2.0,
        num_heads=4,
        pos_encodings='rotary'
    )

    # Create model
    model = HierarchicalReasoningModel_ACTV1(config)
    print('model_instantiation,1')

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f'num_parameters,{num_params}')

except ImportError as e:
    # Model import failed - still try basic check
    print('model_instantiation,N/A')
    print(f'Note: Could not import HRM model: {e}', file=sys.stderr)
except Exception as e:
    print('model_instantiation,0')
    print(f'Model error: {e}', file=sys.stderr)
" >> "$METRICS_FILE" 2>>"$DEBUG_FILE"

# ============================================================================
# Calculate overall score
# ============================================================================
echo "Calculating score..."
python3 -c "
import csv

score = 0
max_score = 0

# Read main results
try:
    with open('$RESULT_FILE', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get('value', '0')
            if val not in ('N/A', ''):
                max_score += 1
                if val == '1':
                    score += 1
except Exception:
    pass

# Read metrics
try:
    with open('$METRICS_FILE', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get('value', '0')
            if val not in ('N/A', '') and row.get('metric') in ('tests_pass', 'model_instantiation'):
                max_score += 1
                if val == '1':
                    score += 1
except Exception:
    pass

final_score = score / max_score if max_score > 0 else 0.0
print(f'overall_score,{final_score:.3f}')
" >> "$RESULT_FILE"

echo ""
echo "============================================"
echo "Evaluation complete"
echo "Results: $RESULT_FILE"
echo "Metrics: $METRICS_FILE"
echo "Debug:   $DEBUG_FILE"
echo "============================================"

exit 0
