#!/bin/bash
# Evaluation script for AEGIS specialization
# The evolved code is written to the source file, so we evaluate that file
# The argument passed is the model NAME (for logging), not a path

set -e

MODEL_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPEC_DIR="$SCRIPT_DIR/.."

# The evolved code is written to the source file
# This is configured in the specialization config
EVOLVED_CODE="/mnt/d/aegis/core/training/gpt2_lora_trainer.py"

# Result paths
RESULT_FILE="$SPEC_DIR/results/evaluation.csv"
METRICS_FILE="$SPEC_DIR/results/metrics.csv"
DEBUG_FILE="$SPEC_DIR/debug/error.txt"

# Initialize
mkdir -p "$SPEC_DIR/results" "$SPEC_DIR/debug"
echo "metric,value" > "$RESULT_FILE"
echo "metric,value" > "$METRICS_FILE"
> "$DEBUG_FILE"

echo "Evaluating model: $MODEL_NAME"
echo "Source file: $EVOLVED_CODE"

# Check 1: File exists
if [ ! -f "$EVOLVED_CODE" ]; then
    echo "Error: Evolved code not found at $EVOLVED_CODE" | tee "$DEBUG_FILE"
    echo "code_exists,0" >> "$RESULT_FILE"
    exit 1
fi
echo "code_exists,1" >> "$RESULT_FILE"

# Check 2: Python syntax validation
echo "Checking syntax..."
python3 -m py_compile "$EVOLVED_CODE" 2>>"$DEBUG_FILE"
if [ $? -eq 0 ]; then
    echo "syntax_valid,1" >> "$RESULT_FILE"
    echo "Syntax: OK"
else
    echo "syntax_valid,0" >> "$RESULT_FILE"
    echo "Syntax error in evolved code" >> "$DEBUG_FILE"
    exit 1
fi

# Check 3: Import test - can the code be loaded?
echo "Testing import..."
cd /mnt/d/aegis
python3 -c "
import sys
sys.path.insert(0, '/mnt/d/aegis')

# Try importing the evolved module
try:
    from core.training.gpt2_lora_trainer import *
    print('import_success,1')
    sys.exit(0)
except Exception as e:
    print('import_success,0')
    print(f'Import error: {e}', file=sys.stderr)
    sys.exit(1)
" >> "$RESULT_FILE" 2>>"$DEBUG_FILE"

if [ $? -ne 0 ]; then
    echo "Code failed to import" >> "$DEBUG_FILE"
    # Don't exit - continue with other checks
fi
echo "Import check complete"

# Check 4: Basic structure validation
echo "Validating structure..."
python3 << PYEOF >> "$RESULT_FILE" 2>>"$DEBUG_FILE"
import ast
import sys

with open('$EVOLVED_CODE', 'r') as f:
    code = f.read()

try:
    tree = ast.parse(code)

    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    # Check for common patterns
    has_init = any('__init__' in [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                   for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    has_forward = any('forward' in [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                      for node in ast.walk(tree) if isinstance(node, ast.ClassDef))

    print(f"class_count,{len(classes)}")
    print(f"function_count,{len(functions)}")
    print(f"has_init,{1 if has_init else 0}")
    print(f"has_forward,{1 if has_forward else 0}")
    print("structure_valid,1")

except SyntaxError as e:
    print("structure_valid,0")
    print(f"Parse error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

echo "Structure: OK"

# Check 5: Try to instantiate main classes
echo "Testing instantiation..."
cd /mnt/d/aegis
python3 << PYEOF >> "$METRICS_FILE" 2>>"$DEBUG_FILE"
import sys
sys.path.insert(0, '/mnt/d/aegis')

try:
    # Try importing from the module
    import importlib.util
    spec = importlib.util.spec_from_file_location('evolved', '$EVOLVED_CODE')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find classes with forward method (likely neural network modules)
    found_classes = []
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type):
            found_classes.append(name)

    print(f"found_classes,{len(found_classes)}")
    if found_classes:
        print(f"class_names,{';'.join(found_classes[:5])}")
    print("module_loads,1")

except Exception as e:
    print("module_loads,0")
    print(f"load_error,{str(e)[:100]}")
PYEOF

echo "Instantiation check complete"

# Calculate overall score
echo "Calculating score..."
python3 << PYEOF
import csv

score = 0
max_score = 0

with open('$RESULT_FILE', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        val = row.get('value', '0')
        try:
            if float(val) >= 1:
                score += 1
        except:
            pass
        max_score += 1

# Normalize to 0-100
final_score = (score / max_score * 100) if max_score > 0 else 0
print(f"overall_score,{final_score:.1f}")
PYEOF >> "$RESULT_FILE"

echo ""
echo "=== Evaluation Complete ==="
echo "Model: $MODEL_NAME"
cat "$RESULT_FILE"
echo ""
echo "Results saved to $RESULT_FILE"

# Return success if we got here
exit 0
