#!/bin/bash
# Start Parallel Data Collection for HRM Training
# Target: 1,000,000 examples

echo "======================================================================"
echo "PARALLEL DATA COLLECTION - 1M EXAMPLES FOR HRM"
echo "======================================================================"
echo ""

# Ask how many instances
echo "How many parallel collectors? (Recommended: 4-8)"
read -p "Number of instances [4]: " NUM_INSTANCES
NUM_INSTANCES=${NUM_INSTANCES:-4}

echo ""
echo "Starting $NUM_INSTANCES parallel data collectors..."
echo "Each will write to: data/hrm_training_data.json"
echo ""

# Start collectors in background
for i in $(seq 1 $NUM_INSTANCES); do
    echo "Starting collector #$i..."
    nohup python collect_data_fast.py > logs/collector_$i.log 2>&1 &
    PIDS[$i]=$!
    echo "  PID: ${PIDS[$i]}"
done

echo ""
echo "======================================================================"
echo "✅ $NUM_INSTANCES collectors started!"
echo "======================================================================"
echo ""
echo "Monitor progress:"
echo "  python -c 'import json; d=json.load(open(\"data/hrm_training_data.json\")); print(f\"{len(d[\"examples\"]):,} / 1,000,000\")'"
echo ""
echo "Stop all collectors:"
echo "  pkill -f collect_data_fast.py"
echo ""
echo "View logs:"
echo "  tail -f logs/collector_1.log"
echo ""
echo "======================================================================"
