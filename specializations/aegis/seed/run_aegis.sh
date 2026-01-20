#!/bin/bash
# AEGIS Launcher Script

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║                    AEGIS Launcher                                    ║"
echo "║         Adaptive Evolutionary General Intelligence System           ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if config exists
if [ ! -f "aegis_config.json" ]; then
    echo "⚠️  Configuration not found. Running auto-setup..."
    python setup_aegis.py
    if [ $? -ne 0 ]; then
        echo "❌ Setup failed. Please fix errors and try again."
        exit 1
    fi
fi

echo ""
echo "Select mode:"
echo "  1) Test system (quick verification)"
echo "  2) Interactive session (communicate with agent)"
echo "  3) Autonomous operation (agent runs independently)"
echo "  4) Run demo (full feature demonstration)"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Running system test..."
        python test_aegis.py
        ;;
    2)
        echo ""
        echo "Starting interactive session..."
        echo "(Type 'help' for commands, 'quit' to exit)"
        echo ""
        python aegis_autonomous.py
        ;;
    3)
        echo ""
        read -p "Number of iterations (default 50): " iterations
        iterations=${iterations:-50}
        echo ""
        echo "Starting autonomous operation for $iterations iterations..."
        echo "(Press Ctrl+C to stop)"
        echo ""
        python -c "from aegis_autonomous import AutonomousAEGIS, AEGISConfig; from core.auto_configure import AutoConfigurator; config = AutoConfigurator.load_config(); aegis = AutonomousAEGIS(config); aegis.start_autonomous_operation(max_iterations=$iterations, think_interval_seconds=5)"
        ;;
    4)
        echo ""
        echo "Starting demo..."
        echo "(This requires interaction - press Enter when prompted)"
        echo ""
        python demo.py
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
