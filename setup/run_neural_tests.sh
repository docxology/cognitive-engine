#!/bin/bash
# Simple script to run neural network tests

# Get the project root directory
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure the visualizations directory exists
mkdir -p visualizations tests/results

# Clear problematic environment variables
unset PYTHONHOME
unset PYTHONPATH

# Add the project root to Python path
export PYTHONPATH="$PROJECT_ROOT"

# Run the tests with the system Python
python3 "$PROJECT_ROOT/tests/run_neural_tests.py" "$@"

# Show the visualization locations if requested
if [[ "$*" == *"--visualize"* ]]; then
    echo ""
    echo "=============================================================================="
    echo "VISUALIZATIONS"
    echo "Neural network visualizations are available at:"
    echo "$PROJECT_ROOT/visualizations"
    
    if [ -f "visualizations/visualization_report.html" ]; then
        echo ""
        echo "Visualization HTML report:"
        echo "$PROJECT_ROOT/visualizations/visualization_report.html"
    fi
    echo "=============================================================================="
fi 