#!/bin/bash
# Shell script wrapper for running neural network tests with proper environment setup

# Get absolute path to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Neural Network Test Runner"
echo "=========================="
echo "Project root: $PROJECT_ROOT"

# Check for Python virtual environment
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "Using Python virtual environment at $PROJECT_ROOT/venv"
    
    # Activate virtual environment
    if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
        PYTHON="$PROJECT_ROOT/venv/bin/python"
    elif [ -f "$PROJECT_ROOT/venv/Scripts/activate" ]; then
        source "$PROJECT_ROOT/venv/Scripts/activate"
        PYTHON="$PROJECT_ROOT/venv/Scripts/python.exe"
    else
        echo "❌ Could not find activate script in virtual environment"
        exit 1
    fi
else
    echo "No virtual environment found. Running with system Python."
    PYTHON="$(which python3)"
    
    # Check if Python is available
    if [ -z "$PYTHON" ]; then
        echo "❌ Python not found. Please install Python 3."
        exit 1
    fi
fi

echo "Using Python: $PYTHON"
echo "Python version: $($PYTHON --version)"

# Clear Python path to avoid conflicts
export PYTHONPATH="$PROJECT_ROOT"

# Run the tests with proper environment setup
echo ""
echo "Running neural tests..."
"$PYTHON" "$PROJECT_ROOT/run_neural_tests.py" "$@"

# Return the exit code
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Neural tests completed successfully"
else
    echo "❌ Neural tests failed with exit code $EXIT_CODE"
fi

# Show location of visualizations
VIS_DIR="$PROJECT_ROOT/visualizations"
if [ -d "$VIS_DIR" ]; then
    echo ""
    echo "=================================================================================="
    echo "VISUALIZATIONS"
    echo "Neural network visualizations are available at:"
    echo "$VIS_DIR"
    
    # Check for HTML report
    HTML_REPORT="$VIS_DIR/visualization_report.html"
    if [ -f "$HTML_REPORT" ]; then
        echo ""
        echo "Visualization HTML report:"
        echo "$HTML_REPORT"
    fi
    echo "=================================================================================="
fi

exit $EXIT_CODE 