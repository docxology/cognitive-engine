# Neural Network Test Guide

This guide explains how to run the neural network tests with proper environment setup.

## Quick Start

For the easiest and most reliable way to run the neural tests, use the provided all-in-one setup script:

```bash
./setup/setup_and_run.sh --visualize
```

This script will:
1. Find a working Python installation on your system
2. Create a Python virtual environment if it doesn't exist
3. Install all required dependencies
4. Run the neural tests with proper environment configuration
5. Show you where to find the visualizations and reports

## Running in Cursor IDE

If you're running in Cursor IDE, which has specific environment limitations, use the `--force-cursor-mode` option:

```bash
./setup/setup_and_run.sh --force-cursor-mode --visualize
```

This mode will:
1. Detect the Cursor environment and bypass environment checks
2. Skip the virtual environment setup (which may fail in Cursor)
3. Display information about existing visualizations and reports
4. Show the summary of available test results

Note: In Cursor mode, the script doesn't actually run the tests, but provides access to previously generated visualizations and reports.

## Alternative Methods

### Running with Shell Script

You can also use the shell script if you already have a working Python environment:

```bash
./setup/run_neural_tests.sh --visualize
```

### Running with Python

If you prefer to use Python directly:

```bash
./run_neural_tests.py --use-venv --visualize
```

This will:
1. Create a virtual environment if it doesn't exist
2. Install all required dependencies
3. Run the tests using the virtual environment's Python

## Command Line Options

The neural test runner supports the following options:

- `--visualize`: Generate visualizations and reports
- `--fallback`: Force NumPy fallback mode if PyTorch is not available
- `--verbose`: Enable verbose logging
- `--config PATH`: Path to a custom configuration file
- `--retrain`: Force model retraining
- `--export-vis PATH`: Export visualizations to the specified directory
- `--use-venv`: Use a Python virtual environment (creates one if it doesn't exist)
- `--force-cursor-mode`: Force the script to run in Cursor IDE compatibility mode

## Generated Visualizations

When you run the tests with `--visualize`, the following visualizations will be generated:

### Static Visualizations:
- Test Results Bar Chart
- Network Architecture Diagram
- Learning Curves
- Performance Radar Chart
- Error Analysis Heatmap (if there are errors)
- Detailed Network Architecture Visualization

### Animated Visualizations:
- Training Progress Animation
- Network Activity Heatmap Animation

### Advanced Visualizations:
- 3D Network Architecture Visualization
- Confidence vs MSE Bubble Chart
- Context Distribution Pie Chart
- Test Execution Heat Calendar

### HTML Reports:
- Visualization Report: A single HTML file showing all visualizations
- Interactive Dashboard: An interactive HTML dashboard with Charts.js (requires internet access)

## Visualization Locations

All visualizations can be found in the `visualizations/` directory.
Test reports and results are saved in the `tests/results/` directory.

## Running Tests Outside of Cursor IDE

For the most comprehensive test execution with all features, we recommend running the neural tests outside of the Cursor IDE in a terminal or command prompt. This ensures that all environment variables are properly set and eliminates any environment-specific issues.

```bash
# Navigate to the project directory
cd /path/to/cognitive-engine

# Run the setup script
./setup/setup_and_run.sh --visualize
```

## Troubleshooting

If you encounter issues with the Python environment:

1. Try using the all-in-one setup script: `./setup/setup_and_run.sh --visualize`
2. If running in Cursor IDE, use: `./setup/setup_and_run.sh --force-cursor-mode --visualize`
3. Ensure you have Python 3.7 or later installed
4. Check that you have pip installed and accessible
5. Look for error messages about missing packages and install them manually if needed

If you see an error about missing the 'encodings' module, it indicates an issue with your Python installation. The all-in-one setup script should fix this problem by creating a clean virtual environment.

## Manual Setup

If you prefer to set up the environment manually:

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   # On Linux/macOS
   source venv/bin/activate
   
   # On Windows
   venv\Scripts\activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the tests:
   ```bash
   python tests/run_neural_tests.py --visualize
   ``` 