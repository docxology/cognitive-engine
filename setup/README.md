# Neural Network Test Setup

This directory contains utilities for setting up the Python environment and running neural network tests.

## Quick Start

To run the neural tests with proper environment setup, use the all-in-one setup script:

```bash
./setup_and_run.sh --visualize
```

Or use the convenient symlink from the project root:

```bash
# From project root
./run_tests.sh --visualize
```

## Available Scripts

- `setup_and_run.sh`: All-in-one script that sets up a virtual environment and runs the tests
- `run_neural_tests.sh`: Shell script for running tests with an existing Python setup
- `run_neural_tests.py`: Python script for running tests with more flexibility

## Running in Cursor IDE

When running in the Cursor IDE, use the `--force-cursor-mode` option:

```bash
./setup_and_run.sh --force-cursor-mode --visualize
```

This mode is specially designed to handle Cursor's environment limitations and will display information about existing visualizations.

## Python Modules

- `environment.py`: Functions for setting up the Python environment
- `venv.py`: Functions for managing Python virtual environments

## Documentation

- `NEURAL_TEST_GUIDE.md`: Comprehensive guide to running neural tests

## Troubleshooting

If you encounter issues with the Python environment, the all-in-one setup script (`setup_and_run.sh`) is the most reliable method as it:

1. Finds a working Python installation 
2. Creates a clean virtual environment
3. Installs all required dependencies
4. Sets up proper environment variables

## Advanced Usage

For advanced users who need more control over the test execution:

```bash
# Run with verbose logging
./setup_and_run.sh --visualize --verbose

# Force model retraining
./setup_and_run.sh --visualize --retrain

# Export visualizations to a custom directory
./setup_and_run.sh --visualize --export-vis /path/to/export/dir

# Run in Cursor IDE compatibility mode
./setup_and_run.sh --force-cursor-mode --visualize
``` 