# Neural Network Tests Quick-Start Guide

This document provides a quick guide to running the neural network test suite.

## 1. One-Step Execution

Run the neural tests with a single command:

```bash
./run_tests.sh --visualize
```

This will:
- Find a working Python installation
- Set up a virtual environment (if needed)
- Install required dependencies
- Run the neural tests
- Generate visualizations and reports

## 2. Running in Cursor IDE

If you're working in Cursor IDE, use:

```bash
./run_tests.sh --force-cursor-mode --visualize
```

This will show information about existing visualizations without trying to execute tests in the Cursor environment.

## 3. Output and Visualizations

After running the tests, you'll find:

- Visualizations in the `visualizations/` directory
- Test reports in the `tests/results/` directory
- A complete HTML visualization report at `visualizations/visualization_report.html`

## 4. Common Options

- `--visualize`: Generate visualizations and reports
- `--verbose`: Show detailed logs during execution
- `--retrain`: Force retraining of the neural model
- `--export-vis PATH`: Export visualizations to a custom directory
- `--force-cursor-mode`: Run in Cursor IDE compatibility mode

## 5. Learn More

For detailed documentation, see:

- [Neural Test Guide](setup/NEURAL_TEST_GUIDE.md) - Comprehensive guide to running tests
- [Setup README](setup/README.md) - Information about the setup utilities 