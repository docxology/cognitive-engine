# Neural Network Testing Documentation

## Overview

The Cognitive Engine includes a comprehensive neural network testing system to validate model performance, generate visualizations, and ensure consistent behavior. This document provides detailed information about the testing system, how to use it, and how to interpret the results.

## Quick Start

Run neural tests with visualizations:

```bash
./run_tests.sh --visualize
```

For more options:

```bash
./run_tests.sh --help
```

## Test System Architecture

The neural testing system consists of:

1. **Test Runner** (`tests/run_neural_tests.py`): Core Python script that executes neural tests
2. **Shell Scripts** (`run_tests.sh`, `setup/setup_and_run.sh`): Environment setup and execution
3. **Configuration** (`tests/data/neural_config.json`): Neural network and test configuration
4. **Visualization System**: Comprehensive visualization generators 
5. **Logging System**: Detailed logging with emoji identifiers
6. **Model Management**: Save, load, and recover neural models

## Test Types

The neural test suite includes:

| Test Type | Description |
|-----------|-------------|
| Inference Tests | Measure accuracy and performance of model inference |
| Learning Tests | Evaluate model training capabilities |
| Error Recovery | Test error handling and recovery mechanisms |
| Visualization Generation | Create comprehensive visualizations |
| Performance Metrics | Calculate and report model performance metrics |

## Visualization Types

The system generates multiple visualization types:

1. **Static Visualizations**:
   - Test Results Summary
   - Network Architecture
   - Learning Curves
   - Performance Radar
   - Error Analysis Heatmap
   - Confidence vs MSE Bubble Chart
   - Context Distribution
   - Test Execution Heat Calendar
   - 3D Network Architecture

2. **Animated Visualizations**:
   - Training Progress Animation
   - Network Activity Heatmap

3. **Reports and Dashboards**:
   - HTML Visualization Report
   - Test Results Summary
   - Detailed Network Architecture
   - Interactive Dashboard (if supported)

## Configuration Options

The neural network can be configured using `tests/data/neural_config.json`:

```json
{
  "layers": [
    {"size": 768, "type": "input"},
    {"size": 256, "type": "hidden"},
    {"size": 128, "type": "hidden"},
    {"size": 64, "type": "output"}
  ],
  "learningRate": 0.001,
  "batchSize": 32,
  "optimizer": "adam",
  "activationFunction": "relu",
  "fallbackMode": false,
  "maxRetries": 3,
  "recoveryPath": "models/recovery"
}
```

## Command Line Options

The test system supports multiple command line options:

| Option | Description |
|--------|-------------|
| `--visualize` | Generate visualizations |
| `--fallback` | Run in fallback mode (NumPy instead of PyTorch) |
| `--config PATH` | Use custom configuration file |
| `--retrain` | Force model retraining |
| `--export-vis PATH` | Export visualizations to custom path |
| `--verbose` | Enable verbose logging |
| `--benchmark` | Run performance benchmarks |
| `--force-cursor-mode` | Run in Cursor IDE compatibility mode |
| `--skip-tests` | Skip tests and only generate visualizations |
| `--generate-report` | Generate HTML report |
| `--interactive` | Enable interactive mode |

## Environment Requirements

The neural testing system requires:

- Python 3.7+ with the following packages:
  - numpy
  - matplotlib
  - torch (optional, falls back to NumPy)
  - seaborn (optional, for enhanced visualizations)
  - pillow (for image processing)
  
## Interpreting Test Results

Test results are reported in multiple formats:

1. **Console Output**: Real-time test progress with emoji indicators
2. **Log Files**: Detailed logs in `tests/results/neural_test.log`
3. **JSON Results**: Machine-readable results in `tests/results/neural_test_results.json`
4. **Visualizations**: Visual representations in the `visualizations/` directory
5. **HTML Report**: Comprehensive report in `visualizations/visualization_report.html`

### Understanding Error Metrics

- **MSE (Mean Squared Error)**: Lower is better, target is below 0.01
- **Inference Time**: Lower is better, measured in seconds
- **Confidence Score**: Higher is better, ranges from 0.0 to 1.0

## Advanced Usage

### Custom Test Data

You can provide custom test data by creating a JSON file:

```json
[
  {
    "input": "Your input text",
    "expectedOutput": "expected_label",
    "metadata": {
      "context": "use_case",
      "confidence": 0.95
    }
  }
]
```

Then run with:
```bash
./run_tests.sh --custom-data path/to/data.json --visualize
```

### Continuous Integration

The neural tests can be integrated into CI/CD pipelines:

```yaml
test_neural:
  script:
    - ./run_tests.sh --visualize --ci-mode
  artifacts:
    paths:
      - visualizations/
      - tests/results/
```

## Troubleshooting

1. **ImportError**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **CUDA Issues**: Set `export CUDA_VISIBLE_DEVICES=""` to force CPU mode
3. **Visualization Errors**: Ensure matplotlib and seaborn are properly installed
4. **Memory Errors**: Reduce batch size in the configuration file
5. **Slow Tests**: Use `--fallback` mode to reduce resource usage

## Reference

For more details, refer to:
- [API Documentation](./api_reference.md)
- [Model Architecture Guide](./model_architecture.md)
- [Visualization Guide](./visualizations.md) 