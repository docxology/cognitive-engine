# Neural Network Test Report

Generated on: 2025-03-10 06:29:50

## Test Summary
- Total Tests: 8
- Passed: 0
- Failed: 8
- Average Inference Time: 0.001355s

## Performance Profile
- Mean MSE: 0.011586
- Median Inference Time: 0.000191s
- 95th Percentile Time: 0.005909s

## Context Analysis
- complex_query_test: 2 tests
- greeting_test: 1 tests
- question_test: 3 tests
- command_test: 1 tests
- unknown: 0 tests

## Confidence Statistics
- Mean Confidence: 0.00
- Std Deviation: 0.00
- Range: [0.00, 0.00]

## Network Architecture
- Total Parameters: 238,016
- Layer Count: 5
- Device: cpu
- Optimizer: adam

## Test Details
### greeting_test
- Passed: False
- MSE: 0.012715
- Inference Time: 0.001115s
- Context: greeting_test
- Confidence: 0.00
- Error: Unknown error

### question_test
- Passed: False
- MSE: 0.010114
- Inference Time: 0.000316s
- Context: question_test
- Confidence: 0.00
- Error: Unknown error

### command_test
- Passed: False
- MSE: 0.011288
- Inference Time: 0.000188s
- Context: command_test
- Confidence: 0.00
- Error: Unknown error

### complex_query_test
- Passed: False
- MSE: 0.010635
- Inference Time: 0.000194s
- Context: complex_query_test
- Confidence: 0.00
- Error: Unknown error

### emotion_test
- Passed: False
- MSE: 0.011322
- Inference Time: 0.000177s
- Context: complex_query_test
- Confidence: 0.00
- Error: Unknown error

### location_test
- Passed: False
- MSE: 0.012372
- Inference Time: 0.000176s
- Context: question_test
- Confidence: 0.00
- Error: Unknown error

### recommendation_test
- Passed: False
- MSE: 0.012658
- Inference Time: 0.000184s
- Context: question_test
- Confidence: 0.00
- Error: Unknown error

### error_handling_test
- Passed: False
- MSE: 1.000000
- Inference Time: 0.008491s
- Error: Error during inference: Error during inference: mat1 and mat2 shapes cannot be multiplied (1x10000 and 768x256)

## Generated Visualizations
- test_results: visualizations/test_results.png
- network_architecture: visualizations/network_architecture.png
- learning_curve: visualizations/learning_curve.png
- performance_radar: visualizations/performance_radar.png
- error_heatmap: visualizations/error_heatmap.png
- training_progress: visualizations/training_progress.gif
- network_activity: visualizations/network_activity.gif
- network_3d: visualizations/network_3d.png
- confidence_mse_bubble: visualizations/confidence_mse_bubble.png
- context_distribution: visualizations/context_distribution.png
- test_heatcalendar: visualizations/test_heatcalendar.png
- network_architecture_detailed: /home/trim/Documents/GitHub/cognitive-engine/visualizations/network_architecture_detailed.png
