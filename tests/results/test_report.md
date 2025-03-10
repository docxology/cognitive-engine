# Neural Network Test Report

Generated on: 2025-03-09 20:26:45

## Test Summary
- Total Tests: 8
- Passed: 0
- Failed: 8
- Average Inference Time: 0.001422s

## Performance Profile
- Mean MSE: 0.011586
- Median Inference Time: 0.000263s
- 95th Percentile Time: 0.006089s

## Context Analysis
- command_test: 1 tests
- complex_query_test: 2 tests
- unknown: 0 tests
- question_test: 3 tests
- greeting_test: 1 tests

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
- Inference Time: 0.001109s
- Context: greeting_test
- Confidence: 0.00
- Error: Unknown error

### question_test
- Passed: False
- MSE: 0.010114
- Inference Time: 0.000432s
- Context: question_test
- Confidence: 0.00
- Error: Unknown error

### command_test
- Passed: False
- MSE: 0.011288
- Inference Time: 0.000302s
- Context: command_test
- Confidence: 0.00
- Error: Unknown error

### complex_query_test
- Passed: False
- MSE: 0.010635
- Inference Time: 0.000225s
- Context: complex_query_test
- Confidence: 0.00
- Error: Unknown error

### emotion_test
- Passed: False
- MSE: 0.011322
- Inference Time: 0.000181s
- Context: complex_query_test
- Confidence: 0.00
- Error: Unknown error

### location_test
- Passed: False
- MSE: 0.012372
- Inference Time: 0.000179s
- Context: question_test
- Confidence: 0.00
- Error: Unknown error

### recommendation_test
- Passed: False
- MSE: 0.012658
- Inference Time: 0.000179s
- Context: question_test
- Confidence: 0.00
- Error: Unknown error

### error_handling_test
- Passed: False
- MSE: 1.000000
- Inference Time: 0.008770s
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
