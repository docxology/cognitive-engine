import argparse
import torch
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from .neural_network import ContinuousNeuralNetwork
import logging
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import traceback
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    handlers=[
        logging.FileHandler('neural_cli.log'),
        logging.StreamHandler()
    ]
)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load network configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file {config_path}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {str(e)}")
        raise

def create_default_config() -> Dict[str, Any]:
    """Create default network configuration"""
    config = {
        'layers': [768, 256, 128, 64],  # First layer must match BERT embedding size
        'learning_rate': 0.001,
        'batch_size': 32,
        'optimizer': 'adam',
        'activation': 'relu',
        # Safe-to-fail settings
        'max_retries': 3,
        'fallback_mode': True,
        'recovery_path': 'models/recovery'
    }
    logging.info("Created default configuration")
    return config

def save_config(config: Dict[str, Any], path: str) -> None:
    """Save network configuration to JSON file"""
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        logging.info(f"Saved configuration to {path}")
    except Exception as e:
        logging.error(f"Error saving config to {path}: {str(e)}")
        # Try to save to an alternative location
        alt_path = f"{path}.backup.{int(time.time())}.json"
        try:
            with open(alt_path, 'w') as f:
                json.dump(config, f, indent=2)
            logging.info(f"Saved configuration to alternative path: {alt_path}")
        except:
            logging.error(f"Failed to save config to alternative location")
            raise

def load_test_data(path: str) -> List[Dict[str, Any]]:
    """Load test data from JSON file"""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        logging.info(f"Loaded test data from {path}")
        return data
    except FileNotFoundError:
        logging.error(f"Test data file not found: {path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in test data file {path}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error loading test data from {path}: {str(e)}")
        raise

def save_test_results(results: Dict[str, Any], path: str) -> None:
    """Save test results to JSON file"""
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved test results to {path}")
    except Exception as e:
        logging.error(f"Error saving test results to {path}: {str(e)}")
        # Try to save to an alternative location
        alt_path = f"{path}.backup.{int(time.time())}.json"
        try:
            with open(alt_path, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Saved test results to alternative path: {alt_path}")
        except:
            logging.error(f"Failed to save test results to alternative location")
            raise

def create_default_test_data() -> List[Dict[str, Any]]:
    """Create default test data"""
    test_data = [
        {
            "name": "greeting_test",
            "input": "Hello, how are you?",
            "expected_output": [0.8, 0.1, 0.1]
        },
        {
            "name": "question_test",
            "input": "What is the weather like today?",
            "expected_output": [0.1, 0.8, 0.1]
        },
        {
            "name": "command_test",
            "input": "Set an alarm for 7am tomorrow",
            "expected_output": [0.1, 0.1, 0.8]
        },
        {
            "name": "complex_query_test",
            "input": "I need to find a restaurant that serves Italian food, is open late, and has good reviews",
            "expected_output": [0.05, 0.85, 0.1]
        },
        {
            "name": "error_handling_test",
            "input": "This is a very long text that should test the model's ability to handle lengthy inputs and potential errors that might occur during processing of complex and verbose text with many different concepts and ideas expressed in a continuous stream",
            "expected_output": [0.2, 0.2, 0.6]
        }
    ]
    logging.info("Created default test data with 5 test cases")
    return test_data

def print_network_summary(network: ContinuousNeuralNetwork) -> None:
    """Print a summary of the neural network"""
    try:
        summary = network.get_network_summary()
        
        if 'error' in summary:
            print(f"\n⚠️ Error getting full network summary: {summary['error']}")
            print(f"Device: {summary['device']}")
            print(f"Fallback mode: {summary['fallback_mode']}")
            return
        
        print("\n===== Neural Network Summary =====")
        print(f"Total parameters: {summary['total_parameters']:,}")
        print(f"Trainable parameters: {summary['trainable_parameters']:,}")
        print(f"Device: {summary['device']}")
        print(f"Optimizer: {summary['optimizer']}")
        print(f"Learning rate: {summary['learning_rate']}")
        print(f"Layer count: {summary['layer_count']}")
        
        print("\nLayer details:")
        for i, layer in enumerate(summary['layers']):
            if layer['type'] == 'Linear':
                print(f"  Layer {i}: {layer['type']} ({layer['input_size']} → {layer['output_size']}) - {layer['parameters']:,} parameters")
            else:
                print(f"  Layer {i}: {layer['type']}")
        
        print(f"\nTraining samples seen: {summary['training_samples_seen']:,}")
        print(f"Inference count: {summary['inference_count']:,}")
        
        if 'error_count' in summary and summary['error_count'] > 0:
            print(f"\n⚠️ Errors encountered: {summary['error_count']}")
            print(f"Fallback mode: {summary['fallback_mode']}")
        
        print("==================================\n")
    except Exception as e:
        print(f"\n⚠️ Error printing network summary: {str(e)}")

def run_tests(network: ContinuousNeuralNetwork, test_data: List[Dict[str, Any]], verbose: bool = True) -> Dict[str, Any]:
    """Run tests on the neural network and print results"""
    if verbose:
        print("\n===== Running Neural Network Tests =====")
    
    start_time = time.time()
    
    try:
        results = network.run_tests(test_data)
        total_time = time.time() - start_time
        
        if verbose:
            error_count = sum(1 for result in results.values() if 'error' in result)
            passed_count = sum(1 for result in results.values() if 'error' not in result and 
                              ('mse' not in result or result['mse'] < 0.1))
            total_tests = len(results)
            
            print(f"Completed {total_tests} tests in {total_time:.2f} seconds")
            print(f"Passed: {passed_count}/{total_tests} ({passed_count/total_tests*100:.1f}%)")
            
            if error_count > 0:
                print(f"⚠️ Errors: {error_count}/{total_tests} ({error_count/total_tests*100:.1f}%)")
            
            # Print individual test results
            for test_name, result in results.items():
                print(f"\nTest: {test_name}")
                
                if 'error' in result:
                    print(f"  ❌ Error: {result['error']}")
                    continue
                
                print(f"  Inference time: {result['inference_time']:.5f} seconds")
                
                if 'mse' in result:
                    print(f"  MSE: {result['mse']:.5f}")
                    score = 1.0 - min(result['mse'], 1.0)
                    status = "✅ PASS" if score > 0.9 else "❌ FAIL"
                    print(f"  Score: {score:.2f} ({status})")
                
                print(f"  Output shape: {np.array(result['output']).shape}")
                
                # Only print the first few values if output is large
                output = np.array(result['output'])
                if len(output) > 5:
                    print(f"  Output values: {output[:5]} ... (truncated)")
                else:
                    print(f"  Output values: {output}")
        
            print("=======================================\n")
        
        return results
    except Exception as e:
        total_time = time.time() - start_time
        logging.error(f"Error running tests: {str(e)}")
        logging.error(traceback.format_exc())
        
        if verbose:
            print(f"❌ Error running tests: {str(e)}")
            print(f"Time elapsed: {total_time:.2f} seconds")
            print("=======================================\n")
        
        # Return minimal results
        return {
            'error': str(e),
            'time_elapsed': total_time
        }

def bridge_to_typescript_tests(network: ContinuousNeuralNetwork, test_data: Optional[List[Dict[str, Any]]] = None) -> None:
    """
    Run tests and save results in format compatible with TypeScript tests
    If test_data is None, load test data from 'tests/data/neural_tests.json'
    """
    output_dir = Path('tests/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load test data if not provided
        if test_data is None:
            try:
                test_data = load_test_data('tests/data/neural_tests.json')
            except FileNotFoundError:
                print("Test data file not found. Creating default test data...")
                test_data = create_default_test_data()
                os.makedirs('tests/data', exist_ok=True)
                with open('tests/data/neural_tests.json', 'w') as f:
                    json.dump(test_data, f, indent=2)
        
        # Run tests
        results = run_tests(network, test_data, verbose=False)
        
        # Handle case where an error occurred during testing
        if 'error' in results:
            ts_results = {
                "testResults": [],
                "metrics": {
                    "totalTests": len(test_data),
                    "passedTests": 0,
                    "failedTests": len(test_data),
                    "totalTime": results.get('time_elapsed', 0),
                    "averageTime": 0,
                    "error": results['error']
                },
                "networkSummary": network.get_network_summary()
            }
        else:
            # Format results for TypeScript tests
            ts_results = {
                "testResults": [],
                "metrics": {
                    "totalTests": len(results),
                    "passedTests": sum(1 for r in results.values() if 'error' not in r and 
                                     ('mse' not in r or r['mse'] < 0.1)),
                    "failedTests": sum(1 for r in results.values() if 'error' in r or 
                                     ('mse' in r and r['mse'] >= 0.1)),
                    "totalTime": sum(r.get('inference_time', 0) for r in results.values()),
                    "averageTime": (sum(r.get('inference_time', 0) for r in results.values()) / 
                                   len(results)) if results else 0
                },
                "networkSummary": network.get_network_summary()
            }
            
            # Add individual test results
            for test_name, result in results.items():
                if 'error' in result:
                    # Format error results
                    ts_results["testResults"].append({
                        "name": test_name,
                        "passed": False,
                        "inferenceTime": 0,
                        "mse": 1.0,  # Maximum error
                        "outputSize": 0,
                        "error": result['error']
                    })
                else:
                    # Format successful results
                    passed = 'mse' not in result or result['mse'] < 0.1
                    ts_results["testResults"].append({
                        "name": test_name,
                        "passed": passed,
                        "inferenceTime": result['inference_time'],
                        "mse": result.get('mse', 0),
                        "outputSize": len(np.array(result['output']).flatten())
                    })
        
        # Save results
        output_path = output_dir / 'neural_test_results.json'
        with open(output_path, 'w') as f:
            json.dump(ts_results, f, indent=2)
        
        print(f"Test results saved to {output_path}")
        print(f"Total tests: {ts_results['metrics']['totalTests']}")
        print(f"Passed: {ts_results['metrics']['passedTests']}")
        print(f"Failed: {ts_results['metrics']['failedTests']}")
        
        if ts_results['metrics'].get('error'):
            print(f"⚠️ Error: {ts_results['metrics']['error']}")
        else:
            print(f"Average inference time: {ts_results['metrics']['averageTime']:.5f} seconds")
        
        # Also create a TypeScript-compatible config
        ts_config_dir = Path('tests/data')
        ts_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert Python-style config to TypeScript-style config
        if hasattr(network, 'config'):
            py_config = network.config
            ts_config = {
                "layers": [
                    {"size": 768, "type": "input"}  # Always include BERT size
                ]
            }
            
            # Add hidden layers
            for i, layer_size in enumerate(py_config.get('layers', [])[1:-1], 1):
                ts_config["layers"].append({
                    "size": layer_size,
                    "type": "hidden"
                })
            
            # Add output layer
            if len(py_config.get('layers', [])) > 1:
                ts_config["layers"].append({
                    "size": py_config['layers'][-1],
                    "type": "output"
                })
            
            # Add other properties
            ts_config["learningRate"] = py_config.get('learning_rate', 0.001)
            ts_config["batchSize"] = py_config.get('batch_size', 32)
            ts_config["optimizer"] = py_config.get('optimizer', 'adam')
            ts_config["activationFunction"] = py_config.get('activation', 'relu')
            
            # Save TypeScript config
            ts_config_path = ts_config_dir / 'ts_neural_config.json'
            with open(ts_config_path, 'w') as f:
                json.dump(ts_config, f, indent=2)
            
            print(f"TypeScript-compatible config saved to {ts_config_path}")
    
    except Exception as e:
        logging.error(f"Error bridging to TypeScript tests: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"❌ Error bridging to TypeScript tests: {str(e)}")
        
        # Try to save minimal error results
        try:
            minimal_results = {
                "testResults": [],
                "metrics": {
                    "totalTests": 0,
                    "passedTests": 0,
                    "failedTests": 0,
                    "totalTime": 0,
                    "averageTime": 0,
                    "error": str(e)
                },
                "networkSummary": {
                    "error": str(e),
                    "fallback_mode": network.fallback_mode if hasattr(network, 'fallback_mode') else False
                }
            }
            
            output_path = output_dir / 'neural_test_results.json'
            with open(output_path, 'w') as f:
                json.dump(minimal_results, f, indent=2)
            
            print(f"Minimal error results saved to {output_path}")
        except:
            logging.error("Failed to save minimal error results")

def generate_visualizations(network: ContinuousNeuralNetwork, output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Generate visualizations for the neural network
    Returns a dictionary mapping visualization types to file paths
    """
    print("\n===== Generating Neural Network Visualizations =====")
    
    try:
        # Generate visualizations
        viz_files = network.visualize_training(output_dir)
        
        # Print visualization file paths
        print(f"Generated {len(viz_files)} visualizations:")
        for viz_type, path in viz_files.items():
            print(f"  - {viz_type}: {path}")
        
        print("=================================================\n")
        return viz_files
    
    except Exception as e:
        logging.error(f"Error generating visualizations: {str(e)}")
        logging.error(traceback.format_exc())
        
        print(f"❌ Error generating visualizations: {str(e)}")
        print("=================================================\n")
        
        return {'error': str(e)}

def backup_model(model_path: str, backup_dir: str = 'models/backups') -> Optional[str]:
    """
    Create a backup of the model file
    Returns path to backup file if successful, None otherwise
    """
    try:
        if not os.path.exists(model_path):
            logging.warning(f"Cannot backup model: file not found {model_path}")
            return None
            
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"{os.path.basename(model_path)}.{timestamp}")
        shutil.copy2(model_path, backup_path)
        logging.info(f"Backed up model from {model_path} to {backup_path}")
        return backup_path
    
    except Exception as e:
        logging.error(f"Error backing up model: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Neural Network CLI')
    parser.add_argument('--config', type=str, help='Path to network configuration file')
    parser.add_argument('--model', type=str, help='Path to saved model file')
    parser.add_argument('--save-path', type=str, default='model.pt', help='Path to save model')
    parser.add_argument('--create-config', action='store_true', help='Create default config file')
    parser.add_argument('--tests', type=str, help='Path to test data file')
    parser.add_argument('--run-tests', action='store_true', help='Run tests and exit')
    parser.add_argument('--ts-bridge', action='store_true', help='Run tests and save results for TypeScript integration')
    parser.add_argument('--summary', action='store_true', help='Print network summary and exit')
    parser.add_argument('--visualize', action='store_true', help='Generate and save visualizations')
    parser.add_argument('--visualize-dir', type=str, help='Directory to save visualizations')
    parser.add_argument('--fallback', action='store_true', help='Enable fallback mode for handling errors')
    parser.add_argument('--backup', action='store_true', help='Create a backup of the model before saving')
    args = parser.parse_args()

    try:
        # Create default config if requested
        if args.create_config:
            config_path = 'network_config.json'
            config = create_default_config()
            save_config(config, config_path)
            print(f"Created default configuration file: {config_path}")
            
            # Also create default test data
            test_data_path = 'tests/data/neural_tests.json'
            test_data = create_default_test_data()
            os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
            with open(test_data_path, 'w') as f:
                json.dump(test_data, f, indent=2)
            print(f"Created default test data file: {test_data_path}")
            return

        # Load configuration
        if args.config:
            try:
                config = load_config(args.config)
            except Exception as e:
                print(f"Error loading config: {str(e)}")
                print("Using default configuration instead")
                config = create_default_config()
        else:
            config = create_default_config()

        # Enable fallback mode if requested
        if args.fallback:
            config['fallback_mode'] = True
            logging.info("Enabling fallback mode for error handling")

        # Initialize network
        try:
            print(f"Initializing neural network on {'fallback mode' if config.get('fallback_mode') else 'normal mode'}...")
            network = ContinuousNeuralNetwork(config)
        except Exception as e:
            print(f"❌ Error initializing network: {str(e)}")
            print("Try using --fallback option to enable fallback mode")
            return

        # Load existing model if specified
        if args.model and Path(args.model).exists():
            try:
                network.load_model(args.model)
                print(f"Loaded model from {args.model}")
            except Exception as e:
                print(f"❌ Error loading model: {str(e)}")
                print("Continuing with freshly initialized model")

        # Print network summary if requested
        if args.summary:
            print_network_summary(network)
            return

        # Load test data
        test_data = None
        if args.tests:
            try:
                test_data = load_test_data(args.tests)
            except Exception as e:
                print(f"Error loading test data: {str(e)}")
                if config.get('fallback_mode', False):
                    print("Creating default test data instead")
                    test_data = create_default_test_data()
                else:
                    sys.exit(1)
        
        # Run tests if requested
        if args.run_tests:
            if test_data is None:
                try:
                    test_data = load_test_data('tests/data/neural_tests.json')
                except FileNotFoundError:
                    print("Test data file not found. Creating default test data...")
                    test_data = create_default_test_data()
                    os.makedirs('tests/data', exist_ok=True)
                    with open('tests/data/neural_tests.json', 'w') as f:
                        json.dump(test_data, f, indent=2)
            
            run_tests(network, test_data)
            
            if args.visualize:
                generate_visualizations(network, args.visualize_dir)
            
            return
        
        # Bridge to TypeScript tests if requested
        if args.ts_bridge:
            bridge_to_typescript_tests(network, test_data)
            
            if args.visualize:
                generate_visualizations(network, args.visualize_dir)
                
            return
            
        # Generate visualizations if requested
        if args.visualize:
            generate_visualizations(network, args.visualize_dir)
            return

        print("\nNeural Network Interactive Mode")
        print("Type 'quit' to exit, 'save' to save the model, 'viz' to visualize training")
        print("Type 'test' to run tests, 'summary' to print network summary")
        print("Type 'help' to see all available commands")

        try:
            while True:
                user_input = input("\nEnter text or command: ").strip()
                
                if user_input.lower() == 'quit' or user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  quit, exit - Exit the program")
                    print("  save - Save the model")
                    print("  viz - Generate visualizations")
                    print("  test - Run tests")
                    print("  summary - Print network summary")
                    print("  backup - Create a backup of the current model")
                    print("  clear - Clear the screen")
                    print("  Any other text - Process through the neural network")
                    continue
                elif user_input.lower() == 'save':
                    # Backup existing model if requested
                    if args.backup:
                        backup_path = backup_model(args.save_path)
                        if backup_path:
                            print(f"Created backup at {backup_path}")
                    
                    try:
                        network.save_model(args.save_path)
                        print(f"Model saved to {args.save_path}")
                    except Exception as e:
                        print(f"❌ Error saving model: {str(e)}")
                    continue
                elif user_input.lower() == 'viz':
                    generate_visualizations(network, args.visualize_dir)
                    continue
                elif user_input.lower() == 'test':
                    if test_data is None:
                        try:
                            test_data = load_test_data('tests/data/neural_tests.json')
                        except FileNotFoundError:
                            print("Test data file not found. Creating default test data...")
                            test_data = create_default_test_data()
                            os.makedirs('tests/data', exist_ok=True)
                            with open('tests/data/neural_tests.json', 'w') as f:
                                json.dump(test_data, f, indent=2)
                    
                    run_tests(network, test_data)
                    continue
                elif user_input.lower() == 'summary':
                    print_network_summary(network)
                    continue
                elif user_input.lower() == 'backup':
                    backup_path = backup_model(args.save_path)
                    if backup_path:
                        print(f"Created backup at {backup_path}")
                    else:
                        print(f"No backup created. Check if {args.save_path} exists.")
                    continue
                elif user_input.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue

                # Process input through network
                try:
                    with torch.no_grad():
                        embeddings = network.embed_text(user_input)
                        output = network.forward(embeddings)
                        
                        print("\nNetwork Output:")
                        print(f"Raw output shape: {output.shape}")
                        print(f"Output values: {output.squeeze().tolist()}")
                        
                        # Additionally, get layer activations to visualize internal state
                        activations = network.get_layer_activations(user_input)
                        print(f"Layer activations: {len(activations)} layers with activations")
                        
                        # Visualize the first activation only if user requests
                        if len(activations) > 0 and input("\nVisualize activations? (y/n): ").lower() == 'y':
                            plt.figure(figsize=(10, 5))
                            act = activations[0].squeeze().numpy()
                            plt.bar(range(len(act)), act)
                            plt.title(f"Layer 1 Activations for: '{user_input}'")
                            plt.xlabel("Neuron")
                            plt.ylabel("Activation")
                            plt.tight_layout()
                            viz_dir = Path(args.visualize_dir) if args.visualize_dir else Path('visualizations')
                            viz_dir.mkdir(exist_ok=True)
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            viz_path = viz_dir / f'activation_{timestamp}.png'
                            plt.savefig(viz_path)
                            plt.close()
                            print(f"Activation visualization saved to {viz_path}")
                except Exception as e:
                    print(f"❌ Error processing input: {str(e)}")
                    if hasattr(network, 'metrics'):
                        network.metrics['error_count'] = network.metrics.get('error_count', 0) + 1
                        
                        # Track error type
                        error_type = type(e).__name__
                        if 'error_types' in network.metrics:
                            if error_type in network.metrics['error_types']:
                                network.metrics['error_types'][error_type] += 1
                            else:
                                network.metrics['error_types'][error_type] = 1

        except KeyboardInterrupt:
            print("\nExiting...")

        # Save final model state
        if args.backup:
            backup_path = backup_model(args.save_path)
            if backup_path:
                print(f"Created backup at {backup_path}")
                
        try:
            network.save_model(args.save_path)
            print(f"Final model state saved to {args.save_path}")
        except Exception as e:
            print(f"❌ Error saving final model state: {str(e)}")
            
    except Exception as e:
        logging.error(f"Unhandled exception in main: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"❌ Unhandled error: {str(e)}")

if __name__ == '__main__':
    main() 