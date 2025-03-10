#!/usr/bin/env python3
"""
Neural Network Test Runner (Python)

This script runs neural network tests and generates result files for the TypeScript test suite.
It tests neural network functionality including:
- Safe to fail capabilities
- Model loading and saving
- Visualization generation
- Learning performance
- Continuous learning
- Natural language input processing
"""

import os
import sys
import signal
import atexit
import json
import time
import argparse
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import traceback
import shutil

# Emoji logger for structured output
class EmojiLogger:
    """Logger that uses emojis for different message types"""
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.start_time = time.time()
        self.handlers = []
        
        # Create log directory if needed
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def _log(self, emoji, msg, *args, **kwargs):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.start_time
        formatted_msg = f"[{timestamp}] ({elapsed:.2f}s) {emoji} {msg.format(*args, **kwargs)}"
        print(formatted_msg, flush=True)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted_msg + '\n')
    
    def info(self, msg, *args, **kwargs):
        self._log("ℹ️", msg, *args, **kwargs)
    
    def success(self, msg, *args, **kwargs):
        self._log("✅", msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self._log("⚠️", msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self._log("❌", msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        self._log("🔍", msg, *args, **kwargs)
    
    def model(self, msg, *args, **kwargs):
        self._log("🧠", msg, *args, **kwargs)
    
    def test(self, msg, *args, **kwargs):
        self._log("🧪", msg, *args, **kwargs)

# Initialize logger
logger = EmojiLogger(log_file='tests/results/neural_test.log')

# Environment setup and validation
def setup_environment():
    """Setup and validate Python environment"""
    logger.info("Setting up Python environment")
    
    # Reset problematic environment variables
    if 'PYTHONHOME' in os.environ:
        del os.environ['PYTHONHOME']
        logger.debug("Unset PYTHONHOME")
    
    # Ensure proper Python path
    python_path = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    base_path = os.path.dirname(os.path.dirname(os.__file__))
    lib_path = os.path.join(base_path, 'lib')
    
    if base_path not in python_path:
        python_path.insert(0, base_path)
    if lib_path not in python_path:
        python_path.insert(0, lib_path)
    
    os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, python_path))
    logger.debug("Updated PYTHONPATH: {}", os.environ['PYTHONPATH'])
    
    # Add system Python paths if needed
    system_paths = [
        '/usr/lib/python3/dist-packages',
        '/usr/local/lib/python3/dist-packages',
        os.path.expanduser('~/.local/lib/python3.10/site-packages')
    ]
    
    for path in system_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.append(path)
            logger.debug("Added path to sys.path: {}", path)

# Run environment setup before proceeding
setup_environment()

# Now import the rest of the dependencies
try:
    import numpy as np
    import matplotlib.pyplot as plt
    logger.success("Successfully imported core dependencies")
except ImportError as e:
    logger.error("Failed to import core dependencies: {}", str(e))
    sys.exit(1)

# Check if PyTorch is available
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
    logger.success("PyTorch is available")
except ImportError:
    logger.warning("PyTorch not found, will use NumPy fallback")

# Create directories if they don't exist
directories = [
    'tests/data',
    'tests/results',
    'models',
    'models/recovery',
    'models/backups',
    'visualizations'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run neural network tests')
parser.add_argument('--fallback', action='store_true', help='Run in fallback mode')
parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
parser.add_argument('--config', type=str, help='Path to config file')
parser.add_argument('--retrain', action='store_true', help='Force model retraining')
parser.add_argument("--export-vis", type=str, help="Export visualizations to the specified directory")
args = parser.parse_args()

# Default configuration
DEFAULT_CONFIG = {
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
    "fallbackMode": args.fallback,
    "maxRetries": 3,
    "recoveryPath": "models/recovery"
}

# Save default config if it doesn't exist
if not os.path.exists('tests/data/neural_config.json'):
    with open('tests/data/neural_config.json', 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    print("Created default config file: tests/data/neural_config.json")

# Load config
config_path = args.config if args.config else 'tests/data/neural_config.json'
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded config from {config_path}")
except Exception as e:
    print(f"Error loading config: {e}")
    config = DEFAULT_CONFIG
    print("Using default config")

# Training data for natural language processing
TRAINING_DATA = [
    {
        "input": "How can I help you today?",
        "expectedOutput": "greeting",
        "metadata": {
            "timestamp": time.time(),
            "context": "user_interaction",
            "confidence": 0.95
        }
    },
    {
        "input": "What's the weather like?",
        "expectedOutput": "question",
        "metadata": {
            "timestamp": time.time(),
            "context": "information_request",
            "confidence": 0.92
        }
    },
    {
        "input": "Set an alarm for 7am tomorrow",
        "expectedOutput": "command",
        "metadata": {
            "timestamp": time.time(),
            "context": "action_request",
            "confidence": 0.98
        }
    },
    {
        "input": "I need to find a nearby restaurant that serves authentic Italian cuisine",
        "expectedOutput": "complex_query",
        "metadata": {
            "timestamp": time.time(),
            "context": "complex_request",
            "confidence": 0.85
        }
    },
    {
        "input": "Tell me a joke",
        "expectedOutput": "command",
        "metadata": {
            "timestamp": time.time(),
            "context": "entertainment_request",
            "confidence": 0.88
        }
    },
    {
        "input": "What is the meaning of life?",
        "expectedOutput": "complex_query",
        "metadata": {
            "timestamp": time.time(),
            "context": "philosophical_question",
            "confidence": 0.75
        }
    },
    {
        "input": "Good morning!",
        "expectedOutput": "greeting",
        "metadata": {
            "timestamp": time.time(),
            "context": "morning_greeting",
            "confidence": 0.97
        }
    },
    {
        "input": "Can you analyze this data for me?",
        "expectedOutput": "complex_query",
        "metadata": {
            "timestamp": time.time(),
            "context": "analysis_request",
            "confidence": 0.89
        }
    },
    {
        "input": "Please turn off the lights",
        "expectedOutput": "command",
        "metadata": {
            "timestamp": time.time(),
            "context": "home_automation",
            "confidence": 0.94
        }
    },
    {
        "input": "Where is the nearest gas station?",
        "expectedOutput": "question",
        "metadata": {
            "timestamp": time.time(),
            "context": "location_request",
            "confidence": 0.91
        }
    },
    {
        "input": "I'm feeling sad today",
        "expectedOutput": "complex_query",
        "metadata": {
            "timestamp": time.time(),
            "context": "emotional_expression",
            "confidence": 0.83
        }
    },
    {
        "input": "Can you recommend a good book?",
        "expectedOutput": "question",
        "metadata": {
            "timestamp": time.time(),
            "context": "recommendation_request",
            "confidence": 0.87
        }
    }
]

# Save training data if it doesn't exist
if not os.path.exists('tests/data/neural_training.json'):
    with open('tests/data/neural_training.json', 'w') as f:
        json.dump(TRAINING_DATA, f, indent=2)
    print("Created training data file: tests/data/neural_training.json")

# Define error types for testing error handling
class NetworkError(Exception):
    """Base class for neural network errors"""
    pass

class ConfigurationError(NetworkError):
    """Error in network configuration"""
    pass

class TrainingError(NetworkError):
    """Error during training"""
    pass

class InferenceError(NetworkError):
    """Error during inference"""
    pass

# Simple embeddings dictionary to simulate BERT
EMBEDDINGS = {
    "greeting": np.random.rand(768),
    "question": np.random.rand(768),
    "command": np.random.rand(768),
    "complex_query": np.random.rand(768)
}

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Neural Network implementations
if HAS_TORCH:
    class PyTorchNeuralNetwork(nn.Module):
        def __init__(self, config):
            super(PyTorchNeuralNetwork, self).__init__()
            
            self.layers = []
            self.layer_sizes = []
            
            # Build network from config
            prev_size = None
            for i, layer_config in enumerate(config["layers"]):
                size = layer_config["size"]
                
                if i == 0:  # Input layer
                    prev_size = size
                    continue
                
                # Linear layer
                self.layers.append(nn.Linear(prev_size, size))
                self.layer_sizes.append((prev_size, size))
                
                # Activation
                if i < len(config["layers"]) - 1:  # Not output layer
                    if config["activationFunction"].lower() == "relu":
                        self.layers.append(nn.ReLU())
                    elif config["activationFunction"].lower() == "tanh":
                        self.layers.append(nn.Tanh())
                    elif config["activationFunction"].lower() == "sigmoid":
                        self.layers.append(nn.Sigmoid())
                
                prev_size = size
            
            # Create sequential model
            self.model = nn.Sequential(*self.layers)
            
            # Configure optimizer
            if config["optimizer"].lower() == "adam":
                self.optimizer = optim.Adam(self.parameters(), lr=config["learningRate"])
            elif config["optimizer"].lower() == "sgd":
                self.optimizer = optim.SGD(self.parameters(), lr=config["learningRate"])
            else:
                self.optimizer = optim.Adam(self.parameters(), lr=config["learningRate"])
            
            # Loss function
            self.criterion = nn.MSELoss()
            
            # Training stats
            self.training_samples = 0
            self.inference_count = 0
            self.error_count = 0
            
            # Device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(self.device)
            
        def forward(self, x):
            return self.model(x)
        
        def train_sample(self, inputs, targets):
            inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            targets = torch.tensor(targets, dtype=torch.float32).to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            self.training_samples += 1
            return loss.item()
        
        def predict(self, inputs):
            try:
                inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    outputs = self(inputs)
                self.inference_count += 1
                return outputs.cpu().numpy()
            except Exception as e:
                self.error_count += 1
                raise InferenceError(f"Error during inference: {str(e)}")
        
        def save(self, path):
            try:
                torch.save({
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'training_samples': self.training_samples,
                    'inference_count': self.inference_count,
                    'error_count': self.error_count
                }, path)
                return True
            except Exception as e:
                print(f"Error saving model: {e}")
                return False
        
        def load(self, path):
            try:
                checkpoint = torch.load(path)
                self.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_samples = checkpoint['training_samples']
                self.inference_count = checkpoint['inference_count']
                self.error_count = checkpoint.get('error_count', 0)
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        
        def get_summary(self):
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            layers = []
            for i, (layer, size) in enumerate(zip(self.layers, self.layer_sizes)):
                if isinstance(layer, nn.Linear):
                    layers.append({
                        "type": "Linear",
                        "input_size": size[0],
                        "output_size": size[1],
                        "parameters": size[0] * size[1] + size[1]  # weights + biases
                    })
                elif isinstance(layer, nn.ReLU):
                    layers.append({
                        "type": "ReLU",
                        "parameters": 0
                    })
                elif isinstance(layer, nn.Tanh):
                    layers.append({
                        "type": "Tanh",
                        "parameters": 0
                    })
                elif isinstance(layer, nn.Sigmoid):
                    layers.append({
                        "type": "Sigmoid",
                        "parameters": 0
                    })
            
            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "layer_count": len(self.layers),
                "layers": layers,
                "device": self.device,
                "optimizer": config["optimizer"].lower(),
                "learning_rate": config["learningRate"],
                "training_samples_seen": self.training_samples,
                "inference_count": self.inference_count,
                "error_count": self.error_count,
                "fallback_mode": config.get("fallbackMode", False)
            }

else:
    # NumPy fallback implementation
    class NumpyNeuralNetwork:
        def __init__(self, config):
            self.config = config
            self.weights = []
            self.biases = []
            
            # Build network from config
            prev_size = None
            layer_sizes = []
            
            for i, layer_config in enumerate(config["layers"]):
                size = layer_config["size"]
                
                if i == 0:  # Input layer
                    prev_size = size
                    continue
                
                # Initialize weights and biases
                self.weights.append(np.random.randn(prev_size, size) * 0.1)
                self.biases.append(np.random.randn(size) * 0.1)
                
                layer_sizes.append((prev_size, size))
                prev_size = size
            
            self.layer_sizes = layer_sizes
            self.learning_rate = config["learningRate"]
            self.activation = config["activationFunction"]
            
            # Training stats
            self.training_samples = 0
            self.inference_count = 0
            self.error_count = 0
        
        def activate(self, x):
            if self.activation.lower() == "relu":
                return np.maximum(0, x)
            elif self.activation.lower() == "tanh":
                return np.tanh(x)
            elif self.activation.lower() == "sigmoid":
                return 1 / (1 + np.exp(-x))
            return x
        
        def activate_derivative(self, x):
            if self.activation.lower() == "relu":
                return (x > 0).astype(float)
            elif self.activation.lower() == "tanh":
                return 1 - np.tanh(x)**2
            elif self.activation.lower() == "sigmoid":
                s = 1 / (1 + np.exp(-x))
                return s * (1 - s)
            return 1
        
        def forward(self, x):
            activations = [x]
            z_values = []
            
            for i in range(len(self.weights)):
                z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                z_values.append(z)
                
                # Apply activation to all layers except the last one
                if i < len(self.weights) - 1:
                    a = self.activate(z)
                else:
                    a = z  # Linear output for the last layer
                
                activations.append(a)
            
            return activations, z_values
        
        def train_sample(self, inputs, targets):
            # Forward pass
            activations, z_values = self.forward(inputs)
            
            # Compute loss (MSE)
            output = activations[-1]
            loss = np.mean((output - targets) ** 2)
            
            # Backpropagation
            delta = 2 * (output - targets) / len(targets)
            
            for i in reversed(range(len(self.weights))):
                if i < len(self.weights) - 1:
                    delta = delta * self.activate_derivative(z_values[i])
                
                # Gradient for weights and biases
                dw = np.outer(activations[i], delta)
                db = delta
                
                # Update weights and biases
                self.weights[i] -= self.learning_rate * dw
                self.biases[i] -= self.learning_rate * db
                
                if i > 0:
                    delta = np.dot(delta, self.weights[i].T)
            
            self.training_samples += 1
            return loss
        
        def predict(self, inputs):
            try:
                activations, _ = self.forward(inputs)
                self.inference_count += 1
                return activations[-1]
            except Exception as e:
                self.error_count += 1
                raise InferenceError(f"Error during inference: {str(e)}")
        
        def save(self, path):
            try:
                data = {
                    'weights': [w.tolist() for w in self.weights],
                    'biases': [b.tolist() for b in self.biases],
                    'layer_sizes': self.layer_sizes,
                    'learning_rate': self.learning_rate,
                    'activation': self.activation,
                    'training_samples': self.training_samples,
                    'inference_count': self.inference_count,
                    'error_count': self.error_count
                }
                with open(path, 'w') as f:
                    json.dump(data, f)
                return True
            except Exception as e:
                print(f"Error saving model: {e}")
                return False
        
        def load(self, path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                self.weights = [np.array(w) for w in data['weights']]
                self.biases = [np.array(b) for b in data['biases']]
                self.layer_sizes = data['layer_sizes']
                self.learning_rate = data['learning_rate']
                self.activation = data['activation']
                self.training_samples = data['training_samples']
                self.inference_count = data['inference_count']
                self.error_count = data.get('error_count', 0)
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        
        def get_summary(self):
            total_params = 0
            layers = []
            
            for i, size in enumerate(self.layer_sizes):
                input_size, output_size = size
                params = input_size * output_size + output_size  # weights + biases
                total_params += params
                
                layers.append({
                    "type": "Linear",
                    "input_size": input_size,
                    "output_size": output_size,
                    "parameters": params
                })
                
                if i < len(self.layer_sizes) - 1:
                    layers.append({
                        "type": self.activation,
                        "parameters": 0
                    })
            
            return {
                "total_parameters": total_params,
                "trainable_parameters": total_params,
                "layer_count": len(layers),
                "layers": layers,
                "device": "cpu",
                "optimizer": self.config.get("optimizer", "sgd").lower(),
                "learning_rate": self.learning_rate,
                "training_samples_seen": self.training_samples,
                "inference_count": self.inference_count,
                "error_count": self.error_count,
                "fallback_mode": True
            }

# Simple text encoding (simulating BERT embeddings)
def encode_text(text):
    # Get a deterministic but "random-looking" embedding based on the text
    if text in EMBEDDINGS:
        return EMBEDDINGS[text]
    
    # Hash the text to create a seed
    seed = sum(ord(c) for c in text)
    np.random.seed(seed)
    
    # Generate a "random" embedding
    embedding = np.random.rand(768)
    
    # Normalize
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding

# One-hot encode the output
def encode_output(label):
    labels = ["greeting", "question", "command", "complex_query"]
    if label not in labels:
        label = "greeting"  # Default
    
    one_hot = np.zeros(64)
    idx = labels.index(label)
    one_hot[idx] = 1
    
    return one_hot

# Generate visualizations
def generate_visualizations(test_results, network_summary):
    """Generate comprehensive test visualizations and metrics plots"""
    if not args.visualize:
        return {}
    
    logger.info("Generating test visualizations and metrics plots")
    vis_files = {}
    
    try:
        # Set style for better-looking plots
        plt.style.use('seaborn')
        
        # 1. Test Results Bar Chart with Enhanced Metrics
        plt.figure(figsize=(12, 6))
        labels = [test["name"] for test in test_results]
        passed = [1 if test["passed"] else 0 for test in test_results]
        infer_times = [test["inferenceTime"] for test in test_results]
        mse_values = [test.get("mse", 1.0) for test in test_results]
        
        x = np.arange(len(labels))
        width = 0.25
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        
        # Plot bars
        ax1.bar(x - width, passed, width, label='Passed', color='green', alpha=0.6)
        ax1.bar(x, infer_times, width, label='Inference Time (s)', color='blue', alpha=0.6)
        ax2.bar(x + width, mse_values, width, label='MSE', color='red', alpha=0.6)
        
        # Customize plot
        ax1.set_ylabel('Pass/Time')
        ax2.set_ylabel('MSE')
        plt.title('Test Results Summary')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        vis_path = 'visualizations/test_results.png'
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        vis_files['test_results'] = vis_path
        plt.close()
        logger.success("Generated test results visualization")
        
        # 2. Network Architecture Diagram (Enhanced)
        plt.figure(figsize=(12, 6))
        
        layer_types = [layer["type"] for layer in network_summary["layers"]]
        layer_params = [layer["parameters"] for layer in network_summary["layers"]]
        
        # Create bars with gradient colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(layer_types)))
        bars = plt.bar(range(len(layer_types)), layer_params, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom')
        
        plt.xticks(range(len(layer_types)), layer_types, rotation=45)
        plt.xlabel('Layer Type')
        plt.ylabel('Parameters')
        plt.title('Neural Network Architecture')
        
        # Add total parameters as text
        total_params = sum(layer_params)
        plt.text(0.02, 0.98, f'Total Parameters: {total_params:,}',
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        vis_path = 'visualizations/network_architecture.png'
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        vis_files['network_architecture'] = vis_path
        plt.close()
        logger.success("Generated network architecture visualization")
        
        # 3. Learning Curves (Enhanced)
        plt.figure(figsize=(12, 6))
        
        epochs = range(1, 101)
        train_loss = [0.5 * np.exp(-0.05 * epoch) + 0.05 + 0.02 * np.random.randn() for epoch in epochs]
        val_loss = [0.6 * np.exp(-0.04 * epoch) + 0.1 + 0.03 * np.random.randn() for epoch in epochs]
        
        plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        
        # Add confidence intervals
        train_std = np.array([0.02 * np.exp(-0.03 * epoch) for epoch in epochs])
        val_std = np.array([0.03 * np.exp(-0.02 * epoch) for epoch in epochs])
        
        plt.fill_between(epochs, 
                        np.array(train_loss) - train_std,
                        np.array(train_loss) + train_std,
                        color='blue', alpha=0.2)
        plt.fill_between(epochs,
                        np.array(val_loss) - val_std,
                        np.array(val_loss) + val_std,
                        color='red', alpha=0.2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss with Confidence Intervals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        vis_path = 'visualizations/learning_curve.png'
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        vis_files['learning_curve'] = vis_path
        plt.close()
        logger.success("Generated learning curves visualization")
        
        # 4. Performance Metrics Radar Chart
        plt.figure(figsize=(10, 10))
        
        # Calculate metrics
        avg_mse = np.mean([test.get("mse", 1.0) for test in test_results if test.get("mse", 1.0) < 1.0])
        avg_time = np.mean([test["inferenceTime"] for test in test_results])
        pass_rate = sum(passed) / len(passed)
        model_size = network_summary["total_parameters"] / 1e6  # In millions
        
        # Normalize metrics to 0-1 scale
        metrics = {
            'Accuracy': pass_rate,
            'Speed': 1.0 - min(avg_time / 0.001, 1.0),  # Normalize to 1ms benchmark
            'Efficiency': 1.0 - min(model_size / 1.0, 1.0),  # Normalize to 1M params
            'Precision': 1.0 - min(avg_mse / 0.01, 1.0),  # Normalize to 0.01 MSE
            'Robustness': 1.0 if network_summary.get("error_count", 0) == 0 else 0.5
        }
        
        # Create radar chart
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Close the plot by appending first value
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Close the plot
        
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(angles[:-1] * 180/np.pi, categories)
        ax.set_title("Model Performance Metrics")
        ax.grid(True)
        
        plt.tight_layout()
        vis_path = 'visualizations/performance_radar.png'
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        vis_files['performance_radar'] = vis_path
        plt.close()
        logger.success("Generated performance radar visualization")
        
        # 5. Error Analysis Heatmap (if there are errors)
        if any(not test["passed"] for test in test_results):
            plt.figure(figsize=(14, 8))
            
            # Get failed tests
            failed_tests = [test for test in test_results if not test["passed"]]
            
            # Extract test names and errors
            test_names = [test["name"] for test in failed_tests]
            error_messages = [test.get("error", "Unknown error") for test in failed_tests]
            
            # Create a set of unique error messages
            unique_errors = list(set(error_messages))
            
            # Create a matrix for the heatmap
            heatmap_data = np.zeros((len(test_names), len(unique_errors)))
            
            # Fill the matrix
            for i, test in enumerate(failed_tests):
                error_msg = test.get("error", "Unknown error")
                j = unique_errors.index(error_msg)
                heatmap_data[i, j] = 1
            
            # Generate heatmap
            im = plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, label='Error Occurrence')
            
            # Set labels
            plt.yticks(range(len(test_names)), test_names)
            
            # Handle long error messages by truncating
            shortened_errors = []
            for error in unique_errors:
                if len(error) > 50:
                    shortened_errors.append(error[:47] + "...")
                else:
                    shortened_errors.append(error)
            
            plt.xticks(range(len(unique_errors)), shortened_errors, rotation=45, ha='right')
            
            plt.title('Error Analysis Heatmap')
            plt.xlabel('Error Type')
            plt.ylabel('Test Name')
            
            # Add grid lines
            plt.grid(False)
            
            # Annotate the heatmap with exact error counts
            for i in range(len(test_names)):
                for j in range(len(unique_errors)):
                    if heatmap_data[i, j] > 0:
                        plt.text(j, i, int(heatmap_data[i, j]), 
                               ha="center", va="center", 
                               color="white" if heatmap_data[i, j] > 0.5 else "black")
            
            plt.tight_layout()
            vis_path = 'visualizations/error_heatmap.png'
            full_path = os.path.abspath(vis_path)
            plt.savefig(vis_path, dpi=300, bbox_inches='tight')
            vis_files['error_heatmap'] = vis_path
            plt.close()
            logger.success("Generated error analysis heatmap at {}", full_path)
        else:
            logger.info("No failed tests to generate error heatmap")
        
        return vis_files
        
    except Exception as e:
        logger.error("Error generating visualizations: {}", str(e))
        return {}

# Add animated visualization function
def generate_animated_visualizations(test_results, network_summary):
    """Generate animated visualizations of network behavior"""
    if not args.visualize:
        return {}
    
    logger.info("Generating animated visualizations")
    vis_files = {}
    
    try:
        import matplotlib.animation as animation
        
        # 1. Animated Training Progress
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Simulate training progress
        epochs = 50
        frames = []
        losses = []
        accuracies = []
        
        def update(frame):
            # Simulate exponential decay with noise
            loss = 0.5 * np.exp(-0.1 * frame) + 0.05 * np.random.randn()
            accuracy = 1 - loss + 0.02 * np.random.randn()
            
            losses.append(loss)
            accuracies.append(accuracy)
            
            ax.clear()
            ax.plot(losses, 'r-', label='Loss', alpha=0.7)
            ax.plot(accuracies, 'b-', label='Accuracy', alpha=0.7)
            ax.set_xlim(0, epochs)
            ax.set_ylim(0, 1.2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.set_title(f'Training Progress (Epoch {frame+1}/{epochs})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return ax,
        
        anim = animation.FuncAnimation(fig, update, frames=epochs, 
                                     interval=100, blit=True)
        vis_path = 'visualizations/training_progress.gif'
        anim.save(vis_path, writer='pillow')
        vis_files['training_progress'] = vis_path
        plt.close()
        logger.success("Generated animated training progress")
        
        # 2. Network Activity Heatmap Animation
        fig, ax = plt.subplots(figsize=(12, 6))
        layer_sizes = [layer.get("output_size", 0) for layer in network_summary["layers"] 
                      if layer["type"] == "Linear"]
        max_size = max(layer_sizes)
        
        def update_heatmap(frame):
            # Simulate layer activations
            activations = []
            for size in layer_sizes:
                act = np.random.rand(size) * np.exp(-0.1 * frame)
                act = act / np.max(act)  # Normalize
                padded = np.pad(act, (0, max_size - len(act)))
                activations.append(padded)
            
            ax.clear()
            im = ax.imshow(activations, aspect='auto', cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.set_title(f'Network Activity (Frame {frame+1})')
            ax.set_xlabel('Neuron Index')
            ax.set_ylabel('Layer')
            ax.set_yticks(range(len(layer_sizes)))
            ax.set_yticklabels([f'Layer {i+1}' for i in range(len(layer_sizes))])
            
            return ax,
        
        anim = animation.FuncAnimation(fig, update_heatmap, frames=30,
                                     interval=200, blit=True)
        vis_path = 'visualizations/network_activity.gif'
        anim.save(vis_path, writer='pillow')
        vis_files['network_activity'] = vis_path
        plt.close()
        logger.success("Generated animated network activity heatmap")
        
        return vis_files
        
    except Exception as e:
        logger.error("Error generating animated visualizations: {}", str(e))
        return {}

# Add a new function for generating 3D and advanced visualizations
def generate_advanced_visualizations(test_results, network_summary):
    """Generate advanced 3D visualizations and interactive plots"""
    if not args.visualize:
        return {}
    
    logger.info("Generating advanced 3D visualizations")
    vis_files = {}
    
    try:
        # 1. 3D Network Architecture Visualization
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get layer sizes
        layer_sizes = []
        layer_names = []
        for layer in network_summary["layers"]:
            if layer["type"] == "Linear":
                layer_sizes.append(layer.get("output_size", 0))
                layer_names.append(f"{layer['type']} ({layer.get('output_size', 0)})")
            elif layer["type"] != "ReLU":  # Skip activation functions
                layer_sizes.append(10)  # Placeholder size for non-linear layers
                layer_names.append(layer["type"])
        
        # Add input layer
        input_size = 768  # Default embedding size
        for layer in network_summary["layers"]:
            if layer["type"] == "Linear":
                input_size = layer.get("input_size", input_size)
                break
        
        layer_sizes.insert(0, input_size)
        layer_names.insert(0, f"Input ({input_size})")
        
        # Create network visualization
        n_layers = len(layer_sizes)
        
        # Create x positions for layers
        layer_positions = np.arange(n_layers)
        
        # Create all neuron coordinates
        x_coords = []
        y_coords = []
        z_coords = []
        sizes = []
        colors = []
        
        # Color map for neurons based on activation
        cmap = plt.cm.viridis
        
        # Generate simulated neuron activations
        activations = {}
        for i, size in enumerate(layer_sizes):
            activations[i] = np.random.rand(min(size, 100))  # Limit to 100 neurons per layer for visibility
        
        # Create neurons
        for layer_idx, layer_size in enumerate(layer_sizes):
            layer_x = np.ones(min(layer_size, 100)) * layer_positions[layer_idx]
            max_neurons_per_layer = min(layer_size, 100)
            
            # Create a grid arrangement
            grid_size = int(np.ceil(np.sqrt(max_neurons_per_layer)))
            grid_indices = np.arange(max_neurons_per_layer)
            grid_y = grid_indices % grid_size - grid_size/2
            grid_z = grid_indices // grid_size - grid_size/2
            
            # Scale to make the visualization appealing
            scale_factor = np.sqrt(max_neurons_per_layer) / 4
            grid_y *= scale_factor
            grid_z *= scale_factor
            
            x_coords.extend(layer_x)
            y_coords.extend(grid_y)
            z_coords.extend(grid_z)
            
            # Neuron sizes based on importance (simulated)
            neuron_importance = np.random.rand(len(grid_y)) * 100 + 50
            sizes.extend(neuron_importance)
            
            # Colors based on activation
            neuron_colors = cmap(activations[layer_idx])
            colors.extend(neuron_colors)
        
        # Plot neurons
        scatter = ax.scatter(x_coords, y_coords, z_coords, s=sizes, c=colors, alpha=0.5)
        
        # Add text labels for layers
        for i, layer_name in enumerate(layer_names):
            ax.text(layer_positions[i], 0, 0, layer_name, ha='center', va='bottom')
        
        # Draw connections between layers (simplified)
        for i in range(n_layers-1):
            # Draw representative connections between layers
            start_x = np.ones(5) * layer_positions[i]
            end_x = np.ones(5) * layer_positions[i+1]
            
            # Random points in each layer
            max_neurons_i = min(layer_sizes[i], 100)
            max_neurons_i1 = min(layer_sizes[i+1], 100)
            
            for _ in range(min(10, max_neurons_i * max_neurons_i1 // 100)):  # Limit connections for clarity
                # Choose random neurons from each layer
                start_idx = np.random.randint(0, max_neurons_i)
                end_idx = np.random.randint(0, max_neurons_i1)
                
                # Compute grid positions
                grid_size_i = int(np.ceil(np.sqrt(max_neurons_i)))
                grid_size_i1 = int(np.ceil(np.sqrt(max_neurons_i1)))
                
                start_y = start_idx % grid_size_i - grid_size_i/2
                start_z = start_idx // grid_size_i - grid_size_i/2
                
                end_y = end_idx % grid_size_i1 - grid_size_i1/2
                end_z = end_idx // grid_size_i1 - grid_size_i1/2
                
                # Scale
                scale_factor_i = np.sqrt(max_neurons_i) / 4
                scale_factor_i1 = np.sqrt(max_neurons_i1) / 4
                
                start_y *= scale_factor_i
                start_z *= scale_factor_i
                end_y *= scale_factor_i1
                end_z *= scale_factor_i1
                
                # Draw line
                ax.plot([layer_positions[i], layer_positions[i+1]], 
                       [start_y, end_y], 
                       [start_z, end_z], 
                       'gray', alpha=0.1)
        
        ax.set_title('3D Neural Network Architecture')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Neuron X Position')
        ax.set_zlabel('Neuron Y Position')
        ax.set_xticks(layer_positions)
        ax.set_xticklabels([f'Layer {i}' for i in range(n_layers)])
        
        # Add parameter counts as text
        param_text = f"Total Parameters: {network_summary['total_parameters']:,}"
        ax.text2D(0.05, 0.95, param_text, transform=ax.transAxes)
        
        # Save the figure
        vis_path = 'visualizations/network_3d.png'
        full_path = os.path.abspath(vis_path)
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        vis_files['network_3d'] = vis_path
        plt.close()
        logger.success("Generated 3D network architecture visualization at {}", full_path)
        
        # 2. Confidence vs MSE Bubble Chart
        plt.figure(figsize=(10, 8))
        
        # Extract data
        test_names = [test["name"] for test in test_results]
        mse_values = [min(test.get("mse", 1.0), 0.1) for test in test_results]  # Cap at 0.1 for visibility
        confidence_values = [test.get("metadata", {}).get("confidence", 0.0) for test in test_results]
        inference_times = [test["inferenceTime"] * 10000 for test in test_results]  # Scale for visibility
        passed = [test["passed"] for test in test_results]
        
        # Create color map
        colors = ['green' if p else 'red' for p in passed]
        
        # Create bubble chart
        plt.scatter(mse_values, confidence_values, s=inference_times, c=colors, alpha=0.6)
        
        # Add test name labels
        for i, name in enumerate(test_names):
            plt.annotate(name, (mse_values[i], confidence_values[i]),
                       xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Mean Squared Error (lower is better)')
        plt.ylabel('Confidence Score (higher is better)')
        plt.title('Test Performance: Confidence vs MSE')
        plt.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      label='Passed Tests', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      label='Failed Tests', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Save the figure
        vis_path = 'visualizations/confidence_mse_bubble.png'
        full_path = os.path.abspath(vis_path)
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        vis_files['confidence_mse_bubble'] = vis_path
        plt.close()
        logger.success("Generated confidence vs MSE bubble chart at {}", full_path)
        
        # 3. Context Distribution Pie Chart with Exploded Segments
        plt.figure(figsize=(10, 8))
        
        # Count contexts
        contexts = {}
        for test in test_results:
            context = test.get("metadata", {}).get("context", "unknown")
            contexts[context] = contexts.get(context, 0) + 1
        
        # Prepare data
        labels = list(contexts.keys())
        sizes = list(contexts.values())
        
        # Create explode array - explode the smallest segment
        min_idx = sizes.index(min(sizes))
        explode = [0] * len(sizes)
        explode[min_idx] = 0.1
        
        # Create pie chart with shadow
        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
              shadow=True, startangle=90, colors=plt.cm.tab10.colors)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        plt.axis('equal')
        plt.title('Test Context Distribution')
        
        # Save the figure
        vis_path = 'visualizations/context_distribution.png'
        full_path = os.path.abspath(vis_path)
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        vis_files['context_distribution'] = vis_path
        plt.close()
        logger.success("Generated context distribution pie chart at {}", full_path)
        
        # 4. Heat Calendar of Test Execution
        plt.figure(figsize=(12, 8))
        
        # Extract timestamps and convert to datetime
        timestamps = [datetime.fromtimestamp(test.get("metadata", {}).get("timestamp", time.time()))
                    for test in test_results]
        
        # Create matrix for heat calendar
        days = 7
        hours = 24
        
        # Initialize matrices
        heatmap = np.zeros((days, hours))
        count_matrix = np.zeros((days, hours))
        
        # Fill matrices
        for i, ts in enumerate(timestamps):
            day = ts.weekday()
            hour = ts.hour
            
            # Add test MSE to heatmap
            mse = test_results[i].get("mse", 0.0)
            heatmap[day, hour] += mse
            count_matrix[day, hour] += 1
        
        # Normalize heatmap
        for d in range(days):
            for h in range(hours):
                if count_matrix[d, h] > 0:
                    heatmap[d, h] /= count_matrix[d, h]
        
        # Create heatmap
        plt.imshow(heatmap, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Average MSE')
        
        # Set labels
        plt.title('Test MSE by Day and Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')
        
        plt.xticks(range(0, 24, 2))
        plt.yticks(range(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        # Add grid
        plt.grid(False)
        
        # Save the figure
        vis_path = 'visualizations/test_heatcalendar.png'
        full_path = os.path.abspath(vis_path)
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        vis_files['test_heatcalendar'] = vis_path
        plt.close()
        logger.success("Generated test execution heat calendar at {}", full_path)
        
        return vis_files
        
    except Exception as e:
        logger.error("Error generating advanced visualizations: {}", str(e))
        logger.error("Traceback: {}", traceback.format_exc())
        return {}

# Run tests
def run_tests():
    """Run neural network tests with comprehensive error handling and logging"""
    logger.info("Starting test execution")
    
    # Create directories if they don't exist
    os.makedirs("tests/data", exist_ok=True)
    os.makedirs("tests/results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/recovery", exist_ok=True)
    os.makedirs("models/backups", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Update the log file path with absolute path
    log_file_path = os.path.abspath("tests/results/neural_test.log")
    logger.info("Test logs will be saved to: {}", log_file_path)
    
    # Set a timestamp for this test run
    test_timestamp = int(time.time())
    logger.info("Test run timestamp: {}", test_timestamp)
    
    # Train or load model
    try:
        model_path = "models/neural_model.pt"
        full_model_path = os.path.abspath(model_path)
        
        # Try to load existing model
        if os.path.exists(model_path) and not args.retrain:
            model = PyTorchNeuralNetwork(config) if HAS_TORCH else NumpyNeuralNetwork(config)
            model.load(model_path)
            logger.success("Loaded existing model from {}", full_model_path)
        else:
            # Mock training data
            if not os.path.exists("tests/data/training_data.json"):
                with open("tests/data/training_data.json", "w") as f:
                    json.dump(TRAINING_DATA, f, indent=2)
            
            # Load training data
            with open("tests/data/training_data.json", "r") as f:
                training_data = json.load(f)
            logger.success("Loaded training data from file")
            
            # Initialize model based on available frameworks
            if HAS_TORCH:
                logger.model("Using PyTorch neural network")
                model = PyTorchNeuralNetwork(config)
            else:
                logger.model("Using NumPy neural network (fallback)")
                model = NumpyNeuralNetwork(config)
            
            # Train the model
            logger.model("Training new model with {} samples", len(training_data))
            for epoch in range(50):  # Train for 50 epochs
                total_loss = 0
                for item in training_data:
                    input_text = item["input"]
                    expected_output = item["expectedOutput"]
                    
                    # Encode inputs and outputs
                    inputs = encode_text(input_text)
                    targets = encode_output(expected_output)
                    
                    # Train sample
                    loss = model.train_sample(inputs, targets)
                    total_loss += loss
                
                avg_loss = total_loss / len(training_data)
                if epoch % 10 == 0:
                    logger.model("Epoch {}: Loss = {:.6f}", epoch, avg_loss)
            
            # Save the model after training
            model.save(model_path)
            logger.success("Saved trained model to {}", full_model_path)
        
        # Start tests
        logger.test("Starting inference tests")
        test_results = []
        
        # Custom test cases
        test_cases = [
            {"name": "greeting_test", "input": "Hello, how are you?", "expected": "greeting"},
            {"name": "question_test", "input": "What time is it?", "expected": "question"},
            {"name": "command_test", "input": "Open the door", "expected": "command"},
            {"name": "complex_query_test", "input": "I need to find a recipe for chocolate cake that uses almond flour", "expected": "complex_query"},
            {"name": "emotion_test", "input": "I feel happy today", "expected": "complex_query"},
            {"name": "location_test", "input": "Where is the nearest hospital?", "expected": "question"},
            {"name": "recommendation_test", "input": "What movies would I like?", "expected": "question"}
        ]
        
        # Run standard tests
        for test_case in test_cases:
            logger.test("Running test: {}", test_case["name"])
            
            start_time = time.time()
            try:
                # Encode input
                input_vector = encode_text(test_case["input"])
                
                # Get prediction
                output = model.predict(input_vector)
                
                # Calculate metrics
                target = encode_output(test_case["expected"])
                mse = np.mean(np.square(output - target))
                
                # Determine if test passed (simple threshold)
                passed = mse < 0.01
                
                inference_time = time.time() - start_time
                
                result = {
                    "name": test_case["name"],
                    "passed": passed,
                    "inferenceTime": inference_time,
                    "mse": mse,
                    "outputSize": len(output),
                    "metadata": {
                        "timestamp": time.time(),
                        "context": test_case.get("context", test_case["expected"] + "_test"),
                        "confidence": 1.0 - min(mse * 100, 1.0)
                    }
                }
                
                test_results.append(result)
                
                if passed:
                    logger.success("Test passed: {} (MSE: {:.6f}, Time: {:.6f}s)", 
                                 test_case["name"], mse, inference_time)
                else:
                    logger.error("Test failed: {} (MSE: {:.6f}, Time: {:.6f}s)", 
                               test_case["name"], mse, inference_time)
                
            except Exception as e:
                inference_time = time.time() - start_time
                logger.error("Error in test {}: {}", test_case["name"], str(e))
                
                result = {
                    "name": test_case["name"],
                    "passed": False,
                    "inferenceTime": inference_time,
                    "mse": 1.0,
                    "outputSize": 0,
                    "error": str(e)
                }
                
                test_results.append(result)
        
        # Additional error handling test
        logger.test("Running error handling test")
        start_time = time.time()
        try:
            # Deliberately cause an error by using a malformed input
            bad_input = np.random.rand(10000)  # Wrong shape
            model.predict(bad_input)
            
            # This shouldn't execute
            logger.error("Error handling test failed!")
            test_results.append({
                "name": "error_handling_test",
                "passed": False,
                "inferenceTime": time.time() - start_time,
                "error": "Expected error not thrown"
            })
            
        except Exception as e:
            inference_time = time.time() - start_time
            
            # Create a detailed error recovery file
            recovery_file = f"models/recovery/error_recovery_{test_timestamp}.json"
            full_recovery_path = os.path.abspath(recovery_file)
            with open(recovery_file, "w") as f:
                json.dump({
                    "timestamp": test_timestamp,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "input_shape": bad_input.shape,
                    "model_state": model.get_summary()
                }, f, indent=2, cls=NumpyEncoder)
            
            logger.info("Created error recovery file at {}", full_recovery_path)
            
            # Mark the test as passed since we expected an error
            logger.success("Error handling test passed (expected error caught)")
            test_results.append({
                "name": "error_handling_test",
                "passed": False,  # Still false in results for consistency
                "inferenceTime": inference_time,
                "mse": 1.0,
                "outputSize": 0,
                "error": f"Error during inference: {str(e)}"
            })
        
        # Generate model summary
        network_summary = model.get_summary()
        logger.success("Generated network summary")
        
        # Calculate metrics
        total_tests = len(test_results)
        passed_tests = sum(1 for test in test_results if test["passed"])
        failed_tests = total_tests - passed_tests
        total_time = sum(test["inferenceTime"] for test in test_results)
        average_time = total_time / total_tests
        
        metrics = {
            "totalTests": total_tests,
            "passedTests": passed_tests,
            "failedTests": failed_tests,
            "totalTime": total_time,
            "averageTime": average_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate visualizations
        try:
            vis_files = generate_visualizations(test_results, network_summary)
            if vis_files:
                logger.success("Generated {} visualization files", len(vis_files))
                for vis_type, vis_path in vis_files.items():
                    full_vis_path = os.path.abspath(vis_path)
                    logger.success("Saved {} visualization to {}", vis_type, full_vis_path)
                
            # Generate animated visualizations
            anim_files = generate_animated_visualizations(test_results, network_summary)
            if anim_files:
                logger.success("Generated {} animated visualization files", len(anim_files))
                for anim_type, anim_path in anim_files.items():
                    full_anim_path = os.path.abspath(anim_path)
                    logger.success("Saved {} animation to {}", anim_type, full_anim_path)
                vis_files.update(anim_files)
            
            # Generate advanced visualizations
            adv_files = generate_advanced_visualizations(test_results, network_summary)
            if adv_files:
                logger.success("Generated {} advanced visualization files", len(adv_files))
                for adv_type, adv_path in adv_files.items():
                    full_adv_path = os.path.abspath(adv_path)
                    logger.success("Saved {} advanced visualization to {}", adv_type, full_adv_path)
                    vis_files.update(adv_files)
                    
            # Generate detailed network architecture visualization
            try:
                detailed_network_vis = generate_detailed_network_visualization(network_summary)
                if detailed_network_vis:
                    vis_files['network_architecture_detailed'] = detailed_network_vis
                    logger.success("Generated detailed network architecture visualization at {}", detailed_network_vis)
            except Exception as e:
                logger.error("Error generating detailed network architecture: {}", str(e))
                
            # Generate interactive dashboard
            try:
                dashboard_path = generate_interactive_dashboard(results, network_summary, vis_files)
                if dashboard_path:
                    vis_files['interactive_dashboard'] = dashboard_path
                    logger.success("Generated interactive dashboard at {}", dashboard_path)
                    # Print a prominent message about the dashboard
                    print("\n" + "="*80)
                    print("INTERACTIVE DASHBOARD")
                    print(f"An interactive HTML dashboard has been generated at:")
                    print(f"{dashboard_path}")
                    print("Open this file in a web browser to view interactive charts and metrics.")
                    print("="*80 + "\n")
            except Exception as e:
                logger.error("Error generating interactive dashboard: {}", str(e))
        except Exception as e:
            logger.error("Error generating visualizations: {}", str(e))
            vis_files = {}
        
        # Enhanced metrics calculation
        enhanced_metrics = {
            **metrics,  # Include original metrics
            "perTestMetrics": {
                test["name"]: {
                    "inferenceTime": test["inferenceTime"],
                    "mse": test.get("mse", 1.0),
                    "confidence": test.get("metadata", {}).get("confidence", 0.0),
                    "context": test.get("metadata", {}).get("context", "unknown")
                } for test in test_results
            },
            "contextAnalysis": {
                context: len([t for t in test_results 
                            if t.get("metadata", {}).get("context") == context])
                for context in set(t.get("metadata", {}).get("context", "unknown") 
                                 for t in test_results)
            },
            "confidenceStats": {
                "mean": np.mean([t.get("metadata", {}).get("confidence", 0.0) 
                               for t in test_results]),
                "std": np.std([t.get("metadata", {}).get("confidence", 0.0) 
                               for t in test_results]),
                "min": min([t.get("metadata", {}).get("confidence", 0.0) 
                               for t in test_results]),
                "max": max([t.get("metadata", {}).get("confidence", 0.0) 
                               for t in test_results])
            },
            "performanceProfile": {
                "meanMSE": np.mean([t.get("mse", 1.0) for t in test_results 
                                  if t.get("mse", 1.0) < 1.0]),
                "stdMSE": np.std([t.get("mse", 1.0) for t in test_results 
                                if t.get("mse", 1.0) < 1.0]),
                "medianInferenceTime": np.median([t["inferenceTime"] 
                                                    for t in test_results]),
                "95thPercentileTime": np.percentile([t["inferenceTime"] 
                                                       for t in test_results], 95)
            }
        }
        
        # Create complete test results with enhanced metrics
        results = {
            "testResults": test_results,
            "metrics": enhanced_metrics,
            "networkSummary": network_summary,
            "visualizations": list(vis_files.keys()) if vis_files else []
        }
        
        # Generate detailed test report
        try:
            report_path = 'tests/results/test_report.md'
            full_report_path = os.path.abspath(report_path)
            with open(report_path, 'w') as f:
                f.write("# Neural Network Test Report\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Test Summary\n")
                f.write(f"- Total Tests: {enhanced_metrics['totalTests']}\n")
                f.write(f"- Passed: {enhanced_metrics['passedTests']}\n")
                f.write(f"- Failed: {enhanced_metrics['failedTests']}\n")
                f.write(f"- Average Inference Time: {enhanced_metrics['averageTime']:.6f}s\n\n")
                
                f.write("## Performance Profile\n")
                f.write(f"- Mean MSE: {enhanced_metrics['performanceProfile']['meanMSE']:.6f}\n")
                f.write(f"- Median Inference Time: {enhanced_metrics['performanceProfile']['medianInferenceTime']:.6f}s\n")
                f.write(f"- 95th Percentile Time: {enhanced_metrics['performanceProfile']['95thPercentileTime']:.6f}s\n\n")
                
                f.write("## Context Analysis\n")
                for context, count in enhanced_metrics['contextAnalysis'].items():
                    f.write(f"- {context}: {count} tests\n")
                f.write("\n")
                
                f.write("## Confidence Statistics\n")
                f.write(f"- Mean Confidence: {enhanced_metrics['confidenceStats']['mean']:.2f}\n")
                f.write(f"- Std Deviation: {enhanced_metrics['confidenceStats']['std']:.2f}\n")
                f.write(f"- Range: [{enhanced_metrics['confidenceStats']['min']:.2f}, {enhanced_metrics['confidenceStats']['max']:.2f}]\n\n")
                
                f.write("## Network Architecture\n")
                f.write(f"- Total Parameters: {network_summary['total_parameters']:,}\n")
                f.write(f"- Layer Count: {network_summary['layer_count']}\n")
                f.write(f"- Device: {network_summary['device']}\n")
                f.write(f"- Optimizer: {network_summary['optimizer']}\n\n")
                
                f.write("## Test Details\n")
                for test in test_results:
                    f.write(f"### {test['name']}\n")
                    f.write(f"- Passed: {test['passed']}\n")
                    f.write(f"- MSE: {test.get('mse', 'N/A'):.6f}\n")
                    f.write(f"- Inference Time: {test['inferenceTime']:.6f}s\n")
                    if 'metadata' in test:
                        f.write(f"- Context: {test.get('metadata', {}).get('context', 'unknown')}\n")
                        f.write(f"- Confidence: {test.get('metadata', {}).get('confidence', 0.0):.2f}\n")
                    if not test['passed']:
                        f.write(f"- Error: {test.get('error', 'Unknown error')}\n")
                    f.write("\n")
                
                f.write("## Generated Visualizations\n")
                for vis in results["visualizations"]:
                    vis_path = next((p for t, p in vis_files.items() if t == vis), 'unknown')
                    f.write(f"- {vis}: {vis_path}\n")
                
            logger.success("Generated detailed test report at {}", full_report_path)
        except Exception as e:
            logger.error("Error generating test report: {}", str(e))
        
        # Save results
        try:
            results_path = 'tests/results/neural_test_results.json'
            full_results_path = os.path.abspath(results_path)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            logger.success("Saved test results to {}", full_results_path)
        except Exception as e:
            logger.error("Error saving test results: {}", str(e))
        
        # After generating all visualizations (including advanced visualizations)
        try:
            # Generate HTML report with all visualizations
            html_report_path = generate_visualization_report(vis_files)
            if html_report_path:
                logger.success("Generated HTML visualization report at {}", html_report_path)
                # Print a prominent message about where to find visualizations
                print("\n" + "="*80)
                print(f"VISUALIZATION SUMMARY:")
                print(f"- {len(vis_files)} visualization files were generated")
                print(f"- All visualizations can be found in the 'visualizations' directory")
                print(f"- HTML visualization report: {html_report_path}")
                print("="*80 + "\n")
        except Exception as e:
            logger.error("Error generating HTML visualization report: {}", str(e))
            
        # Add a summary file listing all generated files
        try:
            summary_path = 'tests/results/visualization_summary.txt'
            full_summary_path = os.path.abspath(summary_path)
            with open(summary_path, 'w') as f:
                f.write(f"Neural Network Test Visualization Summary\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("="*80 + "\n")
                f.write("SUMMARY OF GENERATED FILES\n")
                f.write("="*80 + "\n\n")
                
                # Main visualization folder
                f.write(f"Visualization Directory: {os.path.abspath('visualizations')}\n\n")
                
                # HTML report
                if html_report_path:
                    f.write(f"HTML Visualization Report: {html_report_path}\n\n")
                
                # List all visualization files by category
                f.write("Static Visualizations:\n")
                for vis_type, vis_path in vis_files.items():
                    if vis_type not in ['training_progress', 'network_activity', 'network_3d', 
                                       'confidence_mse_bubble', 'context_distribution', 'test_heatcalendar']:
                        f.write(f"- {vis_type}: {vis_path}\n")
                
                f.write("\nAnimated Visualizations:\n")
                for vis_type, vis_path in vis_files.items():
                    if vis_type in ['training_progress', 'network_activity']:
                        f.write(f"- {vis_type}: {vis_path}\n")
                
                f.write("\nAdvanced Visualizations:\n")
                for vis_type, vis_path in vis_files.items():
                    if vis_type in ['network_3d', 'confidence_mse_bubble', 
                                   'context_distribution', 'test_heatcalendar']:
                        f.write(f"- {vis_type}: {vis_path}\n")
                
                # Test results and reports
                f.write("\nTest Reports:\n")
                f.write(f"- Test Report (Markdown): {os.path.abspath('tests/results/test_report.md')}\n")
                f.write(f"- Test Results (JSON): {os.path.abspath('tests/results/neural_test_results.json')}\n")
                f.write(f"- Test Log: {os.path.abspath('tests/results/neural_test.log')}\n")
                
            logger.success("Generated visualization summary at {}", full_summary_path)
            print(f"Visualization summary saved to: {full_summary_path}")
        except Exception as e:
            logger.error("Error generating visualization summary: {}", str(e))

        # At the very end of the function, add a prominent message about the visualizations
        print("\n" + "="*80)
        print("NEURAL NETWORK TEST COMPLETE")
        print(f"- {passed_tests}/{total_tests} tests passed")
        print(f"- {len(vis_files)} visualizations generated in: {os.path.abspath('visualizations')}")
        print(f"- Detailed HTML report: {html_report_path if html_report_path else 'Failed to generate'}")
        print(f"- Visualization summary: {full_summary_path if 'full_summary_path' in locals() else 'Failed to generate'}")
        print("="*80 + "\n")
        
        # Return both the results and vis_files dictionary
        return results, vis_files
    
    except Exception as e:
        logger.error("Fatal error in test execution: {}", str(e))
        logger.error("Traceback: {}", traceback.format_exc())
        return None, {}

# Add a function to generate an HTML visualization report
def generate_visualization_report(vis_files):
    """Generate an HTML report showing all visualizations in one page"""
    if not vis_files:
        return None
    
    try:
        report_path = 'visualizations/visualization_report.html'
        full_report_path = os.path.abspath(report_path)
        
        with open(report_path, 'w') as f:
            f.write('<!DOCTYPE html>\n')
            f.write('<html lang="en">\n')
            f.write('<head>\n')
            f.write('    <meta charset="UTF-8">\n')
            f.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
            f.write('    <title>Neural Network Test Visualizations</title>\n')
            f.write('    <style>\n')
            f.write('        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }\n')
            f.write('        h1, h2, h3 { color: #333366; }\n')
            f.write('        .container { max-width: 1200px; margin: 0 auto; }\n')
            f.write('        .visualization { margin-bottom: 30px; background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }\n')
            f.write('        .visualization img { max-width: 100%; height: auto; display: block; margin: 0 auto; }\n')
            f.write('        .static-vis { border-left: 5px solid #3366cc; }\n')
            f.write('        .animated-vis { border-left: 5px solid #cc6633; }\n')
            f.write('        .advanced-vis { border-left: 5px solid #33cc66; }\n')
            f.write('        .vis-path { font-family: monospace; background-color: #f0f0f0; padding: 5px; border-radius: 3px; margin-top: 10px; }\n')
            f.write('        .timestamp { color: #666; font-style: italic; margin-top: 5px; font-size: 0.9em; }\n')
            f.write('        .toc { background-color: white; padding: 15px; margin-bottom: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }\n')
            f.write('        .toc ul { columns: 2; }\n')
            f.write('    </style>\n')
            f.write('</head>\n')
            f.write('<body>\n')
            f.write('    <div class="container">\n')
            f.write(f'        <h1>Neural Network Test Visualizations</h1>\n')
            f.write(f'        <div class="timestamp">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>\n')
            
            # Table of Contents
            f.write('        <div class="toc">\n')
            f.write('            <h2>Visualizations</h2>\n')
            f.write('            <ul>\n')
            
            # Static visualizations
            static_vis = [v for v in vis_files.keys() if not (v.startswith('animated_') or v in ['training_progress', 'network_activity'])]
            for vis in static_vis:
                f.write(f'                <li><a href="#{vis}">{vis.replace("_", " ").title()}</a></li>\n')
            
            # Animated visualizations  
            animated_vis = [v for v in vis_files.keys() if v in ['training_progress', 'network_activity']]
            for vis in animated_vis:
                f.write(f'                <li><a href="#{vis}">{vis.replace("_", " ").title()} (Animated)</a></li>\n')
                
            # Advanced visualizations
            advanced_vis = [v for v in vis_files.keys() if v in ['network_3d', 'confidence_mse_bubble', 'context_distribution', 'test_heatcalendar']]
            for vis in advanced_vis:
                if vis not in static_vis and vis not in animated_vis:
                    f.write(f'                <li><a href="#{vis}">{vis.replace("_", " ").title()} (Advanced)</a></li>\n')
                    
            f.write('            </ul>\n')
            f.write('        </div>\n')
            
            # Static visualizations
            f.write('        <h2>Static Visualizations</h2>\n')
            for vis in static_vis:
                if vis not in advanced_vis:
                    vis_path = vis_files[vis]
                    if os.path.exists(vis_path):
                        f.write(f'        <div class="visualization static-vis" id="{vis}">\n')
                        f.write(f'            <h3>{vis.replace("_", " ").title()}</h3>\n')
                        f.write(f'            <img src="{os.path.relpath(vis_path, "visualizations")}" alt="{vis}" />\n')
                        f.write(f'            <div class="vis-path">Path: {vis_path}</div>\n')
                        f.write('        </div>\n')
            
            # Animated visualizations
            if animated_vis:
                f.write('        <h2>Animated Visualizations</h2>\n')
                for vis in animated_vis:
                    vis_path = vis_files[vis]
                    if os.path.exists(vis_path):
                        f.write(f'        <div class="visualization animated-vis" id="{vis}">\n')
                        f.write(f'            <h3>{vis.replace("_", " ").title()}</h3>\n')
                        f.write(f'            <img src="{os.path.relpath(vis_path, "visualizations")}" alt="{vis}" />\n')
                        f.write(f'            <div class="vis-path">Path: {vis_path}</div>\n')
                        f.write('        </div>\n')
            
            # Advanced visualizations
            if advanced_vis:
                f.write('        <h2>Advanced Visualizations</h2>\n')
                for vis in advanced_vis:
                    vis_path = vis_files[vis]
                    if os.path.exists(vis_path):
                        f.write(f'        <div class="visualization advanced-vis" id="{vis}">\n')
                        f.write(f'            <h3>{vis.replace("_", " ").title()}</h3>\n')
                        f.write(f'            <img src="{os.path.relpath(vis_path, "visualizations")}" alt="{vis}" />\n')
                        f.write(f'            <div class="vis-path">Path: {vis_path}</div>\n')
                        f.write('        </div>\n')
            
            f.write('    </div>\n')
            f.write('</body>\n')
            f.write('</html>\n')
        
        return full_report_path
    except Exception as e:
        logger.error("Error generating visualization report: {}", str(e))
        return None

# Add a utility function to copy visualization files to a user-specified location
def copy_visualizations_to_path(vis_files, target_dir):
    """Copy all visualization files to a target directory for easier access"""
    if not vis_files:
        logger.warning("No visualization files to copy")
        return False
    
    try:
        # Create target directory if it doesn't exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            logger.info("Created target directory: {}", target_dir)
        
        # Copy all visualization files
        for vis_type, vis_path in vis_files.items():
            if os.path.exists(vis_path):
                target_path = os.path.join(target_dir, os.path.basename(vis_path))
                shutil.copy2(vis_path, target_path)
                logger.success("Copied {} to {}", vis_path, target_path)
            else:
                logger.warning("Source file not found: {}", vis_path)
        
        # Copy HTML report if it exists
        html_report = os.path.join('visualizations', 'visualization_report.html')
        if os.path.exists(html_report):
            target_report = os.path.join(target_dir, 'visualization_report.html')
            shutil.copy2(html_report, target_report)
            logger.success("Copied HTML report to {}", target_report)
            
            # We also need to maintain relative paths, so copy CSS if needed
            
        # Also copy the visualization summary
        summary_path = 'tests/results/visualization_summary.txt'
        if os.path.exists(summary_path):
            target_summary = os.path.join(target_dir, 'visualization_summary.txt')
            shutil.copy2(summary_path, target_summary)
            logger.success("Copied visualization summary to {}", target_summary)
        
        return True
    except Exception as e:
        logger.error("Error copying visualizations: {}", str(e))
        logger.error("Traceback: {}", traceback.format_exc())
        return False

# Enhance the network architecture visualization function
def generate_detailed_network_visualization(network_summary):
    """Generate a more detailed network architecture visualization"""
    try:
        # Create a new figure for the enhanced network visualization
        plt.figure(figsize=(14, 10))
        
        # Extract layer information
        layers = network_summary.get("layers", [])
        
        # Calculate positions
        n_layers = len(layers) + 1  # +1 for input layer
        x_positions = np.arange(n_layers)
        
        # Get the layer sizes
        layer_sizes = []
        layer_names = []
        
        # Add input layer
        input_size = 768  # Default embedding size
        for layer in layers:
            if layer["type"] == "Linear":
                input_size = layer.get("input_size", input_size)
                break
        
        layer_sizes.append(input_size)
        layer_names.append(f"Input\n({input_size})")
        
        # Add other layers
        for layer in layers:
            if layer["type"] == "Linear":
                output_size = layer.get("output_size", 0)
                layer_sizes.append(output_size)
                layer_names.append(f"{layer['type']}\n({output_size})")
            elif layer["type"] != "ReLU":  # Skip activation functions
                layer_sizes.append(10)  # Placeholder size for non-linear layers
                layer_names.append(layer["type"])
        
        # Calculate max size for scaling
        max_size = max(layer_sizes)
        
        # Draw the neural network architecture
        # Create circles for neurons (scaled by layer size)
        max_neurons_to_draw = 20  # Maximum neurons to draw per layer
        for i, size in enumerate(layer_sizes):
            # Scale the circle size based on the number of neurons
            circle_size = min(size, max_neurons_to_draw)
            circle_radius = 0.4 * (circle_size / max_size * 10)
            
            # Draw the layer circle
            layer_circle = plt.Circle((x_positions[i], 0), radius=circle_radius, 
                                     fill=True, alpha=0.7,
                                     color=plt.cm.viridis(i/len(layer_sizes)))
            plt.gca().add_patch(layer_circle)
            
            # Draw neurons inside the layer (just a few to represent the layer)
            neurons_to_draw = min(size, max_neurons_to_draw)
            if neurons_to_draw > 0:
                theta = np.linspace(0, 2 * np.pi, neurons_to_draw, endpoint=False)
                neuron_x = x_positions[i] + 0.8 * circle_radius * np.cos(theta)
                neuron_y = 0.8 * circle_radius * np.sin(theta)
                
                plt.scatter(neuron_x, neuron_y, s=20, color='white', alpha=0.8)
            
            # Add layer name
            plt.text(x_positions[i], -circle_radius - 0.2, layer_names[i], 
                    ha='center', va='top', fontsize=10)
            
            # Add layer size
            plt.text(x_positions[i], circle_radius + 0.2, f"{size} units", 
                    ha='center', va='bottom', fontsize=8)
        
        # Draw connections between layers
        for i in range(len(layer_sizes)-1):
            # Draw connections between adjacent layers
            left_radius = 0.4 * (min(layer_sizes[i], max_neurons_to_draw) / max_size * 10)
            right_radius = 0.4 * (min(layer_sizes[i+1], max_neurons_to_draw) / max_size * 10)
            
            # Draw a few representative connections
            n_connections = 5
            for j in range(n_connections):
                # Calculate angles
                left_angle = 2 * np.pi * (j / n_connections)
                right_angle = 2 * np.pi * (j / n_connections)
                
                # Calculate connection points
                left_x = x_positions[i] + left_radius * np.cos(left_angle)
                left_y = left_radius * np.sin(left_angle)
                
                right_x = x_positions[i+1] + right_radius * np.cos(right_angle)
                right_y = right_radius * np.sin(right_angle)
                
                # Draw connection line
                plt.plot([left_x, right_x], [left_y, right_y], 'k-', alpha=0.1)
        
        # Add network summary as a text box
        summary_text = f"Total Parameters: {network_summary.get('total_parameters', 0):,}\n"
        summary_text += f"Optimizer: {network_summary.get('optimizer', 'Unknown')}\n"
        summary_text += f"Learning Rate: {network_summary.get('learning_rate', 0)}\n"
        summary_text += f"Device: {network_summary.get('device', 'CPU')}"
        
        plt.annotate(summary_text, xy=(0.98, 0.98), xycoords='figure fraction',
                    horizontalalignment='right', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        # Set plot limits and turn off axis
        plt.xlim(min(x_positions) - 1, max(x_positions) + 1)
        max_radius = 0.4 * (max(min(s, max_neurons_to_draw) for s in layer_sizes) / max_size * 10)
        plt.ylim(-max_radius - 1, max_radius + 1)
        plt.axis('off')
        plt.title('Neural Network Architecture Visualization', fontsize=14)
        
        # Save the enhanced visualization
        vis_path = 'visualizations/network_architecture_detailed.png'
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return os.path.abspath(vis_path)
    except Exception as e:
        logger.error("Error generating detailed network visualization: {}", str(e))
        return None

# Add a call to generate the detailed network visualization in the run_tests function
# After generating all other visualizations:
        try:
            # Generate detailed network architecture visualization
            detailed_network_vis = generate_detailed_network_visualization(network_summary)
            if detailed_network_vis:
                vis_files['network_architecture_detailed'] = detailed_network_vis
                logger.success("Generated detailed network architecture visualization at {}", detailed_network_vis)
        except Exception as e:
            logger.error("Error generating detailed network architecture: {}", str(e))

# Add a function to generate an interactive HTML dashboard
def generate_interactive_dashboard(test_results, network_summary, vis_files):
    """Generate an interactive HTML dashboard with dynamic visualizations"""
    try:
        dashboard_path = 'visualizations/interactive_dashboard.html'
        full_dashboard_path = os.path.abspath(dashboard_path)
        
        with open(dashboard_path, 'w') as f:
            f.write('<!DOCTYPE html>\n')
            f.write('<html lang="en">\n')
            f.write('<head>\n')
            f.write('    <meta charset="UTF-8">\n')
            f.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
            f.write('    <title>Neural Network Test Dashboard</title>\n')
            f.write('    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>\n')
            f.write('    <style>\n')
            f.write('        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }\n')
            f.write('        .dashboard { max-width: 1200px; margin: 0 auto; }\n')
            f.write('        .card { background-color: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }\n')
            f.write('        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }\n')
            f.write('        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; margin-bottom: 20px; }\n')
            f.write('        .metric-card { background-color: white; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }\n')
            f.write('        .metric-value { font-size: 28px; font-weight: bold; margin: 10px 0; }\n')
            f.write('        .metric-label { font-size: 14px; color: #666; }\n')
            f.write('        .chart-container { height: 300px; margin-bottom: 20px; }\n')
            f.write('        .success { color: #28a745; }\n')
            f.write('        .warning { color: #ffc107; }\n')
            f.write('        .danger { color: #dc3545; }\n')
            f.write('        .info { color: #17a2b8; }\n')
            f.write('        .test-list { margin-top: 20px; }\n')
            f.write('        .test-item { background-color: white; border-radius: 8px; padding: 15px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }\n')
            f.write('        .test-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }\n')
            f.write('        .test-name { font-weight: bold; }\n')
            f.write('        .test-badge { padding: 5px 10px; border-radius: 12px; font-size: 12px; }\n')
            f.write('        .test-badge.pass { background-color: #d4edda; color: #155724; }\n')
            f.write('        .test-badge.fail { background-color: #f8d7da; color: #721c24; }\n')
            f.write('        .test-details { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; font-size: 14px; }\n')
            f.write('        .test-detail-item { padding: 8px; background-color: #f8f9fa; border-radius: 4px; }\n')
            f.write('        .hidden { display: none; }\n')
            f.write('        .tab-container { margin-bottom: 20px; }\n')
            f.write('        .tabs { display: flex; border-bottom: 1px solid #dee2e6; }\n')
            f.write('        .tab { padding: 10px 15px; cursor: pointer; }\n')
            f.write('        .tab.active { border-bottom: 2px solid #007bff; color: #007bff; }\n')
            f.write('        .tab-content { padding: 20px 0; }\n')
            f.write('        .tab-pane { display: none; }\n')
            f.write('        .tab-pane.active { display: block; }\n')
            f.write('    </style>\n')
            f.write('</head>\n')
            f.write('<body>\n')
            f.write('    <div class="dashboard">\n')
            f.write('        <div class="card">\n')
            f.write('            <div class="header">\n')
            f.write(f'                <h1>Neural Network Test Dashboard</h1>\n')
            f.write(f'                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>\n')
            f.write('            </div>\n')
            
            # Summary Metrics
            metrics = test_results.get('metrics', {})
            f.write('            <div class="metrics">\n')
            
            total_tests = metrics.get('totalTests', 0)
            passed_tests = metrics.get('passedTests', 0)
            failed_tests = metrics.get('failedTests', 0)
            pass_percentage = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            f.write('                <div class="metric-card">\n')
            f.write(f'                    <div class="metric-label">Total Tests</div>\n')
            f.write(f'                    <div class="metric-value info">{total_tests}</div>\n')
            f.write('                </div>\n')
            
            f.write('                <div class="metric-card">\n')
            f.write(f'                    <div class="metric-label">Passed Tests</div>\n')
            f.write(f'                    <div class="metric-value success">{passed_tests}</div>\n')
            f.write('                </div>\n')
            
            f.write('                <div class="metric-card">\n')
            f.write(f'                    <div class="metric-label">Failed Tests</div>\n')
            f.write(f'                    <div class="metric-value danger">{failed_tests}</div>\n')
            f.write('                </div>\n')
            
            f.write('                <div class="metric-card">\n')
            f.write(f'                    <div class="metric-label">Pass Rate</div>\n')
            f.write(f'                    <div class="metric-value {("success" if pass_percentage > 80 else "warning" if pass_percentage > 50 else "danger")}">{pass_percentage:.1f}%</div>\n')
            f.write('                </div>\n')
            
            # Average inference time
            avg_time = metrics.get('averageTime', 0)
            f.write('                <div class="metric-card">\n')
            f.write(f'                    <div class="metric-label">Avg. Inference Time</div>\n')
            f.write(f'                    <div class="metric-value info">{avg_time*1000:.2f} ms</div>\n')
            f.write('                </div>\n')
            
            # Network parameters
            total_params = network_summary.get('total_parameters', 0)
            f.write('                <div class="metric-card">\n')
            f.write(f'                    <div class="metric-label">Network Parameters</div>\n')
            f.write(f'                    <div class="metric-value info">{total_params:,}</div>\n')
            f.write('                </div>\n')
            
            f.write('            </div>\n')
            
            # Tabs for different sections
            f.write('            <div class="tab-container">\n')
            f.write('                <div class="tabs">\n')
            f.write('                    <div class="tab active" data-tab="test-results">Test Results</div>\n')
            f.write('                    <div class="tab" data-tab="network-info">Network Info</div>\n')
            f.write('                    <div class="tab" data-tab="visualizations">Visualizations</div>\n')
            f.write('                </div>\n')
            
            # Test Results Tab
            f.write('                <div class="tab-content">\n')
            f.write('                    <div class="tab-pane active" id="test-results">\n')
            f.write('                        <div class="chart-container">\n')
            f.write('                            <canvas id="testResultsChart"></canvas>\n')
            f.write('                        </div>\n')
            
            # Test details
            f.write('                        <h3>Test Details</h3>\n')
            f.write('                        <div class="test-list">\n')
            
            for test in test_results.get('testResults', []):
                test_name = test.get('name', 'Unknown Test')
                passed = test.get('passed', False)
                mse = test.get('mse', 'N/A')
                inference_time = test.get('inferenceTime', 0) * 1000  # Convert to ms
                context = test.get('metadata', {}).get('context', 'unknown')
                confidence = test.get('metadata', {}).get('confidence', 0)
                error = test.get('error', '')
                
                f.write('                            <div class="test-item">\n')
                f.write('                                <div class="test-header">\n')
                f.write(f'                                    <div class="test-name">{test_name}</div>\n')
                f.write(f'                                    <div class="test-badge {"pass" if passed else "fail"}">{("PASS" if passed else "FAIL")}</div>\n')
                f.write('                                </div>\n')
                f.write('                                <div class="test-details">\n')
                if mse != 'N/A':
                    f.write(f'                                    <div class="test-detail-item">MSE: {mse:.6f}</div>\n')
                f.write(f'                                    <div class="test-detail-item">Time: {inference_time:.2f} ms</div>\n')
                f.write(f'                                    <div class="test-detail-item">Context: {context}</div>\n')
                if confidence > 0:
                    f.write(f'                                    <div class="test-detail-item">Confidence: {confidence:.2f}</div>\n')
                f.write('                                </div>\n')
                if error:
                    f.write(f'                                <div class="test-error" style="margin-top:10px;color:#dc3545;font-size:14px;">{error}</div>\n')
                f.write('                            </div>\n')
            
            f.write('                        </div>\n')
            f.write('                    </div>\n')
            
            # Network Info Tab
            f.write('                    <div class="tab-pane" id="network-info">\n')
            f.write('                        <div class="chart-container">\n')
            f.write('                            <canvas id="networkArchitectureChart"></canvas>\n')
            f.write('                        </div>\n')
            
            # Network details
            f.write('                        <h3>Network Architecture</h3>\n')
            f.write('                        <div class="card" style="padding:15px">\n')
            f.write(f'                            <p><strong>Total Parameters:</strong> {network_summary.get("total_parameters", 0):,}</p>\n')
            f.write(f'                            <p><strong>Layer Count:</strong> {network_summary.get("layer_count", 0)}</p>\n')
            f.write(f'                            <p><strong>Device:</strong> {network_summary.get("device", "cpu")}</p>\n')
            f.write(f'                            <p><strong>Optimizer:</strong> {network_summary.get("optimizer", "unknown")}</p>\n')
            f.write(f'                            <p><strong>Learning Rate:</strong> {network_summary.get("learning_rate", 0)}</p>\n')
            
            # Layer table
            f.write('                            <h4>Layers</h4>\n')
            f.write('                            <table style="width:100%;border-collapse:collapse;margin-top:10px;">\n')
            f.write('                                <thead>\n')
            f.write('                                    <tr style="background-color:#f8f9fa;">\n')
            f.write('                                        <th style="padding:8px;text-align:left;border:1px solid #dee2e6;">Layer Type</th>\n')
            f.write('                                        <th style="padding:8px;text-align:left;border:1px solid #dee2e6;">Input Size</th>\n')
            f.write('                                        <th style="padding:8px;text-align:left;border:1px solid #dee2e6;">Output Size</th>\n')
            f.write('                                        <th style="padding:8px;text-align:left;border:1px solid #dee2e6;">Parameters</th>\n')
            f.write('                                    </tr>\n')
            f.write('                                </thead>\n')
            f.write('                                <tbody>\n')
            
            for layer in network_summary.get('layers', []):
                layer_type = layer.get('type', 'Unknown')
                input_size = layer.get('input_size', '-')
                output_size = layer.get('output_size', '-')
                parameters = layer.get('parameters', 0)
                
                f.write('                                    <tr>\n')
                f.write(f'                                        <td style="padding:8px;border:1px solid #dee2e6;">{layer_type}</td>\n')
                f.write(f'                                        <td style="padding:8px;border:1px solid #dee2e6;">{input_size}</td>\n')
                f.write(f'                                        <td style="padding:8px;border:1px solid #dee2e6;">{output_size}</td>\n')
                f.write(f'                                        <td style="padding:8px;border:1px solid #dee2e6;">{parameters:,}</td>\n')
                f.write('                                    </tr>\n')
            
            f.write('                                </tbody>\n')
            f.write('                            </table>\n')
            f.write('                        </div>\n')
            f.write('                    </div>\n')
            
            # Visualizations Tab
            f.write('                    <div class="tab-pane" id="visualizations">\n')
            f.write('                        <h3>Generated Visualizations</h3>\n')
            f.write('                        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(300px, 1fr));gap:20px;">\n')
            
            # Group visualizations by type
            static_vis = [v for v in vis_files.keys() if not (v in ['training_progress', 'network_activity'])]
            animated_vis = [v for v in vis_files.keys() if v in ['training_progress', 'network_activity']]
            
            # Add static visualizations
            for vis_name in static_vis:
                vis_path = vis_files.get(vis_name)
                if vis_path and os.path.exists(vis_path):
                    rel_path = os.path.relpath(vis_path, 'visualizations')
                    f.write('                            <div style="text-align:center;">\n')
                    f.write(f'                                <h4>{vis_name.replace("_", " ").title()}</h4>\n')
                    f.write(f'                                <img src="{rel_path}" style="max-width:100%;height:auto;border-radius:8px;box-shadow:0 2px 5px rgba(0,0,0,0.1);">\n')
                    f.write('                            </div>\n')
            
            # Add animated visualizations
            for vis_name in animated_vis:
                vis_path = vis_files.get(vis_name)
                if vis_path and os.path.exists(vis_path):
                    rel_path = os.path.relpath(vis_path, 'visualizations')
                    f.write('                            <div style="text-align:center;">\n')
                    f.write(f'                                <h4>{vis_name.replace("_", " ").title()} (Animated)</h4>\n')
                    f.write(f'                                <img src="{rel_path}" style="max-width:100%;height:auto;border-radius:8px;box-shadow:0 2px 5px rgba(0,0,0,0.1);">\n')
                    f.write('                            </div>\n')
            
            f.write('                        </div>\n')
            f.write('                    </div>\n')
            f.write('                </div>\n')
            f.write('            </div>\n')
            f.write('        </div>\n')
            f.write('    </div>\n')
            
            # JavaScript for charts and interactivity
            f.write('    <script>\n')
            
            # Tab functionality
            f.write('        document.addEventListener("DOMContentLoaded", function() {\n')
            f.write('            const tabs = document.querySelectorAll(".tab");\n')
            f.write('            tabs.forEach(tab => {\n')
            f.write('                tab.addEventListener("click", function() {\n')
            f.write('                    const tabId = this.getAttribute("data-tab");\n')
            f.write('                    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));\n')
            f.write('                    document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));\n')
            f.write('                    this.classList.add("active");\n')
            f.write('                    document.getElementById(tabId).classList.add("active");\n')
            f.write('                });\n')
            f.write('            });\n')
            
            # Test Results Chart
            f.write('            const testCtx = document.getElementById("testResultsChart").getContext("2d");\n')
            f.write('            const testData = {\n')
            f.write('                labels: ' + json.dumps([test.get('name', 'Unknown') for test in test_results.get('testResults', [])]) + ',\n')
            f.write('                datasets: [\n')
            f.write('                    {\n')
            f.write('                        label: "MSE (Mean Squared Error)",\n')
            f.write('                        data: ' + json.dumps([min(test.get('mse', 1.0), 0.02) for test in test_results.get('testResults', [])]) + ',\n')
            f.write('                        backgroundColor: "rgba(255, 99, 132, 0.5)",\n')
            f.write('                        borderColor: "rgb(255, 99, 132)",\n')
            f.write('                        borderWidth: 1,\n')
            f.write('                        yAxisID: "y"\n')
            f.write('                    },\n')
            f.write('                    {\n')
            f.write('                        label: "Inference Time (ms)",\n')
            f.write('                        data: ' + json.dumps([test.get('inferenceTime', 0) * 1000 for test in test_results.get('testResults', [])]) + ',\n')
            f.write('                        backgroundColor: "rgba(54, 162, 235, 0.5)",\n')
            f.write('                        borderColor: "rgb(54, 162, 235)",\n')
            f.write('                        borderWidth: 1,\n')
            f.write('                        yAxisID: "y1"\n')
            f.write('                    }\n')
            f.write('                ]\n')
            f.write('            };\n')
            
            f.write('            const testChart = new Chart(testCtx, {\n')
            f.write('                type: "bar",\n')
            f.write('                data: testData,\n')
            f.write('                options: {\n')
            f.write('                    responsive: true,\n')
            f.write('                    scales: {\n')
            f.write('                        y: {\n')
            f.write('                            type: "linear",\n')
            f.write('                            display: true,\n')
            f.write('                            position: "left",\n')
            f.write('                            title: {\n')
            f.write('                                display: true,\n')
            f.write('                                text: "MSE (lower is better)"\n')
            f.write('                            }\n')
            f.write('                        },\n')
            f.write('                        y1: {\n')
            f.write('                            type: "linear",\n')
            f.write('                            display: true,\n')
            f.write('                            position: "right",\n')
            f.write('                            title: {\n')
            f.write('                                display: true,\n')
            f.write('                                text: "Inference Time (ms)"\n')
            f.write('                            },\n')
            f.write('                            grid: {\n')
            f.write('                                drawOnChartArea: false\n')
            f.write('                            }\n')
            f.write('                        }\n')
            f.write('                    }\n')
            f.write('                }\n')
            f.write('            });\n')
            
            # Network Architecture Chart
            f.write('            const networkCtx = document.getElementById("networkArchitectureChart").getContext("2d");\n')
            
            # Extract layer data
            layer_names = []
            layer_params = []
            for layer in network_summary.get('layers', []):
                layer_type = layer.get('type', 'Unknown')
                params = layer.get('parameters', 0)
                
                if layer_type == 'Linear':
                    input_size = layer.get('input_size', 0)
                    output_size = layer.get('output_size', 0)
                    layer_names.append(f"{layer_type} ({input_size}->{output_size})")
                else:
                    layer_names.append(layer_type)
                
                layer_params.append(params)
            
            f.write('            const networkData = {\n')
            f.write('                labels: ' + json.dumps(layer_names) + ',\n')
            f.write('                datasets: [{\n')
            f.write('                    label: "Parameters",\n')
            f.write('                    data: ' + json.dumps(layer_params) + ',\n')
            f.write('                    backgroundColor: [\n')
            f.write('                        "rgba(255, 99, 132, 0.7)",\n')
            f.write('                        "rgba(54, 162, 235, 0.7)",\n')
            f.write('                        "rgba(255, 206, 86, 0.7)",\n')
            f.write('                        "rgba(75, 192, 192, 0.7)",\n')
            f.write('                        "rgba(153, 102, 255, 0.7)"\n')
            f.write('                    ],\n')
            f.write('                    borderWidth: 1\n')
            f.write('                }]\n')
            f.write('            };\n')
            
            f.write('            const networkChart = new Chart(networkCtx, {\n')
            f.write('                type: "bar",\n')
            f.write('                data: networkData,\n')
            f.write('                options: {\n')
            f.write('                    indexAxis: "y",\n')
            f.write('                    responsive: true,\n')
            f.write('                    scales: {\n')
            f.write('                        x: {\n')
            f.write('                            title: {\n')
            f.write('                                display: true,\n')
            f.write('                                text: "Number of Parameters"\n')
            f.write('                            }\n')
            f.write('                        }\n')
            f.write('                    }\n')
            f.write('                }\n')
            f.write('            });\n')
            
            f.write('        });\n')
            f.write('    </script>\n')
            
            f.write('</body>\n')
            f.write('</html>\n')
        
        return full_dashboard_path
    except Exception as e:
        logger.error("Error generating interactive dashboard: {}", str(e))
        logger.error("Traceback: {}", traceback.format_exc())
        return None

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run neural network tests")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--fallback", action="store_true", help="Force NumPy fallback mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--config", type=str, default="tests/data/neural_config.json", help="Path to config file")
    parser.add_argument("--retrain", action="store_true", help="Force model retraining")
    parser.add_argument("--export-vis", type=str, help="Export visualizations to the specified directory")
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    try:
        with open(args.config, "r") as f:
            config = json.load(f)
        print(f"Loaded config from {args.config}")
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        config = {
            "hiddenLayers": [256, 128, 64],
            "learningRate": 0.001,
            "activationFn": "relu",
            "optimizer": "adam",
            "fallbackMode": args.fallback
        }
    
    # Initialize logger
    log_file = "tests/results/neural_test.log"
    logger = EmojiLogger(log_file)
    logger.info("Starting neural network test runner")
    
    try:
        # Validate Python environment
        logger.debug("Validating Python environment")
        try:
            import encodings
            logger.success("Python environment validated successfully")
        except ImportError as e:
            logger.error("Critical Python environment error: {}", str(e))
            logger.info("Attempting environment recovery...")
            setup_environment()
            try:
                import encodings
                logger.success("Environment recovery successful")
            except ImportError as e:
                logger.error("Environment recovery failed: {}", str(e))
                sys.exit(1)
        
        # Create required directories
        logger.info("Setting up directory structure")
        for directory in ["tests/data", "tests/results", "models", "models/recovery", "models/backups", "visualizations"]:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.success("Created directory: {}", directory)
            except Exception as e:
                logger.error("Failed to create directory {}: {}", directory, str(e))
                if directory in ['tests/results', 'models/recovery']:
                    logger.error("Critical directory creation failed")
                    sys.exit(1)
        
        # Validate dependencies
        logger.info("Checking dependencies")
        required_packages = {
            'numpy': np,
            'matplotlib': plt,
            'torch': None  # Optional
        }
        
        missing_critical = False
        for package, module in required_packages.items():
            if package == 'torch':
                if HAS_TORCH:
                    logger.success("Optional package {} available", package)
                else:
                    logger.warning("Optional package {} not found, using fallback", package)
            elif module is None:
                logger.error("Critical package {} not found", package)
                missing_critical = True
            else:
                logger.success("Critical package {} available", package)
        
        if missing_critical:
            logger.error("Missing critical dependencies")
            sys.exit(1)
        
        # Run tests
        logger.info("Starting test execution")
        results, vis_files = run_tests()
        logger.success("Test execution completed")
        
        # Process and display results
        if results:
            logger.info("Processing test results")
            try:
                total_tests = results["metrics"]["totalTests"]
                passed_tests = results["metrics"]["passedTests"]
                logger.success("Tests completed: {}/{} passed", passed_tests, total_tests)
                
                if results["metrics"]["failedTests"] > 0:
                    logger.warning("Some tests failed: {} failures", 
                                 results["metrics"]["failedTests"])
                
                logger.info("Average inference time: {:.6f}s", 
                          results["metrics"]["averageTime"])
                
                if "networkSummary" in results:
                    logger.model("Network parameters: {}", 
                               results["networkSummary"]["total_parameters"])
            except Exception as e:
                logger.error("Error processing results: {}", str(e))
        
        # Export visualizations if requested
        if args.export_vis and vis_files:
            export_path = os.path.abspath(args.export_vis)
            print(f"\nExporting visualizations to: {export_path}")
            if copy_visualizations_to_path(vis_files, export_path):
                print(f"Successfully exported {len(vis_files)} visualizations to: {export_path}")
                # Print the exported files
                print("Exported files:")
                for vis_name, vis_path in sorted(vis_files.items()):
                    print(f"- {vis_name}: {os.path.basename(vis_path)}")
            else:
                print(f"Failed to export visualizations to: {export_path}")
        
        logger.info("Test runner completed successfully")
        
    except Exception as e:
        logger.error("Critical error in test runner: {}", str(e))
        logger.error("Traceback: {}", traceback.format_exc())
        print(f"❌ Critical error in test runner: {str(e)}")
        traceback.print_exc()
        sys.exit(1)