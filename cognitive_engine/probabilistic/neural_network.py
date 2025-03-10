import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import json
import os
from datetime import datetime
from pathlib import Path
import traceback
import time
import sys

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    handlers=[
        logging.FileHandler('neural_network.log'),
        logging.StreamHandler()
    ]
)

class ContinuousNeuralNetwork(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the neural network with continuous learning capabilities
        
        Args:
            config: Dictionary containing network configuration
                - layers: List of layer sizes
                - learning_rate: Learning rate for optimization
                - batch_size: Batch size for training
                - optimizer: Optimizer type ('adam' or 'sgd')
                - activation: Activation function type
        """
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize safe-to-fail settings
        self.max_retries = config.get('max_retries', 3)
        self.fallback_mode = config.get('fallback_mode', False)
        self.recovery_path = config.get('recovery_path', 'models/recovery')
        
        # Create recovery directory
        os.makedirs(self.recovery_path, exist_ok=True)
        
        try:
            # Initialize BERT for text embedding
            logging.info("Initializing BERT tokenizer and model...")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.bert.to(self.device)
            
            # Build network layers
            self.layers = nn.ModuleList()
            prev_size = 768  # BERT embedding size
            
            for layer_size in config['layers'][1:]:  # Skip input layer size
                self.layers.append(nn.Linear(prev_size, layer_size))
                self.layers.append(self._get_activation())
                prev_size = layer_size
                
            # Initialize optimizer
            self.optimizer = self._get_optimizer()
            
            # Initialize metrics tracking
            self.metrics = {
                'loss_history': [],
                'accuracy_history': [],
                'confusion_matrix': {},
                'memory_usage': [],
                'inference_times': [],
                'last_inputs': [],
                'last_outputs': [],
                'training_time': 0.0,
                'inference_count': 0,
                'test_results': {},
                'error_count': 0,
                'error_types': {},
                'recovery_events': [],
                'performance_history': []
            }
            
            logging.info(f"Initialized neural network on device: {self.device}")
            logging.info(f"Network configuration: {json.dumps(self.config, indent=2)}")
        except Exception as e:
            self.handle_initialization_error(e)
        
    def handle_initialization_error(self, error: Exception) -> None:
        """Handle errors during initialization with fallback options"""
        error_msg = f"Error during initialization: {str(error)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        
        if self.fallback_mode:
            logging.warning("Entering fallback mode with minimal functionality")
            # Set up minimal functionality
            self.layers = nn.ModuleList([nn.Linear(768, 64), nn.ReLU()])
            self.optimizer = optim.Adam(self.parameters(), lr=0.001)
            
            # Initialize minimal metrics
            self.metrics = {
                'loss_history': [],
                'inference_times': [],
                'error_count': 1,
                'error_types': {str(type(error).__name__): 1},
                'recovery_events': [{'timestamp': datetime.now().isoformat(), 'error': str(error)}]
            }
        else:
            # Re-raise the error if not in fallback mode
            raise RuntimeError(f"Failed to initialize neural network: {error_msg}") from error
            
    def _get_activation(self) -> nn.Module:
        """Get activation function based on configuration"""
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        activation = self.config.get('activation', 'relu').lower()
        if activation not in activation_map:
            logging.warning(f"Unknown activation function '{activation}', using ReLU instead")
            return nn.ReLU()
        return activation_map[activation]
        
    def _get_optimizer(self) -> optim.Optimizer:
        """Get optimizer based on configuration"""
        params = self.parameters()
        if self.config.get('optimizer', '').lower() == 'adam':
            return optim.Adam(params, lr=self.config.get('learning_rate', 0.001))
        elif self.config.get('optimizer', '').lower() == 'adamw':
            return optim.AdamW(params, lr=self.config.get('learning_rate', 0.001))
        elif self.config.get('optimizer', '').lower() == 'sgd':
            return optim.SGD(params, lr=self.config.get('learning_rate', 0.001), 
                            momentum=self.config.get('momentum', 0.9))
        return optim.Adam(params, lr=self.config.get('learning_rate', 0.001))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        start_time = datetime.now()
        
        try:
            # Store intermediate activations for visualization
            activations = []
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    activations.append(x.detach().cpu())
            
            # Track inference time
            end_time = datetime.now()
            inference_time = (end_time - start_time).total_seconds()
            self.metrics['inference_times'].append(inference_time)
            self.metrics['inference_count'] += 1
            
            # Track performance metrics
            current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            self.metrics['performance_history'].append({
                'inference_time': inference_time,
                'memory_usage': current_memory,
                'timestamp': datetime.now().isoformat()
            })
            
            logging.debug(f"Forward pass completed in {inference_time:.5f}s, output shape: {x.shape}")
            return x
        except Exception as e:
            return self.handle_forward_error(e, x)
            
    def handle_forward_error(self, error: Exception, x: torch.Tensor) -> torch.Tensor:
        """Handle errors during forward pass with fallback options"""
        error_msg = f"Error during forward pass: {str(error)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        
        # Increment error count
        self.metrics['error_count'] = self.metrics.get('error_count', 0) + 1
        
        # Track error type
        error_type = type(error).__name__
        if error_type in self.metrics['error_types']:
            self.metrics['error_types'][error_type] += 1
        else:
            self.metrics['error_types'][error_type] = 1
            
        # Record recovery event
        self.metrics['recovery_events'].append({
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'type': error_type
        })
        
        if self.fallback_mode:
            # Return a default output in fallback mode
            output_size = self.layers[-2].out_features if isinstance(self.layers[-2], nn.Linear) else 64
            return torch.zeros((x.shape[0], output_size), device=self.device)
        else:
            # Re-raise the error if not in fallback mode
            raise RuntimeError(f"Forward pass failed: {error_msg}") from error
            
    def embed_text(self, text: str) -> torch.Tensor:
        """Convert text input to BERT embeddings"""
        try:
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.bert(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding
                
            logging.debug(f"Text embedded to shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            return self.handle_embedding_error(e, text)
            
    def handle_embedding_error(self, error: Exception, text: str) -> torch.Tensor:
        """Handle errors during text embedding with fallback options"""
        error_msg = f"Error during text embedding: {str(error)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        
        # Increment error count
        self.metrics['error_count'] = self.metrics.get('error_count', 0) + 1
        
        # Track error type
        error_type = type(error).__name__
        if error_type in self.metrics['error_types']:
            self.metrics['error_types'][error_type] += 1
        else:
            self.metrics['error_types'][error_type] = 1
            
        # Record recovery event
        self.metrics['recovery_events'].append({
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'type': error_type,
            'input_length': len(text)
        })
        
        if self.fallback_mode:
            # Return a default embedding in fallback mode
            # Generate a deterministic random embedding based on the text
            random_state = np.random.RandomState(hash(text) % 2**32)
            embedding = torch.tensor(
                random_state.normal(0, 0.1, (1, 768)), 
                dtype=torch.float32, device=self.device
            )
            return embedding
        else:
            # Re-raise the error if not in fallback mode
            raise RuntimeError(f"Text embedding failed: {error_msg}") from error
                
    def train_step(self, input_text: str, target: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single training step
        
        Args:
            input_text: Input text to process
            target: Target tensor
            
        Returns:
            Dictionary containing loss and accuracy metrics
        """
        try:
            self.train()
            self.optimizer.zero_grad()
            
            start_time = datetime.now()
            
            # Process input text
            embeddings = self.embed_text(input_text)
            output = self.forward(embeddings)
            
            # Calculate loss
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            self.optimizer.step()
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            self.metrics['training_time'] += training_time
            
            # Update metrics
            with torch.no_grad():
                accuracy = (output.argmax(dim=1) == target.argmax(dim=1)).float().mean()
                
            metrics = {
                'loss': loss.item(),
                'accuracy': accuracy.item(),
                'training_time': training_time
            }
            
            # Store recent inputs/outputs for debugging
            if len(self.metrics['last_inputs']) >= 5:
                self.metrics['last_inputs'].pop(0)
                self.metrics['last_outputs'].pop(0)
            
            self.metrics['last_inputs'].append(input_text)
            self.metrics['last_outputs'].append(output.detach().cpu().numpy().tolist())
            
            self._update_metrics(metrics)
            logging.info(f"Training step completed - Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
            
            # Auto-save recovery checkpoint periodically
            if len(self.metrics['loss_history']) % 100 == 0:
                recovery_path = os.path.join(self.recovery_path, f"recovery_{len(self.metrics['loss_history'])}.pt")
                self.save_model(recovery_path)
                logging.info(f"Auto-saved recovery checkpoint to {recovery_path}")
                
            return metrics
        except Exception as e:
            return self.handle_training_error(e, input_text)
            
    def handle_training_error(self, error: Exception, input_text: str) -> Dict[str, float]:
        """Handle errors during training with fallback options"""
        error_msg = f"Error during training step: {str(error)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        
        # Increment error count
        self.metrics['error_count'] = self.metrics.get('error_count', 0) + 1
        
        # Track error type
        error_type = type(error).__name__
        if error_type in self.metrics['error_types']:
            self.metrics['error_types'][error_type] += 1
        else:
            self.metrics['error_types'][error_type] = 1
            
        # Record recovery event
        self.metrics['recovery_events'].append({
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'type': error_type,
            'input': input_text[:100] + ('...' if len(input_text) > 100 else '')
        })
        
        # Try to save recovery checkpoint
        try:
            recovery_path = os.path.join(self.recovery_path, f"error_recovery_{int(time.time())}.pt")
            self.save_model(recovery_path)
            logging.info(f"Saved error recovery checkpoint to {recovery_path}")
        except Exception as save_error:
            logging.error(f"Failed to save recovery checkpoint: {save_error}")
            
        if self.fallback_mode:
            # Return a default metrics in fallback mode
            return {
                'loss': 1.0,
                'accuracy': 0.0,
                'training_time': 0.0,
                'error': str(error)
            }
        else:
            # Re-raise the error if not in fallback mode
            raise RuntimeError(f"Training step failed: {error_msg}") from error
            
    def _update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update tracking metrics"""
        self.metrics['loss_history'].append(metrics['loss'])
        self.metrics['accuracy_history'].append(metrics['accuracy'])
        self.metrics['memory_usage'].append(torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)
        
    def visualize_training(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate and save training visualization plots
        
        Args:
            output_dir: Optional directory to save visualizations
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(output_dir) if output_dir else Path('visualizations')
        output_dir.mkdir(exist_ok=True)
        
        visualization_files = {}
        
        try:
            # 1. Basic training metrics (loss, accuracy)
            fig1, ax1 = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot loss history
            ax1[0].plot(self.metrics['loss_history'])
            ax1[0].set_title('Training Loss')
            ax1[0].set_xlabel('Step')
            ax1[0].set_ylabel('Loss')
            ax1[0].grid(True)
            
            # Plot accuracy history
            ax1[1].plot(self.metrics['accuracy_history'])
            ax1[1].set_title('Training Accuracy')
            ax1[1].set_xlabel('Step')
            ax1[1].set_ylabel('Accuracy')
            ax1[1].grid(True)
            
            plt.tight_layout()
            training_metrics_file = output_dir / f'training_metrics_{timestamp}.png'
            plt.savefig(training_metrics_file)
            visualization_files['training_metrics'] = str(training_metrics_file)
            plt.close(fig1)
            
            # 2. Performance metrics (memory, inference time)
            fig2, ax2 = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot memory usage
            ax2[0].plot(self.metrics['memory_usage'])
            ax2[0].set_title('Memory Usage')
            ax2[0].set_xlabel('Step')
            ax2[0].set_ylabel('Memory (bytes)')
            ax2[0].grid(True)
            
            # Plot inference times
            if self.metrics['inference_times']:
                ax2[1].plot(self.metrics['inference_times'])
                ax2[1].set_title('Inference Times')
                ax2[1].set_xlabel('Inference #')
                ax2[1].set_ylabel('Time (seconds)')
                ax2[1].grid(True)
            
            plt.tight_layout()
            performance_metrics_file = output_dir / f'performance_metrics_{timestamp}.png'
            plt.savefig(performance_metrics_file)
            visualization_files['performance_metrics'] = str(performance_metrics_file)
            plt.close(fig2)
            
            # 3. Network architecture visualization
            fig3, ax3 = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot layer weights distribution
            weights = []
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    weights.extend(layer.weight.data.cpu().numpy().flatten())
            
            if weights:
                ax3[0].hist(weights, bins=50)
                ax3[0].set_title('Weight Distribution')
                ax3[0].set_xlabel('Weight Value')
                ax3[0].set_ylabel('Frequency')
                ax3[0].grid(True)
            
            # Plot layer sizes
            layer_sizes = []
            layer_names = []
            for i, layer in enumerate(self.layers):
                if isinstance(layer, nn.Linear):
                    layer_sizes.append(layer.out_features)
                    layer_names.append(f"Layer {i//2+1}")
            
            if layer_sizes:
                ax3[1].bar(layer_names, layer_sizes)
                ax3[1].set_title('Layer Sizes')
                ax3[1].set_ylabel('Size')
                ax3[1].grid(True)
            
            plt.tight_layout()
            architecture_file = output_dir / f'architecture_{timestamp}.png'
            plt.savefig(architecture_file)
            visualization_files['architecture'] = str(architecture_file)
            plt.close(fig3)
            
            # 4. Error and recovery visualization
            if self.metrics.get('error_count', 0) > 0:
                fig4, ax4 = plt.subplots(1, 1, figsize=(12, 6))
                
                # Plot error types
                error_types = list(self.metrics.get('error_types', {}).keys())
                error_counts = [self.metrics['error_types'][t] for t in error_types]
                
                if error_types:
                    ax4.bar(error_types, error_counts)
                    ax4.set_title('Error Types')
                    ax4.set_ylabel('Count')
                    ax4.grid(True)
                    plt.xticks(rotation=45, ha='right')
                    
                    plt.tight_layout()
                    errors_file = output_dir / f'errors_{timestamp}.png'
                    plt.savefig(errors_file)
                    visualization_files['errors'] = str(errors_file)
                    plt.close(fig4)
            
            # Save metrics to JSON for external analysis
            metrics_file = output_dir / f'metrics_{timestamp}.json'
            
            # Create a serializable copy of metrics
            serializable_metrics = {}
            for k, v in self.metrics.items():
                if k == 'last_outputs':
                    serializable_metrics[k] = 'omitted'  # Skip non-serializable tensor data
                elif k == 'confusion_matrix' and isinstance(v, dict):
                    # Convert keys to strings for JSON serialization
                    serializable_metrics[k] = {str(k2): v2 for k2, v2 in v.items()}
                elif k == 'recovery_events':
                    # Already serializable, just copy
                    serializable_metrics[k] = v
                else:
                    # Handle other non-serializable data
                    try:
                        json.dumps({k: v})  # Test if it's serializable
                        serializable_metrics[k] = v
                    except (TypeError, OverflowError):
                        serializable_metrics[k] = str(v)
            
            with open(metrics_file, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
                
            visualization_files['metrics_json'] = str(metrics_file)
            
            logging.info(f"Saved training visualizations to {output_dir}")
            for name, path in visualization_files.items():
                logging.info(f"  - {name}: {path}")
                
            return visualization_files
            
        except Exception as e:
            error_msg = f"Error generating visualizations: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            
            # Try to create a simple error report visualization
            try:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, f"Visualization error: {str(e)}", 
                         ha='center', va='center', fontsize=12)
                plt.axis('off')
                error_file = output_dir / f'visualization_error_{timestamp}.png'
                plt.savefig(error_file)
                plt.close()
                
                visualization_files['error'] = str(error_file)
                return visualization_files
            except:
                logging.error("Failed to create error visualization")
                return {'error': error_msg}
        
    def save_model(self, path: str) -> None:
        """Save model state and metrics"""
        try:
            state = {
                'model_state': self.state_dict(),
                'config': self.config,
                'metrics': {k: v for k, v in self.metrics.items() 
                           if k not in ['last_outputs']}  # Skip tensor data
            }
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            torch.save(state, path)
            logging.info(f"Saved model state to {path}")
        except Exception as e:
            error_msg = f"Error saving model: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            
            # Try to save to alternative location
            try:
                alt_path = f"{path}.backup.{int(time.time())}.pt"
                torch.save(state, alt_path)
                logging.info(f"Saved model state to alternative path: {alt_path}")
            except:
                logging.error("Failed to save model to alternative location")
        
    def load_model(self, path: str) -> None:
        """Load model state and metrics"""
        try:
            state = torch.load(path, map_location=self.device)
            self.load_state_dict(state['model_state'])
            self.config = state['config']
            self.metrics = state['metrics']
            logging.info(f"Loaded model state from {path}")
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            
            # Try to find recovery file
            recovery_files = sorted(Path(self.recovery_path).glob("*.pt"), key=os.path.getmtime, reverse=True)
            if recovery_files and self.fallback_mode:
                try:
                    latest_recovery = str(recovery_files[0])
                    logging.info(f"Attempting to load recovery file: {latest_recovery}")
                    state = torch.load(latest_recovery, map_location=self.device)
                    self.load_state_dict(state['model_state'])
                    self.config = state['config']
                    self.metrics = state['metrics']
                    logging.info(f"Loaded model from recovery file: {latest_recovery}")
                except Exception as recovery_error:
                    logging.error(f"Failed to load recovery file: {recovery_error}")
                    raise RuntimeError(f"Failed to load model and recovery file") from e
            else:
                raise RuntimeError(f"Failed to load model: {error_msg}") from e
    
    def run_tests(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a series of tests on the neural network
        
        Args:
            test_data: List of test cases with input and expected output
        
        Returns:
            Dictionary with test results
        """
        self.eval()
        results = {}
        
        for test_case in test_data:
            test_name = test_case.get('name', 'unnamed_test')
            input_text = test_case.get('input', '')
            expected = test_case.get('expected_output', None)
            
            try:
                start_time = datetime.now()
                
                # Process input
                with torch.no_grad():
                    embeddings = self.embed_text(input_text)
                    output = self.forward(embeddings)
                    
                end_time = datetime.now()
                inference_time = (end_time - start_time).total_seconds()
                
                # Compare with expected output if provided
                if expected is not None:
                    if isinstance(expected, str):
                        # For classification tests
                        result = {'output': output.squeeze().tolist(),
                                 'inference_time': inference_time}
                    else:
                        # For regression tests with numeric outputs
                        expected_tensor = torch.tensor(expected, device=self.device)
                        mse = nn.functional.mse_loss(output, expected_tensor).item()
                        result = {'output': output.squeeze().tolist(),
                                 'mse': mse,
                                 'inference_time': inference_time}
                else:
                    # Just return output for inference tests
                    result = {'output': output.squeeze().tolist(),
                             'inference_time': inference_time}
                
                results[test_name] = result
                logging.info(f"Test '{test_name}' completed in {inference_time:.5f}s")
                
            except Exception as e:
                error_msg = f"Error during test '{test_name}': {str(e)}"
                logging.error(error_msg)
                logging.error(traceback.format_exc())
                
                # Add error information to results
                results[test_name] = {
                    'error': str(e),
                    'inference_time': 0,
                    'output': [],
                    'status': 'failed'
                }
                
                # Increment error count
                self.metrics['error_count'] = self.metrics.get('error_count', 0) + 1
                
                # Track error type
                error_type = type(e).__name__
                if error_type in self.metrics['error_types']:
                    self.metrics['error_types'][error_type] += 1
                else:
                    self.metrics['error_types'][error_type] = 1
        
        # Update metrics
        for name, result in results.items():
            if 'error' not in result:
                if 'mse' in result:
                    self.metrics['test_results'][name] = 1.0 - min(result['mse'], 1.0)  # Convert MSE to score
                else:
                    self.metrics['test_results'][name] = 1.0  # Default score for tests without expected output
            else:
                self.metrics['test_results'][name] = 0.0  # Failed test
        
        return results
    
    def get_layer_activations(self, input_text: str) -> List[torch.Tensor]:
        """
        Get activations for each layer for a given input
        
        Args:
            input_text: Input text to process
            
        Returns:
            List of activations for each layer
        """
        try:
            self.eval()
            
            with torch.no_grad():
                embeddings = self.embed_text(input_text)
                activations = []
                x = embeddings
                
                for layer in self.layers:
                    x = layer(x)
                    if isinstance(layer, nn.Linear):
                        activations.append(x.detach().cpu())
            
            return activations
        except Exception as e:
            error_msg = f"Error getting layer activations: {str(e)}"
            logging.error(error_msg)
            
            if self.fallback_mode:
                # Return empty list in fallback mode
                return []
            else:
                # Re-raise the error if not in fallback mode
                raise RuntimeError(f"Failed to get layer activations: {error_msg}") from e
    
    def get_network_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the network architecture and state
        
        Returns:
            Dictionary with network summary
        """
        try:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            layer_summary = []
            input_size = 768  # BERT embedding size
            
            for i, layer in enumerate(self.layers):
                if isinstance(layer, nn.Linear):
                    layer_summary.append({
                        'type': 'Linear',
                        'input_size': layer.in_features,
                        'output_size': layer.out_features,
                        'parameters': layer.weight.numel() + layer.bias.numel()
                    })
                    input_size = layer.out_features
                elif isinstance(layer, nn.ReLU):
                    layer_summary.append({
                        'type': 'ReLU',
                        'parameters': 0
                    })
                elif isinstance(layer, nn.Tanh):
                    layer_summary.append({
                        'type': 'Tanh',
                        'parameters': 0
                    })
                elif isinstance(layer, nn.Sigmoid):
                    layer_summary.append({
                        'type': 'Sigmoid',
                        'parameters': 0
                    })
                elif isinstance(layer, nn.LeakyReLU):
                    layer_summary.append({
                        'type': 'LeakyReLU',
                        'parameters': 0
                    })
                elif isinstance(layer, nn.ELU):
                    layer_summary.append({
                        'type': 'ELU',
                        'parameters': 0
                    })
            
            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'layer_count': len(self.layers),
                'layers': layer_summary,
                'device': str(self.device),
                'optimizer': self.config.get('optimizer', 'adam'),
                'learning_rate': self.config.get('learning_rate', 0.001),
                'training_samples_seen': len(self.metrics['loss_history']),
                'inference_count': self.metrics['inference_count'],
                'error_count': self.metrics.get('error_count', 0),
                'fallback_mode': self.fallback_mode
            }
        except Exception as e:
            error_msg = f"Error getting network summary: {str(e)}"
            logging.error(error_msg)
            
            # Return minimal information
            return {
                'error': str(e),
                'device': str(self.device),
                'fallback_mode': self.fallback_mode
            }
