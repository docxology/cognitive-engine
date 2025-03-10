# Cognitive Engine

The Cognitive Engine is a hybrid Neuro-Symbolic AI system designed to combine the strengths of symbolic AI and neural networks for enhanced reasoning capabilities.

## Architecture

The engine consists of multiple integrated modules:

### Core Modules

- **Fractal System**: Implements symbol nesting, bindings, templates, and dynamics
  - Core structures for symbolic representation
  - Binding mechanisms for connecting symbols
  - Templates for reusable patterns

- **Probabilistic System**: Handles neural networks, LLMs, and probabilistic inference
  - Neural network interfaces for connectionist approaches
  - LLM integration for large language model capabilities
  - Probabilistic inference engines for reasoning under uncertainty

- **Memory System**: Provides long-term pervasive memory
  - Storage mechanisms for different types of information
  - Retrieval algorithms for efficient memory access

### Specialized Modules

- **Unipixel**: Fractal unit/atom/pixel/basis for each layer of the system
  - Represents object-processes in Active Inference / Free Energy Principle
  - Functions like a row in a database and can be a sub-program

- **MMM (Magical Math Model)**: 7 layers of cognitive motor
  - Pattern recognition across and within layers
  - Mathematical modeling of cognitive processes

- **Code Execution**: Skills for executing and operating code
  - Safe execution of code in various environments
  - Code analysis and repository management
  - Integration with external code repositories

- **PEFF (Paradise Energy Fractal Force)**: System for harmony, optimization, security, ethics, and emotions
  - Ethical reasoning and decision-making
  - System optimization for harmonic balance
  - Security assessment and enforcement
  - Emotional intelligence processing

## Integration

The `engine.py` file in the root of the package serves as the main integration point, combining all modules into a cohesive system through the `HybridCognitiveEngine` class.

## Usage

See the main README.md file for usage examples.

## Module Structure

Each module follows a consistent structure:
- A dedicated directory containing:
  - `__init__.py`: Module initialization and exports
  - `engine.py`: Core integration methods for the module
  - Additional specialized components
- A README.md file explaining the module's purpose and usage
