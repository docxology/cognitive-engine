# Quick Start Guide

This guide will help you get started with the Cognitive Engine, walking you through installation, basic setup, and your first experiment.

## Installation

### Prerequisites

- Python 3.10+
- pip (Python package manager)
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository**:

```bash
git clone https://github.com/docxology/cognitive-engine.git
cd cognitive-engine
```

2. **Create a virtual environment**:

```bash
# Using venv
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt

# Install in development mode (recommended for experimentation)
pip install -e .
```

4. **Verify installation**:

```bash
python -c "from cognitive_engine import HybridCognitiveEngine; print('Installation successful!')"
```

## Basic Usage

### Creating Your First Engine

Create a Python file named `first_example.py` with the following content:

```python
from cognitive_engine import HybridCognitiveEngine

# Initialize the engine with default components
engine = HybridCognitiveEngine()

# Process a simple query
result = engine.process("Explain how neural networks and symbolic AI complement each other")

# Print the neural output
print(result['neural_outputs'])
```

Run the example:

```bash
python first_example.py
```

### Customizing Your Engine

Now let's create a more customized engine with specific settings:

```python
from cognitive_engine import HybridCognitiveEngine
from cognitive_engine.fractal import FractalSystem
from cognitive_engine.probabilistic import LLMInterface, LLMConfig
from cognitive_engine.memory import PervasiveMemory

# Create individual components with custom settings
fractal_system = FractalSystem(levels=7)
llm = LLMInterface(model="gpt-4", config=LLMConfig(temperature=0.7))
memory = PervasiveMemory(storage_config={"persistence_path": "./my_memory_store"})

# Create the engine with custom components
engine = HybridCognitiveEngine(
    symbolic_system=fractal_system,
    neural_system=llm,
    memory_system=memory
)

# Process a query
result = engine.process("Describe the key advantages of hybrid AI systems")
print(result['neural_outputs'])
```

## Working with Core Modules

### Fractal System

```python
from cognitive_engine.fractal import FractalSystem
from cognitive_engine.fractal.binding import BindingMechanism

# Create a fractal system
fs = FractalSystem(levels=7)

# Create nodes
apple = fs.create_root("apple", properties={"type": "fruit", "color": "red"})
fruit = fs.create_root("fruit", properties={"type": "category"})

# Relate the nodes through a binding
mechanism = BindingMechanism(fractal_system=fs)
binding = mechanism.create_binding(apple.id, fruit.id, "is_a")

# Query the bindings
related = mechanism.get_bindings_by_relation("is_a")
print(f"Found {len(related)} 'is_a' bindings")
```

### Probabilistic System

```python
from cognitive_engine.probabilistic import LLMInterface

# Create an LLM interface
llm = LLMInterface(model="gpt-4")

# Generate a response
response = llm.process("Analyze the economic impact of climate change")
print(response)

# Perform reasoning with symbolic context
reasoning = llm.reason(
    query="Does the evidence support the conclusion?",
    symbolic_reasoning={"steps": ["step 1", "step 2"]}
)
print(reasoning)
```

### Memory System

```python
from cognitive_engine.memory import PervasiveMemory

# Create a memory system
memory = PervasiveMemory(storage_config={"persistence_path": "./memory_store"})

# Store information
memory_id = memory.store(
    key="hybrid_ai",
    value="The Cognitive Engine combines symbolic and neural approaches to AI",
    metadata={"topic": "hybrid_ai", "importance": 0.8}
)

# Retrieve information
results = memory.retrieve(query="symbolic neural hybrid", k=5)
for item in results:
    print(item)
    print("---")
```

## Creating a Complete Example

Let's build a more comprehensive example that uses multiple modules:

```python
from cognitive_engine import HybridCognitiveEngine
import time

# Initialize the engine
engine = HybridCognitiveEngine()

# Store some knowledge
engine.memory.store(
    key="climate_change",
    value="Climate change is leading to rising sea levels and more extreme weather events",
    metadata={"topic": "climate_change", "source": "scientific_consensus"}
)

engine.memory.store(
    key="renewable_energy",
    value="Renewable energy sources include solar, wind, hydro, and geothermal power",
    metadata={"topic": "renewable_energy", "source": "energy_research"}
)

# Create some nodes in the fractal system
climate_change = engine.symbolic.create_root(
    "climate_change", properties={"domain": "environment"}
)
renewable_energy = engine.symbolic.create_root(
    "renewable_energy", properties={"domain": "energy"}
)
mitigation = climate_change.create_child(
    "mitigation", properties={"type": "strategy"}
)

# Create unipixels
climate_pixel = engine.create_unipixel(
    name="climate_system", 
    layer=4,
    properties={"domain": "climate_science"}
)

energy_pixel = engine.create_unipixel(
    name="energy_system", 
    layer=4,
    properties={"domain": "energy_science"}
)

# Process a complex query
result = engine.process("How can renewable energy help address climate change?")

print("\nNeural output:")
print(result['neural_outputs'])

reasoning = engine.reason("How can renewable energy help address climate change?")
print("\nIntegrated reasoning:")
print(reasoning['integrated_reasoning'])

# Find patterns between domains
time.sleep(1)  # Give the engine time to process
patterns = engine.find_patterns([climate_pixel['id'], energy_pixel['id']])

print("\nIdentified patterns between climate and energy systems:")
for pattern in patterns.get('patterns', []):
    print(f"- {pattern}")
```

## Next Steps

Now that you've completed the quick start guide, here are some suggestions for next steps:

1. **Explore examples**: Check out the `examples/` directory for more advanced examples
2. **Read the documentation**: Dive deeper into each module's documentation
3. **Try advanced features**:
   - Cross-layer pattern recognition with MMM
   - Code execution capabilities
   - PEFF for ethical reasoning
4. **Contribute**: Consider contributing to the project by fixing bugs or adding features

## Troubleshooting

### Common Issues

- **ImportError**: Make sure you've installed the package correctly with `pip install -e .`
- **API Key Errors**: For LLM integration, ensure you've set the appropriate API keys
- **Memory Storage Errors**: Check that the memory storage path exists and is writable

### Getting Help

- Check the [FAQ](faq.md) for common questions
- Search for similar issues in the GitHub repository
- Ask questions in the community channels

## Further Reading

- [Architecture Overview](architecture.md)
- [Fractal System Documentation](fractal.md)
- [Probabilistic System Documentation](probabilistic.md)
- [Memory System Documentation](memory.md)
- [API Reference](api/index.md)
