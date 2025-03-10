# Quick Start Guide

This guide will help you get started with the Cognitive Engine, walking you through installation, basic setup, and your first experiment.

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/cognitive-engine.git
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

# Print the response
print(result['response'])
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
from cognitive_engine.probabilistic import LLMInterface
from cognitive_engine.memory import PervasiveMemory

# Create individual components with custom settings
fractal_system = FractalSystem(levels=7)
llm = LLMInterface(model="gpt-4", temperature=0.7)
memory = PervasiveMemory(storage_path="./my_memory_store")

# Create the engine with custom components
engine = HybridCognitiveEngine(
    symbolic_system=fractal_system,
    neural_system=llm,
    memory_system=memory
)

# Process a query
result = engine.process("Describe the key advantages of hybrid AI systems")
print(result['response'])
```

## Working with Core Modules

### Fractal System

```python
from cognitive_engine.fractal import FractalSystem, Symbol, SymbolRelation

# Create a fractal system
fs = FractalSystem(levels=7)

# Create symbols
apple = Symbol("apple", properties={"type": "fruit", "color": "red"})
fruit = Symbol("fruit", properties={"type": "category"})

# Create a relation
is_a = SymbolRelation(apple, fruit, relation_type="is_a")

# Add to the fractal system
fs.add_symbol(apple, layer=3)
fs.add_symbol(fruit, layer=4)
fs.add_relation(is_a)

# Query the system
fruits = fs.find_symbols_by_relation(fruit, "is_a", reverse=True)
print(f"Found {len(fruits)} fruits")
```

### Probabilistic System

```python
from cognitive_engine.probabilistic import LLMInterface, PromptManager

# Create an LLM interface
llm = LLMInterface(model="gpt-4")

# Create a prompt manager
prompt_manager = PromptManager()
prompt = prompt_manager.format_prompt(
    template="analyze_topic",
    topic="climate change",
    perspective="economic impact",
    depth="detailed"
)

# Generate a response
response = llm.generate(prompt, max_tokens=500)
print(response)
```

### Memory System

```python
from cognitive_engine.memory import PervasiveMemory

# Create a memory system
memory = PervasiveMemory(storage_path="./memory_store")

# Store information
memory_id = memory.store(
    content="The Cognitive Engine combines symbolic and neural approaches to AI",
    metadata={"topic": "hybrid_ai", "importance": 0.8}
)

# Retrieve information
results = memory.retrieve(query="symbolic neural hybrid", limit=5)
for item in results:
    print(f"Content: {item.content}")
    print(f"Relevance: {item.relevance_score}")
    print("---")
```

## Creating a Complete Example

Let's build a more comprehensive example that uses multiple modules:

```python
from cognitive_engine import HybridCognitiveEngine
from cognitive_engine.fractal import Symbol
import time

# Initialize the engine
engine = HybridCognitiveEngine()

# Store some knowledge
engine.memory_system.store(
    content="Climate change is leading to rising sea levels and more extreme weather events",
    metadata={"topic": "climate_change", "source": "scientific_consensus"}
)

engine.memory_system.store(
    content="Renewable energy sources include solar, wind, hydro, and geothermal power",
    metadata={"topic": "renewable_energy", "source": "energy_research"}
)

# Create some symbols in the fractal system
climate_change = Symbol("climate_change", properties={"domain": "environment"})
renewable_energy = Symbol("renewable_energy", properties={"domain": "energy"})
mitigation = Symbol("mitigation", properties={"type": "strategy"})

engine.symbolic_system.add_symbol(climate_change, layer=4)
engine.symbolic_system.add_symbol(renewable_energy, layer=4)
engine.symbolic_system.add_symbol(mitigation, layer=3)

engine.symbolic_system.add_relation(renewable_energy, climate_change, "mitigates")
engine.symbolic_system.add_relation(renewable_energy, mitigation, "is_a")

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
result = engine.process(
    "How can renewable energy help address climate change?",
    use_memory=True,
    reasoning_depth="deep"
)

print("\nResponse:")
print(result['response'])

print("\nReasoning trace:")
for step in result.get('reasoning_trace', []):
    print(f"Step {step['step']}: {step['description']}")

# Find patterns between domains
time.sleep(1)  # Give the engine time to process
patterns = engine.find_patterns([climate_pixel['id'], energy_pixel['id']])

print("\nIdentified relationships between climate and energy systems:")
for relationship in patterns.get('relationships', []):
    print(f"- {relationship.get('description')}")
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