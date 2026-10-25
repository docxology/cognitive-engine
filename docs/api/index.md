# API Reference

This section provides detailed documentation for the Cognitive Engine API.

## Core API

The main interface to the Cognitive Engine is through the `HybridCognitiveEngine` class:

```python
from cognitive_engine import HybridCognitiveEngine

engine = HybridCognitiveEngine()
```

### Main Methods

| Method | Description |
|--------|-------------|
| `process(input_data)` | Process input data through the hybrid system |
| `reason(query)` | Perform reasoning on a complex question |
| `search_web(query)` | Search the web for information |
| `research_topic(topic, depth)` | Research a topic in depth |
| `create_unipixel(name, layer, properties)` | Create a new fractal unit |
| `find_patterns(element_ids)` | Find patterns between elements |
| `find_cross_layer_patterns()` | Find patterns across all layers |

See the [Engine API](../code_execution.md) for complete details on the main engine API.

## Module APIs

Each module in the Cognitive Engine has its own API:

### Fractal System

```python
from cognitive_engine.fractal import FractalSystem

# Create a fractal system
fs = FractalSystem(levels=7)

# Create a root node and nest a child under it
root = fs.create_root("example", properties={"key": "value"})
child = root.create_child("nested", properties={"layer": 2})
```

See the [Fractal API](../fractal.md) for complete details.

### Probabilistic System

```python
from cognitive_engine.probabilistic import LLMInterface

# Create an LLM interface
llm = LLMInterface(model="gpt-4")

# Process text
result = llm.process("Generate creative ideas for sustainable energy")
```

See the [Probabilistic API](../probabilistic.md) for complete details.

### Memory System

```python
from cognitive_engine.memory import PervasiveMemory

# Create a memory system
memory = PervasiveMemory()

# Store and retrieve information
memory_id = memory.store(key="example", value="Important information", metadata={"topic": "example"})
result = memory.retrieve(query="example", k=5)
```

See the [Memory API](../memory.md) for complete details.

### Specialized Module APIs

- [Unipixel API](../unipixel.md)
- [MMM API](../mmm.md)
- [Code Execution API](../code_execution.md)
- [PEFF API](../peff.md)

## API Design Philosophy

The Cognitive Engine API is designed with the following principles:

1. **Consistency**: Similar operations use similar patterns across modules
2. **Explainability**: Results include explanations of the reasoning process
3. **Flexibility**: Components can be used independently or together
4. **Error Handling**: Clear error messages and graceful failure modes
5. **Type Safety**: Type hints are used throughout the codebase

## Response Formats

Most API methods return a dictionary with standardized fields:

```python
{
    "input": "Analyze the relationship between climate change and economic policy",
    "symbolic_representation": {...},
    "neural_outputs": {...},
    "memory_id": "mem_0001",
    # Engine state snapshots are available via engine.to_json()
}
```

## Authentication and Security

When using external services (e.g., LLMs or the Perplexity search engine), credentials are
passed through the engine configuration:

```python
import os

from cognitive_engine import HybridCognitiveEngine

engine = HybridCognitiveEngine(
    config={
        "perplexity_api_key": os.environ["PERPLEXITY_API_KEY"],
        "perplexity_cache_dir": "./.perplexity_cache",
    }
)
```

## Extending the API

The Cognitive Engine can be extended by supplying custom component implementations to the
engine constructor or by using the module classes directly:

```python
from cognitive_engine import HybridCognitiveEngine
from cognitive_engine.fractal import FractalSystem
from cognitive_engine.probabilistic import LLMInterface

# Create a custom module
class CustomModule:
    def process(self, input_data):
        # Custom processing logic
        return {"result": "processed data"}

# Inject custom components (or your own subclasses of them)
engine = HybridCognitiveEngine(
    symbolic_system=FractalSystem(levels=5),
    neural_system=LLMInterface(model="gpt-4"),
)

# Standalone modules can be driven directly
result = CustomModule().process("input data")
```

## Version Compatibility

The API follows semantic versioning:

- **Major versions**: Potentially breaking changes
- **Minor versions**: New features, no breaking changes
- **Patch versions**: Bug fixes only

Current version: 0.1.0

## API Changelog

### 0.1.0 (Initial Release)

- Core engine implementation
- Basic fractal and probabilistic systems
- Memory storage and retrieval
- Unipixel functionality
