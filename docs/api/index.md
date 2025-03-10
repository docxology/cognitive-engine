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

See the [Engine API](engine.md) for complete details on the main engine API.

## Module APIs

Each module in the Cognitive Engine has its own API:

### Fractal System

```python
from cognitive_engine.fractal import FractalSystem, Symbol, SymbolRelation

# Create a fractal system
fs = FractalSystem(levels=7)

# Add symbols and relations
symbol = Symbol("example", properties={"key": "value"})
fs.add_symbol(symbol, layer=2)
```

See the [Fractal API](fractal.md) for complete details.

### Probabilistic System

```python
from cognitive_engine.probabilistic import LLMInterface, ProbabilisticEngine

# Create an LLM interface
llm = LLMInterface(model="gpt-4")

# Process text
result = llm.process("Generate creative ideas for sustainable energy")
```

See the [Probabilistic API](probabilistic.md) for complete details.

### Memory System

```python
from cognitive_engine.memory import PervasiveMemory

# Create a memory system
memory = PervasiveMemory(storage_path="./memory_store")

# Store and retrieve information
memory.store(content="Important information", metadata={"topic": "example"})
result = memory.retrieve(query="example", limit=5)
```

See the [Memory API](memory.md) for complete details.

### Specialized Module APIs

- [Unipixel API](unipixel.md)
- [MMM API](mmm.md)
- [Code Execution API](code_execution.md)
- [PEFF API](peff.md)

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
    "status": "success",  # or "error"
    "response": "The main response text",
    "reasoning_trace": [
        {"step": 1, "description": "..."},
        {"step": 2, "description": "..."}
    ],
    "confidence": 0.95,  # Confidence score
    "metadata": {
        # Additional information about the response
    }
}
```

## Authentication and Security

When using external services (e.g., LLMs), authentication is managed through:

```python
from cognitive_engine import set_api_credentials

# Set credentials for external services
set_api_credentials(
    service_name="openai",
    api_key="your-api-key",
    additional_params={}
)
```

## Extending the API

The Cognitive Engine can be extended with custom modules:

```python
from cognitive_engine import HybridCognitiveEngine, register_module

# Create a custom module
class CustomModule:
    def process(self, input_data):
        # Custom processing logic
        return {"result": "processed data"}

# Register the module
register_module("custom", CustomModule())

# Use in the engine
engine = HybridCognitiveEngine()
result = engine.modules.custom.process("input data")
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