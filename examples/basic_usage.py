#!/usr/bin/env python3
"""
Basic usage examples for the Cognitive Engine.

This file demonstrates core functionality of the hybrid Neuro-Symbolic AI system
including:
- Engine initialization
- Processing simple queries
- Reasoning with the hybrid system
- Working with the memory system
- Creating and using unipixels
"""

from cognitive_engine import HybridCognitiveEngine
from cognitive_engine.fractal import FractalSystem, Symbol, SymbolRelation
from cognitive_engine.probabilistic import LLMInterface
from cognitive_engine.memory import PervasiveMemory


def main():
    # Example 1: Basic engine initialization
    print("Example 1: Basic Engine Initialization")
    print("-" * 50)
    
    # Initialize with default components
    engine = HybridCognitiveEngine()
    print("Engine initialized with default components")
    print()
    
    # Example 2: Custom engine initialization
    print("Example 2: Custom Engine Initialization")
    print("-" * 50)
    
    # Initialize with custom components
    custom_engine = HybridCognitiveEngine(
        symbolic_system=FractalSystem(levels=7),
        neural_system=LLMInterface(model="gpt-4"),
        memory_system=PervasiveMemory(storage_path="./memory_store")
    )
    print("Engine initialized with custom components")
    print()
    
    # Example 3: Process a simple query
    print("Example 3: Processing a Simple Query")
    print("-" * 50)
    
    # Process a query
    result = engine.process("Explain the relationship between climate change and economic policy")
    print(f"Response: {result['response'][:150]}...")  # Show just the first part
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    print()
    
    # Example 4: Reasoning with the hybrid system
    print("Example 4: Reasoning with the Hybrid System")
    print("-" * 50)
    
    # Perform reasoning
    reasoning_result = engine.reason("How might quantum computing affect cryptographic systems?")
    
    # Display reasoning trace
    print("Reasoning trace:")
    for step in reasoning_result.get('reasoning_trace', [])[:3]:  # Show first 3 steps
        print(f"  Step {step['step']}: {step['description']}")
    
    print(f"\nConclusion: {reasoning_result.get('conclusion', 'No conclusion available')[:150]}...")
    print()
    
    # Example 5: Working with memory
    print("Example 5: Working with Memory")
    print("-" * 50)
    
    # Store information in memory
    engine.memory_system.store(
        content="The Free Energy Principle suggests that biological systems minimize surprise by updating their internal models of the world.",
        metadata={"topic": "cognitive_science", "author": "Karl Friston", "importance": "high"}
    )
    
    # Retrieve from memory
    memory_results = engine.memory_system.retrieve(query="free energy principle", limit=3)
    print(f"Memory retrieval results: {len(memory_results)} items found")
    
    if memory_results:
        print(f"First result: {memory_results[0].get('content', 'No content')[:150]}...")
    print()
    
    # Example 6: Creating and working with unipixels
    print("Example 6: Creating and Working with Unipixels")
    print("-" * 50)
    
    # Create unipixels at different layers
    climate_pixel = engine.create_unipixel(
        name="climate_system", 
        layer=3,
        properties={"domain": "environmental_science"}
    )
    
    economic_pixel = engine.create_unipixel(
        name="economic_system", 
        layer=3,
        properties={"domain": "economics"}
    )
    
    print(f"Created climate unipixel with ID: {climate_pixel.get('id', 'unknown')}")
    print(f"Created economic unipixel with ID: {economic_pixel.get('id', 'unknown')}")
    
    # Find patterns between unipixels
    patterns = engine.find_patterns([climate_pixel.get('id'), economic_pixel.get('id')])
    
    print("\nIdentified relationships:")
    for relationship in patterns.get('relationships', [])[:2]:  # Show first 2 relationships
        print(f"  {relationship.get('type')}: {relationship.get('description')}")
    print()


if __name__ == "__main__":
    main() 