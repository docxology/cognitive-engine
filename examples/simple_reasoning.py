#!/usr/bin/env python
"""
A simple example demonstrating the use of the Hybrid Cognitive Engine.
"""

import sys
import os
import json

# Add the parent directory to the path so we can import cognitive_engine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cognitive_engine import HybridCognitiveEngine


def main():
    """
    Run a simple reasoning example with the Hybrid Cognitive Engine.
    """
    print("Initializing Hybrid Cognitive Engine...")
    engine = HybridCognitiveEngine(config={
        'fractal_levels': 5,
        'use_llm': False  # Use the probabilistic engine instead of LLM
    })
    
    # Store some knowledge in the memory system
    print("\nStoring knowledge in memory...")
    engine.memory.store(
        key="Tree Classification",
        value="Trees are classified as perennial plants with elongated stems or trunks supporting branches and leaves."
    )
    
    engine.memory.store(
        key="Neural Networks",
        value="Neural networks are computing systems inspired by the biological neural networks that constitute animal brains."
    )
    
    # Process a simple query
    query = "How can we use tree structures in neural networks for better symbolic reasoning?"
    
    print(f"\nProcessing query: '{query}'")
    result = engine.reason(query)
    
    # Print the results
    print("\nReasoning Results:")
    print("------------------")
    print(f"Query: {result['query']}")
    print("\nSymbolic Reasoning Steps:")
    for step in result['symbolic_reasoning']['steps']:
        print(f"  Step {step['step']}: {step['operation']} - {step['result']}")
    
    print("\nNeural Reasoning (simulated):")
    # The neural system would normally produce this
    print("  Neural system processed the query in conjunction with symbolic structure")
    
    print("\nIntegrated Conclusion:")
    print(f"  {result['conclusion']}")


if __name__ == "__main__":
    main() 