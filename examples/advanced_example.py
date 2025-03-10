#!/usr/bin/env python
"""
An advanced example demonstrating the use of the Hybrid Cognitive Engine
with the new Unipixel, MMM, and Perplexity API modules.
"""

import sys
import os
import json
import time
from pprint import pprint

# Add the parent directory to the path so we can import cognitive_engine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cognitive_engine import HybridCognitiveEngine, Unipixel


def main():
    """
    Run an advanced example with the Hybrid Cognitive Engine.
    """
    print("\n=== Initializing Hybrid Cognitive Engine ===")
    
    # Create a configuration with API key for Perplexity
    # In production, you would use environment variables for the API key
    config = {
        'fractal_levels': 7,
        'use_llm': False,  # Use the probabilistic engine instead of LLM for this example
        'perplexity_api_key': os.environ.get('PERPLEXITY_API_KEY'),
        'perplexity_cache_dir': './.perplexity_cache'
    }
    
    # Initialize the engine
    engine = HybridCognitiveEngine(config=config)
    
    # Demo 1: Create and work with Unipixels
    demo_unipixels(engine)
    
    # Demo 2: Pattern finding with the Magical Math Model
    demo_patterns(engine)
    
    # Demo 3: Web search with Perplexity API
    if config.get('perplexity_api_key'):
        demo_web_search(engine)
    else:
        print("\n=== Skipping Web Search Demo (No API Key) ===")
        print("Set the PERPLEXITY_API_KEY environment variable to run this demo.")


def demo_unipixels(engine):
    """
    Demonstrate the Unipixel functionality.
    """
    print("\n=== Unipixel Demo ===")
    
    # Create a hierarchy of Unipixels
    print("\nCreating a hierarchy of Unipixels across multiple layers...")
    
    # Create a root Unipixel at layer 0
    root_result = engine.create_unipixel(
        name="Root Concept",
        layer=0,
        properties={
            "description": "The root concept in our cognitive structure",
            "importance": 0.95
        }
    )
    root_id = root_result['id']
    print(f"Created root: {root_result['name']} (ID: {root_id})")
    
    # Create some children at layer 1
    child_concepts = ["Mathematics", "Physics", "Biology"]
    child_ids = []
    
    for concept in child_concepts:
        child_result = engine.create_unipixel(
            name=concept,
            layer=1,
            properties={
                "description": f"A branch of {root_result['name']}",
                "domain": "science"
            },
            parent_id=root_id
        )
        child_ids.append(child_result['id'])
        print(f"Created child: {child_result['name']} (ID: {child_result['id']})")
    
    # Create some grandchildren at layer 2
    math_children = ["Algebra", "Geometry", "Calculus"]
    
    for concept in math_children:
        grandchild_result = engine.create_unipixel(
            name=concept,
            layer=2,
            properties={
                "description": f"A branch of Mathematics",
                "difficulty": 0.7
            },
            parent_id=child_ids[0]  # Mathematics ID
        )
        print(f"Created grandchild: {grandchild_result['name']} (ID: {grandchild_result['id']})")
    
    # Get information about our Unipixel structure
    total_unipixels = len(engine.unipixel_registry.unipixels)
    layer_distribution = {}
    
    for layer in range(7):
        unipixels_at_layer = engine.unipixel_registry.get_by_layer(layer)
        if unipixels_at_layer:
            layer_distribution[layer] = len(unipixels_at_layer)
    
    print(f"\nCreated a total of {total_unipixels} Unipixels")
    print("Distribution across layers:")
    for layer, count in layer_distribution.items():
        print(f"  Layer {layer}: {count} Unipixels")
    
    # Get children of the root
    root_children = engine.unipixel_registry.get_children(root_id)
    print(f"\nChildren of the root ({len(root_children)}):")
    for child in root_children:
        print(f"  - {child.name} (Layer {child.layer})")
    
    # Get children of Mathematics
    math_children = engine.unipixel_registry.get_children(child_ids[0])
    print(f"\nChildren of Mathematics ({len(math_children)}):")
    for child in math_children:
        print(f"  - {child.name} (Layer {child.layer})")


def demo_patterns(engine):
    """
    Demonstrate the pattern finding capabilities of the MMM.
    """
    print("\n=== Magical Math Model (MMM) Demo ===")
    
    # Get all Unipixel IDs to find patterns among
    all_unipixel_ids = list(engine.unipixel_registry.unipixels.keys())
    
    # Find patterns among all Unipixels
    print("\nFinding patterns among all Unipixels...")
    patterns_result = engine.find_patterns(all_unipixel_ids)
    
    print(f"Found {patterns_result['pattern_count']} patterns among {patterns_result['element_count']} elements")
    
    # Display the patterns if any were found
    if patterns_result['pattern_count'] > 0:
        print("\nPatterns found:")
        for i, pattern in enumerate(patterns_result['patterns']):
            print(f"  {i+1}. {pattern['pattern_type']} pattern ({pattern['confidence']:.2f} confidence)")
            print(f"     Description: {pattern['description']}")
            print(f"     Layers involved: {pattern['layers']}")
            print(f"     Elements: {len(pattern['elements'])} elements")
    
    # Find cross-layer patterns
    print("\nFinding patterns across layers...")
    cross_layer_patterns = engine.find_cross_layer_patterns()
    
    print(f"Found {cross_layer_patterns['pattern_count']} cross-layer patterns")
    
    # Display the cross-layer patterns if any were found
    if cross_layer_patterns['pattern_count'] > 0:
        print("\nCross-layer patterns found:")
        for i, pattern in enumerate(cross_layer_patterns['patterns']):
            print(f"  {i+1}. {pattern['pattern_type']} pattern across layers {pattern['layers']}")
            print(f"     Description: {pattern['description']}")
            print(f"     Elements: {len(pattern['elements'])} elements")


def demo_web_search(engine):
    """
    Demonstrate the web search capabilities of the Perplexity API.
    """
    print("\n=== Perplexity API Web Search Demo ===")
    
    # Perform a simple search
    print("\nPerforming a simple web search...")
    query = "What is active inference in cognitive science?"
    
    print(f"Query: '{query}'")
    try:
        search_result = engine.search_web(query)
        
        print("\nSearch Results:")
        print("---------------")
        print(f"Answer: {search_result['answer'][:500]}...")  # Show first 500 chars
        
        print("\nSources:")
        for i, source in enumerate(search_result['sources'][:3]):  # Show first 3 sources
            print(f"  {i+1}. {source.get('title', 'Untitled')} - {source.get('url', 'No URL')}")
        
        # Research a topic with follow-up questions
        print("\nResearching a topic with follow-up questions...")
        topic = "Neuro-Symbolic AI systems"
        
        print(f"Topic: '{topic}'")
        research_result = engine.research_topic(topic, depth=2)
        
        print("\nMain Answer:")
        print(f"{research_result['main_result']['answer'][:500]}...")  # Show first 500 chars
        
        print("\nFollow-up Questions:")
        for i, question in enumerate(research_result['followup_questions']):
            print(f"  {i+1}. {question}")
            
        print("\nTo see full research results, check the memory system or the response object.")
        
    except Exception as e:
        print(f"Web search demo failed: {str(e)}")
        print("Check your Perplexity API key and internet connection.")


if __name__ == "__main__":
    main() 