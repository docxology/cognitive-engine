#!/usr/bin/env python3
"""
Pattern Finding Example for the Cognitive Engine.

This example demonstrates the pattern finding capabilities of the Cognitive Engine,
particularly using the Magical Math Model (MMM) module to identify patterns
across different layers and domains.
"""

import time
from pprint import pprint
from cognitive_engine import HybridCognitiveEngine
from cognitive_engine.fractal import FractalSystem, Symbol, SymbolRelation
from cognitive_engine.unipixel import UnipixelRegistry


def create_knowledge_domains(engine):
    """Create several knowledge domains with unipixels and symbols."""
    print("Creating knowledge domains...")
    
    # Biology domain
    biology_domain = []
    for entity in ["cell", "tissue", "organ", "organism"]:
        # Create symbols
        symbol = Symbol(entity, properties={"domain": "biology", "type": "entity"})
        engine.symbolic_system.add_symbol(symbol, layer=3)
        biology_domain.append(symbol)
        
        # Create unipixels
        pixel = engine.create_unipixel(
            name=f"biology_{entity}",
            layer=3,
            properties={"domain": "biology", "entity_type": entity}
        )
    
    # Add hierarchical relations in biology
    for i in range(len(biology_domain) - 1):
        relation = SymbolRelation(
            biology_domain[i], 
            biology_domain[i+1], 
            relation_type="part_of",
            properties={"strength": 0.9}
        )
        engine.symbolic_system.add_relation(relation)
    
    # Physics domain
    physics_domain = []
    for entity in ["particle", "atom", "molecule", "substance"]:
        # Create symbols
        symbol = Symbol(entity, properties={"domain": "physics", "type": "entity"})
        engine.symbolic_system.add_symbol(symbol, layer=3)
        physics_domain.append(symbol)
        
        # Create unipixels
        pixel = engine.create_unipixel(
            name=f"physics_{entity}",
            layer=3,
            properties={"domain": "physics", "entity_type": entity}
        )
    
    # Add hierarchical relations in physics
    for i in range(len(physics_domain) - 1):
        relation = SymbolRelation(
            physics_domain[i], 
            physics_domain[i+1], 
            relation_type="composes",
            properties={"strength": 0.85}
        )
        engine.symbolic_system.add_relation(relation)
    
    # Social domain
    social_domain = []
    for entity in ["individual", "family", "community", "society"]:
        # Create symbols
        symbol = Symbol(entity, properties={"domain": "social", "type": "entity"})
        engine.symbolic_system.add_symbol(symbol, layer=3)
        social_domain.append(symbol)
        
        # Create unipixels
        pixel = engine.create_unipixel(
            name=f"social_{entity}",
            layer=3,
            properties={"domain": "social", "entity_type": entity}
        )
    
    # Add hierarchical relations in social
    for i in range(len(social_domain) - 1):
        relation = SymbolRelation(
            social_domain[i], 
            social_domain[i+1], 
            relation_type="belongs_to",
            properties={"strength": 0.75}
        )
        engine.symbolic_system.add_relation(relation)
    
    # Store domain data for later use
    return {
        "biology": biology_domain,
        "physics": physics_domain,
        "social": social_domain
    }


def find_cross_domain_patterns(engine, domains):
    """Find patterns across different domains."""
    print("\nFinding cross-domain patterns...")
    
    # Create a meta-domain relationship
    print("Creating meta-domain relationship...")
    hierarchy_symbol = Symbol("hierarchy", properties={"type": "meta_pattern"})
    engine.symbolic_system.add_symbol(hierarchy_symbol, layer=5)
    
    # Connect the first elements of each domain to the hierarchy pattern
    for domain_name, domain_symbols in domains.items():
        meta_relation = SymbolRelation(
            domain_symbols[0],
            hierarchy_symbol,
            relation_type="instance_of",
            properties={"domain": domain_name}
        )
        engine.symbolic_system.add_relation(meta_relation)
    
    # Use MMM to find patterns
    print("Analyzing with Magical Math Model...")
    # Create a list of all unipixels to analyze
    unipixels = []
    registry = engine.unipixel_registry
    
    for domain_name in domains.keys():
        domain_pixels = registry.find_unipixels(
            properties={"domain": domain_name}
        )
        unipixels.extend([pixel.id for pixel in domain_pixels])
    
    # Find patterns across the selected unipixels
    patterns = engine.find_patterns(unipixels)
    
    print("\nDetected patterns across domains:")
    for i, pattern in enumerate(patterns.get('patterns', []), 1):
        print(f"\nPattern {i}:")
        print(f"  Type: {pattern.get('type')}")
        print(f"  Confidence: {pattern.get('confidence')}")
        print(f"  Description: {pattern.get('description')}")
        
        elements = pattern.get('elements', [])
        if elements:
            print("  Elements involved:")
            for elem in elements[:5]:  # Show first 5 elements
                print(f"    - {elem.get('name')} ({elem.get('domain')})")
            if len(elements) > 5:
                print(f"    - ... and {len(elements) - 5} more")
    
    return patterns


def find_hierarchical_structures(engine):
    """Find hierarchical structures within each domain."""
    print("\nIdentifying hierarchical structures...")
    
    # Use the fractal system to find hierarchical patterns
    hierarchies = engine.symbolic_system.find_hierarchical_patterns(
        relation_types=["part_of", "composes", "belongs_to"],
        min_depth=3
    )
    
    print(f"\nFound {len(hierarchies)} hierarchical structures:")
    for i, hierarchy in enumerate(hierarchies, 1):
        print(f"\nHierarchy {i}:")
        print(f"  Domain: {hierarchy.get('domain')}")
        print(f"  Depth: {hierarchy.get('depth')}")
        print(f"  Root: {hierarchy.get('root')}")
        
        levels = hierarchy.get('levels', [])
        if levels:
            print("  Structure:")
            for level_idx, level in enumerate(levels):
                print(f"    Level {level_idx+1}: {' -> '.join(level)}")
    
    return hierarchies


def find_cross_layer_patterns(engine):
    """Find patterns across different layers of the system."""
    print("\nFinding cross-layer patterns...")
    
    # Create some higher-layer abstractions
    print("Creating higher-layer abstractions...")
    
    # Layer 4: Domain concepts
    biology_concept = Symbol("biology_concept", properties={"type": "domain_concept"})
    physics_concept = Symbol("physics_concept", properties={"type": "domain_concept"})
    social_concept = Symbol("social_concept", properties={"type": "domain_concept"})
    
    engine.symbolic_system.add_symbol(biology_concept, layer=4)
    engine.symbolic_system.add_symbol(physics_concept, layer=4)
    engine.symbolic_system.add_symbol(social_concept, layer=4)
    
    # Layer 5: Meta-concepts
    natural_science = Symbol("natural_science", properties={"type": "meta_concept"})
    social_science = Symbol("social_science", properties={"type": "meta_concept"})
    
    engine.symbolic_system.add_symbol(natural_science, layer=5)
    engine.symbolic_system.add_symbol(social_science, layer=5)
    
    # Layer 6: Knowledge domains
    science = Symbol("science", properties={"type": "knowledge_domain"})
    
    engine.symbolic_system.add_symbol(science, layer=6)
    
    # Create connections between layers
    engine.symbolic_system.add_relation(biology_concept, natural_science, "belongs_to")
    engine.symbolic_system.add_relation(physics_concept, natural_science, "belongs_to")
    engine.symbolic_system.add_relation(social_concept, social_science, "belongs_to")
    
    engine.symbolic_system.add_relation(natural_science, science, "subdomain_of")
    engine.symbolic_system.add_relation(social_science, science, "subdomain_of")
    
    # Create corresponding unipixels
    for name, layer in [
        ("biology_concept", 4), ("physics_concept", 4), ("social_concept", 4),
        ("natural_science", 5), ("social_science", 5),
        ("science", 6)
    ]:
        engine.create_unipixel(
            name=name,
            layer=layer,
            properties={"concept_type": name}
        )
    
    # Find patterns across layers
    print("Analyzing cross-layer patterns...")
    cross_layer_patterns = engine.find_cross_layer_patterns()
    
    print("\nDetected cross-layer patterns:")
    for i, pattern in enumerate(cross_layer_patterns.get('patterns', []), 1):
        print(f"\nCross-layer Pattern {i}:")
        print(f"  Type: {pattern.get('type')}")
        print(f"  Layers involved: {pattern.get('layers')}")
        print(f"  Description: {pattern.get('description')}")
        
        if 'structure' in pattern:
            print("  Structure:")
            structure = pattern.get('structure', {})
            for layer, elements in structure.items():
                print(f"    Layer {layer}: {', '.join(elements[:3])}" + 
                     (f" and {len(elements)-3} more" if len(elements) > 3 else ""))
    
    return cross_layer_patterns


def analyze_emergent_properties(engine, patterns):
    """Analyze emergent properties from the discovered patterns."""
    print("\nAnalyzing emergent properties...")
    
    # Use the MMM module to identify emergent properties
    emergent_properties = engine.mmm.analyze_emergent_properties(patterns)
    
    print("\nEmergent properties identified:")
    for i, property in enumerate(emergent_properties, 1):
        print(f"\nProperty {i}: {property.get('name')}")
        print(f"  Type: {property.get('type')}")
        print(f"  Description: {property.get('description')}")
        print(f"  Confidence: {property.get('confidence')}")
        
        if 'examples' in property:
            print("  Examples:")
            for example in property.get('examples', [])[:3]:
                print(f"    - {example}")
    
    return emergent_properties


def main():
    """Main function to demonstrate pattern finding."""
    print("Initializing Cognitive Engine...")
    engine = HybridCognitiveEngine()
    
    # Create knowledge domains
    domains = create_knowledge_domains(engine)
    
    # Give the system a moment to process
    print("Processing knowledge...")
    time.sleep(1)
    
    # Find patterns across domains
    cross_domain_patterns = find_cross_domain_patterns(engine, domains)
    
    # Find hierarchical structures
    hierarchies = find_hierarchical_structures(engine)
    
    # Find cross-layer patterns
    cross_layer_patterns = find_cross_layer_patterns(engine)
    
    # Analyze emergent properties
    emergent_properties = analyze_emergent_properties(
        engine, 
        cross_domain_patterns.get('patterns', []) + cross_layer_patterns.get('patterns', [])
    )
    
    # Summarize findings
    print("\n" + "="*50)
    print("PATTERN FINDING SUMMARY")
    print("="*50)
    print(f"Cross-domain patterns found: {len(cross_domain_patterns.get('patterns', []))}")
    print(f"Hierarchical structures identified: {len(hierarchies)}")
    print(f"Cross-layer patterns discovered: {len(cross_layer_patterns.get('patterns', []))}")
    print(f"Emergent properties analyzed: {len(emergent_properties)}")
    
    # Generate insights
    print("\nGenerating insights...")
    insights = engine.process(
        "What insights can be drawn from the patterns identified across domains and layers?",
        context={
            "cross_domain_patterns": cross_domain_patterns,
            "hierarchies": hierarchies,
            "cross_layer_patterns": cross_layer_patterns,
            "emergent_properties": emergent_properties
        }
    )
    
    print("\nINSIGHTS:")
    print(insights['response'])


if __name__ == "__main__":
    main() 