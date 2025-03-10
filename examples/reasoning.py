#!/usr/bin/env python3
"""
Reasoning Example for the Cognitive Engine.

This example demonstrates the hybrid reasoning capabilities of the Cognitive Engine,
showing how it combines symbolic and neural approaches to solve complex reasoning tasks.
"""

from cognitive_engine import HybridCognitiveEngine
from cognitive_engine.fractal import Symbol, SymbolRelation


def setup_knowledge_base(engine):
    """Set up a knowledge base for the reasoning example."""
    print("Setting up knowledge base...")
    
    # Create climate change domain symbols
    climate_change = Symbol("climate_change", properties={"type": "phenomenon"})
    global_warming = Symbol("global_warming", properties={"type": "phenomenon"})
    greenhouse_effect = Symbol("greenhouse_effect", properties={"type": "phenomenon"})
    carbon_emissions = Symbol("carbon_emissions", properties={"type": "cause"})
    deforestation = Symbol("deforestation", properties={"type": "cause"})
    
    sea_level_rise = Symbol("sea_level_rise", properties={"type": "effect"})
    extreme_weather = Symbol("extreme_weather", properties={"type": "effect"})
    biodiversity_loss = Symbol("biodiversity_loss", properties={"type": "effect"})
    
    renewable_energy = Symbol("renewable_energy", properties={"type": "solution"})
    carbon_capture = Symbol("carbon_capture", properties={"type": "solution"})
    reforestation = Symbol("reforestation", properties={"type": "solution"})
    policy_changes = Symbol("policy_changes", properties={"type": "solution"})
    
    # Add symbols to the fractal system
    for symbol in [
        climate_change, global_warming, greenhouse_effect, carbon_emissions, deforestation,
        sea_level_rise, extreme_weather, biodiversity_loss,
        renewable_energy, carbon_capture, reforestation, policy_changes
    ]:
        engine.symbolic_system.add_symbol(symbol, layer=3)
    
    # Create causal relationships
    engine.symbolic_system.add_relation(greenhouse_effect, global_warming, "causes")
    engine.symbolic_system.add_relation(global_warming, climate_change, "causes")
    
    engine.symbolic_system.add_relation(carbon_emissions, greenhouse_effect, "increases")
    engine.symbolic_system.add_relation(deforestation, carbon_emissions, "increases")
    engine.symbolic_system.add_relation(deforestation, climate_change, "contributes_to")
    
    engine.symbolic_system.add_relation(climate_change, sea_level_rise, "causes")
    engine.symbolic_system.add_relation(climate_change, extreme_weather, "causes")
    engine.symbolic_system.add_relation(climate_change, biodiversity_loss, "causes")
    
    engine.symbolic_system.add_relation(renewable_energy, carbon_emissions, "reduces")
    engine.symbolic_system.add_relation(carbon_capture, carbon_emissions, "reduces")
    engine.symbolic_system.add_relation(reforestation, carbon_emissions, "reduces")
    engine.symbolic_system.add_relation(reforestation, deforestation, "reverses")
    engine.symbolic_system.add_relation(policy_changes, climate_change, "mitigates")
    
    # Store some factual knowledge in memory
    engine.memory_system.store(
        content="Global average temperature has increased by approximately 1.1°C since pre-industrial times.",
        metadata={"topic": "climate_change", "source": "IPCC", "confidence": 0.95}
    )
    
    engine.memory_system.store(
        content="Sea levels rose by about 20 cm in the 20th century, and the rate has accelerated to 3.6 mm per year.",
        metadata={"topic": "sea_level_rise", "source": "NASA", "confidence": 0.9}
    )
    
    engine.memory_system.store(
        content="Renewable energy sources such as solar and wind have become cost-competitive with fossil fuels in many markets.",
        metadata={"topic": "renewable_energy", "source": "IEA", "confidence": 0.85}
    )
    
    engine.memory_system.store(
        content="Current carbon capture technologies can remove CO2 at a cost of $100-300 per ton.",
        metadata={"topic": "carbon_capture", "source": "Research Papers", "confidence": 0.8}
    )
    
    print("Knowledge base setup complete.")


def demonstrate_causal_reasoning(engine):
    """Demonstrate causal reasoning capabilities."""
    print("\n=== Causal Reasoning Example ===")
    
    # Define the reasoning query
    query = "If deforestation increases, what effects would that have on sea levels?"
    
    print(f"Query: {query}\n")
    
    # Perform symbolic causal reasoning
    print("Performing symbolic causal reasoning...")
    
    # Get the symbols for deforestation and sea level rise
    deforestation = engine.symbolic_system.find_symbol_by_name("deforestation")
    sea_level_rise = engine.symbolic_system.find_symbol_by_name("sea_level_rise")
    
    # Find causal paths between these symbols
    causal_paths = engine.symbolic_system.find_paths(
        source=deforestation,
        target=sea_level_rise,
        relation_types=["increases", "contributes_to", "causes"],
        max_length=5
    )
    
    print(f"Found {len(causal_paths)} causal paths:")
    for i, path in enumerate(causal_paths, 1):
        print(f"Path {i}:")
        for step in path:
            print(f"  {step['source']} -- {step['relation']} --> {step['target']}")
    
    # Augment with neural reasoning
    print("\nAugmenting with neural reasoning...")
    
    neural_result = engine.neural_system.generate(
        f"Based on scientific knowledge, explain the specific mechanisms by which {query.lower()}"
    )
    
    print("Neural system response:")
    print(f"{neural_result}\n")
    
    # Hybrid reasoning result
    print("Performing hybrid reasoning...")
    
    result = engine.hybrid_reason(
        query,
        symbolic_steps=["identify_concepts", "find_causal_paths", "evaluate_evidence"],
        neural_steps=["elaborate_mechanisms", "quantify_effects", "assess_confidence"]
    )
    
    print("\nHybrid reasoning result:")
    print(f"Answer: {result['answer']}")
    print("\nReasoning trace:")
    for step in result.get('reasoning_trace', []):
        print(f"Step {step['step']}: {step['description']}")
    
    print(f"\nConfidence: {result.get('confidence', 'N/A')}")
    
    return result


def demonstrate_counterfactual_reasoning(engine):
    """Demonstrate counterfactual reasoning capabilities."""
    print("\n=== Counterfactual Reasoning Example ===")
    
    # Define the reasoning query
    query = "What would happen to global temperatures if all countries immediately switched to 100% renewable energy?"
    
    print(f"Query: {query}\n")
    
    # Perform symbolic counterfactual setup
    print("Setting up counterfactual scenario...")
    
    # Create a counterfactual world state
    counterfactual_state = engine.symbolic_system.create_counterfactual_state(
        changes=[
            {"symbol": "carbon_emissions", "property": "rate", "value": "decreasing_rapidly"},
            {"symbol": "renewable_energy", "property": "adoption", "value": "global_100_percent"}
        ]
    )
    
    # Perform counterfactual simulation
    print("Simulating counterfactual effects...")
    
    simulation_results = engine.symbolic_system.simulate_effects(
        counterfactual_state,
        target_symbols=["global_warming", "climate_change", "sea_level_rise"],
        time_steps=10
    )
    
    print("Simulation results:")
    for time_step, state in enumerate(simulation_results):
        print(f"Time step {time_step}:")
        for symbol, properties in state.items():
            print(f"  {symbol}: {properties}")
    
    # Neural analysis of the counterfactual
    print("\nNeural analysis of counterfactual scenario...")
    
    neural_analysis = engine.neural_system.generate(
        f"Analyzing the scientific plausibility of this scenario: {query}. "
        "Include considerations about inertia in the climate system, existing CO2 levels, "
        "and realistic timelines for climate effects."
    )
    
    print("Neural analysis:")
    print(f"{neural_analysis}\n")
    
    # Hybrid counterfactual reasoning
    print("Performing hybrid counterfactual reasoning...")
    
    result = engine.hybrid_counterfactual_reason(
        query,
        counterfactual_state=counterfactual_state,
        time_horizon_years=50,
        confidence_threshold=0.7
    )
    
    print("\nHybrid counterfactual reasoning result:")
    print(f"Answer: {result['answer']}")
    print("\nScenario timeline:")
    for time_point in result.get('timeline', []):
        print(f"Year {time_point['year']}: {time_point['description']}")
    
    print(f"\nPlausibility: {result.get('plausibility', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    
    return result


def demonstrate_abductive_reasoning(engine):
    """Demonstrate abductive reasoning (inference to best explanation)."""
    print("\n=== Abductive Reasoning Example ===")
    
    # Define the reasoning query
    query = "Why are we observing both extreme heat waves and unusual cold snaps in recent years?"
    
    print(f"Query: {query}\n")
    
    # Gather evidence from memory
    print("Gathering evidence...")
    
    evidence = engine.memory_system.retrieve(
        query="extreme weather patterns temperature anomalies",
        limit=5
    )
    
    print(f"Retrieved {len(evidence)} pieces of evidence:")
    for i, item in enumerate(evidence, 1):
        print(f"{i}. {item.get('content')} (Confidence: {item.get('metadata', {}).get('confidence', 'N/A')})")
    
    # Generate possible hypotheses
    print("\nGenerating possible explanations...")
    
    hypotheses = engine.symbolic_system.generate_hypotheses(
        observation="extreme temperature patterns",
        domain="climate_science",
        max_hypotheses=3
    )
    
    print("Generated hypotheses:")
    for i, hypothesis in enumerate(hypotheses, 1):
        print(f"{i}. {hypothesis['name']}: {hypothesis['description']}")
    
    # Evaluate hypotheses
    print("\nEvaluating hypotheses...")
    
    evaluation = engine.evaluate_hypotheses(
        hypotheses,
        evidence,
        criteria=["consistency", "explanatory_power", "simplicity", "testability"]
    )
    
    print("Hypothesis evaluation:")
    for hypothesis, scores in evaluation.items():
        print(f"\n{hypothesis}:")
        for criterion, score in scores.items():
            print(f"  {criterion}: {score}")
        print(f"  Overall score: {scores.get('overall', 'N/A')}")
    
    # Neural analysis
    print("\nNeural analysis of possible explanations...")
    
    neural_explanation = engine.neural_system.generate(
        f"As a climate scientist, what is the most scientifically sound explanation for this observation: {query}"
    )
    
    print("Neural explanation:")
    print(f"{neural_explanation}\n")
    
    # Hybrid abductive reasoning
    print("Performing hybrid abductive reasoning...")
    
    result = engine.hybrid_abductive_reason(
        query,
        evidence=evidence,
        initial_hypotheses=hypotheses,
        evaluation_criteria=["consistency", "explanatory_power", "simplicity", "testability"]
    )
    
    print("\nHybrid abductive reasoning result:")
    print(f"Best explanation: {result['best_explanation']['name']}")
    print(f"Description: {result['best_explanation']['description']}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    
    print("\nAlternative explanations:")
    for alt in result.get('alternative_explanations', []):
        print(f"- {alt['name']} (Score: {alt.get('score', 'N/A')}): {alt['description']}")
    
    return result


def main():
    """Main function to demonstrate reasoning capabilities."""
    print("Initializing Cognitive Engine...")
    engine = HybridCognitiveEngine()
    
    # Set up the knowledge base
    setup_knowledge_base(engine)
    
    # Demonstrate different types of reasoning
    causal_result = demonstrate_causal_reasoning(engine)
    counterfactual_result = demonstrate_counterfactual_reasoning(engine)
    abductive_result = demonstrate_abductive_reasoning(engine)
    
    # Compare and contrast reasoning approaches
    print("\n=== Comparison of Reasoning Approaches ===")
    print("\nComparing different reasoning methods...")
    
    comparison = engine.process(
        "Compare and contrast the strengths and limitations of causal, "
        "counterfactual, and abductive reasoning as demonstrated in the examples.",
        context={
            "causal_reasoning": causal_result,
            "counterfactual_reasoning": counterfactual_result,
            "abductive_reasoning": abductive_result
        }
    )
    
    print("\nComparison results:")
    print(comparison['response'])
    

if __name__ == "__main__":
    main() 