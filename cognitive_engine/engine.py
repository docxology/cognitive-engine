"""
Main engine class that integrates fractal symbolic system with probabilistic neural systems.
"""

from typing import Dict, List, Any, Optional, Union
import json

from cognitive_engine.fractal import FractalSystem
from cognitive_engine.probabilistic import LLMInterface, ProbabilisticEngine
from cognitive_engine.memory import PervasiveMemory
from cognitive_engine.unipixel import UnipixelRegistry
from cognitive_engine.mmm import MagicalMathModel
from cognitive_engine.perplexity_api import PerplexitySearchEngine


class HybridCognitiveEngine:
    """
    The main hybrid Neuro-Symbolic AI system that integrates:
    1. Fractal symbolic system
    2. Probabilistic neural systems
    3. Long-term pervasive memory
    4. Unipixel registry for fractal unit management
    5. Magical Math Model for pattern finding
    6. Perplexity API for internet search capabilities
    """
    
    def __init__(
        self,
        symbolic_system: Optional[FractalSystem] = None,
        neural_system: Optional[Union[LLMInterface, ProbabilisticEngine]] = None,
        memory_system: Optional[PervasiveMemory] = None,
        unipixel_registry: Optional[UnipixelRegistry] = None,
        mmm: Optional[MagicalMathModel] = None,
        search_engine: Optional[PerplexitySearchEngine] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the hybrid cognitive engine.
        
        Args:
            symbolic_system: Fractal symbolic system component
            neural_system: Neural/probabilistic system component
            memory_system: Long-term memory system
            unipixel_registry: Registry for Unipixel management
            mmm: Magical Math Model for pattern finding
            search_engine: Perplexity search engine for internet access
            config: Additional configuration parameters
        """
        self.config = config or {}
        
        # Initialize systems or use provided ones
        self.symbolic = symbolic_system or FractalSystem(
            levels=self.config.get('fractal_levels', 7)
        )
        
        self.neural = neural_system
        if self.neural is None:
            if self.config.get('use_llm', True):
                self.neural = LLMInterface(model=self.config.get('llm_model', 'gpt-4'))
            else:
                self.neural = ProbabilisticEngine()
        
        self.memory = memory_system or PervasiveMemory()
        
        # Initialize the Unipixel registry
        self.unipixel_registry = unipixel_registry or UnipixelRegistry()
        
        # Initialize the Magical Math Model
        self.mmm = mmm or MagicalMathModel(
            unipixel_registry=self.unipixel_registry,
            fractal_system=self.symbolic
        )
        
        # Initialize the Perplexity search engine
        perplexity_api_key = self.config.get('perplexity_api_key')
        cache_dir = self.config.get('perplexity_cache_dir')
        self.search_engine = search_engine or PerplexitySearchEngine(
            api_key=perplexity_api_key,
            cache_dir=cache_dir,
            memory_system=self.memory
        )
        
    def process(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process information through the hybrid system.
        
        The processing flow:
        1. Symbolic representation and parsing
        2. Memory augmentation
        3. Neural processing for generation/completion
        4. Integration of neural outputs into symbolic representation
        5. Memory update
        
        Args:
            input_data: Input data to process, either as text or structured data
            
        Returns:
            Processed results including symbolic representation and neural outputs
        """
        # Convert input to string if needed for symbolic parsing
        input_str = input_data if isinstance(input_data, str) else str(input_data)
        
        # Step 1: Symbolic parsing and representation
        symbolic_representation = self.symbolic.parse(input_str)
        
        # Step 2: Augment with relevant memories
        memories = self.memory.retrieve(input_str)
        augmented_representation = self.symbolic.augment(symbolic_representation, memories)
        
        # Step 3: Neural processing
        neural_outputs = self.neural.process(
            input_data=input_str,
            symbolic_context=augmented_representation,
            memories=memories
        )
        
        # Step 4: Integration of neural outputs into symbolic representation
        integrated_representation = self.symbolic.integrate(
            augmented_representation, 
            neural_outputs
        )
        
        # Step 5: Update memory with new information
        memory_id = self.memory.store(
            key=input_str,
            value={
                'input': input_data,
                'symbolic': integrated_representation,
                'neural': neural_outputs
            }
        )
        
        # Return the processed results
        return {
            'input': input_data,
            'symbolic_representation': integrated_representation,
            'neural_outputs': neural_outputs,
            'memory_id': memory_id
        }
    
    def reason(self, query: str) -> Dict[str, Any]:
        """
        Perform hybrid reasoning on a query.
        
        Args:
            query: The query or problem statement to reason about
            
        Returns:
            Reasoning results and explanation
        """
        # Similar to process but with emphasis on reasoning structures
        symbolic_query = self.symbolic.parse(query)
        relevant_memories = self.memory.retrieve(query)
        
        # Perform symbolic reasoning
        symbolic_reasoning = self.symbolic.reason(symbolic_query, relevant_memories)
        
        # Enhance with neural reasoning
        neural_reasoning = self.neural.reason(
            query=query,
            symbolic_reasoning=symbolic_reasoning,
            memories=relevant_memories
        )
        
        # Integrate both reasoning approaches
        final_reasoning = self.symbolic.integrate_reasoning(
            symbolic_reasoning,
            neural_reasoning
        )
        
        return {
            'query': query,
            'symbolic_reasoning': symbolic_reasoning,
            'neural_reasoning': neural_reasoning,
            'integrated_reasoning': final_reasoning,
            'conclusion': final_reasoning.get('conclusion', '')
        }
    
    def search_web(self, query: str, save_to_memory: bool = True) -> Dict[str, Any]:
        """
        Search the web for information using the Perplexity API.
        
        Args:
            query: The search query
            save_to_memory: Whether to save the search results to memory
            
        Returns:
            Search results from the Perplexity API
        """
        return self.search_engine.search(query, save_to_memory=save_to_memory)
    
    def research_topic(self, topic: str, depth: int = 2) -> Dict[str, Any]:
        """
        Perform in-depth research on a topic using the Perplexity API.
        
        Args:
            topic: The topic to research
            depth: The depth of research (number of follow-up questions)
            
        Returns:
            Research results including the main topic and follow-up questions
        """
        return self.search_engine.research_topic(topic, depth=depth)
    
    def create_unipixel(
        self,
        name: str,
        layer: int,
        properties: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a Unipixel in the fractal system.
        
        Args:
            name: Name of the Unipixel
            layer: Layer (0-6) the Unipixel belongs to
            properties: Properties of the Unipixel
            parent_id: ID of the parent Unipixel if any
            
        Returns:
            Information about the created Unipixel
        """
        unipixel = self.unipixel_registry.create_unipixel(
            name=name,
            layer=layer,
            properties=properties,
            parent_id=parent_id
        )
        
        return {
            'id': unipixel.id,
            'name': unipixel.name,
            'layer': unipixel.layer,
            'parent_id': unipixel.parent_id,
            'message': f"Created Unipixel '{name}' at layer {layer}"
        }
    
    def find_patterns(self, element_ids: List[str]) -> Dict[str, Any]:
        """
        Find patterns among elements using the MMM.
        
        Args:
            element_ids: IDs of elements to analyze for patterns
            
        Returns:
            Discovered patterns
        """
        patterns = self.mmm.find_patterns(element_ids)
        
        return {
            'element_count': len(element_ids),
            'pattern_count': len(patterns),
            'patterns': [p.to_dict() for p in patterns]
        }
    
    def find_cross_layer_patterns(self) -> Dict[str, Any]:
        """
        Find patterns across different layers using the MMM.
        
        Returns:
            Discovered cross-layer patterns
        """
        patterns = self.mmm.find_cross_layer_patterns()
        
        return {
            'pattern_count': len(patterns),
            'patterns': [p.to_dict() for p in patterns]
        }
    
    def to_json(self) -> str:
        """
        Convert the current state to a JSON string.
        
        Returns:
            JSON representation of the engine state
        """
        # Collect state from all components
        state = {
            'version': '0.1.0',
            'config': self.config,
            'symbolic_system': {
                'type': 'FractalSystem',
                'levels': self.symbolic.levels,
                'node_count': len(self.symbolic.nodes_by_id)
            },
            'neural_system': {
                'type': self.neural.__class__.__name__
            },
            'unipixel_registry': {
                'unipixel_count': len(self.unipixel_registry.unipixels)
            },
            'mmm': {
                'pattern_count': len(self.mmm.patterns)
            }
        }
        
        return json.dumps(state, indent=2) 