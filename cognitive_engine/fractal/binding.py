"""
Symbol binding mechanisms for the Fractal system.
"""

from typing import Dict, List, Any, Optional, Union, Set, Tuple
import uuid


class SymbolBinding:
    """
    A binding between symbols in the fractal system.
    
    Bindings represent relationships between nodes in the fractal hierarchy,
    allowing for complex symbolic structures and reasoning.
    """
    
    def __init__(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a symbol binding.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            relation_type: Type of the relation (e.g., "is-a", "part-of")
            properties: Additional properties of the binding
        """
        self.id = str(uuid.uuid4())
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.properties = properties or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the binding to a dictionary representation.
        
        Returns:
            Dictionary representation of the binding
        """
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type,
            'properties': self.properties
        }


class BindingMechanism:
    """
    Mechanism for creating and managing symbol bindings.
    
    Provides methods for creating, querying, and reasoning with bindings.
    """
    
    def __init__(self, fractal_system=None):
        """
        Initialize the binding mechanism.
        
        Args:
            fractal_system: Reference to the fractal system
        """
        self.fractal_system = fractal_system
        self.bindings = {}
        
    def create_binding(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Optional[SymbolBinding]:
        """
        Create a binding between two nodes.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            relation_type: Type of the relation
            properties: Additional properties of the binding
            
        Returns:
            The created binding, or None if nodes don't exist
        """
        if not self.fractal_system:
            raise ValueError("No fractal system provided")
        
        source_node = self.fractal_system.get_node(source_id)
        target_node = self.fractal_system.get_node(target_id)
        
        if not source_node or not target_node:
            return None
        
        binding = SymbolBinding(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties
        )
        
        self.bindings[binding.id] = binding
        source_node.add_binding(binding)
        
        # Update the graph in the fractal system
        self.fractal_system.graph.add_edge(
            source_id, 
            target_id, 
            binding=binding,
            relation=relation_type
        )
        
        return binding
    
    def get_binding(self, binding_id: str) -> Optional[SymbolBinding]:
        """
        Get a binding by its ID.
        
        Args:
            binding_id: ID of the binding to get
            
        Returns:
            The binding if found, None otherwise
        """
        return self.bindings.get(binding_id)
    
    def get_bindings_by_relation(self, relation_type: str) -> List[SymbolBinding]:
        """
        Get all bindings of a specific relation type.
        
        Args:
            relation_type: Type of relation to filter by
            
        Returns:
            List of bindings matching the relation type
        """
        return [b for b in self.bindings.values() if b.relation_type == relation_type]
    
    def get_bindings_for_node(self, node_id: str) -> List[SymbolBinding]:
        """
        Get all bindings for a specific node (as source).
        
        Args:
            node_id: ID of the node to get bindings for
            
        Returns:
            List of bindings with the node as source
        """
        return [b for b in self.bindings.values() if b.source_id == node_id]
    
    def reason_with_bindings(
        self, 
        start_node_id: str, 
        relation_path: List[str]
    ) -> List[str]:
        """
        Perform reasoning by following a path of relations.
        
        Args:
            start_node_id: ID of the starting node
            relation_path: List of relation types to follow
            
        Returns:
            List of node IDs reachable by following the relation path
        """
        if not self.fractal_system:
            raise ValueError("No fractal system provided")
            
        if not relation_path:
            return [start_node_id]
            
        current_nodes = [start_node_id]
        
        for relation in relation_path:
            next_nodes = []
            for node_id in current_nodes:
                # Find all bindings with this node as source and matching relation
                for binding in self.get_bindings_for_node(node_id):
                    if binding.relation_type == relation:
                        next_nodes.append(binding.target_id)
            
            if not next_nodes:
                # Path can't be followed
                return []
                
            current_nodes = next_nodes
            
        return current_nodes 