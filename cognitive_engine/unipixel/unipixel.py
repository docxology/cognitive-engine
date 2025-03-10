"""
Unipixel: Fractal unit/atom/pixel/basis for each layer of the fractal system.

Each Unipixel represents an object-process in the Active Inference / Free Energy Principle
framework. In practice, each Unipixel functions like a row in a database and can be a sub-program.
"""

from typing import Dict, List, Any, Optional, Union, Callable
import uuid
import time
import json
import networkx as nx


class Unipixel:
    """
    A fundamental unit of the fractal system representing an object-process.
    
    Each Unipixel can:
    1. Store state information
    2. Execute functions/behaviors
    3. Interact with other Unipixels
    4. Adapt based on experience
    5. Scale across different fractal layers
    """
    
    def __init__(
        self,
        name: str,
        layer: int,
        properties: Optional[Dict[str, Any]] = None,
        behaviors: Optional[Dict[str, Callable]] = None,
        parent_id: Optional[str] = None
    ):
        """
        Initialize a Unipixel.
        
        Args:
            name: Name of the Unipixel
            layer: Fractal layer (0-6) this Unipixel belongs to
            properties: State information and properties
            behaviors: Mapping of behavior names to callable functions
            parent_id: ID of parent Unipixel if any
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.layer = layer
        self.properties = properties or {}
        self.behaviors = behaviors or {}
        self.parent_id = parent_id
        self.children_ids = []
        self.connected_ids = []
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.experience_count = 0
        
        # Active Inference / Free Energy Principle components
        self.beliefs = {}  # Prior probabilities
        self.observations = []  # Sensory data
        self.prediction_errors = []  # Differences between beliefs and observations
        self.free_energy = 0.0  # Measure of surprise
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Unipixel to a dictionary representation.
        
        Returns:
            Dictionary representation of the Unipixel
        """
        return {
            'id': self.id,
            'name': self.name,
            'layer': self.layer,
            'properties': self.properties,
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'connected_ids': self.connected_ids,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'experience_count': self.experience_count,
            'beliefs': self.beliefs,
            'observations': self.observations,
            'free_energy': self.free_energy
        }
    
    def update(self, properties: Dict[str, Any]) -> None:
        """
        Update the Unipixel's properties.
        
        Args:
            properties: New properties to update or add
        """
        self.properties.update(properties)
        self.updated_at = time.time()
    
    def add_behavior(self, name: str, behavior_fn: Callable) -> None:
        """
        Add a behavior function to the Unipixel.
        
        Args:
            name: Name of the behavior
            behavior_fn: Function implementing the behavior
        """
        self.behaviors[name] = behavior_fn
        self.updated_at = time.time()
    
    def execute(self, behavior_name: str, *args, **kwargs) -> Any:
        """
        Execute a behavior of the Unipixel.
        
        Args:
            behavior_name: Name of the behavior to execute
            *args: Positional arguments for the behavior
            **kwargs: Keyword arguments for the behavior
            
        Returns:
            Result of the behavior execution
            
        Raises:
            ValueError: If behavior does not exist
        """
        if behavior_name not in self.behaviors:
            raise ValueError(f"Behavior '{behavior_name}' not found")
            
        result = self.behaviors[behavior_name](*args, **kwargs)
        self.experience_count += 1
        self.updated_at = time.time()
        
        return result
    
    def add_child(self, child_id: str) -> None:
        """
        Add a child Unipixel to this Unipixel.
        
        Args:
            child_id: ID of the child Unipixel
        """
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
            self.updated_at = time.time()
    
    def connect(self, other_id: str) -> None:
        """
        Connect this Unipixel to another Unipixel.
        
        Args:
            other_id: ID of the other Unipixel
        """
        if other_id not in self.connected_ids:
            self.connected_ids.append(other_id)
            self.updated_at = time.time()
    
    def update_beliefs(self, new_beliefs: Dict[str, Any]) -> None:
        """
        Update the Unipixel's beliefs (priors).
        
        Args:
            new_beliefs: New beliefs to update or add
        """
        self.beliefs.update(new_beliefs)
        self.updated_at = time.time()
    
    def add_observation(self, observation: Any) -> None:
        """
        Add a new observation to the Unipixel.
        
        Args:
            observation: The observation data
        """
        self.observations.append({
            'data': observation,
            'timestamp': time.time()
        })
        self.updated_at = time.time()
        
        # Update free energy based on this observation
        self._update_free_energy()
    
    def _update_free_energy(self) -> None:
        """
        Update the free energy measure based on current beliefs and observations.
        
        This is a simplified implementation of the Free Energy Principle.
        In a real implementation, this would use more sophisticated methods.
        """
        # Simple implementation: free energy as a measure of surprise
        # For each belief, calculate the difference with observations
        total_error = 0.0
        error_count = 0
        
        for belief_key, belief_value in self.beliefs.items():
            # Find matching observations
            matching_obs = [
                obs['data'].get(belief_key) 
                for obs in self.observations[-5:] 
                if isinstance(obs['data'], dict) and belief_key in obs['data']
            ]
            
            if matching_obs:
                # Calculate average observed value
                avg_observed = sum(matching_obs) / len(matching_obs)
                
                # Calculate squared error
                error = (belief_value - avg_observed) ** 2
                total_error += error
                error_count += 1
                
                # Store prediction error
                self.prediction_errors.append({
                    'key': belief_key,
                    'expected': belief_value,
                    'observed': avg_observed,
                    'error': error,
                    'timestamp': time.time()
                })
        
        # Update free energy (average prediction error)
        if error_count > 0:
            self.free_energy = total_error / error_count


class UnipixelRegistry:
    """
    Registry for managing Unipixels across the system.
    
    Provides mechanisms for creating, retrieving, and connecting Unipixels.
    """
    
    def __init__(self):
        """
        Initialize the Unipixel registry.
        """
        self.unipixels = {}
        self.graph = nx.DiGraph()
    
    def create_unipixel(
        self,
        name: str,
        layer: int,
        properties: Optional[Dict[str, Any]] = None,
        behaviors: Optional[Dict[str, Callable]] = None,
        parent_id: Optional[str] = None
    ) -> Unipixel:
        """
        Create a new Unipixel and register it.
        
        Args:
            name: Name of the Unipixel
            layer: Fractal layer (0-6) this Unipixel belongs to
            properties: State information and properties
            behaviors: Mapping of behavior names to callable functions
            parent_id: ID of parent Unipixel if any
            
        Returns:
            The created Unipixel
        """
        # Create the new Unipixel
        unipixel = Unipixel(
            name=name,
            layer=layer,
            properties=properties,
            behaviors=behaviors,
            parent_id=parent_id
        )
        
        # Register the Unipixel
        self.unipixels[unipixel.id] = unipixel
        
        # Add to the graph
        self.graph.add_node(unipixel.id, unipixel=unipixel)
        
        # Connect to parent if specified
        if parent_id and parent_id in self.unipixels:
            parent = self.unipixels[parent_id]
            parent.add_child(unipixel.id)
            self.graph.add_edge(parent_id, unipixel.id, relationship='parent-child')
        
        return unipixel
    
    def get_unipixel(self, unipixel_id: str) -> Optional[Unipixel]:
        """
        Get a Unipixel by ID.
        
        Args:
            unipixel_id: ID of the Unipixel to get
            
        Returns:
            The Unipixel if found, None otherwise
        """
        return self.unipixels.get(unipixel_id)
    
    def connect_unipixels(self, source_id: str, target_id: str, relationship_type: str = 'connected') -> bool:
        """
        Connect two Unipixels.
        
        Args:
            source_id: ID of the source Unipixel
            target_id: ID of the target Unipixel
            relationship_type: Type of relationship
            
        Returns:
            True if connection was made, False otherwise
        """
        if source_id not in self.unipixels or target_id not in self.unipixels:
            return False
        
        # Update the Unipixels
        source = self.unipixels[source_id]
        source.connect(target_id)
        
        # Update the graph
        self.graph.add_edge(source_id, target_id, relationship=relationship_type)
        
        return True
    
    def get_children(self, unipixel_id: str) -> List[Unipixel]:
        """
        Get all children of a Unipixel.
        
        Args:
            unipixel_id: ID of the parent Unipixel
            
        Returns:
            List of child Unipixels
        """
        if unipixel_id not in self.unipixels:
            return []
        
        unipixel = self.unipixels[unipixel_id]
        return [self.unipixels[child_id] for child_id in unipixel.children_ids if child_id in self.unipixels]
    
    def get_connected(self, unipixel_id: str) -> List[Unipixel]:
        """
        Get all Unipixels connected to a Unipixel.
        
        Args:
            unipixel_id: ID of the Unipixel
            
        Returns:
            List of connected Unipixels
        """
        if unipixel_id not in self.unipixels:
            return []
        
        unipixel = self.unipixels[unipixel_id]
        return [self.unipixels[conn_id] for conn_id in unipixel.connected_ids if conn_id in self.unipixels]
    
    def get_by_layer(self, layer: int) -> List[Unipixel]:
        """
        Get all Unipixels at a specific layer.
        
        Args:
            layer: Layer number (0-6)
            
        Returns:
            List of Unipixels at the specified layer
        """
        return [u for u in self.unipixels.values() if u.layer == layer]
    
    def to_json(self) -> str:
        """
        Convert the registry to a JSON string.
        
        Returns:
            JSON representation of the registry
        """
        data = {uid: unipixel.to_dict() for uid, unipixel in self.unipixels.items()}
        return json.dumps(data, indent=2)
    
    def from_json(self, json_str: str) -> None:
        """
        Load registry data from a JSON string.
        
        Args:
            json_str: JSON string with registry data
        """
        data = json.loads(json_str)
        
        # Clear existing data
        self.unipixels = {}
        self.graph = nx.DiGraph()
        
        # Create Unipixels
        for uid, unipixel_data in data.items():
            unipixel = Unipixel(
                name=unipixel_data['name'],
                layer=unipixel_data['layer'],
                properties=unipixel_data.get('properties', {}),
                parent_id=unipixel_data.get('parent_id')
            )
            
            # Restore all fields
            unipixel.id = uid
            unipixel.children_ids = unipixel_data.get('children_ids', [])
            unipixel.connected_ids = unipixel_data.get('connected_ids', [])
            unipixel.created_at = unipixel_data.get('created_at', time.time())
            unipixel.updated_at = unipixel_data.get('updated_at', time.time())
            unipixel.experience_count = unipixel_data.get('experience_count', 0)
            unipixel.beliefs = unipixel_data.get('beliefs', {})
            unipixel.observations = unipixel_data.get('observations', [])
            unipixel.free_energy = unipixel_data.get('free_energy', 0.0)
            
            # Register the Unipixel
            self.unipixels[uid] = unipixel
            self.graph.add_node(uid, unipixel=unipixel)
        
        # Restore connections
        for uid, unipixel in self.unipixels.items():
            # Parent-child connections
            if unipixel.parent_id and unipixel.parent_id in self.unipixels:
                self.graph.add_edge(unipixel.parent_id, uid, relationship='parent-child')
            
            # Other connections
            for conn_id in unipixel.connected_ids:
                if conn_id in self.unipixels:
                    self.graph.add_edge(uid, conn_id, relationship='connected') 