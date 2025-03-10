"""
Template system for Fractal symbolic structures.

Templates provide reusable patterns for creating complex symbolic structures.
"""

from typing import Dict, List, Any, Optional, Union, Callable
import uuid
import copy


class Template:
    """
    A template for creating complex symbolic structures.
    
    Templates define patterns for nodes, properties, and bindings
    that can be instantiated multiple times with different parameters.
    """
    
    def __init__(
        self,
        name: str,
        nodes: List[Dict[str, Any]],
        bindings: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a template.
        
        Args:
            name: Name of the template
            nodes: List of node definitions
            bindings: List of binding definitions
            metadata: Additional metadata about the template
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.nodes = nodes
        self.bindings = bindings or []
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the template to a dictionary representation.
        
        Returns:
            Dictionary representation of the template
        """
        return {
            'id': self.id,
            'name': self.name,
            'nodes': self.nodes,
            'bindings': self.bindings,
            'metadata': self.metadata
        }
    
    def instantiate(
        self, 
        fractal_system,
        parameters: Dict[str, Any],
        parent_node_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Instantiate the template in a fractal system.
        
        Args:
            fractal_system: The fractal system to instantiate in
            parameters: Parameters to use in instantiation
            parent_node_id: ID of parent node if applicable
            
        Returns:
            Dictionary mapping template node IDs to instantiated node IDs
        """
        # First, create all nodes
        node_mapping = {}
        
        for node_def in self.nodes:
            # Parse template parameters
            name = self._parse_template_value(node_def['name'], parameters)
            level = node_def['level']
            properties = self._parse_template_dict(node_def.get('properties', {}), parameters)
            
            # Determine how to create the node
            if level == 0 and not parent_node_id:
                # Create root node
                node = fractal_system.create_root(name, properties)
            else:
                # Check if parent is specified
                parent_id = node_def.get('parent_id')
                if parent_id:
                    # If parent is from the template, map it
                    if parent_id in node_mapping:
                        parent_id = node_mapping[parent_id]
                    # Otherwise, use the provided parent_node_id
                elif parent_node_id:
                    parent_id = parent_node_id
                else:
                    raise ValueError(f"No parent specified for non-root node {name}")
                
                # Get the parent node
                parent = fractal_system.get_node(parent_id)
                if not parent:
                    raise ValueError(f"Parent node {parent_id} not found")
                
                # Create child node
                node = parent.create_child(name, properties)
            
            # Store the mapping from template ID to actual ID
            node_mapping[node_def['id']] = node.id
        
        # If the template has a binding mechanism, create bindings
        if hasattr(fractal_system, 'binding_mechanism') and fractal_system.binding_mechanism:
            for binding_def in self.bindings:
                # Map source and target IDs
                source_id = node_mapping.get(binding_def['source_id'])
                target_id = node_mapping.get(binding_def['target_id'])
                
                if not source_id or not target_id:
                    continue
                
                # Parse template parameters
                relation_type = self._parse_template_value(binding_def['relation_type'], parameters)
                properties = self._parse_template_dict(binding_def.get('properties', {}), parameters)
                
                # Create the binding
                fractal_system.binding_mechanism.create_binding(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    properties=properties
                )
        
        return node_mapping
    
    def _parse_template_value(self, value: Any, parameters: Dict[str, Any]) -> Any:
        """
        Parse a template value, substituting parameters.
        
        Args:
            value: Template value to parse
            parameters: Parameters to substitute
            
        Returns:
            Parsed value
        """
        if isinstance(value, str) and value.startswith('$'):
            param_name = value[1:]
            if param_name in parameters:
                return parameters[param_name]
        return value
    
    def _parse_template_dict(self, template_dict: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a template dictionary, substituting parameters.
        
        Args:
            template_dict: Template dictionary to parse
            parameters: Parameters to substitute
            
        Returns:
            Parsed dictionary
        """
        result = {}
        for key, value in template_dict.items():
            parsed_key = self._parse_template_value(key, parameters)
            
            if isinstance(value, dict):
                parsed_value = self._parse_template_dict(value, parameters)
            elif isinstance(value, list):
                parsed_value = [self._parse_template_value(item, parameters) for item in value]
            else:
                parsed_value = self._parse_template_value(value, parameters)
                
            result[parsed_key] = parsed_value
            
        return result


class TemplateLibrary:
    """
    A library of templates for reuse.
    
    Provides methods for storing, retrieving, and managing templates.
    """
    
    def __init__(self):
        """
        Initialize the template library.
        """
        self.templates = {}
        
    def add_template(self, template: Template) -> str:
        """
        Add a template to the library.
        
        Args:
            template: Template to add
            
        Returns:
            ID of the added template
        """
        self.templates[template.id] = template
        return template.id
    
    def get_template(self, template_id: str) -> Optional[Template]:
        """
        Get a template by ID.
        
        Args:
            template_id: ID of the template to get
            
        Returns:
            The template if found, None otherwise
        """
        return self.templates.get(template_id)
    
    def get_template_by_name(self, name: str) -> Optional[Template]:
        """
        Get a template by name.
        
        Args:
            name: Name of the template to get
            
        Returns:
            The first template with matching name if found, None otherwise
        """
        for template in self.templates.values():
            if template.name == name:
                return template
        return None
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all templates in the library.
        
        Returns:
            List of template metadata
        """
        return [
            {'id': t.id, 'name': t.name, 'metadata': t.metadata}
            for t in self.templates.values()
        ]
    
    def remove_template(self, template_id: str) -> bool:
        """
        Remove a template from the library.
        
        Args:
            template_id: ID of the template to remove
            
        Returns:
            True if template was removed, False otherwise
        """
        if template_id in self.templates:
            del self.templates[template_id]
            return True
        return False 