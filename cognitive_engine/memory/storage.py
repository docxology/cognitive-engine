"""
Memory storage mechanisms for the Long Term Pervasive Memory system.
"""

from typing import Dict, List, Any, Optional, Union, Set
import uuid
import time
import json
import os


class MemoryNode:
    """
    A node in the memory storage system.
    
    Represents a single memory item with metadata and retrieval information.
    """
    
    def __init__(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ):
        """
        Initialize a memory node.
        
        Args:
            key: Key or identifier for the memory
            value: Content of the memory
            metadata: Additional metadata about the memory
            timestamp: Creation timestamp, defaults to current time
        """
        self.id = str(uuid.uuid4())
        self.key = key
        self.value = value
        self.metadata = metadata or {}
        self.timestamp = timestamp or time.time()
        self.access_count = 0
        self.last_access = self.timestamp
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the memory node to a dictionary representation.
        
        Returns:
            Dictionary representation of the memory node
        """
        return {
            'id': self.id,
            'key': self.key,
            'value': self.value,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'access_count': self.access_count,
            'last_access': self.last_access
        }
    
    def access(self) -> None:
        """
        Record an access to this memory node.
        
        Updates the access count and last access timestamp.
        """
        self.access_count += 1
        self.last_access = time.time()


class MemoryStorage:
    """
    Storage system for the Long Term Pervasive Memory.
    
    Provides mechanisms to store, retrieve, and manage memories.
    """
    
    def __init__(
        self,
        persistence_path: Optional[str] = None,
        max_memories: int = 10000
    ):
        """
        Initialize the memory storage system.
        
        Args:
            persistence_path: Path to persist memories to disk
            max_memories: Maximum number of memories to store
        """
        self.memories = {}
        self.index_by_key = {}
        self.persistence_path = persistence_path
        self.max_memories = max_memories
        
        # Load persisted memories if path provided
        if persistence_path and os.path.exists(persistence_path):
            self.load()
    
    def add(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a memory to storage.
        
        Args:
            key: Key or identifier for the memory
            value: Content of the memory
            metadata: Additional metadata about the memory
            
        Returns:
            ID of the stored memory
        """
        # Create memory node
        node = MemoryNode(key=key, value=value, metadata=metadata)
        
        # Store the memory
        self.memories[node.id] = node
        
        # Update index
        if key in self.index_by_key:
            self.index_by_key[key].append(node.id)
        else:
            self.index_by_key[key] = [node.id]
        
        # Check if we need to trim memories
        if len(self.memories) > self.max_memories:
            self._trim_memories()
        
        # Persist if path provided
        if self.persistence_path:
            self.persist()
        
        return node.id
    
    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a memory by ID.
        
        Args:
            memory_id: ID of the memory to get
            
        Returns:
            The memory node as a dictionary if found, None otherwise
        """
        node = self.memories.get(memory_id)
        if not node:
            return None
        
        # Record access
        node.access()
        
        return node.to_dict()
    
    def get_by_key(self, key: str) -> List[Dict[str, Any]]:
        """
        Get memories by key.
        
        Args:
            key: Key to get memories for
            
        Returns:
            List of memory nodes matching the key
        """
        node_ids = self.index_by_key.get(key, [])
        results = []
        
        for node_id in node_ids:
            node = self.memories.get(node_id)
            if node:
                node.access()
                results.append(node.to_dict())
        
        return results
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if memory was deleted, False otherwise
        """
        node = self.memories.pop(memory_id, None)
        if not node:
            return False
        
        # Update index
        key = node.key
        if key in self.index_by_key:
            self.index_by_key[key] = [nid for nid in self.index_by_key[key] if nid != memory_id]
            if not self.index_by_key[key]:
                del self.index_by_key[key]
        
        # Persist if path provided
        if self.persistence_path:
            self.persist()
        
        return True
    
    def clear(self) -> None:
        """
        Clear all memories.
        """
        self.memories = {}
        self.index_by_key = {}
        
        # Persist empty state if path provided
        if self.persistence_path:
            self.persist()
    
    def persist(self) -> None:
        """
        Persist memories to disk.
        
        Saves the current state of the memory storage system.
        """
        if not self.persistence_path:
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        
        # Convert memories to serializable format
        data = {
            'memories': {mid: node.to_dict() for mid, node in self.memories.items()},
            'index_by_key': self.index_by_key
        }
        
        # Write to file
        with open(self.persistence_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self) -> None:
        """
        Load memories from disk.
        
        Loads the persisted state of the memory storage system.
        """
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return
        
        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct memories
            self.memories = {}
            for mid, node_dict in data.get('memories', {}).items():
                node = MemoryNode(
                    key=node_dict['key'],
                    value=node_dict['value'],
                    metadata=node_dict.get('metadata', {}),
                    timestamp=node_dict.get('timestamp', time.time())
                )
                node.id = mid
                node.access_count = node_dict.get('access_count', 0)
                node.last_access = node_dict.get('last_access', node.timestamp)
                self.memories[mid] = node
            
            # Reconstruct index
            self.index_by_key = data.get('index_by_key', {})
        except Exception as e:
            print(f"Error loading memories: {e}")
    
    def _trim_memories(self) -> None:
        """
        Trim the number of memories to the maximum allowed.
        
        Removes the least recently accessed memories.
        """
        # If we're under the limit, do nothing
        if len(self.memories) <= self.max_memories:
            return
        
        # Sort memories by last access time
        sorted_memories = sorted(
            self.memories.items(),
            key=lambda x: (x[1].last_access, x[1].access_count)
        )
        
        # Determine how many to remove
        to_remove = len(self.memories) - self.max_memories
        
        # Remove the least recently accessed memories
        for i in range(to_remove):
            memory_id, node = sorted_memories[i]
            self.delete(memory_id) 