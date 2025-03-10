"""
Long Term Pervasive Memory system for working with long term memory.
"""

from cognitive_engine.memory.storage import MemoryStorage, MemoryNode
from cognitive_engine.memory.retrieval import MemoryRetrieval, QueryEngine, RelevanceScoring

__all__ = [
    'MemoryStorage',
    'MemoryNode',
    'MemoryRetrieval',
    'QueryEngine',
    'RelevanceScoring',
    'PervasiveMemory',
]

class PervasiveMemory:
    """
    The main class for the Long Term Pervasive Memory system.
    Combines storage and retrieval mechanisms.
    """
    
    def __init__(self, storage_config=None, retrieval_config=None):
        """
        Initialize the Pervasive Memory system.
        
        Args:
            storage_config: Configuration for memory storage
            retrieval_config: Configuration for memory retrieval
        """
        self.storage = MemoryStorage(**(storage_config or {}))
        self.retrieval = MemoryRetrieval(**(retrieval_config or {}))
        
    def store(self, key, value, metadata=None):
        """
        Store information in the memory system.
        
        Args:
            key: Identifier for the memory
            value: Content to store
            metadata: Additional information about the memory
            
        Returns:
            The ID of the stored memory
        """
        return self.storage.add(key, value, metadata)
        
    def retrieve(self, query, k=5):
        """
        Retrieve relevant memories based on a query.
        
        Args:
            query: The query to match against memories
            k: Number of results to return
            
        Returns:
            List of retrieved memories ordered by relevance
        """
        return self.retrieval.query(query, self.storage, k) 