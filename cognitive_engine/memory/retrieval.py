"""
Memory retrieval algorithms for the Long Term Pervasive Memory system.
"""

from typing import Dict, List, Any, Optional, Union, Set, Callable
import re
import time
import math
from collections import Counter


class RelevanceScoring:
    """
    Scoring mechanisms for ranking memory relevance to queries.
    """
    
    @staticmethod
    def keyword_match(query: str, memory: Dict[str, Any]) -> float:
        """
        Score based on keyword matching.
        
        Args:
            query: The query string
            memory: The memory to score
            
        Returns:
            Relevance score between 0 and 1
        """
        # Extract keywords from query
        query_words = set(re.findall(r'\w+', query.lower()))
        
        # Extract words from memory key and value
        key_words = set(re.findall(r'\w+', str(memory.get('key', '')).lower()))
        value_words = set(re.findall(r'\w+', str(memory.get('value', '')).lower()))
        
        # Combine with higher weight for key
        memory_words = key_words.union(value_words)
        
        # Calculate intersection
        matches = query_words.intersection(memory_words)
        
        # Calculate score based on proportion of query words matched
        if not query_words:
            return 0.0
            
        return len(matches) / len(query_words)
    
    @staticmethod
    def recency(memory: Dict[str, Any], current_time: Optional[float] = None) -> float:
        """
        Score based on recency.
        
        Args:
            memory: The memory to score
            current_time: Current time, defaults to current time
            
        Returns:
            Recency score between 0 and 1
        """
        if current_time is None:
            current_time = time.time()
            
        timestamp = memory.get('timestamp', current_time)
        
        # Calculate recency score with exponential decay
        # 30 days = 2592000 seconds
        time_diff = current_time - timestamp
        recency = math.exp(-time_diff / 2592000)
        
        return max(0.0, min(1.0, recency))
    
    @staticmethod
    def frequency(memory: Dict[str, Any]) -> float:
        """
        Score based on access frequency.
        
        Args:
            memory: The memory to score
            
        Returns:
            Frequency score between 0 and 1
        """
        # Get access count, default to 0
        access_count = memory.get('access_count', 0)
        
        # Calculate frequency score with log scaling
        frequency = math.log1p(access_count) / 10.0  # log(1+x)/10 gives reasonable scaling
        
        return max(0.0, min(1.0, frequency))
    
    @staticmethod
    def combine(
        keyword_score: float,
        recency_score: float,
        frequency_score: float,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Combine multiple relevance scores.
        
        Args:
            keyword_score: Score based on keyword matching
            recency_score: Score based on recency
            frequency_score: Score based on frequency
            weights: Weights for each score type
            
        Returns:
            Combined relevance score between 0 and 1
        """
        weights = weights or {
            'keyword': 0.6,
            'recency': 0.3,
            'frequency': 0.1
        }
        
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0
            
        combined_score = (
            keyword_score * weights.get('keyword', 0) +
            recency_score * weights.get('recency', 0) +
            frequency_score * weights.get('frequency', 0)
        ) / total_weight
        
        return combined_score


class QueryEngine:
    """
    Engine for matching queries against memories.
    """
    
    def __init__(
        self,
        scoring_fn: Optional[Callable[[str, Dict[str, Any]], float]] = None,
        scoring_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the query engine.
        
        Args:
            scoring_fn: Custom scoring function for ranking memories
            scoring_weights: Weights for combining relevance scores
        """
        self.scoring_fn = scoring_fn
        self.scoring_weights = scoring_weights or {
            'keyword': 0.6,
            'recency': 0.3,
            'frequency': 0.1
        }
    
    def search(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for memories matching the query.
        
        Args:
            query: The query string
            memories: List of memories to search
            k: Number of results to return
            
        Returns:
            List of memories ordered by relevance
        """
        # Score each memory
        memory_scores = []
        current_time = time.time()
        
        for memory in memories:
            if self.scoring_fn:
                # Use custom scoring function if provided
                score = self.scoring_fn(query, memory)
            else:
                # Use default scoring
                keyword_score = RelevanceScoring.keyword_match(query, memory)
                recency_score = RelevanceScoring.recency(memory, current_time)
                frequency_score = RelevanceScoring.frequency(memory)
                
                score = RelevanceScoring.combine(
                    keyword_score,
                    recency_score,
                    frequency_score,
                    self.scoring_weights
                )
            
            memory_scores.append((memory, score))
        
        # Sort by score (descending)
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return [
            {**memory, 'relevance_score': score}
            for memory, score in memory_scores[:k]
        ]
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess the query for better matching.
        
        Args:
            query: The original query
            
        Returns:
            Preprocessed query
        """
        # Remove punctuation and convert to lowercase
        query = re.sub(r'[^\w\s]', ' ', query).lower()
        
        # Remove stopwords (a very basic implementation)
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with'}
        query_words = query.split()
        query_words = [word for word in query_words if word not in stopwords]
        
        return ' '.join(query_words)


class MemoryRetrieval:
    """
    Memory retrieval system for the Long Term Pervasive Memory.
    
    Provides methods for retrieving and querying memories.
    """
    
    def __init__(
        self,
        query_engine: Optional[QueryEngine] = None,
        cache_size: int = 100
    ):
        """
        Initialize the memory retrieval system.
        
        Args:
            query_engine: Query engine for memory retrieval
            cache_size: Size of the retrieval cache
        """
        self.query_engine = query_engine or QueryEngine()
        self.cache = {}
        self.cache_size = cache_size
        self.cache_keys = []  # LRU queue
    
    def query(
        self,
        query_str: str,
        storage,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query memories from storage.
        
        Args:
            query_str: The query string
            storage: Memory storage to query
            k: Number of results to return
            
        Returns:
            List of memories ordered by relevance
        """
        # Preprocess the query
        processed_query = self.query_engine.preprocess_query(query_str)
        
        # Check if we have cached results
        if processed_query in self.cache:
            # Update cache LRU
            self._update_cache_lru(processed_query)
            return self.cache[processed_query]
        
        # Extract keywords for broader search
        keywords = processed_query.split()
        
        # Get candidate memories from storage
        candidate_memories = []
        
        # First try exact key matches
        for keyword in keywords:
            memories = storage.get_by_key(keyword)
            candidate_memories.extend(memories)
        
        # If not enough candidates, get all memories (up to a limit)
        if len(candidate_memories) < k * 2:
            all_memories = []
            # In a real implementation, we would use a more efficient method
            # This is just a placeholder
            for memory_id in list(storage.memories.keys())[:1000]:
                memory = storage.get(memory_id)
                if memory:
                    all_memories.append(memory)
            
            candidate_memories.extend(all_memories)
        
        # Remove duplicates
        seen_ids = set()
        unique_candidates = []
        for memory in candidate_memories:
            if memory['id'] not in seen_ids:
                seen_ids.add(memory['id'])
                unique_candidates.append(memory)
        
        # Search for relevant memories
        results = self.query_engine.search(processed_query, unique_candidates, k)
        
        # Cache results
        self._cache_results(processed_query, results)
        
        return results
    
    def _cache_results(self, query: str, results: List[Dict[str, Any]]) -> None:
        """
        Cache query results.
        
        Args:
            query: The query string
            results: The query results
        """
        # If cache is full, remove the least recently used item
        if len(self.cache) >= self.cache_size and query not in self.cache:
            oldest_query = self.cache_keys.pop(0)
            self.cache.pop(oldest_query, None)
        
        # Add to cache
        self.cache[query] = results
        self._update_cache_lru(query)
    
    def _update_cache_lru(self, query: str) -> None:
        """
        Update the cache LRU queue.
        
        Args:
            query: The query string that was accessed
        """
        # Remove from current position if exists
        if query in self.cache_keys:
            self.cache_keys.remove(query)
        
        # Add to the end (most recently used)
        self.cache_keys.append(query) 