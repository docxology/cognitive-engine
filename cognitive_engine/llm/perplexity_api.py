"""
Perplexity API: Ability to search the internet for information.

This module adapts OneEarth scripts to be more general and useful
for the Cognitive Engine context, providing search capabilities.
"""

from typing import Dict, List, Any, Optional, Union
import requests
import json
import time
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerplexitySearchResult:
    """
    A search result from the Perplexity API.
    """
    
    def __init__(
        self,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]],
        timestamp: Optional[float] = None
    ):
        """
        Initialize a search result.
        
        Args:
            query: The search query
            answer: The answer text from Perplexity
            sources: List of sources used for the answer
            timestamp: Time when the search was performed
        """
        self.query = query
        self.answer = answer
        self.sources = sources
        self.timestamp = timestamp or time.time()
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the search result to a dictionary representation.
        
        Returns:
            Dictionary representation of the search result
        """
        return {
            'query': self.query,
            'answer': self.answer,
            'sources': self.sources,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to the search result.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerplexitySearchResult':
        """
        Create a search result from a dictionary.
        
        Args:
            data: Dictionary with search result data
            
        Returns:
            A PerplexitySearchResult object
        """
        result = cls(
            query=data['query'],
            answer=data['answer'],
            sources=data['sources'],
            timestamp=data.get('timestamp', time.time())
        )
        result.metadata = data.get('metadata', {})
        return result


class PerplexityAPI:
    """
    Client for the Perplexity API to search the internet for information.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cache_expiry: int = 86400  # 24 hours
    ):
        """
        Initialize the Perplexity API client.
        
        Args:
            api_key: Perplexity API key, taken from environment if not provided
            cache_dir: Directory for caching search results
            cache_expiry: Time in seconds to consider cached results valid
        """
        self.api_key = api_key or os.environ.get('PERPLEXITY_API_KEY')
        if not self.api_key:
            logger.warning("No Perplexity API key provided. API requests will fail.")
        
        self.base_url = "https://api.perplexity.ai"
        self.cache_dir = cache_dir
        self.cache_expiry = cache_expiry
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        # Create cache directory if it doesn't exist
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Store recent search results in memory
        self.recent_results = []
    
    def search(
        self,
        query: str,
        use_cache: bool = True,
        search_options: Optional[Dict[str, Any]] = None
    ) -> PerplexitySearchResult:
        """
        Search for information using the Perplexity API.
        
        Args:
            query: The search query
            use_cache: Whether to use cached results if available
            search_options: Additional options for the search
            
        Returns:
            A PerplexitySearchResult object
            
        Raises:
            Exception: If the API request fails
        """
        # Check cache first if enabled
        if use_cache:
            cached_result = self._get_from_cache(query)
            if cached_result:
                logger.info(f"Using cached result for query: {query}")
                return cached_result
        
        # Prepare search options
        options = {
            "model": "pplx-70b-online",
            "max_tokens": 1024,
            "temperature": 0.0,
            "focus": "internet"
        }
        
        if search_options:
            options.update(search_options)
        
        # Prepare request payload
        payload = {
            "model": options["model"],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant with internet access"},
                {"role": "user", "content": query}
            ],
            "options": {
                "temperature": options["temperature"],
                "max_tokens": options["max_tokens"]
            }
        }
        
        try:
            # Make the API request
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract answer and sources
            answer = data["choices"][0]["message"]["content"]
            
            # Extract sources if available
            sources = []
            if "citations" in data:
                for citation in data["citations"]:
                    source = {
                        "title": citation.get("title", ""),
                        "url": citation.get("url", ""),
                        "text": citation.get("text", "")
                    }
                    sources.append(source)
            
            # Create the search result
            result = PerplexitySearchResult(
                query=query,
                answer=answer,
                sources=sources
            )
            
            # Add some metadata about the request
            result.add_metadata("model", options["model"])
            result.add_metadata("focus", options["focus"])
            
            # Cache the result
            self._save_to_cache(result)
            
            # Add to recent results
            self.recent_results.append(result)
            if len(self.recent_results) > 20:  # Keep only recent 20 results in memory
                self.recent_results.pop(0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching Perplexity API: {str(e)}")
            raise
    
    def extract_facts(self, query: str, num_facts: int = 5) -> List[str]:
        """
        Extract specific facts about a topic.
        
        Args:
            query: The topic to extract facts about
            num_facts: Number of facts to extract
            
        Returns:
            List of fact strings
        """
        # Modify the query to explicitly ask for facts
        factual_query = f"Please provide exactly {num_facts} clear, concise and factual statements about: {query}. Format as a numbered list."
        
        # Search using the modified query
        result = self.search(factual_query)
        
        # Parse the answer to extract the facts
        facts = []
        lines = result.answer.split("\n")
        
        for line in lines:
            line = line.strip()
            # Look for numbered lines that likely contain facts
            if line and (line[0].isdigit() and line[1:3] in ['. ', ') ']):
                # Remove the numbering and add to facts
                fact = line[line.find(' ')+1:].strip()
                facts.append(fact)
        
        # If parsing failed, just split by newlines and take top entries
        if not facts and result.answer:
            facts = [line.strip() for line in result.answer.split("\n") if line.strip()]
            facts = facts[:num_facts]
        
        return facts
    
    def compare_topics(self, topic1: str, topic2: str) -> Dict[str, Any]:
        """
        Compare two topics and identify similarities and differences.
        
        Args:
            topic1: First topic to compare
            topic2: Second topic to compare
            
        Returns:
            Dictionary with comparison results
        """
        comparison_query = f"Compare and contrast {topic1} and {topic2}. Provide key similarities and differences in a structured format."
        
        result = self.search(comparison_query)
        
        # Parse the answer to extract structured comparison
        # This is a simplified parser - in a real implementation, we might use LLM to structure the response better
        lines = result.answer.split("\n")
        similarities = []
        differences = []
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            line_lower = line.lower()
            
            # Check for section headers
            if "similarit" in line_lower:
                current_section = "similarities"
                continue
            elif "difference" in line_lower:
                current_section = "differences"
                continue
            
            # Extract points based on current section
            if current_section == "similarities" and (line.startswith('-') or line.startswith('•') or (line[0].isdigit() and line[1:3] in ['. ', ') '])):
                similarities.append(line.lstrip('-•0123456789.) ').strip())
            elif current_section == "differences" and (line.startswith('-') or line.startswith('•') or (line[0].isdigit() and line[1:3] in ['. ', ') '])):
                differences.append(line.lstrip('-•0123456789.) ').strip())
        
        return {
            "topic1": topic1,
            "topic2": topic2,
            "full_text": result.answer,
            "similarities": similarities,
            "differences": differences,
            "sources": result.sources
        }
    
    def get_recent_results(self) -> List[PerplexitySearchResult]:
        """
        Get the recent search results.
        
        Returns:
            List of recent search results
        """
        return self.recent_results
    
    def _get_cache_path(self, query: str) -> str:
        """
        Get the path to the cache file for a query.
        
        Args:
            query: The search query
            
        Returns:
            Path to the cache file
        """
        if not self.cache_dir:
            return ""
            
        # Create a safe filename from the query
        safe_query = "".join(c if c.isalnum() else "_" for c in query)
        safe_query = safe_query[:100]  # Limit length
        
        return os.path.join(self.cache_dir, f"{safe_query}.json")
    
    def _get_from_cache(self, query: str) -> Optional[PerplexitySearchResult]:
        """
        Get a search result from the cache.
        
        Args:
            query: The search query
            
        Returns:
            The cached search result if found and valid, None otherwise
        """
        if not self.cache_dir:
            return None
            
        cache_path = self._get_cache_path(query)
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check if the cache is expired
            timestamp = data.get('timestamp', 0)
            if time.time() - timestamp > self.cache_expiry:
                return None
                
            return PerplexitySearchResult.from_dict(data)
                
        except Exception as e:
            logger.warning(f"Error reading from cache: {str(e)}")
            return None
    
    def _save_to_cache(self, result: PerplexitySearchResult) -> None:
        """
        Save a search result to the cache.
        
        Args:
            result: The search result to cache
        """
        if not self.cache_dir:
            return
            
        cache_path = self._get_cache_path(result.query)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2)
                
        except Exception as e:
            logger.warning(f"Error saving to cache: {str(e)}")
    
    def clear_cache(self) -> None:
        """
        Clear the search cache.
        """
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return
            
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                except Exception as e:
                    logger.warning(f"Error removing cache file {filename}: {str(e)}")


class PerplexitySearchEngine:
    """
    Advanced search engine built on top of Perplexity API with additional features.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        memory_system = None
    ):
        """
        Initialize the Perplexity search engine.
        
        Args:
            api_key: Perplexity API key
            cache_dir: Directory for caching search results
            memory_system: Optional memory system for storing results
        """
        self.api = PerplexityAPI(api_key, cache_dir)
        self.memory_system = memory_system
    
    def search(
        self, 
        query: str, 
        save_to_memory: bool = True,
        search_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for information and optionally save to memory.
        
        Args:
            query: The search query
            save_to_memory: Whether to save the result to memory
            search_options: Additional options for the search
            
        Returns:
            Dictionary with search results
        """
        result = self.api.search(query, search_options=search_options)
        
        # Format the response
        response = {
            "query": result.query,
            "answer": result.answer,
            "sources": result.sources,
            "timestamp": result.timestamp,
            "metadata": result.metadata
        }
        
        # Save to memory if requested and memory system is available
        if save_to_memory and self.memory_system:
            memory_key = f"search:{query}"
            self.memory_system.store(
                key=memory_key,
                value={
                    "query": query,
                    "answer": result.answer,
                    "sources": result.sources,
                    "timestamp": datetime.fromtimestamp(result.timestamp).strftime('%Y-%m-%d %H:%M:%S')
                }
            )
            response["memory_key"] = memory_key
        
        return response
    
    def research_topic(
        self, 
        topic: str, 
        depth: int = 2,
        save_to_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Perform deeper research on a topic with follow-up questions.
        
        Args:
            topic: The main topic to research
            depth: Depth of research (number of follow-up questions)
            save_to_memory: Whether to save results to memory
            
        Returns:
            Dictionary with research results
        """
        # Initial search
        initial_result = self.api.search(topic)
        
        results = [{
            "query": topic,
            "answer": initial_result.answer,
            "sources": initial_result.sources
        }]
        
        # Generate follow-up questions based on the initial result
        followup_query = f"Based on this information about '{topic}', what are {min(depth, 5)} specific follow-up questions that would deepen understanding of key aspects or fill knowledge gaps?"
        followup_result = self.api.search(followup_query)
        
        # Extract follow-up questions from the result
        followup_questions = []
        for line in followup_result.answer.split("\n"):
            line = line.strip()
            if line and (line.startswith("1.") or line.startswith("2.") or line.startswith("3.") or
                        line.startswith("4.") or line.startswith("5.") or 
                        line.startswith("- ") or line.startswith("• ")):
                question = line.lstrip("123456789.-• ").strip()
                if question.endswith("?"):
                    followup_questions.append(question)
        
        # Limit to the requested depth
        followup_questions = followup_questions[:depth]
        
        # Search for answers to follow-up questions
        for question in followup_questions:
            followup_answer = self.api.search(question)
            results.append({
                "query": question,
                "answer": followup_answer.answer,
                "sources": followup_answer.sources
            })
        
        # Compile the research results
        research_summary = {
            "main_topic": topic,
            "main_result": results[0],
            "followup_questions": followup_questions,
            "followup_results": results[1:],
            "timestamp": time.time()
        }
        
        # Save to memory if requested
        if save_to_memory and self.memory_system:
            memory_key = f"research:{topic}"
            self.memory_system.store(
                key=memory_key,
                value=research_summary
            )
            research_summary["memory_key"] = memory_key
        
        return research_summary
    
    def verify_fact(self, fact: str) -> Dict[str, Any]:
        """
        Verify if a statement is factually correct.
        
        Args:
            fact: The statement to verify
            
        Returns:
            Dictionary with verification results
        """
        verification_query = f"Is the following statement factually correct? '{fact}' Please verify this statement with current information and provide a clear assessment of its accuracy."
        
        result = self.api.search(verification_query)
        
        # Attempt to determine if the response indicates the fact is true or false
        answer_lower = result.answer.lower()
        
        # Simple heuristic to determine the verification result
        is_true = False
        is_false = False
        
        # Check for true/false indicators
        true_indicators = ["is correct", "is accurate", "is true", "factually accurate", "factually correct"]
        false_indicators = ["is incorrect", "is inaccurate", "is false", "not accurate", "not correct", "factually incorrect"]
        
        for indicator in true_indicators:
            if indicator in answer_lower:
                is_true = True
                break
                
        for indicator in false_indicators:
            if indicator in answer_lower:
                is_false = True
                break
        
        # Determine verification status
        if is_true and not is_false:
            verification_status = "verified"
        elif is_false and not is_true:
            verification_status = "refuted"
        else:
            verification_status = "uncertain"
        
        return {
            "fact": fact,
            "verification_status": verification_status,
            "explanation": result.answer,
            "sources": result.sources
        } 