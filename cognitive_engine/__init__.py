"""
cognitive-engine: A hybrid Neuro-Symbolic AI system.
"""

__version__ = '0.1.0'

from cognitive_engine import fractal
from cognitive_engine import probabilistic
from cognitive_engine import memory
from cognitive_engine.unipixel import unipixel
from cognitive_engine import mmm
from cognitive_engine import perplexity_api
from cognitive_engine import code_execution
from cognitive_engine import peff

# Import the main engine class for easier access
from cognitive_engine.engine import HybridCognitiveEngine

# Import key classes from the new modules for easier access
from cognitive_engine.unipixel.unipixel import Unipixel, UnipixelRegistry
from cognitive_engine.mmm import MagicalMathModel, CognitivePattern
from cognitive_engine.perplexity_api import PerplexityAPI, PerplexitySearchEngine
from cognitive_engine.code_execution import CodeExecutor
from cognitive_engine.peff import ParadiseEnergyFractalForce

__all__ = [
    'fractal',
    'probabilistic', 
    'memory',
    'unipixel',
    'mmm',
    'perplexity_api',
    'code_execution',
    'peff',
    'HybridCognitiveEngine',
    'Unipixel',
    'UnipixelRegistry',
    'MagicalMathModel',
    'CognitivePattern',
    'PerplexityAPI',
    'PerplexitySearchEngine',
    'CodeExecutor',
    'ParadiseEnergyFractalForce'
] 