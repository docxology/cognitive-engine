"""
Setup package for the Cognitive Engine.
Contains utilities for setting up Python environments and dependencies.
"""

from setup.environment import setup_environment, check_dependencies, fix_pythonpath
from setup.venv import create_venv, activate_venv, ensure_venv, install_requirements

__all__ = [
    'setup_environment',
    'check_dependencies',
    'fix_pythonpath',
    'create_venv',
    'activate_venv',
    'ensure_venv',
    'install_requirements'
] 