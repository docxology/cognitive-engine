"""
Environment setup utilities for the Cognitive Engine.
Contains functions for setting up the Python environment.
"""

import os
import sys
import importlib
import platform
import subprocess
from pathlib import Path


def fix_pythonpath():
    """
    Fix PYTHONPATH issues by ensuring the current directory is in sys.path
    and removing problematic paths.
    """
    # Get current directory
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Add current directory to sys.path if not already there
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Remove problematic paths (like Cursor's mount paths)
    sys.path = [p for p in sys.path if '/tmp/.mount_Cursor' not in p]
    
    # Reset PYTHONHOME and PYTHONPATH environment variables
    if 'PYTHONHOME' in os.environ:
        del os.environ['PYTHONHOME']
    
    os.environ['PYTHONPATH'] = current_dir
    
    # Return the current directory for reference
    return current_dir


def setup_environment():
    """
    Set up the Python environment for running neural tests.
    Ensures all necessary paths and environment variables are set correctly.
    """
    try:
        # Fix Python path
        project_dir = fix_pythonpath()
        
        # Print environment information
        print(f"Python version: {platform.python_version()}")
        print(f"Python executable: {sys.executable}")
        print(f"Python path: {sys.path}")
        print(f"Current directory: {project_dir}")
        
        # Try to import critical modules
        for module_name in ['encodings', 'os', 'sys', 'json', 'time', 'random']:
            try:
                importlib.import_module(module_name)
                print(f"✅ Successfully imported {module_name}")
            except ImportError as e:
                print(f"❌ Failed to import {module_name}: {str(e)}")
                if module_name == 'encodings':
                    print("Critical module 'encodings' is missing. This will cause issues.")
                    return False
        
        # Create necessary directories
        dirs = [
            "tests/data", 
            "tests/results", 
            "models", 
            "models/recovery", 
            "models/backups", 
            "visualizations"
        ]
        
        for d in dirs:
            os.makedirs(os.path.join(project_dir, d), exist_ok=True)
            print(f"✅ Created directory: {d}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error setting up environment: {str(e)}")
        return False


def check_dependencies():
    """
    Check if all required dependencies are installed.
    Returns a tuple of (success, missing_packages).
    """
    required_packages = [
        'numpy',
        'matplotlib',
        'torch',  # Optional
        'pillow'
    ]
    
    missing_packages = []
    optional_missing = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ Package {package} is installed")
        except ImportError:
            if package == 'torch':
                print(f"⚠️ Optional package {package} is not installed")
                optional_missing.append(package)
            else:
                print(f"❌ Required package {package} is not installed")
                missing_packages.append(package)
    
    success = len(missing_packages) == 0
    return success, missing_packages, optional_missing


def install_package(package_name):
    """
    Install a Python package using pip.
    """
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True)
        print(f"✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {str(e)}")
        return False


if __name__ == "__main__":
    # Test environment setup
    setup_environment()
    success, missing, optional = check_dependencies()
    
    if not success:
        print(f"Missing packages: {', '.join(missing)}")
        print("Installing missing packages...")
        for package in missing:
            install_package(package)
    
    print("Environment setup completed.") 