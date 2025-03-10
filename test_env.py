#!/usr/bin/env python3
"""
Test environment script for neural network dependencies
"""

import sys
import os

def check_dependency(module_name):
    """Check if a module is installed"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def main():
    """Main function"""
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current directory: {os.getcwd()}")
    
    # Check required dependencies
    dependencies = [
        "torch", 
        "numpy", 
        "matplotlib", 
        "transformers", 
        "json", 
        "pathlib"
    ]
    
    print("\nChecking dependencies:")
    for dep in dependencies:
        if check_dependency(dep):
            print(f"✅ {dep} is installed")
        else:
            print(f"❌ {dep} is NOT installed")
    
    # Print system info
    print("\nSystem information:")
    print(f"Platform: {sys.platform}")
    print(f"Path: {sys.path}")

if __name__ == "__main__":
    main() 