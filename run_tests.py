#!/usr/bin/env python3
"""
Wrapper script for running neural network tests with a proper environment.
This script ensures that the Python environment is correctly set up before running the tests.
"""

import os
import sys
import subprocess

def main():
    # Reset Python path to ensure we're using system modules
    os.environ["PYTHONPATH"] = ""
    
    # Get absolute path to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up the command to run
    cmd = [sys.executable, "tests/run_neural_tests.py", "--visualize"]
    
    # Add any additional arguments passed to this script
    cmd.extend(sys.argv[1:])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print(f"Tests completed with exit code {result.returncode}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code {e.returncode}")
        return e.returncode

if __name__ == "__main__":
    sys.exit(main()) 