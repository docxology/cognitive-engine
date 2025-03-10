#!/usr/bin/env python3
"""
Enhanced runner script for neural network tests.
Uses the setup modules to ensure proper environment configuration.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add parent directory to sys.path so that we can import setup modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from setup.environment import setup_environment, check_dependencies, fix_pythonpath
from setup.venv import ensure_venv, install_requirements


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run neural network tests with proper environment setup")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--fallback", action="store_true", help="Force NumPy fallback mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--config", type=str, default="tests/data/neural_config.json", help="Path to config file")
    parser.add_argument("--retrain", action="store_true", help="Force model retraining")
    parser.add_argument("--export-vis", type=str, help="Export visualizations to the specified directory")
    parser.add_argument("--use-venv", action="store_true", help="Use a virtual environment")
    
    return parser.parse_args()


def run_tests_in_venv(args):
    """Run the tests using a virtual environment"""
    print("Setting up a virtual environment for running tests...")
    
    # Ensure we have a virtual environment and that it's activated
    if not ensure_venv():
        print("❌ Failed to set up virtual environment")
        return 1
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return 1
    
    # Path to the Python executable in the virtual environment
    venv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'venv')
    if sys.platform == 'win32':
        python_exe = os.path.join(venv_path, 'Scripts', 'python.exe')
    else:
        python_exe = os.path.join(venv_path, 'bin', 'python')
    
    # Run the tests
    cmd = [python_exe, "tests/run_neural_tests.py"]
    
    # Add arguments
    if args.visualize:
        cmd.append("--visualize")
    if args.fallback:
        cmd.append("--fallback")
    if args.verbose:
        cmd.append("--verbose")
    if args.config:
        cmd.extend(["--config", args.config])
    if args.retrain:
        cmd.append("--retrain")
    if args.export_vis:
        cmd.extend(["--export-vis", args.export_vis])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True)
        print(f"Tests completed with exit code {result.returncode}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code {e.returncode}")
        return e.returncode


def run_tests_directly(args):
    """Run the tests directly without a virtual environment"""
    print("Running tests with the current Python interpreter...")
    
    # Fix Python path
    fix_pythonpath()
    
    # Set up environment
    if not setup_environment():
        print("❌ Failed to set up environment")
        return 1
    
    # Check dependencies
    success, missing_packages, optional_missing = check_dependencies()
    if not success:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using pip and try again.")
        return 1
    
    # Run the tests
    cmd = [sys.executable, "tests/run_neural_tests.py"]
    
    # Add arguments
    if args.visualize:
        cmd.append("--visualize")
    if args.fallback:
        cmd.append("--fallback")
    if args.verbose:
        cmd.append("--verbose")
    if args.config:
        cmd.extend(["--config", args.config])
    if args.retrain:
        cmd.append("--retrain")
    if args.export_vis:
        cmd.extend(["--export-vis", args.export_vis])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True)
        print(f"Tests completed with exit code {result.returncode}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code {e.returncode}")
        return e.returncode


def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_args()
    
    # Change to the project root directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Print environment information
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    # Run tests
    if args.use_venv:
        return run_tests_in_venv(args)
    else:
        return run_tests_directly(args)


if __name__ == "__main__":
    sys.exit(main()) 