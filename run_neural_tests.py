#!/usr/bin/env python3
"""
Main script for running neural network tests.
This script ensures that the Python environment is correctly set up before running the tests.
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """Main entry point for running neural tests"""
    # Get the path to the setup script
    setup_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'setup', 'run_neural_tests.py')
    
    # Check if the setup script exists
    if not os.path.exists(setup_script):
        print(f"❌ Setup script not found at {setup_script}")
        return 1
    
    # Build the command to run
    cmd = [sys.executable, setup_script]
    
    # Add any arguments passed to this script
    cmd.extend(sys.argv[1:])
    
    print(f"Running neural tests with proper environment setup...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True)
        print(f"Neural tests completed with exit code {result.returncode}")
        
        # If successful, show a message about visualizations
        if result.returncode == 0:
            vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')
            if os.path.exists(vis_dir):
                print("\n" + "="*80)
                print("VISUALIZATIONS")
                print(f"Neural network visualizations are available at:")
                print(f"{vis_dir}")
                
                # Check for HTML report
                html_report = os.path.join(vis_dir, 'visualization_report.html')
                if os.path.exists(html_report):
                    print(f"\nVisualization HTML report:")
                    print(f"{html_report}")
                
                print("="*80 + "\n")
        
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Neural tests failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"❌ Error running neural tests: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 