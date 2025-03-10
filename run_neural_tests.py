#!/usr/bin/env python3
"""
Main script for running neural network tests.
"""

import os
import sys
from pathlib import Path

def main():
    """Main entry point for running neural tests"""
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    
    # Clear problematic environment variables in Cursor
    if os.environ.get('PYTHONHOME') and '.mount_Cursor' in os.environ['PYTHONHOME']:
        del os.environ['PYTHONHOME']
    
    # Set Python path to just our project root
    os.environ['PYTHONPATH'] = str(project_root)
    
    # Ensure output directories exist
    Path('visualizations').mkdir(exist_ok=True)
    Path('tests/results').mkdir(exist_ok=True)
    
    # Run the tests
    test_script = project_root / 'tests' / 'run_neural_tests.py'
    if not test_script.exists():
        print(f"❌ Test script not found at {test_script}")
        return 1
        
    # Import and run the tests directly
    sys.path.insert(0, str(project_root))
    from tests.run_neural_tests import run_tests
    
    try:
        results = run_tests()
        if '--visualize' in sys.argv:
            print("\n" + "="*78)
            print("VISUALIZATIONS")
            print(f"Neural network visualizations are available at:")
            print(f"{project_root}/visualizations")
            
            html_report = project_root / 'visualizations' / 'visualization_report.html'
            if html_report.exists():
                print(f"\nVisualization HTML report:")
                print(f"{html_report}")
            print("="*78 + "\n")
        return 0 if results else 1
    except Exception as e:
        print(f"❌ Error running neural tests: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 