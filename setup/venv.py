"""
Virtual environment utilities for the Cognitive Engine.
Contains functions for creating and managing Python virtual environments.
"""

import os
import sys
import subprocess
import platform
import venv
from pathlib import Path


def create_venv(venv_path=None):
    """
    Create a Python virtual environment at the specified path.
    If no path is specified, creates it at 'venv' in the project root.
    
    Returns:
        str: The path to the created virtual environment
    """
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Set default venv path if not specified
    if venv_path is None:
        venv_path = os.path.join(project_root, 'venv')
    
    # Only create the venv if it doesn't exist
    if not os.path.exists(venv_path):
        print(f"Creating virtual environment at {venv_path}...")
        venv.create(venv_path, with_pip=True)
        print(f"✅ Virtual environment created at {venv_path}")
    else:
        print(f"Virtual environment already exists at {venv_path}")
    
    return venv_path


def get_venv_python(venv_path):
    """
    Get the path to the Python executable in the virtual environment.
    
    Args:
        venv_path (str): Path to the virtual environment
        
    Returns:
        str: Path to the Python executable
    """
    if platform.system() == 'Windows':
        python_path = os.path.join(venv_path, 'Scripts', 'python.exe')
    else:
        python_path = os.path.join(venv_path, 'bin', 'python')
    
    return python_path


def activate_venv(venv_path=None):
    """
    Activate the Python virtual environment for the current process.
    If no path is specified, activates the one at 'venv' in the project root.
    
    This modifies sys.path and other environment variables to use the 
    virtual environment's Python and packages.
    
    Returns:
        bool: True if activation was successful, False otherwise
    """
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Set default venv path if not specified
    if venv_path is None:
        venv_path = os.path.join(project_root, 'venv')
    
    if not os.path.exists(venv_path):
        print(f"❌ Virtual environment not found at {venv_path}")
        return False
    
    # Get Python path in the virtual environment
    python_path = get_venv_python(venv_path)
    
    if not os.path.exists(python_path):
        print(f"❌ Python executable not found at {python_path}")
        return False
    
    # Activate the virtual environment
    # This is similar to what bin/activate does but for the current Python process
    
    # First, get site-packages directory
    if platform.system() == 'Windows':
        site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
    else:
        # Get Python version in the venv
        try:
            venv_python_version = subprocess.check_output(
                [python_path, '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'],
                text=True
            ).strip()
            site_packages = os.path.join(venv_path, 'lib', f'python{venv_python_version}', 'site-packages')
        except subprocess.CalledProcessError:
            # Fallback to a common pattern if we can't determine the version
            possible_site_packages = [
                p for p in [os.path.join(venv_path, 'lib', d) for d in os.listdir(os.path.join(venv_path, 'lib'))]
                if os.path.isdir(p) and p.startswith(os.path.join(venv_path, 'lib', 'python'))
            ]
            
            if possible_site_packages:
                site_packages = os.path.join(possible_site_packages[0], 'site-packages')
            else:
                print(f"❌ Could not find site-packages in {venv_path}")
                return False
    
    # Check if site-packages exists
    if not os.path.exists(site_packages):
        print(f"❌ Site-packages not found at {site_packages}")
        return False
    
    # Modify sys.path to prioritize the virtual environment
    sys.path.insert(0, site_packages)
    
    # Update environment variables
    os.environ['VIRTUAL_ENV'] = venv_path
    
    # Update PATH to prioritize virtual environment's binaries
    if platform.system() == 'Windows':
        bin_dir = os.path.join(venv_path, 'Scripts')
    else:
        bin_dir = os.path.join(venv_path, 'bin')
    
    os.environ['PATH'] = os.pathsep.join([bin_dir, os.environ.get('PATH', '')])
    
    print(f"✅ Activated virtual environment at {venv_path}")
    print(f"Python executable: {python_path}")
    return True


def ensure_venv(venv_path=None):
    """
    Ensure a virtual environment exists and is activated.
    If it doesn't exist, creates it.
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Set default venv path if not specified
    if venv_path is None:
        venv_path = os.path.join(project_root, 'venv')
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists(venv_path):
        venv_path = create_venv(venv_path)
    
    # Activate the virtual environment
    return activate_venv(venv_path)


def install_requirements(venv_path=None, requirements_file=None):
    """
    Install required packages in the virtual environment.
    
    Args:
        venv_path (str, optional): Path to the virtual environment
        requirements_file (str, optional): Path to the requirements file
        
    Returns:
        bool: True if installation was successful, False otherwise
    """
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Set default venv path if not specified
    if venv_path is None:
        venv_path = os.path.join(project_root, 'venv')
    
    # Set default requirements file if not specified
    if requirements_file is None:
        requirements_file = os.path.join(project_root, 'requirements.txt')
    
    # Create requirements file if it doesn't exist
    if not os.path.exists(requirements_file):
        # Create a default requirements file with basic neural network dependencies
        with open(requirements_file, 'w') as f:
            f.write('numpy>=1.19.0\n')
            f.write('matplotlib>=3.3.0\n')
            f.write('torch>=1.7.0\n')  # Optional but recommended
            f.write('pillow>=7.0.0\n')
        print(f"✅ Created requirements file at {requirements_file}")
    
    # Get Python path in the virtual environment
    python_path = get_venv_python(venv_path)
    
    if not os.path.exists(python_path):
        print(f"❌ Python executable not found at {python_path}")
        return False
    
    # Install requirements
    try:
        print(f"Installing requirements from {requirements_file}...")
        subprocess.run([python_path, '-m', 'pip', 'install', '-r', requirements_file], check=True)
        print(f"✅ Successfully installed requirements")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {str(e)}")
        return False


if __name__ == "__main__":
    # Test virtual environment setup
    venv_path = create_venv()
    activate_venv(venv_path)
    install_requirements(venv_path)
    print("Virtual environment setup completed.") 