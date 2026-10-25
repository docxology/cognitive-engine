# Installation Guide

This guide provides detailed instructions for installing the Cognitive Engine on various platforms and with different configurations.

## System Requirements

### Minimum Requirements

- **Python**: 3.10 or higher
- **RAM**: 8GB (16GB recommended for larger models)
- **Disk Space**: 1GB for the basic installation
- **Operating System**: Windows 10+, macOS 10.15+, Ubuntu 20.04+ or other Linux distributions

### Recommended Specifications

- **Python**: 3.12 or higher
- **RAM**: 16GB or more
- **Disk Space**: 5GB+ (for storing memory and models)
- **GPU**: NVIDIA GPU with CUDA support (for accelerated neural processing)
- **CPU**: 4+ cores

## Installation Methods

There are several ways to install the Cognitive Engine depending on your needs and setup.

### Method 1: Basic Installation

This is the simplest installation method, suitable for most users.

```bash
# Clone the repository
git clone https://github.com/docxology/cognitive-engine.git
cd cognitive-engine

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Method 2: Installation with Conda

If you use Anaconda or Miniconda for Python environment management:

```bash
# Clone the repository
git clone https://github.com/docxology/cognitive-engine.git
cd cognitive-engine

# Create a conda environment
conda create -n cognitive-engine python=3.12
conda activate cognitive-engine

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Method 3: Installation with GPU Support

For installations that will leverage GPU acceleration:

```bash
# Clone the repository
git clone https://github.com/docxology/cognitive-engine.git
cd cognitive-engine

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or appropriate activation command for your OS

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Platform-Specific Instructions

### Windows

1. Install Python 3.10+ from the [official website](https://www.python.org/downloads/windows/)
2. Install Git from the [official website](https://git-scm.com/download/win)
3. Open Command Prompt or PowerShell
4. Follow the Basic Installation steps above

Additional Windows-specific notes:
- You may need to run Command Prompt or PowerShell as Administrator for some operations
- If you encounter issues with C++ build tools during installation, install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### macOS

1. Install Python 3.10+ (use Homebrew: `brew install python@3.12`)
2. Install Git (use Homebrew: `brew install git`)
3. Open Terminal
4. Follow the Basic Installation steps above

Additional macOS-specific notes:
- If you encounter SSL certificate issues, you may need to run: `pip install --upgrade certifi`
- For Apple Silicon Macs, ensure you're using Python and packages compiled for ARM architecture

### Linux

1. Install Python 3.10+ using your distribution's package manager
   - Ubuntu/Debian: `sudo apt-get install python3.10 python3.10-venv python3-pip`
   - Fedora: `sudo dnf install python3.10 python3-pip`
2. Install Git using your distribution's package manager
   - Ubuntu/Debian: `sudo apt-get install git`
   - Fedora: `sudo dnf install git`
3. Open Terminal
4. Follow the Basic Installation steps above

## Configuration

After installation, you'll need to configure some aspects of the Cognitive Engine.

### API Keys

Credentials for external services are passed through the engine configuration, or picked up
from the environment by the component that needs them:

```python
import os

from cognitive_engine import HybridCognitiveEngine

# Perplexity credentials for web search; the search engine also falls back to
# the PERPLEXITY_API_KEY environment variable when no key is passed.
engine = HybridCognitiveEngine(
    config={
        "perplexity_api_key": os.environ.get("PERPLEXITY_API_KEY"),
        "perplexity_cache_dir": "./.perplexity_cache",
    }
)
```

The LLM interface reads `OPENAI_API_KEY` from the environment when no explicit key is given,
so exporting the keys in your shell is enough:

```bash
export PERPLEXITY_API_KEY=your-perplexity-api-key
export OPENAI_API_KEY=your-openai-api-key
```

There is no automatic `.env` loader; export the variables (or load the file in your shell)
before starting the engine.

### Memory Storage

Configure the memory storage location:

```python
from cognitive_engine.memory import PervasiveMemory

# Create memory with a custom persistence path
memory = PervasiveMemory(storage_config={"persistence_path": "/path/to/your/memory/store"})
```

### GPU Configuration

If you have a compatible GPU, check that PyTorch can see it before building torch-backed
components:

```python
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Torch-backed components use the CUDA device automatically when it is available
from cognitive_engine import HybridCognitiveEngine

engine = HybridCognitiveEngine()
```

## Verification

To verify that your installation is working correctly:

```python
from cognitive_engine import HybridCognitiveEngine

# Create a simple engine
engine = HybridCognitiveEngine()

# Try a simple query
result = engine.process("Hello, Cognitive Engine!")
print(result['neural_outputs'])

# Check if components are loaded correctly
print(f"Fractal System Levels: {engine.symbolic.levels}")
print(f"Neural System Type: {type(engine.neural).__name__}")
print(f"Memory System Type: {type(engine.memory).__name__}")
```

## Troubleshooting

### Common Installation Issues

#### Issue: Missing Dependencies

```
ERROR: Failed building wheel for [package name]
```

**Solution**: Install the required build tools

- Windows: `pip install --upgrade setuptools wheel`
- macOS/Linux: `sudo apt-get install build-essential` or equivalent

#### Issue: CUDA Not Found

```
ImportError: CUDA extension not found
```

**Solution**: Ensure you have the correct CUDA toolkit installed matching your PyTorch version

#### Issue: Out of Memory Errors

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch sizes or model sizes, or use CPU mode instead

#### Issue: Permission Errors

```
PermissionError: [Errno 13] Permission denied
```

**Solution**: Run with appropriate permissions or change the storage path to a location you have write access to

### Advanced Troubleshooting

For more complex issues, you can dump the engine state to check which components loaded:

```bash
python -c "from cognitive_engine import HybridCognitiveEngine; print(HybridCognitiveEngine().to_json())"
```

## Updating

To update to the latest version:

```bash
# Navigate to your installation directory
cd path/to/cognitive-engine

# Pull the latest changes
git pull

# Update dependencies
pip install -r requirements.txt

# Reinstall the package
pip install -e .
```

## Uninstallation

To remove the Cognitive Engine:

```bash
# Deactivate the virtual environment if active
deactivate  # or conda deactivate

# Remove the package
pip uninstall cognitive-engine

# Optionally, remove the entire directory
rm -rf path/to/cognitive-engine
```

## Next Steps

Now that you've installed the Cognitive Engine, you can:

1. Follow the [Quick Start Guide](quickstart.md) to create your first application
2. Explore the [Architecture Documentation](architecture.md) to understand the system
3. Check out the [Examples](examples/index.md) for inspiration
