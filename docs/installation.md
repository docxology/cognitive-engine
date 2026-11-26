# Installation

The Python package is defined by [setup.py](../setup.py) and lists its
dependencies in [requirements.txt](../requirements.txt). Its package
metadata requires Python 3.11 or newer. The TypeScript fixture scripts are
declared separately in [package.json](../package.json).

The standalone symbolic check needs Python and NetworkX. From the
repository root, use uv to run the checked-in implementation against
fixed inputs:

```bash
uv run --no-project --python 3.12 --with 'networkx==3.7' \
  python -m unittest discover -s tests -p test_symbolic_offline.py -v
```

The test loads the fixed source files for the symbolic modules directly.
The package initializer imports the broader hybrid stack, including model
and service adapters. The focused test uses only the symbolic files and
does not initialize the composition layer.

See the [quick start](quickstart.md) to interpret the results.
