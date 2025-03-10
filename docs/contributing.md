# Contributing to Cognitive Engine

Thank you for your interest in contributing to the Cognitive Engine project! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please read and follow our [Code of Conduct](code_of_conduct.md) to help us maintain a positive and inclusive community.

## How to Contribute

There are many ways to contribute to Cognitive Engine:

1. **Code Contributions**: Implement new features or fix bugs
2. **Documentation**: Improve or expand documentation
3. **Testing**: Add or improve tests
4. **Bug Reports**: Report bugs or issues
5. **Feature Requests**: Suggest new features or improvements
6. **Community Support**: Help other users in discussions

## Development Workflow

### Setting Up Your Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/your-username/cognitive-engine.git
cd cognitive-engine

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .
```

### Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the coding standards

3. Write or update tests for your changes

4. Run the tests to ensure everything passes:
   ```bash
   pytest
   ```

5. Update documentation as needed

### Submitting Changes

1. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "Add feature X to improve Y capability"
   ```

2. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Submit a Pull Request (PR) to the main repository

4. Respond to any code review feedback

## Pull Request Guidelines

- Keep PRs focused on a single issue or feature
- Include tests for new functionality
- Update documentation as needed
- Follow the coding style guidelines
- Make sure all tests pass
- Reference any related issues

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use type hints where appropriate (following [PEP 484](https://www.python.org/dev/peps/pep-0484/))
- Write docstrings for all public methods, functions, and classes (following [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings))
- Keep lines to a maximum of 100 characters

### Testing

- Write unit tests for all new functionality
- Maintain or improve code coverage
- Tests should be clear and easy to understand

## Documentation

- Keep documentation up-to-date with code changes
- Follow the existing documentation style
- Provide examples where applicable

## Issue Reporting

When reporting issues, please include:

1. A clear, descriptive title
2. A detailed description of the issue
3. Steps to reproduce the problem
4. Expected behavior
5. Actual behavior
6. System information (OS, Python version, etc.)
7. Any relevant logs or screenshots

## Feature Requests

When suggesting features:

1. Describe the problem you're trying to solve
2. Explain how your proposed feature solves it
3. Provide examples of how the feature would be used
4. Consider potential implementation approaches

## Community

Join our community channels to get help or discuss the project:

- [GitHub Discussions](https://github.com/yourusername/cognitive-engine/discussions)
- [Discord Server](https://discord.gg/your-discord-link)

## License

By contributing to Cognitive Engine, you agree that your contributions will be licensed under the project's [MIT License](../LICENSE). 