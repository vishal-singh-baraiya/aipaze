# Contributing to AIPaze

Thank you for your interest in contributing to AIPaze! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Contribution Workflow](#contribution-workflow)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)
- [Community](#community)

## Code of Conduct

Our community is dedicated to providing a harassment-free experience for everyone. We do not tolerate harassment of participants in any form. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip
- git

### Fork and Clone

1. Fork the AIPaze repository on GitHub
2. Clone your fork locally:

```bash
   git clone https://github.com/your-username/aipaze.git
   cd aipaze
```

3. Add the original repository as an upstream remote:

```bash
   git remote add upstream https://github.com/original-owner/aipaze.git
```

## Development Environment

### Setting Up

1. Create a virtual environment:

```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:

```bash
   pip install -e ".[dev]"
```

3. Install pre-commit hooks:

```bash
   pre-commit install
```

## Contribution Workflow

1. Ensure your fork is up to date:

```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
```

2. Create a new branch for your feature or bug fix:

```bash
   git checkout -b feature/your-feature-name
```

   or

```bash
   git checkout -b fix/issue-you-are-fixing
```

3. Make your changes and commit them with clear, descriptive commit messages:

```bash
   git add .
   git commit -m "Add feature: description of the feature"
```

4. Push your branch to your fork:

```bash
   git push origin feature/your-feature-name
```

5. Create a pull request from your branch to the main AIPaze repository.

## Pull Request Guidelines

- Fill in the pull request template completely
- Include tests for new features or bug fixes
- Update documentation as needed
- Ensure all tests pass
- Keep pull requests focused on a single feature or bug fix
- Reference any related issues

## Coding Standards

We follow these coding standards:

- Use [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style
- Use type hints wherever possible
- Write docstrings in [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Keep functions small and focused on a single task
- Use descriptive variable and function names

We use the following tools to enforce code quality:

- flake8 for linting
- mypy for type checking
- black for code formatting
- isort for import sorting

## Testing

- Write unit tests for all new code
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage

Run tests with:

```bash
pytest
```

## Documentation

- Update documentation for any new features or changes
- Document all public functions, classes, and methods
- Include examples where appropriate
- Keep README and other documentation up to date

## Issue Reporting

When reporting issues, please include:

1. A clear and descriptive title
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Environment information (OS, Python version, etc.)
6. Any relevant logs or error messages

## Feature Requests

We welcome feature requests! When submitting a feature request:

1. Check if the feature has already been requested
2. Provide a clear description of the feature
3. Explain why the feature would be useful
4. Suggest implementation details if possible

## Community

Join our community:

- [Twitter](https://twitter.com/TheVixhal)

## Acknowledgments

Your contributions to AIPaze are greatly appreciated. By contributing, you help make the library better for everyone. Thank you for your time and effort!

## License

By contributing to AIPaze, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).