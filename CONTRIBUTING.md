# Contributing to AEGIS

Thank you for your interest in contributing to **AEGIS**. This document provides guidelines for contributing in a research-grade, professional manner.

## Code of Conduct

By participating, you agree to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

- **Bug reports and feature requests**: Open an [issue](https://github.com/ansulx/AEGIS/issues) with a clear title and description.
- **Code contributions**: Fork the repo, create a branch, make your changes, and open a [pull request](https://github.com/ansulx/AEGIS/pulls).

## Development Setup

```bash
git clone https://github.com/ansulx/AEGIS.git
cd AEGIS
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Code Style

- Use [Black](https://github.com/psf/black) for formatting: `black src tests`
- Use [Ruff](https://github.com/astral-sh/ruff) for linting: `ruff check src tests`
- Run tests: `pytest`

## Pull Request Guidelines

- Keep PRs focused and reasonably sized.
- Include a short description and, if applicable, link to related issues.
- Ensure tests pass and new code is covered where appropriate.

## Research and Citation

If your contribution is part of or supports published work, please add appropriate references in documentation or docstrings.

Thank you for helping make clinical AI safer.
