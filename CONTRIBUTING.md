# Contributing to Adaptive Chunking

Thank you for your interest in contributing! This project follows the
[Ekimetrics Open Source Policy](https://www.ekimetrics.com/).

## Getting started

```bash
git clone https://github.com/ekimetrics/adaptive-chunking.git
cd adaptive-chunking
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
pytest
```

## Submitting changes

1. Fork the repository and create a feature branch from `main`.
2. Write or update tests for your changes.
3. Run `pytest` and make sure all tests pass.
4. Open a pull request with a clear description of what you changed and why.

## Code style

- Follow existing patterns in the codebase.
- Keep imports lazy for heavy dependencies (torch, transformers, maverick, etc.).

## Licensing

By submitting a contribution you agree that your work will be licensed
under the [MIT License](LICENSE) that covers this project.

If your contribution adds or updates a dependency, verify that its license
is compatible with MIT (see the [NOTICE](NOTICE) file for the current list).
Dependencies under copyleft licenses (GPL, AGPL, CC BY-NC-SA, etc.) must
be placed behind an optional extra — never in core `dependencies`.

## Reporting issues

Open an issue on GitHub. Include steps to reproduce, expected behaviour,
and the output you actually see.
