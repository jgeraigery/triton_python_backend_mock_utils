# Contributing Guide

Thank you for helping improve `triton_python_backend_mock_utils`! This project mocks the Triton Python backend utilities so Triton models can be developed and tested outside the Triton Server container. Contributions that increase feature coverage, improve documentation, or expand tests are always welcome.

## Getting Started

1. Fork the repository and clone your fork.
2. Install Python 3.12 and [Poetry](https://python-poetry.org/docs/#installation).
3. Install dependencies:

   ```bash
   poetry install
   ```

4. Activate the virtual environment when developing locally:

   ```bash
   poetry shell
   ```

   You can also run commands without activating the shell via `poetry run <command>`.

## Development Workflow

- Create a feature branch from `main` before making changes.
- Keep your branch focused on a single change so reviews stay lightweight.
- Add or update tests under `tests/` whenever you change behaviour.
- Update documentation (including examples) if user-facing behaviour changes.

### Formatting and Static Checks

The project relies on Black, isort, and other tooling managed by pre-commit hooks.

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

Running the hooks locally prevents CI failures and keeps style consistent.

### Running Tests

Use pytest for the unit test suite. Coverage is optional but encouraged for larger changes.

```bash
poetry run pytest
# Optional coverage report
poetry run coverage run -m pytest
poetry run coverage report
```

## Commit Standards (Required)

This repository uses semantic-release, so **semantic (Conventional) commit messages are required**. Every commit must follow the format:

```
type: short description
```

- `type` examples: `feat`, `fix`, `docs`, `chore`, `test`, `refactor`, `perf`, `ci`.
- Use the imperative mood in the description ("add support for…").
- Commits that break backwards compatibility must include `[BREAKING CHANGE]` in the commit footer.

Commit messages are used to generate changelogs and determine version bumps, so please keep them clear and accurate.
Rebasing and squashing commits before merging is encouraged to maintain a clean history.

## Pull Request Checklist

Before requesting a review:

- Rebase on the latest `main` and resolve conflicts locally.
- Ensure `poetry run pytest` passes.
- Run `poetry run pre-commit run --all-files` and fix any issues it reports.
- Confirm documentation or examples reflect your changes.
- Provide context in the PR description, including rationale and testing notes.

## Reporting Issues

If you find a bug or want to request new functionality:

- Search existing issues to avoid duplicates.
- Include reproduction steps, expected vs. actual behaviour, and environment details.
- For feature requests, describe the problem you are trying to solve and any alternatives considered.

We appreciate your time and contributions. Thank you for helping make Triton model development smoother for everyone!
