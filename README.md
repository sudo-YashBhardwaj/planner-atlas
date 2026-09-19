# Planner Atlas

Planner Atlas studies how planning and optimization expose consequential errors in latent world models, and how those failures can be diagnosed and repaired.

The project is under active development.

## Setup

```bash
uv sync
uv run pre-commit install
```

## Validation

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
