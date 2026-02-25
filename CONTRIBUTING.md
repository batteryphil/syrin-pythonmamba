# Contributing to Syrin

We welcome contributions. Here’s how to get started.

## Development setup

```bash
git clone https://github.com/syrin-labs/syrin-python.git
cd syrin-python
uv sync
```

## Code quality

- **Type checking:** `mypy --strict` must pass on `src/syrin/`.
- **Linting:** `ruff check` and `ruff format` (see `pyproject.toml`).
- **Tests:** `pytest` with `pytest-asyncio` for async tests.

## Submitting changes

1. Open an issue or discussion for larger changes.
2. Fork the repo, create a branch, make your changes.
3. Ensure tests and type/lint checks pass.
4. Open a pull request against `main`.

## Questions

- [GitHub Discussions](https://github.com/syrin-labs/syrin-python/discussions)
- [Discord](https://discord.gg/p4jnKxYKpB)

For full API and design rules, see `docs/ARCHITECTURE.md` and the repository docs.
