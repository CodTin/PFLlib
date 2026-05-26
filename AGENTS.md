# AGENTS.md

## Project Overview

PFLlib is a Personalized Federated Learning library and benchmark. The `pfllib/` package is the refactored core; the original `system/` + `dataset/` layout is the legacy interface still referenced by the README.

## Developer Commands

```bash
make install          # install base deps via uv
make dev              # install with all extras + dev tools
make format           # ruff format + ruff check --fix
make lint            # ruff check (no auto-fix)
make typecheck        # mypy pfllib/
make test             # pytest tests/ -v
make test-unit        # pytest tests/ -v -m "not integration"
make test-integration # pytest tests/ -v -m integration
make clean            # remove caches and build artifacts
```

Always run `format` -> `lint` -> `typecheck` -> `test` in that order after edits.

## Architecture

### Package layout (`pfllib/`)

- **`config.py`** — `ExperimentConfig` (Pydantic BaseModel) holds all hyperparameters. `model_config = {"extra": "allow"}`, so arbitrary fields can be attached at runtime.
- **`registry.py`** — Algorithm and model registration via decorators. Algorithms are `(server_cls, client_cls, uses_head_split)` triples.
- **`cli.py`** — Typer CLI entry point (`pfllib run`, `pfllib generate-data`, `pfllib list`). `python -m pfllib` also works.
- **`runner.py`** — `run_experiment(config)` wires config -> model build -> algorithm lookup -> server.train(). Uses `build_model()` which has dataset-specific branching.
- **`clients/`** — `clientNAME.py`, each extending `clients.base.Client`.
- **`servers/`** — `serverNAME.py`, each extending `servers.base.Server`. **`__init__.py`** performs all `register_algorithm()` calls at import time.
- **`models/`** — Neural network architectures (`cnn.py`, `resnet.py`, `lstm.py`, etc.).
- **`data/generators/`** — Dataset generation scripts (`mnist.py`, `cifar10.py`, etc.), one file per dataset.
- **`data/utils.py`** — `separate_data`, `split_data`, `save_file`, `check`, `ImageDataset`.
- **`data/reader.py`** — `read_data()`, `process_image()`, `process_shakespeare()`, `process_text()`.
- **`optimizers/fed_optimizers.py`** — Custom FL optimizers (`PerAvgOptimizer`, `SCAFFOLDOptimizer`, etc.).

### Registration pattern

Adding a new FL algorithm requires:
1. Create `pfllib/clients/clientNAME.py` and `pfllib/servers/serverNAME.py`
2. Add `register_algorithm("Name", client_cls=ClientNAME, uses_head_split=bool)` in `pfllib/servers/__init__.py`
3. Import client in `pfllib/clients/__init__.py` and server in `pfllib/servers/__init__.py`

`uses_head_split=True` means the model's `.fc` layer is split into a body/head via `BaseHeadSplit`.

### Adding a new model

Add the class to `pfllib/models/` and register it with `register_model()` in `models/__init__.py`. Also add the mapping in `runner.build_model()`.

## Key Conventions

- **Python 3.11+** required (`requires-python = ">=3.11"`)
- **Package manager**: `uv` (not pip directly). Use `uv add` / `uv run`.
- **numpy<2** pin — torch compatibility requires this.
- **Ruff** is the linter/formatter; `E501` and several `F4xx` rules are ignored (see `pyproject.toml`).
- **MyPy** runs with `ignore_missing_imports = true`.
- **Tests** use `pytest` with `markers`: `integration` for slower/data-dependent tests. Default test run includes all tests.
- **Config allows extra fields** — `ExperimentConfig(model_config={"extra": "allow"})` means runtime fields like the actual model object are attached after instantiation.

## CLI Usage

```bash
# Run FL experiment
uv run pfllib run -d MNIST -m CNN -a FedAvg -gr 2000 -did 0

# Generate dataset
uv run pfllib generate-data MNIST --noniid --partition dir

# List available algorithms and models
uv run pfllib list
uv run pfllib list --algorithms
uv run pfllib list --models
```

## Gotchas

- `data/`, `results/`, `outputs/` are gitignored — generated data and experiment outputs are not committed.
- The `run_experiment` function mutates `config.model` from a string to an actual `nn.Module` at runtime.
- `build_model()` in `runner.py` has hardcoded dataset-to-model-dimension mappings; adding a new dataset may require updating these branches.
- `register_algorithm` calls happen at import time in `pfllib/servers/__init__.py` — importing `pfllib.servers` (or `pfllib.clients`) is required before `list_algorithms()` / `get_algorithm()` will work.