.PHONY: install dev format lint typecheck test test-unit test-integration clean

install:
	uv add .

dev:
	uv add --dev ".[dev]"

format:
	uv run ruff format pfllib/ tests/
	uv run ruff check --fix pfllib/ tests/

lint:
	uv run ruff check pfllib/ tests/

typecheck:
	uv run mypy pfllib/

test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/ -v -m "not integration"

test-integration:
	uv run pytest tests/ -v -m integration

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info .eggs/ .mypy_cache/ .pytest_cache/ .ruff_cache/
