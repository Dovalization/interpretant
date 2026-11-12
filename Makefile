.PHONY: install lint type-check test figures clean

install:
	uv sync --all-extras

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

fmt:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

type-check:
	uv run mypy src/

test:
	uv run pytest tests/ -v --cov=src/interpretant --cov-report=term-missing

figures:
	uv run interpretant figures

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
