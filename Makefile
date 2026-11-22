.PHONY: lint test check

lint:
	uv run --extra dev python scripts/lint.py

test:
	uv run --extra dev pytest

check: lint test
