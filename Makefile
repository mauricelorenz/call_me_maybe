PYTHON = python
SRC = src

install:
	uv sync

run:
	uv run $(PYTHON) -m $(SRC)

debug:
	uv run $(PYTHON) -m pdb -m $(SRC)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -name "*.pyc" -delete

lint:
	flake8 src
	mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 src
	mypy src --strict

.PHONY: install run debug clean lint lint-strict
