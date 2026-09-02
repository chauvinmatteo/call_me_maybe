PYTHON = python3
MAIN = src/main.py
MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
--disallow-untyped-defs --check-untyped-defs
VENV = .venv

install:
	uv sync

run:
	uv run $(PYTHON) $(MAIN)

debug:
	uv run $(PYTHON) -m pdb $(MAIN)

clean:
	@rm -rf $$(find . -type d -name "__pycache__") $$(find . -type d -name ".mypy_cache")

lint:
	uv run $(PYTHON) -m flake8 . --exclude $(VENV)
	uv run $(PYTHON) -m mypy . $(MYPY_FLAGS)

lint-strict:
	uv run $(PYTHON) -m flake8 . --exclude $(VENV)
	uv run $(PYTHON) -m mypy . $(MYPY_FLAGS) --strict

.PHONY: install run debug clean lint lint-strict