.PHONY: install test smoke lint format build clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest

smoke:
	dendritron smoke

lint:
	ruff check .
	mypy src/dendritron

format:
	ruff format .
	ruff check --fix .

build:
	python -m build

clean:
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('build', 'dist', '.pytest_cache', 'htmlcov')]"

