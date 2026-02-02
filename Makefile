.PHONY: test setup shell coverage publish-build publish-test publish publish-clean

test:
	uv run pytest

coverage:
	uv run pytest --cov=sqlitesearch --cov-report=term-missing --cov-report=html

setup:
	uv sync --extra dev

shell:
	uv shell

publish-build:
	uv run hatch build

publish-test:
	uv run hatch publish --repo test

publish:
	uv run hatch publish

publish-clean:
	rm -r dist/
