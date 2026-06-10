.PHONY: test setup shell coverage release publish-build publish-test publish publish-clean

test:
	uv run pytest

coverage:
	uv run pytest --cov=sqlitesearch --cov-report=term-missing --cov-report=html

setup:
	uv sync --dev

shell:
	uv shell

# Release: tag the current version (v<version>) and push to trigger CI publish.
# CI workflow: .github/workflows/publish.yml (on tag push v*) -> PyPI.
release:
	@VERSION=$$(grep -E "^__version__" sqlitesearch/__version__.py | sed -E "s/.*['\"]([^'\"]+)['\"].*/\1/"); \
	echo "Releasing v$$VERSION"; \
	git tag "v$$VERSION"; \
	git push origin "v$$VERSION"

# Manual publish (legacy) -- prefer `make release` (tag push -> CI publish).
publish-build:
	uv run hatch build

publish-test:
	uv run hatch publish --repo test

publish:
	uv run hatch publish

publish-clean:
	rm -r dist/
