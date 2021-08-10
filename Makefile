# make file for pytest

.PHONY: test-unit
test-unit:
	pytest --cov=tidyms tests/unit

.PHONY: test-all
test-all:
	pytest --cov=tidyms

.PHONY: coverage
coverage:
	pytest --cov=tidyms && coverage html