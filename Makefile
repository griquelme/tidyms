# make file for pytest

test:
	pytest --cov=tidyms

coverage:
	pytest --cov=tidyms && coverage html