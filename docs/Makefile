# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# dir for bokeh figures
BOKEH_DIR=bokeh
STATIC_DIR=_static
BOKEH_SRC_FILES=$(wildcard $(BOKEH_DIR)/*.py)
BOKEH_HTML_FILES=$(patsubst $(BOKEH_DIR)/%.py,$(STATIC_DIR)/%.html,$(BOKEH_SRC_FILES))

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: html
html: $(BOKEH_HTML_FILES)
	sphinx-build -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: clean
clean:
	rm -r $(BUILDDIR)

$(STATIC_DIR)/%.html: $(BOKEH_DIR)/%.py
	python $< $@