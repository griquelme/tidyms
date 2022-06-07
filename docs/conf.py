# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.pardir))
sys.path.insert(0, os.path.abspath(os.getcwd())) 
from bokeh_plots import create_plots

# -- Project information -----------------------------------------------------

project = 'TidyMS'
copyright = "2020, Bioanalytical Mass Spectrometry Group at CIBION-CONICET"
author = 'Gabriel Riquelme'

# -- generate plot files -----------------------------------------------------
if not os.path.isdir("_static"):
    os.mkdir("_static")

if not os.path.isdir("_build"):
    os.mkdir("_build")

create_plots()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'bokeh.sphinxext.bokeh_plot',
    'matplotlib.sphinxext.plot_directive',
    'numpydoc'
]

add_module_names = False
# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

intersphinx_mapping = {
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# set index.rst as the master doc
master_doc = 'index'

# include __init__ in docs
autoclass_content = 'both'