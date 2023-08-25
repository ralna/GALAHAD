# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GALAHAD C interfaces'
copyright = 'Gould/Orban/Toint, for GALAHAD productions, GALAHAD 4 C/Python interfaces copyright Fowkes/Gould'
author = 'Jaroslav Fowkes & Nick Gould'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
#xtensions = []
extensions = ['sphinx.ext.autodoc','numpydoc','sphinx_math_dollar','sphinx.ext.mathjax']

# Specify the path to Doxyrest extensions for Sphinx:

import sys
#sys.path.insert(1, os.path.abspath('doxyrest-sphinx-dir'))
#sys.path.insert(1, '/share/system/usr/local/src/doxyrest_b/doxyrest/sphinx')
sys.path.insert(1, '/usr/local/src/doxyrest/sphinx')

# Add Doxyrest extensions ``doxyrest`` and ``cpplexer``:
extensions += ['doxyrest', 'cpplexer']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
#xclude_patterns = []
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# prevent MathJax from parsing dollar signs which are ignored by the 
# extension because they should not be parsed as math.
#mathjax_config = {
#    'tex2jax': {
#        'inlineMath': [ ["\\(","\\)"] ],
#        'displayMath': [["\\[","\\]"] ],
#    },
#}

mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'pydata_sphinx_theme'

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#
html_logo = "galahad.small.png"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If you used INTRO_FILE in 'doxyrest-config.lua' to force-include it
# into 'index.rst', exclude it from the Sphinx input (otherwise, there
# will be build warnings):

#exclude_patterns += ['page_index.rst']

# Add any custom static css files (style sheets)
# These paths are relative to html_static_path
html_css_files = [
    'css/custom.css',
]

#rst_epilog = "\n.. include:: .special.rst\n"
