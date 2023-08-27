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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'GALAHAD Python Interface'
copyright = 'Gould/Orban/Toint, for GALAHAD productions, GALAHAD 4 C/Python interfaces copyright Fowkes/Gould'
author = 'Jaroslav Fowkes and Nick Gould'

# The full version, including alpha/beta/rc tags
release = '1.0'
# The short X.Y version
version = release

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = '21 July 2022'
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%d %B %Y'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc','numpydoc','sphinx_math_dollar','sphinx.ext.mathjax']

# Add any paths that contain templates here, relative to this directory.
#templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ['.rst']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

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

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'sphinx_rtd_theme'
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

# Add any custom static css files (style sheets)
# These paths are relative to html_static_path
html_css_files = [
    'css/custom.css',
]
