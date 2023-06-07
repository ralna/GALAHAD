# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GALAHAD C interfaces'
copyright = '2023, Jaroslav Fowkes & Nick Gould'
author = 'Jaroslav Fowkes & Nick Gould'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

# Specify the path to Doxyrest extensions for Sphinx:

#sys.path.insert(1, os.path.abspath('doxyrest-sphinx-dir'))
sys.path.insert(1, '/share/system/usr/local/src/doxyrest_b/doxyrest/sphinx')
# Add Doxyrest extensions ``doxyrest`` and ``cpplexer``:

extensions += ['doxyrest', 'cpplexer']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# If you used INTRO_FILE in 'doxyrest-config.lua' to force-include it
# into 'index.rst', exclude it from the Sphinx input (otherwise, there
# will be build warnings):

#exclude_patterns += ['page_index.rst']
