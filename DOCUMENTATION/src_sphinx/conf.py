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
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'MOLA'
copyright = 'ONERA'
author = 'Luis BERNARDOS'
release = 'v1.18' # sets the default doc version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
'sphinx.ext.napoleon',
'sphinx.ext.autosectionlabel',
'sphinx_copybutton',
'sphinx.ext.intersphinx',
'sphinx.ext.todo',
'sphinx.ext.autosummary',
]

suppress_warnings = ['autosectionlabel.*'] # to get rid of duplicate label warning

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'Converter': ('http://elsa.onera.fr/Cassiopee/', '/home/benoit/Cassiopee/Apps/Modules/Converter/doc/build/objects.inv'),}

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
html_theme = 'pydata_sphinx_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'navbar_end': ['theme-switcher', 'navbar-icon-links', 'version-switcher'],
    "switcher": {
    "json_url": "https://numerics.gitlab-pages.onera.net/mola/Dev/_static/versions.json",
    "version_match": release,
                }
}

html_logo = os.path.join('FIGURES','MOLA_logo.png')

html_favicon = os.path.join('FIGURES','favicon.ico')

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# variables to pass to HTML templating engine
html_context = {
    "default_mode": "light",
    'versions_dropdown': {
        'Dev': 'development',
        'v1.18': 'v1.18',
        'v1.17': 'v1.17',
        'v1.16': 'v1.16',
        'v1.15': 'v1.15',
        'v1.14': 'v1.14',
        'v1.13': 'v1.13'
    },
}

numfig = True

todo_include_todos = True

autodoc_member_order = 'bysource'