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
from pathlib import Path
import sys
from mola import __version__

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

# -- Project information -----------------------------------------------------

project = 'MOLA'
copyright = 'ONERA'
authors = ["Luis Bernardos", "Thomas Bontemps"],
version = __version__
# available versions are to be written in _static/versions.json

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
    'sphinx.ext.autodoc', 
    'sphinx.ext.graphviz',
    'sphinx_design',
    'mola_extension_for_sphinx',
    ]

suppress_warnings = ['autosectionlabel.*'] # to get rid of duplicate label warning

intersphinx_mapping = {
    'python':('https://docs.python.org/3',None),
    'numpy':('https://numpy.org/doc/stable',None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'mpi4py': ('https://mpi4py.readthedocs.io/en/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'maia': ("https://onera.github.io/Maia/1.5/",None),
    'Converter': ('http://elsa.onera.fr/Cassiopee/', '/stck/benoit/Cassiopee/Apps/Modules/Converter/doc/build/objects.inv'),
    }

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
html_style = '_static/css/custom.css'

# html_logo = str(Path('_static/icons/mola_logo.png'))
html_favicon = str(Path('_static/mola_icon.svg'))

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "logo": {
      "image_light": "_static/mola_light.svg",  # should be in _static directory, not in a subdir like _static/icons/
      "image_dark" : "_static/mola_dark.svg",   # should be in _static directory, not in a subdir like _static/icons/
    },

    "navbar_start": ["navbar-logo"], #"navbar-version"],
    "navbar_align": "content",
    "header_links_before_dropdown": 5,
    'navbar_end': ['theme-switcher', 'navbar-icon-links', 'version-switcher'],
    "switcher": {
        "json_url": "https://numerics.gitlab-pages.onera.net/mola/Dev/_static/versions.json",
        "version_match": "Dev",
        },

    "navbar_persistent": ["search-button"],

    "icon_links": [

        {
            'name':'MOLA GitLab',
            'url':'https://gitlab.onera.net/numerics/mola', 
            # 'icon':'_static/icons/gitlab.svg', 
            "icon": "fa-brands fa-gitlab",
            },
        {
            'name':'MOLA GitHub',
            'url':'https://github.com/onera/MOLA', 
            # 'icon':'_static/icons/github-white.svg', 
            "icon": "fa-brands fa-github",
            },
        {
            'name':'Treelab GitHub',
            'url':'https://github.com/Luispain/treelab', 
            'icon':'_static/icons/treelab.png', 
            'type': 'local'
            },

    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

numfig = True
todo_include_todos = True
autodoc_member_order = 'bysource'

rst_epilog = f"""
.. |version| replace:: {__version__}
.. |release| replace:: {__version__}
"""
