from mola import __version__

import pathlib
import sys

## Embedded onera_sphinx_theme
# docs 'source' directory
THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(THIS_DIR.parent/"theme"))
from onera_sphinx_theme import prepare,OST_THEME_DIR

meta_data = {
    "project": "MOLA",
    "copyright": "ONERA",
    "authors":["Luis Bernardos", "Thomas Bontemps"],
    "html_static_path":["_static"],
    "html_logo":"_static/icons/mola_logo.png",  # it also possible to define html_logo_dark and html_logo_light
    "html_favicon": "_static/icons/mola_icon.ico",
    "html_title":"MOLA documentation",
    # "html_logo_text":"MOLA",
    "substitutions": {'authors':'__authors__'},
    "version": __version__,

    # "build_version_switcher": True,
    # "all_versions": [__version__, "1.19"],


    "docsbuilddir":"../html",
    
    "header_navbar_links":{ #list of {"name":...,"target":...} dicts or dict <name:target>
        "User manual":  "user_manual/index",
        "Examples": "examples/index",
        "Tutorials": "tutorials/index",
        "Developer manual": "developer_manual/index",
        "Changelog" : "changelog",
    },
    # "header_navbar_external_links":[
    #     {"url":"https://www.google.com","name":"Google"}
    # ],
    "header_links_before_dropdown":6,
    "header_right_icons":[
        {'url':'https://gitlab.onera.net/numerics/mola', 'icon':'gitlab', 'name':'MOLA GitLab'},
        {'url':'https://github.com/onera/MOLA', 'icon':'_static/icons/github-white.svg', 'name':'MOLA GitHub'},
        {'url':'https://github.com/Luispain/treelab', 'icon':'_static/icons/treelab.png', 'name':'Treelab GitHub'},
        {'url':'https://numerics.gitlab-pages.onera.net/mola/v1.19', 'icon':'_static/icons/mola_v1.png', 'name':'MOLA v1'},
    ],
    "intersphinx_mapping":{
        'python':('https://docs.python.org/3',None),
        'numpy':('https://numpy.org/doc/stable',None),
        'scipy': ('https://docs.scipy.org/doc/scipy/', None),
        'mpi4py': ('https://mpi4py.readthedocs.io/en/stable/', None),
        'matplotlib': ('https://matplotlib.org/stable/', None),
        'maia': ("https://onera.github.io/Maia/1.5/",None),
        'Converter': ('http://elsa.onera.fr/Cassiopee/', '/stck/benoit/Cassiopee/Apps/Modules/Converter/doc/build/objects.inv'),
    },
}

prepare(
    globals(),
    meta_data, # json, yaml or dict
)
# Add other extensions not already loaded with the theme.
# Variables are accessible through globals()
extensions.add('mola_extension_for_sphinx')
extensions.add('sphinx.ext.graphviz')

# remove warnings about duplicate labels
# see https://stackoverflow.com/a/77577337
suppress_warnings = ['autosectionlabel.*'] 
