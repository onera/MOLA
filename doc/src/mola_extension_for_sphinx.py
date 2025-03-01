#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

from docutils import nodes
from docutils.parsers.rst.roles import register_canonical_role

from mola.naming_conventions import *

def mola_name_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    '''
    This function is build from a pattern for a 'role' in sphinx (like `:func:`, `:class:`, etc)

    It defines a new 'role' called `:mola_name:`, such as the given text is replaced by the 
    corresponding constant value found in :mod:`mola.naming_conventions`.

    Example
    -------

        .. code-block:: :mola_name:`FILE_INPUT_WORKFLOW`

        in a docstring will display: :mola_name:`FILE_INPUT_WORKFLOW`

        in the compiled documentation.
    '''
    try:
        value = eval(text)
    except (NameError, ValueError):
        value = f'<UNRESOLVED_LINK for {text}>'

    return [nodes.inline(rawtext, f"{value}")], []

def setup(app):
    register_canonical_role('mola_name', mola_name_role)
    return {'version': '0.1'}
