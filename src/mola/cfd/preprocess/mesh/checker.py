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

from mola.misc import load_source

backend = 'maia'

_possible_backends = ['maia', 'cassiopee', 'treelab']

def set_backend(other_backend : str):
    
    if other_backend not in _possible_backends:
        raise AttributeError(f'requested backend {other_backend} not in allowed ones: {str(_possible_backends)} ')

    backend = other_backend

def is_partitioned_for_use_in_maia(tree):

    tbx = load_source('tbx',f'{backend}_wrapper')
    return tbx.is_partitioned_for_use_in_maia(tree)

def is_distributed_for_use_in_maia(tree):

    tbx = load_source('tbx',f'{backend}_wrapper')
    return tbx.is_distributed_for_use_in_maia(tree)

