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

from collections import Counter

import numpy as np
from mola.dependency_injector.backend_function_caller import call_backend_function



default_backend = 'maia'
possible_backends = ['maia', 'cassiopee', 'treelab']

def is_partitioned_for_use_in_maia(tree, backend=default_backend):
    return call_backend_function('is_partitioned_for_use_in_maia', backend, tree)

def is_distributed_for_use_in_maia(tree, backend=default_backend):
    return call_backend_function('is_distributed_for_use_in_maia', backend, tree)

def assert_zones_have_zone_type_node(tree, backend=default_backend):
    return call_backend_function('assert_zones_have_zone_type_node', backend, tree)


def seems_like_a_single_cgns_node( thing ):
    
    if not isinstance(thing,list):
        return False
    
    if not len(thing) == 4:
        return False
    
    if not isinstance(thing[0], str):
        return False
    
    if not isinstance(thing[3], str):
        return False
    
    if not isinstance(thing[2], list):
        return False
    
    if not (isinstance(thing[1],np.ndarray) or (thing[1] is None) ):
        return False
    
    return True
    

def assert_unique_siblings_names( node ):
    children_names = [n[0] for n in node[2]]
    count = _count_repeated_items(children_names)
    msg = ''
    for name, nb in count.items():
        if nb>1:
            msg += f'node "{name}" is not unique ({nb-1} other)\n'
    if msg:
        raise ValueError(msg)
    
    for n in node[2]:
        assert_unique_siblings_names(n)
    
def _count_repeated_items(lst):
    counts = Counter(lst)
    return {k: v for k, v in counts.items() if v > 1}

