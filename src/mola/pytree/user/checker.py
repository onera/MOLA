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


from mola.dependency_injector.backend_function_caller import call_backend_function

default_backend = 'maia'
possible_backends = ['maia', 'cassiopee', 'treelab']

def is_partitioned_for_use_in_maia(tree, backend=default_backend):
    return call_backend_function('is_partitioned_for_use_in_maia', backend, tree)

def is_distributed_for_use_in_maia(tree, backend=default_backend):
    return call_backend_function('is_distributed_for_use_in_maia', backend, tree)


