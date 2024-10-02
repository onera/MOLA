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

import sys
import functools

from treelab import cgns
from mola.logging import mola_logger, MolaException, MolaAssertionError

def is_mpi4py_imported():
    return 'mpi4py.MPI' in sys.modules

     
def sequential_execution(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not is_mpi4py_imported():
            result = func(*args, **kwargs)
        else:
            import mpi4py.MPI as MPI
            comm = MPI.COMM_WORLD 
            if comm.Get_size() > 1:
                # Execute sequentially only on rank 0
                if comm.Get_rank() == 0:
                    result = func(*args, **kwargs)
                else:
                    result = None
                result = comm.bcast(result, root=0)  
            else:
                result = func(*args, **kwargs)
        return result
    return wrapper



class MaiaParallel():
    '''
    Use this class as a decorator over a function written without taking
    into the type of tree in input. If the input is not a dist_tree, 
    the decorator will transform the input to provide a dist_tree to the 
    decorated functio, and returns the same type as the input.

    The first argument of the decorated function may be a tree or a Workflow.

    Example
    -------

    .. code-block:: python

        from mola.server.parallel import MaiaParallel

        @MaiaParallel
        def foo(tree, param1, param2=None):
            # Implementation of the function considering tree 
            # is already a dist_tree. 
            pass

    '''
     
    def __init__(self, func):
        # The first argument of func must be a tree or a Workflow
        self._original_func = func
        self._func = func

        try:
            import maia
            import mpi4py.MPI as MPI
            self.maia = maia
            self.comm = MPI.COMM_WORLD 
        except ImportError:
            self.maia = None
            self.comm = None

    def __call__(self, *args, **kwargs):
        if not self.maia:
            raise MolaException(f'maia cannot be imported, but it is required by function {self._original_func.__name__}')
        tree = self.get_tree(*args)
        if self.is_dist_tree(tree):
            tree = self.wrapper_dist_tree(tree, *args[1:], **kwargs)
        elif self.is_part_tree(tree):
            tree = self.wrapper_part_tree(tree, *args[1:], **kwargs)
        else: # is a full_tree
            tree = self.wrapper_full_tree(tree, *args[1:], **kwargs)
        tree = cgns.castNode(tree)
        return tree
    
    def get_tree(self, *args):
        if isinstance(args[0], cgns.Node):
            tree = args[0] 
        else:
            # args[0] is assumed to be a Workflow
            workflow = args[0]
            tree = workflow.tree
            def func_with_other_arguments(tree, *args, **kwargs):
                workflow.tree = tree
                return self._original_func(workflow, *args, **kwargs)
            self._func = func_with_other_arguments
        return tree
    
    def is_dist_tree(self, tree):
        return self.maia.pytree.get_node_from_name(tree, ':CGNS#Distribution') is not None

    def is_part_tree(self, tree):
        return self.maia.pytree.get_node_from_name(tree, ':CGNS#GlobalNumbering') is not None

    def is_full_tree(self, tree):
        return not self.is_dist_tree(tree) and not self.is_part_tree(tree)
    
    def wrapper_dist_tree(self, dist_tree, *args, **kwargs):
        self._func(dist_tree, *args, **kwargs)
        return dist_tree
    
    def wrapper_part_tree(self, part_tree, *args, **kwargs):
        raise MolaException('Not implemented for part_tree')
        ## still not functional
        # for gc in part_tree.group(Type='GridConnectivity*'):
        #     if len(gc.name().split('.')) < 2:
        #         # not compliant with maia
        #         gc.setName(f'{gc.name()}.0')
        # dist_tree = self.maia.factory.recover_dist_tree(part_tree, self.comm, data_transfer='ALL')
        # self._func(dist_tree, *args, **kwargs)
        # self.maia.transfer.dist_tree_to_part_tree_all(dist_tree, part_tree, self.comm)
        # return part_tree
    
    def wrapper_full_tree(self, full_tree, *args, **kwargs):
        dist_tree = self.maia.factory.full_to_dist_tree(full_tree, self.comm)
        self._func(dist_tree, *args, **kwargs)
        full_tree = self.maia.factory.dist_to_full_tree(dist_tree, self.comm)
        return full_tree
    

