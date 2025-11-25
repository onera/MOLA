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

import os 
import functools
from treelab import cgns
from mola import __MOLA_PATH__
from mola.misc import run_as_mpi_subprocess
from mola.dependency_injector.retriever import load_source
from ..logging import mola_logger, MolaException, MolaNotImplementedError, redirect_streams_to_null

def get_path_back_in_traceback(step=3):
    import traceback
    stack = traceback.extract_stack()
    previous_filename = stack[-step].filename
    previous_path = os.sep.join(previous_filename.split(os.sep)[:-1])
    return previous_path


def call_solver_specific_function(workflow, function_name, step=3, *args, **kwargs):
    '''
    This is a generic function that is used for calling a solver-specific 
    implementation function contained in a module named ``solver_<NameOfSolver>.py``
    located in the current path (hence, this is context-dependent).
    
    Please note that the solver-specific function always requires a workflow as 
    the first mandatory argument.
    '''
    current_path = get_path_back_in_traceback(step)
    expected_module = os.path.join(current_path, f'solver_{workflow.Solver}.py')

    try:
        solverModule = load_source('solverModule', expected_module)
    except FileNotFoundError as e:
        msg = (f'Missing solver-specific module "solver_{workflow.Solver}.py"'
               f' when requesting "{function_name}" at {current_path}')
        raise MolaException(msg) from e

    try:
        fun = getattr(solverModule, function_name)
    except AttributeError as e:
        msg = f'Function {function_name} not implemented in {expected_module}'
        raise MolaNotImplementedError(msg) from e

    return fun(workflow, *args, **kwargs)


def apply_to_solver(workflow):
    '''
    This is a shortcut for :py:func:`call_solver_specific_function` for 
    ``function_name='apply_to_solver'``.
    '''
    return call_solver_specific_function(workflow, 'apply_to_solver', step=4)


def apply(workflow):
    '''
    This is a shortcut for :py:func:`call_solver_specific_function` for 
    ``function_name='apply'``.
    '''
    return call_solver_specific_function(workflow, 'apply', step=4)

def parallel_execution_with_maia(number_of_cells_limit=1e5, second_tree=False):
    """
    This decorator triggers the following behavior on function `fun(tree, *args, *kwargs)`:

    #. the input `tree` is written in a temporary file.

    #. This file is read in parallel as a distributed tree with Maia, using all the processors availble on 
       the current machin.

    #. The function `fun(tree, *args, *kwargs)` is applied in parallel.

    #. The resulted tree is saved in a temporary file.

    #. This file is finally read, temporary files are deleted, and the output tree is returned.

    Parameters
    ----------

    number_of_cells_limit : int

        If the number of cells in the mesh is below this number, then the function is applied sequentially (like without the decorator).
        Otherwise, the decorator is fully applied and the function is applied in paralllel.
        The default value is 1e5.

    second_tree : bool

        If True, then the decorated function is expected to have two trees as the first two arguments. 
        Both will be written in temporary files. 
        Only one tree is returned.

    Examples 
    --------
    
    .. code-block::python

        @parallel_execution_with_maia()
        def my_function(tree, arg1, arg2, option=None):
            ...

    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            # remove tree from args
            args = list(args)
            tree = args.pop(0)
            assert isinstance(tree, cgns.Tree)

            if tree.numberOfCells() < number_of_cells_limit:
                tree = func(tree, *args, **kwargs)

            else:
                
                tmp_input_filename = '.tmp_mesh_mola_before.cgns'
                tmp_output_filename = '.tmp_mesh_mola_after.cgns'
                with redirect_streams_to_null(): 
                    tree.save(tmp_input_filename)

                # Get module and function names to import it in fun_exec_in_parallel
                func_name = func.__name__
                module_name = os.path.join(__MOLA_PATH__, func.__module__.replace('.', os.path.sep) + '.py')

                def fun_exec_in_parallel(module_name, func_name, tmp_input_filename, tmp_output_filename, *args, **kwargs):
                    import os
                    import maia
                    from mpi4py import MPI
                    from treelab import cgns
                    from mola.cfd import load_source

                    expected_module = os.path.join(module_name)
                    solverModule = load_source('solverModule', expected_module)
                    func = getattr(solverModule, func_name).__wrapped__
                    
                    tmp_tree = maia.io.file_to_dist_tree(tmp_input_filename, MPI.COMM_WORLD)
                    tmp_tree = cgns.castNode(tmp_tree)
                    tmp_tree = func(tmp_tree, *args, **kwargs)
                    maia.io.part_tree_to_file(tmp_tree, tmp_output_filename, MPI.COMM_WORLD, single_file=True)

                if second_tree:
                    tmp_input2_filename = '.tmp_mesh2_mola_before.cgns'
                    source_tree = args[0]
                    assert isinstance(source_tree, cgns.Tree)
                    with redirect_streams_to_null(): 
                        source_tree.save(tmp_input2_filename)
                    args[0] = tmp_input2_filename

                    def fun_exec_in_parallel(module_name, func_name, tmp_input_filename, tmp_output_filename, tmp_input2_filename, *args, **kwargs):
                        import os
                        import maia
                        from mpi4py import MPI
                        from treelab import cgns
                        from mola.cfd import load_source

                        expected_module = os.path.join(module_name)
                        solverModule = load_source('solverModule', expected_module)
                        func = getattr(solverModule, func_name).__wrapped__
                        
                        tmp_tree = maia.io.file_to_dist_tree(tmp_input_filename, MPI.COMM_WORLD)
                        tmp_tree = cgns.castNode(tmp_tree)
                        tmp_tree2 = maia.io.file_to_dist_tree(tmp_input2_filename, MPI.COMM_WORLD)
                        tmp_tree2 = cgns.castNode(tmp_tree2)
                        tmp_tree = func(tmp_tree, tmp_tree2, *args, **kwargs)
                        maia.io.part_tree_to_file(tmp_tree, tmp_output_filename, MPI.COMM_WORLD, single_file=True)

                run_as_mpi_subprocess(fun_exec_in_parallel, None, None, 
                                    module_name, func_name, 
                                    tmp_input_filename, tmp_output_filename,
                                    *args, **kwargs)
                
                tree = cgns.load(tmp_output_filename)
                os.remove(tmp_input_filename)
                os.remove(tmp_output_filename)            

            return tree

        return wrapper
    return decorator
