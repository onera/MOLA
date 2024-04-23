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
from .. import misc
from ..logging import mola_logger, MolaException

def get_path_back_in_traceback(step=3):
    import traceback
    stack = traceback.extract_stack()
    previous_filename = stack[-step].filename
    previous_path = '/'.join(previous_filename.split('/')[:-1])
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
        solverModule = misc.load_source('solverModule', expected_module)
    except FileNotFoundError as e:
        msg = (f'Missing solver-specific module "solver_{workflow.Solver}.py"'
               f' when requesting "{function_name}" at {current_path}')
        raise MolaException(msg) from e

    try:
        fun = getattr(solverModule, function_name)
    except AttributeError as e:
        msg = f'Function {function_name} not implemented in {expected_module}'
        raise MolaException(msg) from e

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

