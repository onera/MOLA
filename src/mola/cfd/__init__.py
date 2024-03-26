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
from mola import misc

def apply_to_solver(workflow):
    '''
    If this function is called from /path/filename.py, it calls the function 
    ``adapt_to_solver(workflow)`` in ``/path/solver_<workflow.Solver>``.
    '''
    current_path = get_path_back_in_traceback()
    solverModule = misc.load_source('solverModule', os.path.join(current_path, f'solver_{workflow.Solver}.py'))
    solverModule.adapt_to_solver(workflow)

def get_path_back_in_traceback(step=3):
    import traceback
    stack = traceback.extract_stack()
    previous_filename = stack[-step].filename
    previous_path = '/'.join(previous_filename.split('/')[:-1])
    return previous_path
