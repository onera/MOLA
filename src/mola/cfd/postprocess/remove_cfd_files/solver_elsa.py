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
import shutil 
import mola.naming_conventions as names

solver_specific_files = names.STATUS_FILES + [
    names.FILE_COMPUTE,
    names.FILE_COPROCESS,
    names.FILE_JOB,
    names.FILE_INPUT_SOLVER,
    names.FILE_COLOG,
]

solver_specific_directories = [
    names.DIRECTORY_OUTPUT,
    names.DIRECTORY_LOG,
    names.DIRECTORY_OVERSET,
]

def apply(workflow):

    for file in solver_specific_files:
        filepath = os.path.join(workflow.RunManagement['RunDirectory'],file)
        try:
            os.unlink(filepath)
        except:
            pass

    for directory in solver_specific_directories:
        dirpath = os.path.join(workflow.RunManagement['RunDirectory'],directory)
        try:
            shutil.rmtree(dirpath)
        except:
            pass

def write_dummy_files_for_testing(workflow):
    for file in solver_specific_files:
        filepath = os.path.join(workflow.RunManagement['RunDirectory'],file)
        with open(filepath,'w') as f: f.write(f'test for {workflow.Solver}')

    for directory in solver_specific_directories:
        dirpath = os.path.join(workflow.RunManagement['RunDirectory'],directory)
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, 'toto.py'),'w') as f:
            f.write(f'test for {workflow.Solver}')

