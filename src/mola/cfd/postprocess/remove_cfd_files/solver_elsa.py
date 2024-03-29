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

solver_specific_files = [
        'COMPLETED',
        'FAILED',
        'NEWJOB_REQUIRED',
        'compute.py',
        'coprocess.py',
        'job.sh',
        'main.cgns']

solver_specific_directories = [
        'OUTPUT',
        'LOGS',
        'OVERSET']

def apply(workflow):

    for file in solver_specific_files:
        try:
            os.unlink(file)
        except:
            pass

    for directory in solver_specific_directories:
        try:
            shutil.rmtree(directory)
        except:
            pass

def write_dummy_files_for_testing(workflow):
    for file in solver_specific_files:
        with open(file,'w') as f: f.write(f'test for {workflow.Solver}')

    for directory in solver_specific_directories:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, 'toto.py'),'w') as f:
            f.write(f'test for {workflow.Solver}')

