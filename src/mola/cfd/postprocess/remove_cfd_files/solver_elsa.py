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

def adapt_to_solver(workflow):

    files_to_remove = [
        'COMPLETED',
        'compute.py',
        'coprocess.py',
        'job.sh',
        'main.cgns']
    directories_to_remove = [
        'OUTPUT',
        'LOGS',
        'OVERSET']
    
    for file in files_to_remove:
        try:
            os.unlink(file)
        except:
            pass

    for directory in directories_to_remove:
        try:
            shutil.rmtree(directory)
        except:
            pass
