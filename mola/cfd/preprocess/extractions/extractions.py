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


def apply(workflow):

    process_extractions_3d(workflow)
    process_extractions_2d(workflow)

    current_path = os.path.dirname(os.path.realpath(__file__))
    solverModule = misc.load_source('solverModule', os.path.join(current_path, f'solver_{workflow.Solver}.py'))
    solverModule.adapt_to_solver(workflow)


def process_extractions_3d(workflow):
    '''
    Process 3D extractions. 
    The conservatives quantities are unconditionnally extracted, plus all the 
    quantities 
    '''

    # TODO Check the name 
    workflow.Extractions.append(dict(type='3D', fields=workflow.Flow['ReferenceState']))


def process_extractions_2d(workflow):
    pass
    
    
