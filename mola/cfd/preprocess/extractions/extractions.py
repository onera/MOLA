#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

from mola import misc


def apply(workflow):

    process_extractions_3d(workflow)
    process_extractions_2d(workflow)

    workflow.SolverParameters = dict()
    solverModule = misc.load_source('solverModule', workflow.Solver)
    solverModule.adapt_to_solver(workflow)


def process_extractions_3d(workflow):
    '''
    Process 3D extractions. 
    The conservatives quantities are unconditionnally extracted, plus all the 
    quantities 
    '''

    # TODO Check the name 
    workflow.Extractions.insert(0, dict(type='3D', fields=workflow.Flow['Conservatives']))


def process_extractions_2d(workflow):
    pass
    
    
