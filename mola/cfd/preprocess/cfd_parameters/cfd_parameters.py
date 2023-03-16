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

import misc as m


def set_cfd_parameters(workflow):

    workflow.SolverParameters = dict()
    
    set_modelling_parameters(workflow)
    set_numerical_parameters(workflow)

    solverModule = m.load_source('solverModule', workflow.Solver)
    solverModule.adapt_to_solver(workflow)


def set_modelling_parameters(workflow):
    
    # Check that all bases have the same dimension
    dimOfBases = set(base.dim() for base in workflow.tree.bases())
    assert len(dimOfBases) == 1, 'All bases have not the same physical dimension'
    workflow.ProblemDimension = list(dimOfBases)[0]


def set_numerical_parameters(workflow):
    pass
