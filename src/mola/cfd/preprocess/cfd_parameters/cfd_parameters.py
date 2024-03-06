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
from mola.logging import mola_logger


def apply(workflow):
    
    set_problem_dimension(workflow)
    set_physical_parameters(workflow)
    set_numerical_parameters(workflow)

    workflow.SolverParameters = dict()
    misc.apply_to_solver(workflow)


def set_problem_dimension(workflow):
    
    # Check that all bases have the same dimension
    dimOfBases = set(base.dim() for base in workflow.tree.bases())
    assert len(dimOfBases) == 1, 'All bases have not the same physical dimension'
    workflow.ProblemDimension = int(list(dimOfBases)[0])

def set_physical_parameters(workflow):

    workflow.Turbulence.setdefault('Model', 'Wilcox2006-klim')
    # TODO: TurbulenceCutOffRatio=1e-8 : Change this value for workflows for external aerodynamics 
    workflow.Turbulence.setdefault('TurbulenceCutOffRatio', 1e-8)
    workflow.Turbulence.setdefault('TransitionMode', None)


def set_numerical_parameters(workflow):

    workflow.Numerics.setdefault('Scheme', 'Jameson')
    workflow.Numerics.setdefault('TimeMarching', 'Steady')
    workflow.Numerics.setdefault('NumberOfIterations', 10000)
    workflow.Numerics.setdefault('MinimumNumberOfIterations', 1000)

    workflow.Numerics.setdefault('IterationAtInitialState', 1)
    workflow.Numerics.setdefault('TimeAtInitialState', 0.)

    # Time marching
    if workflow.Numerics['TimeMarching'] != 'Steady':
        if 'TimeStep' not in workflow.Numerics:
            mola_logger.error(f'TimeStep must be defined to perform a simulation with TimeMarching={workflow.Numerics["TimeMarching"]}')
        
        workflow.Numerics.setdefault('TimeMarchingOrder', 2)

    check_cfl(workflow)
    

def check_cfl(workflow):

    try:
        cfl = workflow.Numerics['CFL']
    except:
        mola_logger.error('CFL is not defined. Please give a value or function in Workflow.Numerics')

    if isinstance(cfl, float):
        pass
    elif isinstance(cfl, int):
        cfl = float(cfl)
    elif isinstance(cfl, dict):
        cfl.setdefault('StartIteration', workflow.Numerics['IterationAtInitialState'])
        mandatoryKeys = ['EndIteration', 'StartValue', 'EndValue']
        CFL_dict_has_all_mandatory_keys = all([(key in cfl) for key in mandatoryKeys])

        if not CFL_dict_has_all_mandatory_keys:
            mola_logger.error(f'If CFL is a dict, it must contains at least {", ".join(mandatoryKeys)}. \
    You may also define StartIteration, otherwise it will be equal to IterationAtInitialState (1 by default).')
        
    else:
        mola_logger.error('CFL must be a scalar or a dict')
