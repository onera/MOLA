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
    
    set_modelling_parameters(workflow)
    set_numerical_parameters(workflow)

    workflow.SolverParameters = dict()
    current_path = os.path.dirname(os.path.realpath(__file__))
    solverModule = misc.load_source('solverModule', os.path.join(current_path, f'solver_{workflow.Solver}.py'))
    solverModule.adapt_to_solver(workflow)


def set_modelling_parameters(workflow):
    
    # Check that all bases have the same dimension
    dimOfBases = set(base.dim() for base in workflow.tree.bases())
    assert len(dimOfBases) == 1, 'All bases have not the same physical dimension'
    workflow.ProblemDimension = int(list(dimOfBases)[0])


def set_numerical_parameters(workflow):

    workflow.Numerics.setdefault('Scheme', 'Jameson')
    workflow.Numerics.setdefault('TimeMarching', 'Steady')
    workflow.Numerics.setdefault('NumberOfIterations', 10000)
    workflow.Numerics.setdefault('MinimumNumberOfIterations', 1000)
    workflow.Numerics.setdefault('TimeStep', None)

    workflow.Numerics.setdefault('IterationAtInitialState', 1)
    workflow.Numerics.setdefault('TimeAtInitialState', 0.)

    # Time marching
    if workflow.Numerics['TimeMarching'] != 'Steady':
        assert workflow.Numerics['TimeStep'] is not None, misc.RED+f'TimeStep must be defined to perform a simulation with TimeMarching={workflow.Numerics["TimeMarching"]}'+misc.ENDC

        workflow.Numerics.setdefault('TimeMarchingOrder', 2)

    # CFL
    if workflow.Numerics['CFL'] is None:
        print(misc.RED+'CFL is not defined. Please give a value or function in Workflow.Numerics'+misc.ENDC)     
    elif isinstance(workflow.Numerics['CFL'], float):
        pass
    elif isinstance(workflow.Numerics['CFL'], int):
        workflow.Numerics['CFL'] = float(workflow.Numerics['CFL'])
    elif dict(workflow.Numerics['CFL']):
        workflow.Numerics['CFL'].setdefault('StartIteration', workflow.Numerics['IterationAtInitialState'])
        mandatoryKeys = ['EndIteration', 'StartValue', 'EndValue']
        ERROR = f'If CFL is a dict, it must contains at least {", ".join(mandatoryKeys)}. \
You may also define StartIteration, otherwise it will be equal to IterationAtInitialState (1 by default).'
        assert all([(key in workflow.Numerics['CFL']) for key in mandatoryKeys]), \
            misc.RED + ERROR + misc.ENDC
    else:
        print(misc.RED+'CFL must be a scalar or a dict'+misc.ENDC)

