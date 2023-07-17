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

    setdefault(workflow)

    current_path = os.path.dirname(os.path.realpath(__file__))
    solverModule = misc.load_source('solverModule', os.path.join(current_path, f'solver_{workflow.Solver}.py'))
    solverModule.adapt_to_solver(workflow)

def setdefault(workflow):

    # NumberOfProcessors must be set before this stage
    # It may have been set during an automatic splitting operation
    ERR_NPROC = misc.RED+f'The value {workflow.RunManagement["NumberOfProcessors"]} for NumberOfProcessors is not allowed. It must be an integer'+misc.ENDC
    assert isinstance(workflow.RunManagement['NumberOfProcessors'], int), ERR_NPROC

    if workflow.RunManagement['Machine'] == 'auto':
        # TODO 'auto' may be too generic. Writing Machine='ONERA' may be better to redirect to sator, spiro, etc.
        # FIXME For now, it is spiro but it may be changed once a dedicated module for file and server management is ready
        workflow.RunManagement['Machine'] = 'spiro'

    if 'TimeLimit' not in workflow.RunManagement:
        # To update depending on the cluster
        if workflow.RunManagement['Machine'] in ['sator', 'spiro']:
            workflow.RunManagement['TimeLimit'] = '0-15:00'
        else:
            print(misc.YELLOW + f'The machine {workflow.RunManagement["Machine"]} is unknown' + misc.ENDC)
            workflow.RunManagement['TimeLimit'] = '0-24:00'

    if 'SlurmConstraint' not in workflow.RunManagement:
        if workflow.RunManagement['Machine'] == 'sator':
            # TODO Remove this constraint if it is not useful anymore
            workflow.RunManagement['SlurmConstraint'] = 'csl'
        else:
            workflow.RunManagement['SlurmConstraint'] = None

    if workflow.RunManagement['Machine'] == 'spiro':
        workflow.RunManagement.setdefault('SlurmQualityOfService', 'c1_test_giga')

