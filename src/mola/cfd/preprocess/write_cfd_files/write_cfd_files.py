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
from mola.cfd import apply_to_solver
from mola import __MOLA_PATH__
from mola.logging import mola_logger, MolaException
from mola.server import server as SV
from mola import misc

def apply(workflow):

    set_default(workflow.RunManagement)
    apply_to_solver(workflow)

def set_default(RunManagement):

    # Set default parameters
    RunManagementDefault = dict(
        RunDirectory='.',
        NumberOfProcessors=None,
        SubmitJob=False,
        SecondsMarginForQuitBeforeTimeOut = 180,
        LauncherCommand = 'auto', # or 'sbatch job.sh', './job.sh'...
        mola_target_path = __MOLA_PATH__,
        FilesAndDirectories=[],
        )
    for key, default_value in RunManagementDefault.items():
        RunManagement.setdefault(key, default_value)
    
    # NumberOfProcessors must be set before this stage
    # It may have been set during an automatic splitting operation
    if not isinstance(RunManagement['NumberOfProcessors'], int):
        raise MolaException(f'The value {RunManagement["NumberOfProcessors"]} for NumberOfProcessors is not allowed. It must be an integer')

    if ('Machine' not in RunManagement) or (RunManagement['Machine'] == 'auto'):
        RunManagement['Machine'] = SV.guess_machine(RunManagement['RunDirectory'])
        
