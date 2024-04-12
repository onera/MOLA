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

from mola.cfd import apply_to_solver
from mola.logging import mola_logger, MolaException
from mola import server as SV

def apply(workflow):

    set_default(workflow.RunManagement)
    apply_to_solver(workflow)

def set_default(RunManagement):

    # Set default parameters
    RunManagementDefault = dict(
        RunDirectory='.',
        NumberOfProcessors=None,
        SecondsMarginForQuitBeforeTimeOut = 180,
        LauncherCommand = 'auto', # or 'sbatch job.sh', './job.sh'...
        FilesAndDirectories=[],
        )
    for key, default_value in RunManagementDefault.items():
        RunManagement.setdefault(key, default_value)
    
    # NumberOfProcessors must be set before this stage
    # It may have been set during an automatic splitting operation
    if not isinstance(RunManagement['NumberOfProcessors'], int):
        raise MolaException(f'The value {RunManagement["NumberOfProcessors"]} for NumberOfProcessors is not allowed. It must be an integer')

    set_default_machine(RunManagement)
    
    RunManagement.setdefault('mola_target_path', SV.get_mola_installation_path(RunManagement['Machine']))
        
    if not SV.run_on_localhost(RunManagement['Machine'], RunManagement['RunDirectory']):
        mola_logger.info(f"> Run on a remote machine ({RunManagement['Machine']}):\n"
                         f"    on path {RunManagement['RunDirectory']}\n"
                         f"    sourcing {RunManagement['mola_target_path']}"
                         )

def set_default_machine(RunManagement):
    if ('Machine' not in RunManagement) or (RunManagement['Machine'] == 'auto'):
        RunManagement['Machine'] = SV.guess_machine(RunManagement['RunDirectory'])

