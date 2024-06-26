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

from mola.logging import mola_logger
from mola.cfd.preprocess.boundary_conditions.boundary_conditions import BoundaryConditionsNames
from mola.cfd.preprocess.motion.solver_sonics import translate_motion_to_sonics

BoundaryConditionsNamesInSONICS = set(v['sonics'] for v in BoundaryConditionsNames.values() if 'sonics' in v)

# For each boundary condition, this generic function does the job
def function_generator(name):
    def set_bc(workflow, *args, **kwargs):
        import miles
        if 'Motion' in kwargs:
            # put elements of dict Motion directly in kwargs (remove the "level" Motion)
            motion = kwargs.pop('Motion')
            motion = translate_motion_to_sonics(motion)
            kwargs['motion'] = motion
        miles.bcfactory(workflow.tree, name, *args, **kwargs)
    return set_bc

# Define functions with the write name to be called from .boundary_conditions
for fun_name in BoundaryConditionsNamesInSONICS:
    locals()[fun_name] = function_generator(fun_name)

