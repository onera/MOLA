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

import numpy as np
from mola.cfd.preprocess.motion import motion
from mola.logging import mola_logger, MolaAssertionError, MolaException

def apply_to_solver(workflow):
    unique_motion = check_unique_motion(workflow.Motion)
    if unique_motion is not None:
        _must_be_unsteady_if_has_motion(workflow.Motion, workflow.Numerics)
        
def check_unique_motion(Motion):
    '''
    Check that every mobile family has the same motion.
    It is not a constraint from Fast, but of the current implement in MOLA.
    '''
    unique_motion = None
    for MotionOnFamily in Motion.values():
        if motion.is_mobile(MotionOnFamily):
            if unique_motion is None:
                unique_motion = MotionOnFamily
            else:
                assert MotionOnFamily == unique_motion
    
    return unique_motion

def _must_be_unsteady_if_has_motion(Motion, Numerics, DefaultAzimutalStepInDegrees=0.5):

    DefaultTimeStep = get_timestep_based_on_azimutal_step(Motion, DefaultAzimutalStepInDegrees)

    if Numerics['TimeMarching'] == 'Steady':
        Numerics["TimeMarching"] = "Unsteady"
        if not 'TimeStep' in Numerics or Numerics['TimeStep'] is None:
            msg = f'fast solver requires unsteady simulation if it has Motion. Using TimeStep={DefaultTimeStep} (ΔΨ={DefaultAzimutalStepInDegrees}°)'
            mola_logger.user_warning(msg)
            Numerics["TimeStep"] = DefaultTimeStep


def get_timestep_based_on_azimutal_step(Motion, delta_psi):

    rpm = get_rpm(Motion)
    dt = delta_psi / ( 6 * rpm)
    return dt

def get_rpm(Motion):
    omega = np.linalg.norm(motion.get_first_found_rotation_speed_vector_at_motion(Motion))
    return omega * 30 / np.pi


def get_rotation_parameter(Motion):
    RotationAxis = motion.get_rotation_axis_from_first_found_motion(Motion)
    RotationAxisOrigin = motion.get_first_found_rotation_axis_origin_vector_at_motion(Motion)

    rotation = [
        RotationAxis[0], 
        RotationAxis[1], 
        RotationAxis[2], 
        RotationAxisOrigin[0], 
        RotationAxisOrigin[1], 
        RotationAxisOrigin[2],
        0., # freq
        0.  # amplitude
    ]
    return rotation

