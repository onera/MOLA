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

def _must_be_unsteady_if_has_motion(Motion, Numerics, DefaultAzimutalStepInDegrees=1.0):

    DefaultTimeStep = get_timestep_based_on_azimutal_step(Motion, DefaultAzimutalStepInDegrees)

    if Numerics['TimeMarching'] == 'Steady':
        msg = f'fast solver requires unsteady simulation if it has Motion. Using TimeStep={DefaultTimeStep} (ΔΨ={DefaultAzimutalStepInDegrees}°)'
        mola_logger.warning(msg)
        Numerics.update(dict(
            TimeMarching = 'Unsteady',
            TimeStep = DefaultTimeStep))


def get_timestep_based_on_azimutal_step(Motion, delta_psi):

    rpm = get_rpm(Motion)
    dt = delta_psi / ( 6 * rpm)
    return dt

def get_rpm(Motion):
    omega = np.linalg.norm(get_first_found_rotation_speed_vector_at_motion(Motion))
    return omega * 30 / np.pi

def get_first_found_rotation_speed_vector_at_motion(Motion):
    for family_name, motion_of_family in Motion.items():
        if 'RotationSpeed' in motion_of_family:
            return np.array(motion_of_family['RotationSpeed'])
    raise MolaException("no RotationSpeed attribute found in Motion")


def get_rotation_parameter(Motion):
    RotationAxis = np.array(Motion['RotationSpeed'])
    assert RotationAxis[1] == RotationAxis[2] == 0
    RotationSpeed = RotationAxis[0]
    # RotationSpeed = np.sqrt(RotationAxis.dot(RotationAxis)) # not working, the sign is always positive!
    if RotationSpeed != 0:
        RotationAxis = np.absolute(RotationAxis / RotationSpeed)
    else:
        RotationAxis = [1., 0., 0.]    

    rotation = [
        RotationAxis[0], 
        RotationAxis[1], 
        RotationAxis[2], 
        Motion['RotationAxisOrigin'][0], 
        Motion['RotationAxisOrigin'][1], 
        Motion['RotationAxisOrigin'][2],
        0., # freq
        0.  # amplitude
    ]
    return rotation

def is_any_family_mobile(Motion):
    for family_name, motion_of_family in Motion.items():
        if 'RotationSpeed' in motion_of_family:
            return True
    return False

