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
from mola.logging import mola_logger, MolaAssertionError

def apply_to_solver(workflow):
    mola_logger.warning("motion to be implemented for FAST solver")

    unique_motion = check_unique_motion(workflow.Motion)
    if unique_motion is not None:
        if workflow.Numerics['TimeMarching'] == 'Steady':
            raise MolaAssertionError('FastS simulation with motion required to be unsteady')
        
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

def is_any_family_mobile(workflow):
    return 'RotationSpeed' in workflow.Motion
