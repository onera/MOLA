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

from mola.cfd import apply_to_solver
from mola.logging import mola_logger, MolaException
from mola.cfd.preprocess.mesh.overset import hasAnyOversetMotion

def apply(workflow):
    '''
    Set Motion for each families
    '''
    
    if not hasAnyOversetMotion(workflow.RawMeshComponents):
        
        # cannot avoid this in general? It is incompatible with overset motion
        set_default_motion_on_families(workflow) 

    apply_to_solver(workflow)

def set_default_motion_on_families(workflow):
    if workflow.Motion is None:
        workflow.Motion = dict()

    for zone in workflow.tree.zones():
        FamilyName = zone.get(Type='FamilyName', Depth=1)
        if FamilyName and FamilyName.value() not in workflow.Motion:
            workflow.Motion[FamilyName.value()] = dict()

    for family, MotionOnFamily in workflow.Motion.items():
        update_motion_with_defaults(MotionOnFamily) 

def update_motion_with_defaults(Motion):

    RotationSpeed = Motion.setdefault('RotationSpeed', [0., 0., 0.])
    if isinstance(RotationSpeed, (int, float)):
        mola_logger.user_warning('No rotation axis for motion: set to x-axis by default.')
        Motion['RotationSpeed'] = [RotationSpeed, 0., 0.]
    Motion.setdefault('RotationAxisOrigin', [0., 0., 0.])
    Motion.setdefault('TranslationSpeed', [0., 0., 0.])

def any_mobile(MotionDictOfDicts):
    for family_name, motion_of_family in MotionDictOfDicts.items():
        if is_mobile(motion_of_family):
            return True
    return False


def is_mobile(Motion):
    return is_rotating(Motion) or is_translating(Motion)

def is_rotating(Motion):
    if 'RotationSpeed' not in Motion:
        return False

    if np.linalg.norm(Motion['RotationSpeed']) < 1e-12:
        return False
    else:
        return True

def is_translating(Motion):
    if 'TranslationSpeed' not in Motion:
        return False

    if np.linalg.norm(Motion['TranslationSpeed']) < 1e-12:
        return False
    else:
        return True
    
def all_families_are_fixed(workflow):
    if any([is_mobile(MotionOnFamily) for MotionOnFamily in workflow.Motion.values()]):
        return False
    else:
        return True
    
def get_rotation_axis_from_first_found_motion(Motion):
    RotationVector = get_first_found_rotation_speed_vector_at_motion(Motion) 
    RotationAxis = RotationVector / np.linalg.norm(RotationVector)
    return RotationAxis


def get_first_found_rotation_speed_vector_at_motion(Motion):
    for family_name, motion_of_family in Motion.items():
        if 'RotationSpeed' in motion_of_family:
            return np.array(motion_of_family['RotationSpeed'])
    raise MolaException("no RotationSpeed attribute found in Motion")


def get_first_found_rotation_axis_origin_vector_at_motion(Motion):
    for family_name, motion_of_family in Motion.items():
        if 'RotationSpeed' in motion_of_family:
            return np.array(motion_of_family['RotationAxisOrigin'])
    raise MolaException("no RotationSpeed attribute found in Motion")


