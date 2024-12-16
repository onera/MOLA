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

def apply(workflow):
    '''
    Set Motion for each families
    '''
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
    if callable(Motion) or any([callable(v) for v in Motion.values()]):
        # complex motion given as a function
        return

    RotationSpeed = Motion.setdefault('RotationSpeed', [0., 0., 0.])
    if isinstance(RotationSpeed, (int, float)):
        mola_logger.warning('No rotation axis for motion: set to x-axis by default.')
        Motion['RotationSpeed'] = [RotationSpeed, 0., 0.]
    Motion.setdefault('RotationAxisOrigin', [0., 0., 0.])
    Motion.setdefault('TranslationSpeed', [0., 0., 0.])


def is_mobile(Motion):
    return is_rotating(Motion) or is_translating(Motion)

def is_rotating(Motion):
    if callable(Motion) or any([callable(v) for v in Motion.values()]):
        # complex motion given as a function
        return True
    if sum(Motion['RotationSpeed']) == 0:
        return False
    else:
        return True

def is_translating(Motion):
    if callable(Motion) or any([callable(v) for v in Motion.values()]):
        # complex motion given as a function
        return True
    if all([v==0 for v in Motion['TranslationSpeed']]):
        return False
    else:
        return True
