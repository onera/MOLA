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
from mola.logging import mola_logger

def apply_to_solver(workflow):
    '''
    Set Motion for each families for the solver elsA.

    The **workflow** must have a **Motion** attribute like this:

    .. code-block:: python
        Motion = dict(
            Rotor = dict(
                RotationSpeed = [1000., 0., 0.],
                RotationAxisOrigin = [0., 0., 0.],
                TranslationSpeed = [0., 0., 0.]
            )
        )

    Parameters
    ----------

        workflow : Workflow object

    '''
  
    for family, MotionOnFamily in workflow.Motion.items():
        famNode = workflow.tree.get(Name=family, Type='Family', Depth=2)

        if not motion.is_mobile(MotionOnFamily):
            continue
        assert_rotation_axis_is_correct(MotionOnFamily)

        mola_logger.debug(f'set motion on {family}: {MotionOnFamily}')

        famNode.setParameters('.Solver#Motion',
                                motion='mobile',
                                **translate_motion_to_elsa(MotionOnFamily)
                                )
 
def assert_rotation_axis_is_correct(Motion):
    # For elsA, the rotation must be around one axis only
    onlyOneRotationComponent = \
        (Motion['RotationSpeed'][0] == Motion['RotationSpeed'][1] == 0) \
    or (Motion['RotationSpeed'][0] == Motion['RotationSpeed'][2] == 0) \
    or (Motion['RotationSpeed'][1] == Motion['RotationSpeed'][2] == 0)
    
    assert onlyOneRotationComponent, 'For elsA, the rotation must be around one axis only'    

def translate_motion_to_elsa(Motion):
    if callable(Motion) or any([callable(v) for v in Motion.values()]):
        raise Exception('Cannot translate a function')
    
    RotationAxis = np.array(Motion['RotationSpeed'])
    RotationSpeed = np.sqrt(RotationAxis.dot(RotationAxis))
    RotationAxis = np.absolute(RotationAxis) / RotationSpeed

    TranslationVector = np.array(Motion['TranslationSpeed'])
    TranslationSpeed = np.sqrt(TranslationVector.dot(TranslationVector))
    if TranslationSpeed != 0:
        TranslationVector /= TranslationSpeed
    else:
        TranslationVector = [1., 0., 0.]

    motion_elsa = dict(
        omega        = RotationSpeed,
        axis_pnt_x   = Motion['RotationAxisOrigin'][0], 
        axis_pnt_y   = Motion['RotationAxisOrigin'][1], 
        axis_pnt_z   = Motion['RotationAxisOrigin'][2],
        axis_vct_x   = RotationAxis[0], 
        axis_vct_y   = RotationAxis[1], 
        axis_vct_z   = RotationAxis[2], 
        transl_vct_x = TranslationVector[0],
        transl_vct_y = TranslationVector[1],
        transl_vct_z = TranslationVector[2],
        transl_speed = TranslationSpeed,
    )

    return motion_elsa
        