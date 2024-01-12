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

from .motion import is_mobile

def adapt_to_solver(workflow):
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

        # TODO Test if zones in that family are modelled with Body Force
        # If yes, they must be with a rotation speed equal to zero

        if not is_mobile(MotionOnFamily):
            continue

        # For elsA, the rotation must be around one axis only
        onlyOneRotationComponent = \
            (MotionOnFamily['RotationSpeed'][0] == MotionOnFamily['RotationSpeed'][1] == 0) \
            or (MotionOnFamily['RotationSpeed'][0] == MotionOnFamily['RotationSpeed'][2] == 0) \
            or (MotionOnFamily['RotationSpeed'][1] == MotionOnFamily['RotationSpeed'][2] == 0)
        
        assert onlyOneRotationComponent, 'For elsA, the rotation must be around one axis only'
        omega = sum(MotionOnFamily['RotationSpeed'])
        
        print(f'setting .Solver#Motion at family {family} (omega={omega}rad/s)')
        famNode.setParameters('.Solver#Motion',
                                motion='mobile',
                                omega=omega,
                                axis_pnt_x=MotionOnFamily['RotationAxisOrigin'][0], 
                                axis_pnt_y=MotionOnFamily['RotationAxisOrigin'][1], 
                                axis_pnt_z=MotionOnFamily['RotationAxisOrigin'][2],
                                axis_vct_x=MotionOnFamily['TranslationSpeed'][0], 
                                axis_vct_y=MotionOnFamily['TranslationSpeed'][1], 
                                axis_vct_z=MotionOnFamily['TranslationSpeed'][2]
                                )
 
    


        