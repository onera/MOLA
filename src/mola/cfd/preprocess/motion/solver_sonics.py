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

from treelab import cgns
from mola.logging import mola_logger, MolaException
from mola.cfd.preprocess.motion.motion import all_families_are_fixed

def apply_to_solver(workflow):
    '''
    Set Motion for each families for the solver SoNICS.

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

    import miles

    if all_families_are_fixed(workflow):
        return

    # HACK error miles: Target Fluid matched invalid CGNS node(s): 'WorkflowParameters/Fluid'.
    WorkflowParameters = workflow.tree.get(Name='WorkflowParameters', Depth=1)
    WorkflowParameters.dettach()
  
    for family, MotionOnFamily in workflow.Motion.items():
        mola_logger.debug(f'set motion on {family}: {MotionOnFamily}')

        motion = translate_motion_to_sonics(MotionOnFamily)
        miles.set_motion(workflow.tree, family, **motion)

    workflow.tree = cgns.castNode(workflow.tree)
    WorkflowParameters.attachTo(workflow.tree)


def translate_motion_to_sonics(Motion, tree=None):
    import miles
    from mpi4py import MPI

    if callable(Motion) or \
        any([callable(v) for k, v in Motion.items() if k != 'RotationSpeed']):
        raise MolaException('Cannot translate a function (except for RotationSpeed)')
    
    if callable(Motion['RotationSpeed']):
        if tree is None:
            raise MolaException('tree must be provided to interpolate motion')
                          
        motion_sonics = dict(
            RotationSpeedX = miles.FunctionInterpolator(tree, Motion['RotationSpeed'], comm=MPI.COMM_WORLD),
            RotationSpeedY = 0., 
            RotationSpeedZ = 0., 
        )

    else:
        motion_sonics = dict(
            RotationSpeedX = Motion['RotationSpeed'][0],
            RotationSpeedY = Motion['RotationSpeed'][1], 
            RotationSpeedZ = Motion['RotationSpeed'][2], 
        )


    motion_sonics.update(dict(
        RotationAxisOriginX = Motion['RotationAxisOrigin'][0],
        RotationAxisOriginY = Motion['RotationAxisOrigin'][1], 
        RotationAxisOriginZ = Motion['RotationAxisOrigin'][2], 
        TranslationSpeedX = Motion['TranslationSpeed'][0], 
        TranslationSpeedY = Motion['TranslationSpeed'][1],
        TranslationSpeedZ = Motion['TranslationSpeed'][2],
    ))

    return motion_sonics
        