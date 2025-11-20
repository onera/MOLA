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
from treelab import cgns
from mola.logging import mola_logger, MolaException
from .tools import (to_partitioned_if_distributed,
                    to_distributed)
from mola.cfd.preprocess.mesh.duplicate import duplicate

def apply(workflow):
    if not all([('Positioning' in component) for component in workflow.RawMeshComponents]):
        return
    
    mola_logger.info("  👇 positioning mesh", rank=0)
    
    warning_flag_import_Transform = False

    duplication_operations = []

    for base in workflow.tree.bases():
        component = workflow.get_component(base.name())
        
        if 'Positioning' not in component: continue

        for operation in component['Positioning']:
            if operation['Type'] == 'Scale':
                s = float(operation['Scale'])
                mola_logger.info(f"    - rescaling component {component['Name']} with factor {s}", rank=0)
                try:
                    rescale_with_maia(base, s)
                except (ImportError, AttributeError):
                    rescale_with_cassiopee(base, s)

            elif operation['Type'] == 'TranslationAndRotation':
                # TODO replace with MOLA meshing operation
                pt1 = np.array(operation['RequestedFrame']['Point'])
                pt0 = np.array(operation['InitialFrame']['Point'])
                translation = pt1 - pt0
                if np.any(translation != 0): 
                    mola_logger.info(f"    - translating component {component['Name']} with vector {translation}", rank=0)
                if np.any(operation['InitialFrame'] != operation['RequestedFrame']):
                    mola_logger.info(f"    - rotate component {component['Name']} around point {pt1} to transform frame {_pretty_print_frame(operation['InitialFrame'])} into {_pretty_print_frame(operation['RequestedFrame'])}", rank=0)
                try:
                    if operation['InitialFrame'] == operation['RequestedFrame']:
                        translate_and_rotate_with_maia(base, translation)

                    elif operation['InitialFrame'] == dict(Point=[0,0,0], Axis1=[0,0,1], Axis2=[1,0,0], Axis3=[0,1,0]) \
                        and operation['RequestedFrame'] == dict(Point=[0,0,0], Axis1=[1,0,0], Axis2=[0,1,0], Axis3=[0,0,1]):
                        translate_and_rotate_with_maia(base, translation)
                        translate_and_rotate_with_maia(base, rotation_center=pt1, rotation_angle=[0,90*np.pi/180,0])
                        translate_and_rotate_with_maia(base, rotation_center=pt1, rotation_angle=[90*np.pi/180,0,0])

                    else:
                        raise MolaException('Positioning not implemented without Cassiopee, except for rotations from Autogrid')

                except (ImportError, AttributeError, MolaException):
                    translate_and_rotate_with_cassiopee(base, translation, pt1, operation['InitialFrame'], operation['RequestedFrame'])
                    
            # elif operation['Type'] == 'DuplicateByRotation':
            #     duplication_operations.append(operation)
        
        for zone in base.zones(): 
            try:
                import Transform.PyTree as T
                T._makeDirect(zone)
            except ModuleNotFoundError:
                if not warning_flag_import_Transform:
                    mola_logger.warning('Cannot check that the mesh is direct after Positioning operations')
                    warning_flag_import_Transform = True # To display this warning only once
    
    # if len(duplication_operations) > 0:   # it is done in a separated method of workflow, because it needs connectivities and families to be correctly done
    #     workflow.tree = duplicate(workflow.tree, duplication_operations)


def rescale_with_cassiopee(t, scale):
    import Transform.PyTree as T
    T._homothety(t, (0,0,0), scale)

def rescale_with_maia(t, scale):
    import maia
    maia.algo.scale_mesh(t, scale)

def translate_and_rotate_with_cassiopee(t, translation, center, InitialFrame, RequestedFrame):
    import Transform.PyTree as T
    T._translate(t, tuple(translation))
    T._rotate(t, tuple(center),
        ( tuple(InitialFrame['Axis1']),
          tuple(InitialFrame['Axis2']),
          tuple(InitialFrame['Axis3']) ),
        ( tuple(RequestedFrame['Axis1']),
          tuple(RequestedFrame['Axis2']),
          tuple(RequestedFrame['Axis3']) ))

def translate_and_rotate_with_maia(t, translation=[0,0,0], rotation_center=[0,0,0], rotation_angle=[0,0,0]):
    import maia
    maia.algo.transform_affine(
        t, 
        translation=translation, 
        rotation_center=rotation_center, 
        rotation_angle=rotation_angle
        )

def _pretty_print_frame(frame):
    def _pretty_axis(axis):
        if np.allclose(axis, [1,0,0]):
            return 'x'
        elif np.allclose(axis, [0,1,0]):
            return 'y'
        elif np.allclose(axis, [0,0,1]):
            return 'z'
        else: 
            return str(axis)
    frame_to_print = f"({_pretty_axis(frame['Axis1'])},{_pretty_axis(frame['Axis2'])},{_pretty_axis(frame['Axis3'])})"
    return frame_to_print