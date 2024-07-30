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

def apply(workflow):
    if not all([('Positioning' in component) for component in workflow.RawMeshComponents]):
        return
    
    tree_was_distributed = bool(workflow.tree.get(':CGNS#Distribution'))
    workflow.tree = to_partitioned_if_distributed(workflow.tree)

    warning_flag_import_Transform = False

    for base in workflow.tree.bases():
        component = workflow.get_component(base.name())
        
        if 'Positioning' not in component: continue

        for operation in component['Positioning']:
            if operation['Type'] == 'Scale':
                s = float(operation['Scale'])
                try:
                    rescale_with_maia(base, s)
                except (ImportError, AttributeError):
                    rescale_with_cassiopee(base, s)

            elif operation['Type'] == 'TranslationAndRotation':
                # TODO replace with MOLA meshing operation
                pt1 = np.array(operation['RequestedFrame']['Point'])
                pt0 = np.array(operation['InitialFrame']['Point'])
                translation = pt1 - pt0
                try:
                    translate_and_rotate_with_cassiopee(base, translation, pt1, operation['InitialFrame'], operation['RequestedFrame'])
                except (ImportError, AttributeError):
                    if operation['InitialFrame'] == operation['RequestedFrame']:
                        translate_and_rotate_with_maia(base, translation)

                    elif operation['InitialFrame'] == dict(Point=[0,0,0], Axis1=[0,0,1], Axis2=[1,0,0], Axis3=[0,1,0]) \
                        and operation['RequestedFrame'] == dict(Point=[0,0,0], Axis1=[1,0,0], Axis2=[0,1,0], Axis3=[0,0,1]):
                        translate_and_rotate_with_maia(base, translation)
                        translate_and_rotate_with_maia(base, rotation_center=pt1, rotation_angle=[0,90*np.pi/180,0])
                        translate_and_rotate_with_maia(base, rotation_center=pt1, rotation_angle=[90*np.pi/180,0,0])

                    else:
                        raise MolaException('Positioning not implemented without Cassiopee, except for rotations from Autogrid')
            
            elif operation['Type'] == 'DuplicateByRotation':
                ...
                # TODO BEWARE!! duplicate Component, and handle it properly! 

        for zone in base.zones(): 
            if cgns.castNode(zone).isStructured():
                try:
                    import Transform.PyTree as T
                    T._makeDirect(zone)
                except ModuleNotFoundError:
                    if not warning_flag_import_Transform:
                        mola_logger.warning('Cannot check that the mesh is direct after Positioning operations')
                        warning_flag_import_Transform = True # To display this warning only once

    if tree_was_distributed: workflow.tree = to_distributed(workflow.tree)

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
