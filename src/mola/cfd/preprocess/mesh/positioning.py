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
import Transform.PyTree as T

def apply(workflow):
    for base in workflow.tree.bases():
        component = workflow.get_component(base.name())
        
        if 'Positioning' not in component: continue

        for operation in component['Positioning']:
            if operation['Type'] == 'scale':
                s = float(operation['Scale'])
                T._homothety(base,(0,0,0),s)

            elif operation['Type'] == 'TranslationAndRotation':
                # TODO replace with MOLA meshing operation
                pt1 = np.array(operation['RequestedFrame']['Point'])
                pt0 = np.array(operation['InitialFrame']['Point'])
                translation = pt1 - pt0
                T._translate(base, tuple(translation))
                T._rotate(base,tuple(pt1),
                    ( tuple(operation['InitialFrame']['Axis1']),
                      tuple(operation['InitialFrame']['Axis2']),
                      tuple(operation['InitialFrame']['Axis3']) ),
                    ( tuple(operation['RequestedFrame']['Axis1']),
                      tuple(operation['RequestedFrame']['Axis2']),
                      tuple(operation['RequestedFrame']['Axis3']) ))

            
            elif operation['Type'] == 'DuplicateByRotation':
                ...
                # TODO BEWARE!! duplicate Component, and handle it properly! 

        for zone in base.zones(): T._makeDirect(zone)
