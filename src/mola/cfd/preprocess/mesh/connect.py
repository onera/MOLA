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

import treelab.cgns as cgns
from mola.logging import mola_logger, MolaException, MolaAssertionError

import Converter.PyTree as C
import Connector.PyTree as X

def apply(workflow):
    for base in workflow.tree.bases():
        component = workflow.get_component(base.name())
        base_name = base.name()
        base_dim = base.dim()

        if 'Connection' not in component: 
            continue
        _check_connections(component['Connection'])

        mola_logger.info(f'Connections for base {base_name}:')

        for operation in component['Connection']:
            ConnectionType = operation['Type']
            mola_logger.info(f'  > connecting type {ConnectionType}')
            try: 
                tolerance = operation['Tolerance']
            except KeyError:
                tolerance = 1e-8
                mola_logger.warning(f'    connection tolerance not defined. Using tolerance={tolerance}')
            
            if ConnectionType == 'Match':
                C._rmBCOfType(base,'BCMatch') # HACK https://elsa.onera.fr/issues/11400
                base_out = X.connectMatch(base, tol=tolerance, dim=base_dim)

            elif ConnectionType == 'NearMatch':
                try: 
                    ratio = operation['Ratio']
                except KeyError:
                    ratio = 2
                    mola_logger.warning(f'    NearMatch ratio was not defined. Using ratio={ratio}')
                base_out = X.connectNearMatch(base, ratio=ratio, tol=tolerance, dim=base_dim)

            elif ConnectionType == 'PeriodicMatch':
                rotationCenter = operation.get('RotationCenter', [0., 0., 0.])
                rotationAngle = operation.get('RotationAngle', [0., 0., 0.])
                translation = operation.get('Translation', [0., 0., 0.])
                mola_logger.info(f'    RotationCenter = {rotationCenter}')
                mola_logger.info(f'    RotationAngle = {rotationAngle}')
                mola_logger.info(f'    Translation = {translation}')

                base_out = X.connectMatchPeriodic(
                    base,
                    rotationCenter=rotationCenter,
                    rotationAngle=rotationAngle,
                    translation=translation,
                    tol=tolerance,
                    dim=base_dim
                    )
            else:
                raise MolaException(f'  Connection type {ConnectionType} not implemented')
            
            base[2] = base_out[2]

    workflow.tree = cgns.castNode(workflow.tree)
    # HACK see https://github.com/Luispain/treelab/issues/6
    # Nodes DimensionalUnits are added by X.connectMatchPeriodic
    workflow.tree.findAndRemoveNodes(Type='DimensionalUnits_t')


def _check_connections(connections):
    '''
    If there is one ConnectionType == 'Match' in **connections**, there must be only one
    and it must be the first element of the list.
    '''
    if len(connections) > 1:
        for i, connection in enumerate(connections):
            if connection['Type'] == 'Match':
                if i != 0:
                    raise MolaAssertionError("Type='Match' cannot be used after another type oc connection")
