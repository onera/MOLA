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

import treelab.cgns as cgns
from mola.logging import mola_logger, MolaException, MolaAssertionError
from mola.server import MaiaParallel

def apply(workflow):
    if not all([('Connection' in component) for component in workflow.RawMeshComponents]):
        return
    
    apply_with_cassiopee(workflow)
        
def apply_with_cassiopee(workflow):

    from mpi4py import MPI
    mpi_size = MPI.COMM_WORLD.Get_size()

    import Converter.PyTree as C
    import Connector.PyTree as X

    for base in workflow.tree.bases():
        component = workflow.get_component(base.name())
        base_name = base.name()
        base_dim = base.dim()

        if 'Connection' not in component: continue
        _check_connections(component['Connection'])

        mola_logger.info(f'Connections for base {base_name}:')

        for operation in component['Connection']:
            if mpi_size > 1:
                raise MolaException('unable to connect mesh using MPI parallel mode and Cassiopee')
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
                mola_logger.debug(f'    RotationCenter = {rotationCenter}')
                mola_logger.debug(f'    RotationAngle = {rotationAngle}')
                mola_logger.debug(f'    Translation = {translation}')

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

def apply_with_maia(workflow):
    component = workflow.RawMeshComponents[0]
    for operation in component['Connection']:
        ConnectionType = operation['Type']
        mola_logger.info(f'  > connecting type {ConnectionType}')
        try: 
            tolerance = operation['Tolerance']
        except KeyError:
            tolerance = 1e-8
            mola_logger.warning(f'    connection tolerance not defined. Using tolerance={tolerance}')
            
        if ConnectionType == 'PeriodicMatch':
            rotation_center = operation.get('RotationCenter', [0., 0., 0.])
            rotation_angle = operation.get('RotationAngle', [0., 0., 0.])
            translation = operation.get('Translation', [0., 0., 0.])
            mola_logger.debug(f'    RotationCenter = {rotation_center}')
            mola_logger.debug(f'    RotationAngle = {rotation_angle}')
            mola_logger.debug(f'    Translation = {translation}')
            # Work only on a top Tree, not on a Base
            connect_periodic_with_maia(workflow.tree, operation['Families'], rotation_center, rotation_angle, translation, tolerance)

        else:
            raise MolaException(f'  Connection type {ConnectionType} not implemented')
    
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


@MaiaParallel
def connect_periodic_with_maia(tree, families, rotation_center, rotation_angle, translation, tol):
    # TODO Should be replace by a function from Miles

    import maia
    from mpi4py import MPI

    def _check_unmatched_faces(tree):
        unmatched_gc = maia.pytree.get_nodes_from_name(tree, '*_unmatched')
        if len(unmatched_gc) > 0:
            raise MolaException(f"Bad geometry (check mesh or tolerance)")

    periodic = dict(
        rotation_angle = np.array(rotation_angle)*np.pi/180.,
        rotation_center = np.array(rotation_center),
        translation = np.array(translation),
    )

    try:
        maia.algo.dist.connect_1to1_families(
            tree,
            families,
            comm=MPI.COMM_WORLD,
            periodic=periodic, 
            tol=tol
        )
        _check_unmatched_faces(tree)
    except ZeroDivisionError:
        # TODO Put this message in debug log
        mola_logger.warning('No Periodic match found. Testing reverting families...')
        # Second test reverting families
        try:
            maia.algo.dist.connect_1to1_families(
                tree,
                families[::-1],
                comm=MPI.COMM_WORLD,
                periodic=periodic, 
                tol=tol
            )
            _check_unmatched_faces(tree)
        except ZeroDivisionError:
            raise MolaException(f"No Periodic match found. Check translation or rotation input data.")
    
