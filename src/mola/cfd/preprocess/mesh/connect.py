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
from mola.cfd.preprocess.mesh.tools import to_distributed

def apply(workflow):

    _clip_small_rotation_angles(workflow.tree)

    if not any([('Connection' in component) 
                and len(component['Connection'])>0 
                for component in workflow.RawMeshComponents]):
        return

    mola_logger.info("  🔗 connecting mesh", rank=0)
    
    reason_for_not_using_maia = get_reason_why_maia_cannot_connect(workflow)
    use_maia = not bool(reason_for_not_using_maia)

    if use_maia:
        apply_with_maia(workflow)

    else:
        try:
            apply_with_cassiopee(workflow)
        except BaseException as e:
            reason_for_not_using_cassiopee = str(e)
            msg =('Unable to connect:\n'
                 f'maia reason: {reason_for_not_using_maia}\n'
                 f'cassiopee reason: {reason_for_not_using_cassiopee}')
            raise MolaException(msg) from e

def _clip_small_rotation_angles(tree, tol=1e-12):
    for perio in tree.group(Type='Periodic'):
        RotationAngle = perio.get(Name='RotationAngle').value()
        for i, angle in enumerate(RotationAngle):
            if abs(angle) < tol:
                RotationAngle[i] = 0. 

def apply_with_cassiopee(workflow):

    mola_logger.debug('connect with cassiopee')

    from mpi4py import MPI
    mpi_size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    if mpi_size > 1:
        from mola.cfd.preprocess.mesh.tools import (to_partitioned_if_distributed,
                                                    to_full_tree_at_rank_0)
        workflow.tree = to_partitioned_if_distributed(workflow.tree) # TODO when https://elsa.onera.fr/issues/11700 fixed
        # workflow.tree = to_full_tree_at_rank_0(workflow.tree)


    import Converter.PyTree as C
    import Connector.PyTree as X
    import Connector.Mpi as Xmpi
    import Converter.Internal as I

    for base in workflow.tree.bases():
        component = workflow.get_component(base.name())
        base_name = base.name()
        base_dim = base.dim()

        if 'Connection' not in component: continue
        # _check_connections(component['Connection'])
        component['Connection'] = _reorder_connections(component['Connection'])
        I._adaptPE2NFace(base)  # For NGon mesh, generate NGonFace nodes if they don't exist using ParentElements n

        mola_logger.info(f'   - connections for base {base_name}:', rank=0)

        for operation in component['Connection']:
            # if mpi_size > 1: raise MolaException('unable to connect mesh using MPI parallel mode and Cassiopee')
            ConnectionType = operation['Type']
            try: 
                tolerance = operation['Tolerance']
            except KeyError:
                tolerance = component['DefaultToleranceForConnection']
                mola_logger.user_warning(f'    connection tolerance not defined. Using tolerance={tolerance}')
            
            if ConnectionType == 'Match':
                C._rmBCOfType(base,'BCMatch') # HACK https://elsa.onera.fr/issues/11400
                # HACK Xmpi.connectMatch works only for structured mesh, 
                # whereas X.connectMatch works also for unstructured mesh.
                # See https://elsa-e.onera.fr/issues/11719
                mola_logger.info(f'    > connecting type {ConnectionType}', rank=0)
                if mpi_size == 1:
                    base_out = X.connectMatch(base, tol=tolerance, dim=base_dim)
                elif base.isStructured():
                    base_out = Xmpi.connectMatch(base, tol=tolerance, dim=base_dim) 
                else:
                    raise MolaAssertionError('connectMatch in parallel works only for structured mesh.')

            elif ConnectionType == 'NearMatch':
                mola_logger.info(f'    > connecting type {ConnectionType}', rank=0)
                try: 
                    ratio = operation['Ratio']
                except KeyError:
                    ratio = 2
                    mola_logger.user_warning(f'    NearMatch ratio was not defined. Using ratio={ratio}')
                base_out = Xmpi.connectNearMatch(base, ratio=ratio, tol=tolerance, dim=base_dim)

            elif ConnectionType == 'PeriodicMatch':
                rotation_center = operation.get('RotationCenter', [0., 0., 0.])
                rotation_angle = operation.get('RotationAngle', [0., 0., 0.])
                translation = operation.get('Translation', [0., 0., 0.])

                msg = ''
                if not np.allclose(rotation_angle, 0.):
                    msg += f' with rotation angle of {rotation_angle} degrees'
                if not np.allclose(rotation_center, 0.):
                    msg += f' around point {rotation_center}'
                if not np.allclose(translation, 0.):
                    msg += f' with translation of {translation} meters'
                if len(msg) == 0:
                    # nothing to do!
                    continue
                mola_logger.info(f'    > connecting type {ConnectionType}{msg}', rank=0)

                if 'Families' in operation:
                    # Remove BC attached to periodic Families if they exists (only needed for maia)
                    # Needed here if the connection is used BEFORE process_mesh for SoNICS
                    for family in operation['Families']:
                        mola_logger.debug(f'remove Family {family}')
                        for bc_node in C.getFamilyBCs(base, family):
                            I._rmNode(base, bc_node)
                        if family_node := I.getNodeFromName1(base, family):
                            I._rmNode(base, family_node)

                if mpi_size > 1:
                    msg = ('cannot make periodic match using Cassiopee and MPI parallel execution:\n'
                           'https://elsa.onera.fr/issues/11706')
                    raise MolaException(msg)
                base_out = X.connectMatchPeriodic(
                    base,
                    rotationCenter=rotation_center,
                    rotationAngle=rotation_angle,
                    translation=translation,
                    tol=tolerance,
                    dim=base_dim
                    )
            else:
                raise MolaException(f'  Connection type {ConnectionType} not implemented')
            
            base[2] = base_out[2]
            
    # try:
    #     from maia.io.fix_tree import fix_point_ranges
    #     if rank==0:print("\033[93m", end='')
    #     fix_point_ranges(workflow.tree)
    #     if rank==0:print("\033[0m", end='')
    # except ModuleNotFoundError:
    #     mola_logger.warning("could not import maia, will not fix PointRange")

    workflow.tree = cgns.castNode(workflow.tree)

def apply_with_maia(workflow):
    mola_logger.debug('connect with maia')
    workflow.tree = to_distributed(workflow.tree)

    component = workflow.RawMeshComponents[0] # CAVEAT this prevents from connecting multiple raw mesh components using maia
    for operation in component['Connection']:
        ConnectionType = operation['Type']
        mola_logger.info(f'    > connecting type {ConnectionType}', rank=0)
            
        if ConnectionType == 'PeriodicMatch':
            rotation_center = operation.get('RotationCenter', [0., 0., 0.])
            rotation_angle = operation.get('RotationAngle', [0., 0., 0.])
            translation = operation.get('Translation', [0., 0., 0.])

            msg = ''
            if not np.allclose(rotation_angle, 0.):
                msg += f' with rotation angle of {rotation_angle} degrees'
            if not np.allclose(rotation_center, 0.):
                msg += f' around point {rotation_center}'
            if not np.allclose(translation, 0.):
                msg += f' with translation of {translation} meters'
            if len(msg) == 0:
                # nothing to do!
                continue
            mola_logger.info(f'    > connecting type {ConnectionType}{msg}', rank=0)

            # Work only on a top Tree, not on a Base
            connect_periodic_with_maia(workflow.tree, operation['Families'], rotation_center, rotation_angle, translation)

        else:
            raise MolaException(f'  Connection type {ConnectionType} not implemented')
    
    workflow.tree = cgns.castNode(workflow.tree)

def get_reason_why_maia_cannot_connect(workflow):

    if workflow.Solver != 'sonics':
        return f'Maia is not used with {workflow.Solver}'
    
    if not workflow.tree.isUnstructured():
        return 'Periodic Match with Maia is possible only for unstructured mesh'

    for component in workflow.RawMeshComponents:
        for connection in component['Connection']:
            if connection['Type'] != 'PeriodicMatch':
                return 'Connection operations are possible only for Type PeriodicMatch without Cassiopee.'
            elif not 'Families' in connection:
                return 'PeriodicMatch with Maia needs Families.'
            elif not len(connection['Families']) == 2:
                return 'Families must be a tuple of length 2.'
            elif not any([workflow.tree.get(Type='Family', Depth=2, Name=fam) for fam in connection['Families']]):
                return (f'Families {connection["Families"]}, '
                    'needed to perform PeriodicMatch operation with Maia, '
                    'cannot be found in the mesh tree.')

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
                
def _reorder_connections(connections):
    '''
    If there is one ConnectionType == 'Match' in **connections**, there must be only one
    and it must be the first element of the list.
    '''
    reordered_connections = []
    connect_match = None
    for connection in connections:
        if connection['Type'] != 'Match':
            reordered_connections.append(connection)
        else:
            connect_match = connection
    
    if connect_match:
        reordered_connections.insert(0, connect_match)

    return reordered_connections

def connect_periodic_with_maia(tree, families, rotation_center, rotation_angle, translation):
    # tolerance is relative with maia, to 0.01 by default
    # TODO Should be replace by a function from Miles

    import maia
    from mpi4py import MPI

    def _check_unmatched_faces(tree):
        unmatched_gc = maia.pytree.get_nodes_from_name(tree, '*_unmatched')
        if len(unmatched_gc) > 0:
            maia.io.dist_tree_to_file(tree, 'debug_bad_geometry.cgns', MPI.COMM_WORLD)
            raise MolaException(f"Bad geometry (check mesh or tolerance)")

    periodic = dict(
        rotation_angle = np.radians(rotation_angle),
        rotation_center = np.array(rotation_center),
        translation = np.array(translation),
    )

    try:
        maia.algo.dist.connect_1to1_families(
            tree,
            families,
            comm=MPI.COMM_WORLD,
            periodic=periodic, 
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
            )
            _check_unmatched_faces(tree)
        except ZeroDivisionError:
            raise MolaException(f"No Periodic match found. Check translation or rotation input data.")
        
    # HACK in SoNICS: for now we need to remove FamilyName in GridConnectivity nodes
    # otherwise there is a bug in SoNICS
    import maia.pytree as PT
    gc_in_families = lambda n: PT.get_label(n) == 'GridConnectivity_t' \
        and PT.get_node_from_label(n, 'FamilyName_t') \
        and PT.get_value(PT.get_node_from_label(n, 'FamilyName_t')) in families
    for node in maia.pytree.get_nodes_from_predicate(tree, gc_in_families):
        maia.pytree.rm_nodes_from_label(node, 'FamilyName_t')
    
