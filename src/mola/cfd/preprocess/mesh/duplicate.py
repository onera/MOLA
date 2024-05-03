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
from mola.server import MaiaParallel
from mola.logging import mola_logger

def duplicate_workflow_with_cassiopee(workflow):
    '''
    Duplicated the input PyTree **t**, already initialized.
    This function perform the following operations:

    #. Duplicate the mesh

    #. Initialize the different blade sectors by rotating the ``FlowSolution#Init``
       node available in the original sector(s)

    #. Update connectivities and periodic boundary conditions

    .. warning:: This function does not rotate vectors in BCDataSet nodes.

    Parameters
    ----------

        t : PyTree
            input tree already initialized, but before setting boundary conditions

        TurboConfiguration : dict
            dictionary as provided by :py:func:`getTurboConfiguration`

    Returns
    -------

        t : PyTree
            tree after duplication
    '''
    import Converter.Internal as I
    import Connector.PyTree as X

    t = workflow.tree

    # Remove connectivities and periodic BCs
    I._rmNodesByType(t, 'GridConnectivity1to1_t')

    angles4ConnectMatchPeriodic = []
    for row, rowParams in workflow.ApplicationContext['Rows'].items():
        nBlades = rowParams['NumberOfBlades']
        nDupli = rowParams['NumberOfBladesSimulated']
        nMesh = rowParams['NumberOfBladesInInitialMesh']
        if nDupli > nMesh:
            duplicate_with_cassiopee(t, row, nBlades, nDupli=nDupli, axis=(1,0,0))

        angle = 360. / nBlades * nDupli
        if not np.isclose(angle, 360.):
            angles4ConnectMatchPeriodic.append(angle)

    # Connectivities
    X.connectMatch(t, tol=1e-8)
    for angle in angles4ConnectMatchPeriodic:
        # Not full 360 simulation: periodic BC must be restored
        t = X.connectMatchPeriodic(t, rotationAngle=[angle, 0., 0.], tol=1e-8)

    # WARNING: Names of BC_t nodes must be unique to use PyPart on globborders
    for l in [2,3,4]: I._correctPyTree(t, level=l)

    workflow.tree = cgns.castNode(t)

def duplicate_with_cassiopee(tree, rowFamily, nBlades, nDupli=None, merge=False, axis=(1,0,0),
    verbose=1, container='FlowSolution#Init',
    vectors2rotate=[['VelocityX','VelocityY','VelocityZ'],['MomentumX','MomentumY','MomentumZ']]):
    '''
    Duplicate **nDupli** times the domain attached to the family **rowFamily**
    around the axis of rotation.

    Parameters
    ----------

        tree : PyTree
            tree to modify

        rowFamily : str
            Name of the CGNS family attached to the row domain to Duplicate

        nBlades : int
            Number of blades in the row. Used to compute the azimuthal length of
            a blade sector.

        nDupli : int
            Number of duplications to make

            .. warning:: This is the number of duplication of the input mesh
                domain, not the wished number of simulated blades. Keep this
                point in mind if there is already more than one blade in the
                input mesh.

        merge : bool
            if :py:obj:`True`, merge all the blocks resulting from the
            duplication.

            .. tip:: This option is useful is the mesh is to split and if a
                globborder will be defined on a BC of the duplicated domain. It
                allows the splitting procedure to provide a 'matricial' ordering
                (see `elsA Tutorial about globborder <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/globborder.html>`_)

        axis : tuple
            axis of rotation given as a 3-tuple of integers or floats

        verbose : int
            level of verbosity:

                * 0: no print

                * 1: print the number of duplications for row **rowFamily** and
                  the total number of blades.

                * 2: print also the name of all duplicated zones

        container : str
            Name of the FlowSolution container to rotate. Default is 'FlowSolution#Init'

        vectors2rotate : :py:class:`list` of :py:class:`list` of :py:class:`str`
            list of vectors to rotate. Each vector is a list of three strings,
            corresponding to each components.
            The default value is:

            >>> vectors2rotate = [['VelocityX','VelocityY','VelocityZ'],
            >>>                   ['MomentumX','MomentumY','MomentumZ']]

            .. note:: 
            
                Rotation of vectors is done with Cassiopee function Transform.rotate. 
                However, it is not useful to put the prefix 'centers:'. It will be 
                added automatically in the function.

    '''
    import Converter.Internal as I
    import Transform.PyTree as T

    OLD_FlowSolutionCenters = I.__FlowSolutionCenters__
    I.__FlowSolutionCenters__ = container

    if nDupli is None:
        nDupli = nBlades # for a 360 configuration
    if nDupli == nBlades:
        if verbose>0: print('Duplicate {} over 360 degrees ({} blades in row)'.format(rowFamily, nBlades))
    else:
        if verbose>0: print('Duplicate {} on {} blades ({} blades in row)'.format(rowFamily, nDupli, nBlades))

    check = False
    vectors = []
    for vec in vectors2rotate:
        vectors.append(vec)
        vectors.append(['centers:'+v for v in vec])

    if I.getType(tree) == 'CGNSBase_t':
        bases = [tree]
    else:
        bases = I.getBases(tree)

    for base in bases:
        for zone in I.getZones(base):
            zone_name = I.getName(zone)
            FamilyNameNode = I.getNodeFromName1(zone, 'FamilyName')
            if not FamilyNameNode: continue
            zone_family = I.getValue(FamilyNameNode)
            if zone_family == rowFamily:
                if verbose>1: print('  > zone {}'.format(zone_name))
                check = True
                zones2merge = [zone]
                for n in range(nDupli-1):
                    ang = 360./nBlades*(n+1)
                    rot = T.rotate(I.copyNode(zone),(0.,0.,0.), axis, ang, vectors=vectors)
                    I.setName(rot, "{}_{}".format(zone_name, n+2))
                    I._addChild(base, rot)
                    zones2merge.append(rot)
                if merge:
                    for node in zones2merge:
                        I.rmNode(base, node)
                    tree_dist = T.merge(zones2merge, tol=1e-8)
                    for i, node in enumerate(I.getZones(tree_dist)):
                        I._addChild(base, node)
                        disk_block = I.getNodeFromName(base, I.getName(node))
                        disk_block[0] = '{}_{:02d}'.format(zone_name, i)
                        I.createChild(disk_block, 'FamilyName', 'FamilyName_t', value=rowFamily)
    if merge: PRE.autoMergeBCs(tree)

    I.__FlowSolutionCenters__ = OLD_FlowSolutionCenters
    assert check, 'None of the zones was duplicated. Check the name of row family'

def duplicate_workflow_with_maia(workflow):
    duplication_parameters = dict()
    for row, rowParams in workflow.ApplicationContext['Rows'].items():
        duplication_parameters[row] = dict(
            number_of_duplications = rowParams['NumberOfBladesSimulated'] - rowParams['NumberOfBladesInInitialMesh'],
            is_360 = rowParams['NumberOfBladesSimulated'] == rowParams['NumberOfBlades']
        )

    if any([p['number_of_duplications']>0 for p in duplication_parameters.values()]):
        mola_logger.info('Duplication:')
    
    workflow.tree = duplicate_with_maia(workflow.tree, duplication_parameters, merge_zones=workflow.tree.isUnstructured())

@MaiaParallel
def duplicate_with_maia(dist_tree, duplication_parameters, merge_zones=False):
    import maia
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # TODO use the following lines when the env uses Maia v>1.3
    # for row, dup_params in duplication_parameters.items():
    #     if dup_params['number_of_duplications'] == 0:
    #         continue
    #     elif dup_params['is_360']:
    #         mola_logger.info(f"  > row {row} is replicated on 360 degrees")
    #         maia.algo.dist.duplicate_family_from_rotation_jns_to_360(dist_tree, row, comm)
    #     else:
    #         plurial = 's' if dup_params['number_of_duplications'] > 1 else ''
    #         mola_logger.info(f"  > row {row} is replicated {dup_params['number_of_duplications']} time"+plurial)
    #         maia.algo.dist.duplicate_family_from_periodic_jns(dist_tree, row, dup_params['number_of_duplications'], comm)
        
    import maia.pytree as PT
    for row, dup_params in duplication_parameters.items():
        if dup_params['number_of_duplications'] == 0:
            continue

        is_zone_in_row = lambda n : PT.get_label(n) == 'Zone_t' and PT.predicate.belongs_to_family(n, row)
        zones_paths = PT.predicates_to_paths(dist_tree, ['CGNSBase_t', is_zone_in_row])

        _, perio_jns = PT.find_periodic_jns(dist_tree)

        if dup_params['is_360']:
            mola_logger.info(f"  > row {row} is replicated on 360 degrees")
            maia.algo.dist.duplicate_from_rotation_jns_to_360(
                dist_tree, 
                zones_paths, 
                perio_jns, 
                comm, 
                apply_to_fields=True  # TODO that becomes the default value with maia v1.4
                )
        else:
            plurial = 's' if dup_params['number_of_duplications'] > 1 else ''
            mola_logger.info(f"  > row {row} is replicated {dup_params['number_of_duplications']} time"+plurial)
            maia.algo.dist.duplicate_from_periodic_jns(
                dist_tree, 
                zones_paths, 
                perio_jns, 
                dup_params['number_of_duplications'], 
                comm, 
                apply_to_fields=True  # TODO that becomes the default value with maia v1.4
                )
    
    if merge_zones:
        maia.algo.dist.merge_connected_zones(dist_tree, comm)    

    return dist_tree

