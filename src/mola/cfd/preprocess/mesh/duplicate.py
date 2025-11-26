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
from mola.logging import mola_logger, MolaException, redirect_streams_to_logger
from mola.cfd.preprocess.mesh.tools import to_distributed, to_full_tree_at_rank_0, compute_azimuthal_extension

def apply(workflow):

    dup_tree = apply_duplication_on_tree(workflow, workflow.tree)
    workflow.tree = dup_tree 

def apply_duplication_on_tree(workflow, tree):
    duplication_operations = []
    for base in tree.bases():
        component = workflow.get_component(base.name())
        if 'Positioning' not in component: 
            continue
        for operation in component['Positioning']:
            if operation['Type'] == 'DuplicateByRotation':
                operation.setdefault('Tolerance', component['DefaultToleranceForConnection'])
                duplication_operations.append(operation)
        
    if len(duplication_operations) > 0:
        mola_logger.info("  ➕ duplicate mesh", rank=0)
        if workflow.SplittingAndDistribution['Strategy'].lower() == 'atpreprocess':
            raise MolaException('Only SplittingAndDistribution Strategy "AtComputation" is compatible with duplication')
        
        tool = 'maia' if workflow.Solver == 'sonics' else 'cassiopee'

        tree = duplicate(tree, duplication_operations, workflow, tool=tool)
    return tree

def duplicate(tree, duplication_operations, workflow, tool='maia'):
    
    with redirect_streams_to_logger(mola_logger, stdout_level='DEBUG'):
        tree.forceShorterZoneNames(max_length=20)

    if tool == 'cassiopee':
        mola_logger.debug('duplicate with cassiopee')
        tree = _duplicate_with_cassiopee(tree, duplication_operations)
        workflow.connect()
    else:
        mola_logger.debug('duplicate with maia')
        tree = _duplicate_with_maia(tree, duplication_operations)

    return tree

def _duplicate_with_cassiopee(tree, duplication_operations):
    '''
    Duplicated the input PyTree **tree**, already initialized.
    This function perform the following operations:

    #. Duplicate the mesh

    #. Initialize the different blade sectors by rotating the ``FlowSolution#Init``
       node available in the original sector(s)

    #. Update connectivities and periodic boundary conditions

    .. warning:: This function does not rotate vectors in BCDataSet nodes.
    '''
    import Converter.Internal as I
    import Connector.PyTree as X

    angles4ConnectMatchPeriodic = []
    for operation in duplication_operations:
        Family = operation['Family']
        NumberOfDuplications = operation['NumberOfDuplications']
        # do this before removing connectivities, because method="from_periodic" uses Periodic nodes
        azimuthal_extension = np.degrees(compute_azimuthal_extension(tree, Family))  # Cassiopee uses degrees by default
        operation['azimuthal_extension'] = azimuthal_extension

        angle = azimuthal_extension * (NumberOfDuplications+1)
        if not np.isclose(angle, 360.):
            angles4ConnectMatchPeriodic.append(angle)

    # Remove connectivities and periodic BCs
    I._rmNodesByType(tree, 'GridConnectivity1to1_t')

    for operation in duplication_operations:

        plurial = 's' if operation['NumberOfDuplications'] > 1 else ''
        mola_logger.info(f"  > row {operation['Family']} is replicated {operation['NumberOfDuplications']} time"+plurial, rank=0)

        __duplicate_with_cassiopee(
            tree, 
            operation['Family'], 
            operation['NumberOfDuplications'], 
            operation['azimuthal_extension'], 
            axis=(1,0,0)
            )

    # WARNING: Names of BC_t nodes must be unique to use PyPart on globborders
    for l in [2,3,4]: I._correctPyTree(tree, level=l)

    return cgns.castNode(tree)

def __duplicate_with_cassiopee(tree, rowFamily, NumberOfDuplications, azimuthal_extension, merge=False, axis=(1,0,0),
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
    mola_logger.debug(f'Duplicate {rowFamily} {NumberOfDuplications} times')

    import Converter.Internal as I
    import Transform.PyTree as T

    OLD_FlowSolutionCenters = I.__FlowSolutionCenters__
    I.__FlowSolutionCenters__ = container

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
            if not FamilyNameNode: 
                continue
            zone_family = I.getValue(FamilyNameNode)
            if zone_family == rowFamily:
                if verbose>1: print(f'  > zone {zone_name}')
                check = True
                zones2merge = [zone]
                for n in range(NumberOfDuplications):
                    ang = azimuthal_extension*(n+1)
                    rot = T.rotate(I.copyNode(zone),(0.,0.,0.), axis, ang, vectors=vectors)
                    I.setName(rot, f"{zone_name}_{n+2}")
                    I._addChild(base, rot)
                    zones2merge.append(rot)
                if merge:
                    for node in zones2merge:
                        I.rmNode(base, node)
                    tree_dist = T.merge(zones2merge, tol=1e-8)
                    for i, node in enumerate(I.getZones(tree_dist)):
                        I._addChild(base, node)
                        disk_block = I.getNodeFromName(base, I.getName(node))
                        disk_block[0] = f'{zone_name}_{i:02d}'
                        I.createChild(disk_block, 'FamilyName', 'FamilyName_t', value=rowFamily)
    # if merge: PRE.autoMergeBCs(tree)

    I.__FlowSolutionCenters__ = OLD_FlowSolutionCenters
    assert check, 'None of the zones was duplicated. Check the name of row family'

def _duplicate_with_maia(tree, duplication_operations, merge_zones=False):
    import maia
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    tree = to_distributed(tree)

    for operation in duplication_operations:
        Family = operation['Family']
        NumberOfDuplications = operation['NumberOfDuplications']

        AzimuthalExtension = compute_azimuthal_extension(tree, Family)
        is_360 = np.isclose(AzimuthalExtension * (NumberOfDuplications+1), 360.)

        if NumberOfDuplications == 0:
            return
        elif is_360:
            mola_logger.info(f"  > row {Family} is replicated on 360 degrees", rank=0)
            maia.algo.dist.duplicate_family_from_rotation_jns_to_360(tree, Family, comm)
        else:
            plurial = 's' if NumberOfDuplications > 1 else ''
            mola_logger.info(f"  > row {Family} is replicated {NumberOfDuplications} time"+plurial, rank=0)
            maia.algo.dist.duplicate_family_from_periodic_jns(tree, Family, NumberOfDuplications, comm)
        tree = cgns.castNode(tree)
        
    if merge_zones:
        maia.algo.dist.merge_connected_zones(tree, comm)    
        tree = cgns.castNode(tree)

    return tree

