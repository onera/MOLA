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

from fnmatch import fnmatch

import Converter.PyTree as C
import Converter.Internal as I
import maia
import maia.pytree as PT
import elsAxdt

from treelab import cgns

from mola.logging import MolaException
import mola.naming_conventions as names
# no relative imports possible for the following line because the current file is called by
# call_solver_specific_function in manager.py
from mola.cfd.coprocess import mola_logger, rank, comm


def perform_extractions(workflow, coprocess_manager):
    output_tree = get_elsa_output_tree(workflow._Skeleton)
    families_to_bctype = C.getFamilyBCNamesDict(output_tree)
    
    for extraction in coprocess_manager.Extractions:
        if extraction['IsToExtract'] == False:
            continue

        mola_logger.debug(f'  update extraction of type {extraction["Type"]}', rank=0)
        
        if extraction['Type'] == 'Restart':
            update_restart_fields(workflow, output_tree)
            extraction['Data'] = workflow.tree
        
        elif extraction['Type'] == '3D':
            extraction['Data'] = extract_fields(output_tree, extraction)

        elif extraction['Type'] == 'BC':
            extraction['Data'] = extract_bc(output_tree, extraction, families_to_bctype)
        
        elif extraction['Type'] == 'IsoSurface':
            extraction['Data'] = extract_isosurface(output_tree, extraction)
        
        elif extraction['Type'] == 'Integral':
            if 'Residuals' in extraction['Fields']:
                extraction['Data'] = extract_residuals(output_tree)

        elif extraction['Type'] == 'Probe':
            extraction['Data'] = extract_probe(output_tree)

        # Remove PyPart nodes for data that are not 3D (important to save them without PyPart)
        if extraction['Type'] not in ['Restart', '3D']:
            extraction['Data'].findAndRemoveNodes(Name=':CGNS#Ppart', Depth=3)

        comm.barrier()

def get_elsa_output_tree(skeleton):
    '''
    Extract the coupling CGNS PyTree from elsAxdt *OUTPUT_TREE* and make
    necessary adaptions, including migration of coordinates fields to
    GridCoordinates_t nodes, renaming of conventional fields names and
    adding the tree's Skeleton.

    Returns
    -------

        t : PyTree
            Coupling adapted PyTree

    '''
    t = elsAxdt.get(elsAxdt.OUTPUT_TREE)
    for tree in [t, skeleton]: 
        ravelBCDataSet(tree) # HACK https://elsa.onera.fr/issues/11219
    t = I.merge([skeleton, t])
    removeEmptyBCDataSet(t)
    forceFamilyBCasFamilySpecified(t) # HACK https://elsa.onera.fr/issues/10928
    t = cgns.castNode(t)
    t.findAndRemoveNodes(Name='FlowSolution#Init*', Type='FlowSolution', Depth=3)
    return t

def update_restart_fields(workflow, output_tree):
    output_tree = cgns.castNode(output_tree)
    for zone in output_tree.zones():
        zone.findAndRemoveNode(Name='FlowSolution#Init')
        FS = zone.get(Name='FlowSolution#EndOfRun')
        if FS is not None: 
            FS.setName('FlowSolution#Init')

    NodesToUpdate = output_tree.group(Name='FlowSolution#Init*', Type='FlowSolution', Depth=3) # for initial field(s) (possible second order restart)
    NodesToUpdate += output_tree.group(Name='FlowSolution#Average', Type='FlowSolution', Depth=3) 
    NodesToUpdate += output_tree.group(Name='BCDataSet#Average') 

    for node in NodesToUpdate:
        path = node.path()
        node_to_update = workflow.tree.getAtPath(path)
        parent = node_to_update.Parent
        node_to_update.remove()
        parent.addChild(node)
    
    workflow.tree = cgns.castNode(workflow.tree)

def extract_fields(output_tree, extraction):

    t = output_tree.copy()
    # HACK Pypart puts WorkflowParameters under the base... need to remove it
    t.findAndRemoveNodes(Name=names.CONTAINER_WORKLFOW_PARAMETERS, Type='UserDefinedData', Depth=2) 
    t.findAndRemoveNodes(Name='GlobalConvergenceHistory', Depth=2)
    t.findAndRemoveNodes(Type='IntegralData', Depth=2)
    t.findAndRemoveNodes(Name='ELSA_TRIGGER')

    for zone in t.zones():
        # Remove FlowSolution nodes that are not the target
        for FS in zone.group(Type='FlowSolution', Depth=1):
            if FS.name() != extraction['Container']:
                FS.remove()
        
        if not zone.get(Type='FlowSolution', Depth=1):
            # no more FlowSolution in the current zone
            # --> remove this zone
            zone.remove()
            continue
            
        # NOTE ZoneBC must be kept for to save tree with PyPart
        zone.findAndRemoveNodes(Type='BCDataSet')
    
    return t

def extract_bc(output_tree, extraction, DictBCNames2Type):
    SurfacesTree = cgns.Tree()
    CellDimension = output_tree.base().dim()

    for BCFamilyName in DictBCNames2Type:
        BCType = DictBCNames2Type[BCFamilyName]
        if fnmatch(BCType, extraction['Source']):
            # Case of source matching one or several names of BC: 'BCWall', 'BCInflow*', '*', etc.
            source = BCType
            family = BCFamilyName
        elif fnmatch(BCFamilyName, extraction['Source']):
            # Case of source matching a family name
            source = BCFamilyName
            family = BCFamilyName
        else:
            continue

        mola_logger.debug(f'  {family=}', rank=0)
        
        zones = C.extractBCOfName(output_tree, f'FamilySpecified:{family}', extrapFlow=False)
        
        # zones = maia.algo.part.extract_part_from_family(output_tree, family, comm, 
        #                                                containers_name=['BCDataSet'])
        
        if extraction['Name'] == 'ByFamily':
            addBase2SurfacesTree(SurfacesTree, BCFamilyName, zones, CellDimension)
        else:
            raise MolaException(f'Not implemented yet: extraction "Name" must be "ByFamily" for "Type"="BC" (now, it is "{extraction["Name"]}")')

    # Remove not needed FlowSolution nodes
    for zone in SurfacesTree.zones():
        for FS in zone.group(Type='FlowSolution', Depth=1):
            if FS.name() != 'FlowSolution#Centers':
                FS.remove()
    
    restore_families(SurfacesTree, output_tree)

    return SurfacesTree

def extract_isosurface(output_tree, extraction):
    if output_tree.isUnstructured():

        if len(extraction['IsoSurfaceField'].split('/')) == 1:
            container = deduce_container_for_slicing(extraction['IsoSurfaceField'])
            extraction['IsoSurfaceField'] = f"{container}/{extraction['IsoSurfaceField']}"

        containers_name = [fs.name() for fs in output_tree.group(Type='FlowSolution')]

        isosurface = maia.algo.part.iso_surface(
                    output_tree, 
                    extraction['IsoSurfaceField'],
                    iso_val=extraction['IsoSurfaceValue'],
                    containers_name=containers_name, 
                    comm=comm,
                    )
        
        isosurface = cgns.castNode(isosurface)

    else:
        mola_logger.warning('skip extraction of type IsoSurface (not implemented yet for structured mesh)', rank=0)
        isosurface = cgns.Tree()
    
    return isosurface

def deduce_container_for_slicing(IsoSurfaceField):
    if IsoSurfaceField in ['CoordinateX', 'CoordinateY', 'CoordinateZ']:
        return 'GridCoordinates'

    elif IsoSurfaceField in ['Radius', 'radius', 'CoordinateR', 'Slice']:
        return 'FlowSolution'

    elif IsoSurfaceField == 'ChannelHeight':
        return 'FlowSolution#Height'
    
    else:
        return 'FlowSolution#EndOfRun'
    
def addBase2SurfacesTree(SurfacesTree, basename, zones, CellDimension=3, PhysicalDimension=3):
    if not zones: 
        return
    
    base = SurfacesTree.get(Name=basename, Type='CGNSBase', Depth=1)
    if not base:
        # create that base
        base = cgns.Base(Parent=SurfacesTree, Name=basename)
        base.setCellDimension(CellDimension-1)
        base.setPhysicalDimension(PhysicalDimension)

    for i, zone in enumerate(zones):
        zone = cgns.castNode(zone)
        # The name of the parent zone is kept in a temporary node .parentZone, 
        # that will be removed before saving
        # There might be a \ in zone name if it is a result of C.ExtractBCOfType
        zoneName = zone.name().split('/')[0]
        cgns.Node(Name='.parentZone', Type='UserDefinedData_t', Value=zoneName, Parent=zone)
        # Rename zones like the base
        zone.setName(f'{basename}_R{rank}N{i}')
        base.addChild(zone)

    return base

def restore_families(surfaces, skeleton):
    '''
    Restore families in the PyTree **surfaces** (e.g read from
    ``'surfaces.cgns'``) based on information in **skeleton** (e.g read from
    ``'main.cgns'``). Also add the ReferenceState to be able to use function
    computeVariables from Cassiopee Post module.

    .. tip:: **skeleton** may be a skeleton tree.

    Parameters
    ----------

        surfaces : PyTree
            tree where zone names are the same as in **skeleton** (or with a
            suffix in '\\<bcname>'), but without information on families and
            ReferenceState.

        skeleton : PyTree
            tree of the full 3D domain with zones, families and ReferenceState.
            No data is needed so **skeleton** may be a skeleton tree.
    '''
    ReferenceState = skeleton.get(Type='ReferenceState', Depth=2) 
    family_nodes = skeleton.group(Type='Family', Depth=2) 

    for base in surfaces.bases():
        if ReferenceState:
            base.addChild(ReferenceState)

        families_in_base = []
        for zone in base.zones():
            parentZone_node = zone.get(Name='.parentZone')
            zone_name = parentZone_node.value()
            zone_in_full_tree = skeleton.get(Name=zone_name, Type='Zone')
            if zone_in_full_tree:  
                fam = zone_in_full_tree.get(Type='FamilyName', Depth=1)
                if fam:
                    zone.addChild(fam)
                    families_in_base.append(fam.value())
            else:
                # This is an extracted BC
                fam = zone.get(Type='FamilyName', Depth=1)
                if fam: 
                    families_in_base.append(fam.value())

            parentZone_node.remove()
            
        for family in family_nodes:
            if family.name() in families_in_base:
                base.addChild(family)
    
def update_elsa_input(new_tree):
    elsAxdt.xdt(elsAxdt.PYTHON,(elsAxdt.RUNTIME_TREE, new_tree, 1))

def end_simulation(workflow):
    elsAxdt.safeInterrupt()

def moveCoordsFromEndOfRunToGridCoords(to):
    '''
    This function is used to make adaptations of the coupling trigger tree
    provided by elsA. The following operations are performed:

    * ``GridCoordinates`` node is created from ``FlowSolution#EndOfRun#Coords``


    Parameters
    ----------

         to : PyTree
            Coupling tree as obtained from function

            >>> elsAxdt.get(elsAxdt.OUTPUT_TREE)

            .. note:: tree **to** is modified
    '''
    FScoords = I.getNodeFromName(to, 'FlowSolution#EndOfRun#Coords')
    if FScoords:
        I._renameNode(to,'FlowSolution#EndOfRun#Coords','GridCoordinates')
        for GridCoordsNode in I.getNodesFromName3(to, 'GridCoordinates'):
            GridLocationNode = I.getNodeFromType1(GridCoordsNode, 'GridLocation_t')
            if I.getValue(GridLocationNode) != 'Vertex':
                zone = I.getParentOfNode(to, GridCoordsNode)
                ERRMSG = ('Extracted coordinates of zone '
                          '%s must be located in Vertex')%I.getName(zone)
                raise MolaException(ERRMSG)
            I.rmNode(to, GridLocationNode)
            I.setType(GridCoordsNode, 'GridCoordinates_t')
    comm.barrier()

def removeEmptyBCDataSet(t):
    for z in I.getZones(t):
        for zbc in I.getNodesFromType1(z,'ZoneBC_t'):
            for bc in I.getNodesFromType1(zbc,'BC_t'):
                for n in bc[2]:
                    if n[0].startswith('BCDataSet'):
                        if not n[2]:
                            I._rmNode(t, n)

def extract_residuals(output_tree):
    if rank == 0:
        residuals = output_tree.base().get(Name='GlobalConvergenceHistory', Depth=2)
        if not residuals:
            return cgns.Tree()
        residuals = cgns.castNode(residuals)
        residuals.findAndRemoveNode(Name='.Solver#Output')
        t = cgns.Tree()
        base = cgns.Base(Name='Base', Parent=t)
        cgns.Zone(Name='Monitoring', Parent=base, Children=[residuals])
        # NOTE maybe it would be better to put the ConvergenceHistory node under the base (not the zone),
        # but for now it seems to be not permitted with treelab
    else:
        t = cgns.Tree()
    
    return t
     
def ravelBCDataSet(t):
    # HACK https://elsa.onera.fr/issues/11219
    # HACK https://elsa-e.onera.fr/issues/10750
    for zone in I.getZones(t):
        for zbc in I.getNodesFromType1(zone,'ZoneBC_t'):
            for bc in I.getNodesFromType1(zbc,'BC_t'):
                for bcds in I.getNodesFromType1(bc,'BCDataSet_t'):
                    for bcd in I.getNodesFromType1(bcds,'BCData_t'):
                        for da in I.getNodesFromType1(bcd,'DataArray_t'):
                            if da[1] is not None:
                                da[1] = da[1].ravel(order='K')

def forceFamilyBCasFamilySpecified(t):
    # https://elsa.onera.fr/issues/10928
    for base in I.getBases(t):
        for zone in I.getZones(base):
            for ZoneBC in I.getNodesFromType1(zone,'ZoneBC_t'):
                for BC in I.getNodesFromType1(ZoneBC,'BC_t'):
                    FamilyNameNode = I.getNodeFromType1(BC,'FamilyName_t')
                    if FamilyNameNode is not None:
                        I.setValue(BC,'FamilySpecified')
                        FamilyName = I.getValue(FamilyNameNode)
                        if not I.getNodeFromName1(base,FamilyName):
                            FamilyAtBase = I.createNode(FamilyName,'Family_t',parent=base)
                            I.createNode('FamilyBC','FamilyBC_t',value='UserDefined',parent=FamilyAtBase)
                        continue

