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
from fnmatch import fnmatch

import Converter.PyTree as C
import Converter.Internal as I
import maia
import maia.pytree as PT
import elsAxdt

from treelab import cgns

from mola.logging import MolaException
# no relative imports possible for the following line because the current file is called by
# call_solver_specific_function in manager.py
from mola.cfd.coprocess import mola_logger, rank, comm
from mola.cfd.coprocess.io.utils import ravelBCDataSet, forceFamilyBCasFamilySpecified


def perform_extractions(workflow, coprocess_manager):
    output_tree = get_elsa_output_tree(workflow._Skeleton, coprocess_manager.iteration)
    # C.convertPyTree2File(output_tree, f'output_tree_{rank}.cgns')

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
            mola_logger.warning('skip extraction of type IsoSurface (not implemented yet)', rank=0)
            extraction['Data'] = None  # extract_isosurface(output_tree, extraction)
        
        elif extraction['Type'] == 'Integral':
            extraction['Data'] = extract_residuals(output_tree)

        
        if extraction['Type'] not in ['Restart', '3D']:
            extraction['Data'].findAndRemoveNodes(Name=':CGNS#Ppart', Depth=3)

        comm.barrier()

def get_elsa_output_tree(skeleton, iteration):
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
    # adaptEndOfRun(t, iteration)
    for tree in [t, skeleton]: 
        ravelBCDataSet(tree) # HACK https://elsa.onera.fr/issues/11219
    # resumeFieldsAveraging(coprocess_manager, t)
    t = I.merge([skeleton, t])
    t = cgns.castNode(t)
    t.findAndRemoveNodes(Name='FlowSolution#Init*', Type='FlowSolution', Depth=3)
    removeEmptyBCDataSet(t)
    forceFamilyBCasFamilySpecified(t) # HACK https://elsa.onera.fr/issues/10928
    return cgns.castNode(t)

def update_restart_fields_maia(workflow, output_tree):    
    part_tree_restart = PT.shallow_copy(output_tree)

    I._renameNode(part_tree_restart, 'FlowSolution#EndOfRun', 'FlowSolution#Init')
    # I._renameNode(part_tree_restart, f'FlowSolution#EndOfRun{iteration-1:04d}', 'FlowSolution#Init-1')
    PT.rm_nodes_from_predicate(part_tree_restart, 
                                lambda n: PT.get_label(n) == 'FlowSolution_t' \
                                and not PT.get_name(n).startswith('FlowSolution#Init' \
                                and not PT.get_name(n) == 'FlowSolution#Average')
                                )
    
    maia.io.part_tree_to_file(part_tree_restart, 'part_tree.cgns', comm)

    maia.transfer.part_tree_to_dist_tree_all(workflow.tree, part_tree_restart, comm)
    workflow.tree = cgns.castNode(workflow.tree)

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
    t.findAndRemoveNodes(Type='UserDefinedData', Depth=2)
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
    isosurface = maia.algo.part.iso_surface(
                output_tree, 
                f"GridCoordinate/{extraction['IsoSurfaceField']}",
                iso_val=extraction['IsoSurfaceValue'],
                containers_name=[], 
                comm=comm,
                )
    
    return cgns.castNode(isosurface)
    
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

def adaptEndOfRun(output_tree, CurrentIteration):
    '''
    This function is used to make adaptations of the coupling trigger tree
    provided by elsA. The following operations are performed:

    * ``GridCoordinates`` node is created from ``FlowSolution#EndOfRun#Coords``
    * adapt name of masking field (``cellnf`` is renamed as ``cellN``)
    * rename ``FlowSolution#EndOfRun`` as ``FlowSolution#Init``

    Parameters
    ----------

         to : PyTree
            Coupling tree as obtained from function

            >>> elsAxdt.get(elsAxdt.OUTPUT_TREE)

            .. note:: tree **to** is modified
    '''
    # moveCoordsFromEndOfRunToGridCoords(output_tree)
    # I._renameNode(output_tree, 'cellnf', 'cellN')
    # I._renameNode(output_tree, 'FlowSolution#EndOfRun', 'FlowSolution#Init')
    # I._renameNode(output_tree, f'FlowSolution#EndOfRun{CurrentIteration-1:04d}', 'FlowSolution#Init-1')

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

def resumeFieldsAveraging(coprocess_manager, t, container_name='FlowSolution#Average'):
    '''
    use any pre-existing average fields contained in ``FlowSolution#Average``
    nodes in order to resume the fields averaging process
    '''
    Skeleton  = coprocess_manager.skeleton
    inititer = coprocess_manager.workflow.Numerics['IterationAtInitialState']
    firstiter = coprocess_manager.workflow.ReferenceValues['CoprocessOptions']['FirstIterationForFieldsAveraging']
    if firstiter is None: 
        return
    firstiter -= 1
    cit = coprocess_manager.iteration

    # adapt 3D fields:
    old = _getDictofNodesFieldsPerZone(Skeleton, container_name)
    tot = _getDictofNodesFieldsPerZone(t, container_name)
    if cit == firstiter:
        ini = _getDictofNodesFieldsPerZone(t, 'FlowSolution#Init')
    for zone_name in tot:
        for field_name in tot[zone_name]:
            if field_name in ['cellN','indicm']: continue
            avg_old = old[zone_name][field_name] # BEWARE this is a CGNS node
            avg_tot = tot[zone_name][field_name] # BEWARE this is a CGNS node
            
            if cit == firstiter:
                avg_old[1] = np.copy(avg_tot[1], order='F')
                avg_tot[1] = np.copy(ini[zone_name][field_name][1], order='F')
                continue

            if cit < firstiter: 
                avg_old[1] = None
                avg_new    = None
            
            else:
                if avg_old[1] is None or avg_tot[1] is None: continue
                if inititer < firstiter:
                    avg_new =  (avg_tot[1]*(cit-inititer+1) \
                            -avg_old[1]*(firstiter-inititer+1))/(cit-firstiter)
                
                else:
                    avg_new =  (avg_old[1]*(inititer-(firstiter+1)) \
                            +avg_tot[1]*(cit-inititer+1))/(cit-firstiter)

                avg_tot[1] = avg_new # update of OUTPUT_TREE

    if cit < firstiter: return

    # adapt BC fields:
    tot = _getDictofNodesBCFieldsPerZone(t, 'BCDataSet#Average')
    comm.barrier()
    old = _getDictofNodesBCFieldsPerZoneAtSkeleton(Skeleton, 'BCDataSet#Average', tot)
    comm.barrier()
    if cit == firstiter:
        ini = _getDictofNodesBCFieldsPerZone(t, 'BCDataSet')
    for zone_name in tot:
        for bcfamily_name in tot[zone_name]:
            for field_name in tot[zone_name][bcfamily_name]:
                if field_name in ['cellN','indicm']: continue
                try:
                    avg_old = old[zone_name][bcfamily_name][field_name] # BEWARE this is a CGNS node
                except KeyError:
                    avg_old = [field_name,None,[],'DataArray_t']

                avg_tot = tot[zone_name][bcfamily_name][field_name] # BEWARE this is a CGNS node
                
                if cit == firstiter:
                    avg_old[1] = np.copy(avg_tot[1], order='F')
                    avg_tot[1] = np.copy(ini[zone_name][bcfamily_name][field_name][1], order='F')
                    continue

                if cit < firstiter:
                    avg_old[1] = None
                    avg_new    = None
                
                else:
                    if avg_old[1] is None or avg_tot[1] is None: continue
                    if inititer < firstiter:
                        avg_new =  (avg_tot[1]*(cit-inititer+1) \
                                -avg_old[1]*(firstiter-inititer+1))/(cit-firstiter)
                    
                    else:
                        avg_new =  (avg_old[1]*(inititer-(firstiter+1)) \
                                +avg_tot[1]*(cit-inititer+1))/(cit-firstiter)

                avg_tot[1] = avg_new # update of OUTPUT_TREE
                avg_old[1] = avg_new # update of OUTPUT_TREE

def _getDictofNodesFieldsPerZone(t, Container):
    fields = dict()
    for base in I.getNodesFromType1(t, 'CGNSBase_t'):
        for zone in I.getNodesFromType1(base, 'Zone_t'):
            zone_name = zone[0]
            fields[zone_name] = dict()
            fs = I.getNodeFromName1(zone, Container)
            if not fs:
                del fields[zone_name]
                continue
            for f in fs[2]:
                if f[3] != 'DataArray_t': continue
                fields[zone_name][f[0]] = f
    return fields

def _getDictofNodesBCFieldsPerZoneAtSkeleton(t, Container, tot):
    fields = dict()
    for base in I.getNodesFromType1(t, 'CGNSBase_t'):
        for zone in I.getNodesFromType1(base, 'Zone_t'):
            zone_name = zone[0]
            fields[zone_name] = dict()
            for bc in I.getNodesFromType(zone,'BC_t'):
                bcfamily_name = bc[0]
                if bcfamily_name not in fields[zone_name]:
                    fields[zone_name][bcfamily_name] = dict()
                bcds = I.getNodeFromName1(bc, Container)
                if bcds:
                    bcds[3] = 'BCDataSet_t'
                    data = I.getNodeFromName1(bcds,'NeumannData')
                    if data:
                        for f in data[2]:
                            if f[3] != 'DataArray_t': continue
                            fields[zone_name][bcfamily_name][f[0]] = f
                    else:
                        try: fields_tot = tot[zone_name][bcfamily_name]
                        except KeyError: continue
                        if not fields_tot: continue
                        nd = I.createUniqueChild(bcds,'NeumannData','BCData_t')
                        for field_name, f in fields_tot.items():
                            field_node = I.createUniqueChild(nd,f[0],'DataArray_t',np.copy(f[1],order='F'))
                            fields[zone_name][bcfamily_name][f[0]] = field_node
                else:
                    try: fields_tot = tot[zone_name][bcfamily_name]
                    except KeyError: continue
                    if not fields_tot: continue
                    bcds = I.createUniqueChild(bc,Container,'BCDataSet_t')
                    nd = I.createUniqueChild(bcds,'NeumannData','BCData_t')
                    for field_name, f in fields_tot.items():
                        field_node = I.createUniqueChild(nd,f[0],'DataArray_t',np.copy(f[1],order='F'))
                        fields[zone_name][bcfamily_name][f[0]] = field_node

    return fields

def _getDictofNodesBCFieldsPerZone(t, Container):
    fields = dict()
    for base in I.getNodesFromType1(t, 'CGNSBase_t'):
        for zone in I.getNodesFromType1(base, 'Zone_t'):
            zone_name = zone[0]
            fields[zone_name] = dict()
            for bc in I.getNodesFromType(zone,'BC_t'):
                bcfamily_name = bc[0]
                if bcfamily_name not in fields[zone_name]:
                    fields[zone_name][bcfamily_name] = dict()
                bcds = I.getNodeFromName1(bc, Container)
                if bcds:
                    data = I.getNodeFromName1(bcds,'NeumannData')
                    if data:
                        for f in data[2]:
                            if f[3] != 'DataArray_t': continue
                            fields[zone_name][bcfamily_name][f[0]] = f
    return fields

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
     
