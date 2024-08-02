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

import os
import glob
import shutil
from fnmatch import fnmatch

import elsAxdt

from treelab import cgns

from mola.logging import MolaException
import mola.naming_conventions as names
# no relative imports possible for the following line because the current file is called by
# call_solver_specific_function in manager.py
from mola.cfd.coprocess import mola_logger, rank, comm
import mola.cfd.postprocess as POST
from mola.cfd.preprocess.mesh.tools import ravel_BCDataSet, remove_empty_BCDataSet, force_FamilyBC_as_FamilySpecified
from mola.cfd.preprocess.mesh.families import get_family_to_BCType

def perform_extractions(workflow, coprocess_manager):
    output_tree = get_elsa_output_tree(workflow._Skeleton)
    families_to_bctype = get_family_to_BCType(output_tree)

    # extract all integral data at once, so once by iteration, 
    # whatever the number of Extractions with type Integral
    integral_data_already_extracted = False 
    
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

        elif extraction['Type'] == 'Residuals':
            extraction['Data'] = extract_residuals(output_tree)
        
        elif extraction['Type'] == 'Integral':
            if not integral_data_already_extracted:
                NormalizationCoefficients = workflow.ApplicationContext.get('NormalizationCoefficient')
                extraction['Data'] = extract_integral(output_tree, NormalizationCoefficients)
                integral_data_already_extracted = True
            else:
                extraction['Data'] = cgns.Tree()

        # elif extraction['Type'] == 'Probe':
        #     extraction['Data'] = extract_probe(output_tree)

        else:
            mola_logger.warning(f"Type of extraction {extraction['Type']} is not available for elsA", rank=0)
            extraction['Data'] = cgns.Tree()

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
    t = cgns.castNode(t)
    t.merge(skeleton)
    ravel_BCDataSet(t) # HACK https://elsa.onera.fr/issues/11219
    remove_empty_BCDataSet(t)
    # force_FamilyBC_as_FamilySpecified(t) # HACK https://elsa.onera.fr/issues/10928
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

        mola_logger.debug(f'  family={family}', rank=0)
    
        data_tree = POST.extract_bc(output_tree, Family=family, BaseName=family)
        data_tree = cgns.castNode(data_tree)
        SurfacesTree.merge(data_tree)
    
    if extraction['Name'] != 'ByFamily':
        # merge all bases and rename the unique base
        base0 =  SurfacesTree.bases()[0]
        base0.setName(extraction['Name'])
        i = 0
        for zone in base0.zones():
            zone.setName(f"{extraction['Name']}_R{rank}N{i}")
            i += 1
        for base in SurfacesTree.bases()[1:]:
            for zone in base.zones():
                zone.setName(f"{extraction['Name']}_R{rank}N{i}")
                i += 1
                zone.moveTo(base0)
            base.remove()

    return SurfacesTree

def extract_isosurface(output_tree, extraction):
    if extraction['IsoSurfaceContainer'] == 'auto':
        extraction['IsoSurfaceContainer'] = deduce_container_for_slicing(extraction['IsoSurfaceField'])

    isosurface = POST.iso_surface(
        output_tree, 
        IsoSurfaceField = extraction['IsoSurfaceField'], 
        IsoSurfaceValue = extraction['IsoSurfaceValue'], 
        IsoSurfaceContainer = extraction['IsoSurfaceContainer'],
        Name = extraction['Name'],
        tool = 'maia' if output_tree.isUnstructured() else 'cassiopee',
        )
    
    return isosurface

def extract_residuals(output_tree):
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

    comm.barrier()
    trees = comm.allgather(t)
    t = cgns.merge(trees)
    comm.barrier() 
    
    return t

def extract_integral(output_tree, NormalizationCoefficients=None):

    def _normalize_data(IntegralDataNode, Family, NormalizationCoefficients):
        data_to_normalize = dict(
            convflux_ro = dict(Name='MassFlow', Coef='FluxCoef'),
            CL = dict(Name='CL', Coef='FluxCoef'),
            CD = dict(Name='CD', Coef='FluxCoef'),
            CY = dict(Name='CY', Coef='FluxCoef'),
            Cn = dict(Name='Cn', Coef='TorqueCoef'),
            Cl = dict(Name='Cl', Coef='TorqueCoef'),
            Cm = dict(Name='Cm', Coef='TorqueCoef'),
        )
        for name, params in data_to_normalize.items():
            new_name = params['Name']
            try:
                coef = NormalizationCoefficients[Family][params['Coef']]
                node = IntegralDataNode.get(Name=name, Type='DataArray')
                cgns.Node(Type='DataArray', Name=new_name, Value=node.value()*coef, Parent=IntegralDataNode)
            except:
                pass
    
    t = cgns.Tree()
    base = cgns.Base(Name='Base', Parent=t)
    zone = cgns.Zone(Name='Integral', Parent=base)
    for IntegralDataNode in output_tree.group(Type='IntegralData', Depth=2):
        Family = IntegralDataNode.name().split('-')[0]
        IntegralDataNode.dettach()
        IntegralDataNode.setName(Family)
        if NormalizationCoefficients:
            _normalize_data(IntegralDataNode, Family, NormalizationCoefficients)
        zone.addChild(IntegralDataNode)

    comm.barrier()
    trees = comm.allgather(t)
    t = cgns.merge(trees)
    comm.barrier() 

    return t

def extract_probe(output_tree):
    mola_logger.warning('skip extraction of type Probe (not implemented yet)', rank=0)
    return cgns.Tree()

def update_elsa_input(new_tree):
    elsAxdt.xdt(elsAxdt.PYTHON,(elsAxdt.RUNTIME_TREE, new_tree, 1))

def end_simulation(workflow):
    elsAxdt.safeInterrupt()

def deduce_container_for_slicing(IsoSurfaceField):
    if IsoSurfaceField in ['CoordinateX', 'CoordinateY', 'CoordinateZ']:
        return 'GridCoordinates'

    elif IsoSurfaceField in ['Radius', 'radius', 'CoordinateR', 'Slice']:
        return 'FlowSolution'

    elif IsoSurfaceField == 'ChannelHeight':
        return 'FlowSolution#Height'
    
    else:
        return 'FlowSolution#EndOfRun'

def move_log_files(w):
    if rank == 0:
        for fn in glob.glob('elsA_MPI*'):
            shutil.move(fn, os.path.join(names.DIRECTORY_LOG, fn))

    comm.barrier()
