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
import warnings

from treelab import cgns

from mola.logging import MolaException, MolaUserError
import mola.naming_conventions as names
# no relative imports possible for the following line because the current file is called by
# call_solver_specific_function in manager.py
from mola.cfd.coprocess import rank, comm
import mola.cfd.postprocess as POST
from mola.cfd.preprocess.mesh.families import get_family_to_BCType

# https://fast.onera.fr/FastS.html#FastS.PyTree._computeVariables
post_fields_using_fast = ['QCriterion', 'Enstrophy'] 

# https://cassiopee.onera.fr/Post.html#Post.computeVariables
post_fields_using_cassiopee_computeVariables = [
    'VelocityX',
    'VelocityY',
    'VelocityZ',
    'VelocityMagnitude',
    'Pressure',
    'Temperature',
    'Enthalpy',
    'Entropy',
    'Mach',
    'ViscosityMolecular',
    'PressureStagnation',
    'TemperatureStagnation',
    'PressureDynamic']

# https://cassiopee.onera.fr/Post.html#Post.computeExtraVariable
post_fields_using_cassiopee_computeExtraVariable = [
    'Vorticity',
    'VorticityMagnitude',
    'ShearStress']


def perform_extractions(workflow, coprocess_manager):
    output_tree = get_output_tree(workflow, coprocess_manager)
    families_to_bctype = get_family_to_BCType(output_tree)

    # extract all integral data at once, so once by iteration, 
    # whatever the number of Extractions with type Integral
    integral_data_already_extracted = False 
    
    for extraction in coprocess_manager.Extractions:
        if extraction['IsToExtract'] == False:
            continue

        coprocess_manager.mola_logger.debug(f'  update extraction of type {extraction["Type"]}', rank=0)
        
        if extraction['Type'] == 'Restart':
            extraction['Data'] = workflow.tree
        
        elif extraction['Type'] == '3D':
            extraction['Data'] = extract_fields(output_tree, extraction)

        elif extraction['Type'] == 'BC':
            extraction['Data'] = extract_bc(output_tree, extraction, families_to_bctype, workflow._metrics)
        
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

        elif extraction['Type'] == 'Probe': 
            extraction['Data'] = extract_probe(output_tree)

        else:
            coprocess_manager.mola_logger.warning(f"Type of extraction {extraction['Type']} is not available for elsA", rank=0)
            extraction['Data'] = cgns.Tree()

        # Remove PyPart nodes for data that are not 3D (important to save them without PyPart)
        if extraction['Type'] not in ['Restart', '3D']:
            if extraction['Data'] is not None:
                extraction['Data'].findAndRemoveNodes(Name=':CGNS#Ppart', Depth=3)

        comm.barrier()

def get_output_tree(workflow, coprocess_manager):
    
    output_tree = workflow.tree.copy()
    for extraction in coprocess_manager.Extractions:
        if extraction['Type'] == '3D':
            compute_missing_fields_at_cell_centers( workflow, output_tree, extraction['Fields'])
    output_tree = cgns.castNode(output_tree)

    return output_tree


def extract_fields(output_tree, extraction) -> cgns.Tree:

    t = output_tree.copy()
    remove_not_requested_fields(t, extraction['Fields'])
    if extraction['GridLocation'] == 'Vertex': put_fields_in_vertex(t)
    if not extraction['GhostCells']: remove_ghost_cells(t)
    rename_flow_solution_container(t, extraction)
    remove_not_requested_containers(t, extraction['Container'])

    return t

def extract_bc(output_tree, extraction, families_to_bctype, metrics):

    # TODO factorize elsa <-> fast

    import FastS.PyTree as FastS

    SurfacesTree = cgns.Tree()

    for BCFamilyName in families_to_bctype:
        BCType = families_to_bctype[BCFamilyName]
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

        data_tree = POST.extract_bc(output_tree, Family=family, BaseName=family)
        data_tree = cgns.castNode(data_tree)

        stress_tree = FastS.createStressNodes(output_tree, [extraction['Source']])

        # TODO optimize by providing stress to integral extractions Data
        stress = FastS._computeStress(output_tree, stress_tree, metrics)
        stress_tree = cgns.castNode(stress_tree)

        for base_data, base_stress in zip(data_tree.bases(), stress_tree.bases()):
            base_stress.setName(base_data.name())
            for zone_data, zone_stress in zip(data_tree.zones(), stress_tree.zones()):
                zone_stress.setName(zone_data.name())

        data_tree.merge(stress_tree)

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
    warnings.warn("extract_residuals TODO -> to be implemented for fast")

def extract_integral(output_tree, NormalizationCoefficients):
    warnings.warn("extract_integral TODO -> to be implemented for fast")

def extract_probe(output_tree):
    warnings.warn("extract_probe TODO -> to be implemented for fast")


def deduce_container_for_slicing(IsoSurfaceField):
    if IsoSurfaceField in ['CoordinateX', 'CoordinateY', 'CoordinateZ']:
        return 'GridCoordinates'

    elif IsoSurfaceField in ['Radius', 'radius', 'CoordinateR', 'Slice']:
        return 'FlowSolution'

    elif IsoSurfaceField == 'ChannelHeight':
        return 'FlowSolution#Height'
    
    else:
        return 'FlowSolution#Centers'


def get_field_names( t : cgns.Tree, container : str ='FlowSolution#Centers') -> list:
    
    zone = t.get(Type='CGNSBase_t',Depth=1).get(Type='Zone_t',Depth=1)
    fs = zone.get(Name=container,Depth=1)
    return [n.name() for n in fs.children() if n.type()=='DataArray_t']


def compute_missing_fields_at_cell_centers( workflow, t : cgns.Tree, field_names : list):
    
    import FastS.PyTree as FastS
    import Post.PyTree as P
    import Converter.PyTree as C
    import Converter.Internal as I

    existing_field_names = get_field_names(t)

    thermodynamic_const = dict(gamma = workflow.Fluid['Gamma'],
                               rgp   = workflow.Fluid['IdealGasConstant'],
                               Cs    = workflow.Fluid['SutherlandConstant'],
                               mus   = workflow.Fluid['SutherlandViscosity'],
                               Ts    = workflow.Fluid['SutherlandTemperature'])

    for requested_field_name in field_names:
        
        if requested_field_name in existing_field_names: continue

        if requested_field_name in post_fields_using_fast:
            FastS._computeVariables(t, workflow._metrics, requested_field_name)
    
        elif requested_field_name in post_fields_using_cassiopee_computeVariables:
            P._computeVariables(t, ["centers:"+requested_field_name], 
                                    **thermodynamic_const)

        elif requested_field_name in post_fields_using_cassiopee_computeExtraVariable: 
            tRef = P.computeExtraVariable(t, "centers:"+requested_field_name,
                                          **thermodynamic_const)
                

            # HACK, because computeExtraVariable does not exist in-place...
            for z_ref, z in zip(I.getZones(tRef), I.getZones(t)):
                fs_ref = I.getNodeFromName1(z_ref,'FlowSolution#Centers')
                fs = I.getNodeFromName1(z,'FlowSolution#Centers')
                fs[2] = fs_ref[2]

        elif requested_field_name.startswith('Momentum'):
            coord = requested_field_name.replace('Momentum','')
            if coord not in ['X','Y','Z']:
                raise MolaUserError('could not extract %s. Available fields: %s'%(requested_field_name,str(existing_field_names)))
            C._initVars(t,'centers:Momentum%s={centers:Velocity%s}*{centers:Density}'%(coord,coord))

        else:
            raise MolaUserError('cannot extract '+requested_field_name)

    cgns.castNode(t)

    
def remove_not_requested_fields( t : cgns.Tree, requested_field_names : list):
    
    if 'Vorticity' in requested_field_names:
        requested_field_names += ['VorticityX', 'VorticityY', 'VorticityZ']

    for zone in t.zones():
        FlowSolution = zone.get(Name='FlowSolution#Centers', Depth=1)
        if FlowSolution is None: raise MolaException('FATAL expected FlowSolution#Centers at '+zone.path())
        for field_node in FlowSolution.group(Type='DataArray_t', Depth=1):
            if field_node.name() not in requested_field_names:
                field_node.remove()


def remove_ghost_cells( t : cgns.Tree ):

    import Converter.Internal as I
    I._rmGhostCells(t,t,2,adaptBCs=1)
    cgns.castNode(t)


def put_fields_in_vertex( t : cgns.Tree ):

    import Converter.PyTree as C

    for field_name in get_field_names(t):
        C._center2Node__(t, 'centers:'+field_name, 0)

    cgns.castNode(t)


def rename_flow_solution_container(t : cgns.Tree, extraction : dict):

    for zone in t.zones():
        if extraction['GridLocation'] == 'Vertex':
            container_name = 'FlowSolution'
        else:
            container_name = 'FlowSolution#Centers'

        flow_solution_node = zone.get(Name=container_name, Depth=1)

        if not flow_solution_node:
            zone.save('debug.cgns')
            raise MolaException('dumping debug.cgns expected finding '+zone.path()+'/'+container_name)

        flow_solution_node.setName(extraction['Container'])
            


def remove_not_requested_containers(t : cgns.Tree, container : str):

    for zone in t.zones():
        # Remove FlowSolution nodes that are not the target
        for FS in zone.group(Type='FlowSolution', Depth=1):
            if FS.name() != container:
                FS.remove()
        
        if not zone.get(Type='FlowSolution', Depth=1):
            # no more FlowSolution in the current zone
            # --> remove this zone
            zone.remove()
            continue