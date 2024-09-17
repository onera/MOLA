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
import numpy as np
from fnmatch import fnmatch
import warnings

from treelab import cgns

from mola.logging import MolaException, MolaUserError
import mola.naming_conventions as names
# no relative imports possible for the following line because the current file is called by
# call_solver_specific_function in manager.py
from mola.cfd.coprocess import rank, comm
from mola.cfd.coprocess.manager import mpi_allgather_and_merge_trees, update_signals_using, get_bc_families_in_extraction, write_extraction_log
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
    
    for extraction in coprocess_manager.Extractions:
        if extraction['IsToExtract'] == False:
            continue

        coprocess_manager.mola_logger.debug(f'  update extraction of type {extraction["Type"]}', rank=0)
        
        if extraction['Type'] == 'Restart':
            extraction['Data'] = workflow.tree
        
        elif extraction['Type'] == '3D':
            extraction['Data'] = extract_fields(output_tree, extraction)

        elif extraction['Type'] == 'BC':
            extraction['Data'] = extract_bc(output_tree, extraction, families_to_bctype, workflow._fast_metrics)
        
        elif extraction['Type'] == 'IsoSurface':
            extraction['Data'] = extract_isosurface(output_tree, extraction)

        elif extraction['Type'] == 'Residuals':
            coprocess_manager.mola_logger.warn('extract residuals requires solving https://github.com/onera/Fast/issues/13')
            extract_residuals(output_tree, extraction)
        
        elif extraction['Type'] == 'Integral':
            extract_integral(output_tree, extraction, workflow)

        elif extraction['Type'] == 'Probe': 
            extraction['Data'] = extract_probe(output_tree)

        else:
            coprocess_manager.mola_logger.warning(f"Type of extraction {extraction['Type']} is not available for elsA", rank=0)
            extraction['Data'] = cgns.Tree()

        # Remove PyPart nodes for data that are not 3D (important to save them without PyPart)
        if extraction['Type'] not in ['Restart', '3D']:
            if extraction['Data'] is not None:
                extraction['Data'].findAndRemoveNodes(Name=':CGNS#Ppart', Depth=3)

        write_extraction_log(extraction)

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

    import FastS.PyTree as FastS

    SurfacesTree = cgns.Tree()

    families_to_extract = get_bc_families_in_extraction(extraction, families_to_bctype)

    for family in families_to_extract:

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
        POST.merge_bases_and_rename_unique_base(SurfacesTree, extraction['Name'])

    for type_to_remove in 'Rind_t', 'ConvergenceHistory_t':
        SurfacesTree.findAndRemoveNodes(Type=type_to_remove)

    remove_spurious_data_from_output(SurfacesTree)

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
    
    remove_spurious_data_from_output(isosurface)
    
    return isosurface


def extract_residuals(output_tree, extraction):
    residuals = output_tree.group(Type='ConvergenceHistory_t', Depth=4)
    if not residuals: return cgns.Tree()

    t = cgns.Tree()
    base = cgns.Base(Name='Residuals', Parent=t)
    for residual in residuals:
        parent_name = residual.parent().name()
        
        node = residual.copy(deep=True)
        
        unstack_residual(node)

        node.setType('FlowSolution_t')
        node.setName('FlowSolution')
        
        cgns.Zone(Name=parent_name, Parent=base, Children=[node])

    current_iteration_signals =  mpi_allgather_and_merge_trees(t)
    
    if 'Data' in extraction and extraction['Data'] is not None:
        and_previous_signals_to_be_updated = extraction['Data']
        update_signals_using(current_iteration_signals, and_previous_signals_to_be_updated)
    else: 
        extraction['Data'] = current_iteration_signals

    

def extract_integral(output_tree, extraction, workflow) -> None:
    
    stress, state = get_stress_and_state(output_tree, extraction['Source'],
                                         workflow._fast_metrics)
    dimensionalize_torque(stress, state)    

    fields = initialize_integral_fields_dict(workflow._coprocess_manager)

    if 'Force' in extraction['Fields']:
        fields.update(dict(ForceX  = np.array([stress['ForceX']]),
                           ForceY  = np.array([stress['ForceY']]),
                           ForceZ  = np.array([stress['ForceZ']])))

    if 'Torque' in extraction['Fields']:
        fields.update(dict(TorqueX  = np.array([stress['Torque0X']]),
                           TorqueY  = np.array([stress['Torque0Y']]),
                           TorqueZ  = np.array([stress['Torque0Z']])))

    if 'MassFlow' in extraction['Fields']:
        fields['MassFlow'] = np.array([stress['m']])

    t = cgns.Tree()
    base = cgns.Base(Name='Integral', Parent=t)
    zone = cgns.utils.newZoneFromDict( extraction['Name'], fields )
    zone.attachTo(base)

    current_iteration_signals = mpi_allgather_and_merge_trees(t)

    if 'Data' in extraction and extraction['Data'] is not None:
        and_previous_signals_to_be_updated = extraction['Data']
        update_signals_using(current_iteration_signals, and_previous_signals_to_be_updated)
    else: 
        extraction['Data'] = current_iteration_signals


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
            FastS._computeVariables(t, workflow._fast_metrics, requested_field_name)
    
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
    import Converter.Internal as I

    for field_name in get_field_names(t):
        C._center2Node__(t, 'centers:'+field_name, 0)

    cgns.castNode(t)

    for fs in t.group(Name=I.__FlowSolutionNodes__, Type='FlowSolution_t',Depth=4):
        GridLocation_node = cgns.Node(Name='GridLocation', Value='Vertex', Type='GridLocation_t')
        GridLocation_node.attachTo(fs, position=0)

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


def unstack_residual( residual : cgns.Node ):

    it_nb_node = residual.get('IterationNumber')
    
    if not it_nb_node: return
    
    it_nb = it_nb_node.value()

    for r in residual.children()[:]:
        if not r.name().startswith('RSD_'): continue
        r.dettach()
        array = r.value()
        it_qty = len(it_nb)
        fields_qty = int( len(array) / it_qty )
        array = np.reshape( array, (fields_qty,it_qty) )
        
        for i in range(fields_qty):
            cgns.Node(Name=r.name()+'_%d'%i, Value=np.copy(array[i,:]), Parent=residual)


def get_stress_and_state(output_tree : cgns.Tree , source : str, metrics) -> list:
    
    import FastS.PyTree as FastS
    stress_tree = FastS.createStressNodes(output_tree, [source])
    stress_list = FastS._computeStress(output_tree, stress_tree, metrics)

    fx, fy, fz, t0x, t0y, t0z, S, m, ForceX, ForceY, ForceZ = stress_list
    stress_dict = dict(fx=fx,
                       fy=fy,
                       fz=fz,
                       t0x=t0x,
                       t0y=t0y,
                       t0z=t0z,
                       S=S,
                       m=m,
                       ForceX=ForceX,
                       ForceY=ForceY,
                       ForceZ=ForceZ)

    stress_tree = cgns.castNode(stress_tree)
    reference_state = stress_tree.get(Type='ReferenceState_t', Depth=3)
    state_dict = reference_state.parent().getParameters(reference_state.name())

    return stress_dict, state_dict

def dimensionalize_torque(stress : dict, state : dict) -> None:

    rho = state['Density']
    one_over_rho = 1.0/rho
    Ux = one_over_rho * state['MomentumX']
    Uy = one_over_rho * state['MomentumY']
    Uz = one_over_rho * state['MomentumZ']

    PressureDynamic = 0.5 * rho * (Ux**2 + Uy**2 + Uz**2)
    LengthReference = 1.0 # 1 meter (cf private comm I.M. 8/8/24)

    torque_coef = PressureDynamic * stress['S'] * LengthReference # M^1 L^2 T^-2
    
    stress['Torque0X'] = float(torque_coef * stress['t0x'])
    stress['Torque0Y'] = float(torque_coef * stress['t0y'])
    stress['Torque0Z'] = float(torque_coef * stress['t0z'])

def remove_spurious_data_from_output( t : cgns.Tree ) -> None:

    for type_to_remove in 'Rind_t', 'ConvergenceHistory_t':
        t.findAndRemoveNodes(Type=type_to_remove)

    for name_to_remove in  '.Solver#define', '.Solver#ownData':
        t.findAndRemoveNodes(Name=name_to_remove)

def initialize_integral_fields_dict(coprocess_manager) -> dict:
    it = coprocess_manager.iteration
    fields = dict(IterationNumber=np.array([it]))

    if coprocess_manager.workflow.Numerics['TimeMarching'] != 'Steady':
        time = coprocess_manager.time
        fields['Time'] = np.array([time])

    return fields

def get_iteration(workflow):
    return workflow._iteration

def get_status(workflow):
    return workflow._status