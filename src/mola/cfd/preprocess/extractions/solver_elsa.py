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

import copy
from treelab import cgns
import mola.naming_conventions as names
from mola.logging import mola_logger, MolaException
from mola.cfd.preprocess.solver_specific_tools.solver_elsa import translate_to_elsa
from mola.cfd.preprocess.extractions.extractions import get_familiesBC_nodes, get_bc_families_to_extract

# FIXME Check the writingframe, following what has been done in mola v1

def apply_to_solver(workflow):

    check_not_extraction_BC_if_unstructured(workflow)  # FIXME also tests in worklow/test_workflow.py are currently deactivated
    add_extractions_for_restart(workflow)
    add_extractions_for_overset_components(workflow)
    process_extractions_of_type_field(workflow)
    process_extractions_of_type_bc_and_integral(workflow)
    add_trigger(workflow.tree)
    for Extraction in workflow.Extractions: 
        if Extraction['Type'] == 'Residuals':
            add_global_convergence_history(workflow, Extraction['ExtractionPeriod'])
            # In elsA, the extraction period is defined by add_global_convergence_history
            # Hence, the update of residuals by MOLA can be done at SavePeriod (more is useless)
            Extraction['ExtractionPeriod'] = Extraction['SavePeriod']
        # elif Extraction['Type'] == 'Integral':
        #     Extraction['ExtractionPeriod'] = Extraction['SavePeriod']
            
def check_not_extraction_BC_if_unstructured(workflow):
    if not workflow.tree.isStructured():
        for extraction in workflow.Extractions:
            if extraction['Type'] == 'BC':
                raise MolaException((
                    'Extraction with Type="BC" is currently not possible with elsA with MOLA.\n'
                    f'In this case, extraction on {extraction["Source"]} of {extraction["Fields"]} is not possible.'
                ))

def add_extractions_for_overset_components(workflow):
    if workflow.has_overset_component():
        workflow._interface.add_to_Extractions_3D(
            Fields    = ['CoordinateX', 'CoordinateY', 'CoordinateZ'], 
            Container = 'FlowSolution#EndOfRun#Coords', 
            GridLocation = 'Vertex',
            Frame     = 'absolute',
        )

def add_global_convergence_history(workflow, ExtractionPeriod=1):
    for base in workflow.tree.bases():
        GlobalConvergenceHistory = cgns.Node(Parent=base, Name='GlobalConvergenceHistory', Value=0, Type='UserDefinedData')
        cgns.Node(Parent=GlobalConvergenceHistory, Name='NormDefinitions', Value='ConvergenceHistory', Type='Descriptor')
        GlobalConvergenceHistory.setParameters('.Solver#Output',
                                        period=ExtractionPeriod,
                                        writingmode=0,
                                        var='residual_cons residual_turb'
                                        )

def add_extractions_for_restart(workflow):
    workflow.tree.findAndRemoveNodes(Name='FlowSolution#EndOfRun', Type='FlowSolution')

    workflow._interface.add_to_Extractions_Restart(
        Container='FlowSolution#EndOfRun', 
        Fields=list(workflow.Flow['ReferenceState']),
        Frame='relative'
        )

def process_extractions_of_type_field(workflow):

    # For 3D averaged field : 
    #   dict(type='3D', Container='FlowSolution#Average', fields=[...], options=dict(average='time', period_init='inactive'))

    add_GridLocation = workflow.SplittingAndDistribution['Splitter'].lower() != 'maia'

    add_cellN_field = workflow.has_overset_component()

    for zone in workflow.tree.zones():
        for Extraction in workflow.Extractions:

            if Extraction['Type'] in ['3D', 'Restart'] and is_zone_in_extraction_family(zone, Extraction):
                add_3d_extraction_to_zone(zone, Extraction, add_GridLocation, add_cellN_field)

            elif Extraction['Type'] == 'IsoSurface' and is_zone_in_extraction_family(zone, Extraction):
                Fields = Extraction.get('Fields')
                if Fields is None or len(Fields) == 0:
                    continue

                import inspect                
                signature = inspect.signature(workflow._interface.add_to_Extractions_3D)
                extraction3D = dict((name, param.default) for name, param in signature.parameters.items() if name != 'self')
                extraction3D['Fields'] = Fields
                extraction3D['GridLocation'] = 'Vertex'
                extraction3D['Container'] = names.CONTAINER_OUTPUT_FIELDS_AT_VERTEX
                extraction3D['Frame'] = Extraction['Frame']
                extraction3D['OtherOptions'] = dict()
                
                add_3d_extraction_to_zone(zone, extraction3D, add_GridLocation, add_cellN_field)
            
            elif Extraction['Type'] == 'Probe' and is_zone_in_extraction_family(zone, Extraction):
                Fields = Extraction.get('Fields')
                if Fields is None or len(Fields) == 0:
                    continue

                import inspect                
                signature = inspect.signature(workflow._interface.add_to_Extractions_Probe)
                extraction3D = dict((name, param.default) for name, param in signature.parameters.items() if name != 'self')
                extraction3D['Fields'] = Fields
                extraction3D['GridLocation'] = 'CellCenter'
                extraction3D['Container'] = 'FlowSolution#Probes'
                extraction3D['OtherOptions'] = dict()
                
                add_3d_extraction_to_zone(zone, extraction3D, add_GridLocation, add_cellN_field)

def is_zone_in_extraction_family(zone, Extraction):
    try:
        has_a_corresponding_FamilyName = zone.get(Type='FamilyName', Value=Extraction['Family'], Depth=1)
        has_a_corresponding_AditionnalFamilyName = zone.get(Type='AditionnalFamilyName', Value=Extraction['Family'], Depth=1)
        if has_a_corresponding_FamilyName or has_a_corresponding_AditionnalFamilyName:
            return True
        else:
            return False
    except KeyError:
        # No Family is given as a filter: no filter is applied
        return True

def add_3d_extraction_to_zone(zone, Extraction, add_GridLocation=True, add_cellN_field=False): 
    
    if Extraction['Fields'] == []: 
        mola_logger.user_warning(f'Caution: the list of fields in Extraction of name {Extraction["Name"]} is empty')
        return

    EoRnode = zone.get(Name=Extraction['Container'], Type='FlowSolution', Depth=1) 
    if EoRnode is None:
        EoRnode = zone.setParameters(
            Extraction['Container'], 
            ContainerType='FlowSolution', 
            )
        if add_GridLocation:
            cgns.Node(Parent=EoRnode, Name='GridLocation', Type='GridLocation', Value=Extraction['GridLocation'])

    elsa_var_list = translate_to_elsa(Extraction['Fields'], type='var')

    if add_cellN_field and Extraction['Container'] != "FlowSolution#EndOfRun#Coords":
        if 'cellN' not in Extraction['Fields']:
            Extraction['Fields'] += ['cellN']
        celln_var = translate_to_elsa('cellN', type='var')
        if celln_var not in elsa_var_list:
            elsa_var_list += [celln_var]
    
    # Set GridLocation
    if Extraction['GridLocation'] == 'CellCenter':
        loc = 'cell'
    elif Extraction['GridLocation'] == 'Vertex':
        loc = 'node'
    else:
        raise MolaException(f'no defined GridLocation for 3D extraction: {Extraction["GridLocation"]}. Choose CellCenter or Vertex.')
    


    solver_output_name = '.Solver#Output'
    options = Extraction.get('OtherOptions', dict())
    output_keys = dict(
        loc           = loc,
        period        = 1,
        writingmode   = 2,
        writingframe  = Extraction['Frame'],
        var           = elsa_var_list,
        **options
    )

    add_variables_in_right_solver_output(EoRnode, output_keys)

def add_variables_in_right_solver_output(root_node, output_keys, solver_output_name='.Solver#Output', force_new_solver_output=False):

    def create_names_generator():
        # Generator that returns .Solver#Output, .Solver#Output#2, .Solver#Output#3, ...
        yield solver_output_name
        for suffix in range(2, 100):
            yield f"{solver_output_name}#{suffix}"

    names_generator = create_names_generator()

    # Search for a .Solver#Output with same parameters
    for name in names_generator:
        solver_ouput = root_node.get(Name=name, Depth=1)
        if not solver_ouput:
            break  # this name is not already used
        elif force_new_solver_output:
            continue
        else:
            params = root_node.getParameters(solver_ouput.name())
            if all([
                key not in params or params[key] == value
                for key, value in output_keys.items() if key != 'var'
            ]):
                # All parameters correspond! This .SolverOutput node can be simply updated just by modifying its "var" node
                update_existing_solver_output(solver_ouput, output_keys)
                return name

    root_node.setParameters(name, **output_keys)
    return name
                
def process_extractions_of_type_bc_and_integral(workflow):

    familiesBC = get_familiesBC_nodes(workflow.tree)

    for Extraction in workflow.Extractions:
        if Extraction['Type'] not in ['Integral', 'BC']: 
            continue 

        families_to_extract = get_bc_families_to_extract(workflow, Extraction, familiesBC)

        for family in families_to_extract:
            add_2d_extractions_in_SolverOutput(family, Extraction, workflow)

def add_2d_extractions_in_SolverOutput(FamilyNode, Extraction, workflow):
    
    bc_type = FamilyNode.get(Name='FamilyBC').value()

    fields_to_extract = adapt_variables_for_2d_extraction(workflow, Extraction, bc_type)

    if fields_to_extract != []:

        elsa_var_list = translate_to_elsa(fields_to_extract, type='var')       
        output_keys = get_BC_solver_output_params(workflow, Extraction, bc_type, elsa_var_list)

        # solver_output_name = '.Solver#Output#1'
        # SolverOutput_node = FamilyNode.get(Name=solver_output_name, Depth=1)
        # if not SolverOutput_node:
        #     FamilyNode.setParameters(solver_output_name, **output_keys)
        # else:
        #     n = 2
        #     while SolverOutput_node is not None and n < 100:
        #         solver_output_name = f'.Solver#Output#{n}'
        #         SolverOutput_node = FamilyNode.get(Name=solver_output_name, Depth=1)
        #         n += 1
        #     FamilyNode.setParameters(solver_output_name, **output_keys)
        # Extraction['_ElsaSolverOutputName'] = solver_output_name
        solver_output_name = add_variables_in_right_solver_output(FamilyNode, output_keys)
        Extraction['_ElsaSolverOutputName'] = solver_output_name

    else:
        raise MolaException(f'the list of fields to extract on family {FamilyNode.name()} is empty')

def update_existing_solver_output(SolverOutput_node, output_keys):
    for key, value in output_keys.items():
        if key != 'var':
            # update node value, or add new node if it was not already in the tree
            cgns.Node(Parent=SolverOutput_node, Name=key, Value=value, Type='DataArray')
        else:
            var_node = SolverOutput_node.get(Name='var')
            var_value = var_node.value()
            if isinstance(var_value, str):
                var_value = [var_value]
            if isinstance(value, str):
                value = [value]
            for element in value:
                if element not in var_value:
                    var_value.append(element)
            var_node.setValue(var_value)

def get_BC_solver_output_params(workflow, Extraction, bc_type, elsa_var_list) -> dict:


    output_keys = dict(
        period        = Extraction["ExtractionPeriod"],

        # TODO make ticket:
        # BUG with writingmode=2 and Cfdpb.compute() (required by unsteady overset) 
        # wall extractions ignored during coprocess
        # BEWARE : contradiction in doc :  http://elsa.onera.fr/restricted/MU_tuto/latest/MU-98057/Textes/Attribute/extract.html#extract.writingmode 
        #                        versus :  http://elsa.onera.fr/restricted/MU_tuto/latest/MU_Annexe/CGNS/CGNS.html#Solver-Output
        writingmode   = 2, # NOTE requires extract_filtering='inactive'

        fluxcoeff     = 1.0,
        writingframe  = Extraction['Frame'],
    )

    
    if Extraction["Type"] == "BC":
        requested_location = Extraction["GridLocation"]
        if requested_location == "CellCenter":
            output_keys["loc"] = 'interface'
        elif requested_location == "Vertex":
            output_keys["loc"] = 'node'
        else:
            extraction_name = Extraction["Name"]
            raise MolaException(f"requested location {requested_location} for Extraction {extraction_name} not supported for elsA")
    
    elif Extraction["Type"] == "Integral":
        output_keys["loc"] = 'interface' # always required by elsA for integrals
        
    else:
        raise MolaException('UNEXPECTED TYPE WHEN SETTING loc TO SOLVER OUTPUT AT EXTRACTION'+Extraction['Name'])

    
    is_wall = 'Wall' in bc_type
    is_inviscid_wall = is_wall and ('Inviscid' in bc_type or workflow.Turbulence['Model']=='Euler')
    is_viscous_wall = is_wall and not is_inviscid_wall

    if is_wall:
        output_keys.update(dict(
            pinf = workflow.Flow['Pressure'],
            torquecoeff   = 1.0,
            xtorque       = 0.0,
            ytorque       = 0.0,
            ztorque       = 0.0,
        ))


        if is_viscous_wall:

            if Extraction['Frame'] == 'absolute' and not workflow.tree.isStructured():
                output_keys["writingframe"] = "relative" # TODO identify elsA ticket
                mola_logger.user_warning(f"Extraction {Extraction['Name']} requested absolute frame, but elsA cannot extract bc wall quantities in absolute frame for not structured grids. Switching to relative.")
            
            boundary_layer_requested = any([v.startswith('bl_') for v in elsa_var_list])
            
            if boundary_layer_requested:
            
                output_keys.update(dict(
                    delta_compute = workflow.SolverParameters['model']['delta_compute'],
                    vortratiolim  = workflow.SolverParameters['model']['vortratiolim'],
                    shearratiolim = workflow.SolverParameters['model']['shearratiolim'],
                    pressratiolim = workflow.SolverParameters['model']['pressratiolim'],
                    geomdepdom    = 2, # see #8127#note-26
                    delta_cell_max= 300,
                ))

    if "OtherOptions" in Extraction:
        output_keys.update(Extraction["OtherOptions"])
        output_keys.update(Extraction["OtherOptions"])

    output_keys['var'] = elsa_var_list

    return output_keys

def adapt_variables_for_2d_extraction(workflow, Extraction, ExtractBCType):
    ExtractVariablesList = copy.deepcopy(Extraction['Fields'])

    # TODO since in unstructured it is now possible
    if not workflow.tree.isStructured():
        if 'BoundaryLayer' in ExtractVariablesList:
            ExtractVariablesList.remove('BoundaryLayer')

    if ExtractBCType == 'BCWallInviscid':
        ViscousKeys = ['BoundaryLayer', 'yPlus', 'SkinFriction',
                       'geomdepdom','delta_cell_max','delta_compute',
                       'vortratiolim','shearratiolim','pressratiolim']
        for vk in ViscousKeys:
            try:
                ExtractVariablesList.remove(vk)
            except ValueError:
                pass
    else:

        if 'TransitionMode' in workflow.Turbulence:
            if workflow.Turbulence['TransitionMode'] == 'NonLocalCriteria-LSTT':
                extraVariables = ['intermittency', 'clim', 'how', 'origin',
                                'lambda2', 'turb_level', 'n_tot_ag', 'n_crit_ag',
                                'r_tcrit_ahd', 'r_theta_t1', 'line_status', 'crit_indicator']
                ExtractVariablesList.extend(extraVariables)

            elif workflow.Turbulence['TransitionMode'] == 'Imposed':
                extraVariables = ['intermittency', 'clim']
                ExtractVariablesList.extend(extraVariables)
    
    return ExtractVariablesList



def add_trigger(t, coprocessFilename=names.FILE_COPROCESS):
    '''
    Add ``.Solver#Trigger`` node to all zones.

    Parameters
    ----------

        t : PyTree
            the main tree. It is modified.

        coprocessFilename : str
            the name of the coprocess file.

    '''
    FamilyName = cgns.Node(Name='ELSA_TRIGGER', Type='AdditionalFamilyName', Value='ELSA_TRIGGER')
    for zone in t.zones():
        zone.addChild(FamilyName)

    Family = cgns.Node(Name='ELSA_TRIGGER', Type='Family')
    for base in t.bases():
        base.addChild(Family)

    AllZonesFamilyNodes = t.group(Name='ELSA_TRIGGER', Type='Family', Depth=2)
    for n in AllZonesFamilyNodes:
        n.setParameters('.Solver#Trigger',
                 next_state=16,
                 next_iteration=1,
                 file=coprocessFilename)

