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

from treelab import cgns
import mola.naming_conventions as names
from mola.logging import mola_logger, MolaException
from mola.cfd.preprocess.solver_specific_tools.solver_elsa import translate_to_elsa

import copy

# FIXME Check the writingframe, following what has been done in mola v1

def apply_to_solver(workflow):

    add_extractions_for_restart(workflow)
    add_extractions_for_overset_components(workflow)
    process_extractions_3d(workflow)
    process_extractions_2d(workflow)
    process_extractions_1d(workflow)
    add_trigger(workflow.tree)
    for Extraction in workflow.Extractions: 
        if Extraction['Type'] == 'Residuals':
            add_global_convergence_history(workflow, Extraction['ExtractionPeriod'])
            # In elsA, the extraction period is defined by add_global_convergence_history
            # Hence, the update of residuals by MOLA can be done at SavePeriod (more is useless)
            Extraction['ExtractionPeriod'] = Extraction['SavePeriod']
        elif Extraction['Type'] == 'Integral':
            Extraction['ExtractionPeriod'] = Extraction['SavePeriod']
            

def add_extractions_for_overset_components(workflow):
    if workflow.has_overset_component():
        workflow._interface.add_to_Extractions_3D(
            Fields    = list(workflow.Flow['Conservatives']), 
            Container = 'FlowSolution#Overset', 
            Frame     = 'absolute'
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
        )

def process_extractions_3d(workflow):

    # For 3D averaged field : 
    #   dict(type='3D', Container='FlowSolution#Average', fields=[...], options=dict(average='time', period_init='inactive'))

    # For coordinates : 
    #    dict(type='3D', Container='FlowSolution#EndOfRun#Coords', fields=['CoordinateX', 'CoordinateY', 'CoordinateZ'], GridLocation='Vertex', Frame='absolute')

    for zone in workflow.tree.zones():
        for Extraction in workflow.Extractions:
            if Extraction['Type'] in ['3D', 'Restart'] and is_zone_in_extraction_family(zone, Extraction):

                add_3d_extraction_to_zone(zone, Extraction)

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

def add_3d_extraction_to_zone(zone, Extraction):
    
    if 'Container' in Extraction:
        EoRnode = zone.get(Name=Extraction['Container'], Type='FlowSolution', Depth=1) 
    else:
        EoRnode = None

    options = Extraction.get('OtherOptions', dict())
    if not EoRnode:
        create_new_container_for_3d_extraction(zone, Extraction['Fields'], Extraction['Container'], 
                                               Extraction['GridLocation'], Extraction['Frame'], options)
    else:
        add_3d_extraction_to_existing_container(EoRnode, Extraction['Fields'], Extraction['GridLocation'], Extraction['Frame'])

def create_new_container_for_3d_extraction(zone, Fields2Extract, container_name, GridLocation, frame, OtherOptions):
    EoRnode = zone.setParameters(container_name, 
                                ContainerType='FlowSolution', 
                                **dict((field, None) for field in Fields2Extract)
                                )
    cgns.Node(Parent=EoRnode, Name='GridLocation', Type='GridLocation', Value=GridLocation)
    EoRnode.setParameters('.Solver#Output',
                            period=1,
                            writingmode=2,
                            writingframe=frame,
                            **OtherOptions)
    
def add_3d_extraction_to_existing_container(Container, Fields2Extract, GridLocation, frame):
    try:
        # Check compatibility
        ExistingGridLocation = Container.get(Type='GridLocation', Depth=1)
        assert GridLocation == ExistingGridLocation.value()

        writingframe = Container.get(Name='writingframe')
        assert frame == writingframe.value()

        # Add variables that are not already in this FlowSolution
        for field in Fields2Extract:
            if not Container.get(Name=field, Type='DataArray', Depth=1):
                cgns.Node(Parent=Container, Name=field, Type='DataArray')

    except AssertionError:
        raise MolaException('several 3D extractions are incompatible together')


def process_extractions_2d(workflow):

    families = workflow.tree.group(Type='Family', Depth=2)
    familiesBC = [node for node in families if node.get(Type='FamilyBC', Depth=1)]

    for Extraction in workflow.Extractions:

        if Extraction['Type'] != 'BC': continue 

        requested_source = Extraction['Source']

        for familyBC in familiesBC:

            family = familyBC.parent()
            family_name = family.name()
            bc_type = familyBC.value() 

            if requested_source not in [family_name, bc_type]: continue # TODO : allow regex ?
            
            add_2d_extractions_in_SolverOutput(family, Extraction, workflow)


def process_extractions_1d(workflow):
    ...


def add_2d_extractions_in_SolverOutput(FamilyNode, Extraction, workflow):
    
    bc_type = FamilyNode.get(Name='FamilyBC').value()

    fields_to_extract = adapt_variables_for_2d_extraction(workflow, Extraction, bc_type)

    if fields_to_extract != []:

        elsa_var_list = translate_to_elsa(fields_to_extract, type='var')       

        solver_output_name = '.Solver#Output#'+Extraction['Name'] # note that we may have several outputs (e.g. different requested frames)

        raise_error_if_solver_output_already_defined(solver_output_name, FamilyNode)

        output_keys = get_BC_solver_output_params(workflow, Extraction, bc_type, elsa_var_list)

        FamilyNode.setParameters(solver_output_name, **output_keys)
        
    else:
        mola_logger.warning(f'Caution: the list of fields to extract on family {FamilyNode.name()} is empty')


def raise_error_if_solver_output_already_defined(solver_output_name, FamilyNode):
    
    solver_output_already_defined = bool(FamilyNode.get(Name=solver_output_name, Depth=1))

    if solver_output_already_defined:
        raise MolaException(f'{solver_output_name} already defined in {FamilyNode.path()}')



def get_BC_solver_output_params(workflow, Extraction, bc_type, elsa_var_list) -> dict:

    # get loc
    requested_location = Extraction["GridLocation"]
    if requested_location == "CellCenter":
        loc = 'interface'
    elif requested_location == "Vertex":
        loc = 'node'
    else:
        extraction_name = Extraction["Name"]
        raise MolaException(f"requested location {requested_location} for Extraction {extraction_name} not supported for elsA")


    output_keys = dict(
        period        = Extraction["ExtractionPeriod"],

        # TODO make ticket:
        # BUG with writingmode=2 and Cfdpb.compute() (required by unsteady overset) 
        # wall extractions ignored during coprocess
        # BEWARE : contradiction in doc :  http://elsa.onera.fr/restricted/MU_tuto/latest/MU-98057/Textes/Attribute/extract.html#extract.writingmode 
        #                        versus :  http://elsa.onera.fr/restricted/MU_tuto/latest/MU_Annexe/CGNS/CGNS.html#Solver-Output
        writingmode   = 2, # NOTE requires extract_filtering='inactive'

        loc           = loc,
        fluxcoeff     = 1.0,
        writingframe  = Extraction['Frame'],
    )

    is_wall = 'Wall' in bc_type
    is_inviscid_wall = is_wall and 'Inviscid' in bc_type
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
                output_keys["writingframe"] = "relative"
                mola_logger.warning(f"Extraction {Extraction['Name']} requested absolute frame, but elsA cannot extract bc wall quantities in absolute frame for not structured grids. Switching to relative.")
            
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

    if not workflow.tree.isStructured():
        if 'BoundaryLayer' in ExtractVariablesList:
            ExtractVariablesList.remove('BoundaryLayer')

    if ExtractBCType == 'BCWallInviscid':
        ViscousKeys = ['BoundaryLayer', 'yPlus', 
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

