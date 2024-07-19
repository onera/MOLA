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
    add_trigger(workflow.tree)
    for Extraction in workflow.Extractions: 
        if Extraction['Type'] == 'Residuals':
            add_global_convergence_history(workflow, Extraction['ExtractionPeriod'])
            # In elsA, the extraction period is defined by add_global_convergence_history
            # Hence, the update of residuals by MOLA can be done at SavePeriod (more is useless)
            Extraction['ExtractionPeriod'] = Extraction['SavePeriod']
            break

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
    EoRnode = zone.get(Name=Extraction['Container'], Type='FlowSolution', Depth=1) 
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

    # Get elsA parameters for extractions, depending on the type of the BC
    default_bc_parameters, default_bc_wall_parameters = get_default_parameters_for_2d_extractions(workflow.SolverParameters, workflow.Flow['Pressure'])
    
    FamilyNodes = workflow.tree.group(Type='Family', Depth=2)
    BCFamilyNodes = [node for node in FamilyNodes if node.get(Type='FamilyBC', Depth=1)]

    # Among all extractions, get all the BCType that are asked
    AllBCExtractions = []
    for Extraction in workflow.Extractions:
        if Extraction['Type'] == 'BC' and Extraction['Source'].startswith('BC'):
            AllBCExtractions.append(Extraction['Source'])

    for Extraction in workflow.Extractions:

        is_integral_on_bc = Extraction['Type'] == 'Integral' \
            and (Extraction['Source'].startswith('BC') or Extraction['Source'] in BCFamilyNodes)
        if Extraction['Type'] != 'BC' and not is_integral_on_bc:
            # extraction not handled with that function
            continue

        # TODO : manage the case with no BCType given but a Family instead
        ExtractBCTypeRequired = Extraction['Source'] # It may contain *

        for FamilyNode in BCFamilyNodes:
            FamilyBCNode = FamilyNode.get(Type='FamilyBC', Value=ExtractBCTypeRequired, Depth=1)
            if FamilyBCNode:
                ExtractBCType = FamilyBCNode.value()

                ExtractVariablesList = adapt_variables_for_2d_extraction(workflow, Extraction, ExtractBCType)
                add_2d_extractions_in_SolverOutput(FamilyNode, ExtractBCType, ExtractVariablesList, default_bc_parameters, default_bc_wall_parameters)

def get_default_parameters_for_2d_extractions(SolverParameters, pinf):
    # Default keys to write in the .Solver#Output of the Family node
    # The node 'var' will be fill later depending on the BCType
    default_bc_parameters = dict(
        period        = 1,

        # TODO make ticket:
        # BUG with writingmode=2 and Cfdpb.compute() (required by unsteady overset) 
        # wall extractions ignored during coprocess
        # BEWARE : contradiction in doc :  http://elsa.onera.fr/restricted/MU_tuto/latest/MU-98057/Textes/Attribute/extract.html#extract.writingmode 
        #                        versus :  http://elsa.onera.fr/restricted/MU_tuto/latest/MU_Annexe/CGNS/CGNS.html#Solver-Output
        writingmode   = 2, # NOTE requires extract_filtering='inactive'

        loc           = 'interface',
        fluxcoeff     = 1.0,
        writingframe  = 'absolute',
        geomdepdom    = 2, # see #8127#note-26
        delta_cell_max= 300,
    )

    # Keys to write in the .Solver#Output for wall Families
    default_bc_wall_parameters = dict()
    default_bc_wall_parameters.update(default_bc_parameters)
    default_bc_wall_parameters.update(dict(
        delta_compute = SolverParameters['model']['delta_compute'],
        vortratiolim  = SolverParameters['model']['vortratiolim'],
        shearratiolim = SolverParameters['model']['shearratiolim'],
        pressratiolim = SolverParameters['model']['pressratiolim'],
        pinf          = pinf,
        torquecoeff   = 1.0,
        xtorque       = 0.0,
        ytorque       = 0.0,
        ztorque       = 0.0,
        writingframe  = 'relative', # absolute incompatible with unstructured mesh
    ))
    return default_bc_parameters, default_bc_wall_parameters

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

def add_2d_extractions_in_SolverOutput(FamilyNode, ExtractBCType, ExtractVariablesList, default_bc_parameters, default_bc_wall_parameters):
    if ExtractVariablesList != []:
        varList = translate_to_elsa(ExtractVariablesList, type='var')
        SolverOutput = FamilyNode.get(Name='.Solver#Output', Depth=1) 
        
        if not SolverOutput:
            mola_logger.debug('setting .Solver#Output to FamilyNode '+FamilyNode.name())
            if 'BCWall' in ExtractBCType:
                SolverOutputKeys = dict(**default_bc_wall_parameters, var=' '.join(varList))
            else:
                SolverOutputKeys = dict(**default_bc_parameters, var=' '.join(varList))
            FamilyNode.setParameters('.Solver#Output', **SolverOutputKeys)
        else:
            mola_logger.debug('adding variables in .Solver#Output to FamilyNode '+FamilyNode.name())
            # Add variables that are not already in the node
            varNode = SolverOutput.get(Name='var', Depth=1)
            varListAlreadyPresent = varNode.value()
            if isinstance(varListAlreadyPresent, str):
                # only one variable in node var, so varListAlreadyPresent is a str
                # Careful, doing list(varListAlreadyPresent) gives a wrong result!
                # For example, list('psta') = ['p', 's', 't', 'a']
                varListAlreadyPresent = [varListAlreadyPresent]
            newVarList = copy.deepcopy(varListAlreadyPresent)
            for var in varList:
                if not var in varListAlreadyPresent:
                    newVarList.append(var)
            varNode.setValue(' '.join(newVarList))
    else:
        mola_logger.warning(f'Caution: the list of fields to extract on {FamilyNode.name()} is empty')


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

