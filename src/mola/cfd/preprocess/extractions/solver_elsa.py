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

from mola import (cgns, misc)
from mola.cfd.preprocess.solver_specific_tools.solver_elsa import translate_to_elsa

import copy

# FIXME Check the writingframe, following what has been done in mola v1

def adapt_to_solver(workflow):

    add_extractions_for_overset_components(workflow)
    process_extractions_3d(workflow)
    process_extractions_2d(workflow)
    add_trigger(workflow.tree)
    add_global_convergence_history(workflow)

def add_extractions_for_overset_components(workflow):
    if workflow.has_overset_component():
        workflow.Extractions.append(
            dict(
                type      = '3D', 
                fields    = workflow.Flow['Conservatives'], 
                Container = 'FlowSolution#EndOfRun#Relative', 
                Frame     = 'relative'
            )
        )

def add_global_convergence_history(workflow):
    for base in workflow.tree.bases():
        GlobalConvergenceHistory = cgns.Node(Parent=base, Name='GlobalConvergenceHistory', Value=0, Type='ConvergenceHistory')
        cgns.Node(Parent=GlobalConvergenceHistory, Name='NormDefinitions', Value='ConvergenceHistory', Type='Descriptor')

def process_extractions_3d(workflow):

    workflow.tree.findAndRemoveNodes(Name='FlowSolution#EndOfRun', Type='FlowSolution')

    for zone in workflow.tree.zones():

        for Extraction in workflow.Extractions:
            if Extraction['type'] != '3D':
                continue

            # Filter by Family
            Family = Extraction.get('Family', None)
            if Family:
                if not zone.get(Type='FamilyName', Value=Family, Depth=1) \
                 and not zone.get(Type='AditionnalFamilyName', Value=Family, Depth=1):
                    continue

            Container = Extraction.get('Container', 'FlowSolution#EndOfRun')
            GridLocation = Extraction.get('GridLocation', 'CellCenter')
            Frame = Extraction.get('Frame', 'absolute')
            Fields2Extract = Extraction['fields']

            EoRnode = zone.get(Name=Container, Type='FlowSolution', Depth=1) 
            if not EoRnode:
                # Creation of a new FlowSolution node
                EoRnode = zone.setParameters(Container, 
                                            ContainerType='FlowSolution', 
                                            **dict((field, None) for field in Fields2Extract)
                                            )
                cgns.Node(Parent=EoRnode, Name='GridLocation', Type='GridLocation', Value=GridLocation)
                EoRnode.setParameters('.Solver#Output',
                                        period=1,
                                        writingmode=2,
                                        writingframe=Frame)
            else:
                # Check compatibility
                try:
                    ExistingGridLocation = EoRnode.get(Type='GridLocation', Depth=1)
                    assert GridLocation == ExistingGridLocation.value()

                    writingframe = EoRnode.get(Name='writingframe')
                    assert Frame == writingframe.value()

                except AssertionError:
                    print(misc.RED+'several 3D extractions are incompatible together'+misc.ENDC)

                # Add variables that are not already in this FlowSolution
                for field in Fields2Extract:
                    if not EoRnode.get(Name=field, Type='DataArray', Depth=1):
                        cgns.Node(Parent=EoRnode, Name=field, Type='DataArray')

def process_extractions_2d(workflow):

    # Default keys to write in the .Solver#Output of the Family node
    # The node 'var' will be fill later depending on the BCType
    BCKeys = dict(
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
    BCWallKeys = dict()
    BCWallKeys.update(BCKeys)
    BCWallKeys.update(dict(
        delta_compute = workflow.SolverParameters['model']['delta_compute'],
        vortratiolim  = workflow.SolverParameters['model']['vortratiolim'],
        shearratiolim = workflow.SolverParameters['model']['shearratiolim'],
        pressratiolim = workflow.SolverParameters['model']['pressratiolim'],
        pinf          = workflow.Flow['Pressure'],
        torquecoeff   = 1.0,
        xtorque       = 0.0,
        ytorque       = 0.0,
        ztorque       = 0.0,
        writingframe  = 'relative', # absolute incompatible with unstructured mesh
    ))
    
    FamilyNodes = workflow.tree.group(Type='Family', Depth=2)

    AllBCExtractions = []
    for Extraction in workflow.Extractions:
        if Extraction['type'] == 'bc':
            AllBCExtractions.append(Extraction['BCType'])

    for Extraction in workflow.Extractions:

        if Extraction['type'] != 'bc':
            continue

        # TODO : manage the case with no BCType given but a Family instead
        ExtractBCTypeRequired = Extraction['BCType'] # It may contain *
        ExtractVariablesListDefault = Extraction['fields']

        for FamilyNode in FamilyNodes:
            FamilyBCNode = FamilyNode.get(Type='FamilyBC', Value=ExtractBCTypeRequired, Depth=1)
            if FamilyBCNode:
                ExtractVariablesList = copy.deepcopy(ExtractVariablesListDefault)
                ExtractBCType = FamilyBCNode.value()

                if not workflow.tree.isStructured():
                    if 'BoundaryLayer' in Extraction['fields']:
                        Extraction['fields'].remove('BoundaryLayer')

                if ExtractBCType == 'BCWallInviscid':
                    ViscousKeys = [
                        'BoundaryLayer', 'yPlus',
                        'geomdepdom','delta_cell_max','delta_compute',
                        'vortratiolim','shearratiolim','pressratiolim']
                    for vk in ViscousKeys:
                        try:
                            ExtractVariablesList.remove(vk)
                        except ValueError:
                            pass
                else:

                    if workflow.Turbulence['TransitionMode'] == 'NonLocalCriteria-LSTT':
                        extraVariables = ['intermittency', 'clim', 'how', 'origin',
                            'lambda2', 'turb_level', 'n_tot_ag', 'n_crit_ag',
                            'r_tcrit_ahd', 'r_theta_t1', 'line_status', 'crit_indicator']
                        ExtractVariablesList.extend(extraVariables)

                    elif workflow.Turbulence['TransitionMode'] == 'Imposed':
                        extraVariables = ['intermittency', 'clim']
                        ExtractVariablesList.extend(extraVariables)

                if ExtractVariablesList != []:
                    varList = translate_to_elsa(ExtractVariablesList)
                    SolverOutput = FamilyNode.get(Name='.Solver#Output', Depth=1) 
                    
                    if not SolverOutput:
                        print('setting .Solver#Output to FamilyNode '+FamilyNode.name())
                        if 'BCWall' in ExtractBCType:
                            SolverOutputKeys = dict(**BCWallKeys, var=' '.join(varList))
                        else:
                            SolverOutputKeys = dict(**BCKeys, var=' '.join(varList))
                        FamilyNode.setParameters('.Solver#Output', **SolverOutputKeys)
                    else:
                        print('adding variables in .Solver#Output to FamilyNode '+FamilyNode.name())
                        # Add variables that are not already in the node
                        varNode = SolverOutput.get(Name='var', Depth=1)
                        varListAlreadyPresent = varNode.value().split() 
                        newVarList = copy.deepcopy(varListAlreadyPresent)
                        for var in varList:
                            if not var in varListAlreadyPresent:
                                newVarList.append(var)
                        varNode.setValue(' '.join(newVarList))
                else:
                    print(misc.YELLOW+f'Caution: the list of fields to extract on {FamilyNode.name()} is empty'+misc.ENDC)
                    # raise ValueError(misc.RED+f'Did not added anything since:\nExtractVariablesList={ExtractVariablesList}'+misc.ENDC)


def add_trigger(t, coprocessFilename='coprocess.py'):
    '''
    Add ``.Solver#Trigger`` node to all zones.

    Parameters
    ----------

        t : PyTree
            the main tree. It is modified.

        coprocessFilename : str
            the name of the coprocess file.

            .. note:: it is recommended using ``'coprocess.py'``

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

