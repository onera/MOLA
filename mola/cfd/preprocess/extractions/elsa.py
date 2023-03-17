#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

from mola import (cgns, misc)
from ..boundary_conditions.elsa import getFamilyBCTypeFromFamilyBCName

import copy

import Converter.elsAProfile as EP


def adapt_to_solver(workflow):

    if workflow.has_overset_component():
        workflow.extractions.append(
            dict(
                type      = '3D', 
                fields    = workflow.Flow['Conservatives'], 
                Container = 'FlowSolution#EndOfRun#Relative', 
                Frame     = 'relative'
            )
        )

    process_extractions_3d(workflow)
    process_extractions_2d(workflow)
    add_trigger(workflow.tree)
    EP._addGlobalConvergenceHistory(workflow.tree)
    cgns.castNode(workflow.tree)


def process_extractions_3d(workflow):

    workflow.tree.findAndRemoveNodes(Name='FlowSolution#EndOfRun', Type='FlowSolution')

    for zone in workflow.tree.zones():

        for Extraction in workflow.Extractions():
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
            Fields2Extract = translate_to_elsa(Extraction['fields'])

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
                        cgns.Node(Pärent=EoRnode, Name=field, Type='DataArray')


def addAverageFieldExtractions(t, ReferenceValues, firstIterationForAverage=1):
    '''
    Include time averaged fields extraction information to CGNS tree using
    information contained in dictionary **ReferenceValues**.

    Parameters
    ----------

        t : PyTree
            prepared grid as produced by :py:func:`prepareMesh4ElsA` function.

            .. note:: tree **t** is modified

        ReferenceValues : dict
            dictionary as produced by :py:func:`computeReferenceValues` function

        firstIterationForAverage : int
            Iteration to start the computation of time average. All the following iterations
            will be taken into account to compute the average.

    '''

    Fields2Extract = ReferenceValues['Fields'] + ReferenceValues['FieldsAdditionalExtractions']

    for zone in I.getZones(t):

        EoRnode = I.createNode('FlowSolution#EndOfRun#Average', 'FlowSolution_t',
                                parent=zone)
        I.createNode('GridLocation','GridLocation_t', value='CellCenter', parent=EoRnode)
        for fieldName in Fields2Extract:
            I.createNode(fieldName, 'DataArray_t', value=None, parent=EoRnode)
        J.set(EoRnode, '.Solver#Output',
              period=1,
              writingmode=2,
              writingframe='absolute',
              average='time',
              period_init=firstIterationForAverage,  #First iteration to consider to compute time average
               )


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
        delta_compute = workflow.SolverParamters['model']['delta_compute'],
        vortratiolim  = workflow.SolverParamters['model']['vortratiolim'],
        shearratiolim = workflow.SolverParamters['model']['shearratiolim'],
        pressratiolim = workflow.SolverParamters['model']['pressratiolim'],
        pinf          = workflow.Flow['Pressure'],
        torquecoeff   = 1.0,
        xtorque       = 0.0,
        ytorque       = 0.0,
        ztorque       = 0.0,
        writingframe  = 'relative', # absolute incompatible with unstructured mesh
        geomdepdom    = 2,  # see #8127#note-26
        delta_cell_max= 300,
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
        ExtractBCType = Extraction['BCType']
        ExtractVariablesListDefault = Extraction['fields']

        for FamilyNode in FamilyNodes:
            ExtractVariablesList = copy.deepcopy(ExtractVariablesListDefault)

            if FamilyNode.get(Type='FamilyBC', Name=ExtractBCType, Depth=1):

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
                    varDict = dict(var=' '.join(ExtractVariablesList))
                    print('setting .Solver#Output to FamilyNode '+FamilyNode.name())
                    if 'BCWall' in ExtractBCType:
                        BCWallKeys.update(varDict)
                        FamilyNode.setParameters('.Solver#Output', **translate_to_elsa(BCWallKeys))
                    else:
                        BCKeys.update(varDict)
                        FamilyNode.setParameters('.Solver#Output',**translate_to_elsa(BCKeys))
                else:
                    raise ValueError(misc.RED+f'Did not added anything since:\nExtractVariablesList={ExtractVariablesList}'+misc.ENDC)


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
    FamilyName = cgns.Node(Name='ELSA_TRIGGER', Type='FamilyName', Value='ELSA_TRIGGER')
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


def translate_to_elsa(Variables):
    '''
    Translate names in **Variables** from CGNS standards to elsA names for
    boundary conditions.

    Parameters
    ----------

        Variables : :py:class:`dict` or :py:class:`list` or :py:class:`str`
            Could be eiter:

                * a :py:class:`dict` with keys corresponding to variables names

                * a :py:class:`list` of variables names

                * a :py:class:`str` as a single variable name

    Returns
    -------

        NewVariables : :py:class:`dict` or :py:class:`list` or :py:class:`str`
            Depending on the input type, return the same object with variable
            names translated to elsA standards.

    '''
    CGNS2ElsaDict = dict(
        PressureStagnation       = 'stagnation_pressure',
        EnthalpyStagnation       = 'stagnation_enthalpy',
        TemperatureStagnation    = 'stagnation_temperature',
        Pressure                 = 'pressure',
        MassFlow                 = 'globalmassflow',
        SurfacicMassFlow         = 'surf_massflow',
        VelocityUnitVectorX      = 'txv',
        VelocityUnitVectorY      = 'tyv',
        VelocityUnitVectorZ      = 'tzv',
        TurbulentSANuTilde       = 'inj_tur1',
        TurbulentEnergyKinetic   = 'inj_tur1',
        TurbulentDissipationRate = 'inj_tur2',
        TurbulentDissipation     = 'inj_tur2',
        TurbulentLengthScale     = 'inj_tur2',
        VelocityCorrelationXX    = 'inj_tur1',
        VelocityCorrelationXY    = 'inj_tur2', 
        VelocityCorrelationXZ    = 'inj_tur3',
        VelocityCorrelationYY    = 'inj_tur4', 
        VelocityCorrelationYZ    = 'inj_tur5', 
        VelocityCorrelationZZ    = 'inj_tur6',
        
        BoundaryLayer            = 'bl_quantities_2d bl_quantities_3d bl_ue',
        NormalVector             = 'normalvector',
        Friction                 = 'frictionvector', 
        yPlus                    = 'yplusmeshsize',
        MomentumFlux             = 'flux_rou flux_rov flux_row',
        TorqueFlux               = 'torque_rou torque_rov torque_row',

    )
    if 'VelocityCorrelationXX' in Variables:
        # For RSM models
        CGNS2ElsaDict['TurbulentDissipationRate'] = 'inj_tur7'

    elsAVariables = CGNS2ElsaDict.values()

    if isinstance(Variables, dict):
        NewVariables = dict()
        for var, value in Variables.items():
            if var in CGNS2ElsaDict:
                NewVariables[CGNS2ElsaDict[var]] = value
            else:
                NewVariables[var] = value
        return NewVariables
    elif isinstance(Variables, list):
        NewVariables = []
        for var in Variables:
            if var in elsAVariables:
                NewVariables.append(var)
            else:
                NewVariables.append(CGNS2ElsaDict[var])
        return NewVariables
    elif isinstance(Variables, str):
        if Variables in elsAVariables:
            return CGNS2ElsaDict[Variables]
    else:
        raise TypeError('Variables must be of type dict, list or string')

