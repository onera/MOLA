'''
MOLA - WorkflowCompressor.py

WORKFLOW COMPRESSOR

Collection of functions designed for Workflow Compressor

BEWARE:
There is no equivalent of Preprocess prepareMesh4ElsA.
prepareMainCGNS4ElsA takes as an input a CGNS file assuming that the following
elements are already set:
    #. connectivities
    #. boundary conditions
    #. splitting and distribution
    #. families
    #. (optional) parametrization with channel height in a FlowSolution#Height node

File history:
31/08/2021 - T. Bontemps - Creation
'''

import sys
import os
import numpy as np
import pprint
import scipy.optimize

# BEWARE: in Python v >= 3.4 rather use: importlib.reload(setup)
import imp


import Converter.PyTree as C
import Converter.Internal as I
import Distributor2.PyTree as D2

from . import InternalShortcuts as J
from . import Preprocess as PRE


def prepareMainCGNS4ElsA(FILE_MESH='mesh.cgns', ReferenceValuesParams={},
        NumericalParams={}, PostParameters={},
        BodyForceInputData=[], writeOutputFields=True):
    '''
    This is mainly a function similar to Preprocess prepareMainCGNS4ElsA
    but adapted to compressor computations. Its purpose is adapting the CGNS to
    elsA.

    INPUTS

    t - (PyTree or None) - the grid as produced by buildMesh()

    OUTPUT

    None. Writes setup.py, main.cgns and eventually OUTPUT/fields.cgns
    '''

    def addFieldExtraction(fieldname):
        try:
            FieldsExtr = ReferenceValuesParams['FieldsAdditionalExtractions']
            if fieldname not in FieldsExtr.split():
                FieldsExtr += ' '+fieldname
        except:
            ReferenceValuesParams['FieldsAdditionalExtractions'] = fieldname


    t = C.convertFile2PyTree(FILE_MESH)

    hasBCOverlap = True if C.extractBCOfType(t, 'BCOverlap') else False


    if hasBCOverlap: PRE.addFieldExtraction('ChimeraCellType')
    if BodyForceInputData: PRE.addFieldExtraction('Temperature')

    FluidProperties = PRE.computeFluidProperties()
    ReferenceValues = computeReferenceValues(t, FluidProperties,
                                             **ReferenceValuesParams)

    NProc = max([I.getNodeFromName(z,'proc')[1][0][0] for z in I.getZones(t)])+1
    ReferenceValues['NProc'] = int(NProc)
    ReferenceValuesParams['NProc'] = int(NProc)
    elsAkeysCFD      = PRE.getElsAkeysCFD()
    elsAkeysModel    = PRE.getElsAkeysModel(FluidProperties, ReferenceValues)
    if BodyForceInputData: NumericalParams['useBodyForce'] = True
    elsAkeysNumerics = PRE.getElsAkeysNumerics(ReferenceValues, **NumericalParams)
    #TurboConfiguration = setTurboConfiguration(**TurboConfiguration)
    PostParameters = setPostParameters(**PostParameters)

    AllSetupDics = dict(FluidProperties=FluidProperties,
                        ReferenceValues=ReferenceValues,
                        elsAkeysCFD=elsAkeysCFD,
                        elsAkeysModel=elsAkeysModel,
                        elsAkeysNumerics=elsAkeysNumerics,
                        #TurboConfiguration=TurboConfiguration,
                        PostParameters=PostParameters)
    if BodyForceInputData: AllSetupDics['BodyForceInputData'] = BodyForceInputData

    t = newCGNSfromSetup(t, AllSetupDics, initializeFlow=True, FULL_CGNS_MODE=False)
    to = PRE.newRestartFieldsFromCGNS(t)
    PRE.saveMainCGNSwithLinkToOutputFields(t,to,writeOutputFields=writeOutputFields)


    print('REMEMBER : configuration shall be run using %s%d%s procs'%(J.CYAN,
                                               ReferenceValues['NProc'],J.ENDC))


def computeReferenceValues(t, FluidProperties, Massflow, PressureStagnation,
        TemperatureStagnation, Surface, TurbulenceLevel=0.001,
        Viscosity_EddyMolecularRatio=0.1, TurbulenceModel='Wilcox2006-klim',
        TurbulenceCutoff=1.0, TransitionMode=None, CoprocessOptions={},
        Length=1.0, TorqueOrigin=[0., 0., 0.],
        FieldsAdditionalExtractions='ViscosityMolecular Viscosity_EddyMolecularRatio Pressure Temperature PressureStagnation TemperatureStagnation Mach'):
    '''
    This function is the Compressor's equivalent of PRE.computeReferenceValues().
    The main difference is that in this case reference values are set through
    Massflow, total Pressure Pt, total Temperature Tt and Surface.

    Please, refer to PRE.computeReferenceValues() doc for more details.
    '''
    # Fluid properties local shortcuts
    Gamma   = FluidProperties['Gamma']
    RealGas = FluidProperties['RealGas']
    cv      = FluidProperties['cv']
    cp      = FluidProperties['cp']

    # Compute variables
    Mach  = machFromMassflow(Massflow, Surface, Pt=PressureStagnation,
                            Tt=TemperatureStagnation)
    Temperature  = TemperatureStagnation / (1. + 0.5*(Gamma-1.) * Mach**2)
    Pressure  = PressureStagnation / (1. + 0.5*(Gamma-1.) * Mach**2)**(Gamma/(Gamma-1))
    Density = Pressure / (Temperature * RealGas)
    SoundSpeed  = np.sqrt(Gamma * RealGas * Temperature)
    Velocity  = Mach * SoundSpeed

    # REFERENCE VALUES COMPUTATION
    mus = FluidProperties['SutherlandViscosity']
    Ts  = FluidProperties['SutherlandTemperature']
    S   = FluidProperties['SutherlandConstant']
    ViscosityMolecular = mus * (Temperature/Ts)**1.5 * ((Ts + S)/(Temperature + S))

    ReferenceValues = PRE.computeReferenceValues(FluidProperties,
        Density=Density,
        Velocity=Velocity,
        Temperature=Temperature,
        AngleOfAttackDeg = 0.0,
        AngleOfSlipDeg = 0.0,
        YawAxis = [0.,0.,1.],
        PitchAxis = [0.,1.,0.],
        TurbulenceLevel=TurbulenceLevel,
        Surface=Surface,
        Length=Length,
        TorqueOrigin=TorqueOrigin,
        TurbulenceModel=TurbulenceModel,
        Viscosity_EddyMolecularRatio=Viscosity_EddyMolecularRatio,
        TurbulenceCutoff=TurbulenceCutoff,
        TransitionMode=TransitionMode,
        CoprocessOptions=CoprocessOptions,
        FieldsAdditionalExtractions=FieldsAdditionalExtractions)

    addKeys = dict(
        PressureStagnation = PressureStagnation,
        TemperatureStagnation = TemperatureStagnation,
        Massflow = Massflow,
        RotationSpeed = getRotationSpeedOfRows(t)
        )

    ReferenceValues.update(addKeys)

    return ReferenceValues


def setTurboConfiguration(t, rowNames):
    '''
    Construct a dictionary of values concerning the compressor properties.
    ::

        row_names = ''

    Returns
    -------

        TurboConfiguration : dict
            set of compressor properties
    '''

    TurboConfiguration = dict(
        rowNames      = ['row_1'],
        nb_blades     = [16],
        periodicities = [16],
        RotationSpeedDict = getRotationSpeedOfRows(t),
        x_fct         = [-999.0, 0.0742685]
        )
    return TurboConfiguration

def setPostParameters(IsoSurfaces={}, Variables=[]):
    '''
    Construct a dictionary with the Co(Post)processing options.

    Parameters
    ----------

        IsoSurfaces : dict
            dictionary defining the isoSurfaces to compute. Each key sets a
            variable, and the associated value is a list defining the levels.
            Example :
                IsoSurfaces   = dict(
                    CoordinateX   = [-0.2432, 0.0398],
                    ChannelHeight = [0.1, 0.5, 0.9]
                    )

        Variables : list of str
            list of the variable names to compute and add on the isoSurfaces.
            They are computed by the function computeVariables in Cassiopee
            Post module and must be among the following entries:
            'Pressure', 'PressureStagnation', 'TemperatureStagnation', 'Entropy'

    Returns
    -------

        PostParameters : dict
            set of Co(Post)processing options
    '''
    PostParameters = dict(
        IsoSurfaces   = IsoSurfaces,
        Variables     = Variables
        )
    return PostParameters

def getRotationSpeedOfRows(t):
    '''
    Get the rotationnal speed of each row in the PyTree <t>

    Parameters
    ----------

        t : PyTree
            PyTree with declared families (Family_t) for each row with a
            ``.Solver#Motion`` node.

    Returns
    -------

        omegaDict : dict
            dictionary with the rotation speed associated to each row family
            name.
    '''
    omegaDict = dict()
    for node in I.getNodesFromName(t, '.Solver#Motion'):
        rowNode, pos = I.getParentOfNode(t, node)
        if I.getType(rowNode) != 'Family_t':
            continue
        omega = I.getValue(I.getNodeFromName(node, 'omega'))
        rowName = I.getName(rowNode)
        omegaDict[rowName] = omega

    return omegaDict

def newCGNSfromSetup(t, AllSetupDictionaries, initializeFlow=True,
                     FULL_CGNS_MODE=False, dim=3):
    '''
    This is mainly a function similar to Preprocess newCGNSfromSetup
    but adapted to compressor computations. Its purpose is creating the main
    CGNS tree and writes the ``setup.py`` file.

    The only differences with Preprocess newCGNSfromSetup are:
    #. addSolverBC is not applied, in order not to disturb the special
       turbomachinery BCs
    #. extraction of coordinates in FlowSolution#EndOfRun#Coords is desactivated
    '''
    t = I.copyRef(t)

    PRE.addTrigger(t)
    PRE.addExtractions(t, AllSetupDictionaries['ReferenceValues'],
                      AllSetupDictionaries['elsAkeysModel'], extractCoords=False)
    PRE.addReferenceState(t, AllSetupDictionaries['FluidProperties'],
                         AllSetupDictionaries['ReferenceValues'])
    PRE.addGoverningEquations(t, dim=dim) # TODO replace dim input by elsAkeysCFD['config'] info
    if initializeFlow:
        PRE.newFlowSolutionInit(t, AllSetupDictionaries['ReferenceValues'])
    if FULL_CGNS_MODE:
        PRE.addElsAKeys2CGNS(t, [AllSetupDictionaries['elsAkeysCFD'],
                             AllSetupDictionaries['elsAkeysModel'],
                             AllSetupDictionaries['elsAkeysNumerics']])

    AllSetupDictionaries['ReferenceValues']['NProc'] = int(max(D2.getProc(t))+1)
    AllSetupDictionaries['ReferenceValues']['CoreNumberPerNode'] = 28

    PRE.writeSetup(AllSetupDictionaries)

    return t


def massflowFromMach(Mx, S, Pt=101325.0, Tt=288.25, r=287.04, gamma=1.4):
    return S * Pt * (gamma/r/Tt)**0.5 * Mx / (1. + 0.5*(gamma-1.) * Mx**2) ** ((gamma+1) / 2 / (gamma-1))

def machFromMassflow(massflow, S, Pt=101325.0, Tt=288.25, r=287.04, gamma=1.4):
    if isinstance(massflow, (list, tuple, np.ndarray)):
        Mx = []
        for i, MF in enumerate(massflow):
            Mx.append(machFromMassflow(MF, S, Pt=Pt, Tt=Tt, r=r, gamma=gamma))
        if isinstance(massflow, np.ndarray):
            Mx = np.array(Mx)
        return Mx
    else:
        # Check that massflow is lower than the chocked massflow
        chocked_massflow = massflowFromMach(1., S, Pt=Pt, Tt=Tt, r=r, gamma=gamma)
        assert massflow < chocked_massflow, "Massflow ({:6.3f}kg/s) is greater than the chocked massflow ({:6.3f}kg/s)".format(massflow, chocked_massflow)
        # Massflow as a function of Mach number
        f = lambda Mx: massflowFromMach(Mx, S, Pt, Tt, r, gamma)
        # Objective function
        g = lambda Mx: f(Mx) - massflow
        # Search for the corresponding Mach Number between 0 and 1
        Mx = scipy.optimize.brentq(g, 0, 1)
        return Mx
