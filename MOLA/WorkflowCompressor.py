'''
MOLA - WorkflowCompressor.py

WORKFLOW COMPRESSOR

Collection of functions designed for Workflow Compressor

BEWARE:
There is no equivalent of Preprocess ``prepareMesh4ElsA``.
``prepareMainCGNS4ElsA`` takes as an input a CGNS file assuming that the following
elements are already set:
    * connectivities
    * boundary conditions
    * splitting and distribution
    * families
    * (optional) parametrization with channel height in a ``FlowSolution#Height`` node

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
        NumericalParams={}, TurboConfiguration={}, PostParameters={},
        BodyForceInputData=[], writeOutputFields=True):
    '''
    This is mainly a function similar to Preprocess :py:func:`prepareMainCGNS4ElsA`
    but adapted to compressor computations. Its purpose is adapting the CGNS to
    elsA.

    Parameters
    ----------

        t : PyTree
            A grid that already contains:
                * connectivities
                * boundary conditions
                * splitting and distribution
                * families
                * (optional) parametrization with channel height in a ``FlowSolution#Height`` node

    Returns
    -------

        files : None
            A number of files are written:

            * ``main.cgns``
                main CGNS file to be read directly by elsA

            * ``OUTPUT/fields.cgns``
                file containing the initial fields (if ``writeOutputFields=True``)

            * ``setup.py``
                ultra-light file containing all relevant info of the simulation
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
    ReferenceValues = computeReferenceValues(FluidProperties,
                                             **ReferenceValuesParams)

    NProc = max([I.getNodeFromName(z,'proc')[1][0][0] for z in I.getZones(t)])+1
    ReferenceValues['NProc'] = int(NProc)
    ReferenceValuesParams['NProc'] = int(NProc)
    elsAkeysCFD      = PRE.getElsAkeysCFD()
    elsAkeysModel    = PRE.getElsAkeysModel(FluidProperties, ReferenceValues)
    if BodyForceInputData: NumericalParams['useBodyForce'] = True
    elsAkeysNumerics = PRE.getElsAkeysNumerics(ReferenceValues, **NumericalParams)
    TurboConfiguration = setTurboConfiguration(**TurboConfiguration)
    PostParameters = setPostParameters(**PostParameters)

    AllSetupDics = dict(FluidProperties=FluidProperties,
                        ReferenceValues=ReferenceValues,
                        elsAkeysCFD=elsAkeysCFD,
                        elsAkeysModel=elsAkeysModel,
                        elsAkeysNumerics=elsAkeysNumerics,
                        TurboConfiguration=TurboConfiguration,
                        PostParameters=PostParameters)
    if BodyForceInputData: AllSetupDics['BodyForceInputData'] = BodyForceInputData

    t = newCGNSfromSetup(t, AllSetupDics, initializeFlow=True, FULL_CGNS_MODE=False)
    to = PRE.newRestartFieldsFromCGNS(t)
    PRE.saveMainCGNSwithLinkToOutputFields(t,to,writeOutputFields=writeOutputFields)


    print('REMEMBER : configuration shall be run using %s%d%s procs'%(J.CYAN,
                                               ReferenceValues['NProc'],J.ENDC))


def computeReferenceValues(FluidProperties, Massflow, PressureStagnation,
        TemperatureStagnation, Surface, TurbulenceLevel=0.001,
        Viscosity_EddyMolecularRatio=0.1, TurbulenceModel='Wilcox2006-klim',
        TurbulenceCutoff=1.0, TransitionMode=None, CoprocessOptions={},
        Length=1.0, TorqueOrigin=[0., 0., 0.],
        FieldsAdditionalExtractions='ViscosityMolecular Viscosity_EddyMolecularRatio Pressure Temperature PressureStagnation TemperatureStagnation Mach'):
    '''
    This function is the Compressor's equivalent of :py:func:`PRE.computeReferenceValues()`.
    The main difference is that in this case reference values are set through
    ``Massflow``, total Pressure ``PressureStagnation``, total Temperature
    ``TemperatureStagnation`` and ``Surface``.

    Please, refer to :py:func:`PRE.computeReferenceValues()` doc for more details.
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
        )

    ReferenceValues.update(addKeys)

    return ReferenceValues


def setTurboConfiguration(ShaftRotationSpeed=0., HubRotationSpeed=[], Rows={}):
    '''
    Construct a dictionary concerning the compressor properties.

    Parameters
    ----------

        ShaftRotationSpeed : py:class:`float`
            Shaft speed in rad/s

            .. attention:: only for single shaft configuration

            .. attention:: Pay attention to the sign of **ShaftRotationSpeed**

        HubRotationSpeed : :py:class:list of :py:class:tuple
            Hub rotation speed. Each tuple (``xmin``, ``xmax``) corresponds to a
            ``CoordinateX`` interval where the speed at hub wall is
            ``ShaftRotationSpeed``. It is zero outside these intervals.

        Rows : py:class:`dict`
            This dictionary has one entry for each row domain. The key names
            must be the family names in the CGNS Tree.
            For each family name, the following entries are expected:

                * RotationSpeed : py:class:`float` or py:class:`str`
                    Rotation speed in rad/s. Set ``'auto'`` to automatically
                    set ``ShaftRotationSpeed``.

                    .. attention:: Use **RotationSpeed**=``'auto'`` for rotors
                    only.

                    .. attention:: Pay attention to the sign of
                    **RotationSpeed**

                * NumberOfBlades : py:class:`int`
                    The number of blades in the row

                * NumberOfBladesSimulated : py:class:`int`
                    The number of blades in the computational domain. Set to
                    ``<NumberOfBlades>`` for a full 360 simulation.

                * PlaneIn : py:class:`float`, optional
                    Position (in ``CoordinateX``) of the inlet plane for this
                    row. This plane is used for post-processing and convergence
                    monitoring.

                * PlaneOut : py:class:`float`, optional
                    Position of the outlet plane for this row.

    Returns
    -------

        TurboConfiguration : :py:class:`dict`
            set of compressor properties
    '''

    TurboConfiguration = dict(
        ShaftRotationSpeed = ShaftRotationSpeed,
        HubRotationSpeed   = HubRotationSpeed,
        Rows               = Rows
        )
    for row, rowParams in TurboConfiguration['Rows'].items():
        for key, value in rowParams.items():
            if key == 'RotationSpeed' and value == 'auto':
                rowParams[key] = ShaftRotationSpeed

    return TurboConfiguration

def setPostParameters(IsoSurfaces={}, Variables=[]):
    '''
    Construct a dictionary with the Co(Post)processing options.

    Parameters
    ----------

        IsoSurfaces : :py:class:`dict`
            dictionary defining the isoSurfaces to compute. Each key sets a
            variable, and the associated value is a list defining the levels.
            For example:

            ::

                IsoSurfaces   = dict(
                    CoordinateX   = [-0.2432, 0.0398],
                    ChannelHeight = [0.1, 0.5, 0.9]
                    )

        Variables : :py:class:`list` of :py:class:`str`, optional
            list of the variable names to compute and add on the isoSurfaces.
            They are computed by the function :py:func:`computeVariables` in
            Cassiopee Post module and must be among the following entries:
            ``'Pressure'``, ``'PressureStagnation'``,
            ``'TemperatureStagnation'``, ``'Entropy'``

    Returns
    -------

        PostParameters : :py:class:`dict`
            set of Co(Post)processing options
    '''
    PostParameters = dict(
        IsoSurfaces   = IsoSurfaces,
        Variables     = Variables
        )
    return PostParameters

def getRotationSpeedOfRows(t):
    '''
    Get the rotationnal speed of each row in the PyTree ``<t>``

    Parameters
    ----------

        t : PyTree
            PyTree with declared families (Family_t) for each row with a
            ``.Solver#Motion`` node.

    Returns
    -------

        omegaDict : :py:class:`dict`
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
    This is mainly a function similar to Preprocess :py:func:`newCGNSfromSetup`
    but adapted to compressor computations. Its purpose is creating the main
    CGNS tree and writes the ``setup.py`` file.

    The only differences with Preprocess newCGNSfromSetup are:
    #. addSolverBC is not applied, in order not to disturb the special
       turbomachinery BCs
    #. extraction of coordinates in ``FlowSolution#EndOfRun#Coords`` is
        desactivated
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


def massflowFromMach(Mx, S, Pt=101325.0, Tt=288.25, r=287.053, gamma=1.4):
    '''
    Compute the massflow rate through a section.

    Parameters
    ----------

        Mx : :py:class:`float`
            Mach number in the normal direction to the section.

        S : :py:class:`float`
            Surface of the section.

        Pt : :py:class:`float`
            Stagnation pressure of the flow.

        Tt : :py:class:`float`
            Stagnation temperature of the flow.

        r : :py:class:`float`
            Specific gas constant.

        gamma : :py:class:`float`
            Ratio of specific heats of the gas.


    Returns
    -------

        massflow : :py:class:`float`
            Value of massflow through the section.
    '''
    return S * Pt * (gamma/r/Tt)**0.5 * Mx / (1. + 0.5*(gamma-1.) * Mx**2) ** ((gamma+1) / 2 / (gamma-1))

def machFromMassflow(massflow, S, Pt=101325.0, Tt=288.25, r=287.053, gamma=1.4):
    '''
    Compute the Mach number normal to a section from the massflow rate.

    Parameters
    ----------

        massflow : :py:class:`float`
            Massflow rate through the section.

        S : :py:class:`float`
            Surface of the section.

        Pt : :py:class:`float`
            Stagnation pressure of the flow.

        Tt : :py:class:`float`
            Stagnation temperature of the flow.

        r : :py:class:`float`
            Specific gas constant.

        gamma : :py:class:`float`
            Ratio of specific heats of the gas.


    Returns
    -------

        Mx : :py:class:`float`
            Value of the Mach number in the normal direction to the section.
    '''
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
