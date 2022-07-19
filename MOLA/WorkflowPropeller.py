'''
MOLA - WorkflowPropeller.py

WORKFLOW PROPELLER

Collection of functions designed for CFD simulations of propellers in axial
flight conditions

File history:
22/03/2022 - L. Bernardos - Creation
'''

import os
import numpy as np

import Converter.PyTree    as C
import Converter.Internal  as I
import Distributor2.PyTree as D2
import Post.PyTree         as P
import Generator.PyTree    as G
import Transform.PyTree    as T
import Connector.PyTree    as X

from . import InternalShortcuts as J
from . import Wireframe as W
from . import Preprocess        as PRE
from . import JobManager        as JM
from . import WorkflowCompressor as WC

def prepareMesh4ElsA(mesh='mesh.cgns', splitAndDistributeOptions={'maximum_allowed_nodes':3},
                     match_tolerance=1e-7, periodic_match_tolerance=1e-7):
    '''
    Exactly like :py:func:`MOLA.Preprocess.prepareMesh4ElsA`
    '''
    if isinstance(mesh,str):
        t = C.convertFile2PyTree(mesh)
    elif I.isTopTree(mesh):
        t = mesh
    else:
        raise ValueError('parameter mesh must be either a filename or a PyTree')

    blade_number = getPropellerKinematic(t)[0]
    InputMeshes = [dict(file='mesh.cgns',
                        baseName='Base',
                        SplitBlocks=True,
                        BoundaryConditions=[
                            dict(name='blade_wall',
                                 type='FamilySpecified:BLADE',
                                 familySpecifiedType='BCWall'),
                            dict(name='spinner_wall',
                                 type='FamilySpecified:SPINNER',
                                 familySpecifiedType='BCWall'),
                            dict(name='farfield',
                                 type='FamilySpecified:FARFIELD',
                                 familySpecifiedType='BCFarfield',
                                 location='special',
                                 specialLocation='fillEmpty')],
                        Connection=[
                            dict(type='Match',
                                 tolerance=match_tolerance),
                            dict(type='PeriodicMatch',
                                 tolerance=periodic_match_tolerance,
                                 rotationCenter=[0.,0.,0.],
                                 rotationAngle=[360./float(blade_number),0.,0.])])]



    return PRE.prepareMesh4ElsA(InputMeshes, splitAndDistributeOptions)

def cleanMeshFromAutogrid(t, **kwargs):
    '''
    Exactly like :py:func:`MOLA.WorkflowCompressor.cleanMeshFromAutogrid`
    '''
    return WC.cleanMeshFromAutogrid(t, **kwargs)

def prepareMainCGNS4ElsA(mesh='mesh.cgns',
        RPM=0., AxialVelocity=0., ReferenceTurbulenceSetAtRelativeSpan=0.75,
        SpinnerRotationInterval=(-1e6,+1e6),
        ReferenceValuesParams=dict(
            FieldsAdditionalExtractions=['q_criterion'],
            CoprocessOptions=dict(
                RequestedStatistics=['std-Thrust','std-Power'],
                ConvergenceCriteria=[dict(Family='BLADE',
                                          Variable='std-Thrust',
                                          Threshold=1e-3)],
                AveragingIterations = 1000,
                ItersMinEvenIfConverged = 1000,
                UpdateArraysFrequency = 100,
                UpdateSurfacesFrequency = 500,
                UpdateFieldsFrequency = 2000)),
        NumericalParams={},
        Extractions=[dict(type='AllBCWall'),
                     dict(type='IsoSurface',field='CoordinateX',value=1.0),
                     dict(type='IsoSurface',field='CoordinateX',value=2.0),
                     dict(type='IsoSurface',field='q_criterion',value=10.0)],
        writeOutputFields=True,
        Initialization={'method':'uniform'},
        JobInformation={},
        FULL_CGNS_MODE=False,
        COPY_TEMPLATES=True):
    '''
    This is mainly a function similar to :func:`MOLA.Preprocess.prepareMainCGNS4ElsA`
    but adapted to propeller mono-chanel computations. Its purpose is adapting
    the CGNS to elsA, setting numerical and physical parameters as well as
    extractions and convergence criteria.

    Parameters
    ----------

        mesh : :py:class:`str` or PyTree
            if the input is a :py:class:`str`, then such string specifies the
            path to file (usually named ``mesh.cgns``) where the result of
            function :py:func:`prepareMesh4ElsA` has been writen. Otherwise,
            **mesh** can directly be the PyTree resulting from :func:`prepareMesh4ElsA`

        RPM : float
            revolutions per minute of the blade

        AxialVelocity : float
            axial (advance) velocity of the propeller in :math:`m/s`

        ReferenceTurbulenceSetAtRelativeSpan : float
            relative span (radial) position used for computing the kinematic
            velocity employed as reference for setting freestream turbulence
            quantities

        ReferenceValuesParams : dict
            Python dictionary containing the
            Reference Values and other relevant data of the specific case to be
            run using elsA. For information on acceptable values, please
            see the documentation of function :func:`computeReferenceValues`.

            .. note:: internally, this dictionary is passed as *kwargs* as follows:

                >>> MOLA.Preprocess.computeReferenceValues(arg, **ReferenceValuesParams)

        NumericalParams : dict
            dictionary containing the numerical
            settings for elsA. For information on acceptable values, please see
            the documentation of function :func:`MOLA.Preprocess.getElsAkeysNumerics`

            .. note:: internally, this dictionary is passed as *kwargs* as follows:

                >>> MOLA.Preprocess.getElsAkeysNumerics(arg, **NumericalParams)

        Extractions : :py:class:`list` of :py:class:`dict`
            List of extractions to perform during the simulation. See
            documentation of :func:`MOLA.Preprocess.prepareMainCGNS4ElsA`

        writeOutputFields : bool
            if :py:obj:`True`, write initialized fields overriding
            a possibly existing ``OUTPUT/fields.cgns`` file. If :py:obj:`False`, no
            ``OUTPUT/fields.cgns`` file is writen, but in this case the user must
            provide a compatible ``OUTPUT/fields.cgns`` file to elsA (for example,
            using a previous computation result).

        Initialization : dict
            dictionary defining the type of initialization, using the key
            **method**. See documentation of :func:`MOLA.Preprocess.initializeFlowSolution`

        JobInformation : dict
            Dictionary containing information to update the job file. For
            information on acceptable values, please see the documentation of
            function :func:`MOLA.JobManager.updateJobFile`

        FULL_CGNS_MODE : bool
            if :py:obj:`True`, put all elsA keys in a node ``.Solver#Compute``
            to run in full CGNS mode.

        COPY_TEMPLATES : bool
            If :py:obj:`True` (default value), copy templates files in the
            current directory.
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

    ReferenceValuesParamsDefault = dict(
        FieldsAdditionalExtractions=['q_criterion'],
        CoprocessOptions=dict(
            RequestedStatistics=['std-Thrust','std-Power'],
            ConvergenceCriteria=[dict(Family='BLADE',
                                      Variable='std-Thrust',
                                      Threshold=1e-3)],
            AveragingIterations = 1000,
            ItersMinEvenIfConverged = 1000,
            UpdateArraysFrequency = 100,
            UpdateSurfacesFrequency = 500,
            UpdateFieldsFrequency = 2000))

    ReferenceValuesParams.update(ReferenceValuesParamsDefault)
    ReferenceValuesParams['Velocity'] = AxialVelocity

    def addFieldExtraction(fieldname):
        print('adding %s'%fieldname)
        try:
            ReferenceValuesParams['FieldsAdditionalExtractions'].append(fieldname)
        except:
            ReferenceValuesParams['FieldsAdditionalExtractions'] = [fieldname]


    if isinstance(mesh,str):
        t = C.convertFile2PyTree(mesh)
    elif I.isTopTree(mesh):
        t = mesh
    else:
        raise ValueError('parameter mesh must be either a filename or a PyTree')

    nb_blades, Dir = getPropellerKinematic(t)
    span = maximumSpan(t)


    hasBCOverlap = True if C.extractBCOfType(t, 'BCOverlap') else False

    if hasBCOverlap: addFieldExtraction('ChimeraCellType')

    IsUnstructured = PRE.hasAnyUnstructuredZones(t)

    omega = -Dir * RPM * np.pi / 30.

    TangentialVelocity = abs(omega)*span*ReferenceTurbulenceSetAtRelativeSpan
    VelocityUsedForScalingAndTurbulence = np.sqrt(TangentialVelocity**2 +
                                                  AxialVelocity**2)

    ReferenceValuesParams['VelocityUsedForScalingAndTurbulence'] = VelocityUsedForScalingAndTurbulence

    RowTurboConfDict = {}
    for b in I.getBases(t):
        RowTurboConfDict[b[0]+'Zones'] = {'RotationSpeed':omega,
                                          'NumberOfBlades':nb_blades,
                                          'NumberOfBladesInInitialMesh':nb_blades}
    TurboConfiguration = WC.getTurboConfiguration(t, ShaftRotationSpeed=omega,
                                HubRotationSpeed=[SpinnerRotationInterval],
                                Rows=RowTurboConfDict)
    FluidProperties = PRE.computeFluidProperties()
    if not 'Surface' in ReferenceValuesParams:
        ReferenceValuesParams['Surface'] = 1.0

    MainDirection = np.array([1,0,0]) # Strong assumption here
    YawAxis = np.array([0,0,1])
    PitchAxis = np.cross(YawAxis, MainDirection)
    ReferenceValuesParams.update(dict(PitchAxis=PitchAxis, YawAxis=YawAxis))

    ReferenceValues = PRE.computeReferenceValues(FluidProperties,
                                                 **ReferenceValuesParams)
    ReferenceValues['RPM'] = RPM
    ReferenceValues['Workflow'] = 'Propeller'
    ReferenceValues['NumberOfBlades'] = nb_blades
    ReferenceValues['AxialVelocity'] = AxialVelocity
    ReferenceValues['MaximumSpan'] = span

    if I.getNodeFromName(t, 'proc'):
        JobInformation['NumberOfProcessors'] = int(max(PRE.getProc(t))+1)
        Splitter = None
    else:
        JobInformation['NumberOfProcessors'] = 'free'
        Splitter = 'PyPart'

    elsAkeysCFD      = PRE.getElsAkeysCFD(nomatch_linem_tol=1e-6, unstructured=IsUnstructured)
    elsAkeysModel    = PRE.getElsAkeysModel(FluidProperties, ReferenceValues, unstructured=IsUnstructured)
    elsAkeysNumerics = PRE.getElsAkeysNumerics(ReferenceValues,
                            unstructured=IsUnstructured, **NumericalParams)

    PRE.initializeFlowSolution(t, Initialization, ReferenceValues)

    WC.setBoundaryConditions(t, {}, TurboConfiguration,
                            FluidProperties, ReferenceValues)

    WC.computeFluxCoefByRow(t, ReferenceValues, TurboConfiguration)

    AllSetupDics = dict(FluidProperties=FluidProperties,
                        ReferenceValues=ReferenceValues,
                        elsAkeysCFD=elsAkeysCFD,
                        elsAkeysModel=elsAkeysModel,
                        elsAkeysNumerics=elsAkeysNumerics,
                        TurboConfiguration=TurboConfiguration,
                        Extractions=Extractions,
                        Splitter=Splitter,
                        JobInformation=JobInformation)

    PRE.addTrigger(t)
    PRE.addExtractions(t, AllSetupDics['ReferenceValues'],
                          AllSetupDics['elsAkeysModel'], extractCoords=False)

    if elsAkeysNumerics['time_algo'] != 'steady':
        PRE.addAverageFieldExtractions(t, AllSetupDics['ReferenceValues'],
            AllSetupDics['ReferenceValues']['CoprocessOptions']['FirstIterationForAverage'])

    PRE.addReferenceState(t, AllSetupDics['FluidProperties'],
                         AllSetupDics['ReferenceValues'])
    dim = int(AllSetupDics['elsAkeysCFD']['config'][0])
    PRE.addGoverningEquations(t, dim=dim)
    AllSetupDics['ReferenceValues']['NumberOfProcessors'] = int(max(PRE.getProc(t))+1)
    PRE.writeSetup(AllSetupDics)

    if FULL_CGNS_MODE:
        PRE.addElsAKeys2CGNS(t, [AllSetupDics['elsAkeysCFD'],
                                 AllSetupDics['elsAkeysModel'],
                                 AllSetupDics['elsAkeysNumerics']])

    PRE.saveMainCGNSwithLinkToOutputFields(t,writeOutputFields=writeOutputFields)

    if not Splitter:
        print('REMEMBER : configuration shall be run using %s%d%s procs'%(J.CYAN,
                                                   JobInformation['NumberOfProcessors'],J.ENDC))
    else:
        print('REMEMBER : configuration shall be run using %s'%(J.CYAN + \
            Splitter + J.ENDC))

    if COPY_TEMPLATES:
        JM.getTemplates('Standard', otherWorkflowFiles=['monitor_loads.py'],
                JobInformation=JobInformation)

def getPropellerKinematic(t):
    mesh_params = I.getNodeFromName(t,'.MeshingParameters')
    if mesh_params is None:
        raise ValueError(J.FAIL+'node .MeshingParameters not found in tree'+J.ENDC)

    try:
        nb_blades = int(I.getValue(I.getNodeFromName(mesh_params,'blade_number')))
    except:
        ERRMSG = 'could not find .MeshingParameters/blade_number in tree'
        raise ValueError(J.FAIL+ERRMSG+J.ENDC)

    try:
        Dir = int(I.getValue(I.getNodeFromName(mesh_params,'RightHandRuleRotation')))
        Dir = +1 if Dir else -1
    except:
        ERRMSG = 'could not find .MeshingParameters/RightHandRuleRotation in tree'
        raise ValueError(J.FAIL+ERRMSG+J.ENDC)

    return nb_blades, Dir

def maximumSpan(t):
    zones = C.extractBCOfName(t,'FamilySpecified:BLADE')
    W.addDistanceRespectToLine(zones, [0,0,0],[-1,0,0],FieldNameToAdd='span')
    return C.getMaxValue(zones,'span')

def _extendArraysWithPropellerQuantities(arrays, IntegralDataName, setup):
    arraysSubset = arrays[IntegralDataName]

    try:
        FX = arraysSubset['MomentumXFlux']
        MX = arraysSubset['TorqueX']
        blade_number = setup.ReferenceValues['NumberOfBlades']
        RPM = setup.ReferenceValues['RPM']
        AxialVelocity = setup.ReferenceValues['AxialVelocity']
        Density = setup.ReferenceValues['Density']
        span = setup.ReferenceValues['MaximumSpan']
    except KeyError:
        return

    RPS = RPM / 60.
    diameter = span * 2

    Thrust = - blade_number * FX
    Power = blade_number * MX * RPM * np.pi / 30.
    CT = Thrust / (Density * RPS**2 * diameter**4)
    CP = Power / (Density * RPS**3 * diameter**5)
    FM = np.sqrt(2./np.pi)* np.sign(CT)*np.abs(CT)**1.5 / CP
    eta = AxialVelocity*Thrust/Power

    arraysSubset['Thrust']=Thrust
    arraysSubset['Power']=Power
    arraysSubset['CT']=CT
    arraysSubset['CP']=CP
    arraysSubset['FigureOfMeritHover']=FM
    arraysSubset['PropulsiveEfficiency']=eta
