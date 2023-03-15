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

'''
MOLA - WorkflowPropeller.py

WORKFLOW PROPELLER

Collection of functions designed for CFD simulations of propellers in axial
flight conditions

File history:
22/03/2022 - L. Bernardos - Creation
'''

import MOLA
if not MOLA.__ONLY_DOC__:
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

def prepareMesh4ElsA(mesh='mesh.cgns', splitOptions={'maximum_allowed_nodes':3},
                     match_tolerance=1e-7, periodic_match_tolerance=1e-7):
    '''
    An adaptation of  :py:func:`MOLA.Preprocess.prepareMesh4ElsA`.

    Parameters
    ----------

        mesh : :py:class:`str`
            Mesh filename issued from automatic mesh generation, including BC families.

        splitOptions : dict
            Exactly the keyword arguments of :py:func:`~MOLA.Preprocess.splitAndDistribute`

        match_tolerance : float
            small tolerance for constructing the match connectivity.

        periodic_match_tolerance : float
            small tolerance for constructing the periodic match connectivity

    Returns
    -------

        t : tree
            resulting preprocessed tree, that can be sent to :py:func:`prepareMainCGNS4ElsA`
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



    return PRE.prepareMesh4ElsA(InputMeshes, splitOptions=splitOptions)

def clean_mesh_from_autogrid(t, **kwargs):
    '''
    Exactly like :py:func:`MOLA.WorkflowCompressor.clean_mesh_from_autogrid`
    '''
    return WC.clean_mesh_from_autogrid(t, **kwargs)

def prepareMainCGNS4ElsA(mesh='mesh.cgns',
        RPM=0., AxialVelocity=0., ReferenceTurbulenceSetAtRelativeSpan=0.75,
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
        OverrideSolverKeys={},
        Extractions=[dict(type='AllBCWall'),
                     dict(type='IsoSurface',field='CoordinateX',value=1.0),
                     dict(type='IsoSurface',field='CoordinateX',value=2.0),
                     dict(type='IsoSurface',field='q_criterion',value=20.0)],
        writeOutputFields=True,
        Initialization={'method':'uniform'},
        JobInformation={},
        SubmitJob=False,
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

        OverrideSolverKeys : :py:class:`dict` of maximum 3 :py:class:`dict`
            exactly the same as in :py:func:`MOLA.Preprocess.prepareMainCGNS4ElsA`

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

        SubmitJob : bool
            if :py:obj:`True`, submit the SLURM job based on information contained
            in **JobInformation**

            .. note::
                only relevant if **COPY_TEMPLATES** is py:obj:`True` and
                **JobInformation** is provided

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
    toc = J.tic()
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
    SpinnerRotationInterval=(-1e6,+1e6)
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
    ReferenceValues['NumberOfBlades'] = nb_blades
    ReferenceValues['AxialVelocity'] = AxialVelocity
    ReferenceValues['MaximumSpan'] = span

    if I.getNodeFromName(t, 'proc'):
        JobInformation['NumberOfProcessors'] = int(max(PRE.getProc(t))+1)
        Splitter = None
    else:
        Splitter = 'PyPart'

    elsAkeysCFD      = PRE.getElsAkeysCFD(nomatch_linem_tol=1e-6, unstructured=IsUnstructured)
    elsAkeysModel    = PRE.getElsAkeysModel(FluidProperties, ReferenceValues, unstructured=IsUnstructured)
    elsAkeysNumerics = PRE.getElsAkeysNumerics(ReferenceValues,
                            unstructured=IsUnstructured, **NumericalParams)

    PRE.initializeFlowSolution(t, Initialization, ReferenceValues)

    WC.setMotionForRowsFamilies(t, TurboConfiguration)
    WC.setBC_Walls(t, TurboConfiguration)

    WC.computeFluxCoefByRow(t, ReferenceValues, TurboConfiguration)

    allowed_override_objects = ['cfdpb','numerics','model']
    for v in OverrideSolverKeys:
        if v == 'cfdpb':
            elsAkeysCFD.update(OverrideSolverKeys[v])
        elif v == 'numerics':
            elsAkeysNumerics.update(OverrideSolverKeys[v])
        elif v == 'model':
            elsAkeysModel.update(OverrideSolverKeys[v])
        else:
            raise AttributeError('OverrideSolverKeys "%s" must be one of %s'%(v,
                                                str(allowed_override_objects)))

    AllSetupDicts = dict(Workflow='Propeller',
                        Splitter=Splitter,
                        JobInformation=JobInformation,
                        TurboConfiguration=TurboConfiguration,
                        FluidProperties=FluidProperties,
                        ReferenceValues=ReferenceValues,
                        elsAkeysCFD=elsAkeysCFD,
                        elsAkeysModel=elsAkeysModel,
                        elsAkeysNumerics=elsAkeysNumerics,
                        Extractions=Extractions)

    PRE.addTrigger(t)
    PRE.addExtractions(t, AllSetupDicts['ReferenceValues'],
                          AllSetupDicts['elsAkeysModel'], extractCoords=False,
                          BCExtractions=ReferenceValues['BCExtractions'])

    if elsAkeysNumerics['time_algo'] != 'steady':
        PRE.addAverageFieldExtractions(t, AllSetupDicts['ReferenceValues'],
            AllSetupDicts['ReferenceValues']['CoprocessOptions']['FirstIterationForAverage'])

    PRE.addReferenceState(t, AllSetupDicts['FluidProperties'],
                         AllSetupDicts['ReferenceValues'])
    dim = int(AllSetupDicts['elsAkeysCFD']['config'][0])
    PRE.addGoverningEquations(t, dim=dim)
    AllSetupDicts['ReferenceValues']['NumberOfProcessors'] = int(max(PRE.getProc(t))+1)
    PRE.writeSetup(AllSetupDicts)

    if FULL_CGNS_MODE:
        PRE.addElsAKeys2CGNS(t, [AllSetupDicts['elsAkeysCFD'],
                                 AllSetupDicts['elsAkeysModel'],
                                 AllSetupDicts['elsAkeysNumerics']])

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
        if 'DIRECTORY_WORK' in JobInformation:
            PRE.sendSimulationFiles(JobInformation['DIRECTORY_WORK'],
                                    overrideFields=writeOutputFields)

        for i in range(SubmitJob):
            singleton = False if i==0 else True
            JM.submitJob(JobInformation['DIRECTORY_WORK'], singleton=singleton)

    ElapsedTime = str(PRE.datetime.timedelta(seconds=J.tic()-toc))
    hours, minutes, seconds = ElapsedTime.split(':')
    ElapsedTimeHuman = hours+' hours '+minutes+' minutes and '+seconds+' seconds'
    msg = 'prepareMainCGNS took '+ElapsedTimeHuman
    print(J.BOLD+msg+J.ENDC)


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
