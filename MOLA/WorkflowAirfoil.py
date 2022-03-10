'''
MOLA - WorkflowAirfoil.py

WORKFLOW AIRFOIL

Collection of functions designed for Workflow Airfoil

File history:
15/02/2021 - L. Bernardos - Creation
'''

import sys
import os
import numpy as np
import pprint

import Converter.PyTree as C
import Converter.Internal as I
import Geom.PyTree as D
import Transform.PyTree as T

from . import InternalShortcuts as J
from . import Preprocess as PRE
from . import GenerativeShapeDesign as GSD
from . import Wireframe as W
from . import JobManager as JM

CONVERGENCE_THRESHOLD = 1e-4

def checkDependencies():
    '''
    Make a series of functional tests in order to determine if the user
    environment is correctly set for using the Workflow Airfoil
    '''
    JM.checkDependencies()

    print('\nChecking XFoil...')
    from . import XFoil
    XFoilResults = XFoil.computePolars('naca 0012', [1e6], [0.1], [3.0], )
    print(J.GREEN+'XFoil OK'+J.ENDC)

    print('\nVERIFICATIONS TERMINATED')


def launchBasicStructuredPolars(PREFIX_JOB, FILE_GEOMETRY, AER, machine,
        DIRECTORY_WORK, AoARange, MachRange, ReynoldsOverMach,
        AdditionalCoprocessOptions={}, AdditionalTransitionZones={},
        ImposedWallFields=[], TransitionalModeReynoldsThreshold=0,
        ExtraElsAParams={}):
    '''
    User-level function designed to launch a structured-type polar of airfoil.

    Parameters
    ----------

        PREFIX_JOB : str
            an arbitrary prefix for the jobs

        FILE_GEOMETRY : str
            path where the airfoil coordinates file is
            located.

            .. attention:: the airfoil must exist at provided path

        AER : str
            full AER code for launching simulations on SATOR

        machine : str
            name of the machine ``'sator'``, ``'spiro'``, ``'eos'``...

            .. warning:: only ``'sator'`` has been tested

        DIRECTORY_WORK : str
            the working directory at computation server.

            .. note:: indicated directory may not exist. In this case, it will
                be created.

        AoARange : list
            angle-of-attack to consider. Increasing positive values
            must come first. Then, the decreasing negative values. For example:

            >>> [0, 2, 4, 6, 8, 9, 10, -1, -2, -3]

        MachRange : list
            mach numbers to consider

        ReynoldsOverMach : float
            :math:`Re/M` value to consider in order to
            compute Reynolds number for each case.

        AdditionalCoprocessOptions : dict
            values overriding the default set of ``CoprocessOptions``, which
            is:

            ::

                dict(UpdateFieldsFrequency   = 5000,
                     UpdateArraysFrequency   = 100,
                     UpdateSurfacesFrequency = 1000,
                     AveragingIterations     = 10000,
                     ItersMinEvenIfConverged = 3000,
                     TimeOutInSeconds        = 54000.0, # 15 h * 3600 s/h = 53100 s
                     InitialTimeOutInSeconds = 54000.0,
                     SecondsMargin4QuitBeforeTimeOut = 300.,
                     RequestedStatistics=['std-CL','std-CD','std-Cm'],
                     ConvergenceCriteria = [dict(Family='AIRFOIL',
                                                 Variable='std-CL',
                                                 Threshold=CONVERGENCE_THRESHOLD)])


        AdditionalTransitionZones : dict
            values overriding the default
            set of ``TransitionZones``, which defines the regions for transition
            criteria behavior.

            ::

                TransitionZones = dict( TopOrigin = 0.003,
                                        BottomOrigin = 0.05,
                                        TopLaminarImposedUpTo = 0.01,
                                        TopLaminarIfFailureUpTo = 0.07,
                                        TopTurbulentImposedFrom = 0.95,
                                        BottomLaminarImposedUpTo = 0.10,
                                        BottomLaminarIfFailureUpTo = 0.50,
                                        BottomTurbulentImposedFrom = 0.95)

            can be used as template

        ImposedWallFields : :py:class:`list` of :py:class:`dict`
            used to impose specific fields of intermittency or clim. For example:

            ::

                ImposedWallFields = [
                     dict(fieldname    = 'intermittency_imposed',
                          side         = 'top',
                          Xref         = TopXtr,
                          LengthScale  = TopLengthScale,
                          a_boost      = a_boost,
                          sa           = sa,
                          sb           = sb,
                          ),

                     dict(fieldname    = 'clim_imposed',
                          side         = 'top',
                          Xref         = TopXtr+TopLengthScale*sc,
                          LengthScale  = 1e-6,
                          a_boost      = 1.0,
                          sa           = 1e-3,
                          sb           = 0.0,
                          ),

                     dict(fieldname    = 'intermittency_imposed',
                          side         = 'bottom',
                          Xref         = BottomXtr,
                          LengthScale  = BottomLengthScale,
                          a_boost      = a_boost,
                          sa           = sa,
                          sb           = sb,
                          ),

                     dict(fieldname    = 'clim_imposed',
                          side         = 'bottom',
                          Xref         = BottomXtr+BottomLengthScale*sc,
                          LengthScale  = 1e-6,
                          a_boost      = 1.0,
                          sa           = 1e-3,
                          sb           = 0.0,
                          ),
                    ]

            can be used as template

        TransitionalModeReynoldsThreshold : float
            Reynolds number threshold
            value below which transitional computation is performed.

            .. hint:: For always making non-transitional computation, set this value to ``0``

        ExtraElsAParams : dict
            parameters overriding the default values
            of the attributes sent to :py:func:`prepareMainCGNS4ElsA` of this module.

    Returns
    -------

        None : None
            File ``JobsConfiguration.py`` is writen and polar builder job is
            launched
    '''

    # TODO appropriately sort AoARange
    a = np.atleast_1d(AoARange)
    a = np.unique(a)
    aP = a[a>=0]
    aM = a[a<0]
    aP = np.sort(aP)
    aM = np.sort(aM)[::-1]
    AoARange = np.hstack((aP,aM))

    MachRange = np.unique(MachRange)

    AoAMatrix, MachMatrix  = np.meshgrid(AoARange, MachRange)
    ReynoldsMatrix = MachMatrix * ReynoldsOverMach

    AoA_  =      AoAMatrix.ravel(order='K')
    M_    =     MachMatrix.ravel(order='K')
    Re_   = ReynoldsMatrix.ravel(order='K')
    NewJobs = AoA_ == AoARange[0]

    JobsQueues = []
    for i, (AoA, Reynolds, Mach, NewJob) in enumerate(zip(AoA_, Re_, M_, NewJobs)):

        print('Assembling run {} AoA={} Re={} M={} | NewJob = {}'.format(
                i, AoA, Reynolds, Mach, NewJob))

        if NewJob:
            JobName = PREFIX_JOB+'%d'%i
            writeOutputFields = True
        else:
            writeOutputFields = False

        CASE_LABEL = '%06.2f'%abs(AoA)+'_'+JobName # CAVEAT tol AoA >= 0.01 deg
        if AoA < 0: CASE_LABEL = 'M'+CASE_LABEL

        meshParams = getMeshingParameters()
        meshParams['References'].update({'Reynolds':Reynolds})
        if 'options' not in machine: meshParams['options'] = {}
        if machine == 'sator':
            meshParams['options']['Nproc']=28
        elif machine == 'sator-new':
            meshParams['options']['Nproc']=48
        elif machine == 'spiro':
            meshParams['options']['Nproc']=24
        elif machine in ['ld', 'eos']:
            meshParams['options']['Nproc']=8


        EffectiveMach = np.maximum(Mach, 0.2) # TODO externalize this threshold

        if ImposedWallFields:
            TransitionMode = 'Imposed'

        elif Reynolds <= TransitionalModeReynoldsThreshold:
            TransitionMode = 'NonLocalCriteria-LSTT'

        else:
            TransitionMode = None


        CoprocessOptions = dict(UpdateFieldsFrequency   = 5000,
                                UpdateArraysFrequency   = 100,
                                UpdateSurfacesFrequency = 1000,
                                AveragingIterations     = 10000,
                                ItersMinEvenIfConverged = 3000,
                                TimeOutInSeconds        = 54000.0, # 15 h * 3600 s/h = 53100 s
                                InitialTimeOutInSeconds = 54000.0,
                                SecondsMargin4QuitBeforeTimeOut = 300.,
                                RequestedStatistics=['std-CL','std-CD','std-Cm'],
                                ConvergenceCriteria = [dict(Family='AIRFOIL',
                                                            Variable='std-CL',
                                                            Threshold=CONVERGENCE_THRESHOLD)])
        CoprocessOptions.update(AdditionalCoprocessOptions)

        TransitionZones = dict(
                              TopOrigin = 0.003,
                           BottomOrigin = 0.05,
                  TopLaminarImposedUpTo = 0.01,
                TopLaminarIfFailureUpTo = 0.07,
                TopTurbulentImposedFrom = 0.95,
               BottomLaminarImposedUpTo = 0.10,
             BottomLaminarIfFailureUpTo = 0.50,
             BottomTurbulentImposedFrom = 0.95,
                               )

        TransitionZones.update(AdditionalTransitionZones)

        # ImposedWallFields = [
        #     dict(fieldname    = 'intermittency_imposed',
        #          side         = 'top',
        #          Xref         = TopXtr,
        #          LengthScale  = TopLengthScale,
        #          a_boost      = a_boost,
        #          sa           = sa,
        #          sb           = sb,
        #          ),

        #     dict(fieldname    = 'clim_imposed',
        #          side         = 'top',
        #          Xref         = TopXtr+TopLengthScale*sc,
        #          LengthScale  = 1e-6,
        #          a_boost      = 1.0,
        #          sa           = 1e-3,
        #          sb           = 0.0,
        #          ),

        #     dict(fieldname    = 'intermittency_imposed',
        #          side         = 'bottom',
        #          Xref         = BottomXtr,
        #          LengthScale  = BottomLengthScale,
        #          a_boost      = a_boost,
        #          sa           = sa,
        #          sb           = sb,
        #          ),

        #     dict(fieldname    = 'clim_imposed',
        #          side         = 'bottom',
        #          Xref         = BottomXtr+BottomLengthScale*sc,
        #          LengthScale  = 1e-6,
        #          a_boost      = 1.0,
        #          sa           = 1e-3,
        #          sb           = 0.0,
        #          ),
        # ]


        elsAParams = dict(Reynolds=Reynolds,
                          MachPolar=Mach,
                          Mach=EffectiveMach,
                          AngleOfAttackDeg=AoA,
                          writeOutputFields=writeOutputFields,
                          TransitionMode=TransitionMode,
                          TurbulenceLevel=0.1 * 1e-2,
                          TurbulenceModel='Wilcox2006-klim',
                          InitialIteration=1, NumberOfIterations=50000,
                          NumericalScheme='ausm+',
                          TimeMarching='steady')

        elsAParams.update(ExtraElsAParams)

        JobsQueues.append( dict(ID=i, CASE_LABEL=CASE_LABEL, NewJob=NewJob,
            JobName=JobName, meshParams=meshParams, elsAParams=elsAParams,
            CoprocessOptions=CoprocessOptions, TransitionZones=TransitionZones,
            ImposedWallFields=ImposedWallFields,) )

    JM.saveJobsConfiguration(JobsQueues, AER, machine, DIRECTORY_WORK,
                             FILE_GEOMETRY)

    JM.launchJobsConfiguration(templatesFolder=JM.MOLA_PATH+'/TEMPLATES/WORKFLOW_AIRFOIL')


def buildMesh(FILE_GEOMETRY,
              meshParams={'References':{'Reynolds':1e6,'DeltaYPlus':0.5}},
              save_meshParams=True,
              save_mesh=True):
    '''
    Builds the mesh around an airfoil

    Parameters
    ----------

        FILE_GEOMETRY : str
            location where airfoil geometry is located

        meshParams : dict
            literally, the optional parameters to pass to
            :py:func:`MOLA.GenerativeShapeDesign.extrudeAirfoil2D` function

        save_meshParams : bool
            if :py:obj:`True`, saves the ``meshParams.py`` file

        save_mesh : bool
            if :py:obj:`True`, saves the ``mesh.cgns`` file

    Returns
    -------

        t : PyTree
            resulting mesh tree

        meshParamsUpdated : :py:class:`dict`
            **meshParams** with additional parameters
            resulting from the grid constuction process
    '''
    if FILE_GEOMETRY.endswith('.dat') or FILE_GEOMETRY.endswith('.txt') or \
        '.' not in FILE_GEOMETRY:
        try:
            closeTol = meshParams['options']['TEclosureTolerance']
            airfoilCurve = W.airfoil(FILE_GEOMETRY, ClosedTolerance=closeTol)
        except KeyError:
            airfoilCurve = W.airfoil(FILE_GEOMETRY)
    else:
        airfoilCurve = C.convertFile2PyTree(FILE_GEOMETRY)
        airfoilCurve, = I.getZones(airfoilCurve)


    t, meshParamsUpdated = GSD.extrudeAirfoil2D(airfoilCurve, **meshParams)



    if save_meshParams:
        params2write='meshParams='+pprint.pformat(meshParamsUpdated)
        with open('meshParams.py','w') as f: f.write(params2write)

    if save_mesh: C.convertPyTree2File(t, 'mesh.cgns')

    return t, meshParamsUpdated


def prepareMainCGNS4ElsA(mesh, meshParams={},
                    writeOutputFields=True,
                    Reynolds=1e6, Mach=0.3,
                    MachPolar=None,
                    AngleOfAttackDeg=0.0,
                    TurbulenceLevel=0.001,
                    TransitionMode=None, TurbulenceModel='Wilcox2006-klim',
                    InitialIteration = 1, NumberOfIterations = 30000,
                    NumericalScheme='jameson',
                    TimeMarching='steady', timestep=0.01,
                    CFLparams=dict(vali=1.,valf=10.,iteri=1,
                                           iterf=1000,function_type='linear'),
                    CoprocessOptions={'ConvergenceCriteria': [dict(Family='AIRFOIL',
                                                Variable='std-CL',
                                                Threshold=CONVERGENCE_THRESHOLD)]},
                    ImposedWallFields=[], TransitionZones={},
                    FieldsAdditionalExtractions=['ViscosityMolecular',
                        'Viscosity_EddyMolecularRatio','Mach'],
                    Extractions=[{'type':'AllBCWall'}],
                    Initialization={'method':'uniform'}):
    '''
    This is mainly a function similar to :py:func:`MOLA.Preprocess.prepareMainCGNS4ElsA`
    but adapted to airfoil computations. Its purpose is adapting the CGNS to
    elsA.

    Parameters
    ----------

        mesh : :py:class:`str` or PyTree
            if the input is a :py:class:`str`, then such string specifies the
            path to file (usually named ``mesh.cgns``) where the result of
            function :py:func:`buildMesh` has been writen. Otherwise,
            **mesh** can directly be the PyTree resulting from :py:func:`buildMesh`

        meshParams : dict
            provide the mesh parameters dictionary as
            generated by :py:func:`buildMesh`

        writeOutputFields : bool
            write file ``OUTPUT/fields.cgns``

        Reynolds : float
            set the Reynolds number, :math:`Re_c`  conditions for the simulation

        Mach : float
            set the actual Mach number employed by the simulation.

            .. attention:: as elsA is a compressible solver, this value shall
                **not** be lower than ``0.2``

        MachPolar : float
            if provided, set the *virtual* Mach number
            considered in the Polars.

            .. note:: this value **can** be lower than ``0.2``, as the actual
                Mach number may be limited to 0.2

        AngleOfAttackDeg : float
            angle of attack in [degree]

        TurbulenceLevel : float
            turbulence level

        TransitionMode : str
            if provided, can be one of:
            ``'NonLocalCriteria-LSTT'``, ``'NonLocalCriteria-Step'``, ``'Imposed'``

        TurbulenceModel : str
            any turbulence model keyword supported by
            :py:func:`MOLA.Preprocess.prepareMainCGNS4ElsA`

        InitialIteration : int
            initial iteration of computation

        NumberOfIterations : int
            maximum number of iterations

        NumericalScheme : str
            any value accepted by :py:func:`MOLA.Preprocess.getElsAkeysNumerics`

        TimeMarching : str
            any value accepted by :py:func:`MOLA.Preprocess.getElsAkeysNumerics`

        timestep : float
            the timestep in unsteady simulations

            .. note:: see any value accepted by :py:func:`MOLA.Preprocess.getElsAkeysNumerics` doc

        CFLParams : dict
            any accepted value by any value accepted by :py:func:`MOLA.Preprocess.getElsAkeysNumerics`

        CoprocessOptions : dict
            any accepted value by :py:func:`MOLA.Preprocess.computeReferenceValues`

        ImposeWallFields : list
            any accepted value by :py:func:`launchBasicStructuredPolars`

        TransitionZones : dict
            any accepted value by :py:func:`launchBasicStructuredPolars`

        FieldsAdditionalExtractions : :py:class:`list` of :py:class:`str`
            list of CGNS names for extracting additional flowfield quantities

        Extractions : :py:class:`list` of :py:class:`dict`
            .. danger:: **doc this** # TODO

        Initialization : dict
            dictionary defining the type of initialization, using the key
            **method**. This latter is mandatory and should be one of the
            following:

            * **method** = 'uniform' : the Flow Solution is initialized uniformly
              using the **ReferenceValues**.

            * **method** = 'copy' : the Flow Solution is initialized by copying
              the FlowSolution container of another file. The file path is set by
              using the key **file**. The container might be set with the key
              **container** ('FlowSolution#Init' by default).

            * **method** = 'interpolate' : the Flow Solution is initialized by
              interpolating the FlowSolution container of another file. The file
              path is set by using the key **file**. The container might be set
              with the key **container** ('FlowSolution#Init' by default).

            Default method is 'uniform'.

    Returns
    -------

        None : None
            Writes ``setup.py``, ``main.cgns`` and eventually ``OUTPUT/fields.cgns``
    '''

    if isinstance(mesh,str):
        t = C.convertFile2PyTree(mesh)
    elif I.isTopTree(mesh):
        t = mesh
    else:
        raise ValueError('parameter mesh must be either a filename or a PyTree')

    if not t: t = C.convertFile2PyTree(file_mesh)
    else: t = I.copyRef(t)

    FluidProperties = PRE.computeFluidProperties()

    try:
        AirfoilWallFamilyName = meshParams['options']['AirfoilFamilyName']
    except:
        AirfoilWallFamilyName = 'AIRFOIL'
        print(J.WARN+'did not find AirfoilFamilyName in meshParams. Using "AIRFOIL"'+J.ENDC)

    NonLocalModes = ('NonLocalCriteria-LSTT','NonLocalCriteria-Step','Imposed')
    NonLocalTransition = True if TransitionMode in NonLocalModes else False

    if NonLocalTransition and 'intermittency' not in FieldsAdditionalExtractions:
        FieldsAdditionalExtractions.append('intermittency')


    CoprocessOpts = dict()
    CoprocessOpts.update(CoprocessOptions)

    ReferenceValues = computeReferenceValues(Reynolds, Mach, meshParams,
        FluidProperties,
        AngleOfAttackDeg=AngleOfAttackDeg,
        TurbulenceLevel=TurbulenceLevel,
        TurbulenceModel='Wilcox2006-klim',
        TransitionMode=TransitionMode,
        CoprocessOptions=CoprocessOpts,
        FieldsAdditionalExtractions=FieldsAdditionalExtractions,
                            )
    ReferenceValues['ImposedWallFields'] = ImposedWallFields
    ReferenceValues['TransitionZones'] = TransitionZones
    ReferenceValues['MachPolar']=MachPolar

    elsAkeysCFD      = PRE.getElsAkeysCFD(config='2d')
    elsAkeysModel    = PRE.getElsAkeysModel(FluidProperties, ReferenceValues)
    elsAkeysNumerics = PRE.getElsAkeysNumerics(ReferenceValues,
                                               NumericalScheme=NumericalScheme,
                                               TimeMarching=TimeMarching,
                                               timestep=timestep,
                                               inititer=InitialIteration,
                                               niter=NumberOfIterations,
                                               CFLparams=CFLparams)

    if NonLocalTransition:
        setBCDataSetWithNonLocalTransition(t, AirfoilWallFamilyName, ReferenceValues)

    AllSetupDics = dict(FluidProperties=FluidProperties,
                        ReferenceValues=ReferenceValues,
                        elsAkeysCFD=elsAkeysCFD,
                        elsAkeysModel=elsAkeysModel,
                        elsAkeysNumerics=elsAkeysNumerics,
                        Extractions=Extractions)


    t = PRE.newCGNSfromSetup(t, AllSetupDics, Initialization=Initialization,
                            FULL_CGNS_MODE=False)
    PRE.saveMainCGNSwithLinkToOutputFields(t, writeOutputFields=writeOutputFields)

    print(J.CYAN+'REMEMBER : configuration shall be run using %d procs'%ReferenceValues['NProc']+J.ENDC)


def computeReferenceValues(Reynolds, Mach, meshParams, FluidProperties,
        Temperature=288.15, AngleOfAttackDeg=0.0, AngleOfSlipDeg=0.0,
        YawAxis=[0.,1.,0.], PitchAxis=[0.,0.,-1.],
        TurbulenceLevel=0.001,
        TurbulenceModel='Wilcox2006-klim', Viscosity_EddyMolecularRatio=0.1,
        TurbulenceCutoff=1.0, TransitionMode=None, CoprocessOptions={},
        FieldsAdditionalExtractions=[]):
    '''
    This function is the Airfoil's equivalent of :py:func:`MOLA.Preprocess.computeReferenceValues` .
    The main difference is that in this case reference values are set through
    **Reynolds** and **Mach** number.

    Please, refer to :py:func:`MOLA.Preprocess.computeReferenceValues` doc for more details.
    '''

    try:
        if Reynolds > meshParams['References']['Reynolds']:
            MSG = "WARNING: Reynolds of flow is %g which is greater than Reynolds of mesh %g.\n"%(Reynolds,meshParams['References']['Reynolds'])
            MSG+= "This may provoke DeltaYPlus > desired value (%g).\n"%meshParams['References']['DeltaYPlus']
            MSG+= "Consider decreasing Reynolds number or creating newly adapted mesh."
            print(J.WARN+MSG+J.ENDC)
    except:
        pass

    DefaultCoprocessOptions = dict(    # Default values for WorkflowAirfoil
        UpdateFieldsFrequency   = 2000,
        UpdateArraysFrequency    =100,
        NewSurfacesFrequency    = 500,
        AveragingIterations     = 3000,
        MaxConvergedCLStd       = 1e-6,
        ItersMinEvenIfConverged = 3000,
        TimeOutInSeconds        = 54000.0, # 15 h * 3600 s/h = 53100 s
        SecondsMargin4QuitBeforeTimeOut = 900.,
    )
    DefaultCoprocessOptions.update(CoprocessOptions) # User-provided values


    # Fluid properties local shortcuts
    Gamma   = FluidProperties['Gamma']
    IdealGasConstant = FluidProperties['IdealGasConstant']
    cv      = FluidProperties['cv']
    cp      = FluidProperties['cp']

    # REFERENCE VALUES COMPUTATION
    T   = Temperature
    mus = FluidProperties['SutherlandViscosity']
    Ts  = FluidProperties['SutherlandTemperature']
    S   = FluidProperties['SutherlandConstant']

    try:
        Length = meshParams['References']['Chord']
    except KeyError:
        Length = 1.0
        print(J.WARN+'Chord not found at meshParams. Using Chord=1.0'+J.ENDC)

    try:
        Depth = meshParams['References']['Depth']
    except KeyError:
        Depth = 1.0
        print(J.WARN+'Depth not found at meshParams. Using Depth=1.0'+J.ENDC)

    try:
        TorqueOrigin = meshParams['References']['TorqueOrigin']
    except KeyError:
        TorqueOrigin = [0.25,0.0,0.0]
        print(J.WARN+'TorqueOrigin not found at meshParams. Using TorqueOrigin='+str(TorqueOrigin)+J.ENDC)


    Surface = Length * Depth

    ViscosityMolecular = mus * (T/Ts)**1.5 * ((Ts + S)/(T + S))
    Velocity = Mach * np.sqrt( Gamma * IdealGasConstant * Temperature )
    Density  = Reynolds * ViscosityMolecular / (Velocity * Length)

    ReferenceValues = PRE.computeReferenceValues(FluidProperties,
        Density=Density,
        Velocity=Velocity,
        Temperature=Temperature,
        AngleOfAttackDeg=AngleOfAttackDeg,
        AngleOfSlipDeg=AngleOfSlipDeg,
        YawAxis=YawAxis,
        PitchAxis=PitchAxis,
        TurbulenceLevel=TurbulenceLevel,
        Surface=Surface,
        Length=Length,
        TorqueOrigin=TorqueOrigin,
        TurbulenceModel=TurbulenceModel,
        Viscosity_EddyMolecularRatio=Viscosity_EddyMolecularRatio,
        TurbulenceCutoff=TurbulenceCutoff,
        TransitionMode=TransitionMode,
        CoprocessOptions=DefaultCoprocessOptions,
        FieldsAdditionalExtractions=FieldsAdditionalExtractions)

    return ReferenceValues


def setBCDataSetWithNonLocalTransition(t, AirfoilWallFamilyName, ReferenceValues):
    '''
    This function is used to set ``BCDataSet`` CGNS nodes relative to transition
    (computed or imposed) of an airfoil tree.

    Parameters
    ----------

        t : PyTree
            as produced by :py:func:`buildMesh`

            .. note:: tree **t** is modified.

        AirfoilWallFamilyName : str
            the family name of the airfoil's walls

        ReferenceValues : dict
            as produced :py:func:`computeReferenceValues`

    '''

    def findOrigin(SideCurve, OriginXoverC):
        OriginField, = J.invokeFields(SideCurve, ['origin'])
        x = J.getx(SideCurve)
        XoverC = (x - xmin)/Chord
        IndexOrigin = np.where(XoverC>OriginXoverC)[0][0]
        OriginField[IndexOrigin] = 1

    def findHow(SideCurve, LaminarImposedUpTo, LaminarIfFailureUpTo,
                TurbulentImposedFrom):
        C._initVars(SideCurve,
                    'XoverC=({CoordinateX} - %g)/%g'%(xmin, Chord))
        x = '{XoverC}'
        LaminarIfFailureRegion = '2*(%s>=%g)*(%s<%g)'%(x,LaminarImposedUpTo,x,LaminarIfFailureUpTo)
        TurbulentIfFailureRegion = '+3*(%s>=%g)*(%s<%g)'%(x,LaminarIfFailureUpTo,x,TurbulentImposedFrom)
        TurbulentImposedRegion = '+1*(%s>=%g)'%(x,TurbulentImposedFrom)

        HowEquation = 'how='+LaminarIfFailureRegion+TurbulentIfFailureRegion+TurbulentImposedRegion
        C._initVars(SideCurve, HowEquation)

    def setOriginAndHow():
        J.migrateFields([TopSideCurve, BottomSideCurve], AirfoilCenters,
                        keepMigrationDataForReuse=False)
        for AirfoilCentersZone in AirfoilCenters:
            OriginField, = J.getVars(AirfoilCentersZone, ['origin'])
            if OriginField.any():
                OriginIndices, = np.where(OriginField==1)
                x = J.getx(AirfoilCentersZone)
                for OriginIndex in OriginIndices:
                    try:
                        isFirstindexIncreasing = (x[OriginIndex+1] -
                                                  x[OriginIndex]) > 0
                    except IndexError:
                        isFirstindexIncreasing = (x[OriginIndex] -
                                                  x[OriginIndex-1]) > 0

                    if not isFirstindexIncreasing:
                        OriginField[OriginIndex] = -1



    def setOriginAtFaceCenters(XoverC, side):

        CutDirection = np.array([1.0,0.0,0.0])
        CutPoint = np.array([xmin+XoverC*Chord,0,0])
        PlaneCoefs = CutDirection[0],CutDirection[1],CutDirection[2],-CutDirection.dot(CutPoint)
        C._initVars(AirfoilShape,'SliceVar=%0.12g*{CoordinateX}+%0.12g*{CoordinateY}+%0.12g*{CoordinateZ}+%0.12g'%PlaneCoefs)
        Slice, = P.isoSurfMC(AirfoilShape,'SliceVar',value=0.0)
        xyzSlice = J.getxyz(Slice)
        IntersectionTop = xyzSlice[0][0],xyzSlice[1][0],xyzSlice[2][0]
        IntersectionBottom = xyzSlice[0][1],xyzSlice[1][1],xyzSlice[2][1]
        if IntersectionTop[1] < IntersectionBottom[1]:
            IntersectionTop, IntersectionBottom = IntersectionBottom, IntersectionTop
        if side == 'Top':
            ValuePoint = IntersectionTop
        elif side == 'Bottom':
            ValuePoint = IntersectionBottom
        else:
            raise AttributeError('side %s not recognized'%side)

        Zone2setValue,_ = J.getNearestZone(AirfoilCenters, ValuePoint)
        ind = D.getNearestPointIndex(Zone2setValue, ValuePoint)[0]
        field = J.getVars(Zone2setValue,['origin'])[0]
        I._rmNodesByName(Zone2setValue,'SliceVar')

        field[ind] = +1 if side == 'Top' else -1 # CAVEAT : Mesh needs to be oriented clockwise in i-direction

    def setHowAtFaceCenters(AirfoilCenters):
        x = '{CoordinateX}'
        LaminarIfFailureRegion = '2*(%s>=%g)*(%s<%g)'%(x,LaminarImposedUpTo,x,LaminarIfFailureUpTo)
        TurbulentIfFailureRegion = '+3*(%s>=%g)*(%s<%g)'%(x,LaminarIfFailureUpTo,x,TurbulentImposedFrom)
        TurbulentImposedRegion = '+1*(%s>=%g)'%(x,TurbulentImposedFrom)

        HowEquation = 'how='+LaminarIfFailureRegion+TurbulentIfFailureRegion+TurbulentImposedRegion
        C._initVars(AirfoilCenters,HowEquation)

    def imposeWallFields(ImposedWallFields):
        '''
        Example of input (through ReferenceValues['ImposedWallFields']):

        ImposedWallFields = [
        dict(fieldname    = 'intermittency_imposed',
             side         = 'top',
             Xref         = TopXtr,
             LengthScale  = TopLengthScale,
             a_boost      = a_boost,
             sa           = sa,
             sb           = sb,
             ),

        dict(fieldname    = 'clim_imposed',
             side         = 'top',
             Xref         = TopXtr+TopLengthScale*sc,
             LengthScale  = 1e-6,
             a_boost      = 1.0,
             sa           = 1e-3,
             sb           = 0.0,
             ),

        dict(fieldname    = 'intermittency_imposed',
             side         = 'bottom',
             Xref         = BottomXtr,
             LengthScale  = BottomLengthScale,
             a_boost      = a_boost,
             sa           = sa,
             sb           = sb,
             ),

        dict(fieldname    = 'clim_imposed',
             side         = 'bottom',
             Xref         = BottomXtr+BottomLengthScale*sc,
             LengthScale  = 1e-6,
             a_boost      = 1.0,
             sa           = 1e-3,
             sb           = 0.0,
             ),
        ]

        '''


        for c in (TopSideCurve, BottomSideCurve):
            x = J.getx(c)
            if x[-1] < x[0]:
                T._reorder(c,(-1,2,3))

        isTransitionImposed = False

        for iwf in ImposedWallFields:
            whichSide = iwf['side'].lower()

            if whichSide == 'top':
                curve2modify = TopSideCurve

            elif whichSide == 'bottom':
                curve2modify = BottomSideCurve

            else:
                ERRMSG = 'side {} not recognized'.format(iwf['side'])
                raise AttributeError(ERRMSG)

            if not isTransitionImposed:
                isTransitionImposed = iwf['fieldname']=='intermittency_imposed'

            W.setImposedFieldLSTT(curve2modify,
                fieldname     = iwf['fieldname'],
                Xref          = iwf['Xref'],
                LengthScale   = iwf['LengthScale'],
                a_boost       = iwf['a_boost'],
                sa            = iwf['sa'],
                sb            = iwf['sb'])


        # This is less expensive than extractMesh (see #7643)
        J.migrateFields([TopSideCurve, BottomSideCurve],
                        AirfoilCenters, keepMigrationDataForReuse=False)
        C._rmVars(AirfoilCenters, 's')

        return isTransitionImposed


    FamilyBCname = 'FamilySpecified:%s'%AirfoilWallFamilyName

    AirfoilVertex   = C.extractBCOfName(t,FamilyBCname, reorder=False)
    I._rmNodesFromType(AirfoilVertex,'FlowSolution_t')
    AirfoilShape,  = T.merge(AirfoilVertex)
    AirfoilShape   = C.node2Center(AirfoilShape)
    AirfoilCenters = [C.node2Center(faz) for faz in AirfoilVertex]

    xmin = C.getMinValue(AirfoilShape,'CoordinateX')
    xmax = C.getMaxValue(AirfoilShape,'CoordinateX')
    Chord = xmax-xmin

    TopSideCurve, BottomSideCurve = W.splitAirfoil(AirfoilShape,
        FieldCriterion='CoordinateY', SideChoiceCriteriaPriorities=['field'])
    I._rmNodesByType([TopSideCurve, BottomSideCurve], 'FlowSolution_t')

    try:
        ImposedWallFields = ReferenceValues['ImposedWallFields']
        isTransitionImposed = imposeWallFields(ImposedWallFields)
    except KeyError:
        isTransitionImposed = False

    if not isTransitionImposed:

        # TODO: Make default values fonction of AoA
        try: TransZones = ReferenceValues['TransitionZones']
        except KeyError: TransZones = {}

        try: TopOrigin = TransZones['TopOrigin']
        except KeyError: TopOrigin = 0.003

        try: BottomOrigin = TransZones['BottomOrigin']
        except KeyError: BottomOrigin = 0.05

        try: TopLaminarImposedUpTo = TransZones['TopLaminarImposedUpTo']
        except KeyError: TopLaminarImposedUpTo = 0.01

        try: TopLaminarIfFailureUpTo = TransZones['TopLaminarIfFailureUpTo']
        except KeyError: TopLaminarIfFailureUpTo = 0.07

        try: TopTurbulentImposedFrom = TransZones['TopTurbulentImposedFrom']
        except KeyError: TopTurbulentImposedFrom = 0.95

        try: BottomLaminarImposedUpTo = TransZones['BottomLaminarImposedUpTo']
        except KeyError: BottomLaminarImposedUpTo = 0.10

        try: BottomLaminarIfFailureUpTo = TransZones['BottomLaminarIfFailureUpTo']
        except KeyError: BottomLaminarIfFailureUpTo = 0.50

        try: BottomTurbulentImposedFrom = TransZones['BottomTurbulentImposedFrom']
        except KeyError: BottomTurbulentImposedFrom = 0.95

        findOrigin(TopSideCurve, TopOrigin)
        findOrigin(BottomSideCurve, BottomOrigin)
        findHow(TopSideCurve, TopLaminarImposedUpTo,
                TopLaminarIfFailureUpTo, TopTurbulentImposedFrom)
        findHow(BottomSideCurve, BottomLaminarImposedUpTo,
                BottomLaminarIfFailureUpTo, BottomTurbulentImposedFrom)
        setOriginAndHow()

    C._rmVars(AirfoilCenters,'XoverC')
    # Put values in BCDataSet
    AirfoilBCs = C.getFamilyBCs(t,AirfoilWallFamilyName)
    for z in AirfoilCenters: z[0] = z[0].split('/')[-1]
    for BCnode in AirfoilBCs:
        BCDataSet = I.newBCDataSet(name='BCDataSet', value='UserDefined', gridLocation='FaceCenter', parent=BCnode)
        NeumannData = I.newBCData('NeumannData', parent=BCDataSet)
        AirfoilCentersZone, = [z for z in AirfoilCenters if z[0]==BCnode[0]]
        VarNames, = C.getVarNames(AirfoilCentersZone, excludeXYZ=True)
        for varname in VarNames:
            field, = J.getVars(AirfoilCentersZone, [varname])
            I.newDataArray(varname, value=field, parent=NeumannData)


def computeTransitionCriterion(walls, criterion='in_ahd_gl'):
    '''
    This function calls the transition criterion computation, as a
    post-process routine. This routine requires **walls** to include
    boundary-layer post-processed quantities in ``FlowSolution``.

    Parameters
    ----------

        walls : PyTree, base, zone, list of zones
            1-cell-depth surface defining the wall of an airfoil

        criterion : str
            choice of the computed transition criterion.

            Currently available options:

            * ``'in_ahd_gl'``
                employs :py:func:`computeAHDGleyzes`

    Returns
    -------

        TransitionLines : :py:class:`list` of zone
            top and bottom structured curves
            including the post-processed flowfields issued from criterion
            computation
    '''
    TransitionLines = getTransitionLines(walls)


    if criterion == 'in_ahd_gl':
        computeAHDGleyzes(TransitionLines)

    return TransitionLines

def computeAHDGleyzes(TransitionLines, Hswitch=2.8):
    '''
    Adds the following fields to provided **TransitionLines** :
    ``ReynoldsThetaCritical``, ``ReynoldsThetaTransition``,
    ``TotalAmplificationFactor`` and ``MeanPohlhausen``

    Requires the following fields to be present at input TransitionLines:

    * ``ReynoldsTheta`` or (``runit`` and (``Theta`` or ``theta11`` or ``theta11i``))
    * ``lambda2``
    * ``H`` or ``hi``
    * ``turb_level`` (in %)

    Parameters
    ----------

        TransitionLines : :py:class:`list` of zone
            structured curves as obtained from airfoil walls by function
            :py:func:`getTransitionLines`

            .. note:: zones **TransitionLines** are modified

        Hswitch : float
            threshold value used to switch from AHD criterion to Gleyzes criterion.

    '''
    import scipy.integrate as si
    MAX, MIN = np.maximum, np.minimum
    def explim(v): return np.exp(MIN(v,10.0))
    def loglim(v): return np.log(MAX(v,1e-30))



    for line in TransitionLines:
        ReCr, ReTr, Ntot, MeanL2 = J.invokeFields(line,
                                                ['ReynoldsThetaCritical',
                                                 'ReynoldsThetaTransition',
                                                 'TotalAmplificationFactor',
                                                 'MeanPohlhausen'])
        ReT = getReynoldsTheta(line)
        L2  = getQuantityFromTransitionLine(line, ['lambda2'])
        H   = getQuantityFromTransitionLine(line, ['H','hi'])
        Tu  = getQuantityFromTransitionLine(line, ['turb_level']) * 1e-2
        s   = getQuantityFromTransitionLine(line, ['s'])

        Tu[:] = MAX(Tu,0.0001*1e-2)

        ReCr[:] = explim((52./MAX(H,0.5))-14.8)
        scr = J.interpolate__(0.0, ReT-ReCr, s)
        ReCrValue = J.interpolate__(0.0, ReT-ReCr, ReCr)

        # BEWARE ! in scipy v >= 1.6.0 cumptrapz becomes cumulative_trapezoid
        # MeanL2 = scipy.integrate.cumulative_trapezoid()
        Critical = s >= scr
        FactorMeanL2 = (1./(s[Critical]-scr))

        MeanL2[Critical] = FactorMeanL2*si.cumtrapz(L2[Critical],
                                                     s[Critical],
                                                     initial=0)


        ReTr[Critical] = -206.*explim(25.7*MeanL2[Critical]) * \
                                 (loglim(16.8*Tu[Critical]) \
                                     -2.77*MeanL2[Critical])+ReCrValue


        Ntot[Critical] = MAX(0.,-8.43 -2.4*loglim(0.01*(1./0.168)*explim(2.77*MeanL2[Critical]-
            ((ReT[Critical]-ReCrValue)/(206.*explim(25.7*MeanL2[Critical]))))))

        ssw = J.interpolate__(Hswitch, H, s)
        Switch = s >= ssw

        Nsw = float(J.interpolate__(ssw, s, Ntot))

        # Gleyze's function
        Bgl = H*0.
        Upper = H > 3.36
        Lower = H < 2.8
        Middle = np.logical_not(Upper) * np.logical_not(Lower)
        Bgl[Upper] = -162.11093/(H[Upper]**1.1)
        Bgl[Middle] = -73*explim(-1.56486*(H[Middle]-3.02))
        Bgl[Lower] = -103*explim(-4.12633*(H[Lower]-2.8))

        Ntot[Switch] = Nsw+si.cumtrapz(-2.4/Bgl[Switch],ReT[Switch], initial=Nsw)


def getReynoldsTheta(TransitionLine):
    '''
    Add and return ``ReynoldsTheta`` quantity from a zone containing fields
    at ``FlowSolution`` (nodes):
    ``ReynoldsTheta`` or (``runit`` and (``Theta`` or ``theta11`` or ``theta11i``))

    Parameters
    ----------

        TransitionLine : zone
            zone containing the required fields

    Returns
    -------

        ReynoldsTheta : numpy.ndarray
            numpy array of ReynoldsTheta :math:`Re_\\theta`
    '''
    ReynoldsTheta = I.getNodeFromName(TransitionLine, 'ReynoldsTheta')
    if ReynoldsTheta: return ReynoldsTheta[1]
    C._initVars(TransitionLine,'ReynoldsTheta',0)

    ReynoldsUnity = I.getNodeFromName(TransitionLine, 'runit')
    if not ReynoldsUnity:
        raise ValueError('Missing ReynoldsTheta or runit fields')

    Theta = getQuantityFromTransitionLine(TransitionLine,
                                           ['Theta','theta11','theta11i'])
    ReynoldsTheta, = J.getVars(TransitionLine,['ReynoldsTheta'])
    ReynoldsTheta[:] = Theta * ReynoldsUnity[1]

    return ReynoldsTheta


def getQuantityFromTransitionLine(TransitionLine, PossibleNames):
    '''
    Get an arbitrary quantity from nodes ``FlowSolution`` container of a
    zone, named after a set of possible names. If no one is found, then
    raises an error. If field ``'s'`` (dimensional curvilinear-abscissa in this
    context) is requested, then it is computed.

    Parameters
    ----------

        TransitionLine : zone
            structured curve with requested fields located
            at vertex of container ``FlowSolution``

        PossibleNames : :py:class:`list` of :py:class:`str`
            list of fields requested. First matching item is kept.

    Returns
    -------

        quantity : numpy.ndarray
            the detected numpy array
    '''
    for QuantityName in PossibleNames:
        quantity = I.getNodeFromName(TransitionLine, QuantityName)
        if quantity: return quantity[1]

    if QuantityName == 's':
        s = W.gets(TransitionLine)
        s *= D.getLength(TransitionLine)
        return s

    raise ValueError('missing field. Must be in %s'%str(PossibleNames))


def getTransitionLines(walls):
    '''
    Given a 1-cell-depth surface defining the wall of an airfoil, produce two
    curves corresponding to the Top and Bottom sides of the airfoil and
    adds the dimensional curvilinear-abscissa field ``{s}``

    Parameters
    ----------

        walls : PyTree, base, zone, list of zones
            1-cell-depth surface defining the wall of an airfoil.

    Returns
    -------

        Top : zone
            structured curve corresponding to the top side of the airfoil

        Bottom : zone
            structured curve corresponding to the bottom side of the airfoil
    '''
    Top, Bottom = mergeWallsAndSplitAirfoilSides(walls)
    for line in Top, Bottom:
        s = W.gets(line)
        s *= D.getLength(line)
    return Top, Bottom


def mergeWallsAndSplitAirfoilSides(t):
    '''
    Given a 1-cell-depth surface defining the wall of an airfoil, produce two
    curves corresponding to the Top and Bottom sides of the airfoil.

    Parameters
    ----------

        walls : PyTree, base, zone, list of zones
            1-cell-depth surface defining the wall of an airfoil.

    Returns
    -------

        Top : zone
            structured curve corresponding to the top side of the airfoil

        Bottom : zone
            structured curve corresponding to the bottom side of the airfoil
    '''

    tRef = I.copyRef(t)
    I._rmNodesByType(tRef, 'FlowSolution_t')
    tRef = T.merge(tRef)
    foil, = C.node2Center(tRef)

    TopSide, BottomSide = W.splitAirfoil(foil,
                                         FirstEdgeSearchPortion = 0.9,
                                         SecondEdgeSearchPortion = -0.9,
                                         RelativeRadiusTolerance = 1e-2,
                                         MergePointsTolerance = 1e-10)


    tOut =C.newPyTree(['Airfoil',[TopSide, BottomSide]])
    tCtr = C.node2Center(t)
    J.migrateFields(tCtr, tOut)

    return I.getZones(tOut)


def getMeshingParameters():
    '''
    This function reads the data contained in ``MeshingParameters.py`` and
    converts it into a dictionary ``meshParams``.

    If the ``MeshingParameters.py`` file is non-existent, then the following
    default dictionary is returned :

    >>> meshParams = {'References':{'DeltaYPlus':0.5, 'Reynolds':1e6},}

    Returns
    -------

        meshParams : dict
            to be employed in :py:func:`buildMesh` function
    '''

    meshParams = {'References':{'DeltaYPlus':0.5, 'Reynolds':1e6},}

    try:
        import MeshingParameters
        meshParams = MeshingParameters.meshParams
    except ImportError:
        print('Inexistent MeshingParameters.py')
        print('Will use default meshing values')
    except AttributeError:
        print('MeshingParameters.py does not have meshParams dictionary')
        print('Will use default meshing values')

    return meshParams


def getAirfoilCurveFromSurfacesFile(filepath):
    '''
    Wrap of :py:func:`getAirfoilCurveFromSurfaces` function, but using a filepath
    (tyically, ``surfaces.cgns``) instead using the tree pointer.
    '''

    t = C.convertFile2PyTree(filepath)

    return getAirfoilCurveFromSurfaces(t)

def getAirfoilCurveFromSurfaces(t):
    '''
    Extract the airfoil contour from a ``surfaces.cgns`` tree

    Parameters
    ----------

        t : PyTree
            tree containing flow fields

    Returns
    -------

        wall : zone
            surface of the airfoil's contour, forced to be at :math:`z=0`
    '''

    # missing at some zones TODO: make elsA ticket
    NotHomogeneousFields = ('origin','clim_imposed','intermittency_imposed')
    for fieldname in NotHomogeneousFields: I._rmNodesByName(t, fieldname)

    # elsA bug : somme numpy have wrong sizes. To circumvent this problem:
    for zone in I.getZones(t):
        for field_node in I.getNodeFromName(zone, 'FlowSolution#Centers')[2]:
            field = field_node[1]
            if isinstance(field, str): continue
            if len(field.shape) == 1:
                I.setValue(field_node, field.reshape((field.shape[0],1),
                           order='F'))

    for FlowSolution in I.getNodesFromType(t,'FlowSolution_t'):
        I._sortByName(FlowSolution, recursive=True)

    t = T.merge(I.getZones(t))

    wall, = I.getZones(t)
    Ni = I.getZoneDim(wall)[1]
    wall = T.subzone(wall,(1,1,1),(Ni,1,1))
    z = J.getz(wall)
    z *= 0

    return wall


def getBoundaryLayerEdgesFromAirfoilCurve(wall,
                                Thicknesses2Plot=['delta','delta1','theta11']):
    '''
    Obtain a set of zones representing the boundary-layer characteristic
    thicknesses provided by user.

    Parameters
    ----------

        wall : zone
            wall surface containing normals fields ``nx`` ``ny`` ``nz``

        Thickness2Plot : :py:class:`list` of :py:class:`str`
            names of the characteristic thicknesses. Each one must be contained
            in **wall**

    Returns
    -------

        surfaces : list
            the surfaces (structured zones) whose coordinates correspond to the
            requested characteristic thicknesses
    '''
    z = I.copyRef(wall)
    C.center2Node__(z,'centers:nx',cellNType=0)
    C.center2Node__(z,'centers:ny',cellNType=0)
    C.center2Node__(z,'centers:nz',cellNType=0)
    for Thickness2Plot in Thicknesses2Plot:
        C.center2Node__(z,'centers:'+Thickness2Plot,cellNType=0)
    I._rmNodesByName(z,'FlowSolution#Centers')
    C._normalize(z,['nx','ny','nz'])

    NewZones2add = []
    for Thickness2Plot in Thicknesses2Plot:
        for coord in ('x','y','z'): C._initVars(z,'d%s={%s}*{n%s}'%(coord,Thickness2Plot,coord))
        z2 = T.deform(z,['dx','dy','dz'])
        I._rmNodesByType(z2,'FlowSolution_t')
        z2[0] += '_%s'%Thickness2Plot
        NewZones2add += [z2]

    return NewZones2add


def addPressureAndFrictionCoefficientsToAirfoilCurve(wall, setupfilepath=None,
        PressureDynamic=None, PressureStatic=None):
    '''
    Add ``Cp`` and ``Cf`` fields to the airfoil's surface.

    .. note:: information contained in **setupfilepath** takes priority over
        explicitly provided values (**PressureDynamic** and **PressureStatic**)

    Parameters
    ----------

        wall : zone
            as provided by :py:func:`getAirfoilCurveFromSurfaces` . It must
            contain fields ``Pressure`` ``SkinFrictionX`` ``SkinFrictionY``
            ``SkinFrictionZ`` ``nx`` ``ny`` ``nz`` at ``FlowSolution#Centers``.

            .. note:: zone **wall** is modified

        setupfilepath : str
            ``setup.py`` file containing case information. (Most prioritary)

        PressureDynamic : float
            Dynamic pressure for normaization

        PressureStatic : float
            Static pressure for normaization

    '''

    if setupfilepath:
        setup = J.load_source('setup', setupfilepath)
        PressureDynamic = setup.ReferenceValues['PressureDynamic']
        PressureStatic = setup.ReferenceValues['Pressure']

    else:
        if not all([PressureDynamic, PressureStatic]):
            raise ValueError('Must provide either setup.py or needed reference values')

    Cp, Cf = J.invokeFields(wall,['Cp','Cf'], locationTag='centers:')
    P, fx, fy, fz, nx, ny, nz = J.getVars(wall,['Pressure','SkinFrictionX',
        'SkinFrictionY', 'SkinFrictionZ','nx','ny','nz'],
        Container='FlowSolution#Centers')


    Cf[:] = (ny*fx - nx*fy)/(np.sqrt( ny*ny + nx*nx + nz*nz )*PressureDynamic)
    Cf[ny<0] *= -1
    Cp[:] = ( P - PressureStatic ) / PressureDynamic


def getRangesOfStructuredPolar(config):
    '''
    Compute Polar ranges of AngleOfAttack, Mach and Reynolds following a
    Struct_AoA_Mach convention using as input the **config**, which can be
    obtained using :py:func:`MOLA.JobManager.getJobsConfiguration` function.

    Parameters
    ----------

        config : module
            ``JobsConfiguration.py`` data as a mobule object as
            returned by function :py:func:`MOLA.JobManager.getJobsConfiguration`

    Returns
    -------

        AoAs : numpy.ndarray
            1D vector of considered angles-of-attack

        Mach : numpy.ndarray
            1D vector of considered Mach numbers

        Reynolds : numpy.ndarray
            1D vector of considered Reynolds numbers
    '''
    AoAs = np.array(sorted(list(set([case['elsAParams']['AngleOfAttackDeg'] for case in config.JobsQueues]))))

    # BEWARE: Reynolds may not be linear on Mach, and can possibly be doubled
    Mach, Reynolds = [] , []
    for case in config.JobsQueues:
        NewMach = case['elsAParams']['MachPolar']
        if NewMach not in Mach:
            Mach.append( NewMach )
            Reynolds.append( case['elsAParams']['Reynolds'] )

    Mach, Reynolds = J.sortListsUsingSortOrderOfFirstList(Mach, Reynolds)

    return AoAs, Mach, Reynolds


def getReynoldsFromCaseLabel(config, CASE_LABEL):
    '''
    Determine the employed Reynolds number :math:`Re_c` of a case identified by
    a given **CASE_LABEL**

    Parameters
    ----------

        config : module
            ``JobsConfiguration.py`` data as a mobule object as
            returned by function :py:func:`MOLA.JobManager.getJobsConfiguration`

        CASE_LABEL : str
            unique identifying label of the requested case.
            Can be determined by :py:func:`getCaseLabelFromAngleOfAttackAndMach`

    Returns
    -------

        Reynolds : float
            the employed Reynolds number :math:`Re_c`
    '''
    for case in config.JobsQueues:
        if case['CASE_LABEL'] == CASE_LABEL:
            return case['elsAParams']['Reynolds']

    raise ValueError('no case %s found'%CASE_LABEL)


def getCaseLabelFromAngleOfAttackAndMach(config, AoA, Mach):
    '''
    Get the **CASE_LABEL** corresponding to the CFD run of from given **AoA**
    and **Mach**.

    Parameters
    ----------

        config : module
            ``JobsConfiguration.py`` data as a mobule object as
            returned by function :py:func:`MOLA.JobManager.getJobsConfiguration`

        AoA : float
            requested angle of attack

        Mach : float
            requested (virtual) Mach number

    Returns
    -------

        CASE_LABEL : str
            unique identifying label of the requested case
    '''
    for case in config.JobsQueues:
        if np.isclose(case['elsAParams']['MachPolar'], Mach) and \
           np.isclose(case['elsAParams']['AngleOfAttackDeg'], AoA):

            return case['CASE_LABEL']

    raise ValueError('no case found for AoA=%g, M=%g at %s'%(AoA,Mach,config.DIRECTORY_WORK))


def printConfigurationStatus(DIRECTORY_WORK, useLocalConfig=True):
    '''
    Print the current status of a Polars computation.


    Parameters
    ----------

        DIRECTORY_WORK : str
            directory where ``JobsConfiguration.py`` file is located

        useLocalConfig : bool
            if :py:obj:`True`, use the local ``JobsConfiguration.py``
            file instead of retreiving it from **DIRECTORY_WORK**

    Returns
    -------

        None : None
            prints to standard output the status of the computation matrix.
            Three possible tags exist:

            * ``PD``
                which means *pending*. The case run is waiting for previous cases to end.

            * ``GO``
                the case is *running*.

            * ``OK``
                the case terminated (but not necessarily converged)

            * ``KO``
                the case crashed

            * ``TO``
                the case terminated in time-out. This case is theoretically
                impossible. If you see status ``TO``, please contact the
                author of this module.

    '''
    config = JM.getJobsConfiguration(DIRECTORY_WORK, useLocalConfig, filename='JobsConfiguration.py')
    AoA, Mach, Reynolds = getRangesOfStructuredPolar(config)
    nM = len(Mach)
    nA = len(AoA)
    NcolMax = 79
    FirstCol = 10
    Ndigs = int((NcolMax-FirstCol)/nM)
    ColFmt = r'{:^'+str(Ndigs)+'g}'
    ColStrFmt = r'{:^'+str(Ndigs)+'s}'
    TagStrFmt = r'{:>'+str(FirstCol)+'s}'
    TagFmt = r'{:>'+str(FirstCol-2)+'g} |'

    JobNames = [getCaseLabelFromAngleOfAttackAndMach(config, AoA[0], m).split('_')[-1] for m in Mach]
    print('')
    print(TagStrFmt.format('JobName |')+''.join([ColStrFmt.format(j) for j in JobNames]))
    print(TagStrFmt.format('Reynolds |')+''.join([ColFmt.format(r) for r in Reynolds]))
    print(TagStrFmt.format('Mach |')+''.join([ColFmt.format(m) for m in Mach]))
    print(TagStrFmt.format('AoA |')+''.join(['_' for m in range(NcolMax-FirstCol)]))

    for a in AoA:
        Line = TagFmt.format(a)
        for m in Mach:
            CASE_LABEL = getCaseLabelFromAngleOfAttackAndMach(config, a, m)
            status = JM.statusOfCase(config, CASE_LABEL)
            if status == 'COMPLETED':
                msg = J.GREEN+ColStrFmt.format('OK')+J.ENDC
            elif status == 'FAILED':
                msg = J.FAIL+ColStrFmt.format('KO')+J.ENDC
            elif status == 'TIMEOUT':
                msg = J.WARN+ColStrFmt.format('TO')+J.ENDC
            elif status == 'RUNNING':
                msg = ColStrFmt.format('GO')
            else:
                msg = ColStrFmt.format('PD') # Pending
            Line += msg
        print(Line)


def buildPolar(JobsConfiguration, PolarName='Polar',
     BigAngleOfAttackAoA=[-180, -160, -140, -120, -100, -80, -60, -40,-30,
                           30, 40, 60, 80, 100, 120, 140, 160, 180],
     BigAngleOfAttackCl=[ 0, 0.73921, 1.13253, 0.99593, 0.39332, -0.39332,
                         -0.99593, -1.13253, -0.99593, 0.99593, 1.13253,
                         0.99593, 0.39332, -0.39332, -0.99593, -1.13253,
                         -0.73921, 0],
     BigAngleOfAttackCd=[4e-05, 0.89256, 1.54196, 1.94824,  2.1114,  2.03144,
                         1.70836,  1.14216,  0.76789, 0.76789, 1.14216, 1.70836,
                         2.03144,  2.1114,  1.94824,  1.54196,  0.89256, 4e-05],
     BigAngleOfAttackCm=[0, 0.24998, 0.46468,  0.5463, 0.53691,  0.51722,
                        0.49436,  0.40043,  0.31161,-0.31161,-0.40043,-0.49436,
                        -0.51722, -0.53691,  -0.5463, -0.46468, -0.24998,0],
     FILE_ARRAYS='arrays.cgns'):
    '''
    Constructs a **PyZonePolar** (that can be saved to a file ``Polars.cgns``)
    containing the information of the polars predictions.

    Parameters
    ----------

        DIRECTORY_WORK : str
            directory where ``JobsConfiguration.py`` file is located

        PolarName : str
            name to provide to the new **PyZonePolar**

        useLocalConfig : bool
            if :py:obj:`True`, use the local ``JobsConfiguration.py``
            file instead of retreiving it from **DIRECTORY_WORK**

        BigAngleOfAttackAoA : :py:class:`lisf` of :py:class:`float`
            monotonically increasing range of big angles of attack
            :math:`\in [-180, 180]` degrees.

        BigAngleOfAttackCl : :py:class:`lisf` of :py:class:`float`
            corresponding value of lift coefficient for big angles of attack

        BigAngleOfAttackCd : :py:class:`lisf` of :py:class:`float`
            corresponding value of drag coefficient for big angles of attack

        BigAngleOfAttackCm : :py:class:`lisf` of :py:class:`float`
            corresponding value of pitch coefficient for big angles of attack

    Returns
    -------

        PolarsDict : :py:class:`dict`
            contains the polars results

        PyZonePolar : zone
            special CGNS zone containing polars results
    '''

    config = JobsConfiguration
    DIRECTORY_WORK = config.DIRECTORY_WORK
    AoA, Mach, Reynolds = getRangesOfStructuredPolar(config)
    nM = len(Mach)
    nA = len(AoA)

    lenBigAoA = len(BigAngleOfAttackAoA)
    for d, v in zip([BigAngleOfAttackCl,BigAngleOfAttackCd,BigAngleOfAttackCm],
                    ['Cl','Cd','Cm']):
        if lenBigAoA != len(d):
            ERRMSG = ('the length of {baoa}{v} must be equal to the length of'
                       ' {baoa}AoA').format(baoa='BigAngleOfAttack', v=v)
            raise ValueError(ERRMSG)

    PolarsDict = {}
    for i, a in enumerate(AoA):
        for j, m in enumerate(Mach):
            print(J.MAGE+'\nPROCESSING RUN %d of %d'%(1+j+nM*i, nA*nM)+J.ENDC)
            CASE_LABEL = getCaseLabelFromAngleOfAttackAndMach(config, a, m)
            status = JM.statusOfCase(config, CASE_LABEL)

            if status == 'FAILED':
                for v in PolarsDict:
                    if len(PolarsDict[v].shape) == 2:
                        if v.startswith('std-'):
                            PolarsDict[v][i,j] = 1.0
                        else:
                            PolarsDict[v][i,j] = 0.0
                    elif len(PolarsDict[v].shape) == 3:
                        PolarsDict[v][i,j,:] = 0.0

            try:
                arrays = JM.getCaseArrays(config, CASE_LABEL, FILE_ARRAYS=FILE_ARRAYS)
            except:
                for v in PolarsDict:
                    if len(PolarsDict[v].shape) == 2:
                        PolarsDict[v][i,j] = 0.0
                    elif len(PolarsDict[v].shape) == 3:
                        PolarsDict[v][i,j,:] = 0.0
                    arrays = None

            if arrays:
                for v in arrays:
                    try:
                        Matrix = PolarsDict[v]
                    except KeyError:
                        Matrix = np.zeros((nA,nM), dtype=np.float, order='F')
                        Matrix[:i,:j] = 0.0
                        Matrix[i,:j]  = 0.0
                        Matrix[:i,j]  = 0.0
                        PolarsDict[v] = Matrix
                    Matrix[i,j] = arrays[v]


            try:
                distr = getCaseDistributions(config, CASE_LABEL)
            except:
                for v in PolarsDict:
                    if len(PolarsDict[v].shape) == 3:
                        PolarsDict[v][i,j,:] = 0.0
                distr = None

            if distr:
                for v in distr:
                    try:
                        Matrix = PolarsDict[v]
                    except KeyError:
                        nS = distr[v].shape[-1]
                        Matrix = np.zeros((nA,nM,nS), dtype=np.float, order='F')
                        Matrix[:i,:j,:] = 0.0
                        Matrix[i,:j,:]  = 0.0
                        Matrix[:i,j,:]  = 0.0
                        PolarsDict[v] = Matrix
                    Matrix[i,j,:] = distr[v]

    # change between 3D and 2D convention
    PolarsDict['Cl'] = PolarsDict['avg-CL']
    PolarsDict['Cd'] = PolarsDict['avg-CD']

    default_Big={}
    default_Big['AoA']= np.array(BigAngleOfAttackAoA,dtype=np.float64, order='F')
    default_Big['Cl'] = np.array(BigAngleOfAttackCl,dtype=np.float64, order='F')
    default_Big['Cd'] = np.array(BigAngleOfAttackCd,dtype=np.float64, order='F')
    default_Big['Cm'] = np.array(BigAngleOfAttackCm,dtype=np.float64, order='F')

    IntegralFieldsNames = [v for v in PolarsDict if len(PolarsDict[v].shape) == 2]
    DistributionFieldsNames = [v for v in PolarsDict if len(PolarsDict[v].shape) == 3]
    try:
        DistributionFieldsNames.remove('CoordinateX')
        DistributionFieldsNames.remove('CoordinateY')
        DistributionFieldsNames.remove('CoordinateZ')
    except:
        pass

    IntegralFields = [PolarsDict[v] for v in IntegralFieldsNames]

    PyZonePolar = J.createZone(PolarName, IntegralFields, IntegralFieldsNames)

    # Add .Polar#Range node
    children = [['AngleOfAttack',     AoA],
                ['Mach',             Mach],
                ['Reynolds',     Reynolds],]
    BigAoADict = {'Cl':BigAngleOfAttackCl, 'Cd':BigAngleOfAttackCd, 'Cm':BigAngleOfAttackCm}
    for v in BigAoADict:
        BigAoADict[v] = default_Big['AoA'], default_Big[v]
        children += [['BigAngleOfAttack%s'%v, BigAoADict[v][0]]]
    J._addSetOfNodes(PyZonePolar,'.Polar#Range',children)

    # Add out-of-range big Angle Of Attack values
    children = [['BigAngleOfAttack%s'%v, BigAoADict[v][1]] for v in BigAoADict]
    J._addSetOfNodes(PyZonePolar,'.Polar#OutOfRangeValues',children)

    # Add Foil-data information
    children = []
    for var in DistributionFieldsNames:
        children += [[var, PolarsDict[var]]]
    if len(children) > 0:
        J._addSetOfNodes(PyZonePolar,'.Polar#FoilValues',children)

    # Add Foil Geometry
    if ('CoordinateX' in PolarsDict) and ('CoordinateY' in PolarsDict):
        RankCoords = len(PolarsDict['CoordinateX'].shape)
        if RankCoords == 3:
            Xcoord = PolarsDict['CoordinateX'][0,0,:]
            Ycoord = PolarsDict['CoordinateY'][0,0,:]
        elif RankCoords == 2:
            Xcoord = PolarsDict['CoordinateX'][0,:]
            Ycoord = PolarsDict['CoordinateY'][0,:]
        else:
            Xcoord = PolarsDict['CoordinateX']
            Ycoord = PolarsDict['CoordinateY']
        children = [['CoordinateX', Xcoord],
                    ['CoordinateY', Ycoord]]
        J._addSetOfNodes(PyZonePolar,'.Polar#FoilGeometry',children)

    # Add .Polar#Interp node
    PolarsDict['PyZonePolarKind'] = 'Struct_AoA_Mach'
    children=[
    ['PyZonePolarKind',PolarsDict['PyZonePolarKind']],
    ]
    if PolarsDict['PyZonePolarKind'] == 'Unstr_AoA_Mach_Reynolds':
        children += [['Algorithm','RbfInterpolator']]
    elif PolarsDict['PyZonePolarKind'] == 'Struct_AoA_Mach':
        children += [['Algorithm','RectBivariateSpline']]
    J._addSetOfNodes(PyZonePolar,'.Polar#Interp',children)

    return PolarsDict, PyZonePolar

def convertSurfaces2OrientedAirfoilCurveAtVertex(SurfacesTree):
    '''
    Convert a tree (loaded from e.g. a ``surfaces.cgns`` file) containing airfoil
    walls into a merged oriented curve with flow solutions at vertex.
    The resulting wall can be then employed in funtions such as
    :py:func:`getTransitionLines`

    Parameters
    ----------

        SurfacesTree : PyTree
            a tree as got from reading ``surfaces.cgns``

    Returns
    -------

        wall : zone
            curve as a structured zone
    '''

    wall = getAirfoilCurveFromSurfaces(SurfacesTree)

    for fieldname in C.getVarNames(wall,excludeXYZ=True,loc='centers')[0]:
        C.center2Node__(wall,fieldname,cellNType=0)

    I._rmNodesByName(wall,'FlowSolution#Centers')

    W.putAirfoilClockwiseOrientedAndStartingFromTrailingEdge(wall)

    return wall

def addRelevantWallFieldsFromElsAFieldsAtVertex(wall, PressureDynamic,
                                                PressureStatic):
    '''
    Add ``Cp``, ``Cf``, ``ReynoldsTheta`` fields to an airfoil's wall

    Parameters
    ----------

        wall : zone
            airfoil curve  containing fields:
            ``Pressure``,
            ``SkinFrictionX``, ``SkinFrictionY``, ``SkinFrictionZ``,
            ``nx``, ``ny``, ``nz``,
            ``theta11``, ``runit``

        PressureDynamic : float
            dynamic pressure for normalization

        PressureStatic : float
            static pressure for normalization

    See also
    --------
    addPressureAndFrictionCoefficientsToAirfoilCurve

    '''

    Cp, Cf, ReT = J.invokeFields(wall,['Cp','Cf','ReynoldsTheta'])
    P, fx, fy, fz, nx, ny, nz, theta, runit = J.getVars(wall,['Pressure',
        'SkinFrictionX', 'SkinFrictionY', 'SkinFrictionZ', 'nx','ny','nz',
        'theta11','runit'])

    Cf[:] = (ny*fx - nx*fy)/(np.sqrt( nx*nx + ny*ny + nz*nz )*PressureDynamic)
    Cf[ny<0] *= -1
    Cp[:] = ( P - PressureStatic ) / PressureDynamic
    ReT[:] = runit * theta


def getCaseDistributions(config, CASE_LABEL):
    '''
    Repatriate and process the remote ``OUTPUT/surface.cgns`` file and return its
    contents in a form of a Python dictionary.

    Parameters
    ----------

        config : module
            ``JobsConfiguration.py`` data as a mobule object as
            obtained from :py:func:`MOLA.JobManager.getJobsConfiguration`

        CASE_LABEL : str
            unique identifying label of the requested case.
            Can be determined by :py:func:`getCaseLabelFromAngleOfAttackAndMach`

    Returns
    ------

        SurfDict : dict
            contains foilwise distributions like this,

            ::

                {'Cf':1D array,
                 'Cp':1D array,
                 'CoordinateX', 1D array,
                 'CoordinateY', 1D array,
                    ...    }


    '''
    FILE_SURFACES = 'surfaces.cgns'
    FILE_SETUP = 'setup.py'
    JobTag = '_'.join(CASE_LABEL.split('_')[1:])

    SourceSurf = os.path.join(config.DIRECTORY_WORK, JobTag, CASE_LABEL,
                                                        'OUTPUT', FILE_SURFACES)

    SourceSetup = os.path.join(config.DIRECTORY_WORK, JobTag, CASE_LABEL,
                               FILE_SETUP)

    for FILE, Source in zip([FILE_SURFACES, FILE_SETUP],[SourceSurf, SourceSetup]):
        try:
            JM.repatriate(Source, FILE,
                                    removeExistingDestinationPath=True)
        except:
            print(J.WARN+'could not retrieve %s of case %s'%(FILE,CASE_LABEL)+J.ENDC)
            return

    setup = J.load_source('setup', FILE_SETUP)
    SurfsTree = C.convertFile2PyTree( FILE_SURFACES )
    foil = convertSurfaces2OrientedAirfoilCurveAtVertex(SurfsTree)
    addRelevantWallFieldsFromElsAFieldsAtVertex(foil,
                                       setup.ReferenceValues['PressureDynamic'],
                                       setup.ReferenceValues['Pressure'],)
    SurfDict = J.getVars2Dict(foil, C.getVarNames(foil,excludeXYZ=True)[0])
    x,y,z = J.getxyz(foil)
    SurfDict.update( dict(CoordinateX=x, CoordinateY=y, CoordinateZ=z) )

    return SurfDict


def getCaseFields(config, CASE_LABEL):
    '''
    Repatriate remote ``OUTPUT/fields.cgns`` file.

    Parameters
    ----------

        config : module
            ``JobsConfiguration.py`` data as a mobule object as
            obtained from :py:func:`MOLA.JobManager.getJobsConfiguration`

        CASE_LABEL : str
            unique identifying label of the requested case.
            Can be determined by :py:func:`getCaseLabelFromAngleOfAttackAndMach`
    '''
    FILE_FIELDS = 'fields.cgns'
    JobTag = '_'.join(CASE_LABEL.split('_')[1:])

    SourceSurf = os.path.join(config.DIRECTORY_WORK, JobTag, CASE_LABEL,
                                                        'OUTPUT', FILE_FIELDS)

    for FILE, Source in zip([FILE_FIELDS],[SourceSurf]):
        try:
            JM.repatriate(Source, FILE,
                                    removeExistingDestinationPath=True)
        except:
            print(J.WARN+'could not retrieve %s of case %s'%(FILE,CASE_LABEL)+J.ENDC)
            return



def compareAgainstXFoil(AirfoilKeyword, config, CASE_LABEL, DistributedLoads,
                        XFoilOptions={}, Field2Compare='Cp'):
    '''
    Compare a CFD prediction with a XFOIL prediction through a plot.

    Parameters
    ----------

        AirfoilKeyword : str
            keyword of airfoil (``*.dat`` file or ``naca XXXX``)

        config : module
            ``JobsConfiguration.py`` data as a mobule object as
            obtained from :py:func:`MOLA.JobManager.getJobsConfiguration`

        CASE_LABEL : str
            unique identifying label of the requested case.
            Can be determined by :py:func:`getCaseLabelFromAngleOfAttackAndMach`

        DistributedLoads : dict
            as result of :py:func:`getCaseDistributions`

        XFoilOptions : dict
            literally, optional parameters to introduce to
            :py:func:`MOLA.XFoil.computePolars` function

        Field2Compare : str
            field to compare between CFD and XFOIL.
            Possible values: ``'Cp'`` or ``'Cf'``

    Returns
    -------

        None : None
            a matplotlib figure is shown

            .. warning:: if this function is employed in a server without
                graphic output, this may cause a segmentation fault. Please
                be sure to enable graphics
    '''

    if Field2Compare == 'Cp':
        ylabel = '$C_p$'
        invert_yaxis = True

    elif Field2Compare == 'Cf':
        ylabel = '$C_f$'
        invert_yaxis = False

    else:
        raise ValueError('Field2Compare must be "Cp" or "Cf"')

    from . import XFoil
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    XFoilParams = dict(Ncr=1)
    XFoilParams.update(XFoilOptions)

    for case in config.JobsQueues:
        if case['CASE_LABEL'] == CASE_LABEL:
            Reynolds = case['elsAParams']['Reynolds']
            Mach = case['elsAParams']['MachPolar']
            AoA = case['elsAParams']['AngleOfAttackDeg']
            break

    XFoilResults = XFoil.computePolars(AirfoilKeyword, [Reynolds], [Mach],
                                       [AoA], **XFoilParams)


    fig, ax = plt.subplots(1,1,dpi=150)
    ax.plot(XFoilResults['x'].flatten(), XFoilResults[Field2Compare].flatten(),
        linestyle='None', marker='o',mfc='None',color='k', label='XFoil')
    ax.plot(DistributedLoads['CoordinateX'],DistributedLoads[Field2Compare],
            color='red', label='CFD')
    if invert_yaxis: ax.invert_yaxis()
    ax.set_ylabel(ylabel)
    ax.set_xlabel('$x/c$')
    minLocX = AutoMinorLocator()
    ax.xaxis.set_minor_locator(minLocX)
    minLocY = AutoMinorLocator()
    ax.yaxis.set_minor_locator(minLocY)
    ax.xaxis.grid(True, which='major')
    ax.xaxis.grid(True, which='minor',linestyle=':')
    ax.yaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='minor',linestyle=':')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


def correctPolar(t, useBigRangeValuesIf_StdCLisHigherThan=0.0005,
                    Fields2Correct=['Cl','Cd','Cm']):
    '''
    Given a PyTree containing **PyZonePolars**, this function replaces
    integral arrays whose lift coefficient standard-deviation ``std-CL``
    ( :math:`\sigma(c_L)` ) are higher
    than a user-provided threshold, with a linear interpolation of
    *BigAngleOfAttack* data.

    Parameters
    ----------

        t : PyTree
            tree containing **PyZonePolars**

            .. note:: tree **t** is modified

        useBigRangeValuesIf_StdCLisHigherThan : float
            standard deviation
            threhold of lift coefficient ``std-CL`` ( :math:`\sigma(c_L)` )
            from which fields requested by user are to be replaced (associated
            runs are considered poorly converged, so to be replaced by interpolated
            data)

        Fields2Correct : :py:class:`list` of :py:class:`str`
            fields to correct. Any field
            contained in ``FlowSolution`` of input **t** is acceptable.

    '''
    for zone in I.getZones(t):
        stdCL, AoA = J.getVars(zone, ['std-CL','AngleOfAttack'])
        PolarRanges = I.getNodeFromName1(zone,'.Polar#Range')
        if AoA is None:
            AoA = I.getNodeFromName1(PolarRanges,'AngleOfAttack')[1]
            AoA = np.broadcast_to(AoA,stdCL.shape[::-1]).T

        PolarOutOfRanges = I.getNodeFromName1(zone,'.Polar#OutOfRangeValues')

        for fieldname in Fields2Correct:
            Field, = J.getVars(zone, [fieldname])
            BigAoA = I.getNodeFromName1(PolarRanges,
                                       'BigAngleOfAttack'+fieldname)[1]
            FieldOutOfRange = I.getNodeFromName1(PolarOutOfRanges,
                                                'BigAngleOfAttack'+fieldname)[1]
            Field_ = Field.ravel(order='F')
            AoA_   =   AoA.ravel(order='F')
            stdCL_ = stdCL.ravel(order='F')
            Ncases = len(Field_)

            for i in range(Ncases):
                if stdCL_[i] > useBigRangeValuesIf_StdCLisHigherThan:
                    newValue = J.interpolate__(AoA_[i], BigAoA, FieldOutOfRange)
                    Field_[i] = newValue
