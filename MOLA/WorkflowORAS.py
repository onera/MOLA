'''
MOLA - WorkflowORAS.py

WORKFLOW ORAS

Collection of functions designed for CFD simulations of Open Rotor And Stator (ORAS)

File history:
01/04/2022 - M. Balmaseda - Creation
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
from . import Preprocess        as PRE
from . import JobManager        as JM
from . import WorkflowCompressor as WC

def prepareMesh4ElsA(mesh, **kwargs):
    '''
    Exactly like :py:func:`MOLA.Preprocess.prepareMesh4ElsA`
    '''
    return WC.prepareMesh4ElsA(mesh, **kwargs)

def prepareMainCGNS4ElsA(mesh='mesh.cgns', ReferenceValuesParams={},
        NumericalParams={}, Extractions=[],
        writeOutputFields=True, Initialization={'method':'uniform'}, TurboConfiguration={},
        BoundaryConditions=[],bladeFamilyNames=['Blade'],
        JobInformation={}, SubmitJob=False, FULL_CGNS_MODE=False, COPY_TEMPLATES=True):
    '''
    This is mainly a function similar to :func:`MOLA.Preprocess.prepareMainCGNS4ElsA`
    but adapted to ORAS mono-chanel computations. Its purpose is adapting
    the CGNS to elsA.

    Parameters
    ----------

        mesh : :py:class:`str` or PyTree
            if the input is a :py:class:`str`, then such string specifies the
            path to file (usually named ``mesh.cgns``) where the result of
            function :py:func:`prepareMesh4ElsA` has been writen. Otherwise,
            **mesh** can directly be the PyTree resulting from :func:`prepareMesh4ElsA`

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

        RPM : float
            revolutions per minute of the blade

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

    def addFieldExtraction(fieldname):
        try:
            FieldsExtr = ReferenceValuesParams['FieldsAdditionalExtractions']
            if fieldname not in FieldsExtr.split():
                FieldsExtr += ' '+fieldname
        except:
            ReferenceValuesParams['FieldsAdditionalExtractions'] = fieldname

    if isinstance(mesh,str):
        t = C.convertFile2PyTree(mesh)
    elif I.isTopTree(mesh):
        t = mesh
    else:
        raise ValueError('parameter mesh must be either a filename or a PyTree')

    FluidProperties = PRE.computeFluidProperties()
    if not 'Surface' in ReferenceValuesParams:
        ReferenceValuesParams['Surface'] = 1.0

    MainDirection = np.array([1,0,0]) # Strong assumption here
    YawAxis = np.array([0,0,1])
    PitchAxis = np.cross(YawAxis, MainDirection)
    ReferenceValuesParams.update(dict(PitchAxis=PitchAxis, YawAxis=YawAxis))

    ReferenceValues = PRE.computeReferenceValues(FluidProperties, **ReferenceValuesParams)


    if I.getNodeFromName(t, 'proc'):
        JobInformation['NumberOfProcessors'] = int(max(PRE.getProc(t))+1)
        Splitter = None
    else:
        Splitter = 'PyPart'

    if ('ChorochronicInterface' or 'stage_choro') in (bc['type'] for bc in BoundaryConditions):
      MSG = 'Chorochronic interface detected'
      print(J.WARN + MSG + J.ENDC)
      CHORO_TAG = True
      updateChoroTimestep(t, Rows =TurboConfiguration['Rows'], NumericalParams = NumericalParams)
    else:
        CHORO_TAG = False

    elsAkeysCFD      = PRE.getElsAkeysCFD(nomatch_linem_tol=1e-4)
    elsAkeysModel    = PRE.getElsAkeysModel(FluidProperties, ReferenceValues)
    elsAkeysNumerics = PRE.getElsAkeysNumerics(ReferenceValues, **NumericalParams)

    if CHORO_TAG == True and Initialization['method'] != 'copy':
        MSG = 'Flow initialization failed'
        print(J.FAIL + MSG + J.ENDC)
        raise ValueError('No initial solution provided. Chorochronic simulations must be initialized from a mixing plane solution obtained on the same mesh.')
    PRE.initializeFlowSolution(t, Initialization, ReferenceValues)

    WC.setBoundaryConditions(t, BoundaryConditions, TurboConfiguration,
                            FluidProperties,ReferenceValues, bladeFamilyNames=bladeFamilyNames)

    WC.computeFluxCoefByRow(t, ReferenceValues, TurboConfiguration)

    AllSetupDics = dict(Workflow='ORAS',
                        Splitter=Splitter,
                        TurboConfiguration=TurboConfiguration,
                        FluidProperties=FluidProperties,
                        ReferenceValues=ReferenceValues,
                        elsAkeysCFD=elsAkeysCFD,
                        elsAkeysModel=elsAkeysModel,
                        elsAkeysNumerics=elsAkeysNumerics,
                        Extractions=Extractions)

    BCExtractions = dict(
    BCWall = ['normalvector', 'frictionvector','psta', 'bl_quantities_2d', 'yplusmeshsize'],
    )

    PRE.addTrigger(t)
    PRE.addExtractions(t, AllSetupDics['ReferenceValues'],
                          AllSetupDics['elsAkeysModel'], extractCoords=False,BCExtractions=BCExtractions)

    if elsAkeysNumerics['time_algo'] != 'steady':
        if 'FirstIterationForAverage' in AllSetupDics['ReferenceValues']['CoprocessOptions'].keys():
            PRE.addAverageFieldExtractions(t, AllSetupDics['ReferenceValues'],
                AllSetupDics['ReferenceValues']['CoprocessOptions']['FirstIterationForAverage'])
        else:
            PRE.addAverageFieldExtractions(t, AllSetupDics['ReferenceValues'])

    PRE.addReferenceState(t, AllSetupDics['FluidProperties'],
                         AllSetupDics['ReferenceValues'])
    dim = int(AllSetupDics['elsAkeysCFD']['config'][0])
    PRE.addGoverningEquations(t, dim=dim)
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
        JM.getTemplates('Compressor', JobInformation=JobInformation)

        if 'DIRECTORY_WORK' in JobInformation:
            PRE.sendSimulationFiles(JobInformation['DIRECTORY_WORK'],
                                    overrideFields=writeOutputFields)

        if SubmitJob: JM.submitJob(JobInformation['DIRECTORY_WORK'])


def updateChoroTimestep(t, Rows, NumericalParams):
    
    if 'timestep' not in NumericalParams.keys():
        MSG = 'Time-step not provided by the user. Computating of a suitable time-step based on stage properties.'
        print(J.WARN + MSG + J.ENDC)
        rowNameList = list(Rows.keys())
    
        Nblade_Row1 = Rows[rowNameList[0]]['NumberOfBlades']
        Nblade_Row2 = Rows[rowNameList[1]]['NumberOfBlades']
        omega_Row1 = Rows[rowNameList[0]]['RotationSpeed']
        omega_Row2 = Rows[rowNameList[1]]['RotationSpeed']

        per_Row1 = (2*np.pi)/(Nblade_Row2*np.abs(omega_Row1-omega_Row2))
        per_Row2 = (2*np.pi)/(Nblade_Row1*np.abs(omega_Row1-omega_Row2))

        gcd =np.gcd(Nblade_Row1,Nblade_Row2)
        
        DeltaT = gcd*2*np.pi/(np.abs(omega_Row1-omega_Row2)*Nblade_Row1*Nblade_Row2) #Largest time step that is a fraction of the period of both Row1 and Row2.
        MSG = 'DeltaT : %s'%(DeltaT)
        print(J.WARN + MSG + J.ENDC)
        Nquo = 10
        time_step = DeltaT/Nquo
    
        NewNquo = Nquo
        while time_step*np.abs(omega_Row1-omega_Row2)*180./np.pi> 0.06:
            NewNquo = NewNquo+10
            time_step = DeltaT/NewNquo

    
        NumericalParams['timestep'] = time_step

    
    else:
        MSG = 'Time-step provided by the user.'
        print(J.WARN + MSG + J.ENDC)

    MSG = 'Nquo : %s'%(NewNquo)
    print(J.WARN + MSG + J.ENDC)    
    
    MSG = 'Time step : %s'%(NumericalParams['timestep'])
    print(J.WARN + MSG + J.ENDC)

    MSG = 'Number of time step per period for row 1 : %s'%(per_Row1/NumericalParams['timestep'])
    print(J.WARN + MSG + J.ENDC)

    MSG = 'Number of time step per period for row 2 : %s'%(per_Row2/NumericalParams['timestep'])
    print(J.WARN + MSG + J.ENDC) 