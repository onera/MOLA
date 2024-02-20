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

'''
MOLA - WorkflowORAS.py

WORKFLOW ORAS

Collection of functions designed for CFD simulations of Open Rotor And Stator (ORAS)

File history:
01/04/2022 - M. Balmaseda - Creation
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
from . import Preprocess        as PRE
from . import JobManager        as JM
from . import WorkflowCompressor as WC

def prepareMesh4ElsA(mesh, **kwargs):
    '''
    Exactly like :py:func:`MOLA.WorkflowCompressor.prepareMesh4ElsA`
    '''
    return WC.prepareMesh4ElsA(mesh, **kwargs)

def prepareMainCGNS4ElsA(mesh='mesh.cgns', ReferenceValuesParams={},
        NumericalParams={}, OverrideSolverKeys ={}, Extractions=[],
        BodyForceInputData={}, writeOutputFields=True, Initialization={'method':'uniform'}, 
        TurboConfiguration={},
        BoundaryConditions=[],bladeFamilyNames=['Blade'],
        JobInformation={}, SubmitJob=False, FULL_CGNS_MODE=False, templates=dict(), 
        secondOrderRestart=False):
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

        OverrideSolverKeys : :py:class:`dict` of maximum 3 :py:class:`dict`
            exactly the same as in :py:func:`MOLA.Preprocess.prepareMainCGNS4ElsA`

        RPM : float
            revolutions per minute of the blade

        Extractions : :py:class:`list` of :py:class:`dict`
            List of extractions to perform during the simulation. See
            documentation of :func:`MOLA.Preprocess.prepareMainCGNS4ElsA`

        BodyForceInputData : :py:class:`dict`
            if provided, each key in this :py:class:`dict` is the name of a row family to model
            with body-force. The associated value is a sub-dictionary, with the following 
            potential entries:

                * model (:py:class:`dict`): the name of the body-force model to apply. Available models 
                  are: 'hall', 'blockage', 'Tspread', 'constant'.

                * rampIterations (:py:class:`dict`): The number of iterations to apply a ramp on source terms, 
                  starting from `BodyForceInitialIteration` (in `ReferenceValues['CoprocessOptions']`). 
                  If not given, there is no ramp (source terms are fully applied from the `BodyForceInitialIteration`).

                * other optional parameters depending on the **model** 
                  (see dedicated functions in :mod:`MOLA.BodyForceTurbomachinery`).

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

        FULL_CGNS_MODE : bool
            if :py:obj:`True`, put all elsA keys in a node ``.Solver#Compute``
            to run in full CGNS mode.

        templates : dict
            Main files to copy for the workflow. 
            By default, it is filled with the following values:

            .. code-block::python

                templates = dict(
                    job_template = '$MOLA/TEMPLATES/job_template.sh',
                    compute = '$MOLA/TEMPLATES/<WORKFLOW>/compute.py',
                    coprocess = '$MOLA/TEMPLATES/<WORKFLOW>/coprocess.py',
                    otherWorkflowFiles = ['monitor_loads.py'],
                )

        secondOrderRestart : bool
            If :py:obj:`True`, and if NumericalParams['time_algo'] is 'gear' or 'DualTimeStep' 
            (second order time integration schemes), prepare a second order restart, and allow 
            the automatic restart of such a case. By default, the value is :py:obj:`False`.

            .. important:: 
            
                This behavior works only if elsA reaches the final iteration given by ``niter``.
                If the simulation stops because of the time limit or because all convergence criteria
                have been reached, then the restart will be done at the first order, without raising an error.
                
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
    
    if isinstance(mesh,str):
        t = J.load(mesh)
    elif I.isTopTree(mesh):
        t = mesh
    else:
        raise ValueError('parameter mesh must be either a filename or a PyTree')

    IsUnstructured = PRE.hasAnyUnstructuredZones(t)
    TurboConfiguration = WC.getTurboConfiguration(t, BodyForceInputData=BodyForceInputData, **TurboConfiguration)
    FluidProperties = PRE.computeFluidProperties()
    if not 'Surface' in ReferenceValuesParams:
        ReferenceValuesParams['Surface'] = 1.0

    MainDirection = np.array([1,0,0]) # Strong assumption here
    YawAxis = np.array([0,0,1])
    PitchAxis = np.cross(YawAxis, MainDirection)
    ReferenceValuesParams.update(dict(PitchAxis=PitchAxis, YawAxis=YawAxis))

    ReferenceValues = PRE.computeReferenceValues(FluidProperties, **ReferenceValuesParams)
    PRE.appendAdditionalFieldExtractions(ReferenceValues, Extractions)


    if I.getNodeFromName(t, 'proc'):
        JobInformation['NumberOfProcessors'] = int(max(PRE.getProc(t))+1)
        Splitter = None
    else:
        Splitter = 'PyPart'

    if ('ChorochronicInterface' or 'stage_choro') in (bc['type'] for bc in BoundaryConditions):
      MSG = 'Chorochronic interface detected'
      print(J.WARN + MSG + J.ENDC)
      CHORO_TAG = True
      updateChoroTimestep(t, Rows = TurboConfiguration['Rows'], NumericalParams = NumericalParams)
    else:
        CHORO_TAG = False

    if BodyForceInputData: 
        NumericalParams['useBodyForce'] = True
        PRE.tag_zones_with_sourceterm(t)
    elsAkeysCFD      = PRE.getElsAkeysCFD(nomatch_linem_tol=1e-4, unstructured=IsUnstructured)
    elsAkeysModel    = PRE.getElsAkeysModel(FluidProperties, ReferenceValues, unstructured=IsUnstructured)
    elsAkeysNumerics = PRE.getElsAkeysNumerics(ReferenceValues, **NumericalParams, unstructured=IsUnstructured)

    if CHORO_TAG == True and Initialization['method'] != 'copy':
            MSG = 'Flow initialization failed. No initial solution provided. Chorochronic simulations must be initialized from a mixing plane solution obtained on the same mesh'
            print(J.FAIL + MSG + J.ENDC)
            raise Exception(J.FAIL + MSG + J.ENDC)
    
    if secondOrderRestart:
        secondOrderRestart = True if elsAkeysNumerics['time_algo'] in ['gear', 'dts'] else False
    
    if not 'PeriodicTranslation' in TurboConfiguration and \
        any([rowParams['NumberOfBladesSimulated'] > rowParams['NumberOfBladesInInitialMesh'] \
            for rowParams in TurboConfiguration['Rows'].values()]):
        t = WC.duplicateFlowSolution(t, TurboConfiguration)

    PRE.initializeFlowSolution(t, Initialization, ReferenceValues, secondOrderRestart=secondOrderRestart)

    WC.setMotionForRowsFamilies(t, TurboConfiguration)
    WC.setBoundaryConditions(t, BoundaryConditions, TurboConfiguration,
                            FluidProperties,ReferenceValues, bladeFamilyNames=bladeFamilyNames)    

    WC.computeFluxCoefByRow(t, ReferenceValues, TurboConfiguration)

    WC.addMonitoredRowsInExtractions(Extractions, TurboConfiguration)

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

    AllSetupDicts = dict(Workflow='ORAS',
                        Splitter=Splitter,
                        JobInformation=JobInformation,
                        TurboConfiguration=TurboConfiguration,
                        FluidProperties=FluidProperties,
                        ReferenceValues=ReferenceValues,
                        elsAkeysCFD=elsAkeysCFD,
                        elsAkeysModel=elsAkeysModel,
                        elsAkeysNumerics=elsAkeysNumerics,
                        Extractions=Extractions)
                         
    if BodyForceInputData: 
        AllSetupDicts['BodyForceInputData'] = BodyForceInputData

    PRE.addTrigger(t)

    is_unsteady = AllSetupDicts['elsAkeysNumerics']['time_algo'] != 'steady'
    avg_requested = AllSetupDicts['ReferenceValues']['CoprocessOptions']['FirstIterationForFieldsAveraging'] is not None

    if is_unsteady and not avg_requested:
        msg =('WARNING: You are setting an unsteady simulation, but no field averaging\n'
              'will be done since CoprocessOptions key "FirstIterationForFieldsAveraging"\n'
              'is set to None. If you want fields average extraction, please set a finite\n'
              'positive value to "FirstIterationForFieldsAveraging" and relaunch preprocess')
        print(J.WARN+msg+J.ENDC)

    PRE.addExtractions(t, AllSetupDicts['ReferenceValues'],
                          AllSetupDicts['elsAkeysModel'],
                          extractCoords=False,
                          BCExtractions=ReferenceValues['BCExtractions'],
                          add_time_average= is_unsteady and avg_requested,
                          secondOrderRestart=secondOrderRestart)


    PRE.addReferenceState(t, AllSetupDicts['FluidProperties'],
                         AllSetupDicts['ReferenceValues'])
    dim = int(AllSetupDicts['elsAkeysCFD']['config'][0])
    PRE.addGoverningEquations(t, dim=dim)
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

    templates.setdefault('otherWorkflowFiles', [])
    if 'monitor_loads.py' not in templates['otherWorkflowFiles']:
        templates['otherWorkflowFiles'].append('monitor_loads.py')
    JM.getTemplates('Compressor', templates, JobInformation=JobInformation)
    if 'DIRECTORY_WORK' in JobInformation:
        PRE.sendSimulationFiles(JobInformation['DIRECTORY_WORK'], overrideFields=writeOutputFields)

    for i in range(SubmitJob):
        singleton = False if i==0 else True
        JM.submitJob(JobInformation['DIRECTORY_WORK'], singleton=singleton)

    J.printElapsedTime('prepareMainCGNS4ElsA took ', toc)

def updateChoroTimestep(t, Rows, NumericalParams):
    '''
    Compute the timestep for chorochronic simulations if not provided.
    
    Parameters
    ----------

        t : PyTree
            Tree to modify

        Rows : :py:class:`dict`
            Dictionary of Rows as provided in TurboConfiguration for the prepareMainCGNS function.

        NumericalParams : :py:class:`dict`
            dictionary containing the numerical settings for elsA. Similar to that required in prepareMainCGNS function.

    '''   
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

    if 'timestep' not in NumericalParams.keys():
        MSG = 'Time-step not provided by the user. Computating of a suitable time-step based on stage properties.'
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
        NewNquo = DeltaT/NumericalParams['timestep']
        Nquo_round = np.round(NewNquo)
        print()
        if np.absolute(NewNquo-Nquo_round)>1e-08:
            MSG = 'Choice of time-step does no seem to be suited for the case. Check the following parameters:'
            print(J.WARN + MSG + J.ENDC)

    MSG = 'Nquo : %s'%(NewNquo)
    print(J.WARN + MSG + J.ENDC)    
    
    MSG = 'Time step : %s'%(NumericalParams['timestep'])
    print(J.WARN + MSG + J.ENDC)

    MSG = 'Number of time step per period for row 1 : %s'%(per_Row1/NumericalParams['timestep'])
    print(J.WARN + MSG + J.ENDC)

    MSG = 'Number of time step per period for row 2 : %s'%(per_Row2/NumericalParams['timestep'])
    print(J.WARN + MSG + J.ENDC) 


def setRadiusAsChannelHeight(t):
    '''
    Compute the variable *ChannelHeight* from a mesh PyTree **t**. This function
    relies on the ETC module.

    Parameters
    ----------

        t : PyTree
            input mesh tree

        fsname : str
            Name of the ``FlowSolution_t`` container to stock the variable at
            nodes *ChannelHeight*.

    Returns
    -------

        t : PyTree
            modified tree

    '''
    OLD_FlowSolutionNodes = I.__FlowSolutionNodes__
    print(J.CYAN + 'Adding Radius as ChannelHeight in the mesh...' + J.ENDC)

    I._rmNodesByName(t, 'FlowSolution#Height')
    I.__FlowSolutionNodes__ = 'FlowSolution#Height'
    C._initVars(t, '{ChannelHeight} = sqrt({CoordinateY}*{CoordinateY}+{CoordinateZ}*{CoordinateZ})')
    for node in I.getNodesFromNameAndType(t, 'FlowSolution#Height', 'FlowSolution_t'):
        I.newGridLocation(value='Vertex', parent=node)

    I.__FlowSolutionNodes__ = OLD_FlowSolutionNodes
    print(J.GREEN + 'done.' + J.ENDC)
    return t

def computeLoadRadialDistribution(surface, row, torque_center= None):

    def searchBladeInTree(row):
        famnames = ['*BLADE*'.format(row), '*Blade*'.format(row),
                    '*AUBE*'.format(row), '*Aube*'.format(row)]
        for famname in famnames:
            for bladeSurface in I.getNodesFromNameAndType(surface, famname, 'CGNSBase_t'):
                if I.getNodeFromNameAndType(bladeSurface, row, 'Family_t') and I.getZones(bladeSurface) != []:
                    return bladeSurface

    try:
        setup = J.load_source('setup', 'setup.py')
        TurboConfiguration = setup.TurboConfiguration
        ReferenceValues = setup.ReferenceValues
        reference_pressure = ReferenceValues['Pressure']
        print('Setup OK')
    except:
        setup = None

    distribution = np.linspace(0,1,50)

    blade_surf = searchBladeInTree(row)
    
    if torque_center == None:
        if 'TorqueCenter' in TurboConfiguration['Rows'][row].keys():
            torque_center = TurboConfiguration['Rows'][row]['TorqueCenter'], #to be retrieved
        else:
            torque_center = ReferenceValues['TorqueOrigin'] #to be retrieved
    
    print('Torque center:',torque_center)

    sectionalLoads = computeLoadRadialDistributionInAnnularConfiguration(blade_surf, distribution=distribution, slicing_method='AbscissaBased', geometrical_parameters=dict(start_point=[0.,0.,0.],end_point=None,axis_direction=[1.,0.,0.]),torque_center=torque_center, reference_pressure=reference_pressure, CustomVariable=None)
    sectionalLoads = I.renameNode(sectionalLoads, 'SectionalLoads', f'{row}_SectionalLoads')
    I.addChild(surface, sectionalLoads)
    

def computeLoadRadialDistributionInAnnularConfiguration(surface, distribution, slicing_options=dict(slicing_method='SpanBased',custom_variable=None), geometrical_parameters=dict(start_point=None,end_point=None, axis_direction=None),
        torque_center=[0,0,0], reference_pressure=0.):
    '''
    Compute the sectional loads (spanwise distributions) along a direction 
    from a set of surfaces

    Parameters
    ----------
    
        surface : PyTree, Base, Zone or :py:class:`list` of Zone
            surfaces from which sectional loads are to be computed

            .. note::
                surfaces contained in **t** must contain the following fields
                 (preferrably at centers): ``Pressure``, ``SkinFrictionX``,
                ``SkinFrictionY``, ``SkinFrictionZ``. It may also contain 
                normals ``nx``, ``ny``, ``nz``. Otherwise they are computed.

        slicing_method : str
            Options:
            - SpanBased: computes the span based on 2 points (see below) provided by the user.
            - AbscissaBased: computes the abscissa based on the distance d to the axis provided by the user
              Abscissa = (d-dmin)/(dmax-dmin)


        start_point : 3-float :py:class:`list` or :py:class:`tuple` or :py:class:`numpy`
            :math:`(x,y,z)` coordinates of the starting point from which 
            sectional loads are to be computed

        end_point : 3-float :py:class:`list` or :py:class:`tuple` or :py:class:`numpy`
            :math:`(x,y,z)` coordinates of the end point up to which 
            sectional loads are to be computed
        
        axis_direction : 3-float :py:class:`list` or :py:class:`tuple` or :py:class:`numpy`
            :math:`(x,y,z)` direction of the reference axis along which 
            sectional loads are to be computed

        distribution : 1D :py:class:`float` list or :py:class:`numpy`
            dimensionless coordinate (from *start_point* to *end_point*) used 
            for discretizing the sectional loads. This must be :math:`\in [0,1]`.

            .. hint:: for example 

                >>> distribution = np.linspace(0,1,200)

        torque_center : 3-float :py:class:`list` or :py:class:`tuple` or :py:class:`numpy`
            center for computation the torque contributions

        reference_pressure : float
            Reference pressure. Put ambiant pressure as a reference for integration over a surface that is not closed (such as blades).

    '''
    import MOLA.Wireframe as W
    
    def Abscissa(d): return (d-dmin)/(dmax-dmin)
    def Theta (y, z): return np.arctan2(z,y)
    def ThetaProjection(vecty, vectz, Theta): return vectz*np.cos(Theta)-vecty*np.sin(Theta)
    def RProjection(vecty, vectz, Theta): return vecty*np.cos(Theta)+vectz*np.sin(Theta)
  
    
    if slicing_options['slicing_method'] == 'SpanBased':
        if geometrical_parameters['start_point'] == None or geometrical_parameters['end_point'] == None:
            ERRMSG = 'Span based sectional load computation requires both start_point and end_point as input parameters'
            raise ValueError(ERRMSG)
        else:
            Post.computeAndAddSpanToSurface(surface, geometrical_parameters['start_point'], geometrical_parameters['end_point'])
            slicing_var = 'Span'

    elif slicing_options['slicing_method'] == 'AbscissaBased':
        if geometrical_parameters['start_point'] == None or geometrical_parameters['axis_direction']== None:
            ERRMSG = 'Abscissa based sectional load computation requires both start_point and axis_direction as input parameters'
            raise ValueError(ERRMSG)
        else:
            W.addDistanceRespectToLine(surface, np.array(geometrical_parameters['start_point']), np.array(geometrical_parameters['axis_direction']), FieldNameToAdd='Distance2Axis')
            
            dmin = C.getMinValue(surface, 'Distance2Axis')
            print('dmin=',dmin)
            dmax = C.getMaxValue(surface, 'Distance2Axis')
            print('dmax=',dmax)
            surface = C.initVars(surface,'Abscissa', Abscissa, ['Distance2Axis']) 
            slicing_var = 'Abscissa'

    
    elif slicing_options['slicing_method'] == 'Custom':
            if slicing_options['custom_variable'] == None:
                ERRMSG = 'The user needs to provide a CustomVariable value when performing custom variable based sectional load computation.'
                raise ValueError(ERRMSG)
            else:
            
                dmin = C.getMinValue(surface, CustomVariable)
                dmax = C.getMaxValue(surface, CustomVariable)
                slicing_var = CustomVariable
               
    print('dmin=',dmin)
    print('dmax=',dmax)

    SectionalForceX      = []
    SectionalForceY      = []
    SectionalForceZ      = []
    SectionalForceTheta  = []
    SectionalForceR      = []
    SectionalTorqueX     = []
    SectionalTorqueX2    = []
    SectionalTorqueY     = []
    SectionalTorqueZ     = []
    SectionalTorqueTheta = []
    SectionalTorqueR     = []
    SectionalSpan        = []

    sectionalLoads = I.newCGNSBase('SectionalLoads', cellDim=1, physDim=3, parent=None)

    for d in distribution:
        if slicing_options['slicing_method'] != 'Custom':
            section = Post.isoSurface(surface, fieldname=slicing_var, value=d, container='FlowSolution')
        else:
            value = d*(dmax-dmin)+dmin
            section = Post.isoSurface(surface, fieldname=slicing_var, value=value, container='FlowSolution')
        if not section: continue
        section = T.join(section)

        I.__FlowSolutionNodes__ = 'BCDataSetV'
        section = C.initVars(section,'Theta', Theta, ['CoordinateY','CoordinateZ'])
        section = C.initVars(section,'ntheta', ThetaProjection, ['ny','nz','Theta'])
        section = C.initVars(section,'nr', RProjection, ['ny','nz','Theta'])
        section = C.initVars(section,'SkinFrictionTheta', ThetaProjection, ['SkinFrictionY','SkinFrictionZ','Theta'])
        section = C.initVars(section,'SkinFrictionR', RProjection, ['SkinFrictionY','SkinFrictionZ','Theta'])
        
        C._normalize(section,['nx','ny','nz'])
        C._initVars(section, 'fx=-({Pressure}-%s)*{nx}+{SkinFrictionX}'%(reference_pressure))
        C._initVars(section, 'fy=-({Pressure}-%s)*{ny}+{SkinFrictionY}'%(reference_pressure))
        C._initVars(section, 'fz=-({Pressure}-%s)*{nz}+{SkinFrictionZ}'%(reference_pressure))
    
        C._initVars(section, 'ftheta', ThetaProjection, ['fy','fz','Theta'])
        C._initVars(section, 'fr', RProjection, ['fy','fz','Theta'])
        C._initVars(section, '{mx}={Distance2Axis}*{ftheta}')
        C._initVars(section, '{mr}=-({CoordinateX}-'+str(torque_center[0])+')*{ftheta}')
        C._initVars(section, '{mtheta}=-{Distance2Axis}*{fx}+({CoordinateX}-'+str(torque_center[0])+')*{fr}')
        
        # computation of sectional forces
        SectionalForceX += [ -P.integ(section,'fx')[0] ]
        SectionalForceY += [ -P.integ(section,'fy')[0] ]
        SectionalForceZ += [ -P.integ(section,'fz')[0] ]
        SectionalForceTheta += [ -P.integ(section,'ftheta')[0] ]
        SectionalForceR     += [ -P.integ(section,'fr')[0] ]
    
        # computation of sectional torques
        STorqueX, STorqueY, STorqueZ = P.integMoment(section, center=torque_center,
                                    vector=['fx','fy','fz'])
        
        SectionalTorqueX     += [ -STorqueX ]
        SectionalTorqueY     += [ -STorqueY ]
        SectionalTorqueZ     += [ -STorqueZ ]
        SectionalTorqueX2    += [ -P.integ(section,'mx')[0] ]
        SectionalTorqueTheta += [ -P.integ(section,'mtheta')[0] ]
        SectionalTorqueR     += [ -P.integ(section,'mr')[0] ]
    
        if slicing_options['slicing_method'] != 'Custom':
            SectionalSpan        += [ d ]
        else:
            SectionalSpan        += [ value ]

    sloads = dict(SectionalForceX=np.array(SectionalForceX),SectionalForceY=np.array(SectionalForceY),SectionalForceZ=np.array(SectionalForceZ),
                  SectionalForceTheta=np.array(SectionalForceTheta),SectionalForceR=np.array(SectionalForceR),
                  SectionalTorqueX=np.array(SectionalTorqueX), SectionalTorqueY=np.array(SectionalTorqueY),
                  SectionalTorqueZ=np.array(SectionalTorqueZ), SectionalTorqueX4Check=np.array(SectionalTorqueX4Check), 
                  SectionalTorqueTheta=np.array(SectionalTorqueTheta), SectionalTorqueR=np.array(SectionalTorqueR),
                  SectionalSpan=np.array(SectionalSpan))
     
 
    varValues = []
    varNames = []

    for key in sloads.keys():
        varNames.append(key)
        varValues.append(sloads[key])
        
    sectionalLoads = J.createZone('SectionalLoads',Arrays=varValues,Vars=varNames) 

    return sectionalLoads




def computePressureCoefficent(surface,row, hlist=all, distribution=np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98]), slicing_options=dict(slicing_method='SpanBased',custom_variable=None)):

    def searchBladeInTree(row):
        famnames = ['*BLADE*'.format(row), '*Blade*'.format(row),
                    '*AUBE*'.format(row), '*Aube*'.format(row)]
        for famname in famnames:
            for bladeSurface in I.getNodesFromNameAndType(surface, famname, 'CGNSBase_t'):
                if I.getNodeFromNameAndType(bladeSurface, row, 'Family_t') and I.getZones(bladeSurface) != []:
                    return bladeSurface

    try:
        setup = J.load_source('setup', 'setup.py')
        TurboConfiguration = setup.TurboConfiguration
        ReferenceValues = setup.ReferenceValues
        FluidProperties = setup.FluidProperties
        Pinf = ReferenceValues['Pressure']
        Roinf = ReferenceValues['Density']
        Minf = ReferenceValues['Mach']
        Gamma = FluidProperties['Gamma']
    except:
        setup = None   

    
    blade_surf = searchBladeInTree(row)
    blade_slices = Post.computeCp(blade_surf, distribution=distribution, slicing_options=slicing_options, 
                   geometrical_parameters=dict(start_point=[0.,0.,0],end_point=None, axis_direction=[1.,0.,0]), 
                   reference_state = dict(reference_pressure=Pinf, reference_density=Roinf, reference_mach=Minf, gamma=Gamma,rotation_speed = TurboConfiguration['Rows'][row]['RotationSpeed']))
    blade_slices = I.renameNode(blade_slices, 'Slices', f'{row}_Slices')
    # fsnodes = I.getNodesFromType(blade_slices,'FlowSolution_t')
    # for node in fsnodes:
    #     I.setName(node,'FlowSolution')
    I.addChild(surface, blade_slices)