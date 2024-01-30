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
PREPROCESS module

It implements a collection of routines for preprocessing of CFD simulations

23/12/2020 - L. Bernardos - creation by recycling
'''

import MOLA
from . import InternalShortcuts as J
from . import GenerativeShapeDesign as GSD
from . import GenerativeVolumeDesign as GVD
from . import ExtractSurfacesProcessor as ESP
from . import JobManager as JM

if not MOLA.__ONLY_DOC__:
    from multiprocessing.sharedctypes import Value
    import sys
    import os
    import shutil
    import pprint
    import glob
    import numpy as np
    from itertools import product
    import copy
    from timeit import default_timer as tic
    import datetime

    import Converter.PyTree as C
    import Converter.Internal as I
    import Connector.PyTree as X
    import Transform.PyTree as T
    import Generator.PyTree as G
    import Geom.PyTree as D
    import Post.PyTree as P
    import Distributor2.PyTree as D2
    import Converter.elsAProfile as EP
    import Intersector.PyTree as XOR
    import Dist2Walls.PyTree as DTW


load = J.load 
save = J.save

DEBUG = False
DIRECTORY_OVERSET='OVERSET' 

K_OMEGA_TWO_EQN_MODELS = ['Wilcox2006-klim', 'Wilcox2006-klim-V',
            'Wilcox2006', 'Wilcox2006-V', 'SST-2003', 
            'SST-V2003', 'SST', 'SST-V',  'BSL', 'BSL-V']

K_OMEGA_MODELS = K_OMEGA_TWO_EQN_MODELS + [ 'SST-2003-LM2009',
                 'SST-V2003-LM2009', 'SSG/LRR-RSM-w2012']

AvailableTurbulenceModels = K_OMEGA_MODELS + ['smith', 'SA']


def prepareMesh4ElsA(InputMeshes, splitOptions={}, globalOversetOptions={}):
    '''
    This is a macro-function used to prepare the mesh for an elsA computation
    from user-provided instructions in form of a list of python dictionaries.

    The sequence of operations performed are the following:

    #. load and assemble the meshes, eventually generate background
    #. apply transformations
    #. apply connectivity
    #. set the boundary conditions
    #. split the mesh
    #. distribute the mesh
    #. add and group families
    #. perform overset preprocessing
    #. make final elsA-specific adaptations of CGNS data

    Parameters
    ----------

        InputMeshes : :py:class:`list` of :py:class:`dict`
            User-provided data.

            Each list item corresponds to a CGNS Base element (in the sense of an
            overset component). Most prioritary items (less likely to be masked)
            must be placed first of the list. Hence, background grid (most likely to
            be masked) must always be last item of the list.

            Each list item is a Python dictionary with special keywords.
            Possible pairs of keywords and its associated values are presented:

            * file : :py:class:`str` or PyTree
                if it is a :py:class:`str`, then it is the
                path of the file containing the grid. It can also be the special
                keyword ``'GENERATE'``. Alternatively, user can provide directly
                a PyTree (in-memory).

                .. attention:: each component must have a unique base. If
                    input file have several bases, please separate each base
                    into different files.

                .. note::
                    if special keyword ``'GENERATE'`` is provided, the automatic 
                    cartesian background grid generation algorithm is launched.
                    In this case, please consult 
                    :py:func:`MOLA.GenerativeVolumeDesign.buildCartesianBackground` 
                    for relevant options.

            * baseName : :py:class:`str`
                the new name to give to the component.

                .. attention:: each base name must be unique

            * Transform : :py:class:`dict`
                see :py:func:`transform` doc for more information on accepted
                values.

            * Connection : :py:class:`list` of :py:class:`dict`
                see :py:func:`connectMesh` doc for more information on accepted
                values.

                .. attention:: if the mesh is to be split, user must provide
                    the **Connection** attribute

            * BoundaryConditions : :py:class:`list` of :py:class:`dict`
                see :py:func:`setBoundaryConditions` doc for more information on
                accepted values.

            * OversetOptions : :py:class:`dict`
                see :py:func:`addOversetData` doc for more information on
                accepted values.

            * Motion : :py:class:`dict`
                Specifies if the block corresponds to a rotating overset grid.
                This is specifically designed for propeller or rotor blades.

                Acceptable keys are:

                * NumberOfBlades : :py:class:`int`
                    Specifies the number of blades of the component that will be
                    duplicated. New blades will be added to the main tree as 
                    new CGNS Bases and will be named ``<baseName>_#``.

                * InitialFrame : :py:class:`dict`
                    Specifies the position of the original mesh as placed in 
                    **file**. Possible keys are:

                    RotationCenter : 3-float :py:class:`list`
                        :math:`(x,y,z)` coordinates of the rotation point

                    RotationAxis : 3-float :py:class:`list`
                        :math:`(x,y,z)` components of the rotation axis

                    BladeDirection : 3-float :py:class:`list`
                        :math:`(x,y,z)` components of the direction of the blade,
                        from root to tip.

                    RightHandRuleRotation : :py:class:`bool`
                        if :py:obj:`True`, the rotation orientation of the blade
                        in the input **file** follows the right-hand-rule.

                * RequestedFrame : :py:class:`dict`
                    Specifies the requested final position of the component.
                    Possible keys are:

                    RotationCenter : 3-float :py:class:`list`
                        :math:`(x,y,z)` coordinates of the rotation point

                    RotationAxis : 3-float :py:class:`list`
                        :math:`(x,y,z)` components of the rotation axis

                    BladeDirection : 3-float :py:class:`list`
                        :math:`(x,y,z)` components of the direction of the blade,
                        from root to tip.

                    RightHandRuleRotation : :py:class:`bool`
                        if :py:obj:`True`, the rotation orientation of the blade
                        will follow the right-hand-rule.

            * SplitBlocks : :py:class:`bool`
                if :py:obj:`True`, allow for splitting this component in
                order to satisfy the user-provided rules of total number of used
                processors and block points load during simulation. If :py:obj:`False`,
                the component is protected against splitting.

                .. attention:: split operation results in loss of connectivity information.
                    Hence, if ``SplitBlocks=True`` , then user must specify connection
                    rules in list **Connection**.

                .. danger:: you must **NOT** split blocks requiring *NearMatch*,
                    like e.g. octree cartesian structured blocks commonly used 
                    as background mesh in overset computations.

        splitOptions : dict
            All optional parameters passed to function :py:func:`splitAndDistribute`

        globalOversetOptions : dict
            All optional parameters passed to function :py:func:`addOversetData`

    Returns
    -------

        t : PyTree
            the pre-processed mesh tree (usually saved as ``mesh.cgns``)

            .. important:: This tree is **NOT** ready for elsA computation yet !
                The user shall employ function :py:func:`prepareMainCGNS4ElsA`
                as next step
    '''
    toc = tic()
    t = getMeshesAssembled(InputMeshes)
    if hasAnyUnstructuredZones(t):
        t = convertUnstructuredMeshToNGon(t,
                mergeZonesByFamily=False if splitOptions else True)
    transform(t, InputMeshes)
    t = connectMesh(t, InputMeshes)
    setBoundaryConditions(t, InputMeshes)
    if splitOptions: t = splitAndDistribute(t, InputMeshes, **splitOptions)
    addFamilies(t, InputMeshes)
    _writeBackUpFiles(t, InputMeshes)
    t = addOversetData(t, InputMeshes, **globalOversetOptions)
    adapt2elsA(t, InputMeshes)
    J.checkEmptyBC(t)

    J.printElapsedTime('prepareMesh4ElsA took:', toc)

    return t

def _writeBackUpFiles(t, InputMeshes):
    for b, meshInfo in zip(I.getBases(t), InputMeshes):
        try: backup_file = meshInfo['backup_file']
        except KeyError: continue
        if backup_file: J.save(b, backup_file)


def prepareMainCGNS4ElsA(mesh, ReferenceValuesParams={}, OversetMotion={},
        NumericalParams={}, OverrideSolverKeys={}, 
        Extractions=[{'type':'AllBCWall'}], BoundaryConditions=[],
        Initialization=dict(method='uniform'),
        BodyForceInputData=[], writeOutputFields=True,
        JobInformation={}, SubmitJob=False, templates=dict(), secondOrderRestart=False):
    r'''
    This macro-function takes as input a preprocessed grid file (as produced
    by function :py:func:`prepareMesh4ElsA` ) and adds all remaining information
    required by elsA computation.

    .. important:: Most of this adaptations are elsA-specific

    Several operations are performed on the main tree:

    #. add ``.Solver#BC`` nodes
    #. add trigger nodes
    #. add extraction nodes
    #. add reference state nodes
    #. add governing equations nodes
    #. initialize flowfields
    #. create links between ``FlowSolution#Init`` and ``OUTPUT/fields.cgns``

    Parameters
    ----------

        mesh : :py:class:`str` or PyTree
            if the input is a :py:class:`str`, then such string specifies the
            path to file (usually named ``mesh.cgns``) where the result of
            function :py:func:`prepareMesh4ElsA` has been writen. Otherwise,
            **mesh** can directly be the PyTree resulting from :py:func:`prepareMesh4ElsA`

        ReferenceValuesParams : dict
            Python dictionary containing the
            Reference Values and other relevant data of the specific case to be
            run using elsA. For information on acceptable values, please
            see the documentation of function :py:func:`computeReferenceValues` .

            .. note:: internally, this dictionary is passed as *kwargs* as follows:

                >>> PRE.computeReferenceValues(arg, **ReferenceValuesParams)

        OversetMotion : :py:class:`dict` of :py:class:`dict`.
            Set a motion (kinematic) law to each grid component (Base).
            The value of the :py:class:`dict` must correspond to a given component
            (**baseName** in **InputMeshes**). Each value is another :py:class:`dict`
            defining the kinematic parameters. Acceptable keys are:

            * RPM : :py:class:`int`
                Rotation speed (in Revolutions Per Minute) of the component.

            * Function : :py:class:`dict`
                A :py:class:`dict` with pairs of keywords and values defining 
                the elsA motion function. By default, ``'rotor_motion'`` function 
                is employed using as rotation point and rotation vector the 
                information provided by user in **Motion** parameter of function 
                :py:func:`prepareMesh4ElsA`.

                .. hint:: for information on elsA kinematic functions, please 
                    see `this page <http://elsa.onera.fr/restricted/MU_tuto/latest/MU-98057/Textes/Attribute/function.html?#attributes-of-the-function-class>`_. For detailed information on elsA ``rotor_motion`` 
                    function, please see `this page <http://elsa.onera.fr/restricted/MU_tuto/latest/MU-98057/Textes/HelicopterBladePosition.html>`_
                                      
        Extractions : :py:class:`list` of :py:class:`dict`
            List of extractions to perform during the simulation. For now, only
            surfacic extractions may be asked. See documentation of :func:`MOLA.Coprocess.extractSurfaces` for further details on the
            available types of extractions.

        BoundaryConditions : :py:class:`list` of :py:class:`dict`
            List of boundary conditions to set on the given mesh.
            For details, refer to documentation of :py:func:`MOLA.WorfklowCompressor.setBoundaryConditions`


        NumericalParams : dict
            dictionary containing the numerical
            settings for elsA. For information on acceptable values, please see
            the documentation of function :py:func:`getElsAkeysNumerics`

            .. note:: internally, this dictionary is passed as *kwargs* as follows:

                >>> PRE.getElsAkeysNumerics(arg, **NumericalParams)

        OverrideSolverKeys : :py:class:`dict` of maximum 3 :py:class:`dict`
            this is a dictionary containing up to three dictionaries with 
            meaningful keys ``cfdpb``, ``model`` and ``numerics``. The 
            information contained in each one of the aforementioned dictionaries 
            are user-specified elsA keys that will override (or add to) any
            previous key inferred by MOLA. For example:

            ::

                OverrideSolverKeys = {
                    'cfdpb':    dict(metrics_type    = 'barycenter'),
                    'model':    dict(k_prod_compute  = 'from_kato',
                                     walldistcompute = 'gridline'),
                    'numerics': dict(gear_iteration  = 20,
                                     av_mrt          = 0.3),
                                      }

        Initialization : dict
            dictionary defining the type of initialization, using the key
            **method**. See documentation of :func:`initializeFlowSolution`

        BodyForceInputData : :py:class:`list` of :py:class:`dict`
            if provided, each item of this list constitutes a body-force modeling component.
            Currently acceptable pairs of keywords and associated values are:

            * name : :py:class:`str`
                the name to provide to the bodyforce component

            * proc : :py:class:`int`
                sets the processor at which the bodyforce component
                is associated for Lifting-Line operations.

                .. note:: **proc** must be :math:`\in (0, \mathrm{NumberOfProcessors}-1)`

            * FILE_LiftingLine : :py:class:`str`
                path to the LiftingLine CGNS file to
                consider for the bodyforce modeling element.

                .. attention:: LiftingLine curve must be placed in native location
                    as resulting from :py:func:`MOLA.LiftingLine.buildLiftingLine` function !
                    This is required for coherent use of relocation functions
                    employed by this method.

            * FILE_Polars : :py:class:`str`
                path to the Airfoil 2D polars CGNS file
                to employ for the lifting-line used as bodyforce modeling element

            * NumberOfBlades : :py:class:`int`
                the number of blades that the propeller or rotor is composed of

            * RotationCenter : :py:class:`list` of 3 :py:class:`float`
                the :math:`(x,y,z)` coordinates of the rotation center of the rotor.

            * RotationAxis : :py:class:`list` of 3 :py:class:`float`
                unitary vector (in absolute :math:`(x,y,z)` frame) where
                the rotor is located. **RotationAxis** points towards the desired
                Thrust direction of the propeller.

            * GuidePoint : :py:class:`list` of 3 :py:class:`float`
                :math:`(x,y,z)` coordinates of the point where
                the first blade will be approximately pointing to.

                .. attention:: This will be probably deprecated as it does not
                    seem useful in a body-force context

            * RightHandRuleRotation : :py:class:`bool`
                if :py:obj:`True`, the propeller rotates
                around **RotationAxis** vector following the right-hand-rule
                direction. if :py:obj:`False`, rotation follows the left-hand-rule.

            * NumberOfAzimutalPoints : :py:class:`int`
                number of azimutal points used to discretize the bodyforce disk

            * buildBodyForceDiskOptions : :py:class:`dict`
                Additional options defining the bodyforce element. Acceptable
                pairs of keywords and their associated values are:

                * ``'RPM'`` : :py:class:`float`
                    the number of revolutions per minute of the rotor,
                    if ``'CommandType'`` is not ``'RPM'``

                * ``'Pitch'`` : :py:class:`float`
                    The pitch to be applied to the blades if
                    ``'CommandType'`` is not ``'Pitch'``

                * ``'CommandType'`` : :py:class:`str`
                    Can be one of:

                    * ``'Pitch'``
                        adjusts the pitch in order to verify a constraint

                    * ``'RPM'``
                        adjusts the RPM in order to verify a constraint

                * ``'Constraint'`` : :py:class:`str`
                    constraint type to be satisfied by
                    controling the command defined by ``'CommandType'``:

                    * ``'Thrust'``
                        Aims a *thrust* value, to be specified in
                        ``'ConstraintValue'`` in [N]

                    * ``'Power'``
                        Aims a *power* value, to be specified in
                        ``'ConstraintValue'`` in [W]

                * ``'ConstraintValue'`` : :py:class:`float`
                    constraint value to be satisfied by
                    controling the command defined by ``'CommandType'``

                * ``'ValueTol'`` : :py:class:`float`
                    tolerance of ``'ConstraintValue'`` to determine
                    if the constraint is satisfied

                * ``'AttemptCommandGuess'`` : :py:class:`list` of 2 :py:class:`float`
                    Used as search bounds ``[min, max]`` for the trim procedure.

                    .. tip:: use as many sets of ``[min, max]`` items as
                        the desired number of attempts for trimming

                * ``'StackOptions'`` : :py:class:`dict`
                    parameters to be passed as  *kwargs* to function
                    :py:func:`MOLA.LiftingLine.stackBodyForceComponent`. Refer
                    to the documentation for more information on acceptable values

        writeOutputFields : bool
            if :py:obj:`True`, write initialized fields overriding
            a possibly existing ``OUTPUT/fields.cgns`` file. If :py:obj:`False`, no
            ``OUTPUT/fields.cgns`` file is writen, but in this case the user must
            provide a compatible ``OUTPUT/fields.cgns`` file to elsA (for example,
            using a previous computation result).

        JobInformation : dict
            Dictionary containing information to update the job file. For
            information on acceptable values, please see the documentation of
            function :func:`MOLA.JobManager.updateJobFile`

        SubmitJob : :py:class:`bool` or :py:class:`int`
            if :py:obj:`True`, submit the SLURM job based on information contained
            in **JobInformation**. If **SubmitJob** is an :py:class:`int` ``>1``,
            then will submit the specified number of jobs in queue with singleton
            dependency (useful for long simulations).

        templates : dict
            Main files to copy for the workflow. 
            By default, it is filled with the following values:

            .. code-block::python

                templates = dict(
                    job_template = '$MOLA/TEMPLATES/job_template.sh',
                    compute = '$MOLA/TEMPLATES/<WORKFLOW>/compute.py',
                    coprocess = '$MOLA/TEMPLATES/<WORKFLOW>/coprocess.py',
                    otherWorkflowFiles = [],
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

    toc = tic()
    if isinstance(mesh,str):
        t = C.convertFile2PyTree(mesh)
    elif I.isTopTree(mesh):
        t = mesh
    else:
        raise ValueError('parameter mesh must be either a filename or a PyTree')

    useBCOverlap = hasBCOverlap(t)

    if useBCOverlap:
        addFieldExtraction(ReferenceValuesParams, 'ChimeraCellType')
    if BodyForceInputData:
        addFieldExtraction(ReferenceValuesParams, 'Temperature')


    IsUnstructured = hasAnyUnstructuredZones(t)

    ReferenceValuesParams.setdefault('CoprocessOptions', dict())
    ReferenceValuesParams['CoprocessOptions'].setdefault('TagSurfacesWithIteration', 'auto')

    FluidProperties = computeFluidProperties()
    ReferenceValues = computeReferenceValues(FluidProperties,
                                             **ReferenceValuesParams)
    appendAdditionalFieldExtractions(ReferenceValues, Extractions)

    BCExtractions = ReferenceValues['BCExtractions']

    if I.getNodeFromName(t, 'proc'):
        JobInformation['NumberOfProcessors'] = int(max(getProc(t))+1)
        Splitter = None
    else:
        Splitter = 'PyPart'

    elsAkeysCFD      = getElsAkeysCFD(unstructured=IsUnstructured)
    elsAkeysModel    = getElsAkeysModel(FluidProperties, ReferenceValues,
                                        unstructured=IsUnstructured)
    if useBCOverlap: NumericalParams['useChimera'] = True
    if BodyForceInputData: 
        NumericalParams['useBodyForce'] = True
        tag_zones_with_sourceterm(t)
    elsAkeysNumerics = getElsAkeysNumerics(ReferenceValues,
                                unstructured=IsUnstructured, **NumericalParams)
    
    if secondOrderRestart:
        secondOrderRestart = True if elsAkeysNumerics['time_algo'] in ['gear', 'dts'] else False

    if useBCOverlap and not OversetMotion:
        elsAkeysNumerics['chm_interpcoef_frozen'] = 'active'
        elsAkeysNumerics['chm_conn_io'] = 'read' # NOTE ticket 8259
        elsAkeysNumerics['chm_impl_interp'] = 'none'
        elsAkeysNumerics['chm_ovlp_minimize'] = 'inactive'
        elsAkeysNumerics['chm_ovlp_thickness'] = 2
        elsAkeysNumerics['chm_preproc_method'] = 'mask_based'
        elsAkeysNumerics['chm_conn_fprefix'] = DIRECTORY_OVERSET+'/overset'
        
    if ReferenceValues['CoprocessOptions']['TagSurfacesWithIteration'] == 'auto': 
        if OversetMotion and elsAkeysNumerics['time_algo'] != 'steady':
            ReferenceValues['CoprocessOptions']['TagSurfacesWithIteration'] = True
            elsAkeysModel['walldistcompute'] = 'gridline'
        else:
            ReferenceValues['CoprocessOptions']['TagSurfacesWithIteration'] = False

    if BoundaryConditions:
        from . import WorkflowCompressor as WC
        WC.setBoundaryConditions(t, BoundaryConditions, dict(),
                                FluidProperties, ReferenceValues,
                                bladeFamilyNames=[])


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


    AllSetupDicts = dict(Workflow='Standard',
                        Splitter=Splitter,
                        JobInformation=JobInformation,
                        FluidProperties=FluidProperties,
                        ReferenceValues=ReferenceValues,
                        OversetMotion=OversetMotion,
                        elsAkeysCFD=elsAkeysCFD,
                        elsAkeysModel=elsAkeysModel,
                        elsAkeysNumerics=elsAkeysNumerics,
                        Extractions=Extractions)

    if BodyForceInputData: AllSetupDicts['BodyForceInputData'] = BodyForceInputData

    t = newCGNSfromSetup(t, AllSetupDicts, Initialization=Initialization,
                         FULL_CGNS_MODE=False, BCExtractions=BCExtractions, secondOrderRestart=secondOrderRestart)
    saveMainCGNSwithLinkToOutputFields(t,writeOutputFields=writeOutputFields)


    if not Splitter:
        print('REMEMBER : configuration shall be run using %s%d%s procs'%(J.CYAN,
                                                   JobInformation['NumberOfProcessors'],J.ENDC))
    else:
        print('REMEMBER : configuration shall be run using %s'%(J.CYAN + \
            Splitter + J.ENDC))

    JM.getTemplates('Standard', templates, JobInformation=JobInformation)
    if 'DIRECTORY_WORK' in JobInformation:
        sendSimulationFiles(JobInformation['DIRECTORY_WORK'], overrideFields=writeOutputFields)

    for i in range(SubmitJob):
        singleton = False if i==0 else True
        JM.submitJob(JobInformation['DIRECTORY_WORK'], singleton=singleton)

    ElapsedTime = str(datetime.timedelta(seconds=tic()-toc))
    hours, minutes, seconds = ElapsedTime.split(':')
    ElapsedTimeHuman = hours+' hours '+minutes+' minutes and '+seconds+' seconds'
    msg = 'prepareMainCGNS took '+ElapsedTimeHuman
    print(J.BOLD+msg+J.ENDC)

def tag_zones_with_sourceterm(t):
    '''
    Add node xdt_nature='sourceterm' that is mandatory to use body force.
    See https://elsa.onera.fr/issues/11496#note-6
    '''
    if I.getNodeFromName(t, 'FlowSolution#DataSourceTerm'):
        zones = [z for z in I.getZones(t) if I.getNodeFromName1(z, 'FlowSolution#DataSourceTerm')]
    else:
        zones = I.getZones(t)
    for zone in zones:
        solverParam = I.getNodeFromName1(zone, '.Solver#Param')
        if not solverParam:
            solverParam = I.createChild(zone, '.Solver#Param', 'UserDefinedData_t')
        I.newDataArray('xdt_nature', value='sourceterm', parent=solverParam)

def getMeshesAssembled(InputMeshes):
    '''
    This function reads the grid files provided by the user-provided list
    **InputMeshes** and merges all components into a unique tree, returned by the
    function.

    Parameters
    ----------

        InputMeshes : :py:class:`list` of :py:class:`dict`
            each component (:py:class:`dict`) is associated to a base.
            Two keys are *compulsory*:

            * file : :py:class:`str` or PyTree
                the CGNS file containing the grid and possibly
                other CGNS information. If a :py:class:`str` is provided, then
                it is interpreted as a file name in absolute or relative path.
                Alternatively, top CGNS PyTree can be provided in memory.

                .. danger:: It must contain only 1 base

            * baseName : :py:class:`str`
                the name to attribute to the new base associated to the grid
                component

    Returns
    -------

        t : PyTree
            assembled tree

    '''
    print('assembling meshes...')
    Trees = []
    NewComponents = dict()
    for i, meshInfo in enumerate(InputMeshes):
        filename = meshInfo['file']
        if filename == 'GENERATE': continue
        if not isinstance(filename, str):
            if not I.isTopTree(filename):
                MSG = 'the value of key "file" of InputMeshes item %d shall be'%i
                MSG+= ' either a str or a top CGNS PyTree.'
                raise ValueError(J.FAIL+MSG+J.ENDC)
            t = meshInfo['file']
            meshInfo['file'] = 'user_provided_in_memory'
        else:
            t = J.load(filename)
        bases = I.getBases(t)
        if len(bases) != 1:
            raise ValueError('InputMesh element in %s must contain only 1 base'%filename)
        base = bases[0]
        NewBases, NewMeshInfos = duplicateBlades(base, meshInfo)
    
        if NewBases:
            new_tree = C.newPyTree(NewBases)
            NewComponents[i] = NewMeshInfos

        else:
            try: 
                base[0] = meshInfo['baseName']
            except KeyError:
                meshInfo['baseName'] = base[0]
            J.set(base,'.MOLA#InputMesh',**meshInfo)
            new_tree = C.newPyTree([base]) 
        
        Trees += [ new_tree ]

    OriginalInputMeshes = copy.copy(InputMeshes)
    for k in NewComponents:
        original_item = OriginalInputMeshes[k]
        new_index = InputMeshes.index(original_item)
        InputMeshes[new_index:new_index] = NewComponents[k]
        InputMeshes.remove(original_item)

    if Trees: t = I.merge(Trees)
    else: t = C.newPyTree([])
    t_cart = GVD.buildCartesianBackground(t, InputMeshes)
    if t_cart is not None: t = I.merge([t, t_cart])
    t = I.correctPyTree(t, level=3)

    return t


def transform(t, InputMeshes):
    '''
    This function applies the ``'Transform'`` key contained in user-provided
    list of Python dictionaries **InputMeshes** as introduced in
    function :py:func:`prepareMesh4ElsA` documentation.

    .. important:: in all cases, this function forces the grid to be direct

    Parameters
    ----------

        t : PyTree
            Assembled PyTree as obtained from :py:func:`getMeshesAssembled`

            .. note:: tree **t** is modified

        InputMeshes : :py:class:`list` of :py:class:`dict`
            preprocessing information provided by user as defined in
            :py:func:`prepareMesh4ElsA` doc.

            Acceptable values concerning the :py:class:`dict` associated to
            the key ``'Transform'`` are the following:

                * 'scale' : :py:class:`float`
                    Scaling factor to apply to the grid component.

                    .. tip:: use this option to transform a grid built in
                             milimeters into meters

                * 'rotate' : :py:class:`list` of :py:class:`tuple`
                    List of rotation to apply to the grid component. Each rotation
                    is defined by 3 elements:

                        * a 3-tuple corresponding to the center coordinates

                        * a 3-tuple corresponding to the rotation axis

                        * a float (or integer) defining the angle of rotation in
                          degrees

                    .. tip:: this option is useful to change the orientation of
                             a mesh built in Autogrid 5.

    '''
    makeStructuredAdaptations(t)
    for meshInfo in InputMeshes:
        if 'Transform' not in meshInfo: continue

        base = I.getNodeFromName1(t, meshInfo['baseName'])

        if isinstance(meshInfo['Transform'], list):
            raise AttributeError(J.FAIL+'Transform attribute must be a dict, not a list'+J.ENDC)

        if 'scale' in meshInfo['Transform']:
            s = float(meshInfo['Transform']['scale'])
            T._homothety(base,(0,0,0),s)

        if 'rotate' in meshInfo['Transform']:
            for center, axis, ang in meshInfo['Transform']['rotate']:
                T._rotate(base, center, axis, ang)
    makeStructuredAdaptations(t)

def makeStructuredAdaptations(t):
    zones = getStructuredZones(t)
    for zone in zones:
        type, Ni, Nj, Nk, dim = I.getZoneDim(zone)
        for gc in I.getNodesFromType1(zone,'GridCoordinates_t'):
            for n in I.getNodesFromType1(gc,'DataArray_t'):
                if n[1] is None: continue
                if len(n[1].shape) != dim:
                    n[1] = n[1].reshape((Ni,Nj,Nk))
    T._makeDirect(zones)

def connectMesh(t, InputMeshes):
    '''
    This function applies connectivity to the grid following instructions given
    by the list of dictionaries associated to ``Connection`` key of each
    :py:class:`dict` item in **InputMeshes**.


    Parameters
    ----------

        t : PyTree
            assembled tree

            .. note:: tree **t** is modified

        InputMeshes : :py:class:`list` of :py:class:`dict`
            user-provided preprocessing instructions as introduced in
            :py:func:`prepareMesh4ElsA` documentation.

            Each item contained in the list associated to key ``Connection`` is
            a Python dictionary representing a connection instruction.

            Possible values for connection instructions:

            * ``'type'`` : :py:class:`str`
                indicates the type of the connection. Two possibilities:

                * ``'Match'``
                    makes an exact match within a prescribed **tolerance**

                * ``'NearMatch'``
                    makes a near-match operation using prescribed **tolerance**
                    and **ratio**

                * ``'PeriodicMatch'``
                    makes a periodic match operation using prescribed **tolerance**
                    and **angles**

            * ``'tolerance'`` : :py:class:`float`
                employed tolerance for the connection instruction

            * ``'ratio'`` : :py:class:`int`
                employed ratio for connection ``'type':'NearMatch'``

            * ``'rotationCenter'`` : :py:class:`list` of :py:class:`float`
                center of rotation if **type** = ``'PeriodicMatch'``.
                As passed to function *connectMatchPeriodic* of module
                ``Connector.PyTree`` in Cassiopee.
                The default value is [0., 0., 0.]

            * ``'rotationAngle'`` : :py:class:`list` of :py:class:`float`
                rotation angles for the three axes, in degrees.
                Relevant only for **type** = ``'PeriodicMatch'``.
                As passed to function *connectMatchPeriodic* of module
                ``Connector.PyTree`` in Cassiopee.
                The default value is [0., 0., 0.]

            * ``'translation'`` : :py:class:`list` of :py:class:`float`
                Vector of translation if for **type** = ``'PeriodicMatch'``.
                As passed to function *connectMatchPeriodic* of module
                ``Connector.PyTree`` in Cassiopee.
                The default value is [0., 0., 0.]

    Returns
    -------

        t : PyTree
            Modified tree

            .. note:: this returned tree is only needed for ``'PeriodicMatch'``
                operation.

    '''
    for meshInfo in InputMeshes:
        if 'Connection' not in meshInfo: continue
        for ConnectParams in meshInfo['Connection']:
            base = I.getNodeFromName1(t, meshInfo['baseName'])
            baseDim = I.getValue(base)[-1]
            ConnectionType = ConnectParams['type']
            print('connecting type {} at base {}'.format(ConnectionType,
                                                         meshInfo['baseName']))
            try: tolerance = ConnectParams['tolerance']
            except KeyError:
                print('connection tolerance not defined. Using tol=1e-8')
                tolerance = 1e-8

            if ConnectionType == 'Match':
                C._rmBCOfType(base,'BCMatch') # HACK https://elsa.onera.fr/issues/11400
                base_out = X.connectMatch(base, tol=tolerance, dim=baseDim)
            elif ConnectionType == 'NearMatch':
                try: ratio = ConnectParams['ratio']
                except KeyError:
                    print('NearMatch ratio was not defined. Using ratio=2')
                    ratio = 2

                base_out = X.connectNearMatch(base, ratio=ratio,
                                      tol=tolerance,
                                      dim=baseDim)
            elif ConnectionType == 'PeriodicMatch':
                try: rotationCenter = ConnectParams['rotationCenter']
                except: rotationCenter = [0., 0., 0.]
                try: rotationAngle = ConnectParams['rotationAngle']
                except: rotationAngle = [0., 0., 0.]
                try: translation = ConnectParams['translation']
                except: translation = [0., 0., 0.]
                base_out = X.connectMatchPeriodic(base,
                                                  rotationCenter=rotationCenter,
                                                  rotationAngle=rotationAngle,
                                                  translation=translation,
                                                  tol=tolerance,
                                                  dim=baseDim)
            else:
                ERRMSG = 'Connection type %s not implemented'%ConnectionType
                raise AttributeError(ERRMSG)
            base[2] = base_out[2]
    return t

def setBoundaryConditions(t, InputMeshes):
    '''
    This function is used for setting the boundary-conditions (including
    *BCOverlap*) to a grid by means of the user-provided set of preprocessing
    instructions **InputMeshes**.

    This function expects that item in **InputMeshes** contains a list of
    Python dictionaries as value contained in key  ``BoundaryConditions``.

    Each item of the provided list specifies an instruction for setting a BC.


    Parameters
    ----------

        t : PyTree
            Assembled PyTree as obtained from :py:func:`getMeshesAssembled`

            .. note:: tree **t** is modified

        InputMeshes : :py:class:`list` of :py:class:`dict`
            preprocessing information provided by user as defined in
            :py:func:`prepareMesh4ElsA` doc.

            Acceptable pairs of keywords and associated values for the python dictionary
            contained in the list attributed to ``BoundaryConditions`` are:

            * ``'location'`` : :py:class:`str`
                specifies the location of the boundary-condition to
                be set. Acceptable values are:

                * ``'imin', 'imax', 'jmin', 'jmax', 'kmin'`` or ``'kmax'``
                    which corresponds to the boundaries of a structured zone.

                * ``'special'``
                    this keyword indicates that a special location is specified.
                    Then, additional instructions are provided through keyword
                    ``'specialLocation'`` (see next)

            * ``'specialLocation'`` : :py:class:`str`
                must be specified if ``'location':'specialLocation'``.
                Currently available special locations are the following:

                * ``'plane#TAG#'``
                    sets the boundary-condition at windows that *entirely*
                    lays on a plane provided by user. Possible values of ``#TAG#``:

                    * ``'YZ'``
                        plane :math:`OYZ` (:math:`x=0`)

                    * ``'XZ'``
                        plane :math:`OXZ` (:math:`y=0`)

                    * ``'XY'``
                        plane :math:`OXY` (:math:`z=0`)

                * 'fillEmpty'
                    sets the boundary-condition at all windows with
                    undefined BC or connectivity.

                * 'fillEmptyAfterRemovingExistingBCType'
                    like the precedent, except
                    that it removes any pre-existing BCType as indicated by user.
                    The BCType to remove before appying fillEmpty is indicated
                    by ``'BCType2Remove'`` key (see next)

            * ``'BCType2Remove'`` : :py:class:`str`
                Specify the type of BC to remove before applying
                fillEmptyBC.

                .. note:: only relevant if user specifies:
                    ``'location':'special'`` AND
                    ``'specialLocation':'fillEmptyAfterRemovingExistingBCType'``

            * ``'name'`` : :py:class:`str`
                name of the BC to specify

            * ``'type'`` : :py:class:`str`
                can be any compatible with :py:class:`Converter.PyTree.addBC2Zone`
                including family specification (starting with ``'FamilySpecified:'``).

            * ``'familySpecifiedType'`` : :py:class:`str`
                Specifies the actual BC type if
                ``'type'`` instruction started with ``'FamilySpecified:'``.

            .. warning:: this interface can be significantly change in
                future versions

    '''
    print('setting boundary conditions...')

    for meshInfo in InputMeshes:
        if 'BoundaryConditions' not in meshInfo: continue
        print('setting BC at base '+meshInfo['baseName'])
        base = I.getNodeFromName1(t, meshInfo['baseName'])
        FillEmptyBC = False
        for zone in I.getZones(base):
            for BCinfo in meshInfo['BoundaryConditions']:
                if 'type' not in BCinfo:
                        MSGERR = ('Boundary-Condition setup error at base {}:\n'
                            'Mandatory key "type" is missing').format(
                                meshInfo['baseName'])
                        raise ValueError(J.FAIL+MSGERR+J.ENDC)

                try:
                    location = BCinfo['location']
                except:
                    if not BCinfo['type'].startswith('FamilySpecified:'):
                        MSGERR = ('Boundary-Condition setup error at base {}:\n'
                            'Key "location" is missing, which is only supported for\n'
                            'BCFamilies already present in the input mesh. However, in that\n'
                            'case it is expected that the user-provided BC component type\n'
                            'must start with the term "FamilySpecified:", which is\n'
                            'not verified, because user provided type: {}').format(
                                meshInfo['baseName'], BCinfo['type'])
                        raise ValueError(J.FAIL+MSGERR+J.ENDC)
                    continue

                if location == 'special':
                    SpecialLocation = BCinfo['specialLocation']

                    if SpecialLocation.startswith('plane'):
                        WindowTags = getWindowTagsAtPlane(zone,
                                                          planeTag=SpecialLocation)
                        for winTag in WindowTags:
                            C._addBC2Zone(zone, BCinfo['name'], BCinfo['type'],
                                                winTag)

                    elif SpecialLocation == 'fillEmpty':
                        FillEmptyBC = BCinfo

                    elif SpecialLocation == 'fillEmptyAfterRemovingExistingBCType':
                        C._rmBCOfType(base, BCinfo['BCType2Remove'])
                        FillEmptyBC = BCinfo

                    else:
                        print('specialLocation %s not implemented, ignoring'%SpecialLocation)

                else:
                    C._addBC2Zone(zone, BCinfo['name'], BCinfo['type'], location)

        if FillEmptyBC:
            baseDim = I.getValue(base)[-1]
            C._fillEmptyBCWith(base,FillEmptyBC['name'],BCinfo['type'],dim=baseDim)
    print('setting boundary conditions... done')

def getWindowTagsAtPlane(zone, planeTag='planeXZ', tolerance=1e-8):
    '''
    Returns the windows keywords of a structured zone that entirely lies (within
    a geometrical tolerance) on a plane provided by user.

    Parameters
    ----------

        zone : zone
            a structured zone

        planeTag : str
            a keyword used to specify the requested plane.
            Possible tags: ``'planeXZ'``, ``'planeXY'`` or ``'planeYZ'``

        tolerance : float
            maximum geometrical distance allowed to all window
            coordinates to be satisfied if the window is a valid candidate

    Returns
    -------

        WindowTagsAtPlane : :py:class:`list` of :py:class:`str`
            A list containing any of the
            following window tags: ``'imin', 'imax', 'jmin', 'jmax', 'kmin', 'kmax'``.

            .. important:: If no window lies on the plane, the function returns an empty list.
                If more than one window entirely lies on the plane, then the returned
                list will have several items.
    '''
    WindowTags = ('imin','imax','jmin','jmax','kmin','kmax')
    Windows = [GSD.getBoundary(zone, window=w) for w in WindowTags]

    if planeTag.endswith('XZ') or planeTag.endswith('ZX'):
        DistanceVariable = 'CoordinateY'
    elif planeTag.endswith('XY') or planeTag.endswith('YX'):
        DistanceVariable = 'CoordinateZ'
    elif planeTag.endswith('YZ') or planeTag.endswith('ZY'):
        DistanceVariable = 'CoordinateX'
    else:
        raise AttributeError('planeTag %s not implemented'%planeTag)

    WindowTagsAtPlane = []
    for window, tag in zip(Windows, WindowTags):
        PositiveDistance = C.getMaxValue(window, DistanceVariable)
        NegativeDistance = C.getMinValue(window, DistanceVariable)
        if abs(PositiveDistance) > tolerance: continue
        if abs(NegativeDistance) > tolerance: continue
        WindowTagsAtPlane += [tag]

    return WindowTagsAtPlane

def addFamilies(t, InputMeshes, tagZonesWithBaseName=True):
    '''
    This function is used to set all required CGNS nodes involving families of
    zones and families of BC. It also groups ``UserDefined`` BC families by name.

    Parameters
    ----------

        t : PyTree
            assembled tree

            .. note:: tree **t** is modified

        InputMeshes : :py:class:`list` of :py:class:`dict`
            user-provided preprocessing
            instructions as described in :py:func:`prepareMesh4ElsA` doc
    '''
    print('adding families...')
    for meshInfo in InputMeshes:
        base = I.getNodeFromName1(t, meshInfo['baseName'])

        if tagZonesWithBaseName:
            FamilyZoneName = meshInfo['baseName']+'Zones'
            C._tagWithFamily(base, FamilyZoneName)
            C._addFamily2Base(base, FamilyZoneName)



        I._correctPyTree(base, level=7)
        
        if 'BoundaryConditions' in meshInfo:
            for BCinfo in meshInfo['BoundaryConditions']:
                if BCinfo['type'].startswith('FamilySpecified:') and 'familySpecifiedType' in BCinfo:
                    BCName = BCinfo['type'][16:]
                    try:
                        BCType = BCinfo['familySpecifiedType']
                    except:
                        BCType = 'UserDefined'
                    if BCType == 'BCOverlap':
                        # TODO: HACK Check tickets closure
                        ERRMSG=('BCOverlap must be fully defined directly on zones'
                            ' instead of indirectly using FAMILIES.\n'
                            'This option will be acceptable once Cassiopee tickets #7868'
                            ' and #7869 are solved.')
                        raise ValueError(J.FAIL+ERRMSG+J.ENDC)
                    print('Setting BCName %s of BCType %s at base %s'%(BCName, BCType, base[0]))
                    C._addFamily2Base(base, BCName, bndType=BCType)

        all_family_names = [n[0] for n in I.getNodesFromType1(base,'Family_t')]
        for bc in I.getNodesFromType(base, 'BC_t'):
            for fn in I.getNodesFromType1(bc,'FamilyName_t'):
                FamilyName = I.getValue(fn)
                if FamilyName not in all_family_names:
                    all_family_names += [ FamilyName ]
                    f = I.createUniqueChild(base,FamilyName,'Family_t')
                    I.createUniqueChild(f,'FamilyBC','FamilyBC_t',value='UserDefined')

    groupUserDefinedBCFamiliesByName(t)
    adaptFamilyBCNamesToElsA(t)

def splitAndDistribute(t, InputMeshes, mode='auto', cores_per_node=48,
                       minimum_number_of_nodes=1,
                       maximum_allowed_nodes=20,
                       maximum_number_of_points_per_node=1e9,
                       only_consider_full_node_nproc=True,
                       NumberOfProcessors=None, SplitBlocks=True):
    '''
    Distribute a PyTree **t**, with optional splitting.

    Returns a new split and distributed PyTree.

    .. important:: only **InputMeshes** where ``'SplitBlocks':True`` are split.

    Parameters
    ----------

        t : PyTree
            assembled tree

        InputMeshes : :py:class:`list` of :py:class:`dict`
            user-provided preprocessing
            instructions as described in :py:func:`prepareMesh4ElsA` doc

        mode : str
            choose the mode of splitting and distribution among these possibilities:

            * ``'auto'``
                automatically search for the optimum distribution verifying
                the constraints given by **maximum_allowed_nodes** and
                **maximum_number_of_points_per_node**

                .. note:: **NumberOfProcessors** is ignored if **mode** = ``'auto'``, as it
                    is automatically computed by the function. The resulting
                    **NumberOfProcessors** is a multiple of **cores_per_node**

            * ``'imposed'``
                the number of processors is imposed using parameter **NumberOfProcessors**.

                .. note:: **cores_per_node** and **maximum_allowed_nodes**
                    parameters are ignored.

            * ``'pypart'``
                the mesh will be split and distributed on-the-run using PyPart.
                In this case, this function is not effective.

        cores_per_node : int
            number of available CPU cores per node.

            .. note:: only relevant if **mode** = ``'auto'``

        minimum_number_of_nodes : int
            Establishes the minimum number of nodes for the automatic research of
            **NumberOfProcessors**.

            .. note:: only relevant if **mode** = ``'auto'``

        maximum_allowed_nodes : int
            Establishes a boundary of maximum usable nodes. The resulting
            number of processors is the product **cores_per_node** :math:`\\times`
            **maximum_allowed_nodes**

            .. note:: only relevant if **mode** = ``'auto'``

        maximum_number_of_points_per_node : int
            Establishes a boundary of maximum points per node. This value is
            important in order to reduce the required RAM memory for each one
            of the nodes. It raises a :py:obj:`ValueError` if at least one node
            does not satisfy this condition.

        only_consider_full_node_nproc : bool
            if :py:obj:`True` and **mode** = ``'auto'``, then the number of
            processors considered for the optimum search distribution is a
            multiple of **cores_per_node**, in order to employ each node at its
            full capacity. If :py:obj:`False`, then any processor number from
            **cores_per_node** up to **cores_per_node** :math:`\\times` **maximum_allowed_nodes**
            is explored

            .. note:: only relevant if **mode** = ``'auto'``

        NumberOfProcessors : int
            number of processors to be imposed when **mode** = ``'imposed'``

            .. attention:: if **mode** = ``'auto'``, this parameter is ignored

        SplitBlocks : bool
            default value of **SplitBlocks** if it does not exist in the InputMesh
            component.


    Returns
    -------

        t : PyTree
            new distributed *(and possibly split)* tree

    '''

    if mode.lower() == 'pypart':
        print(J.WARN+'mesh shall be split and distributed using PyPart'+J.ENDC)
        return t

    print('splitting and distributing mesh...')
    for meshInfo in InputMeshes:
        if 'SplitBlocks' not in meshInfo:
            meshInfo['SplitBlocks'] = SplitBlocks


    TotalNPts = C.getNPts(t)

    if mode == 'auto':

        startNProc = cores_per_node*minimum_number_of_nodes+1
        if not only_consider_full_node_nproc: startNProc -= cores_per_node 
        endNProc = maximum_allowed_nodes*cores_per_node+1

        if NumberOfProcessors is not None and NumberOfProcessors > 0:
            print(J.WARN+'User requested NumberOfProcessors=%d, switching to mode=="imposed"'%NumberOfProcessors+J.ENDC)
            mode = 'imposed'

        elif minimum_number_of_nodes == maximum_allowed_nodes:
            if only_consider_full_node_nproc:
                NumberOfProcessors = minimum_number_of_nodes*cores_per_node
                print(J.WARN+'User constrained to NumberOfProcessors=%d, switching to mode=="imposed"'%NumberOfProcessors+J.ENDC)
                mode = 'imposed'

        elif minimum_number_of_nodes > maximum_allowed_nodes:
            raise ValueError(J.FAIL+'minimum_number_of_nodes > maximum_allowed_nodes'+J.ENDC)

        elif minimum_number_of_nodes < 1:
            raise ValueError(J.FAIL+'minimum_number_of_nodes must be at least equal to 1'+J.ENDC)


    if mode == 'auto':

        if only_consider_full_node_nproc:
            NProcCandidates = np.array(list(range(startNProc-1,
                                                  (endNProc-1)+cores_per_node,
                                                  cores_per_node)))
        else:
            NProcCandidates = np.array(list(range(startNProc, endNProc)))

        EstimatedAverageNodeLoad = TotalNPts / (NProcCandidates / cores_per_node)
        NProcCandidates = NProcCandidates[EstimatedAverageNodeLoad < maximum_number_of_points_per_node]

        if len(NProcCandidates) < 1:
            raise ValueError(('maximum_number_of_points_per_node is too likely to be exceeded.\n'
                              'Try increasing maximum_allowed_nodes and/or maximum_number_of_points_per_node'))

        Title1= ' number of  | number of  | max pts at | max pts at | percent of | average pts|'
        Title = ' processors | zones      | any proc   | any node   | imbalance  | per proc   |'
        
        Ncol = len(Title)
        print('-'*Ncol)
        print(Title1)
        print(Title)
        print('-'*Ncol)
        Ndigs = len(Title.split('|')[0]) + 1
        ColFmt = r'{:^'+str(Ndigs)+'g}'

        AllNZones = []
        AllVarMax = []
        AllAvgPts = []
        AllMaxPtsPerNode = []
        AllMaxPtsPerProc = []
        for i, NumberOfProcessors in enumerate(NProcCandidates):
            _, NZones, varMax, meanPtsPerProc, MaxPtsPerNode, MaxPtsPerProc = _splitAndDistributeUsingNProcs(t,
                InputMeshes, NumberOfProcessors, cores_per_node, maximum_number_of_points_per_node,
                raise_error=False)
            AllNZones.append( NZones )
            AllVarMax.append( varMax )
            AllAvgPts.append( meanPtsPerProc )
            AllMaxPtsPerNode.append( MaxPtsPerNode )
            AllMaxPtsPerProc.append( MaxPtsPerProc )

            if AllNZones[i] == 0:
                start = J.FAIL
                end = '  <== EXCEEDED nb. pts. per node with %d'%AllMaxPtsPerNode[i]+J.ENDC
            else:
                start = end = ''
            Line = start + ColFmt.format(NumberOfProcessors)
            if AllNZones[i] == 0:
                Line += end
            else:
                Line += ColFmt.format(AllNZones[i])
                Line += ColFmt.format(AllMaxPtsPerProc[i])
                Line += ColFmt.format(AllMaxPtsPerNode[i])
                Line += ColFmt.format(AllVarMax[i] * 100)
                Line += ColFmt.format(AllAvgPts[i])
                Line += end

            print(Line)
            if cores_per_node>1 and (NumberOfProcessors%cores_per_node==0):
                print('-'*Ncol)
            

        BestOption = np.argmin( AllMaxPtsPerProc )

        for i, NumberOfProcessors in enumerate(NProcCandidates):
            if i == BestOption and AllNZones[i] > 0:
                Line = start = J.GREEN + ColFmt.format(NumberOfProcessors)
                end = '  <== BEST'+J.ENDC
                Line += ColFmt.format(AllNZones[i])
                Line += ColFmt.format(AllMaxPtsPerProc[i])
                Line += ColFmt.format(AllMaxPtsPerNode[i])
                Line += ColFmt.format(AllVarMax[i] * 100)
                Line += ColFmt.format(AllAvgPts[i])
                Line += end
                print(Line)
                break
        tRef = _splitAndDistributeUsingNProcs(t, InputMeshes, NProcCandidates[BestOption],
                cores_per_node, maximum_number_of_points_per_node, raise_error=True)[0]

        I._correctPyTree(tRef,level=3)
        tRef = connectMesh(tRef, InputMeshes)

    elif mode == 'imposed':

        tRef = _splitAndDistributeUsingNProcs(t, InputMeshes, NumberOfProcessors, cores_per_node,
                                 maximum_number_of_points_per_node, raise_error=True)[0]

        I._correctPyTree(tRef,level=3)
        tRef = connectMesh(tRef, InputMeshes)

    showStatisticsAndCheckDistribution(tRef, CoresPerNode=cores_per_node)

    return tRef

def _splitAndDistributeUsingNProcs(t, InputMeshes, NumberOfProcessors, cores_per_node,
                         maximum_number_of_points_per_node, raise_error=False):

    if DEBUG: print('attempting distribution for NumberOfProcessors= %d ...'%NumberOfProcessors)

    tRef = I.copyRef(t)
    TotalNPts = C.getNPts(tRef)
    ProcPointsLoad = TotalNPts / NumberOfProcessors
    basesToSplit, basesBackground = getBasesBasedOnSplitPolicy(tRef, InputMeshes)

    remainingNProcs = NumberOfProcessors * 1
    baseName2NProc = dict()

    for base in basesBackground:
        baseNPts = C.getNPts(base)
        baseNProc = int( baseNPts / ProcPointsLoad )
        baseName2NProc[base[0]] = baseNProc
        remainingNProcs -= baseNProc


    if basesToSplit:

        tToSplit = I.merge([C.newPyTree([b[0],I.getZones(b)]) for b in basesToSplit])

        removeMatchAndNearMatch(tToSplit)
        tSplit = T.splitSize(tToSplit, 0, type=0, R=remainingNProcs,
                             minPtsPerDir=5)
        NbOfZonesAfterSplit = len(I.getZones(tSplit))
        HasDegeneratedZones = False
        if NbOfZonesAfterSplit < remainingNProcs:
            MSG = 'WARNING: nb of zones after split (%d) is less than expected procs (%d)'%(NbOfZonesAfterSplit, remainingNProcs)
            MSG += '\nattempting T.splitNParts()...'
            print(J.WARN+MSG+J.ENDC)
            tSplit = T.splitNParts(tToSplit, remainingNProcs)
            splitZones = I.getZones(tSplit)
            if len(splitZones) < remainingNProcs:
                MSG = ('could not split sufficiently. Try manually splitting '
                       'mesh and set SplitBlocks=False')
                raise ValueError(J.FAIL+MSG+J.ENDC)
            for zone in splitZones:
                zoneDims = I.getZoneDim(zone)
                if zoneDims[0] == 'Structured':
                    dims = zoneDims[1:-1]
                    for NPts, dir in zip(dims, ['i', 'j', 'k']):
                        if NPts < 5:
                            if NPts < 3:
                                MSG = J.FAIL+'ERROR: zone %s has %d pts in %s direction'%(zone[0],NPts,dir)+J.ENDC
                                HasDegeneratedZones = True
                            else:
                                MSG = J.WARN+'WARNING: zone %s has %d pts in %s direction'%(zone[0],NPts,dir)+J.ENDC
                            print(MSG)

        if HasDegeneratedZones:
            raise ValueError(J.FAIL+'grid has degenerated zones. See previous print error messages'+J.ENDC)

        for splitbase in I.getBases(tSplit):
            basename = splitbase[0]
            base = I.getNodeFromName2(tRef, basename)
            if not base: raise ValueError('unexpected !')
            I._rmNodesByType(base, 'Zone_t')
            base[2].extend( I.getZones(splitbase) )

        tRef = I.merge([tRef,tSplit])

        NZones = len( I.getZones( tRef ) )
        if NumberOfProcessors > NZones:
            if raise_error:
                MSG = ('Requested number of procs ({}) is higher than the final number of zones ({}).\n'
                       'You may try the following:\n'
                       ' - Reduce the number of procs\n'
                       ' - increase the number of grid points').format( NumberOfProcessors, NZones)
                raise ValueError(J.FAIL+MSG+J.ENDC)
            return tRef, 0, np.inf, np.inf, np.inf, np.inf

    NZones = len( I.getZones( tRef ) )
    if NumberOfProcessors > NZones:
        if raise_error:
            MSG = ('Requested number of procs ({}) is higher than the final number of zones ({}).\n'
                   'You may try the following:\n'
                   ' - set SplitBlocks=True to more grid components\n'
                   ' - Reduce the number of procs\n'
                   ' - increase the number of grid points').format( NumberOfProcessors, NZones)
            raise ValueError(J.FAIL+MSG+J.ENDC)
        else:
            return tRef, 0, np.inf, np.inf, np.inf, np.inf

    # NOTE see Cassiopee BUG #8244 -> need algorithm='fast'
    silence = J.OutputGrabber()
    with silence:
        tRef, stats = D2.distribute(tRef, NumberOfProcessors, algorithm='fast', useCom='all')

    behavior = 'raise' if raise_error else 'silent'

    if hasAnyEmptyProc(tRef, NumberOfProcessors, behavior=behavior):
        return tRef, 0, np.inf, np.inf, np.inf, np.inf

    HighestLoad = getNbOfPointsOfHighestLoadedNode(tRef, cores_per_node)
    HighestLoadProc = getNbOfPointsOfHighestLoadedProc(tRef)

    if HighestLoad > maximum_number_of_points_per_node:
        if raise_error:
            raise ValueError('exceeded maximum_number_of_points_per_node (%d>%d)'%(HighestLoad,
                                                maximum_number_of_points_per_node))
        return tRef, 0, np.inf, np.inf, np.inf, np.inf


    return tRef, NZones, stats['varMax'], stats['meanPtsPerProc'], HighestLoad, HighestLoadProc

def getNbOfPointsOfHighestLoadedNode(t, cores_per_node):
    NPtsPerNode = {}
    for zone in I.getZones(t):
        Proc, = getProc(zone)
        Node = Proc//cores_per_node
        try: NPtsPerNode[Node] += C.getNPts(zone)
        except KeyError: NPtsPerNode[Node] = C.getNPts(zone)

    nodes = list(NPtsPerNode)
    NodesLoad = np.zeros(max(nodes)+1, dtype=int)
    for node in NPtsPerNode: NodesLoad[node] = NPtsPerNode[node]
    HighestLoad = np.max(NodesLoad)

    return HighestLoad

def getNbOfPointsOfHighestLoadedProc(t):
    NPtsPerProc = {}
    for zone in I.getZones(t):
        Proc, = getProc(zone)
        try: NPtsPerProc[Proc] += C.getNPts(zone)
        except KeyError: NPtsPerProc[Proc] = C.getNPts(zone)

    procs = list(NPtsPerProc)
    ProcsLoad = np.zeros(max(procs)+1, dtype=int)
    for proc in NPtsPerProc: ProcsLoad[proc] = NPtsPerProc[proc]
    HighestLoad = np.max(ProcsLoad)

    return HighestLoad

def _isMaximumNbOfPtsPerNodeExceeded(t, maximum_number_of_points_per_node, cores_per_node):
    NPtsPerNode = {}
    for zone in I.getZones(t):
        Proc, = getProc(zone)
        Node = (Proc//cores_per_node)+1
        try: NPtsPerNode[Node] += C.getNPts(zone)
        except KeyError: NPtsPerNode[Node] = C.getNPts(zone)

    for node in NPtsPerNode:
        if NPtsPerNode[node] > maximum_number_of_points_per_node: return True
    return False

def hasAnyEmptyProc(t, NumberOfProcessors, behavior='raise', debug_filename=''):
    '''
    Check the proc distribution of a tree and raise an error (or print message)
    if there are any empty proc.

    Parameters
    ----------

        t : PyTree
            tree with node ``.Solver#Param/proc``

        NumberOfProcessors : int
            initially requested number of processors for distribution

        behavior : str
            if empty processors are found, this parameter specifies the behavior
            of the function:

            * ``'raise'``
                Raises a :py:obj:`ValueError`, stopping execution

            * ``'print'``
                Prints a message onto the termina, execution continues

            * ``'silent'``
                No error, no print; execution continues

        debug_filename : str
            if given, then writes the input tree **t** before the designed
            exceptions are raised or in case some proc is empty.

    Returns
    -------

        hasAnyEmptyProc : bool
            :py:obj:`True` if any processor has no attributed zones
    '''
    Proc2Zones = dict()
    UnaffectedProcs = list(range(NumberOfProcessors))

    for z in I.getZones(t):
        proc = int(D2.getProc(z))

        if proc < 0:
            if debug_filename: C.convertPyTree2File(t, debug_filename)
            raise ValueError('zone %s is not distributed'%z[0])

        if proc in Proc2Zones:
            Proc2Zones[proc].append( I.getName(z) )
        else:
            Proc2Zones[proc] = [ I.getName(z) ]

        try: UnaffectedProcs.remove( proc )
        except ValueError: pass


    if UnaffectedProcs:
        hasAnyEmptyProc = True
        if debug_filename: C.convertPyTree2File(t, debug_filename)
        MSG = J.FAIL+'THERE ARE UNAFFECTED PROCS IN DISTRIBUTION!!\n'
        MSG+= 'Empty procs: %s'%str(UnaffectedProcs)+J.ENDC
        if behavior == 'raise':
            raise ValueError(MSG)
        elif behavior == 'print':
            print(MSG)
        elif behavior != 'silent':
            raise ValueError('behavior %s not recognized'%behavior)
    else:
        hasAnyEmptyProc = False

    return hasAnyEmptyProc

def getBasesBasedOnSplitPolicy(t,InputMeshes):
    '''
    Returns two different lists, one with bases to split and other with bases
    not to split. The filter is done depending on the boolean value
    of ``SplitBlocks`` key provided by user for each component of **InputMeshes**.

    Parameters
    ----------

        t : PyTree
            assembled tree

        InputMeshes : :py:class:`list` of :py:class:`dict`
            user-provided preprocessing
            instructions as described in :py:func:`prepareMesh4ElsA` doc

    Returns
    -------

        basesToSplit : :py:class`list` of base
            bases that are to be split.

        basesNotToSplit : :py:class`list` of base
            bases that are NOT to be split.
    '''
    basesToSplit = []
    basesNotToSplit = []
    for meshInfo in InputMeshes:
        base = I.getNodeFromName1(t,meshInfo['baseName'])
        if meshInfo['SplitBlocks']:
            basesToSplit += [base]
        else:
            basesNotToSplit += [base]

    return basesToSplit, basesNotToSplit

def showStatisticsAndCheckDistribution(tNew, CoresPerNode=28):
    '''
    Print statistics on the distribution of a PyTree and also indicates the load
    attributed to each computational node.

    Parameters
    ----------

        tNew : PyTree
            tree where distribution was done.

        CoresPerNode : int
            number of processors per node.

    '''
    ProcDistributed = getProc(tNew)
    ResultingNProc = max(ProcDistributed)+1
    Distribution = D2.getProcDict(tNew)

    NPtsPerProc = {}
    for zone in I.getZones(tNew):
        Proc, = getProc(zone)
        try: NPtsPerProc[Proc] += C.getNPts(zone)
        except KeyError: NPtsPerProc[Proc] = C.getNPts(zone)

    NPtsPerNode = {}
    for zone in I.getZones(tNew):
        Proc, = getProc(zone)
        Node = (Proc//CoresPerNode)+1
        try: NPtsPerNode[Node] += C.getNPts(zone)
        except KeyError: NPtsPerNode[Node] = C.getNPts(zone)


    ListOfProcs = list(NPtsPerProc.keys())
    ListOfNPts = [NPtsPerProc[p] for p in ListOfProcs]
    ArgNPtsMin = np.argmin(ListOfNPts)
    ArgNPtsMax = np.argmax(ListOfNPts)

    MSG = '\nTotal number of processors is %d\n'%ResultingNProc
    MSG += 'Total number of zones is %d\n'%len(I.getZones(tNew))
    MSG += 'Proc %d has lowest nb. of points with %d\n'%(ListOfProcs[ArgNPtsMin],
                                                    ListOfNPts[ArgNPtsMin])
    MSG += 'Proc %d has highest nb. of points with %d\n'%(ListOfProcs[ArgNPtsMax],
                                                    ListOfNPts[ArgNPtsMax])
    print(MSG)

    for node in NPtsPerNode:
        print('Node %d has %d points'%(node,NPtsPerNode[node]))

    print(J.CYAN+'TOTAL NUMBER OF POINTS: '+'{:,}'.format(C.getNPts(tNew)).replace(',',' ')+'\n'+J.ENDC)

    for p in range(ResultingNProc):
        if p not in ProcDistributed:
            raise ValueError('Bad proc distribution! rank %d is empty'%p)

def addOversetData(t, InputMeshes, depth=2, optimizeOverlap=False,
                   prioritiesIfOptimize=[], double_wall=0,
                   saveMaskBodiesTree=True,
                   overset_in_CGNS=False, # see elsA #10545
                   CHECK_OVERSET=True,
                   ):
    '''
    This function performs all required preprocessing operations for a
    overlapping configuration. This includes masks production, setting
    interpolating regions and computing interpolating coefficients. This may 
    also include unsteady overset masking operations. 

    Global overset options are provided by the optional arguments of the
    function.


    Parameters
    ----------

        t : PyTree
            assembled tree

        InputMeshes : :py:class:`list` of :py:class:`dict`
            user-provided preprocessing instructions as described in
            :py:func:`prepareMesh4ElsA` .

            Component-specific instructions for overlap settings are provided
            through **InputMeshes** component by means of keyword ``OversetOptions``, which
            accepts a Python dictionary with several allowed pairs of keywords and their
            associated values:

            * ``'BlankingMethod'`` : :py:class:`str`
                currently, two blanking methods are allowed:

                * ``'blankCellsTri'``
                    makes use of Connector's function of same name

                * ``'blankCells'``
                    makes use of Connector's function of same name

            * ``'BlankingMethodOptions'`` : :py:class:`dict`
                literally, all options to provide to Connector's function
                specified by ``'BlankingMethod'`` key.

            * ``'NCellsOffset'`` : :py:class:`int`
                if provided, then masks constructed from *BCOverlap*
                boundaries are built by producing an offset towards the interior of
                the grid following the number of cells provided by this value. This
                option is well suited if cell size around BCOverlaps are similar and
                they have also similar size with respect to background grids (which
                should be the case for proper quality of the interpolations).

                .. important:: ``'NCellsOffset'`` and ``'OffsetDistanceOfOverlapMask'``
                    **MUST NOT** be defined **simultaneously** (for a same **InputMeshes** item)
                    as they use very different masking techniques !

            * ``'OffsetDistanceOfOverlapMask'`` : :py:class:`float`
                if set, then masks constructed
                from *BCOverlap* boundaries are built by producing a  offset towards the
                interior of the grid following a normal distance provided by this value.
                This option is better suited than ``'NCellsOffset'`` if cell sizes are
                irregular. However, this strategy is more costly and less robust.
                It is recommended to try ``'NCellsOffset'`` in priority.

                .. important:: ``'NCellsOffset'`` and ``'OffsetDistanceOfOverlapMask'``
                    **MUST NOT** be defined **simultaneously** (for a same **InputMeshes** item)
                    as they use very different masking techniques !

            * ``'CreateMaskFromWall'`` : :py:class:`bool`
                If :py:obj:`False`, then walls of this component
                will not be used for masking. This shall only be used if user knows
                a priori that this component's walls are not masking any grid. If
                this is the case, then user can put this value to :py:obj:`False` in order to
                slightly accelerate the preprocess time.

                .. note:: by default, the value of this key is :py:obj:`True`.

            * ``'OnlyMaskedByWalls'`` : :py:class:`bool`
                if :py:obj:`True`, then this overset component
                is strongly protected against masking. Only other component's walls
                are allowed to mask this component.

                .. hint:: you should use ```OnlyMaskedByWalls=True`` **except**
                    for background grids.

            * ``'ForbiddenOverlapMaskingThisBase'`` : :py:class:`list` of :py:class:`str`
                This is a list of
                base names (names of **InputMeshes** components) whose masking bodies
                built from their *BCOverlap* are not allowed to mask this component.
                This is used to protect this component from being masked by other
                component's masks (only affects masks constructed from offset of
                overlap bodies, this does not include masks constructed from walls).

        depth : int
            depth of the interpolation region.

        prioritiesIfOptimize : list
            literally, the
            priorities argument passed to :py:func:`Connector.PyTree.optimizeOverlap`.

        double_wall : bool
            if :py:obj:`True`, double walls exist

        saveMaskBodiesTree : bool
            if :py:obj:`True`, then saves the file ``masks.cgns``,
            allowing the user to analyze if masks have been properly generated.

        overset_in_CGNS : bool
            if :py:obj:`True`, then include all interpolation data in CGNS using
            ``ID_*`` nodes.

            .. danger::
                beware of elsA bug `10545 <https://elsa.onera.fr/issues/10545>`_
        
        CHECK_OVERSET : bool
            if :py:obj:`True`, then make an extrapolated-orphan cell diagnosis
            when making unsteady motion overset preprocess.

    Returns
    -------

        t : PyTree
            new pytree including ``cellN`` values at ``FlowSolution#Centers``
            and elsA's ``ID*`` nodes including interpolation coefficients information.

    '''

    from . import UnsteadyOverset as UO


    if not hasAnyOversetData(InputMeshes): return t  

    try: os.makedirs(DIRECTORY_OVERSET)
    except: pass

    print('building masking bodies...')
    baseName2BodiesDict = getMaskingBodiesAsDict(t, InputMeshes)

    bodies = []
    for meshInfo in InputMeshes:
        bodies.extend(baseName2BodiesDict[meshInfo['baseName']])

    if saveMaskBodiesTree:
        treeLikeList = []
        for bn in baseName2BodiesDict:
            flat_bodies = _flattenBodies(baseName2BodiesDict[bn])
            treeLikeList.extend([bn, flat_bodies])
        tMask = I.copyRef(C.newPyTree(treeLikeList))
        I._correctPyTree(tMask,level=3)
        C.convertPyTree2File(tMask, os.path.join(DIRECTORY_OVERSET,
                                                'CHECK_ME_mask.cgns'))


    BlankingMatrix = getBlankingMatrix(bodies, InputMeshes)

    # TODO -> RB: applyBCOverlaps after cellN2OversetHoles so that 
    # output files assign cellN=0 on masked overlaps instead of cellN=2
    t = X.applyBCOverlaps(t, depth=depth)

    print('Static blanking...')
    t_blank = staticBlanking(t, bodies, BlankingMatrix, InputMeshes)
    if hasAnyOversetMotion(InputMeshes):
        StaticBlankingMatrix  = getBlankingMatrix(bodies, InputMeshes,
                                                    StaticOnly=True)
        t = staticBlanking(t, bodies, StaticBlankingMatrix, InputMeshes)
    else:
        StaticBlankingMatrix = BlankingMatrix
        t = t_blank
    print('... static blanking done.')

    print('setting hole interpolated points...')
    t = X.setHoleInterpolatedPoints(t, depth=depth)

    if prioritiesIfOptimize:
        print('Optimizing overlap...')
        t = X.optimizeOverlap(t, double_wall=double_wall,
                              priorities=prioritiesIfOptimize)
        print('... optimization done.')

    print('maximizing blanked cells...')
    t = X.maximizeBlankedCells(t, depth=depth)


    if overset_in_CGNS:
        prefixFile = ''
    else:
        prefixFile = os.path.join(DIRECTORY_OVERSET,'overset')

    print('cellN2OversetHoles and applyBCOverlaps...')
    t = X.cellN2OversetHoles(t)
    t = X.applyBCOverlaps(t, depth=depth) # TODO ->  see previous RB note
    print('... cellN2OversetHoles and applyBCOverlaps done.')


    if hasAnyOversetMotion(InputMeshes):
        if CHECK_OVERSET:
            print('Checking overset assembly...')
            t_blank = X.setHoleInterpolatedPoints(t_blank, depth=depth)
            if prioritiesIfOptimize:
                t_blank = X.optimizeOverlap(t_blank, double_wall=double_wall,
                                    priorities=prioritiesIfOptimize)
            t_blank = X.maximizeBlankedCells(t_blank, depth=depth)          
            t_blank = X.cellN2OversetHoles(t_blank)
            t_blank = X.applyBCOverlaps(t_blank, depth=depth) 
            t_blank = muted_setInterpolations(t_blank, loc='cell', sameBase=0,
                    double_wall=double_wall, storage='inverse', solver='elsA',
                    check=True, nGhostCells=2, prefixFile='')
            print('Checking overset assembly... done')

        print('Writing static masking files...')
        EP.buildMaskFiles(t, fileDir=DIRECTORY_OVERSET, prefixBase=True)
        print('Writing static masking files... done')

    else:
        print('Computing interpolation coefficients...')
        t = muted_setInterpolations(t, loc='cell', sameBase=0,
                double_wall=double_wall, storage='inverse', solver='elsA',
                check=True, nGhostCells=2, prefixFile=prefixFile)
        print('... interpolation coefficients built.')

    if CHECK_OVERSET:
        if not hasAnyOversetMotion(InputMeshes): t_blank = t
        TreesDiagnosis = []
        anyOrphan = False
        for diagnosisType in ['orphan', 'extrapolated']:
            tAux = X.chimeraInfo(t_blank, type=diagnosisType)
            for base in I.getBases(tAux):
                CriticalPoints = X.extractChimeraInfo(base, type=diagnosisType)
                if CriticalPoints:
                    nCells = C.getNCells(CriticalPoints)
                    TreesDiagnosis += [ C.newPyTree([base[0]+'_'+diagnosisType,
                                                    CriticalPoints]) ]
                    msg = 'base %s has %d %s cells'%(base[0],nCells,diagnosisType)
                    if diagnosisType == 'orphan':
                        anyOrphan = True
                        print(J.FAIL+'DANGER: %s'%msg+J.ENDC)
                    elif diagnosisType == 'extrapolated':
                        print(J.WARN+'WARNING: %s'%msg+J.ENDC)

        if TreesDiagnosis:
            TreeDiagnosis = I.merge(TreesDiagnosis)
            diagnosis_file = os.path.join(DIRECTORY_OVERSET,
                                        'CHECK_ME_OverlapCriticalCells.cgns')
            C.convertPyTree2File(TreeDiagnosis, diagnosis_file)
            start = J.FAIL if anyOrphan else J.WARN
            print(start+'Please check '+J.BOLD+diagnosis_file+J.ENDC)
        else:
            print(J.GREEN+'Congratulations! no extrapolated or orphan cells!'+J.ENDC)


    print('adding unsteady overset data...')
    DynamicBlankingMatrix = BlankingMatrix - StaticBlankingMatrix
    UO.addMaskData(t, InputMeshes, bodies, DynamicBlankingMatrix)
    BodyNames = [getBodyName( body ) for body in bodies]
    UO.setMaskedZonesOfMasks(t, InputMeshes, DynamicBlankingMatrix, BodyNames)
    UO.setMaskParameters(t, InputMeshes)
    I._rmNodesByName(t,'OversetHoles')
    # UO.removeOversetHolesOfUnsteadyMaskedGrids(t)
    print('adding unsteady overset data... done')

    if not overset_in_CGNS: I._rmNodesByName(t,'ID_*')

    return t


def staticBlanking(t, bodies, BlankingMatrix, InputMeshes):
    # see ticket #7882
    for ibody, body in enumerate(bodies):
        BlankingVector = np.atleast_2d(BlankingMatrix[:,ibody]).T
        BaseNameOfBody = getBodyParentBaseName(getBodyName(body))
        meshInfo = getMeshInfoFromBaseName(BaseNameOfBody, InputMeshes)
        try: BlankingMethod = meshInfo['OversetOptions']['BlankingMethod']
        except KeyError: BlankingMethod = 'blankCellsTri'

        try: UserSpecifiedBlankingMethodOptions = meshInfo['OversetOptions']['BlankingMethodOptions']
        except KeyError: UserSpecifiedBlankingMethodOptions = {}
        BlankingMethodOptions = dict(blankingType='center_in')
        BlankingMethodOptions.update(UserSpecifiedBlankingMethodOptions)

        bodyInterface = [[body]] if isinstance(body[0],str) else [body]

        if BlankingMethod == 'blankCellsTri':
            t = X.blankCellsTri(t, bodyInterface, BlankingVector,
                                    **BlankingMethodOptions)

        elif BlankingMethod == 'blankCells':
            t = X.blankCells(t, bodyInterface, BlankingVector,
                                **BlankingMethodOptions)

        else:
            raise ValueError('BlankingMethod "{}" not recognized'.format(BlankingMethod))
    return t 

@J.mute_stdout
def muted_setInterpolations(*args, **kwargs):
    return X.setInterpolations(*args, **kwargs)


def hasAnyOversetMotion(InputMeshes):
    '''
    Determine if at least one item in **InputMeshes** has a motion overset kind
    of assembly.

    Parameters
    ----------

        InputMeshes : :py:class:`list` of :py:class:`dict`
            as described by :py:func:`prepareMesh4ElsA`

    Returns
    -------

        bool : bool
            :py:obj:`True` if has moving overset assembly. :py:obj:`False` otherwise.
    '''
    if hasAnyOversetData(InputMeshes):
        for meshInfo in InputMeshes:
            if 'Motion' in meshInfo:
                return True
    return False


def getBlankingMatrix(bodies, InputMeshes, StaticOnly=False):
    '''
    .. attention:: this is a **private-level** function. Users shall employ
        user-level function :py:func:`addOversetData`.

    This function constructs the blanking matrix :math:`BM_{ij}`, such that
    :math:`BM_{ij}=1` means that :math:`i`-th basis is blanked by :math:`j`-th
    body. If :math:`BM_{ij}=0`, then :math:`i`-th basis is **not** blanked by
    :math:`j`-th body.

    Parameters
    ----------

        bodies : :py:class:`list` of :py:class:`zone`
            list of watertight surfaces used for blanking (masks)

            .. attention:: unstructured *TRI* surfaces must be oriented inwards

        InputMeshes : :py:class:`list` of :py:class:`dict`
            user-provided preprocessing instructions as described in
            :py:func:`prepareMesh4ElsA` .

    Returns
    -------

        BlankingMatrix : numpy.ndarray
            2D matrix of shape :math:`N_B \\times N_m`  where :math:`N_B` is the
            total number of bases and :math:`N_m` is the total number of
            masking surfaces.
    '''


    # BM(i,j)=1 means that ith basis is blanked by jth body
    Nbases  = len( InputMeshes )
    Nbodies = len( bodies )
    

    BaseNames = [meshInfo['baseName'] for meshInfo in InputMeshes]
    BodyNames = [getBodyName( body ) for body in bodies]

    # Initialization: all bodies mask all bases
    BlankingMatrix = np.ones((Nbases, Nbodies))

    # do not allow bodies issued of a given base to mask its own parent base
    for i, j in product(range(Nbases), range(Nbodies)):
        BaseName = BaseNames[i]
        BodyName = BodyNames[j]
        BodyParentBaseName = getBodyParentBaseName(BodyName)
        if BaseName == BodyParentBaseName:
            BlankingMatrix[i, j] = 0

    # user-provided masking protections
    for i, meshInfo in enumerate(InputMeshes):
        # if 'Motion' in meshInfo: BlankingMatrix[i, :] = 0 # will need unsteady mask
        try:
            Forbidden = meshInfo['OversetOptions']['ForbiddenOverlapMaskingThisBase']
        except KeyError:
            continue

        for j, BodyName in enumerate(BodyNames):
            BodyParentBaseName = getBodyParentBaseName(BodyName)
            if BodyName.startswith('overlap') and BodyParentBaseName in Forbidden:
                BlankingMatrix[i, j] = 0
        

    # masking protection using key "OnlyMaskedByWalls"
    for i, meshInfo in enumerate(InputMeshes):
        try: OnlyMaskedByWalls = meshInfo['OversetOptions']['OnlyMaskedByWalls']
        except KeyError: continue
        if OnlyMaskedByWalls:
            for j, BodyName in enumerate(BodyNames):
                if not BodyName.startswith('wall'):
                    BlankingMatrix[i, j] = 0

    if StaticOnly:
        # TODO optimize by fixed masking of components with same rigid motion
        for j, BodyName in enumerate(BodyNames):
            BodyParentBaseName = getBodyParentBaseName(BodyName)
            bodyInfo = getMeshInfoFromBaseName(BodyParentBaseName,InputMeshes)
            MovingBody = True if 'Motion' in bodyInfo else False
            if MovingBody or BodyName.startswith('overlap'):
                BlankingMatrix[:,j] = 0

        for i, meshInfo in enumerate(InputMeshes):
            MovingBase = True if 'Motion' in meshInfo else False
            if MovingBase: BlankingMatrix[i,:] = 0

    print('BaseNames (rows) = %s'%str(BaseNames))
    print('BodyNames (columns) = %s'%str(BodyNames))
    msg = 'BlankingMatrix:' if not StaticOnly else 'static BlankingMatrix:'
    print(msg)
    print(np.array(BlankingMatrix,dtype=int))

    return BlankingMatrix

def getMaskingBodiesAsDict(t, InputMeshes):
    '''
    This function generates a python dictionary of the following structure:

    >>> baseName2BodiesDict['<basename>'] = [list of zones]

    The list of zones correspond to the masks produced at the base named
    **<basename>**.

    Parameters
    ----------

        t : PyTree
            assembled PyTree as generated by :py:func:`getMeshesAssembled`

        InputMeshes : :py:class:`list` of :py:class:`dict`
            user-provided preprocessing
            instructions as described in :py:func:`prepareMesh4ElsA`

    Returns
    -------

        baseName2BodiesDict : :py:class:`dict`
            each value is a list of zones (the masking bodies of the base)
    '''
    baseName2BodiesDict = {}
    for base in I.getBases(t):
        basename = base[0]
        baseName2BodiesDict[basename] = []

        # Currently allowed masks are built using BCWall (hard mask) and
        # BCOverlap (soft mask)

        meshInfo = getMeshInfoFromBaseName(basename, InputMeshes)
        try:
            CreateMaskFromWall = meshInfo['OversetOptions']['CreateMaskFromWall']
        except KeyError:
            CreateMaskFromWall = True

        if CreateMaskFromWall:
            wallTag = 'wall-'+basename
            print('building mask surface %s'%wallTag)
            walls = getWalls(base, SuffixTag=wallTag)
            if walls: baseName2BodiesDict[basename].extend( walls )
            else: print('no wall found at %s'%basename)


        try:
            CreateMaskFromOverlap = meshInfo['OversetOptions']['CreateMaskFromOverlap']
        except KeyError:
            CreateMaskFromOverlap = True

        if not CreateMaskFromOverlap: continue

        if 'OversetOptions' not in meshInfo:
            print(('No OversetOptions dictionary defined for base {}.\n'
            'Will not search overlap masks in this base.').format(basename))
            continue


        NCellsOffset = None
        try: NCellsOffset = meshInfo['OversetOptions']['NCellsOffset']
        except KeyError: pass

        OffsetDistance = None
        try: OffsetDistance = meshInfo['OversetOptions']['OffsetDistanceOfOverlapMask']
        except KeyError: pass

        MatchTolerance = 1e-8
        try:
            for ConDict in meshInfo['Connection']:
                if ConDict['type'] == 'Match':
                    MatchTolerance = ConDict['tolerance']
                    break
        except KeyError: pass


        overlapTag = 'overlap-'+basename

        if NCellsOffset is not None:
            print('building mask surface %s by cells offset'%overlapTag)
            overlap = getOverlapMaskByCellsOffset(base, SuffixTag=overlapTag,
                                               NCellsOffset=NCellsOffset)

        elif OffsetDistance is not None:
            print('building mask surface %s by negative extrusion'%overlapTag)
            niter = None
            try:
                niter=meshInfo['OversetOptions']['MaskOffsetNormalsSmoothIterations']
            except KeyError: pass
            overlap = getOverlapMaskByExtrusion(base, SuffixTag=overlapTag,
                                           OffsetDistanceOfOverlapMask=OffsetDistance,
                                           MatchTolerance=MatchTolerance,
                                           MaskOffsetNormalsSmoothIterations=niter)
        if overlap: baseName2BodiesDict[basename].append( overlap )
        else: print('no overlap found at %s'%basename)
    return baseName2BodiesDict

def getWalls(t, SuffixTag=None):
    '''
    Get closed watertight surfaces from walls (defined using ``BCWall*``)

    Parameters
    ----------

        t : PyTree
            assembled tree

        SuffixTag : str
            if provided, include a tag on newly created zone names

    Returns
    -------

        walls - list
            closed watertight surfaces (TRI)
    '''
    # note: this works also for BCWall* defined by families
    walls = C.extractBCOfType(t, 'BCWall*', reorder=True)
    if not walls: return []
    
    tR = I.copyRef(t)
    _ungroupBCsByBCType(tR, forced_starting='BCWall')
    I._groupBCByBCType(tR, btype='BCWall', name=SuffixTag)
    walls = ESP.extractSurfacesByOffsetCellsFromBCFamilyName(tR,
                                    BCFamilyName=SuffixTag, NCellsOffset=0)


    # walls = C.extractBCOfType(tR, 'BCWall*', reorder=True)
    # if not walls: return []
    ## walls = buildWatertightBodiesFromSurfaces(walls,
    ##                                          imposeNormalsOrientation='inwards',
    ##                                          SuffixTag=SuffixTag)
    
    # EXPERIMENTAL : directly use composite walls
    if SuffixTag:
        for w in I.getZones(walls): w[0] = SuffixTag
    walls = [walls]

    return walls

def getOverlapMaskByExtrusion(t, SuffixTag=None, OffsetDistanceOfOverlapMask=0.,
                              MatchTolerance=1e-8,
                              MaskOffsetNormalsSmoothIterations=None):
    '''
    Build the overlap mask by negative extrusion from *BCOverlap* boundaries.

    Parameters
    ----------

        t :  PyTree
            the assembled PyTree containing boundary conditions.

        SuffixTag : str
            The suffix to attribute to the new mask name.

        OffsetDistanceOfOverlapMask : float
            distance of negative extrusion to apply from *BCOverlap*

        MatchTolerance : float
            small value used for merging auxiliar surface patches.

        MaskOffsetNormalsSmoothIterations : int
            number of iterations of normal smoothing employed for computing
            the extrusion direction.

    Returns
    -------

        mask : zone
            unstructured zone consisting in a watertight closed surface
            that can be employed as a mask.
    '''

    # get Overlap masks and merge them without making them watertight
    mask = C.extractBCOfType(t, 'BCOverlap', reorder=True)
    if not mask: return
    mask = C.convertArray2Tetra(mask)
    mask = T.join(mask)

    mask = G.mmgs(mask, ridgeAngle=45., hmin=OffsetDistanceOfOverlapMask/8.,
                        hmax=OffsetDistanceOfOverlapMask/2., hausd=0.01,
                        grow=1.1, optim=0)

    if GSD.isClosed(mask, tol=MatchTolerance):

        print(('Applying offset={} to closed mask {}').format(
                            OffsetDistanceOfOverlapMask,SuffixTag))

        if SuffixTag: mask[0] = SuffixTag

        mask = applyOffset2ClosedMask(mask, OffsetDistanceOfOverlapMask,
                                      niter=MaskOffsetNormalsSmoothIterations)


    else:
        # get support surface where open mask will be constrained when
        # applying offset distance. Note that support do not include BCOverlap
        # nor Match nor NearMatch

        print(('Applying offset={} to open mask {}').format(
                            OffsetDistanceOfOverlapMask,SuffixTag))

        SupportSurface = C.extractBCOfType(t, 'BC*', reorder=True)
        SupportSurface = C.convertArray2Tetra(SupportSurface)
        SupportSurface = T.join(SupportSurface)


        mask = applyOffset2OpenMask(mask,
                                    OffsetDistanceOfOverlapMask,
                                    SupportSurface,
                                    niter=MaskOffsetNormalsSmoothIterations)

        mask = buildWatertightBodyFromSurfaces([mask],
                                             imposeNormalsOrientation='inwards')

    if SuffixTag: mask[0] = SuffixTag

    return mask

def getOverlapMaskByCellsOffset(base, SuffixTag=None, NCellsOffset=2):
    '''
    Build the overlap mask by selecting a fringe of cells from overlap
    boundaries.

    Parameters
    ----------

        t : PyTree
            the assembled PyTree containing boundary conditions

        SuffixTag : str
            The suffix to attribute to the new mask name

        NCellsOffset : int
            number of cells to offset the mask from *BCOverlap*

        MatchTolerance : float
            small value used for merging auxiliar surface patches

    Returns
    -------

        mask : zone
            unstructured zone consisting in a watertight closed surface
            that can be employed as a mask.
    '''
    # make a temporary tree as elsAProfile.overlapGC2BC() does not accept bases
    t = C.newPyTree([])
    t[2].append(base)
    EP._overlapGC2BC(t)
    FamilyName = 'F_OV_'+base[0]
    mask = ESP.extractSurfacesByOffsetCellsFromBCFamilyName(t, FamilyName,
                                                            NCellsOffset)

    if not mask: return
    mask = ESP.extractWindows(t)
    # mask = C.convertArray2Tetra(mask)
    # mask = T.join(mask)

    # if GSD.isClosed(mask, tol=MatchTolerance):
    #     mask = T.reorderAll(mask, dir=-1) # force normal pointing inwards
    # else:
    #     mask = buildWatertightBodyFromSurfaces([mask],
    #                                          imposeNormalsOrientation='inwards')

    if SuffixTag:
        for m in I.getZones(mask): m[0] = SuffixTag

    return mask

def applyOffset2ClosedMask(mask, offset, niter=None):
    '''
    .. warning:: this is a **private-level** function.

    Creates an offset of a surface removing
    geometrical singularities.

    Parameters
    ----------

        mask : zone
            input surface

        offset : float
            offset distance

        niter : integer
            number of iterations to deform normals *(deprecated)*

    Returns
    -------

        NewClosedMask : zone
            mask surface
    '''
    if not offset: return mask
    mask = T.reorderAll(mask, dir=-1) # force normals to point inwards
    C._initVars(mask, 'offset', offset)
    G._getNormalMap(mask)
    mask = C.center2Node(mask,['centers:sx','centers:sy','centers:sz'])
    I._rmNodesByName(mask,'FlowSolution#Centers')
    # if niter: T._deformNormals(mask, 'offset', niter=niter)
    C._normalize(mask,['sx','sy','sz'])
    C._initVars(mask, 'dx={offset}*{sx}')
    C._initVars(mask, 'dy={offset}*{sy}')
    C._initVars(mask, 'dz={offset}*{sz}')
    T._deform(mask, vector=['dx','dy','dz'])
    I._rmNodesByType(mask,'FlowSolution_t')

    NewClosedMask = removeSingularitiesOnMask(mask)

    return NewClosedMask

def applyOffset2OpenMask(mask, offset, support, niter=None):
    '''

    .. warning:: this is a **private-level** function.

    Applies an offset to an open surface,
    while respecting a constraint on a given support and removing geometrical
    singularities.

    Parameters
    ----------

        mask : zone
            input surface

        offset : float
            offset distance

        support : zone
            zone employed of support during the extrusion process.

        niter : integer
            number of iterations to deform normals

    Returns
    -------

        NewClosedMask : zone
            mask surface

    '''
    if not niter: niter = 0
    if not offset: return mask
    if not support: raise AttributeError('support is required')

    ExtrusionDistribution = D.line((0,0,0),(offset,0,0),10)
    C._initVars(ExtrusionDistribution,'growthfactor',0.)
    C._initVars(ExtrusionDistribution,'growthiters',0.)
    C._initVars(ExtrusionDistribution,'normalfactor',0.)
    C._initVars(ExtrusionDistribution,'normaliters',float(niter))

    mask = T.reorderAll(mask, dir=-1) # force normals to point inwards

    Constraint = dict(kind='Projected',
                      curve=P.exteriorFaces(mask),
                      surface=support,
                      ProjectionMode='ortho',)
    tMask = C.newPyTree(['Base',[mask]])
    tExtru = GVD.extrude(tMask, [ExtrusionDistribution], [Constraint],
                         printIters=False, growthEquation='')
    ExtrudeLayerBase = I.getNodesFromName2(tExtru,'ExtrudeLayerBase')
    NewOpenMaskZones = I.getZones(ExtrudeLayerBase)
    if len(NewOpenMaskZones) > 1:
        raise ValueError(J.FAIL+'Unexpected number of NewOpenMask'+J.ENDC)
    NewOpenMask = NewOpenMaskZones[0]
    NewClosedMask = removeSingularitiesOnMask(NewOpenMask)

    return NewClosedMask

def removeSingularitiesOnMask(mask):
    '''
    Remove geometrical singularities that may have arised after the negative
    extrusion process.

    .. danger:: this function is not sufficiently robust

    Parameters
    ----------

        mask : zone
            surface of the mask including singularities.

    Returns
    -------

        NewClosedMask : zone
            new zone without geometrical singularities.
    '''
    
    mask = XOR.conformUnstr(mask, left_or_right=0, itermax=1)
    masks = T.splitManifold(mask)


    if len(masks) > 1:
        # assumed that closed masks are singular
        openMasks = [m for m in masks if not GSD.isClosed(m)]
        LargeSurfaces = openMasks

        if not LargeSurfaces:
            print(J.WARN+"WRONG MASK - ATTEMPTING SMOOTHING"+J.ENDC)
            mask = T.join(mask)
            C.convertPyTree2File(mask,'debug_maskBeforeSmooth.cgns')
            T._smooth(mask, eps=0.5, type=1, niter=200)
            G._close(mask, tol=1e-3)
            mask = XOR.conformUnstr(mask, left_or_right=0, itermax=1)
            masks = T.splitManifold(mask)
            LargeSurfaces, _ = GSD.filterSurfacesByArea(masks, ratio=0.50)
            body = T.join(LargeSurfaces)
            G._close(body)
            ClosedBody = T.reorderAll(body, dir=-1)
            C.convertPyTree2File(body,'debug_body.cgns')
            return ClosedBody


    else:
        LargeSurfaces = masks

    NewClosedMask = buildWatertightBodyFromSurfaces(LargeSurfaces)

    return NewClosedMask

def buildWatertightBodyFromSurfaces(walls, imposeNormalsOrientation='inwards'):
    '''
    Given a set of surfaces, this function creates a single manifold and
    watertight closed unstructured surface.

    Parameters
    ----------

        walls : list
            list of zones (the surfaces or patches of surfaces)

        imposeNormalsOrientation : str
            can be ``'inwards'`` or ``'outwards'``

            .. tip:: set **imposeNormalsOrientation** = ``'inwards'`` to use
                result as blanking mask

    Returns
    -------

        body : zone
            closed watertight surface (TRI)
    '''

    walls = C.convertArray2Tetra(walls)
    walls = T.join(walls)
    walls = T.splitManifold(walls)
    walls,_ = GSD.filterSurfacesByArea(walls, ratio=0.5)
    body = G.gapsmanager(walls)
    body = T.join(body)
    G._close(body)
    bodyZones = I.getZones(body)
    if len(bodyZones) > 1:
        raise ValueError(J.FAIL+'Unexpected number of body zones'+J.ENDC)
    body = bodyZones[0]

    if imposeNormalsOrientation == 'inwards':
        body = T.reorderAll(body, dir=-1)
    elif imposeNormalsOrientation == 'outwards':
        body = T.reorderAll(body, dir=1)

    return body

def buildWatertightBodiesFromSurfaces(walls, imposeNormalsOrientation='inwards',
                                             SuffixTag=''):
    '''
    Given a set of surfaces, this function creates a set of manifold and
    watertight closed unstructured surfaces.

    Parameters
    ----------

        walls : list
            list of zones (the surfaces or patches of surfaces)

        imposeNormalsOrientation : str
            can be ``'inwards'`` or ``'outwards'``

            .. tip:: set **imposeNormalsOrientation** = ``'inwards'`` to use
                result as blanking mask

        SuffixTag : str
            tag to add as suffix to new zones

    Returns
    -------

        bodies : list
            list of zones, which are closed watertight surfaces (TRI)

    '''

    walls = C.convertArray2Tetra(walls)
    walls = T.join(walls)
    walls = T.splitManifold(walls)
    bodies = []
    for manifoldWall in walls:
        body = G.gapsmanager(manifoldWall)
        body = T.join(body)
        G._close(body)
        bodyZones = I.getZones(body)
        if len(bodyZones) > 1:
            raise ValueError(J.FAIL+'Unexpected number of body zones'+J.ENDC)
        body = bodyZones[0]        
        if imposeNormalsOrientation == 'inwards':
            body = T.reorderAll(body, dir=-1)
        elif imposeNormalsOrientation == 'outwards':
            body = T.reorderAll(body, dir=1)
        if SuffixTag: body[0] = SuffixTag
        bodies.append(body)
    if bodies: I._correctPyTree(bodies, level=-3)

    return bodies

def removeMatchAndNearMatch(t):
    '''
    Remove ``GridConnectivity1to1_t`` and ``Abbuting`` type of connectivity.

    Parameters
    ----------

        t : PyTree
            tree to modify
    '''
    I._rmNodesByType(t, 'GridConnectivity1to1_t')
    for GridConnectivityNode in I.getNodesFromType(t, 'GridConnectivity_t'):
        if I.getNodesFromValue(GridConnectivityNode, 'Abbuting'):
            I.rmNode(t, GridConnectivityNode)

def computeFluidProperties(Gamma=1.4, IdealGasConstant=287.053, Prandtl=0.72,
        PrandtlTurbulence=0.9, SutherlandConstant=110.4,
        SutherlandViscosity=1.78938e-05, SutherlandTemperature=288.15,
        cvAndcp=None):

    '''
    Construct a dictionary of values concerning the fluid properties of air.

    Please note reference default reference values:

    Reference elsA Theory Manual v4.2.01, Table 2.1, Section 2.1.1.5:

    ::

        IdealGasConstant = 287.053
        Gamma = 1.4
        SutherlandConstant = 110.4
        SutherlandTemperature = 288.15
        SutherlandViscosity = 1.78938e-5

    Default values for air in elsA documentation for model object:

    ::

        PrandtlTurbulence = 0.9
        Prandtl = 0.72 # (BEWARE depends on temperature, different from Table 2.1)

    Returns
    -------

        FluidProperties : dict
            set of fluid properties constants
    '''


    if cvAndcp is None:
        cv                = IdealGasConstant/(Gamma-1.0)
        cp                = Gamma * cv
    else:
        cv, cp = cvAndcp

    FluidProperties = dict(
    Gamma                 = Gamma,
    IdealGasConstant               = IdealGasConstant,
    cv                    = cv,
    cp                    = cp,
    Prandtl               = Prandtl,
    PrandtlTurbulence     = PrandtlTurbulence,
    SutherlandConstant    = SutherlandConstant,
    SutherlandViscosity   = SutherlandViscosity,
    SutherlandTemperature = SutherlandTemperature,
    )

    return FluidProperties

def computeReferenceValues(FluidProperties, Density=1.225, Temperature=288.15,
        Velocity=0.0, VelocityUsedForScalingAndTurbulence=None,
        AngleOfAttackDeg=0.0, AngleOfSlipDeg=0.0,
        YawAxis=[0.,0.,1.], PitchAxis=[0.,-1.,0.],
        TurbulenceLevel=0.001,
        Surface=1.0, Length=1.0, TorqueOrigin=[0,0,0],
        TurbulenceModel='Wilcox2006-klim', Viscosity_EddyMolecularRatio=0.1,
        TurbulenceCutoff=0.1, TransitionMode=None,
        WallDistance=None,
        CoprocessOptions={},
        FieldsAdditionalExtractions=['ViscosityMolecular','ViscosityEddy','Mach'],
        BCExtractions=dict(BCWall=['normalvector', 'frictionvector',
                        'psta', 'bl_quantities_2d', 'yplusmeshsize', 'bl_ue_vector',
                        'flux_rou','flux_rov','flux_row','torque_rou','torque_rov','torque_row'])):
    '''
    Compute ReferenceValues dictionary used for pre/co/postprocessing a CFD
    case. It contains multiple information and is mostly self-explanatory.
    Some information contained in this dictionary can used to setup elsA's objects.

    The following is a list of attributes for creating the ReferenceValues
    dictionary. Please note that any variable may be modified/added after the
    creation of the dictionary. The inputs of this function are the most
    commonly modified parameters of a case.

    Parameters
    ----------

        FluidProperties : dict
            as produced by :py:func:`computeFluidProperties`

        Density : float
            Air density in [kg/m3].

        Temperature : float
            air external static temperature in [Kelvin].

        Velocity : float
            farfield true-air-speed magnitude in [m/s]

        VelocityUsedForScalingAndTurbulence : :py:class:`float` or :py:obj:`None`
            velocity magnitude in [m/s] employed for the computation of
            Reynolds, Mach, PressureDynamic and turbulence quantities. By default,
            this value is :py:obj:`None`, which means that **Velocity** is used.

            .. note::
                if ``Velocity=0``, you *must* provide a non-zero value
                of **VelocityUsedForScalingAndTurbulence** in order to compute
                the characteristic quantities of the simulation

        AngleOfAttackDeg : float
            .. note:: see :py:func:`getFlowDirections`

        AngleOfSlipDeg : float
            .. note:: see :py:func:`getFlowDirections`

        YawAxis : float
            .. note:: see :py:func:`getFlowDirections`

        PitchAxis : float
            .. note:: see :py:func:`getFlowDirections`

        TurbulenceLevel : float
            Turbulence intensity used at farfield

        Surface : float
            Reference surface for coefficients computations in [m2]

        Length : float
            Reference length for coefficients computations and
            Reynolds computation, expressed in [m]

        TorqueOrigin : :py:class:`list` of 3 :py:class:`float`
            Reference location coordinates origin :math:`(x,y,z)` for the moments
            computation, expressed in [m]

        TurbulenceModel : str
            Some `NASA's conventional turbulence model <https://turbmodels.larc.nasa.gov/>`_
            available in elsA are included:
            ``'SA'``, ``'Wilcox2006-klim'``, ``'Wilcox2006-klim-V'``,
            ``'Wilcox2006'``, ``'Wilcox2006-V'``, ``'SST-2003'``, 
            ``'SST-V2003'``, ``'SST'``, ``'SST-V'``,  ``'BSL'``, ``'BSL-V'``,
            ``'SST-2003-LM2009'``, ``'SST-V2003-LM2009'``, ``'SSG/LRR-RSM-w2012'``.

            other non-conventional turbulence models:
            ``'smith'`` and ``'smith-V'`` reference `doi:10.2514/6.1995-232 <http://doi.org/10.2514/6.1995-232>`_

        Viscosity_EddyMolecularRatio : float
            Expected ratio of eddy to molecular viscosity at farfield

        TurbulenceCutoff : float
            Ratio of farfield turbulent quantities used for imposing a cutoff.

        TransitionMode : str
            .. attention:: not implemented in workflow standard

        WallDistance : dict or str
            Method to compute wall distance
            Str type: name of the method (e.g. 'mininterf_ortho4')
            Dict type: must contain 'compute' key with name of method as value (e.g. 'mininterf_ortho4'), can contain 'periodic' key in order to take the periodicity into account (e.g. 'two_dir'). The solver can be chosen between 'elsa' and 'cassiopee' (key 'software') and specific extraction can be added  thanks to the key 'extract' (value=boolean True or False).

            .. code-block:: python

               WallDistance=dict(compute='mininterf_ortho4', periodic='two_dir', extract=True, software='elsa')

        CoprocessOptions : dict
            Override default coprocess options dictionary with this paramter.
            Default options are:

            ::

                RequestedStatistics=[],
                UpdateFieldsFrequency   = 2000,
                UpdateArraysFrequency   = 50,
                UpdateSurfacesFrequency = 500,
                TagSurfacesWithIteration= False,
                AveragingIterations     = 3000,
                ItersMinEvenIfConverged = 1000,
                TimeOutInSeconds        = 54000.0, # 15 h * 3600 s/h = 53100 s
                SecondsMargin4QuitBeforeTimeOut = 900.,
                BodyForceInitialIteration = 1,
                BodyForceComputeFrequency = 50,
                BodyForceSaveFrequency    = 100,
                ConvergenceCriteria = [],
                FirstIterationForFieldsAveraging=None,
                DeleteZoneBCGT=False

        FieldsAdditionalExtractions : :py:class:`list` of :py:class:`str`
            elsA or CGNS keywords of fields to be extracted.
            additional fields to be included as extraction.

            .. note:: primitive conservative variables required for restart are
                automatically included

        BCExtractions : dict
            determine the BC (surfacic) extractions.
            See :py:func:`addSurfacicExtractions` doc for relevant parameters.

    Returns
    -------

        ReferenceValues : dict
            dictionary containing all reference values of the simulation
    '''


    DefaultCoprocessOptions = dict(            # Default values
        RequestedStatistics=[],
        UpdateFieldsFrequency   = 2000,
        UpdateArraysFrequency   = 50,
        UpdateSurfacesFrequency = 500,
        TagSurfacesWithIteration= False,
        AveragingIterations     = 3000,
        ItersMinEvenIfConverged = 1000,
        TimeOutInSeconds        = 54000.0, # 15 h * 3600 s/h = 53100 s
        SecondsMargin4QuitBeforeTimeOut = 900.,
        BodyForceInitialIteration = 1,
        BodyForceComputeFrequency = 50,
        BodyForceSaveFrequency    = 100,
        ConvergenceCriteria = [],
        FirstIterationForFieldsAveraging=None,
    )
    DefaultCoprocessOptions.update(CoprocessOptions) # User-provided values

    FreestreamIsTooLow = np.abs(Velocity) < 1e-5
    if FreestreamIsTooLow and VelocityUsedForScalingAndTurbulence is None:
        ERRMSG = 'Velocity is too low (%g). '%Velocity
        ERRMSG+= 'You must provide a non-zero value for VelocityUsedForScalingAndTurbulence'
        raise ValueError(J.FAIL+ERRMSG+J.ENDC)

    if VelocityUsedForScalingAndTurbulence is not None:
        if VelocityUsedForScalingAndTurbulence <= 0:
            ERRMSG = 'VelocityUsedForScalingAndTurbulence must be positive'
            raise ValueError(J.FAIL+ERRMSG+J.ENDC)
    else:
        VelocityUsedForScalingAndTurbulence = np.abs( Velocity )

    RequestedStatistics = DefaultCoprocessOptions['RequestedStatistics']
    for criterion in DefaultCoprocessOptions['ConvergenceCriteria']:
        VariableName = criterion['Variable']
        if VariableName not in RequestedStatistics:
            if any([VariableName.startswith(i) for i in ['std-','rsd-','avg-']]):
                RequestedStatistics.append( VariableName )

    ReferenceValues = dict(
    CoprocessOptions   = DefaultCoprocessOptions,
    AngleOfAttackDeg   = AngleOfAttackDeg,
    AngleOfSlipDeg     = AngleOfSlipDeg,
    TurbulenceLevel    = TurbulenceLevel,
    TurbulenceModel    = TurbulenceModel,
    TransitionMode     = TransitionMode,
    Viscosity_EddyMolecularRatio = Viscosity_EddyMolecularRatio,
    TurbulenceCutoff   = TurbulenceCutoff,
    WallDistance       = WallDistance,
    )

    # Fluid properties local shortcuts
    Gamma   = FluidProperties['Gamma']
    IdealGasConstant = FluidProperties['IdealGasConstant']
    cv      = FluidProperties['cv']

    # REFERENCE VALUES COMPUTATION
    T   = Temperature
    mus = FluidProperties['SutherlandViscosity']
    Ts  = FluidProperties['SutherlandTemperature']
    S   = FluidProperties['SutherlandConstant']

    ViscosityMolecular = mus * (T/Ts)**1.5 * ((Ts + S)/(T + S))
    Mach = VelocityUsedForScalingAndTurbulence /np.sqrt( Gamma * IdealGasConstant * Temperature )
    Reynolds = Density * VelocityUsedForScalingAndTurbulence * Length / ViscosityMolecular
    Pressure = Density * IdealGasConstant * Temperature
    PressureDynamic = 0.5 * Density * VelocityUsedForScalingAndTurbulence **2
    FluxCoef        = 1./(PressureDynamic * Surface)
    TorqueCoef      = 1./(PressureDynamic * Surface*Length)

    # Reference state (farfield)
    FlowDir=getFlowDirections(AngleOfAttackDeg,AngleOfSlipDeg,YawAxis,PitchAxis)
    DragDirection, SideDirection, LiftDirection= FlowDir
    MomentumX =  Density * Velocity * DragDirection[0]
    MomentumY =  Density * Velocity * DragDirection[1]
    MomentumZ =  Density * Velocity * DragDirection[2]
    EnergyStagnationDensity = Density * ( cv * Temperature + 0.5 * Velocity **2)

    # -> for k-omega models
    TurbulentEnergyKineticDensity = Density*1.5*(TurbulenceLevel**2)*(VelocityUsedForScalingAndTurbulence**2)
    TurbulentDissipationRateDensity = Density * TurbulentEnergyKineticDensity / (Viscosity_EddyMolecularRatio * ViscosityMolecular)

    # -> for Smith k-l model
    k = TurbulentEnergyKineticDensity/Density
    omega = TurbulentDissipationRateDensity/Density
    TurbulentLengthScaleDensity = Density*k*18.0**(1./3.)/(np.sqrt(2*k)*omega)

    # -> for k-kL model
    TurbulentEnergyKineticPLSDensity = TurbulentLengthScaleDensity*k

    # -> for Menter-Langtry assuming acceleration factor F(lambda_theta)=1
    IntermittencyDensity = Density * 1.0
    if TurbulenceLevel*100 <= 1.3:
        MomentumThicknessReynoldsDensity = Density * (1173.51 - 589.428*(TurbulenceLevel*100) + 0.2196*(TurbulenceLevel*100)**(-2.))
    else:
        MomentumThicknessReynoldsDensity = Density * ( 331.50*(TurbulenceLevel*100-0.5658)**(-0.671) )

    # -> for RSM models
    ReynoldsStressXX = ReynoldsStressYY = ReynoldsStressZZ = (2./3.) * TurbulentEnergyKineticDensity
    ReynoldsStressXY = ReynoldsStressXZ = ReynoldsStressYZ = 0.
    ReynoldsStressDissipationScale = TurbulentDissipationRateDensity

    TurbulentSANuTilde = None

    Fields = ['Density','MomentumX','MomentumY','MomentumZ','EnergyStagnationDensity']
    ReferenceState = [
    float(Density),
    float(MomentumX),
    float(MomentumY),
    float(MomentumZ),
    float(EnergyStagnationDensity)
    ]

    if   TurbulenceModel == 'SA':
        FieldsTurbulence  = ['TurbulentSANuTildeDensity']

        def computeEddyViscosityFromNuTilde(NuTilde):
            '''
            Compute ViscosityEddy using Eqn. (A1) of DOI:10.2514/6.1992-439
            '''
            Cnu1 = 7.1
            Nu = ViscosityMolecular / Density
            f_nu1 = (NuTilde/Nu)**3 / ((NuTilde/Nu)**3 + Cnu1**3)
            ViscosityEddy = Density * NuTilde * f_nu1

            return ViscosityEddy

        def residualEddyViscosityRatioFromGivenNuTilde(NuTilde):
            return Viscosity_EddyMolecularRatio - computeEddyViscosityFromNuTilde(NuTilde)/ViscosityMolecular

        sol = J.secant(residualEddyViscosityRatioFromGivenNuTilde,
            x0=Viscosity_EddyMolecularRatio*ViscosityMolecular/Density,
            x1=1.5*Viscosity_EddyMolecularRatio*ViscosityMolecular/Density,
            ftol=Viscosity_EddyMolecularRatio*0.001,
            bounds=(1e-14,1.e6))

        NuTilde = sol['root']
        TurbulentSANuTilde = float(NuTilde*Density)

        ReferenceStateTurbulence = [float(TurbulentSANuTilde)]


    elif TurbulenceModel in K_OMEGA_TWO_EQN_MODELS:
        FieldsTurbulence  = ['TurbulentEnergyKineticDensity','TurbulentDissipationRateDensity']
        ReferenceStateTurbulence = [float(TurbulentEnergyKineticDensity), float(TurbulentDissipationRateDensity)]

    elif TurbulenceModel == 'smith':
        FieldsTurbulence  = ['TurbulentEnergyKineticDensity','TurbulentLengthScaleDensity']
        ReferenceStateTurbulence = [float(TurbulentEnergyKineticDensity), float(TurbulentLengthScaleDensity)]

    elif 'LM2009' in TurbulenceModel:
        FieldsTurbulence = ['TurbulentEnergyKineticDensity','TurbulentDissipationRateDensity',
                    'IntermittencyDensity','MomentumThicknessReynoldsDensity']

        ReferenceStateTurbulence = [float(TurbulentEnergyKineticDensity), float(TurbulentDissipationRateDensity), float(IntermittencyDensity), float(MomentumThicknessReynoldsDensity),]

    elif TurbulenceModel == 'SSG/LRR-RSM-w2012':
        FieldsTurbulence = ['ReynoldsStressXX', 'ReynoldsStressXY', 'ReynoldsStressXZ',
                            'ReynoldsStressYY', 'ReynoldsStressYZ', 'ReynoldsStressZZ',
                            'ReynoldsStressDissipationScale']
        ReynoldsStressDissipationScale = TurbulentDissipationRateDensity
        ReferenceStateTurbulence = [
        float(ReynoldsStressXX),
        float(ReynoldsStressXY),
        float(ReynoldsStressXZ),
        float(ReynoldsStressYY),
        float(ReynoldsStressYZ),
        float(ReynoldsStressZZ),
        float(ReynoldsStressDissipationScale)]

    else:
        raise AttributeError('Turbulence model %s not implemented in workflow. Must be in: %s'%(TurbulenceModel,str(AvailableTurbulenceModels)))
    Fields         += FieldsTurbulence
    ReferenceState += ReferenceStateTurbulence

    if isinstance(WallDistance,dict):
        if WallDistance.get('extract',False) and ('TurbulentDistance' not in FieldsAdditionalExtractions):
            FieldsAdditionalExtractions += ['TurbulentDistance','TurbulentDistanceIndex']

    # Update ReferenceValues dictionary
    ReferenceValues.update(dict(
    Reynolds                         = Reynolds,
    Mach                             = Mach,
    Length                           = Length,
    Surface                          = Surface,
    DragDirection                    = list(DragDirection),
    SideDirection                    = list(SideDirection),
    LiftDirection                    = list(LiftDirection),
    Velocity                         = Velocity,
    Temperature                      = Temperature,
    Pressure                         = Pressure,
    PressureDynamic                  = PressureDynamic,
    ViscosityMolecular               = ViscosityMolecular,
    ViscosityEddy                    = ReferenceValues['Viscosity_EddyMolecularRatio'] * ViscosityMolecular,
    FluxCoef                         = FluxCoef,
    TorqueCoef                       = TorqueCoef,
    TorqueOrigin                     = TorqueOrigin,
    Density                          = Density,
    MomentumX                        = MomentumX,
    MomentumY                        = MomentumY,
    MomentumZ                        = MomentumZ,
    EnergyStagnationDensity          = EnergyStagnationDensity,
    TurbulentSANuTilde               = TurbulentSANuTilde,
    TurbulentEnergyKineticDensity    = TurbulentEnergyKineticDensity,
    TurbulentDissipationRateDensity  = TurbulentDissipationRateDensity,
    TurbulentLengthScaleDensity      = TurbulentLengthScaleDensity,
    TurbulentEnergyKineticPLSDensity = TurbulentEnergyKineticPLSDensity,
    IntermittencyDensity             = IntermittencyDensity,
    MomentumThicknessReynoldsDensity = MomentumThicknessReynoldsDensity,
    ReynoldsStressXX                 = ReynoldsStressXX,
    ReynoldsStressYY                 = ReynoldsStressYY,
    ReynoldsStressZZ                 = ReynoldsStressZZ,
    ReynoldsStressXY                 = ReynoldsStressXY,
    ReynoldsStressXZ                 = ReynoldsStressXZ,
    ReynoldsStressYZ                 = ReynoldsStressYZ,
    ReynoldsStressDissipationScale   = ReynoldsStressDissipationScale,
    Fields                           = Fields,
    FieldsTurbulence                 = FieldsTurbulence,
    FieldsAdditionalExtractions      = FieldsAdditionalExtractions,
    BCExtractions                    = BCExtractions,
    ReferenceStateTurbulence         = ReferenceStateTurbulence,
    ReferenceState                   = ReferenceState,
    ))

    if TransitionMode is not None:
        ReferenceValues.update(dict(
        TopOrigin                   = 0.002,
        BottomOrigin                = 0.010,
        TopLaminarImposedUpTo       = 0.001,
        TopLaminarIfFailureUpTo     = 0.2,
        TopTurbulentImposedFrom     = 0.995,
        BottomLaminarImposedUpTo    = 0.001,
        BottomLaminarIfFailureUpTo  = 0.2,
        BottomTurbulentImposedFrom  = 0.995
        ))

    ReferenceValues['PREPROCESS_SCRIPT'] = main_script_path = os.path.abspath(__import__('__main__').__file__)

    return ReferenceValues

def getElsAkeysCFD(config='3d', unstructured=False, **kwargs):
    '''
    Create a dictionary of pairs of elsA keyword/values to be employed as
    cfd problem object.

    Parameters
    ----------

        config : str
            elsa keyword config (``'2d'`` or ``'3d'``)

        unstructured : bool
            if :py:obj:`True`, add keys adapted for unstructured mesh.

        kwargs : dict
            additional parameters for elsA *cfd* object

    Returns
    -------

        elsAkeysCFD : dict
            dictionary containing key/value for elsA *cfd* object
    '''
    elsAkeysCFD = dict(config=config,
        extract_filtering='inactive' # NOTE required with writingmode=2 for NeumannData in coprocess
        )

    if unstructured:
        elsAkeysCFD.update(dict(
            metrics_as_unstruct='active',
            metrics_type='barycenter'))

    elsAkeysCFD.update(kwargs)
    return elsAkeysCFD

def getElsAkeysModel(FluidProperties, ReferenceValues, unstructured=False, **kwargs):
    '''
    Produce the elsA model object keys as a Python dictionary.

    Parameters
    ----------

        FluidProperties : dict
            as obtained from :py:func:`computeFluidProperties`

        ReferenceValues : dict
            as obtained from :py:func:`computeReferenceValues`

        unstructured : bool
            if :py:obj:`True`, add keys adapted for unstructured mesh

        kwargs : dict
            additional parameters for elsA *model* object

    Returns
    -------

        elsAkeysModel : dict
            dictionary containing key/value for elsA *model* object

    '''
    TurbulenceModel = ReferenceValues['TurbulenceModel']
    TransitionMode  = ReferenceValues['TransitionMode']

    elsAkeysModel = dict(
    cv               = FluidProperties['cv'],
    fluid            = 'pg',
    gamma            = FluidProperties['Gamma'],
    phymod           = 'nstur',
    prandtl          = FluidProperties['Prandtl'],
    prandtltb        = FluidProperties['PrandtlTurbulence'],
    visclaw          = 'sutherland',
    suth_const       = FluidProperties['SutherlandConstant'],
    suth_muref       = FluidProperties['SutherlandViscosity'],
    suth_tref        = FluidProperties['SutherlandTemperature'],

    # Boundary-layer computation parameters
    vortratiolim    = 1e-3,
    shearratiolim   = 2e-2,
    pressratiolim   = 1e-3,
    linearratiolim  = 1e-3,
    delta_compute   = 'first_order_bl',
    )

    if unstructured:
        elsAkeysModel['walldistcompute'] = 'mininterf'
    else:
        elsAkeysModel['walldistcompute'] = 'mininterf_ortho'
    WallDistance = ReferenceValues.get('WallDistance',None)
    if isinstance(WallDistance,dict):
        elsAkeysModel['walldistcompute'] = WallDistance.get('compute',elsAkeysModel['walldistcompute'])
        elsAkeysModel['walldistperio']   = WallDistance.get('periodic','none')
    elif isinstance(WallDistance,str):
        elsAkeysModel['walldistcompute'] = WallDistance

    if TurbulenceModel == 'SA':
        addKeys4Model = dict(
        turbmod        = 'spalart',
            )

    elif TurbulenceModel == 'Wilcox2006-klim':
        addKeys4Model = dict(
        turbmod        = 'komega_kok',
        kok_diff_cor   = 'wilcox2006',
        sst_cor        = 'active',
        sst_version    = 'wilcox2006',
        k_prod_limiter = 20.,
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
            )

    elif TurbulenceModel == 'Wilcox2006-klim-V':
        addKeys4Model = dict(
        turbmod        = 'komega_kok',
        kok_diff_cor   = 'wilcox2006',
        sst_cor        = 'active',
        sst_version    = 'wilcox2006',
        k_prod_limiter = 20.,
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
            )

    elif TurbulenceModel == 'Wilcox2006':
        addKeys4Model = dict(
        turbmod        = 'komega_kok',
        kok_diff_cor   = 'wilcox2006',
        sst_cor        = 'active',
        sst_version    = 'wilcox2006',
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
            )

    elif TurbulenceModel == 'Wilcox2006-V':
        addKeys4Model = dict(
        turbmod        = 'komega_kok',
        kok_diff_cor   = 'wilcox2006',
        sst_cor        = 'active',
        sst_version    = 'wilcox2006',
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
            )


    elif TurbulenceModel == 'SST-2003':
        addKeys4Model = dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'std_sij',
        k_prod_limiter = 10.,
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
            )

    elif TurbulenceModel == 'SST-V2003':
        addKeys4Model = dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'std_sij',
        k_prod_limiter = 10.,
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
            )

    elif TurbulenceModel == 'SST':
        addKeys4Model = dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'standard',
        k_prod_limiter = 20.,
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
            )

    elif TurbulenceModel == 'SST-V':
        addKeys4Model = dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'standard',
        k_prod_limiter = 20.,
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
            )

    elif TurbulenceModel == 'BSL':
        addKeys4Model = dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'inactive',
        k_prod_limiter = 20.,
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
            )

    elif TurbulenceModel == 'BSL-V':
        addKeys4Model = dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'inactive',
        k_prod_limiter = 20.,
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
            )

    elif TurbulenceModel == 'smith':
        addKeys4Model = dict(
        turbmod        = 'smith',
        k_prod_compute = 'from_sij',
            )

    elif TurbulenceModel == 'smith-V':
        addKeys4Model = dict(
        turbmod        = 'smith',
        k_prod_compute = 'from_vorticity',
            )

    elif TurbulenceModel == 'SST-2003-LM2009':
        addKeys4Model = dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'std_sij',
        k_prod_limiter = 10.,
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
        trans_mod      = 'menter',
            )

    elif TurbulenceModel == 'SST-V2003-LM2009':
        addKeys4Model = dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'std_sij',
        k_prod_limiter = 10.,
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
        trans_mod      = 'menter',
            )

    elif TurbulenceModel == 'SSG/LRR-RSM-w2012':
        addKeys4Model = dict(
        turbmod          = 'rsm',
        rsm_name         = 'ssg_lrr_bsl',
        rsm_diffusion    = 'isotropic',
        rsm_bous_limiter = 10.0,
        omega_prolong    = 'linear_extrap',
                            )

    # Transition Settings
    if TransitionMode == 'NonLocalCriteria-LSTT':
        if 'LM2009' in TurbulenceModel:
            raise AttributeError(J.FAIL+"Modeling incoherency! cannot make Non-local transition criteria with Menter-Langtry turbulence model"+J.ENDC)
        addKeys4Model.update(dict(
        freqcomptrans     = 1,
        trans_crit        = 'in_ahd_gl_comp',
        trans_max_bubble  = 'inactive',
        ext_turb_lev      = ReferenceValues['TurbulenceLevel'] * 100,
        intermittency     = 'limited',
        interm_thick_coef = 1.2,
        ext_turb_lev_lim  = 'constant_tu',
        trans_shift       = 1.0,
        delta_compute     = elsAkeysModel['delta_compute'],
        vortratiolim      = elsAkeysModel['vortratiolim'],
        shearratiolim     = elsAkeysModel['shearratiolim'],
        pressratiolim     = elsAkeysModel['pressratiolim'],
        firstcomptrans    = 1,
        lastcomptrans     = int(1e9),
        trans_comp_h      = 'h_calc',
        trans_gl_ctrl_h1  = 3.0,
        trans_gl_ctrl_h2  = 3.2,
        trans_gl_ctrl_h3  = 3.6,
        # LSTT specific parameters (see ticket #6501)
        trans_crit_order       = 'first_order',
        trans_crit_extrap      = 'active',
        intermit_region        = 'LSTT', # TODO: Not read in fullCGNS -> https://elsa.onera.fr/issues/8145
        intermittency_form     = 'LSTT19',
        trans_h_crit_ahdgl     = 2.8,
        ahd_n_extract          = 'active',
            ))

        if TurbulenceModel in K_OMEGA_MODELS:  addKeys4Model['prod_omega_red'] = 'active'

    # Transition Settings
    if TransitionMode == 'NonLocalCriteria-Step':
        if 'LM2009' in TurbulenceModel:
            raise AttributeError(J.FAIL+"Modeling incoherency! cannot make Non-local transition criteria with Menter-Langtry turbulence model"+J.ENDC)
        addKeys4Model.update(dict(
        freqcomptrans     = 1,
        trans_crit        = 'in_ahd_comp',
        trans_max_bubble  = 'inactive',
        ext_turb_lev      = ReferenceValues['TurbulenceLevel'] * 100,
        intermittency     = 'limited',
        interm_thick_coef = 1.2,
        ext_turb_lev_lim  = 'constant_tu',
        trans_shift       = 1.0,
        delta_compute     = elsAkeysModel['delta_compute'],
        vortratiolim      = elsAkeysModel['vortratiolim'],
        shearratiolim     = elsAkeysModel['shearratiolim'],
        pressratiolim     = elsAkeysModel['pressratiolim'],
        firstcomptrans    = 1,
        lastcomptrans     = int(1e9),
        trans_comp_h      = 'h_calc',
        intermittency_form     = 'LSTT19',
        trans_h_crit_ahdgl     = 2.8,
        ahd_n_extract          = 'active',
            ))

        if TurbulenceModel in K_OMEGA_MODELS:  addKeys4Model['prod_omega_red'] = 'active'


    elif TransitionMode == 'Imposed':
        if 'LM2009' in TurbulenceModel:
            raise AttributeError(J.FAIL+"Modeling incoherency! cannot make imposed transition with Menter-Langtry turbulence model"+J.ENDC)
        addKeys4Model.update(dict(
        intermittency       = 'full',
        interm_thick_coef   = 1.2,
        delta_compute       = elsAkeysModel['delta_compute'],
        vortratiolim        = elsAkeysModel['vortratiolim'],
        shearratiolim       = elsAkeysModel['shearratiolim'],
        pressratiolim       = elsAkeysModel['pressratiolim'],
        intermittency_form  = 'LSTT19',
            ))

        if TurbulenceModel in K_OMEGA_MODELS:  addKeys4Model['prod_omega_red'] = 'active'

    elsAkeysModel.update(addKeys4Model)

    elsAkeysModel.update(kwargs)

    return elsAkeysModel

def getElsAkeysNumerics(ReferenceValues, NumericalScheme='jameson',
        TimeMarching='steady', inititer=1, niter=30000,
        CFLparams=dict(vali=1.,valf=10.,iteri=1,iterf=1000,function_type='linear'),
        itime=0., timestep=0.01, useBodyForce=False, useChimera=False,
        unstructured=False, **kwargs):
    '''
    Get the Numerics elsA keys as a Python dictionary.

    Parameters
    ----------

        ReferenceValues : dict
            as got from :py:func:`computeReferenceValues`

        NumericalScheme : str
            one of: (``'jameson'``, ``'ausm+'``, ``'roe'``)

        TimeMarching : str
            One of: (``'steady'``, ``'UnsteadyFirstOrder'``, ``'gear'``,
            ``'DualTimeStep'``)

        inititer : int
            initial iteration

        niter : int
            total number of iterations to run

        CFLparams : dict
            indicates the CFL function to be employed

        itime : float
            initial time

        timestep : float
            timestep for unsteady simulation (in seconds)

        useBodyForce : bool
            :py:obj:`True` if bodyforce is employed

        useChimera : bool
            :py:obj:`True` if chimera (static) is employed

        unstructured : bool
            if :py:obj:`True`, add keys adapted for unstructured mesh

        kwargs : dict
            additional parameters for elsA *numerics* object

    Returns
    -------

        elsAkeysNumerics : dict
            contains *numerics* object elsA keys/values
    '''
    

    NumericalScheme = NumericalScheme.lower() # avoid case-type mistakes
    elsAkeysNumerics = dict()
    CFLparams['name'] = 'f_cfl'
    for v in ('vali', 'valf'): CFLparams[v] = float(CFLparams[v])
    for v in ('iteri', 'iterf'): CFLparams[v] = int(CFLparams[v])

    elsAkeysNumerics.update(dict(viscous_fluxes='5p_cor'))
    if NumericalScheme == 'jameson':
        addKeys = dict(
        flux               = 'jameson',
        avcoef_k2          = 0.5,
        avcoef_k4          = 0.016,
        avcoef_sigma       = 1.0,
        filter             = 'incr_new+prolong',
        cutoff_dens        = 0.005,
        cutoff_pres        = 0.005,
        cutoff_eint        = 0.005,
        artviscosity       = 'dismrt',
        av_mrt             = 0.3,
        av_border          = 'current', # default elsA is 'dif0null', but JCB, JM, LC use 'current' see https://elsa.onera.fr/issues/10624
        av_formul          = 'current', # default elsA is 'new', but JCB, JM, LC use 'current' see https://elsa.onera.fr/issues/10624
        )
        if not unstructured:
            addKeys.update(dict(
                artviscosity       = 'dismrt',
                av_mrt             = 0.3,
            ))
        else:
            # Martinelli correction not available for unstructured grids
            addKeys['artviscosity'] = 'dissca'

    elif NumericalScheme == 'ausm+':
        addKeys = dict(
        flux               = 'ausmplus_pmiles',
        ausm_wiggle        = 'inactive',
        ausmp_diss_cst     = 0.04,
        ausmp_press_vel_cst= 0.04,
        ausm_tref          = float(ReferenceValues['Temperature']),
        ausm_pref          = float(ReferenceValues['Pressure']),
        ausm_mref          = float(ReferenceValues['Mach']),
        limiter            = 'third_order',
        )
    elif NumericalScheme == 'roe':
        addKeys = dict(
        flux               = 'roe',
        limiter            = 'valbada',
        psiroe             = 0.01,
        )
    else:
        raise AttributeError('Numerical scheme shortcut %s not recognized'%NumericalScheme)
    elsAkeysNumerics.update(addKeys)

    addKeys = dict(
        inititer           = int(inititer),
        niter              = int(niter),
        ode                = 'backwardeuler',
        implicit           = 'lussorsca',
        ssorcycle          = 4,
        freqcompres        = 1,
    )
    if TimeMarching == 'steady':
        addKeys.update(dict(
            time_algo          = 'steady',
            global_timestep    = 'inactive',
            timestep_div       = 'divided',  # timestep divided by 2 at the boundaries ; should not be used in unsteady simulations
            cfl_fct            = CFLparams['name'],
            residual_type      = 'explicit_novolum',
        ))
        addKeys['.Solver#Function'] = CFLparams
    else:
        addKeys.update(dict(
            timestep           = float(timestep),
            itime              = float(itime),
            restoreach_cons    = 1e-2,
        ))
        if TimeMarching == 'gear':
            addKeys.update(dict(
                time_algo          = 'gear',
                gear_iteration     = 20,
            ))
        elif TimeMarching == 'DualTimeStep':
            addKeys.update(dict(
                time_algo          = 'dts',
                dts_timestep_lim   = 'active',
                cfl_dts            = 20.,
            ))
        elif TimeMarching == 'UnsteadyFirstOrder':
            addKeys.update(dict(
                time_algo          = 'unsteady',
            ))
        else:
            raise AttributeError('TimeMarching scheme shortcut %s not recognized'%TimeMarching)
    elsAkeysNumerics.update(addKeys)

    if useBodyForce:
        addKeys = dict(misc_source_term='active')
    else:
        addKeys = dict(misc_source_term='inactive')
    elsAkeysNumerics.update(addKeys)



    if useChimera:
        addKeys = dict(chm_double_wall='active',
                       chm_double_wall_tol=2000.,
                       chm_orphan_treatment= 'neighbourgsmean',
                       chm_impl_interp='none',
                       chm_interp_depth=2)


    addKeys.update(dict(
    multigrid     = 'none',
    t_harten      = 0.01,
    muratiomax    = 1.e+20,
        ))

    ReferenceStateTurbulence = ReferenceValues['ReferenceStateTurbulence']
    TurbulenceCutoff         = ReferenceValues['TurbulenceCutoff']
    if len(ReferenceStateTurbulence)==7:  # RSM
        addKeys['t_cutvar1'] = TurbulenceCutoff*ReferenceStateTurbulence[0]
        addKeys['t_cutvar2'] = TurbulenceCutoff*ReferenceStateTurbulence[3]
        addKeys['t_cutvar3'] = TurbulenceCutoff*ReferenceStateTurbulence[5]
        addKeys['t_cutvar4'] = TurbulenceCutoff*ReferenceStateTurbulence[6]

    elif len(ReferenceStateTurbulence)>4: # unsupported 
        raise ValueError('UNSUPPORTED NUMBER OF TURBULENT FIELDS')
    
    else:
        for i in range(len(ReferenceStateTurbulence)):
            addKeys['t_cutvar%d'%(i+1)] = TurbulenceCutoff*ReferenceStateTurbulence[i]
    elsAkeysNumerics.update(addKeys)

    if unstructured:
        elsAkeysNumerics.update(dict(
            implconvectname = 'vleer', # only available for unstructured mesh, see https://elsa-e.onera.fr/issues/6492
            viscous_fluxes  = '5p_cor2', # adapted to unstructured mesh # TODO Set 5p_cor2 for structured mesh also ?
        ))

    elsAkeysNumerics.update(kwargs)

    # Handle incompatibilities
    if (elsAkeysNumerics['implicit'] == 'lussorscawf') and (ReferenceValues['TurbulenceModel'] == 'SSG/LRR-RSM-w2012'):
        # HACK see elsA issue https://elsa.onera.fr/issues/11312
        raise Exception(J.FAIL+'lussorscawf and SSG/LRR-RSM-w2012 turbulence model are incompatible'+J.ENDC)
    if unstructured and (elsAkeysNumerics['implicit'] == 'lussorsca') and ('LM2009' in ReferenceValues['TurbulenceModel']):
        raise Exception(J.FAIL+'With unstructured mesh, lussorsca and LM2009 transition model are incompatible.'+J.ENDC)

    return elsAkeysNumerics

def newCGNSfromSetup(t, AllSetupDictionaries, Initialization=None,
                     FULL_CGNS_MODE=False,  extractCoords=True, BCExtractions={},
                     secondOrderRestart=False):
    '''
    Given a preprocessed grid using :py:func:`prepareMesh4ElsA` and setup information
    dictionaries, this function creates the main CGNS tree and writes the
    ``setup.py`` file.

    Parameters
    ----------

        t : PyTree
            preprocessed tree as performed by :py:func:`prepareMesh4ElsA`

        AllSetupDictionaries : dict
            dictionary containing at least the
            dictionaries: ``ReferenceValues``, ``elsAkeysCFD``,
            ``elsAkeysModel`` and ``elsAkeysNumerics``.

        Initialization : dict
            dictionary defining the type of flow initialization. If :py:obj:`None`,
            no initialization is performed, else it depends on the key 'method'.

        FULL_CGNS_MODE : bool
            if :py:obj:`True`, add elsa keys in ``.Solver#Compute`` CGNS container

        extractCoords : bool
            if :py:obj:`True`, then create a ``FlowSolution`` container named
            ``FlowSolution#EndOfRun#Coords`` to perform coordinates extraction.

        BCExtractions : dict
            dictionary to indicate variables to extract. Each key corresponds to
            a ``BCType`` or a pattern to search in BCType of each FamilyBC (for
            instance, *BCInflow* matches *BCInflowSubsonic* and
            *BCInflowSupersonic*).
            The value associated to each key is a :py:class:`list` of
            :py:class:`str` corresponding to variable names to extract (in elsA
            convention).

            To see default extracted variables, see :py:func:`addSurfacicExtractions`

        secondOrderRestart : bool
            if :py:obj:`True`, duplicate the node ``'FlowSolution#Init'`` into a
            new node ``'FlowSolution#Init-1'`` to mimic the expected
            behavior for the next restart.

    Returns
    -------

        tNew : PyTree
            CGNS tree containing all required data for elsA computation

    '''
    t = I.copyRef(t)

    addTrigger(t)

    if 'OversetMotion' in AllSetupDictionaries and AllSetupDictionaries['OversetMotion']:
        addOversetMotion(t, AllSetupDictionaries['OversetMotion'])
        includeAbsoluteFieldsForSurfacesPostprocessing = True
    else:
        includeAbsoluteFieldsForSurfacesPostprocessing = False

    is_unsteady = AllSetupDictionaries['elsAkeysNumerics']['time_algo'] != 'steady'
    avg_requested = AllSetupDictionaries['ReferenceValues']['CoprocessOptions']['FirstIterationForFieldsAveraging'] is not None

    if is_unsteady and not avg_requested:
        msg =('WARNING: You are setting an unsteady simulation, but no field averaging\n'
              'will be done since CoprocessOptions key "FirstIterationForFieldsAveraging"\n'
              'is set to None. If you want fields average extraction, please set a finite\n'
              'positive value to "FirstIterationForFieldsAveraging" and relaunch preprocess')
        print(J.WARN+msg+J.ENDC)

    addExtractions(t, AllSetupDictionaries['ReferenceValues'],
                      AllSetupDictionaries['elsAkeysModel'],
                      extractCoords=extractCoords, BCExtractions=BCExtractions,
                      includeAbsoluteFieldsForSurfacesPostprocessing=includeAbsoluteFieldsForSurfacesPostprocessing,
                      add_time_average= is_unsteady and avg_requested,
                      secondOrderRestart=secondOrderRestart)
    addReferenceState(t, AllSetupDictionaries['FluidProperties'],
                         AllSetupDictionaries['ReferenceValues'])
    dim = int(AllSetupDictionaries['elsAkeysCFD']['config'][0])
    addGoverningEquations(t, dim=dim)
    if Initialization:
        initializeFlowSolution(t, Initialization, AllSetupDictionaries['ReferenceValues'], secondOrderRestart=secondOrderRestart)

    if FULL_CGNS_MODE:
        addElsAKeys2CGNS(t, [AllSetupDictionaries['elsAkeysCFD'],
                             AllSetupDictionaries['elsAkeysModel'],
                             AllSetupDictionaries['elsAkeysNumerics']])

    writeSetup(AllSetupDictionaries)

    return t


def addOversetMotion(t, OversetMotion):
    if not OversetMotion: return
    bases = I.getBases(t)
    bases_names = [b[0] for b in bases]
    NewOversetMotion = dict()
    for k in OversetMotion:
        base_found = bool([b for b in bases if b[0]==k])
        if base_found: continue
        base_candidates = [b for b in bases if b[0].startswith(k)]
        never_found = True
        for i, b in enumerate(base_candidates):
            try:
                base_found = [b for b in base_candidates if b[0]==k+'_%d'%(i+1)][0]
                never_found = False
            except IndexError:
                continue
            NewOversetMotion[base_found[0]] = OversetMotion[k]
        if never_found:
            msg=('tried to set motion to component %s or inherited, but never found.'
                 '\nAvailable component names are: %s')%(k,str(bases_names))
            raise ValueError(J.FAIL+msg+J.ENDC)
    OversetMotion.update(NewOversetMotion)

    for base in bases:

        motion_keys = dict( motion=1, omega=0.0, transl_speed=0.0,
                            axis_ang_1=1, axis_ang_2=1 )

        try:             OversetMotionData = OversetMotion[base[0]]
        except KeyError: OversetMotionData = dict(RPM=0.0)


        FamilyMotionName = 'MOTION_'+base[0]
        for z in I.getZones(base):
            I.createUniqueChild(z,FamilyMotionName,'FamilyName_t',
                                    value=FamilyMotionName)
        family = I.createChild(base, FamilyMotionName, 'Family_t')
        I.createChild(family,'FamilyBC','FamilyBC_t',value='UserDefined')

        rc, ra, td = _getMotionDataFromMeshInfo(base)
        motion_keys['function_name']=FamilyMotionName
        motion_keys['omega']=OversetMotionData['RPM']*np.pi/30.
        motion_keys['axis_pnt_x']=rc[0]
        motion_keys['axis_pnt_y']=rc[1]
        motion_keys['axis_pnt_z']=rc[2]
        motion_keys['axis_vct_x']=ra[0]
        motion_keys['axis_vct_y']=ra[1]
        motion_keys['axis_vct_z']=ra[2]
        motion_keys['transl_vct_x']=td[0]
        motion_keys['transl_vct_y']=td[1]
        motion_keys['transl_vct_z']=td[2]
        
        _setMobileCoefAtBCsExceptOverlap(base, mobile_coef=-1.0)

        J.set(family,'.Solver#Motion', **motion_keys)

        MeshInfo = J.get(base,'.MOLA#InputMesh') 
        try: is_duplicated = bool(MeshInfo['DuplicatedFrom'] != base[0])
        except KeyError: is_duplicated = False

        
        phase = 0.0

        if is_duplicated:
            blade_id = int(base[0].split('_')[-1])
            blade_nb = MeshInfo['Motion']['NumberOfBlades']
            try:
                RH=MeshInfo['Motion']['RequestedFrame']['RightHandRuleRotation']
                sign = 1 if RH else -1
            except KeyError:
                sign = 1
            psi0_b = (blade_id-1)*sign*(360.0/float(blade_nb)) + phase
        else:
            psi0_b = phase

        try: bd = MeshInfo['Motion']['RequestedFrame']['BladeDirection']
        except KeyError: bd = [1,0,0]
        bd = np.array(bd,dtype=float)

        default_rotor_motion = dict(type='rotor_motion',
            initial_angles=[0.,psi0_b],
            alp0=0.,
            alp_pnt=[0.,0.,0.],
            alp_vct=[0.,1.,0.],
            rot_pnt=[rc[0],rc[1],rc[2]],
            rot_vct=[ra[0],ra[1],ra[2]],
            rot_omg=motion_keys['omega'],
            span_vct=bd,
            pre_lag_pnt=[0.,0.,0.],
            pre_lag_vct=[0.,0.,1.],
            pre_lag_ang=0.,
            pre_con_pnt=[0.,0.,0.],
            pre_con_vct=[0.,1.,0.],
            pre_con_ang=0.,
            del_pnt=[0.,0.,0.],
            del_vct=[0.,0.,1.],
            del0=0.,
            bet_pnt=[0.,0.,0.],
            bet_vct=[0.,1.,0.],
            bet0=0.,
            tet_pnt=[0.,0.,0.],
            tet_vct=[1.,0.,0.],
            tet0=0.)        
        

        try:
            function_motion_type = OversetMotionData['Function']['type']
        except KeyError:
            function_motion_type = 'rotor_motion'
            try:
                OversetMotionData['Function']['type'] = 'rotor_motion'
            except KeyError:
                OversetMotionData['Function'] = dict(type='rotor_motion')

        if function_motion_type == 'rotor_motion':
            default_rotor_motion.update(OversetMotionData['Function'])
            J.set(family,'.MOLA#Motion',**default_rotor_motion)
        else:
            J.set(family,'.MOLA#Motion',**OversetMotionData['Function'])
        
        MOLA_Motion = I.getNodeFromName1(family,'.MOLA#Motion')
        I.setValue(MOLA_Motion, FamilyMotionName)



def _getMotionDataFromMeshInfo(base):
    defaultRotationCenter = np.array([0.,0.,0.],order='F')
    defaultRotationAxis = np.array([0.,0.,1.],order='F')
    defaultTranslationDirection = np.array([1.,0.,0.],order='F')
    default = defaultRotationCenter, defaultRotationAxis, defaultTranslationDirection

    MeshInfo = J.get(base,'.MOLA#MeshInfo')
    if not MeshInfo: return default
    try: MotionData = MeshInfo['Motion']
    except KeyError: return default

    try: RotationCenter = MotionData['RotationCenter']
    except KeyError: RotationCenter = defaultRotationCenter

    try: RotationAxis = MotionData['RotationAxis']
    except KeyError: RotationAxis = defaultRotationAxis

    try: TranslationDirection = MotionData['TranslationDirection']
    except KeyError: TranslationDirection = defaultTranslationDirection

    return RotationCenter, RotationAxis, TranslationDirection


def _setMobileCoefAtBCsExceptOverlap(t, mobile_coef=-1.0):
    for base in I.getBases(t):
        for family in I.getNodesFromType1(base,'Family_t'):
            BCType = getFamilyBCTypeFromFamilyBCName(base, family[0])
            if BCType and BCType != 'BCOverlap':
                SolverBC = I.getNodeFromName1(family,'.Solver#BC')
                if not SolverBC:
                    J.set(family, '.Solver#BC', mobile_coef=mobile_coef)
                else:
                    I.createUniqueChild(SolverBC,'mobile_coef',
                                        'DataArray_t', value=mobile_coef)




def saveMainCGNSwithLinkToOutputFields(t, DIRECTORY_OUTPUT='OUTPUT',
                               MainCGNSFilename='main.cgns',
                               FieldsFilename='fields.cgns',
                               writeOutputFields=True):
    '''
    Saves the ``main.cgns`` file including linsk towards ``OUTPUT/fields.cgns``
    file, which contains ``FlowSolution#Init`` fields.

    Parameters
    ----------

        t : PyTree
            fully preprocessed PyTree

        DIRECTORY_OUTPUT : str
            folder containing the file ``fields.cgns``

            .. note:: it is advised to use ``'OUTPUT'``

        MainCGNSFilename : str
            name for main CGNS file.

            .. note:: it is advised to use ``'main.cgns'``

        FieldsFilename : str
            name of CGNS file containing initial fields

            .. note:: it is advised to use ``'fields.cgns'``

        writeOutputFields : bool
            if :py:obj:`True`, write ``fields.cgns`` file

    Returns
    -------

        None - None
            files ``main.cgns`` and eventually ``OUTPUT/fields.cgns`` are written
    '''
    print('gathering links between main CGNS and fields')
    AllCGNSLinks = []
    include_zone_bc_link = I.getNodeFromName(t,'.Solver#Output#Average') is not None
    for b in I.getBases(t):
        for z in b[2]:
            if z[3] != 'Zone_t': continue
            for fs in I.getNodesFromName(z, 'FlowSolution#Init*') + I.getNodesFromName(z, 'FlowSolution#Average'):
                currentNodePath='/'.join([b[0], z[0], fs[0]])
                targetNodePath=currentNodePath
                AllCGNSLinks += [['.',
                                DIRECTORY_OUTPUT+'/'+FieldsFilename,
                                '/'+targetNodePath,
                                currentNodePath]]

            if include_zone_bc_link:
                zbc = I.getNodeFromType1(z,'ZoneBC_t')
                if zbc:
                    for bc in I.getNodesFromType1(zbc, 'BC_t'):
                        currentNodePath='/'.join([b[0], z[0], zbc[0], bc[0], 'BCDataSet#Average'])
                        bcdsavg = I.createNode('BCDataSet#Average', 'BCDataSet_t', parent=bc)

                        targetNodePath=currentNodePath
                        AllCGNSLinks += [['.',
                                        DIRECTORY_OUTPUT+'/'+FieldsFilename,
                                        '/'+targetNodePath,
                                        currentNodePath]]

    print('saving PyTrees with links')
    to = I.copyRef(t)
    I._renameNode(to, 'FlowSolution#Centers', 'FlowSolution#Init')

    # HACK required in order to avoid AssertionError at line 771 in
    # etc/pypart/PpartCGNS/LayoutsS.pxi, Layouts.splitBCDataSet 
    for b in I.getBases(to):
        for z in b[2]:
            if z[3] != 'Zone_t': continue
            zbc = I.getNodeFromType1(z,'ZoneBC_t')
            if zbc:
                for bc in I.getNodesFromType1(zbc, 'BC_t'):
                    bcdsavg = I.getNodeFromName1(bc, 'BCDataSet#Average')
                    if bcdsavg: bcdsavg[3] = 'UserDefinedData_t'

    if writeOutputFields:
        try: os.makedirs(DIRECTORY_OUTPUT)
        except: pass
        C.convertPyTree2File(to, os.path.join(DIRECTORY_OUTPUT, FieldsFilename))
    C.convertPyTree2File(t, MainCGNSFilename, links=AllCGNSLinks)

def addTrigger(t, coprocessFilename='coprocess.py'):
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

    C._tagWithFamily(t,'ALLZONES',add=True)
    for b in I.getBases(t): C._addFamily2Base(b, 'ALLZONES')
    AllZonesFamilyNodes = I.getNodesFromName2(t,'ALLZONES')
    for n in AllZonesFamilyNodes:
        J.set(n, '.Solver#Trigger',
                 next_state=16,
                 next_iteration=1,
                 file=coprocessFilename)

def addExtractions(t, ReferenceValues, elsAkeysModel, extractCoords=True,
        BCExtractions=dict(), includeAbsoluteFieldsForSurfacesPostprocessing=False,
        add_time_average=False, secondOrderRestart=False):
    '''
    Include surfacic and field extraction information to CGNS tree using
    information contained in dictionaries **ReferenceValues** and
    **elsAkeysModel**.

    Parameters
    ----------

        t : PyTree
            prepared grid as produced by :py:func:`prepareMesh4ElsA` function.

            .. note:: tree **t** is modified

        ReferenceValues : dict
            dictionary as produced by :py:func:`computeReferenceValues` function

        elsAkeysModel : dict
            dictionary as produced by :py:func:`getElsAkeysModel` function

        extractCoords : bool
            if :py:obj:`True`, then create a ``FlowSolution`` container named
            ``FlowSolution#EndOfRun#Coords`` to perform coordinates extraction.

        BCExtractions : dict
            dictionary to indicate variables to extract. Each key corresponds to
            a ``BCType`` or a pattern to search in BCType of each FamilyBC (for
            instance, *BCInflow* matches *BCInflowSubsonic* and
            *BCInflowSupersonic*).
            The value associated to each key is a :py:class:`list` of
            :py:class:`str` corresponding to variable names to extract (in elsA
            convention).

            To see default extracted variables, see :py:func:`addSurfacicExtractions`
        
        includeAbsoluteFieldsForSurfacesPostprocessing : bool
            if :py:obj:`True`, then creates an additional container
            'FlowSolution#EndOfRun#Absolute' where restart fields will be 
            extracted

        add_time_average : bool
            if :py:obj:`True`, include additional ``FlowSolution#Average`` for 
            time-averaged field extraction and ``.Solver#Output#Average`` 
            for time-averaging fields at boundary-conditions.
        
        secondOrderRestart : bool
            if :py:obj:`True`, activate the saving of the flow solution at the
            penultimate iteration to allow a second order restart.

    '''
    addSurfacicExtractions(t, ReferenceValues, elsAkeysModel,
        BCExtractions=BCExtractions, add_time_average=add_time_average)
    addFieldExtractions(t, ReferenceValues, extractCoords=extractCoords,
        add_time_average=add_time_average, secondOrderRestart=secondOrderRestart)
    if includeAbsoluteFieldsForSurfacesPostprocessing:
        addFieldExtractions(t, ReferenceValues, extractCoords=False,
        includeAdditionalExtractions=True, container='FlowSolution#EndOfRun#Absolute',
        ReferenceFrame='absolute', secondOrderRestart=secondOrderRestart)
    
    for base in I.getBases(t):
        # Create GlobalConvergenceHistory to follow convergence in the OUTPUT8TREE during and at the end of simulation
        # see https://elsa.onera.fr/issues/9703
        GlobalConvergenceHistory = I.createNode('GlobalConvergenceHistory', 'UserDefinedData_t', value=0, parent=base)
        I.createNode('NormDefinitions', 'Descriptor_t', value='ConvergenceHistory', parent=GlobalConvergenceHistory)
        J.set(GlobalConvergenceHistory, '.Solver#Output', period=1, writingmode=0, var='residual_cons residual_turb')


def addSurfacicExtractions(t, ReferenceValues, elsAkeysModel, BCExtractions={},
        add_time_average=False):
    '''
    Include surfacic extraction information to CGNS tree using information
    contained in dictionaries **ReferenceValues** and **elsAkeysModel**.

    Parameters
    ----------

        t : PyTree
            prepared grid as produced by :py:func:`prepareMesh4ElsA` function.

            .. note:: tree **t** is modified

        ReferenceValues : dict
            dictionary as produced by :py:func:`computeReferenceValues` function

        elsAkeysModel : dict
            dictionary as produced by :py:func:`getElsAkeysModel` function

        BCExtractions : dict
            dictionary to indicate variables to extract. Each key corresponds to
            a ``BCType`` or a pattern to search in BCType of each FamilyBC (for
            instance, *BCInflow* matches *BCInflowSubsonic* and
            *BCInflowSupersonic*).
            The value associated to each key is a :py:class:`list` of
            :py:class:`str` corresponding to variable names to extract (in elsA
            convention).

        add_time_average : bool
            if :py:obj:`True`, include additional ``.Solver#Output#Average`` 
            extraction node for time-averaging fields at boundary-conditions.
    '''

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
    BCWallKeysDefault = dict()
    BCWallKeysDefault.update(BCKeys)
    BCWallKeysDefault.update(dict(
        delta_compute = elsAkeysModel['delta_compute'],
        vortratiolim  = elsAkeysModel['vortratiolim'],
        shearratiolim = elsAkeysModel['shearratiolim'],
        pressratiolim = elsAkeysModel['pressratiolim'],
        pinf          = ReferenceValues['Pressure'],
        torquecoeff   = 1.0,
        xtorque       = 0.0,
        ytorque       = 0.0,
        ztorque       = 0.0,
        writingframe  = 'relative', # absolute incompatible with unstructured mesh
        geomdepdom    = 2,  # see #8127#note-26
        delta_cell_max= 300,
    ))
    
    FamilyNodes = I.getNodesFromType2(t, 'Family_t')
    for ExtractBCType, ExtractVariablesListDefault in BCExtractions.items():
        for FamilyNode in FamilyNodes:
            BCWallKeys = copy.deepcopy(BCWallKeysDefault)
            ExtractVariablesList = copy.deepcopy(ExtractVariablesListDefault)
            FamilyName = I.getName( FamilyNode )
            BCType = getFamilyBCTypeFromFamilyBCName(t, FamilyName)
            if not BCType: continue

            if ExtractBCType in BCType:
                if ExtractBCType != BCType and BCType in BCExtractions:
                    # There is a more specific ExtractBCType
                    continue

                if 'BCWall' in BCType:

                    for zone in I.getZones(t):
                        if I.getZoneType(zone) == 2: # unstructured zone
                            # Remove extraction of bl_quantities, see https://elsa-e.onera.fr/issues/6479
                            var2remove = ['bl_quantities_2d', 'bl_quantities_3d', 'bl_ue_vector']
                            for var in var2remove:
                                if var in ExtractVariablesList:
                                    ExtractVariablesList.remove(var)
                            break

                    if 'Inviscid' in BCType:
                        ViscousKeys = ['bl_quantities_2d', 'bl_quantities_3d', 'bl_ue_vector',
                            'yplusmeshsize', 'frictionvector']
                        for vk in ViscousKeys:
                            try:
                                ExtractVariablesList.remove(vk)
                            except ValueError:
                                pass

                        param2remove = ['geomdepdom','delta_cell_max','delta_compute',
                            'vortratiolim','shearratiolim','pressratiolim']
                        for p in param2remove:
                            try:
                                BCWallKeys.pop(p)
                            except ValueError:
                                pass
                    else:
                        TransitionMode = ReferenceValues['TransitionMode']

                        if TransitionMode == 'NonLocalCriteria-LSTT':
                            extraVariables = ['intermittency', 'clim', 'how', 'origin',
                                'lambda2', 'turb_level', 'n_tot_ag', 'n_crit_ag',
                                'r_tcrit_ahd', 'r_theta_t1', 'line_status', 'crit_indicator']
                            ExtractVariablesList.extend(extraVariables)

                        elif TransitionMode == 'Imposed':
                            extraVariables = ['intermittency', 'clim']
                            ExtractVariablesList.extend(extraVariables)

                if ExtractVariablesList != []:
                    varDict = dict(var=' '.join(ExtractVariablesList))
                    print('setting .Solver#Output to FamilyNode '+FamilyNode[0])
                    if 'BCWall' in BCType:
                        BCWallKeys.update(varDict)
                        params = BCWallKeys
                        J.set(FamilyNode, '.Solver#Output',**BCWallKeys)
                    else:
                        BCKeys.update(varDict)
                        params = BCKeys
                    J.set(FamilyNode, '.Solver#Output',**params)
                    if add_time_average:
                        print('setting .Solver#Output#Average to FamilyNode '+FamilyNode[0])
                        avg_params = dict(average='time',
                                          period_init='inactive')

                        avg_params.update(params)
                        J.set(FamilyNode, '.Solver#Output#Average',**avg_params)

                else:
                    raise ValueError('did not added anything since:\nExtractVariablesList=%s'%str(ExtractVariablesList))

def addFieldExtractions(t, ReferenceValues, extractCoords=False,
        includeAdditionalExtractions=True, container='FlowSolution#EndOfRun',
        ReferenceFrame='relative', add_time_average=False, secondOrderRestart=False):
    '''
    Include fields extraction information to CGNS tree using
    information contained in dictionary **ReferenceValues**.

    Parameters
    ----------

        t : PyTree
            prepared grid as produced by :py:func:`prepareMesh4ElsA` function.

            .. note:: tree **t** is modified

        ReferenceValues : dict
            dictionary as produced by :py:func:`computeReferenceValues` function

        extractCoords : bool
            if :py:obj:`True`, then create a ``FlowSolution`` container named
            ``FlowSolution#EndOfRun#Coords`` to perform coordinates extraction.

        includeAdditionalExtractions : bool
            if :py:obj:`True`, will include fields listed in in 
            ReferenceValues['FieldsAdditionalExtractions']

        container : str
            name of the container of the field extraction

        ReferenceFrame : str
            ``'absolute'`` or ``'relative'``

        add_time_average : bool
            if :py:obj:`True`, include additional ``FlowSolution#Average`` for 
            time-averaged field extraction
        
        secondOrderRestart : bool
            if :py:obj:`True`, activate the saving of the flow solution at the
            penultimate iteration to allow a second order restart.

    '''

    Fields2Extract = ReferenceValues['Fields'][:] 
    if includeAdditionalExtractions: Fields2Extract += ReferenceValues['FieldsAdditionalExtractions']
    I._rmNodesByName(t, container)
    for zone in I.getZones(t):
        if extractCoords:
            EoRnode = I.createNode('FlowSolution#EndOfRun#Coords', 'FlowSolution_t',
                                    parent=zone)
            I.createNode('GridLocation','GridLocation_t', value='Vertex', parent=EoRnode)
            for fieldName in ('CoordinateX', 'CoordinateY', 'CoordinateZ'):
                I.createNode(fieldName, 'DataArray_t', value=None, parent=EoRnode)
            J.set(EoRnode, '.Solver#Output',
                  period=1,
                  writingmode=2,
                  writingframe='absolute')

        EoRnode = I.createNode(container, 'FlowSolution_t', parent=zone)
        I.createNode('GridLocation','GridLocation_t', value='CellCenter', parent=EoRnode)
        for fieldName in Fields2Extract:
            I.createNode(fieldName, 'DataArray_t', value=None, parent=EoRnode)       
        SolverOutputKeys = dict(
            period=1,
            writingmode=2,
            writingframe=ReferenceFrame)
        if secondOrderRestart:
            SolverOutputKeys['exact_restart'] = 'active'
        J.set(EoRnode, '.Solver#Output', **SolverOutputKeys)

        if add_time_average:
            Fields2Extract = ReferenceValues['Fields'] + \
                             ReferenceValues['FieldsAdditionalExtractions']

            EoRnode = I.createNode('FlowSolution#Average', 'FlowSolution_t',
                                    parent=zone)
            I.createNode('GridLocation','GridLocation_t', value='CellCenter', parent=EoRnode)
            for fieldName in Fields2Extract:
                I.createNode(fieldName, 'DataArray_t', value=None, parent=EoRnode)
            J.set(EoRnode, '.Solver#Output',
                period=1,
                writingmode=2,
                writingframe=ReferenceFrame,
                average='time',
                period_init='inactive')


def addGoverningEquations(t, dim=3):
    '''
    Add the nodes corresponding to `newFlowEquationSet_t`

    Parameters
    ----------

        t : PyTree
            prepared grid as produced by :py:func:`prepareMesh4ElsA` function.

            .. note:: tree **t** is modified


        dim : int
            dimension of the equations (``2`` or ``3``)

    '''
    I._rmNodesByType(t, 'newFlowEquationSet_t')
    for b in I.getBases(t):
        FES_n = I.newFlowEquationSet(parent=b)
        I.newGoverningEquations(value='NSTurbulent',parent=FES_n)
        I.createNode('EquationDimension', 'EquationDimension_t',
                      value=dim, parent=FES_n)

def addElsAKeys2CGNS(t, AllElsAKeys):
    '''
    Include node ``.Solver#Compute`` , where elsA keys are set in full CGNS mode.

    Parameters
    ----------

        t : PyTree
            main CGNS tree

            .. note:: tree **t** is modified

        AllElsAKeys : :py:class:`list` of :py:class:`dict`
            include all dictinaries of elsA keys to be set into ``.Solver#Compute``

    '''
    I._rmNodesByName(t, '.Solver#Compute')
    AllComputeModels = dict()
    for ElsAKeys in AllElsAKeys: AllComputeModels.update(ElsAKeys)
    for b in I.getBases(t): J.set(b, '.Solver#Compute', **AllComputeModels)

def initializeFlowSolution(t, Initialization, ReferenceValues, secondOrderRestart=False):
    '''
    Initialize the flow solution in tree **t**.

    Parameters
    ----------

        t : PyTree
            preprocessed tree as performed by :py:func:`prepareMesh4ElsA`

        Initialization : dict
            dictionary defining the type of initialization, using the key
            **method**. This latter is mandatory and should be one of the
            following:

            * **method** = :py:obj:`None` : the Flow Solution is not initialized.

            * **method** = ``'uniform'`` : the Flow Solution is initialized uniformly
              using the **ReferenceValues**.

            * **method** = ``'copy'`` : the Flow Solution is initialized by copying
              the FlowSolution container of another file. The file path is set by
              using the key **file**. The container might be set with the key
              **container** (``'FlowSolution#Init'`` by default).

            * **method** = ``'interpolate'`` : the Flow Solution is initialized by
              interpolating the FlowSolution container of another file. The file
              path is set by using the key **file**. The container might be set
              with the key **container** (``'FlowSolution#Init'`` by default).

            Default method is ``'uniform'``.

        ReferenceValues : dict
            dictionary as got from :py:func:`computeReferenceValues`

        secondOrderRestart : bool
            if :py:obj:`True`, duplicate the node ``'FlowSolution#Init'`` into a
            new node ``'FlowSolution#Init-1'`` to mimic the expected
            behavior for the next restart.

    '''
    if not 'container' in Initialization:
        Initialization['container'] = 'FlowSolution#Init'

    if Initialization['method'] is None:
        pass
    elif Initialization['method'] == 'uniform':
        print(J.CYAN + 'Initialize FlowSolution with uniform reference values' + J.ENDC)
        initializeFlowSolutionFromReferenceValues(t, ReferenceValues)
    elif Initialization['method'] == 'interpolate':
        print(J.CYAN + 'Initialize FlowSolution by interpolation from {}'.format(Initialization['file']) + J.ENDC)
        initializeFlowSolutionFromFileByInterpolation(t, ReferenceValues,
            Initialization['file'], container=Initialization['container'])
    elif Initialization['method'] == 'copy':
        print(J.CYAN + 'Initialize FlowSolution by copy of {}'.format(Initialization['file']) + J.ENDC)
        if not 'keepTurbulentDistance' in Initialization:
            Initialization['keepTurbulentDistance'] = False
        initializeFlowSolutionFromFileByCopy(t, ReferenceValues, Initialization['file'],
            container=Initialization['container'],
            keepTurbulentDistance=Initialization['keepTurbulentDistance'])
    else:
        raise Exception(J.FAIL+'The key "method" of the dictionary Initialization is mandatory'+J.ENDC)

    for zone in I.getZones(t):
        if not I.getNodeFromName1(zone, 'FlowSolution#Init'):
            MSG = 'FlowSolution#Init is missing in zone {}'.format(I.getName(zone))
            raise ValueError(J.FAIL + MSG + J.ENDC)
    
    if secondOrderRestart:
        for zone in I.getZones(t):
            FSnode = I.copyTree(I.getNodeFromName1(zone, 'FlowSolution#Init'))
            I.setName(FSnode, 'FlowSolution#Init-1')
            I.addChild(zone, FSnode)
        

def initializeFlowSolutionFromReferenceValues(t, ReferenceValues):
    '''
    Invoke ``FlowSolution#Init`` fields using information contained in
    ``ReferenceValue['ReferenceState']`` and ``ReferenceValues['Fields']``.

    .. note:: This is equivalent as a *uniform* initialization of flow.

    Parameters
    ----------

        t : PyTree
            main CGNS PyTree where fields are going to be initialized

            .. note:: tree **t** is modified

        ReferenceValues : dict
            as produced by :py:func:`computeReferenceValues`

    '''
    print('invoking FlowSolution#Init with uniform fields using ReferenceState')
    I._renameNode(t,'FlowSolution#Centers','FlowSolution#Init')
    FieldsNames = ReferenceValues['Fields']
    I.__FlowSolutionCenters__ = 'FlowSolution#Init'
    for i in range(len(ReferenceValues['ReferenceState'])):
        FieldName = FieldsNames[i]
        FieldValue= ReferenceValues['ReferenceState'][i]
        C._initVars(t,'centers:%s'%FieldName,FieldValue)

    # TODO : This should not be required.
    # FieldsNamesAdd = ReferenceValues['FieldsAdditionalExtractions'].split(' ')
    # for fn in FieldsNamesAdd:
    #     try:
    #         ValueOfField = ReferenceValues[fn]
    #     except KeyError:
    #         ValueOfField = 1.0
    #     C._initVars(t,'centers:%s'%fn,ValueOfField)
    I.__FlowSolutionCenters__ = 'FlowSolution#Centers'

    FlowSolInit = I.getNodesFromName(t,'FlowSolution#Init')
    I._rmNodesByName(FlowSolInit, 'ChimeraCellType')

def initializeFlowSolutionFromFileByInterpolation(t, ReferenceValues, sourceFilename, container='FlowSolution#Init'):
    '''
    Initialize the flow solution of **t** from the flow solution in the file
    **sourceFilename**.
    Modify the tree **t** in-place.

    Parameters
    ----------

        t : PyTree
            Tree to initialize

        ReferenceValues : dict
            as produced by :py:func:`computeReferenceValues`

        sourceFilename : str
            Name of the source file for the interpolation.

        container : str
            Name of the ``'FlowSolution_t'`` node use for the interpolation.
            Default is 'FlowSolution#Init'

    '''
    sourceTree = C.convertFile2PyTree(sourceFilename)
    OLD_FlowSolutionCenters = I.__FlowSolutionCenters__
    I.__FlowSolutionCenters__ = container
    sourceTree = C.extractVars(sourceTree, ['centers:{}'.format(var) for var in ReferenceValues['Fields']])

    I._rmNodesByType(sourceTree, 'BCDataSet_t')
    I._rmNodesByNameAndType(sourceTree, '*EndOfRun*', 'FlowSolution_t')
    P._extractMesh(sourceTree, t, mode='accurate', extrapOrder=0)
    if container != 'FlowSolution#Init':
        I._rmNodesByName(t, 'FlowSolution#Init')
        I.renameNode(t, container, 'FlowSolution#Init')
    I.__FlowSolutionCenters__ = OLD_FlowSolutionCenters

def initializeFlowSolutionFromFileByCopy(t, ReferenceValues, sourceFilename,
        container='FlowSolution#Init', keepTurbulentDistance=False):
    '''
    Initialize the flow solution of **t** by copying the flow solution in the file
    **sourceFilename**.
    Modify the tree **t** in-place.

    Parameters
    ----------

        t : PyTree
            Tree to initialize

        ReferenceValues : dict
            as produced by :py:func:`computeReferenceValues`

        sourceFilename : str
            Name of the source file.

        container : str
            Name of the ``'FlowSolution_t'`` node to copy.
            Default is 'FlowSolution#Init'

        keepTurbulentDistance : bool
            if :py:obj:`True`, copy also fields ``'TurbulentDistance'`` and
            ``'TurbulentDistanceIndex'``.

            .. danger::
                The restarted simulation must be submitted with the same
                CPU distribution that the previous one ! It is due to the field
                ``'TurbulentDistanceIndex'`` that indicates the index of the
                nearest wall, and this index varies with the distribution.

    '''
    sourceTree = C.convertFile2PyTree(sourceFilename)
    OLD_FlowSolutionCenters = I.__FlowSolutionCenters__
    I.__FlowSolutionCenters__ = container
    varNames = copy.deepcopy(ReferenceValues['Fields'])
    if keepTurbulentDistance:
        varNames += ['TurbulentDistance', 'TurbulentDistanceIndex']

    sourceTree = C.extractVars(sourceTree, ['centers:{}'.format(var) for var in varNames])

    for base in I.getBases(t):
        basename = I.getName(base)
        for zone in I.getNodesFromType1(base, 'Zone_t'):
            zonename = I.getName(zone)
            zonepath = '{}/{}'.format(basename, zonename)
            FSpath = '{}/{}'.format(zonepath, container)
            FlowSolutionInSourceTree = I.getNodeFromPath(sourceTree, FSpath)
            if FlowSolutionInSourceTree:
                I._rmNodesByNameAndType(zone, container, 'FlowSolution_t')
                I._append(t, FlowSolutionInSourceTree, zonepath)
            else:
                ERROR_MSG = 'The node {} is not found in {}'.format(FSpath, sourceFilename)
                raise Exception(J.FAIL+ERROR_MSG+J.ENDC)

    if container != 'FlowSolution#Init':
        I.renameNode(t, container, 'FlowSolution#Init')

    I.__FlowSolutionCenters__ = OLD_FlowSolutionCenters

def writeSetup(AllSetupDictionaries, setupFilename='setup.py'):
    '''
    Write ``setup.py`` file using a dictionary of dictionaries containing setup
    information

    Parameters
    ----------

        AllSetupDictionaries : dict
            contains all dictionaries to be included in ``setup.py``

        setupFilename : str
            name of setup file

    '''

    Lines = '#!/usr/bin/python\n'
    Lines+= "'''\nMOLA %s setup.py file automatically generated in PREPROCESS\n"%MOLA.__version__
    Lines+= "Path to MOLA: %s\n"%MOLA.__MOLA_PATH__
    Lines+= "Commit SHA: %s\n'''\n\n"%MOLA.__SHA__

    for SetupDict in AllSetupDictionaries:
        Lines+=SetupDict+"="+pprint.pformat(AllSetupDictionaries[SetupDict])+"\n\n"

    with open(setupFilename,'w') as f: f.write(Lines)

    try: os.remove(setupFilename+'c')
    except: pass

def writeSetupFromModuleObject(setup, setupFilename='setup.py'):
    '''
    Write ``setup.py`` file using "setup" module object as got from an import
    operation.

    Parameters
    ----------

        setup : module object
            as got from instruction

            >>> import setup

        setupFilename : str
            name of the new setup file to write

    '''
    Lines = '#!/usr/bin/python\n'
    Lines+= "'''\nMOLA %s setup.py file automatically generated in PREPROCESS\n"%MOLA.__version__
    Lines+= "Path to MOLA: %s\n"%MOLA.__MOLA_PATH__
    Lines+= "Commit SHA: %s\n'''\n\n"%MOLA.__SHA__

    for SetupItem in dir(setup):
        if not SetupItem.startswith('_'):
            Lines+=SetupItem+"="+pprint.pformat(getattr(setup, SetupItem))+"\n\n"

    with open(setupFilename,'w') as f: f.write(Lines)

    try: os.remove(setupFilename+'c')
    except: pass

def addReferenceState(t, FluidProperties, ReferenceValues):
    '''
    Add ``ReferenceState`` node to CGNS using user-provided conditions

    Parameters
    ----------

        t : PyTree
            main CGNS tree where ReferenceState will be set

            .. note:: tree **t** is modified

        FluidProperties : dict
            as produced by :py:func:`computeFluidProperties`

        ReferenceValues : dict
            as produced by :py:func:`computeReferenceValues`

    '''
    bases = I.getBases(t)
    # BEWARE: zip behavior change between Python 2 and 3.
    # This strategy cannot apply with Python 3.
    # RefState = zip(ReferenceValues['Fields'].split(' '),
    #                ReferenceValues['ReferenceState'])
    RefState = []
    for f, rs in zip(ReferenceValues['Fields'], ReferenceValues['ReferenceState']):
        RefState += [[f,rs]]

    for i in ('Reynolds','Mach','Pressure','Temperature'):
        RefState += [[i,ReferenceValues[i]]]
    RefState += [
        ['Cv',   FluidProperties['cv']], # Beware of case difference
        ['Gamma',FluidProperties['Gamma']],
        ['Mus',  FluidProperties['SutherlandViscosity']],
        ['Cs',   FluidProperties['SutherlandConstant']],
        ['Ts',   FluidProperties['SutherlandTemperature']],
        ['Pr',   FluidProperties['Prandtl']],
                ]
    for b in bases:
        J._addSetOfNodes(b, 'ReferenceState',
         RefState,
         type1='ReferenceState_t',
         type2='DataArray_t')

def removeEmptyOversetData(t, silent=True):
    '''
    Remove spurious 0-length lists or numpy arrays created during overset
    preprocessing.

    Parameters
    ----------

        t : PyTree
            main CGNS to clean

            .. note:: tree **t** is modified

        silent : bool
            if :py:obj:`False`, then it prints information on the
            performed operations

    '''
    OversetNodeNames = ('InterpolantsDonor',
                        'InterpolantsType',
                        'InterpolantsVol',
                        'FaceInterpolantsDonor',
                        'FaceInterpolantsType',
                        'FaceInterpolantsVol',
                        'FaceDirection',
                        'PointListExtC',
                        'PointListDonor',
                        'FaceListExtC',
                        'FaceListDonor',
                        'PointList',
                        'PointListDonorExtC',
                        'FaceList',
                        'FaceListDonorExtC',
                        )

    print('cleaning empty chimera nodes...')
    OPL_ns = I.getNodesFromName(t,'OrphanPointList')
    for opl in OPL_ns:
        # ID_node, _ = I.getParentOfNode(t, opl)
        # print(J.WARN+'removing %s'%opl[0]+J.ENDC)
        I.rmNode(t,opl)

    for zone in I.getZones(t):
        for OversetNodeName in OversetNodeNames:
            OversetNodes = I.getNodesFromName(zone, OversetNodeName)
            for OversetNode in OversetNodes:
                OversetValue = OversetNode[1]
                if OversetValue is None or len(OversetValue)==0:
                    # if not silent:
                    #     STR = J.WARN, zone[0], OversetNode[0], J.ENDC
                    #     print('%szone %s removing empty overset %s node%s'%STR)
                    I.rmNode(t, OversetNode)

def getFlowDirections(AngleOfAttackDeg, AngleOfSlipDeg, YawAxis, PitchAxis):
    '''
    Compute the main flow directions from angle of attack and slip and aircraft
    yaw and pitch axis. The resulting directions can be used to impose inflow
    conditions and to compute aero-forces (Drag, Side, Lift) by projection of
    cartesian (X, Y, Z) forces onto the corresponding Flow Direction.

    Parameters
    ----------

        AngleOfAttackDeg : float
            Angle-of-attack in degree. A positive
            angle-of-attack has an analogous impact as making a rotation of the
            aircraft around the **PitchAxis**, and this will likely contribute in
            increasing the Lift force component.

        AngleOfSlipDeg : float
            Angle-of-attack in degree. A positive
            angle-of-slip has an analogous impact as making a rotation of the
            aircraft around the **YawAxis**, and this will likely contribute in
            increasing the Side force component.

        YawAxis : array of 3 :py:class:`float`
            Vector indicating the Yaw-axis of the
            aircraft, which commonly points towards the top side of the aircraft.
            A positive rotation around **YawAxis** is commonly produced by applying
            left-pedal rudder (rotation towards the left side of the aircraft).
            This left-pedal rudder application will commonly produce a positive
            angle-of-slip and thus a positive side force.

        PitchAxis : array of 3 :py:class:`float`
            Vector indicating the Pitch-axis of the
            aircraft, which commonly points towards the right side of the
            aircraft. A positive rotation around **PitchAxis** is commonly produced
            by pulling the elevator, provoking a rotation towards the top side
            of the aircraft. By pulling the elevator, a positive angle-of-attack
            is created, which commonly produces an increase of Lift force.

    Returns
    -------

        DragDirection : array of 3 :py:class:`float`
            Vector indicating the main flow
            direction. The Drag force is obtained by projection of the absolute
            (X, Y, Z) forces onto this vector. The inflow vector for reference
            state is also obtained by projection of the momentum magnitude onto
            this vector.

        SideDirection : array of 3 :py:class:`float`
            Vector normal to the main flow
            direction pointing towards the Side direction. The Side force is
            obtained by projection of the absolute (X, Y, Z) forces onto this
            vector.

        LiftDirection : array of 3 :py:class:`float`
            Vector normal to the main flow
            direction pointing towards the Lift direction. The Lift force is
            obtained by projection of the absolute (X, Y, Z) forces onto this
            vector.
    '''

    def getDirectionFromLine(line):
        x,y,z = J.getxyz(line)
        Direction = np.array([x[-1]-x[0], y[-1]-y[0], z[-1]-z[0]])
        Direction /= np.sqrt(Direction.dot(Direction))
        return Direction

    # Yaw axis must be exact
    YawAxis    = np.array(YawAxis, dtype=np.float64)
    YawAxis   /= np.sqrt(YawAxis.dot(YawAxis))

    # Pitch axis may be approximate
    PitchAxis  = np.array(PitchAxis, dtype=np.float64)
    PitchAxis /= np.sqrt(PitchAxis.dot(PitchAxis))

    # Roll axis is inferred
    RollAxis  = np.cross(PitchAxis, YawAxis)
    RollAxis /= np.sqrt(RollAxis.dot(RollAxis))

    # correct Pitch axis
    PitchAxis = np.cross(YawAxis, RollAxis)
    PitchAxis /= np.sqrt(PitchAxis.dot(PitchAxis))

    # FlowLines are used to infer the final flow direction
    DragLine = D.line((0,0,0),(1,0,0),2);DragLine[0]='Drag'
    SideLine = D.line((0,0,0),(0,1,0),2);SideLine[0]='Side'
    LiftLine = D.line((0,0,0),(0,0,1),2);LiftLine[0]='Lift'
    FlowLines = [DragLine, SideLine, LiftLine]

    # Put FlowLines in Aircraft's frame
    zero = (0,0,0)
    InitialFrame =  [       [1,0,0],         [0,1,0],       [0,0,1]]
    AircraftFrame = [list(RollAxis), list(PitchAxis), list(YawAxis)]
    T._rotate(FlowLines, zero, InitialFrame, AircraftFrame)
    DragDirection = getDirectionFromLine(DragLine)
    SideDirection = getDirectionFromLine(SideLine)
    LiftDirection = getDirectionFromLine(LiftLine)

    # Apply Flow angles with respect to Airfraft's frame
    T._rotate(FlowLines, zero, SideDirection, -AngleOfAttackDeg)
    DragDirection = getDirectionFromLine(DragLine)
    SideDirection = getDirectionFromLine(SideLine)
    LiftDirection = getDirectionFromLine(LiftLine)

    T._rotate(FlowLines, zero, LiftDirection,  AngleOfSlipDeg)
    DragDirection = getDirectionFromLine(DragLine)
    SideDirection = getDirectionFromLine(SideLine)
    LiftDirection = getDirectionFromLine(LiftLine)

    return DragDirection, SideDirection, LiftDirection

def getMeshInfoFromBaseName(baseName, InputMeshes):
    '''
    .. note:: this is a private-level function.

    Used to pick the right InputMesh item associated to the
    **baseName** value.

    Parameters
    ----------

        baseName : str
            name of the base

        InputMeshes : :py:class:`list` of :py:class:`dict`
            same input as introduced in :py:func:`prepareMesh4ElsA`

    Returns
    -------

        meshInfo : dict
            item contained in **InputMeshes** with same base name as requested
            by **baseName**
    '''
    for meshInfo in InputMeshes:
        if meshInfo['baseName'] == baseName:
            return meshInfo

def getFamilyBCTypeFromFamilyBCName(t, FamilyBCName):
    '''
    Get the *BCType* of BCs defined by a given family BC name.

    Parameters
    ----------

        t : PyTree
            main CGNS tree

        FamilyBCName : str
            requested name of the *FamilyBC*

    Returns
    -------

        BCType : str
            the resulting *BCType*. Returns:py:obj:`None` if **FamilyBCName** is not
            found
    '''
    FamilyNode = I.getNodeFromNameAndType(t, FamilyBCName, 'Family_t')
    if not FamilyNode: return

    FamilyBCNode = I.getNodeFromType1(FamilyNode, 'FamilyBC_t')
    if not FamilyBCNode: return

    FamilyBCNodeType = I.getValue(FamilyBCNode)
    if FamilyBCNodeType != 'UserDefined': return FamilyBCNodeType

    SolverBC = I.getNodeFromName1(FamilyNode,'.Solver#BC')
    if SolverBC:
        SolverBCType = I.getNodeFromName1(SolverBC,'type')
        if SolverBCType:
            BCType = I.getValue(SolverBCType)
            return BCType

    SolverOverlap = I.getNodeFromName1(FamilyNode,'.Solver#Overlap')
    if SolverOverlap: return 'BCOverlap'

    BCnodes = I.getNodesFromType(t, 'BC_t')
    for BCnode in BCnodes:
        FamilyNameNode = I.getNodeFromName1(BCnode, 'FamilyName')
        if not FamilyNameNode: continue

        FamilyNameValue = I.getValue( FamilyNameNode )
        if FamilyNameValue == FamilyBCName:
            BCType = I.getValue( BCnode )
            if BCType != 'FamilySpecified': return BCType
            break


def hasBCOverlap(t):
    '''
    Determines whether the input tree **t** employs Overlap boundary-conditions
    or not.

    Parameters
    ----------

        t : PyTree
            main CGNS where Overlap BC may exist

    Returns
    -------

        Result : bool
            :py:obj:`True` if BC Overlap exist on **t**, :py:obj:`False` otherwise
    '''

    hasBCOfTypeOverlap = bool(C.extractBCOfType(t, 'BCOverlap'))
    if hasBCOfTypeOverlap: return True
    hasSolverOverlap = bool(I.getNodeFromName(t,'.Solver#Overlap'))

    return hasSolverOverlap

def groupUserDefinedBCFamiliesByName(t):
    '''
    It is an extension of ``Converter.PyTree.groupBCByBCType``.
    This function follows ``UserDefined`` *BCType* by its actual defining value
    located at ``FamilyBC_t``, and use its value to group Families.

    .. note:: this distinction does not apply to *BCOverlap*.

    Parameters
    ----------

        t : PyTree
            main CGNS tree where ``UserDefined`` families are to be grouped
    '''
    for b in I.getBases(t):
        FamilyBCName2Type = C.getFamilyBCNamesDict(b)
        for FamilyName in FamilyBCName2Type:
            FamilyBCType = FamilyBCName2Type[FamilyName]
            if FamilyBCType == 'UserDefined':
                BCType = getFamilyBCTypeFromFamilyBCName(b, FamilyName)
                if BCType == 'BCOverlap': continue
                I._groupBCByBCType(b, btype=BCType, name=FamilyName)

def adapt2elsA(t, InputMeshes):
    '''
    This function is similar to :py:func:`Converter.elsAProfile.convert2elsAxdt`,
    except that it employs **InputMeshes** information in order to precondition
    unnecessary operations. It also cleans spurious 0-length data CGNS nodes that
    can be generated during overset preprocessing.
    '''

    if hasAnyNearMatch(t, InputMeshes):
        print(J.CYAN+'adapting NearMatch to elsA'+J.ENDC)
        EP._adaptNearMatch(t)

    if hasAnyOversetData(InputMeshes):
        print('adapting overset data to elsA...')
        EP._overlapGC2BC(t)
        EP._rmGCOverlap(t)
        EP._fillNeighbourList(t, sameBase=0)
        EP._prefixDnrInSubRegions(t)
        removeEmptyOversetData(t, silent=False)

    forceFamilyBCasFamilySpecified(t) # https://elsa.onera.fr/issues/10928
    I._createElsaHybrid(t, method=1)

def forceFamilyBCasFamilySpecified(t):
    # https://elsa.onera.fr/issues/10928
    for base in I.getBases(t):
        for zone in I.getZones(base):
            for ZoneBC in I.getNodesFromType1(zone,'ZoneBC_t'):
                for BC in I.getNodesFromType1(ZoneBC,'BC_t'):
                    if I.getNodeFromType1(BC,'FamilyName_t') is not None:
                        I.setValue(BC,'FamilySpecified')
                        continue

def hasAnyNearMatch(t, InputMeshes):
    '''
    Determine if configuration has a connectivity of type ``NearMatch``.

    Parameters
    ----------

        t : PyTree
            input tree to test

        InputMeshes : :py:class:`list` of :py:class:`dict`
            as described by :py:func:`prepareMesh4ElsA`

    Returns
    -------

        bool : bool
            :py:obj:`True` if has ``NearMatch`` connectivity. :py:obj:`False` otherwise.
    '''
    for meshInfo in InputMeshes:
        try: Connection = meshInfo['Connection']
        except KeyError: continue

        for ConnectionInfo in Connection:
            isNearMatch = ConnectionInfo['type'] == 'NearMatch'
            if isNearMatch: return True
    
    for base in I.getBases(t):
        for zone in I.getZones(base):
            for zgc in I.getNodesFromType1(zone,'ZoneGridConnectivity_t'):
                for gc in I.getNodesFromType1(zgc, 'GridConnectivity_t'):
                    gct = I.getNodeFromType1(gc, 'GridConnectivityType_t')
                    if gct:
                        gct_value = I.getValue(gct)
                        if isinstance(gct_value,str) and gct_value == 'Abutting':
                            return True

    return False

def hasAnyPeriodicMatch(InputMeshes):
    '''
    Determine if at least one item in **InputMeshes** has a connectivity of
    type ``PeriodicMatch``.

    Parameters
    ----------

        InputMeshes : :py:class:`list` of :py:class:`dict`
            as described by :py:func:`prepareMesh4ElsA`

    Returns
    -------

        bool : bool
            :py:obj:`True` if has ``PeriodicMatch`` connectivity. :py:obj:`False` otherwise.
    '''

    for meshInfo in InputMeshes:
        try: Connection = meshInfo['Connection']
        except KeyError: continue

        for ConnectionInfo in Connection:
            isPeriodicMatch = ConnectionInfo['type'] == 'PeriodicMatch'
            if isPeriodicMatch: return True

    return False

def hasAnyOversetData(InputMeshes):
    '''
    Determine if at least one item in **InputMeshes** has an overset kind of
    assembly.

    Parameters
    ----------

        InputMeshes : :py:class:`list` of :py:class:`dict`
            as described by :py:func:`prepareMesh4ElsA`

    Returns
    -------

        bool : bool
            :py:obj:`True` if has overset assembly. :py:obj:`False` otherwise.
    '''
    for meshInfo in InputMeshes:
        if 'OversetOptions' in meshInfo:
            return True
    return False


def getStructuredZones(t):
    zones = []
    for z in I.getZones(t):
        if I.getZoneType(z) == 1:
            zones += [ z ]
    return zones

def getUnstructuredZones(t):
    zones = []
    for z in I.getZones(t):
        if I.getZoneType(z) == 2:
            zones += [ z ]
    return zones


def hasAnyUnstructuredZones(t):
    '''
    Determine if at least one zone in **t** is unstructured.

    Parameters
    ----------

        t : PyTree
            input tree to test

    Returns
    -------

        bool : bool
            :py:obj:`True` if at least one zone in **t** is unstructured,
            :py:obj:`False` otherwise.
    '''
    # Test if there are unstructured zones in mesh
    IsUnstructured = False
    for zone in I.getZones(t):
        if I.getZoneType(zone) == 2: # unstructured zone
            IsUnstructured = True
            break
    return IsUnstructured

def getProc(t):
    '''
    Wrap of :py:func:`Distributor2.PyTree.getProc` function, such that the
    returned object is always a 1D numpy.array of :py:class:`int`.

    Parameters
    ----------

        t : PyTree, base, zone or list of zones

    Returns
    -------

        ProcArray : 1D numpy.ndarray
            proc number attributed to each zone of **t**
    '''
    return  np.array(D2.getProc(t), order='F', ndmin=1)

def autoMergeBCs(t, familyNames=None):
    '''
    Merge BCs that belong to the same family for all zones of a PyTree.
    This fonction work for either structured or unstructured mesh.

    Parameters
    ----------

        t : PyTree
            input tree (structured or unstructured)

        familyNames : :py:class:`list` of :py:obj:`None`
            restrict the merge operation to the listed family name(s). If
            :py:obj:`None`, merge zones for all BC families.
    '''
    for zone in I.getZones(t):
        if I.getZoneType(zone) == 1: # structured zone
            autoMergeBCsStructured(zone, familyNames)
        elif I.getZoneType(zone) == 2: # unstructured zone
            autoMergeBCsUnstructured(zone, familyNames)

def autoMergeBCsStructured(t, familyNames=None):
    '''
    Merge BCs that are contiguous, belong to the same family and are of the same
    type, for all zones of a PyTree

    Parameters
    ----------

        t : PyTree
            input tree

        familyNames : :py:class:`list` of :py:obj:`None`
            restrict the merge operation to the listed family name(s). If
            :py:obj:`None`, merge zones for all BC families.
    '''
    if familyNames is None: familyNames = J.getBCFamilies(t)

    def getBCInfo(bc):
        pt  = I.getNodeFromName(bc, 'PointRange')
        fam = I.getNodeFromName(bc, 'FamilyName')
        if not fam:
            fam = I.createNode('FamilyName', 'FamilyName_t', value='Unknown')
        return I.getName(bc), I.getValue(fam), pt

    def areContiguous(PointRange1, PointRange2):
        '''
        Check if subZone of the same block defined by PointRange1 and PointRange2
        are contiguous.

        Parameters
        ----------

            PointRange1 : PyTree
                PointRange (PyTree of type ``IndexRange_t``) of a ``BC_t`` node

            PointRange2 : PyTree
                Same as PointRange2

        Returns
        -------
            dimension : int
                an integer of value -1 if subZones are not contiguous, and of value
                equal to the direction along which subzone are contiguous else.

        '''
        assert I.getType(PointRange1) == 'IndexRange_t' \
            and I.getType(PointRange2) == 'IndexRange_t', \
            'Arguments are not IndexRange_t'

        pt1 = I.getValue(PointRange1)
        pt2 = I.getValue(PointRange2)
        if pt1.shape != pt2.shape:
            return -1
        spaceDim = pt1.shape[0]
        indSpace = 0
        MatchingDims = []
        for dim in range(spaceDim):
            if pt1[dim, 0] == pt2[dim, 0] and pt1[dim, 1] == pt2[dim, 1]:
                indSpace += 1
                MatchingDims.append(dim)
        if indSpace != spaceDim-1 :
            # matching dimensions should form a hyperspace of original space
            return -1

        for dim in [d for d in range(spaceDim) if d not in MatchingDims]:
            if pt1[dim][0] == pt2[dim][1] or pt2[dim][0] == pt1[dim][1]:
                return dim

        return -1

    for block in I.getNodesFromType(t, 'Zone_t'):
        if I.getNodeFromType(block, 'ZoneBC_t'):
            somethingWasMerged = True
            while somethingWasMerged : # recursively attempts to merge bcs until nothing possible is left
                somethingWasMerged = False
                bcs = I.getNodesFromType(block, 'BC_t')
                zoneBcsOut = I.copyNode(I.getNodeFromType(block, 'ZoneBC_t')) # a duplication of all BCs of current block
                mergedBcs = []
                for bc1 in bcs:
                    bcName1, famName1, pt1 = getBCInfo(bc1)
                    for bc2 in [b for b in bcs if b is not bc1]:
                        bcName2, famName2, pt2 = getBCInfo(bc2)
                        # check if bc1 and bc2 can be merged
                        mDim = areContiguous(pt1, pt2)
                        if bc1 not in mergedBcs and bc2 not in mergedBcs \
                            and mDim>=0 \
                            and famName1 == famName2 \
                            and famName1 in familyNames :
                            # does not check inward normal index, necessarily the same if subzones are contiguous
                            newPt = np.zeros(np.shape(pt1[1]),dtype=np.int32,order='F')
                            for dim in range(np.shape(pt1[1])[0]):
                                if dim != mDim :
                                    newPt[dim,0] = pt1[1][dim, 0]
                                    newPt[dim,1] = pt1[1][dim, 1]
                                else :
                                    newPt[dim,0] = min(pt1[1][dim, 0], pt2[1][dim, 0])
                                    newPt[dim,1] = max(pt1[1][dim, 1], pt2[1][dim, 1])
                            # new BC inheritates from the name of first BC
                            bc = I.createNode(bcName1, 'BC_t', bc1[1])
                            I.createChild(bc, pt1[0], 'IndexRange_t', value=newPt)
                            I.createChild(bc, 'FamilyName', 'FamilyName_t', value=famName1)
                            # TODO : include case with flow solution

                            I._rmNodesByName(zoneBcsOut, bcName1)
                            I._rmNodesByName(zoneBcsOut, bcName2)
                            I.addChild(zoneBcsOut, bc)
                            mergedBcs.append(bc1)
                            mergedBcs.append(bc2)
                            somethingWasMerged = True
                            # print('BCs {} and {} were merged'.format(bcName1, bcName2))

                block = I.rmNodesByType(block,'ZoneBC_t')
                I.addChild(block,zoneBcsOut)
                del(zoneBcsOut)

def autoMergeBCsUnstructured(t, familyNames=None):
    '''
    Merge BCs that belong to the same family for all zones of a PyTree

    Parameters
    ----------

        t : PyTree
            input tree

        familyNames : :py:class:`list` of :py:obj:`None`
            restrict the merge operation to the listed family name(s). If
            :py:obj:`None`, merge zones for all BC families.
    '''
    if familyNames is None: familyNames = J.getBCFamilies(t)

    for zone in I.getZones(t):
        for familyName in familyNames:
            BCs = C.getFamilyBCs(zone, familyName)
            if len(BCs) < 2:
                # Either no BC or just one BC, so no merge operation required
                continue
            ptlList = [I.getValue(I.getNodeFromType(BC, 'IndexArray_t')) for BC in BCs]
            newBC = I.copyTree(BCs[0])
            PointList = I.getNodeFromType(newBC, 'IndexArray_t')
            newPointListValue = np.concatenate(ptlList, axis=1)
            I.setValue(PointList, newPointListValue)
            zoneBC = I.getNodeFromType1(zone, 'ZoneBC_t')
            for BC in BCs: I.rmNode(zoneBC, BC)
            I.addChild(zoneBC, newBC)

def checkFamiliesInZonesAndBC(t, behavior='add'):
    '''
    Check that each zone and each BC is attached to a family (so there must be
    a ``FamilyName_t`` node). In case of family missing, raise an exception or
    add family depending on the value of parameter behavior.

    Parameters
    ----------

        t : PyTree
            PyTree to check

        behavior : str
            if ``'raise'``, then raise an exception. 
            If ``'add'``, then add the family zone tag as ``basename+'Zones'``.

            .. note::
                In all cases, FamilyBC without FamilyName_t raises an exception
    '''
    for base in I.getBases(t):
        if any([pattern in I.getName(base) for pattern in ['Numeca', 'meridional_base', 'tools_base']]):
            # Base specific to Autogrid
            continue
        # Check that each zone is attached to a family
        for zone in I.getZones(base):
            if not I.getNodeFromType1(zone, 'FamilyName_t'):
                FAILMSG = 'Each zone must be attached to a Family:\n'
                FAILMSG += 'Zone {} has no node of type FamilyName_t'.format(I.getName(zone))
                if behavior=='raise':
                    raise Exception(J.FAIL+FAILMSG+J.ENDC)
                elif behavior=='add':
                    newFamilyTag = base[0]+'Zones'
                    FAILMSG += '\nAdding tag '+newFamilyTag
                    print(J.WARN+FAILMSG+J.ENDC)
                    C._tagWithFamily(base, 'BaseZones')
                    C._addFamily2Base(base, 'BaseZones')
                    I._correctPyTree(base, level=7)
                else:
                    raise Exception(f'behavior {behavior} not recognized')
            # Check that each BC is attached to a family
            for bc in I.getNodesFromType(zone, 'BC_t'):
                if I.getValue(bc) in ['BCDegenerateLine', 'BCDegeneratePoint']:
                    continue
                if not I.getNodeFromType1(bc, 'FamilyName_t'):
                    FAILMSG = 'Each BC must be attached to a Family:\n'
                    FAILMSG += 'BC {} in zone {} has no node of type FamilyName_t'.format(I.getName(bc), I.getName(zone))
                    raise Exception(J.FAIL+FAILMSG+J.ENDC)

def extendListOfFamilies(FamilyNames):
    '''
    For each <NAME> in the list **FamilyNames**, add Name, name and NAME.
    '''
    ExtendedFamilyNames = copy.deepcopy(FamilyNames)
    for fam in FamilyNames:
        newNames = [fam.lower(), fam.upper(), fam.capitalize()]
        for name in newNames:
            if name not in ExtendedFamilyNames:
                ExtendedFamilyNames.append(name)
    return ExtendedFamilyNames

def computeDistance2Walls(t, WallFamilies=[], verbose=False, wallFilename=None):
    '''
    Compute the distance to the walls and add nodes ``'TurbulentDistance'`` and
    ``'TurbulentDistanceIndex'`` into ``'FlowSolution#Centers'``. This function
    works in-place and returns None. Boundary conditions must have a ``BCType``,
    that may be FamilySpecified or not.

    Walls are identified as boundary conditions of type ``'BCWall*'``. They also
    might be searched by pattern in their ``BCType`` with **WallFamilies**.

    .. important::
        Be careful when using this function on a partial mesh with periodicity
        conditions. If this kind of BC is not equidistant to the walls, the
        computed ``'TurbulentDistance'`` will be wrong !
        Except if options are specified to take periodicity into account

    Parameters
    ----------

        t : PyTree
            input mesh tree

        WallFamilies : list
            List of patterns to search in family names to identify wall surfaces.
            Names are not case-sensitive (automatic conversion to lower, uper and
            capitalized cases).

        verbose : bool
            if :py:obj:`True`, print families recognized as walls.

        wallFilename : :py:class:`str` or :py:obj:`None`
            if not :py:obj:`None`, write the wall surfaces in the file named
            **wallFilename**.

    '''
    def __adaptZoneNames(zone):
        zn = I.getName(zone)
        if   '/' in zn:  zn = zn.split('/')[-1]
        elif "\\" in zn: zn = zn.split('\\')[-1]
        I.setName(zone,zn)

    def __duplicateWalls(walls):
        for bw in I.getBases(walls):
            for zw in I.getZones(bw):
                sp = I.getNodeFromName2(zw,".Solver#Param")
                if sp:
                    __adaptZoneNames(zw)
                    ang1 = I.getNodeFromName1(sp,'axis_ang_1'); ang1 = I.getValue(ang1)
                    ang2 = I.getNodeFromName1(sp,'axis_ang_2'); ang2 = I.getValue(ang2)
                    angle = float(ang2)/float(ang1)*360.
                    xc = I.getNodeFromName1(sp,'axis_pnt_x'); xc = I.getValue(xc)
                    yc = I.getNodeFromName1(sp,'axis_pnt_y'); yc = I.getValue(yc)
                    zc = I.getNodeFromName1(sp,'axis_pnt_z'); zc = I.getValue(zc)
                    vx = I.getNodeFromName1(sp,'axis_vct_x'); vx = I.getValue(vx)
                    vy = I.getNodeFromName1(sp,'axis_vct_y'); vy = I.getValue(vy)
                    vz = I.getNodeFromName1(sp,'axis_vct_z'); vz = I.getValue(vz)
                    if angle != 0.:
                        zdupPos = T.rotate(zw,(xc,yc,zc), (vx,vy,vz),angle)
                        zdupNeg = T.rotate(zw,(xc,yc,zc), (vx,vy,vz),-angle)
                        zdupPos[0] += 'dup_pos'
                        zdupNeg[0] += 'dup_neg'
                        I.addChild(bw,zdupPos)
                        I.addChild(bw,zdupNeg)

    WallFamilies = extendListOfFamilies(WallFamilies)

    print('Compute distance to walls...')

    BCs, _, BCTypes = C.getBCs(t)
    walls = [] # will be a list on Zone_t nodes
    wallBCTypes = set()
    for BC, BCType in zip(BCs, BCTypes):
        FamilyBCType = getFamilyBCTypeFromFamilyBCName(t, BCType)
        # print('isStdNode=',I.isStdNode(BC))
        if FamilyBCType is not None and 'BCWall' in FamilyBCType:
            wallBCTypes.add(FamilyBCType)
            if I.isStdNode(BC) == 0: # list of pytree nodes
                tmpBC = BC
            elif I.isStdNode(BC) == -1: # pytree node
                tmpBC = [BC]
            else: raise TypeError('BC is not a PyTree node or a list of PyTree nodes')
            for tbc in tmpBC:
                I._rmNodesByType(tbc,'FlowSolution_t')
                I._adaptZoneNamesForSlash(tbc)
                walls.append(tbc)
        elif any([pattern in BCType for pattern in WallFamilies]):
            wallBCTypes.add(BCType)
            if I.isStdNode(BC) == 0: # list of pytree nodes
                tmpBC = BC
            elif I.isStdNode(BC) == -1: # pytree node
                tmpBC = [BC]
            else: raise TypeError('BC is not a PyTree node or a list of PyTree nodes')
            for tbc in tmpBC:
                I._rmNodesByType(tbc,'FlowSolution_t')
                I._adaptZoneNamesForSlash(tbc)
                walls.append(tbc)
    if verbose:
        print('List of BCTypes recognized as walls:')
        for BCType in wallBCTypes:
            print('  {}'.format(BCType))
    walls = C.newPyTree(['WALLS', walls]) # as walls is a list of Zone_t nodes 
    I._adaptZoneNamesForSlash(walls)
    __duplicateWalls(walls)

    if wallFilename:
        C.convertPyTree2File(walls, wallFilename)

    container_cell_save = I.__FlowSolutionCenters__
    I.__FlowSolutionCenters__ = 'FlowSolution#Init'
    DTW._distance2Walls(t, walls)
    EP._addTurbulentDistanceIndex(t)
    I.__FlowSolutionCenters__ = container_cell_save

def convert2Unstructured(t, merge=True, tol=1e-6):
    '''
    Convert to unstructured mesh and merge zones by Family. Recover all BCs and
    all zones and BC Families.

    Parameters
    ----------

        t : PyTree
            input tree

        merge : bool
            if :py:obj:`True`, merge zones by Family, and recover all BCs and
            all Families (zones and BC). Else, no merge is performed.

        tol : float
            Tolerance to recover BCs.

    Returns
    -------

        t : PyTree
            unstructured tree

    '''
    print('Convert to unstructured mesh')
    # Important : Delete Autogrid5 bases, otherwise the function is much more longer to run
    I._rmNodesByName(t, 'Numeca*')
    I._rmNodesByName(t, 'meridional_base')
    I._rmNodesByName(t, 'tools_base')

    t = C.convertArray2NGon(t, recoverBC=1)
    if merge:
        t = mergeUnstructuredMeshByFamily(t, tol=tol)

    return t

def mergeUnstructuredMeshByFamily(t, tol=1e-6):
    '''
    Merge zones by Family. Recover all BCs and all zones and BC Families.

    Parameters
    ----------

        t : PyTree
            Unstructured tree

        tol : float
            Tolerance to recover BCs.

    Returns
    -------

        t : PyTree
            Merged unstructured tree

    '''
    for base in I.getBases(t):

        for family in I.getNodesFromType(base, 'Family_t'):
            if I.getNodeFromType(family, 'FamilyBC_t'):
                continue

            familyName = I.getName(family)
            zones =  C.getFamilyZones(base, familyName)
            if len(zones) < 2: continue # no merge needed

            # Save BCs information
            # IMPORTANT: Saving and recovering BCs must be done by zones family
            # to avoid to recover both sides of a rotor-stator interface on each
            # adjacent zone
            (BCs, BCNames, BCTypes) = C.getBCs(zones)

            # Merge zones in family
            mergedZone = T.join(zones)
            I.setName(mergedZone, '{}_zone'.format(familyName))
            C._tagWithFamily(mergedZone, familyName)
            I.addChild(base, mergedZone)
            for zone in zones: I.rmNode(base, zone)

            # Recover BCs
            for zone in C.getFamilyZones(base, familyName):
                C._recoverBCs(zone, (BCs, BCNames, BCTypes), tol=tol)

    autoMergeBCsUnstructured(t)

    return t


def sendSimulationFiles(DIRECTORY_WORK, overrideFields=True):
    ElementsToSend = ['setup.py', 'main.cgns']
    if os.path.exists('OVERSET'): ElementsToSend += ['OVERSET']
    if overrideFields: ElementsToSend += ['OUTPUT/fields.cgns']
    ElementsToSend += glob.glob('state_radius*') # https://elsa.onera.fr/issues/11304
    setup = J.load_source('setup','setup.py')
    try: BodyForceInputData = setup.BodyForceInputData
    except: BodyForceInputData = []

    for b in BodyForceInputData:
        for k in b:
            if k.startswith('FILE_') and b[k] not in ElementsToSend:
                ElementsToSend += [ b[k] ]


    files2copyString = ', '.join([s.split('/')[-1] for s in ElementsToSend])
    print(f'Copying simulation elements ({files2copyString}) to {DIRECTORY_WORK}')
    for elt in ElementsToSend:
        JM.repatriate(elt, os.path.join(DIRECTORY_WORK, elt))


def isUnsteadyMask(body, InputMeshes):
    body_name = getBodyName(body)
    base_name = getBodyParentBaseName(body_name)
    meshInfo = getMeshInfoFromBaseName(base_name)
    if 'Motion' in meshInfo: return True
    return False

def getBodyParentBaseName(BodyName):
    return '-'.join(BodyName.split('-')[1:])

def getBodyName(body):
    if isinstance(body[0],str): return body[0]
    else: return body[0][0]

def _flattenBodies( bodies ):
    bodies_zones = []
    for b in bodies:
        if isinstance(b[0],str):
            bodies_zones.append(b)
        elif isinstance(b[0],list):
            for bb in b:
                if not isinstance(bb[0],str):
                    raise ValueError('wrong bodies container')
                bodies_zones.append(bb)
    return bodies_zones

def _ungroupBCsByBCType(t, forced_starting=''):
    for BC in I.getNodesFromType(t,'BC_t'):
        BCvalue = I.getValue(BC)
        if BCvalue == 'FamilySpecified':
            FamilyBC = I.getValue(I.getNodeFromName1(BC,'FamilyName'))
            BCType = getFamilyBCTypeFromFamilyBCName(t, FamilyBC)
            if forced_starting:
                if BCType.startswith(forced_starting):
                    BCType = forced_starting
            I.setValue(BC,BCType)


def duplicateBlades(base, meshInfo):
    try: MotionInfo = meshInfo['Motion']
    except KeyError: return [], []
    
    try: NumberOfBlades = MotionInfo['NumberOfBlades']
    except KeyError: NumberOfBlades = 1

    try: InitialFrame = MotionInfo['InitialFrame']
    except KeyError: InitialFrame = None

    try: RequestedFrame = MotionInfo['RequestedFrame']
    except KeyError: RequestedFrame = None

    if not InitialFrame and not RequestedFrame:
        return [], []
    
    elif RequestedFrame and not InitialFrame:
        InitialFrame = RequestedFrame
        MotionInfo['InitialFrame'] = InitialFrame
         
    elif InitialFrame and not RequestedFrame:
        RequestedFrame = InitialFrame
        MotionInfo['RequestedFrame'] = RequestedFrame

    from .RotatoryWings import placeRotorAndDuplicateBlades
    NewBases = placeRotorAndDuplicateBlades(base,
                                    InitialFrame['RotationCenter'],
                                    InitialFrame['RotationAxis'],
                                    InitialFrame['BladeDirection'],
                                    InitialFrame['RightHandRuleRotation'],
                                    RequestedFrame['RotationCenter'],
                                    RequestedFrame['RotationAxis'],
                                    RequestedFrame['BladeDirection'], 
                                    RequestedFrame['RightHandRuleRotation'],
                                    AzimutalDuplicationNumber=NumberOfBlades,
                                    orthonormal_tolerance_in_degree=0.5)

    NewMeshInfos = []
    for i, base in enumerate(NewBases):
        suffix = '_%d'%(i+1)
        base[0] = meshInfo['baseName'] + suffix
        newMeshInfo = copy.deepcopy(meshInfo)
        newMeshInfo['baseName'] = base[0]
        if i > 0: newMeshInfo['DuplicatedFrom'] = meshInfo['baseName']+'_1'
        
        try: BCinfo = newMeshInfo['BoundaryConditions']
        except KeyError: BCinfo = []
        for bc in BCinfo:
            if bc['type'].startswith('FamilySpecified:'):
                bc['type'] += suffix
                bc['name'] += suffix

        J.set(base,'.MOLA#InputMesh',**newMeshInfo)
        NewMeshInfos += [newMeshInfo]

    return NewBases, NewMeshInfos

def removeFamilies(t, families_to_remove):
    '''
    Remove useless families by their name on tree (both *Family_t*, 
    *FamilyName_t* and *BC_t*
    nodes are removed).

    Parameters
    ----------

        t : PyTree
            working tree

            .. note:: tree **t** is modified

        families_to_remove : :py:class:`list` of :py:class:`str`
            names of the families to be removed
    '''
    all_family_types = I.getNodesFromType(t,'Family_t')
    all_family_name_types = I.getNodesFromType(t,'FamilyName_t')
    all_bc_types = I.getNodesFromType(t,'BC_t')
    for f in families_to_remove:
        for n in all_family_types+all_family_name_types+all_bc_types:
            if n[0] == f or I.getValue(n) == f:
                I.rmNode(t, n)

def renameNodes(t, rename_dict={}):
    '''
    a shortcut of I._renameNode function, literally:

    ::

        for old, new in rename_dict.items():
            I._renameNode(t, old, new)

    '''
    for old, new in rename_dict.items():
        I._renameNode(t, old, new)


def adaptFamilyBCNamesToElsA(t):
    '''
    Due to https://elsa.onera.fr/issues/11090
    '''
    for n in I.getNodesFromType(t, 'FamilyBC_t'):
        n[0] = 'FamilyBC'

def convertUnstructuredMeshToNGon(t, mergeZonesByFamily=True):
    print('making unstructured mesh adaptations for elsA:')
    from mpi4py import MPI
    import maia
    
    t = maia.factory.full_to_dist_tree(t, MPI.COMM_WORLD, owner=None)
    
    if J.anyNotNGon(t):
        print(' -> some cells are not NGon : converting to NGon')
        try:
            tRef = I.copyRef(t)
            maia.algo.dist.generate_ngon_from_std_elements(tRef, MPI.COMM_WORLD)
            t = tRef
        except BaseException as e:
            print(J.WARN+f'WARNING: could not convert to NGon using maia, received error:')
            print(e)
            print('attempting using Cassiopee...'+J.ENDC)
            try:
                tRef = I.copyRef(t)
                uns = getUnstructuredZones(tRef)[0]
                I.printTree(uns,'tree1.txt')
                C._convertArray2NGon(uns, recoverBC=1)
                t = tRef
            except BaseException as e2:
                print(J.FAIL)
                C.convertPyTree2File(t,'debug.cgns')
                print('could not convert to NGon using Cassiopee, check debug.cgns')
                t = C.convertFile2PyTree('debug.cgns')
                uns = getUnstructuredZones(t)[0]
                I.printTree(uns,'tree2.txt')
                C._convertArray2NGon(uns, recoverBC=1)
                print(J.GREEN+'OK'+J.ENDC)


    if mergeZonesByFamily:
        print(' -> merging zones by family')
        zonePathsByFamily = dict()
        for base in I.getBases(t):
            for zone in I.getZones(base):
                if I.getZoneType(zone) == 1: continue # structured zone
                zone_path = base[0] + '/' + zone[0]
                FamilyNameNode = I.getNodeFromType1(zone, 'FamilyName_t')
                if FamilyNameNode: 
                    FamilyName = I.getValue(FamilyNameNode)
                    if FamilyName not in zonePathsByFamily:
                        zonePathsByFamily[FamilyName] = [zone_path]
                    else:
                        zonePathsByFamily[FamilyName] += [zone_path]
                else:
                    if 'unspecified' not in zonePathsByFamily:
                        zonePathsByFamily['unspecified'] = [zone_path]
                    else:
                        zonePathsByFamily['unspecified'] += [zone_path]

        for family, zone_paths in zonePathsByFamily.items():
            if len(zone_paths) < 2: continue
            print(f' --> merging zones of family {family}')
            try:
                maia.algo.dist.merge_zones(t, zone_paths, MPI.COMM_WORLD)
                MergedZone = I.getNodeFromName3(t,'MergedZone')
                MergedZone[0] = family + 'Zone'
            except BaseException as e:
                print(J.WARN+f'WARNING: could not merge zones using maia, received error:')
                print(str(e))
                print('will not merge zones'+J.ENDC)

    print(' -> enforcing ngon_pe_local')
    maia.algo.seq.enforce_ngon_pe_local(t) # required by elsA ?

    t = maia.factory.dist_to_full_tree(t, MPI.COMM_WORLD, target=0)
    I._fixNGon(t) # required ?
    print('finished unstructured mesh adaptations for elsA')
    print('zones names:')
    for z in I.getZones(t):
        print(z[0])

    return t

def addFieldExtraction(ReferenceValues, fieldname):
    try:
        if fieldname not in ReferenceValues['FieldsAdditionalExtractions']:
            print('adding %s'%fieldname)
            ReferenceValues['FieldsAdditionalExtractions'].append(fieldname)
    except:
        print('adding %s'%fieldname)
        ReferenceValues['FieldsAdditionalExtractions'] = [fieldname]

def appendAdditionalFieldExtractions(ReferenceValues, Extractions):
    field_names = set()
    for e in Extractions:
        if 'field' in e:
            field_names.update([e['field']])
        elif e['type'] == 'Probe':
            field_names.update(e['variables'])
        else:
            continue

    for field_name in field_names:
        if field_name.startswith('Coordinate') or field_name == 'ChannelHeight':
            continue
        addFieldExtraction(ReferenceValues, field_name)

def addBC2Zone(*args, **kwargs):
    '''
    Workaround of Converter._addBC2Zone (in-place) function in order to circumvent 
    https://elsa.onera.fr/issues/11236

    '''
    zone = args[0]
    if len(zone) != 4: 
        raise ValueError('first argument must be a zone')
    else:
        try:
            is_zone = zone[3] == 'Zone_t'
        except:
            raise ValueError('first argument must be a zone')
        if not is_zone:
            raise ValueError('first argument must be a zone')

    is_structured = I.getZoneType(zone) == 1


    if is_structured: return C._addBC2Zone(*args,**kwargs)

    C._addBC2Zone(*args,**kwargs)

    # HACK https://elsa.onera.fr/issues/11236
    for n in I.getNodesFromType(zone,'ZoneBC_t'):
        I._renameNode(n,'ElementRange','PointRange')
    
    for n in I.getNodesFromType(zone,'BC_t'):
        I.createUniqueChild(n, 'GridLocation', 'GridLocation_t', value='FaceCenter')

    for n in I.getNodesFromType(zone,'Elements_t'): n[0] = n[0].replace('/','_')

_addBC2Zone = addBC2Zone

def _convert_mesh_to_ngon(filename_in, filename_out):

    from mpi4py import MPI
    import maia

    if not filename_in:
        raise AttributeError('must provide input filename as first argument')

    if not filename_out:
        filename_out = filename_in.replace('.cgns','_ngon.cgns')

    t = maia.io.file_to_dist_tree(filename_in, MPI.COMM_WORLD)
    maia.algo.dist.generate_ngon_from_std_elements(t, MPI.COMM_WORLD)
    maia.io.dist_tree_to_file(t, filename_out, MPI.COMM_WORLD)