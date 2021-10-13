'''
PREPROCESS module

It implements a collection of routines for preprocessing of CFD simulations

23/12/2020 - L. Bernardos - creation by recycling
'''

import sys
import os
import pprint
import numpy as np
from itertools import product

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
import MOLA
from . import InternalShortcuts as J
from . import GenerativeShapeDesign as GSD
from . import GenerativeVolumeDesign as GVD
from . import ExtractSurfacesProcessor as ESP

def prepareMesh4ElsA(InputMeshes, NProcs=None, ProcPointsLoad=250000):
    '''
    This is a macro-function used to prepare the mesh for an elsA computation
    from user-provided instructions in form of a list of python dictionaries.

    The sequence of operations performed are the following:

    #. load and assemble the meshes
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

            * file : :py:class:`str`
                path of the file containing the grid.

                .. attention:: each component must have a unique base. If
                    input file have several bases, please separate each base
                    into different files.

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

            * SplitBlocks : :py:class:`bool`
                if :py:obj:`True`, allow for splitting this component in
                order to satisfy the user-provided rules of total number of used
                processors and block points load during simulation. If :py:obj:`False`,
                the component is protected against splitting.

                .. attention:: split operation results in loss of connectivity information.
                    Hence, if ``SplitBlocks=True`` , then user must specify connection
                    rules in list **Connection**.

        NProcs : int
            If a positive integer is provided, then the
            distribution of the tree (and eventually the splitting) will be done in
            order to satisfy a total number of processors provided by this value.
            If not provided (:py:obj:`None`) then the number of procs is automatically
            determined using as information **ProcPointsLoad** variable.

        ProcPointsLoad : int
            this is the desired number of grid points
            attributed to each processor. If **SplitBlocks** = :py:obj:`True`, then it is used to
            split zones that have more points than **ProcPointsLoad**. If
            **NProcs** = :py:obj:`None` , then **ProcPointsLoad** is used to determine
            the **NProcs** to be used.

    Returns
    -------

        t : PyTree
            the pre-processed mesh tree (usually saved as ``mesh.cgns``)

            .. important:: This tree is **NOT** ready for elsA computation yet !
                The user shall employ function :py:func:`prepareMainCGNS4ElsA`
                as next step
    '''

    t = getMeshesAssembled(InputMeshes)
    transform(t, InputMeshes)
    t = connectMesh(t, InputMeshes)
    setBoundaryConditions(t, InputMeshes)
    t = splitAndDistribute(t, InputMeshes,
                              NProcs=NProcs,
                              ProcPointsLoad=ProcPointsLoad)
    addFamilies(t, InputMeshes)
    t = addOversetData(t, InputMeshes, saveMaskBodiesTree=True)
    adapt2elsA(t, InputMeshes)
    J.checkEmptyBC(t)

    return t


def prepareMainCGNS4ElsA(mesh, ReferenceValuesParams={},
        NumericalParams={}, Extractions=[{'type':'AllBCWall'}],
        BodyForceInputData=[], writeOutputFields=True):
    '''
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
    #. initialize flowfields (uniformly)
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

        NumericalParams : dict
            dictionary containing the numerical
            settings for elsA. For information on acceptable values, please see
            the documentation of function :py:func:`getElsAkeysNumerics`

            .. note:: internally, this dictionary is passed as *kwargs* as follows:

                >>> PRE.getElsAkeysNumerics(arg, **NumericalParams)

        Extractions : :py:class:`list` of :py:class:`dict`
            .. danger:: **doc this** # TODO

        BodyForceInputData : :py:class:`list` of :py:class:`dict`
            if provided, each item of this list constitutes a body-force modeling component.
            Currently acceptable pairs of keywords and associated values are:

            * name : :py:class:`str`
                the name to provide to the bodyforce component

            * proc : :py:class:`int`
                sets the processor at which the bodyforce component
                is associated for Lifting-Line operations.

                .. note:: **proc** must be :math:`\in (0, \mathrm{NProc}-1)`

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

    hasBCOverlap = True if C.extractBCOfType(t, 'BCOverlap') else False


    if hasBCOverlap: addFieldExtraction('ChimeraCellType')
    if BodyForceInputData: addFieldExtraction('Temperature')



    FluidProperties = computeFluidProperties()
    ReferenceValues = computeReferenceValues(FluidProperties,
                                             **ReferenceValuesParams)

    NProc = max([I.getNodeFromName(z,'proc')[1][0][0] for z in I.getZones(t)])+1
    ReferenceValues['NProc'] = int(NProc)
    ReferenceValuesParams['NProc'] = int(NProc)
    elsAkeysCFD      = getElsAkeysCFD()
    elsAkeysModel    = getElsAkeysModel(FluidProperties, ReferenceValues)
    if BodyForceInputData: NumericalParams['useBodyForce'] = True
    elsAkeysNumerics = getElsAkeysNumerics(ReferenceValues, **NumericalParams)

    AllSetupDics = dict(FluidProperties=FluidProperties,
                        ReferenceValues=ReferenceValues,
                        elsAkeysCFD=elsAkeysCFD,
                        elsAkeysModel=elsAkeysModel,
                        elsAkeysNumerics=elsAkeysNumerics,
                        Extractions=Extractions)

    if BodyForceInputData: AllSetupDics['BodyForceInputData'] = BodyForceInputData

    t = newCGNSfromSetup(t, AllSetupDics, initializeFlow=True,
                         FULL_CGNS_MODE=False)
    to = newRestartFieldsFromCGNS(t)
    saveMainCGNSwithLinkToOutputFields(t,to,writeOutputFields=writeOutputFields)


    print('REMEMBER : configuration shall be run using %s%d%s procs'%(J.CYAN,
                                               ReferenceValues['NProc'],J.ENDC))



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

            * file : :py:class:`str`
                the CGNS file containing the grid and possibly
                other CGNS information.

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
    for meshInfo in InputMeshes:
        filename = meshInfo['file']
        t = C.convertFile2PyTree(filename)
        bases = I.getBases(t)
        if len(bases) != 1:
            raise ValueError('InputMesh element in %s must contain only 1 base'%filename)
        base = bases[0]
        base[0] = meshInfo['baseName']
        Trees += [C.newPyTree([base])]

    t = I.merge(Trees)
    t = I.correctPyTree(t, level=-3)

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

                    .. tip:: use this option to transform a grid built in milimeters
                        into meters

                * 'rotate' : :py:class:`list` of :py:class:`tuple`
                    List of rotation to apply to the grid component. Each rotation
                    is defined by 3 elements:
                        * a 3-tuple corresponding to the center coordinates
                        * a 3-tuple corresponding to the rotation axis
                        * a float (or integer) defining the angle of rotation in degrees

                    .. tip:: this option is useful to change the orientation of
                        a mesh built in Autogrid 5.

    '''
    for meshInfo in InputMeshes:
        if 'Transform' not in meshInfo: continue

        base = I.getNodeFromName1(t, meshInfo['baseName'])

        if 'scale' in meshInfo['Transform']:
            s = float(meshInfo['Transform']['scale'])
            T._homothety(base,(0,0,0),s)

        if 'rotate' in meshInfo['Transform']:
            for center, axis, ang in meshInfo['Transform']['rotate']:
                T._rotate(base, center, axis, ang)
    T._makeDirect(t)


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

            * ``'angles'`` : :py:class:`list` of :py:class:`float`
                employed list of angles (in degree) for connection
                ``'type':'PeriodicMatch'``

    Returns
    -------

        t : PyTree
            Modified tree

            .. note:: this returned tree is only needed for ``'PeriodicMatch'``
                operation.

    '''
    for meshInfo in InputMeshes:
        if 'Connection' not in meshInfo: continue
        base = I.getNodeFromName1(t, meshInfo['baseName'])
        baseDim = I.getValue(base)[-1]
        for ConnectParams in meshInfo['Connection']:
            ConnectionType = ConnectParams['type']
            print('connecting type {} at base {}'.format(ConnectionType,
                                                         meshInfo['baseName']))
            if ConnectionType == 'Match':
                X.connectMatch(base, tol=ConnectParams['tolerance'], dim=baseDim)
            elif ConnectionType == 'NearMatch':
                X.connectNearMatch(t, ratio=ConnectParams['ratio'],
                                      tol=ConnectParams['tolerance'],
                                      dim=baseDim)
            elif ConnectionType == 'PeriodicMatch':
                for angle in ConnectParams['angles']:
                    print('  angle = {:g} deg ({} blades)'.format(angle, int(360./angle)))
                    t = X.connectMatchPeriodic(t,
                                            rotationCenter=[0.,0.,0.],
                                            rotationAngle=[angle,0.,0.],
                                            tol=ConnectParams['tolerance'],
                                            dim=baseDim,
                                            unitAngle='Degree')
            else:
                ERRMSG = 'Connection type %s not implemented'%ConnectionType
                raise AttributeError(ERRMSG)
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
                location = BCinfo['location']
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

        if 'BoundaryConditions' not in meshInfo: continue

        I._correctPyTree(base, level=7)
        for BCinfo in meshInfo['BoundaryConditions']:
            if BCinfo['type'].startswith('FamilySpecified:') and 'familySpecifiedType' in BCinfo:
                BCName = BCinfo['type'][16:]
                BCType = BCinfo['familySpecifiedType']
                if BCType == 'BCOverlap':
                    # TODO: Check tickets closure
                    ERRMSG=('BCOverlap must be fully defined directly on zones'
                        ' instead of indirectly using FAMILIES.\n'
                        'This option will be acceptable once Cassiopee tickets #7868'
                        ' and #7869 are solved.')
                    raise ValueError(J.FAIL+ERRMSG+J.ENDC)
                print('Setting BCName %s of BCType %s at base %s'%(BCName, BCType, base[0]))
                C._addFamily2Base(base, BCName, bndType=BCType)

    groupUserDefinedBCFamiliesByName(t)



def splitAndDistribute(t, InputMeshes, NProcs=None, ProcPointsLoad=2e5):
    '''
    Split a PyTree **t** using the desired proc points load **ProcPointsLoad**.
    Distribute the PyTree **t** using a user-provided **NProcs**. If **NProcs**
    is not provided, then it is automatically computed.

    Returns a new split and distributed PyTree.

    .. note:: only **InputMeshes** where ``'SplitBlocks':True`` are split.

    Parameters
    ----------

        t : PyTree
            assembled tree

        InputMeshes : :py:class:`list` of :py:class:`dict`
            user-provided preprocessing
            instructions as described in :py:func:`prepareMesh4ElsA` doc

        NProcs : int
            If a positive integer is provided, then the
            distribution of the tree (and eventually the splitting) will be done in
            order to satisfy a total number of processors provided by this value.
            If not provided (:py:obj:`None`) then the number of procs is automatically
            determined using as information **ProcPointsLoad** variable.

        ProcPointsLoad : int
            this is the desired number of grid points
            attributed to each processor. If **SplitBlocks** = :py:obj:`True`, then it is used to
            split zones that have more points than **ProcPointsLoad**. If
            **NProcs** = :py:obj:`None` , then **ProcPointsLoad** is used to determine
            the **NProcs** to be used.

    Returns
    -------

        t : PyTree
            new distributed *(and possibly split)* tree

    '''
    print('splitting and distributing mesh...')

    tRef = I.copyRef(t)

    basenames = [I.getName(base) for base in I.getBases(tRef)]

    basesToSplit, basesBackground = getBasesBasedOnSplitPolicy(tRef, InputMeshes)

    if basesToSplit:

        removeMatchAndNearMatch(tRef)

        '''
        # Option 1
        if NProcs:
            NPts4SplitSize = int(C.getNPts(tRef)/NProcs)
        else:
            NPts4SplitSize = ProcPointsLoad
        BasesAndZonesList = []
        for base in basesToSplit:
            NewBase = T.splitSize(base, NPts4SplitSize, minPtsPerDir=11)
            zones = I.getZones(NewBase)
            BasesAndZonesList.append( [ base[0], zones ] )
        '''


        # Option 2
        BasesAndZonesList = []
        for base in basesToSplit:
            zones = []
            for zone in I.getZones(base):
                ZoneNPts = C.getNPts(zone)
                if ZoneNPts <= ProcPointsLoad:
                    zones += [zone]
                    continue
                NParts2Split = int(np.round(ZoneNPts / float(ProcPointsLoad)))+1
                SplitZones = T.splitNParts([zone], NParts2Split,
                                            dirs=[1,2,3], recoverBC=True)
                SplitZonesNPts = [C.getNPts(z) for z in SplitZones]
                # print('zone %s with %d pts split in %d parts resulting in max %d pts min %d pts'%(zone[0],ZoneNPts,NParts2Split,max(SplitZonesNPts), min(SplitZonesNPts)))
                zones.extend(SplitZones)
            BasesAndZonesList.append( [ base[0], zones ] )


        # replace old zones with newly split zones only at concerned bases
        for BaseNameAndZones in BasesAndZonesList:
            basename = BaseNameAndZones[0]
            base = I.getNodeFromName2(tRef, basename)
            if not base: raise ValueError('unexpected !')
            I._rmNodesByType(base, 'Zone_t')
            base[2].extend(BaseNameAndZones[1])

        I._correctPyTree(tRef,level=3)

        tRef = connectMesh(tRef, InputMeshes)

    if NProcs is None:
        NProcs = int(np.round(C.getNPts(tRef) / float(ProcPointsLoad)))-1
    print('distributing through %d procs...'%NProcs)
    # NOTE see BUG ticket #8244 . Need algorithm='fast'
    tRef, stats = D2.distribute(tRef, NProcs, algorithm='fast', useCom='all')

    D2.printProcStats(tRef)
    showStatisticsAndCheckDistribution(tRef, stats)

    return tRef


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
        try:
            if meshInfo['SplitBlocks']:
                basesToSplit += [base]
            else:
                basesNotToSplit += [base]
        except KeyError:
            basesNotToSplit += [base]

    return basesToSplit, basesNotToSplit


def showStatisticsAndCheckDistribution(tNew, stats, CoresPerNode=28):
    '''
    Print statistics on the distribution of a PyTree and also indicates the load
    attributed to each computational node.

    Parameters
    ----------

        tNew : PyTree
            tree where distribution was done.

        stats : dict
            result of :py:func:`Distributor2.PyTree.distribute`

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

    MSG = ('SHOWING DISTRIBUTION STATISTICS\n'
           'Average number of points per processor: {meanPtsPerProc}\n'
           'Block with maximum relative load variation is {varMax}\n'
           ).format(**stats)
    MSG += 'Total number of processors is %d\n'%ResultingNProc
    MSG += 'Total number of zones is %d\n'%len(I.getZones(tNew))
    MSG += 'Proc %d has lowest nb. of points %d\n'%(ListOfProcs[ArgNPtsMin],
                                                    ListOfNPts[ArgNPtsMin])
    MSG += 'Proc %d has highest nb. of points %d\n'%(ListOfProcs[ArgNPtsMax],
                                                    ListOfNPts[ArgNPtsMax])
    print(MSG)

    for node in NPtsPerNode:
        print('Node %d has %d points'%(node,NPtsPerNode[node]))

    print('TOTAL NUMBER OF POINTS: %d'%C.getNPts(tNew))

    for p in range(ResultingNProc):
        if p not in ProcDistributed:
            raise ValueError('Bad proc distribution! rank %d is empty'%p)


def addOversetData(t, InputMeshes, depth=2, optimizeOverlap=False,
                   prioritiesIfOptimize=[], double_wall=0,
                   saveMaskBodiesTree=False):
    '''
    This function performs all required preprocessing operations for a STATIC
    overlapping configuration. This includes masks production, setting
    interpolating regions and computing interpolating coefficients.

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

            * ``'ForbiddenOverlapMaskingThisBase'`` : :py:class:`list` of :py:class:`str`
                This is a list of
                base names (names of **InputMeshes** components) whose masking bodies
                built from their *BCOverlap* are not allowed to mask this component.
                This is used to protect this component from being masked by other
                component's masks (only affects masks constructed from offset of
                overlap bodies, this does not include masks constructed from walls).

        depth : int
            depth of the interpolation region.

        optimizeOverlap : bool
            if :py:obj:`True`, then applies :py:func:`Connector.PyTree.optimizeOverlap` function.

        prioritiesIfOptimize : list
            literally, the
            priorities argument passed to :py:func:`Connector.PyTree.optimizeOverlap`.

            .. note:: only relevant if **optimizeOverlap** is set to :py:obj:`True`.

        double_wall : bool
            if :py:obj:`True`, double walls exist

        saveMaskBodiesTree : bool
            if :py:obj:`True`, then saves the file ``masks.cgns``,
            allowing the user to analyze if masks have been properly generated.

    Returns
    -------

        t : PyTree
            new pytree including ``cellN`` values at ``FlowSolution#Centers``
            and elsA's ``ID*`` nodes including interpolation coefficients information.

    '''


    if not hasAnyOversetData(InputMeshes): return t

    DIRECTORY_OVERSET = 'OVERSET'


    print('building masking bodies...')
    baseName2BodiesDict = getMaskingBodiesAsDict(t, InputMeshes)

    bodies = []
    for meshInfo in InputMeshes:
        bodies.extend(baseName2BodiesDict[meshInfo['baseName']])

    if saveMaskBodiesTree:
        tMask = C.newPyTree(['MASK', bodies])
        C.convertPyTree2File(tMask, 'mask.cgns')


    BlankingMatrix = getBlankingMatrix(bodies, InputMeshes)

    t = X.applyBCOverlaps(t, depth=depth)

    print('Blanking...')
    # see ticket #7882
    for ibody, body in enumerate(bodies):
        BlankingVector = np.atleast_2d(BlankingMatrix[:,ibody]).T
        BaseNameOfBody = ''.join(body[0].split('-')[1:])
        meshInfo = getMeshInfoFromBaseName(BaseNameOfBody, InputMeshes)
        try: BlankingMethod = meshInfo['OversetOptions']['BlankingMethod']
        except KeyError: BlankingMethod = 'blankCellsTri'

        try: UserSpecifiedBlankingMethodOptions = meshInfo['OversetOptions']['BlankingMethodOptions']
        except KeyError: UserSpecifiedBlankingMethodOptions = {}
        BlankingMethodOptions = dict(blankingType='center_in')
        BlankingMethodOptions.update(UserSpecifiedBlankingMethodOptions)

        if BlankingMethod == 'blankCellsTri':
            t = X.blankCellsTri(t, [[body]], BlankingVector,
                                    **BlankingMethodOptions)

        elif BlankingMethod == 'blankCells':
            t = X.blankCells(t, [[body]], BlankingVector,
                                **BlankingMethodOptions)

        else:
            raise ValueError('BlankingMethod "{}" not recognized'.format(BlankingMethod))

    print('... blanking done.')

    print('setting hole interpolated points...')
    t = X.setHoleInterpolatedPoints(t, depth=depth)

    if optimizeOverlap:
        print('Optimizing overlap...')
        t = X.optimizeOverlap(t, double_wall=double_wall,
                              priorities=prioritiesIfOptimize)
        print('... optimization done.')

    print('maximizing blanked cells...')
    t = X.maximizeBlankedCells(t, depth=depth)

    print('Computing interpolation coefficients...')
    try: os.makedirs(DIRECTORY_OVERSET)
    except: pass
    t = X.setInterpolations(t, loc='cell', sameBase=0, double_wall=double_wall,
                            storage='inverse', solver='elsA', check=True,
                            nGhostCells=2,
                            prefixFile=os.path.join(DIRECTORY_OVERSET,
                                                    'OvstData'),)
    print('... interpolation coefficients built.')

    for diagnosisType in ['orphan', 'extrapolated']:
        tAux = X.chimeraInfo(t, type=diagnosisType)
        CriticalPoints = X.extractChimeraInfo(tAux, type=diagnosisType)
        if CriticalPoints:
            C.convertPyTree2File(CriticalPoints, diagnosisType+'.cgns')

    t = X.cellN2OversetHoles(t)

    return t


def getBlankingMatrix(bodies, InputMeshes):
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

    def getBodyParentBaseName(BodyName):
        return '-'.join(BodyName.split('-')[1:])

    # BM(i,j)=1 means that ith basis is blanked by jth body
    Nbases  = len( InputMeshes )
    Nbodies = len( bodies )

    BaseNames = [meshInfo['baseName'] for meshInfo in InputMeshes]
    BodyNames = [I.getName( body ) for body in bodies]

    # Initialization: all bodies mask all bases
    BlankingMatrix = np.ones((Nbases, Nbodies))

    # do not allow bodies issued of a given base to mask its own parent base
    for i, j in product(range(Nbases), range(Nbodies)):
        BaseName = BaseNames[i]
        BodyName = BodyNames[j]
        BodyParentBaseName = getBodyParentBaseName(BodyName)
        if BaseName in BodyParentBaseName:
            BlankingMatrix[i, j] = 0

    # user-provided masking protections
    for i, meshInfo in enumerate(InputMeshes):
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

    print('BaseNames = %s'%str(BaseNames))
    print('BodyNames = %s'%str(BodyNames))
    print('BlankingMatrix:')
    print(BlankingMatrix)

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

        try:
            OversetOptions = meshInfo['OversetOptions']
        except KeyError:
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
                                               NCellsOffset=NCellsOffset,
                                               MatchTolerance=MatchTolerance)

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
    walls = buildWatertightBodiesFromSurfaces(walls,
                                             imposeNormalsOrientation='inwards',
                                             SuffixTag=SuffixTag)
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


def getOverlapMaskByCellsOffset(base, SuffixTag=None, NCellsOffset=2,
                                   MatchTolerance=1e-8):
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
    mask = C.convertArray2Tetra(mask)
    mask = T.join(mask)

    if GSD.isClosed(mask, tol=MatchTolerance):
        mask = T.reorderAll(mask, dir=-1) # force normal pointing inwards
    else:
        mask = buildWatertightBodyFromSurfaces([mask],
                                             imposeNormalsOrientation='inwards')

    if SuffixTag: mask[0] = SuffixTag

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
    NewOpenMask, = I.getZones(ExtrudeLayerBase)

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
    Name = mask[0]
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
            LargeSurfaces, SmallSurfaces=GSD.filterSurfacesByArea(masks,
                ratio=0.50)
            body = T.join(LargeSurfaces)
            G._close(body)
            ClosedBody = T.reorderAll(body, dir=-1)
            C.convertPyTree2File(body,'debug_body.cgns')
            return body


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
    body, = I.getZones(body)

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
        body, = I.getZones(body)
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


def computeFluidProperties(Gamma=1.4, RealGas=287.053, Prandtl=0.72,
        PrandtlTurbulence=0.9, SutherlandConstant=110.4,
        SutherlandViscosity=1.78938e-05, SutherlandTemperature=288.15,
        cvAndcp=None):

    '''
    Construct a dictionary of values concerning the fluid properties of air.

    Please note reference default reference values:

    Reference elsA Theory Manual v4.2.01, Table 2.1, Section 2.1.1.5:

    ::

        RealGas = 287.053
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
        cv                = RealGas/(Gamma-1.0)
        cp                = Gamma * cv
    else:
        cv, cp = cvAndcp

    FluidProperties = dict(
    Gamma                 = Gamma,
    RealGas               = RealGas,
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
        Velocity=0.0, AngleOfAttackDeg=0.0, AngleOfSlipDeg=0.0,
        YawAxis=[0.,0.,1.], PitchAxis=[0.,-1.,0.],
        TurbulenceLevel=0.001,
        Surface=1.0, Length=1.0, TorqueOrigin=[0,0,0],
        TurbulenceModel='Wilcox2006-klim', Viscosity_EddyMolecularRatio=0.1,
        TurbulenceCutoff=0.1, TransitionMode=None, CoprocessOptions={},
        FieldsAdditionalExtractions = ['ViscosityMolecular','ViscosityEddy',
                                       'Mach']):
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
            ``'SA'``, ``'BSL'``, ``'BSL-V'``, ``'SST-2003'``, ``'SST'``,
            ``'SST-V'``, ``'Wilcox2006-klim'``, ``'SST-2003-LM2009'``,
            ``'SSG/LRR-RSM-w2012'``

            other non-conventional turbulence models:
            ``'smith'`` reference `doi:10.2514/6.1995-232 <http://doi.org/10.2514/6.1995-232>`_

        Viscosity_EddyMolecularRatio : float
            Expected ratio of eddy to molecular viscosity at farfield

        TurbulenceCutoff : float
            Ratio of farfield turbulent quantities used for imposing a cutoff.

        TransitionMode : str
            .. attention:: not implemented in workflow standard

        CoprocessOptions : dict
            Override default coprocess options dictionary with this paramter.
            Default options are:

            ::

                'UpdateRestartFrequency' : 1000,
                'UpdateLoadsFrequency'   : 20,
                'NewSurfacesFrequency'   : 500,
                'AveragingIterations'    : 3000,
                'MaxConvergedCLStd'      : 1e-4,
                'ItersMinEvenIfConverged': 1000,
                'TimeOutInSeconds'       : 53100.0, # 14.75 h * 3600 s/h = 53100 s
                'SecondsMargin4QuitBeforeTimeOut' : 900.,
                'ConvergenceCriterionFamilyName' : '', # Add familyBCname

        FieldsAdditionalExtractions : :py:class:`list` of :py:class:`str`
            elsA or CGNS keywords of fields to be extracted.
            additional fields to be included as extraction.

            .. note:: primitive conservative variables required for restart are
                automatically included


    Returns
    -------

        ReferenceValues : dict
            dictionary containing all reference values of the simulation
    '''


    DefaultCoprocessOptions = dict(            # Default values
        UpdateFieldsFrequency  = 2000,
        UpdateLoadsFrequency    = 50,
        NewSurfacesFrequency    = 500,
        AveragingIterations     = 3000,
        MaxConvergedCLStd       = 1e-4,
        ItersMinEvenIfConverged = 1000,
        TimeOutInSeconds        = 54000.0, # 15 h * 3600 s/h = 53100 s
        SecondsMargin4QuitBeforeTimeOut = 900.,
        ConvergenceCriterionFamilyName = '', # Add familyBCname
    )
    DefaultCoprocessOptions.update(CoprocessOptions) # User-provided values

    ReferenceValues = dict(
    CoprocessOptions   = DefaultCoprocessOptions,
    AngleOfAttackDeg   = AngleOfAttackDeg,
    AngleOfSlipDeg     = AngleOfSlipDeg,
    TurbulenceLevel    = TurbulenceLevel,
    TurbulenceModel    = TurbulenceModel,
    TransitionMode     = TransitionMode,
    Viscosity_EddyMolecularRatio = Viscosity_EddyMolecularRatio,
    TurbulenceCutoff   = TurbulenceCutoff,
    )

    # Fluid properties local shortcuts
    Gamma   = FluidProperties['Gamma']
    RealGas = FluidProperties['RealGas']
    cv      = FluidProperties['cv']
    cp      = FluidProperties['cp']

    # REFERENCE VALUES COMPUTATION
    T   = Temperature
    mus = FluidProperties['SutherlandViscosity']
    Ts  = FluidProperties['SutherlandTemperature']
    S   = FluidProperties['SutherlandConstant']

    ViscosityMolecular = mus * (T/Ts)**1.5 * ((Ts + S)/(T + S))
    Mach = Velocity / np.sqrt( Gamma * RealGas * Temperature )
    Reynolds = Density * Velocity * Length / ViscosityMolecular
    Pressure = Density * RealGas * Temperature
    PressureDynamic = 0.5 * Density * Velocity **2
    FluxCoef        = 1./(PressureDynamic * Surface)
    TorqueCoef      = 1./(PressureDynamic * Surface*Length)

    # Reference state (farfield)
    FlowDir=getFlowDirections(AngleOfAttackDeg,AngleOfSlipDeg,YawAxis,PitchAxis)
    DragDirection, SideDirection, LiftDirection= FlowDir
    MomentumX =  Density * Velocity * DragDirection[0]
    MomentumY =  Density * Velocity * DragDirection[1]
    MomentumZ =  Density * Velocity * DragDirection[2]
    EnergyStagnationDensity = Density * (cv * Temperature + 0.5 * Velocity **2)

    # -> for k-omega models
    TurbulentEnergyKineticDensity = Density * 1.5* TurbulenceLevel**2 * Velocity**2
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


    elif TurbulenceModel in ('BSL','BSL-V','SST-2003','SST','SST-V','Wilcox2006-klim'):
        FieldsTurbulence  = ['TurbulentEnergyKineticDensity','TurbulentDissipationRateDensity']
        ReferenceStateTurbulence = [float(TurbulentEnergyKineticDensity), float(TurbulentDissipationRateDensity)]

    elif TurbulenceModel == 'smith':
        FieldsTurbulence  = ['TurbulentEnergyKineticDensity','TurbulentLengthScaleDensity']
        ReferenceStateTurbulence = [float(TurbulentEnergyKineticDensity), float(TurbulentLengthScaleDensity)]

    elif TurbulenceModel == 'SST-2003-LM2009':
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

    return ReferenceValues


def getElsAkeysCFD(config='3d'):
    '''
    Create a dictionary of pairs of elsA keyword/values to be employed as
    cfd problem object.

    Parameters
    ----------

        config : str
            elsa keyword config (``'2d'`` or ``'3d'``)

    Returns
    -------

        elsAkeysCFD : dict
            dictionary containing key/value for elsA *cfd* object
    '''
    elsAkeysCFD      = dict(
        config=config,
        extract_filtering='inactive')
    return elsAkeysCFD


def getElsAkeysModel(FluidProperties, ReferenceValues):
    '''
    Produce the elsA model object keys as a Python dictionary.

    Parameters
    ----------

        FluidProperties : dict
            as obtained from :py:func:`computeFluidProperties`

        ReferenceValues : dict
            as obtained from :py:func:`computeReferenceValues`

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
    walldistcompute  = 'mininterf_ortho',

    # Boundary-layer computation parameters
    vortratiolim    = 1e-3,
    shearratiolim   = 2e-2,
    pressratiolim   = 1e-3,
    linearratiolim  = 1e-3,
    delta_compute   = 'first_order_bl',
    )

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
        omega_prolong  = 'infinit_extrap',
            )

    elif TurbulenceModel == 'SST-2003':
        addKeys4Model = dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'std_sij',
        k_prod_limiter = 10.,
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'infinit_extrap',
            )

    elif TurbulenceModel == 'SST-V2003':
        addKeys4Model = dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'std_sij',
        k_prod_limiter = 10.,
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'infinit_extrap',
            )

    elif TurbulenceModel == 'SST':
        addKeys4Model = dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'standard',
        k_prod_limiter = 20.,
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'infinit_extrap',
            )

    elif TurbulenceModel == 'SST-V':
        addKeys4Model = dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'standard',
        k_prod_limiter = 20.,
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'infinit_extrap',
            )

    elif TurbulenceModel == 'BSL':
        addKeys4Model = dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'inactive',
        k_prod_limiter = 20.,
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'infinit_extrap',
            )

    elif TurbulenceModel == 'BSL-V':
        addKeys4Model = dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'inactive',
        k_prod_limiter = 20.,
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'infinit_extrap',
            )

    elif TurbulenceModel == 'smith':
        addKeys4Model = dict(
        turbmod        = 'smith',
        k_prod_compute = 'from_sij',
            )

    elif TurbulenceModel == 'SST-2003-LM2009':
        addKeys4Model = dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'std_sij',
        k_prod_limiter = 10.,
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'infinit_extrap',
        trans_mod      = 'menter',
            )

    elif TurbulenceModel == 'SSG/LRR-RSM-w2012':
        addKeys4Model = dict(
        turbmod        = 'rsm',
        rsm_name       = 'ssg_lrr_bsl',
        sst_version    = 'std_sij',
        k_prod_limiter = 10.,
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'infinit_extrap',
        trans_mod      = 'menter',
            )

    # Transition Settings
    if TransitionMode == 'NonLocalCriteria-LSTT':
        if TurbulenceModel == 'SST-2003-LM2009':
            raise AttributeError(FAIL+"Modeling incoherency! cannot make Non-local transition criteria with Menter-Langtry turbulence model"+ENDC)
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
        intermit_region        = 'LSTT', # TODO: Not read in fullCGNS -> make ticket
        intermittency_form     = 'LSTT19',
        trans_h_crit_ahdgl     = 2.8,
        ahd_n_extract          = 'active',
            ))

        if TurbulenceModel in ('BSL','BSL-V','SST-2003','SST','SST-V','Wilcox2006-klim'):  addKeys4Model['prod_omega_red'] = 'active'

    # Transition Settings
    if TransitionMode == 'NonLocalCriteria-Step':
        if TurbulenceModel == 'SST-2003-LM2009':
            raise AttributeError(FAIL+"Modeling incoherency! cannot make Non-local transition criteria with Menter-Langtry turbulence model"+ENDC)
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

        if TurbulenceModel in ('BSL','BSL-V','SST-2003','SST','SST-V','Wilcox2006-klim'):  addKeys4Model['prod_omega_red'] = 'active'


    elif TransitionMode == 'Imposed':
        if TurbulenceModel == 'SST-2003-LM2009':
            raise AttributeError(FAIL+"Modeling incoherency! cannot make imposed transition with Menter-Langtry turbulence model"+ENDC)
        addKeys4Model.update(dict(
        intermittency       = 'full',
        interm_thick_coef   = 1.2,
        delta_compute       = elsAkeysModel['delta_compute'],
        vortratiolim        = elsAkeysModel['vortratiolim'],
        shearratiolim       = elsAkeysModel['shearratiolim'],
        pressratiolim       = elsAkeysModel['pressratiolim'],
        intermittency_form  = 'LSTT19',
            ))

        if TurbulenceModel in ('BSL','BSL-V','SST-2003','SST','SST-V','Wilcox2006-klim'):  addKeys4Model['prod_omega_red'] = 'active'

    elsAkeysModel.update(addKeys4Model)

    return elsAkeysModel


def getElsAkeysNumerics(ReferenceValues, NumericalScheme='jameson',
        TimeMarching='steady', inititer=1, Niter=30000,
        CFLparams=dict(vali=1.,valf=10.,iteri=1,iterf=1000,function_type='linear'),
        timestep=0.01, useBodyForce=False, useChimera=False):
    '''
    Get the Numerics elsA keys as a Python dictionary.

    Parameters
    ----------

        ReferenceValues : dict
            as got from :py:func:`computeReferenceValues`

        NumericalScheme : str
            one of: (``'jameson'``, ``'ausm+'``, ``'roe'``)

        TimeMarching : str
            One of: (``'steady'``, ``'gear'``, ``'DualTimeStep'``)

        inititer : int
            initial iteration

        Niter : int
            total number of iterations to run

        CFLparams : dict
            indicates the CFL function to be employed

        timestep : float
            timestep for unsteady simulation (in seconds)

        useBodyForce : bool
            :py:obj:`True` if bodyforce is employed

        useChimera : bool
            :py:obj:`True` if chimera (static) is employed

    Returns
    -------

        elsAkeysNumerics : dict
            contains *numerics* object elsA keys/values
    '''
    DIRECTORY_OVERSET='OVERSET' # TODO: adapt

    NumericalScheme = NumericalScheme.lower() # avoid case-type mistakes
    elsAkeysNumerics = dict()
    CFLparams['name'] = 'f_cfl'
    for v in ('vali', 'valf'): CFLparams[v] = float(CFLparams[v])
    for v in ('iteri', 'iterf'): CFLparams[v] = int(CFLparams[v])
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
        )
    elif NumericalScheme == 'ausm+':
        addKeys = dict(
        flux               = 'ausmplus_pmiles',
        ausm_wiggle        = 'inactive',
        ausmp_diss_cst     = 0.04,
        ausmp_press_vel_cst= 0.04,
        ausm_tref          = ReferenceValues['Temperature'],
        ausm_pref          = ReferenceValues['Pressure'],
        ausm_mref          = ReferenceValues['Mach'],
        limiter            = 'third_order',
        )
    elif NumericalScheme == 'roe':
        addKeys = dict(
        flux               = 'roe',
        limiter            = 'valbada',
        psiroe             = 0.01,
        viscous_fluxes     = '5p_cor',
        )
    else:
        raise AttributeError('Numerical scheme shortcut %s not recognized'%NumericalScheme)
    elsAkeysNumerics.update(addKeys)


    if TimeMarching == 'steady':
        addKeys = dict(
        time_algo          = 'steady',
        inititer           = int(inititer),
        niter              = int(Niter),
        global_timestep    = 'inactive',
        ode                = 'backwardeuler',
        implicit           = 'lussorsca',
        ssorcycle          = 4,
        timestep_div       = 'divided',
        cfl_fct            = CFLparams['name'],
        freqcompres        = 1,
        residual_type      = 'explimpl',
        )
        addKeys['.Solver#Function'] = CFLparams
    elif TimeMarching == 'gear':
        addKeys = dict(
        time_algo          = 'gear',
        gear_iteration     = 20,
        timestep           = float(timestep),
        inititer           = int(inititer),
        niter              = int(Niter),
        ode                = 'backwardeuler',
        residual_type      = 'implicit',
        freqcompres        = 1,
        restoreach_cons    = 1e-2,
        viscous_fluxes     = '5p_cor',
            )
    elif TimeMarching == 'DualTimeStep':
        addKeys = dict(
        time_algo          = 'dts',
        timestep           = float(timestep),
        dts_timestep_lim   = 'active',
        inititer           = int(inititer),
        niter              = int(Niter),
        ode                = 'backwardeuler',
        residual_type      = 'implicit',
        cfl_dts            = 20.,
        freqcompres        = 1,
        viscous_fluxes     = '5p_cor',
            )

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
                       chm_interp_depth=2,
                       chm_interpcoef_frozen='active', # TODO: make conditional if provided Motion
                       chm_conn_io='read', # NOTE ticket 8259
                       # # Overset by external files: (should not be used)
                       # chm_ovlp_minimize='inactive',
                       # chm_preproc_method='mask_based',
                       # chm_conn_fprefix=DIRECTORY_OVERSET+'/OvstData',

                       )

    addKeys.update(dict(
    multigrid     = 'none',
    t_harten      = 0.01,
    harten_type   = 2,  # see Development #7765 on https://elsa-e.onera.fr/issues/7765
                        # Incompability between default value harten_type=1
                        # and default value vel_formulation='absolute'
    muratiomax    = 1.e+20,
        ))

    ReferenceStateTurbulence = ReferenceValues['ReferenceStateTurbulence']
    TurbulenceCutoff         = ReferenceValues['TurbulenceCutoff']
    for i in range(len(ReferenceStateTurbulence)):
        addKeys['t_cutvar%d'%(i+1)] = TurbulenceCutoff*ReferenceStateTurbulence[i]
    elsAkeysNumerics.update(addKeys)

    return elsAkeysNumerics


def newCGNSfromSetup(t, AllSetupDictionaries, initializeFlow=True,
                     FULL_CGNS_MODE=False, dim=3):
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

        initializeFlow : bool
            if :py:obj:`True`, calls :py:func:`newFlowSolutionInit`, which
            creates ``FlowSolution#Init`` fields used for initialization of the flow

        FULL_CGNS_MODE : bool
            if :py:obj:`True`, add elsa keys in ``.Solver#Compute`` CGNS container

        dim : int
            dimension of the problem (``2`` or ``3``).

    Returns
    -------

        tNew : PyTree
            CGNS tree containing all required data for elsA computation

    '''
    t = I.copyRef(t)

    addSolverBC(t)
    addTrigger(t)
    addExtractions(t, AllSetupDictionaries['ReferenceValues'],
                      AllSetupDictionaries['elsAkeysModel'])
    addReferenceState(t, AllSetupDictionaries['FluidProperties'],
                         AllSetupDictionaries['ReferenceValues'])
    addGoverningEquations(t, dim=dim) # TODO replace dim input by elsAkeysCFD['config'] info
    if initializeFlow:
        newFlowSolutionInit(t, AllSetupDictionaries['ReferenceValues'])
    if FULL_CGNS_MODE:
        addElsAKeys2CGNS(t, [AllSetupDictionaries['elsAkeysCFD'],
                             AllSetupDictionaries['elsAkeysModel'],
                             AllSetupDictionaries['elsAkeysNumerics']])

    AllSetupDictionaries['ReferenceValues']['NProc'] = int(max(getProc(t))+1)
    AllSetupDictionaries['ReferenceValues']['CoreNumberPerNode'] = 28

    writeSetup(AllSetupDictionaries)

    return t

def newRestartFieldsFromCGNS(t):
    '''
    .. warning:: this function interface will change
    '''
    print('invoking EndOfRun from restart')
    to = I.copyRef(t)
    for zone in I.getZones(to):
        FlowSolutionInit = I.getNodeFromName1(zone, 'FlowSolution#Init')
        if not FlowSolutionInit:
            I.printTree(zone, color=True)
            raise ValueError(zone[0])

        '''
        FlowSolutionCenters = I.getNodeFromName1(zone, 'FlowSolution#Centers')

        if FlowSolutionCenters:
            I._rmNodesByType(FlowSolutionInit, 'GridLocation_t')
            FlowSolutionCenters[2].extend(FlowSolutionInit[2])
            I.rmNode(zone, FlowSolutionInit)
        else:
            I._renameNode(zone, 'FlowSolution#Init', 'FlowSolution#Centers')
        '''

    return to


def saveMainCGNSwithLinkToOutputFields(t, to, DIRECTORY_OUTPUT='OUTPUT',
                               MainCGNSFilename='main.cgns',
                               FieldsFilename='fields.cgns',
                               MainCGNS_FlowSolutionName='FlowSolution#Init',
                               Fields_FlowSolutionName='FlowSolution#Init',
                               writeOutputFields=True):
    '''
    Saves the ``main.cgns`` file including linsk towards ``OUTPUT/fields.cgns``
    file, which contains ``FlowSolution#Init`` fields.

    Parameters
    ----------

        t : PyTree
            fully preprocessed PyTree

        to : PyTree
            reference copy of t

            .. warning:: this input will be removed in future as it is useless

        DIRECTORY_OUTPUT : str
            folder containing the file ``fields.cgns``

            .. note:: it is advised to use ``'OUTPUT'``

        MainCGNSFilename : str
            name for main CGNS file.

            .. note:: it is advised to use ``'main.cgns'``

        FieldsFilename : str
            name of CGNS file containing initial fields

            .. note:: it is advised to use ``'fields.cgns'``

        MainCGNS_FlowSolutionName : str
            name of container of initial fields at ``main.cgns``

            .. important:: it is strongly recommended using ``'FlowSolution#Init'``

        Fields_FlowSolutionName : str
            name of container of initial fields at ``fields.cgns``

            .. important:: it is strongly recommended using ``'FlowSolution#Init'``

        writeOutputFields : bool
            if :py:obj:`True`, write ``fields.cgns`` file

    Returns
    -------

        None - None
            files ``main.cgns`` and eventually ``OUTPUT/fields.cgns`` are written
    '''
    print('gathering links between main CGNS and fields')
    AllCGNSLinks = []
    for b in I.getBases(t):
        for z in b[2]:
            fs = I.getNodeFromName1(z,MainCGNS_FlowSolutionName)
            if not fs: continue
            for field in I.getChildren(fs):
                if I.getType(field) == 'DataArray_t':
                    currentNodePath='/'.join([b[0], z[0], fs[0], field[0]])
                    targetNodePath=currentNodePath.replace(MainCGNS_FlowSolutionName,
                                                           Fields_FlowSolutionName)
                    # TODO: Cassiopee BUG ! targetNodePath must be identical
                    # to currentNodePath...
                    AllCGNSLinks += [['.',
                                      DIRECTORY_OUTPUT+'/'+FieldsFilename,
                                      '/'+targetNodePath,
                                      currentNodePath]]


    try: os.makedirs(DIRECTORY_OUTPUT)
    except: pass
    print('saving PyTrees with links')
    to = I.copyRef(t)
    I._renameNode(to, 'FlowSolution#Centers', 'FlowSolution#Init')
    if writeOutputFields:
        C.convertPyTree2File(to, os.path.join(DIRECTORY_OUTPUT, FieldsFilename))
    C.convertPyTree2File(t, MainCGNSFilename, links=AllCGNSLinks)


def addSolverBC(t):
    '''
    Increase family integer value to ``.Solver#BC`` at ``FamilyBC_t`` nodes

    Parameters
    ----------

        t : PyTree
            the main tree. It is modified.
    '''
    FamilyNodes = I.getNodesFromType2(t, 'Family_t')
    for i, fn in enumerate(FamilyNodes):
        if I.getNodeFromType(fn, 'FamilyBC_t'):
            J.set(fn, '.Solver#BC', family=i+1)


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


def addExtractions(t, ReferenceValues, elsAkeysModel, extractCoords=True):
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

    '''
    addSurfacicExtractions(t, ReferenceValues, elsAkeysModel)
    addFieldExtractions(t, ReferenceValues, extractCoords=extractCoords)
    EP._addGlobalConvergenceHistory(t)


def addSurfacicExtractions(t, ReferenceValues, elsAkeysModel):
    '''
    Include surfacic extraction information to CGNS tree using
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
    '''

    FamilyNodes = I.getNodesFromType2(t, 'Family_t')
    for FamilyNode in FamilyNodes:
        FamilyName = I.getName( FamilyNode )
        BCType = getFamilyBCTypeFromFamilyBCName(t, FamilyName)
        if not BCType or not 'BCWall' in BCType: continue

        # TODO 'bl_ue' CGNS extraction FAILS in v4.2.01
        WallVariables = ('normalvector'
                         ' SkinFrictionX SkinFrictionY SkinFrictionZ'
                         ' psta'
                         ' flux_rou flux_rov flux_row'
                         ' torque_rou torque_rov torque_row')

        if 'Inviscid' not in BCType:
            WallVariables += ' bl_quantities_2d yplusmeshsize'

            TransitionMode = ReferenceValues['TransitionMode']

            if TransitionMode == 'NonLocalCriteria-LSTT':
                WallVariables += ' intermittency clim how origin lambda2 turb_level n_tot_ag n_crit_ag r_tcrit_ahd r_theta_t1 line_status crit_indicator'

            elif TransitionMode == 'Imposed':
                WallVariables += ' intermittency clim'

        print('setting .Solver#Output to FamilyNode '+FamilyNode[0])
        J.set(FamilyNode, '.Solver#Output',
            var           = WallVariables,
            period        = 1,
            writingmode   = 2,
            loc           = 'interface',
            delta_compute = elsAkeysModel['delta_compute'],
            vortratiolim  = elsAkeysModel['vortratiolim'],
            shearratiolim = elsAkeysModel['shearratiolim'],
            pressratiolim = elsAkeysModel['pressratiolim'],
            fluxcoeff     = 1.0,
            torquecoeff   = 1.0,
            pinf          = ReferenceValues['Pressure'],
            xtorque       = 0.0,
            ytorque       = 0.0,
            ztorque       = 0.0,
            force_extract = 1,
            )


def addFieldExtractions(t, ReferenceValues, extractCoords=False):
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

    '''

    Fields2Extract = ReferenceValues['Fields'] + ReferenceValues['FieldsAdditionalExtractions']

    '''
    # Do not respect order, see #7764
    EP._addFlowSolutionEoR(t,
        name='',
        variables=ReferenceValues['Fields']+' '+
                  ReferenceValues['FieldsAdditionalExtractions'],
        protocol='iteration',
        writingFrame='absolute')
    '''
    I._rmNodesByName(t, 'FlowSolution#EndOfRun')
    for zone in I.getZones(t):
        if extractCoords:
            EoRnode = I.createNode('FlowSolution#EndOfRun#Coords', 'FlowSolution_t',
                                    parent=zone)
            GridLocationNode = I.createNode('GridLocation','GridLocation_t',
                                            value='Vertex', parent=EoRnode)
            for fieldName in ('CoordinateX', 'CoordinateY', 'CoordinateZ'):
                I.createNode(fieldName, 'DataArray_t', value=None, parent=EoRnode)
            J.set(EoRnode, '.Solver#Output',
                  period=1,
                  writingmode=2,
                  writingframe='absolute',
                  force_extract=1,
                   )

        EoRnode = I.createNode('FlowSolution#EndOfRun', 'FlowSolution_t',
                                parent=zone)
        GridLocationNode = I.createNode('GridLocation','GridLocation_t',
                                        value='CellCenter', parent=EoRnode)
        for fieldName in Fields2Extract:
            I.createNode(fieldName, 'DataArray_t', value=None, parent=EoRnode)
        J.set(EoRnode, '.Solver#Output',
              period=1,
              writingmode=2,
              writingframe='absolute',
              force_extract=1,
               )


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


def newFlowSolutionInit(t, ReferenceValues):
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

    Lines = ['#!/usr/bin/python\n']
    Lines = ["'''\nMOLA %s setup.py file automatically generated in PREPROCESS\n'''\n"%MOLA.__version__]

    for SetupDict in AllSetupDictionaries:
        Lines+=[SetupDict+"="+pprint.pformat(AllSetupDictionaries[SetupDict])+"\n"]

    AllLines = '\n'.join(Lines)

    with open(setupFilename,'w') as f: f.write(AllLines)

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
    Lines = ['#!/usr/bin/python\n']
    Lines = ["'''\nMOLA %s setup.py file automatically generated in PREPROCESS\n'''\n"%MOLA.__version__]

    for SetupItem in dir(setup):
        if not SetupItem.startswith('_'):
            Lines+=[SetupItem+"="+pprint.pformat(getattr(setup, SetupItem))+"\n"]

    AllLines = '\n'.join(Lines)

    with open(setupFilename,'w') as f: f.write(AllLines)

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
    FieldsNames = ReferenceValues['Fields']
    for i in range(len(FieldsNames)):
        RefState += [[FieldsNames[i],ReferenceValues['ReferenceState'][i]]]
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

    OPL_ns = I.getNodesFromName(t,'OrphanPointList')
    for opl in OPL_ns:
        ID_node, _ = I.getParentOfNode(t, opl)
        print(J.WARN+'removing %s'%opl[0]+J.ENDC)
        I.rmNode(t,opl)

    for zone in I.getZones(t):
        for OversetNodeName in OversetNodeNames:
            OversetNodes = I.getNodesFromName(zone, OversetNodeName)
            for OversetNode in OversetNodes:
                OversetValue = OversetNode[1]
                if OversetValue is None or len(OversetValue)==0:
                    if not silent:
                        STR = J.WARN, zone[0], OversetNode[0], J.ENDC
                        print('%szone %s removing empty overset %s node%s'%STR)
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
    YawAxis    = np.array(YawAxis, dtype=np.float)
    YawAxis   /= np.sqrt(YawAxis.dot(YawAxis))

    # Pitch axis may be approximate
    PitchAxis  = np.array(PitchAxis, dtype=np.float)
    PitchAxis /= np.sqrt(PitchAxis.dot(PitchAxis))

    # Roll axis is inferred
    RollAxis  = np.cross(PitchAxis, YawAxis)
    RollAxis /= np.sqrt(RollAxis.dot(RollAxis))

    # correct Pitch axis
    PitchAxis = np.cross(YawAxis, RollAxis)
    PitchAxis /= np.sqrt(PitchAxis.dot(PitchAxis))

    # FlowLines are used to infer the final flow direction
    DragLine = D.line((0,0,0),(1,0,0),2)
    SideLine = D.line((0,0,0),(0,1,0),2)
    LiftLine = D.line((0,0,0),(0,0,1),2)
    FlowLines = [DragLine, SideLine, LiftLine]

    # Put FlowLines in Aircraft's frame
    zero = (0,0,0)
    InitialFrame =  [       [1,0,0],         [0,1,0],       [0,0,1]]
    AircraftFrame = [list(RollAxis), list(PitchAxis), list(YawAxis)]
    T._rotate(FlowLines, zero, InitialFrame, AircraftFrame)

    # Apply Flow angles with respect to Airfraft's frame
    T._rotate(FlowLines, zero, list(PitchAxis), -AngleOfAttackDeg)
    T._rotate(FlowLines, zero,   list(YawAxis),  AngleOfSlipDeg)

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
    FamilyNode = I.getNodeFromName3(t, FamilyBCName)
    if not FamilyNode: return

    FamilyBCNode = I.getNodeFromName1(FamilyNode, 'FamilyBC')
    if not FamilyBCNode: return

    FamilyBCNodeType = I.getValue(FamilyBCNode)
    if FamilyBCNodeType != 'UserDefined': return FamilyBCNodeType

    BCnodes = I.getNodesFromType(t, 'BC_t')
    BCType = None
    for BCnode in BCnodes:
        FamilyNameNode = I.getNodeFromName1(BCnode, 'FamilyName')
        if not FamilyNameNode: continue

        FamilyNameValue = I.getValue( FamilyNameNode )
        if FamilyNameValue == FamilyBCName:
            BCType = I.getValue( BCnode )
            break

    return BCType


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
    if hasAnyNearMatch(InputMeshes):
        print('adapting NearMatch to elsA...')
        EP._adaptNearMatch(t)


    if hasAnyPeriodicMatch(InputMeshes):
        print('adapting PeriodicMatch to elsA...')
        EP._adaptPeriodicMatch(t, clean=True)

    if hasAnyOversetData(InputMeshes):
        print('adapting overset data to elsA...')
        EP._overlapGC2BC(t)
        EP._rmGCOverlap(t)
        EP._fillNeighbourList(t, sameBase=0)
        EP._prefixDnrInSubRegions(t)
        removeEmptyOversetData(t, silent=False)


def hasAnyNearMatch(InputMeshes):
    '''
    Determine if at least one item in **InputMeshes** has a connectivity of
    type ``NearMatch``.

    Parameters
    ----------

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
        try:
            OversetOptions = meshInfo['OversetOptions']
            return True
        except KeyError:
            continue

    return False


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
