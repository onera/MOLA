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

import os
from itertools import product
import numpy as np
import mola.naming_conventions as names
import mola.cfd.preprocess.overset.UnsteadyOverset as UO
from mola.cfd.postprocess.extractions_with_cassiopee.extract_bc import getWalls
import mola.pytree.InternalShortcuts as J
import mola.mesh.ExtractSurfacesProcessor as ESP
import mola.mesh.surface as GSD
import mola.mesh.volume as GVD
from mola.logging import mola_logger

from treelab import cgns

import Converter.PyTree as C
import Converter.Internal as I
import Converter.elsAProfile as EP
import Connector.PyTree as X
import Transform.PyTree as T
import Generator.PyTree as G
import Intersector.PyTree as XOR
import Geom.PyTree as D
import Post.PyTree as P

def addOversetData(t, InputMeshes, depth=2, optimizeOverlap=False,
                   prioritiesIfOptimize=[], double_wall=0,
                   saveMaskBodiesTree=True,
                   overset_in_CGNS=False, # see elsA #10545
                   CHECK_OVERSET=True,
                   run_directory='.',
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
                beware of `elsA bug 10545 <https://elsa.onera.fr/issues/10545>`__
        
        CHECK_OVERSET : bool
            if :py:obj:`True`, then make an extrapolated-orphan cell diagnosis
            when making unsteady motion overset preprocess.

    Returns
    -------

        t : PyTree
            new pytree including ``cellN`` values at ``FlowSolution#Centers``
            and elsA's ``ID*`` nodes including interpolation coefficients information.

    '''

    if not hasAnyOversetData(InputMeshes): return t  

    mola_logger.info("  📎 adding overset data", rank=0)

    overset_path = os.path.join(run_directory,names.DIRECTORY_OVERSET)
    try: os.makedirs(overset_path)
    except: pass

    print('building masking bodies...')
    baseName2BodiesDict = getMaskingBodiesAsDict(t, InputMeshes)

    bodies = []
    for meshInfo in InputMeshes:
        bodies.extend(baseName2BodiesDict[meshInfo['Name']])

    if saveMaskBodiesTree:
        save_masks(baseName2BodiesDict, overset_path)

    BlankingMatrix = getBlankingMatrix(bodies, InputMeshes)

    # TODO -> RB: applyBCOverlaps after cellN2OversetHoles so that 
    # output files assign cellN=0 on masked overlaps instead of cellN=2
    t = X.applyBCOverlaps(t, depth=depth)

    print('Static blanking...')
    t_blank = staticBlanking(t, bodies, BlankingMatrix, InputMeshes)
    if hasAnyOversetMotion(InputMeshes):
        StaticBlankingMatrix  = getBlankingMatrix(bodies, InputMeshes,
                                    StaticOnly=True, FullBlankMatrix=BlankingMatrix)
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
        prefixFile = os.path.join(overset_path,'overset')

    print('cellN2OversetHoles and applyBCOverlaps...')
    t = X.cellN2OversetHoles(t)
    t = X.applyBCOverlaps(t, depth=depth) # TODO ->  see previous RB note
    print('... cellN2OversetHoles and applyBCOverlaps done.')

    if hasAnyOversetMotion(InputMeshes):
        add_mola_input_mesh_info(t,InputMeshes)

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
        EP.buildMaskFiles(t, fileDir=overset_path, prefixBase=True)
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
            diagnosis_file = os.path.join(overset_path,
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

    if hasAnyNearMatch(t, InputMeshes):
        print(J.CYAN+'adapting NearMatch to elsA'+J.ENDC)
        EP._adaptNearMatch(t)

    print('adapting overset data to elsA...')
    # EP._overlapGC2BC(t) # may provoke BC duplication in mola v2, not required then ?
    from mola.pytree.user import checker; checker.assert_unique_siblings_names(t)
    EP._rmGCOverlap(t)
    EP._fillNeighbourList(t, sameBase=0)
    _hackChimGroupFamilies(t)
    EP._prefixDnrInSubRegions(t)
    removeEmptyOversetData(t, silent=False)


    return cgns.castNode(t)

def add_mola_input_mesh_info(t,InputMeshes):
    for base in I.getBases(t):
        base_name = base[0]
        meshInfo = getMeshInfoFromBaseName(base_name, InputMeshes)
        if 'OversetMotion' in meshInfo:
            J.set(base,'.MOLA#InputMesh',**dict(OversetMotion=meshInfo['OversetMotion']))
        else:
            J.set(base,'.MOLA#InputMesh',**dict(Nothing=0))


def save_masks( base_name_to_bodies : dict, overset_path):
    treeLikeList = []
    for bn, bodies in base_name_to_bodies.items():
        bodies_zones = I.getZones(bodies)
        treeLikeList.extend([bn, bodies_zones])
    tMask = I.copyRef(C.newPyTree(treeLikeList))
    I._correctPyTree(tMask,level=3)
    C.convertPyTree2File(tMask, os.path.join(overset_path,
                                            'CHECK_ME_mask.cgns'))    

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

def _flattenBodies( bodies ):
    bodies_zones = []
    for b in bodies:
        if len(b) == 0:
            raise RuntimeError(f"{b}")

        if isinstance(b[0],str):
            bodies_zones.append(b)
        elif isinstance(b[0],list):
            for bb in b:
                if not isinstance(bb[0],str):
                    raise ValueError('wrong bodies container')
                bodies_zones.append(bb)
    return bodies_zones

def getBlankingMatrix(bodies, InputMeshes, StaticOnly=False, FullBlankMatrix=None):
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
    

    BaseNames = [meshInfo['Name'] for meshInfo in InputMeshes]
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
        
    for i, meshInfo in enumerate(InputMeshes):
        
        # masking protection using key "OnlyMaskedByWalls"
        try: OnlyMaskedByWalls = meshInfo['OversetOptions']['OnlyMaskedByWalls']
        except KeyError: continue
        
        if OnlyMaskedByWalls:
            for j, BodyName in enumerate(BodyNames):
                if not BodyName.startswith('wall'):
                    BlankingMatrix[i, j] = 0

    if StaticOnly:
        for i, meshInfo in enumerate(InputMeshes):
            MovingBase = True if 'OversetMotion' in meshInfo else False

            for j, BodyName in enumerate(BodyNames):
                BodyParentBaseName = getBodyParentBaseName(BodyName)
                bodyInfo = getMeshInfoFromBaseName(BodyParentBaseName,InputMeshes)
                MovingBody = True if 'OversetMotion' in bodyInfo else False

                # if MovingBase and MovingBody: rigidMotionToBeImplementedHere # TODO
                
                if not MovingBody and not MovingBase: continue # will make static blank

                BlankingMatrix[i,j] = 0


        # HACK elsA does not allow for static blanking a base if already dynamic blanked
        # by other moving bodies
        Dynamic = FullBlankMatrix - BlankingMatrix
        for i in range(Dynamic.shape[0]):
            if any(Dynamic[i,:]):
                BlankingMatrix[i,:] = 0
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
            if 'OversetMotion' in meshInfo:
                return True
    return False

def getBodyName(body):
    if isinstance(body[0],str): return body[0]
    else: return body[0][0]

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
        if meshInfo['Name'] == baseName:
            return meshInfo

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

    if SuffixTag:
        for m in I.getZones(mask):
            m[0] = SuffixTag

    return mask

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

def getBodyParentBaseName(BodyName):
    return '-'.join(BodyName.split('-')[1:])

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

def _hackChimGroupFamilies(t):
    '''
    This is a HACK for circumventing `elsA Issue #11552`__
    also check `MOLA Issue #196 <https://gitlab.onera.net/numerics/mola/-/issues/196>`__
    '''
    for family_name_node in I.getNodesFromType(t,'FamilyName_t'):
        family_name_node_value = I.getValue(family_name_node)
        if family_name_node_value.startswith('ChimGroup'):
            family_name_node[0] = family_name_node_value
            family_name_node[3] = 'AdditionalFamilyName_t'


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
            isNearMatch = ConnectionInfo['Type'] == 'NearMatch'
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
