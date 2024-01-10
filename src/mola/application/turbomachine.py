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



def parametrizeChannelHeight(t, lin_axis=None):
    '''
    Compute the variable *ChannelHeight* from a mesh PyTree **t**. This function
    relies on the turbo module.

    .. important::

        Dependency to *turbo* module. See file:///stck/jmarty/TOOLS/turbo/doc/html/index.html

    Parameters
    ----------

        t : PyTree
            input mesh tree

        lin_axis : :py:obj:`None` or :py:class:`str`
            Axis for linear configuration.
            If :py:obj:`None`, the configuration is annular (default case), else
            the configuration is linear.
            'XY' means that X-axis is the streamwise direction and Y-axis is the
            spanwise direction.(see turbo documentation)

    Returns
    -------

        t : PyTree
            modified tree

    '''
    import turbo.height as TH

    def plot_hub_and_shroud_lines(t):
        # Get geometry
        hub     = I.getNodeFromName(t, 'Hub')
        xHub    = I.getValue(I.getNodeFromName(hub, 'CoordinateX'))
        yHub    = I.getValue(I.getNodeFromName(hub, 'CoordinateY'))
        shroud  = I.getNodeFromName(t, 'Shroud')
        xShroud = I.getValue(I.getNodeFromName(shroud, 'CoordinateX'))
        yShroud = I.getValue(I.getNodeFromName(shroud, 'CoordinateY'))
        # Import matplotlib
        import matplotlib.pyplot as plt
        # Plot
        plt.figure()
        plt.plot(xHub, yHub, '-', label='Hub')
        plt.plot(xShroud, yShroud, '-', label='Shroud')
        plt.axis('equal')
        plt.grid()
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        # Save
        plt.savefig('shroud_hub_lines.png', dpi=150, bbox_inches='tight')
        return 0

    print(J.CYAN + 'Add ChannelHeight in the mesh...' + J.ENDC)
    OLD_FlowSolutionNodes = I.__FlowSolutionNodes__
    I.__FlowSolutionNodes__ = 'FlowSolution#Height'

    fd = 2 # stderr file identifier
    def _redirect_stderr(to):
        sys.stderr.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stderr = os.fdopen(fd, 'w') # Python writes to fd


    silence_stdout = J.OutputGrabber(stream=sys.stdout)
    silence_stderr = J.OutputGrabber(stream=sys.stderr)
    message = None

    with silence_stdout:

        if not lin_axis:
            numpy_major, numpy_minor, numpy_micro = np.__version__.split('.')
            if int(numpy_major) < 2 and int(numpy_minor) > 23:
                # HACK see https://elsa-e.onera.fr/issues/11277
                if PRE.hasAnyUnstructuredZones(t):
                    message = J.WARN + 'Cannot use method=2 of generateHLinesAxial because of https://elsa-e.onera.fr/issues/11277.\n'
                    message += 'Use the function parametrizeChannelHeight on spiro (not on the login machine!).' + J.ENDC
                    raise Exception(message)
                else:
                    message = J.WARN + 'Cannot use method=2 of generateHLinesAxial because of https://elsa-e.onera.fr/issues/11277.\n'
                    message += 'Switch to method=1 (only for structured grids).\n'
                    message += 'To use method=2, use the function parametrizeChannelHeight on spiro (not on the login machine!).' + J.ENDC
                    print(message)
                    endlinesTree = TH.generateHLinesAxial(t, filename='shroud_hub_lines.plt', method=1)
            else:
                endlinesTree = TH.generateHLinesAxial(t, filename='shroud_hub_lines.plt', method=2)
            try: 
                plot_hub_and_shroud_lines(endlinesTree)
            except: 
                pass

            # - Generation of the mask file
            with silence_stderr:
                m = TH.generateMaskWithChannelHeight(t, 'shroud_hub_lines.plt')
            os.remove('shroud_hub_lines.plt')
        else:
            m = TH.generateMaskWithChannelHeightLinear(t, lin_axis=lin_axis)

        # - Generation of the ChannelHeight field
        TH._computeHeightFromMask(t, m, writeMask='mask.cgns', lin_axis=lin_axis)
    
    if message: print(message)

    I.__FlowSolutionNodes__ = OLD_FlowSolutionNodes
    print(J.GREEN + 'done.' + J.ENDC)
    return t

def duplicate(tree, rowFamily, nBlades, nDupli=None, merge=False, axis=(1,0,0),
    verbose=1, container='FlowSolution#Init',
    vectors2rotate=[['VelocityX','VelocityY','VelocityZ'],['MomentumX','MomentumY','MomentumZ']]):
    '''
    Duplicate **nDupli** times the domain attached to the family **rowFamily**
    around the axis of rotation.

    Parameters
    ----------

        tree : PyTree
            tree to modify

        rowFamily : str
            Name of the CGNS family attached to the row domain to Duplicate

        nBlades : int
            Number of blades in the row. Used to compute the azimuthal length of
            a blade sector.

        nDupli : int
            Number of duplications to make

            .. warning:: This is the number of duplication of the input mesh
                domain, not the wished number of simulated blades. Keep this
                point in mind if there is already more than one blade in the
                input mesh.

        merge : bool
            if :py:obj:`True`, merge all the blocks resulting from the
            duplication.

            .. tip:: This option is useful is the mesh is to split and if a
                globborder will be defined on a BC of the duplicated domain. It
                allows the splitting procedure to provide a 'matricial' ordering
                (see `elsA Tutorial about globborder <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/globborder.html>`_)

        axis : tuple
            axis of rotation given as a 3-tuple of integers or floats

        verbose : int
            level of verbosity:

                * 0: no print

                * 1: print the number of duplications for row **rowFamily** and
                  the total number of blades.

                * 2: print also the name of all duplicated zones

        container : str
            Name of the FlowSolution container to rotate. Default is 'FlowSolution#Init'

        vectors2rotate : :py:class:`list` of :py:class:`list` of :py:class:`str`
            list of vectors to rotate. Each vector is a list of three strings,
            corresponding to each components.
            The default value is:

            >>> vectors2rotate = [['VelocityX','VelocityY','VelocityZ'],
            >>>                   ['MomentumX','MomentumY','MomentumZ']]

            .. note:: Rotation of vectors is done with Cassiopee function
                      Transform.rotate. However, it is not useful to put the
                      prefix 'centers:'. It will be added automatically in the
                      function.

    '''
    OLD_FlowSolutionCenters = I.__FlowSolutionCenters__
    I.__FlowSolutionCenters__ = container

    if nDupli is None:
        nDupli = nBlades # for a 360 configuration
    if nDupli == nBlades:
        if verbose>0: print('Duplicate {} over 360 degrees ({} blades in row)'.format(rowFamily, nBlades))
    else:
        if verbose>0: print('Duplicate {} on {} blades ({} blades in row)'.format(rowFamily, nDupli, nBlades))

    check = False
    vectors = []
    for vec in vectors2rotate:
        vectors.append(vec)
        vectors.append(['centers:'+v for v in vec])

    if I.getType(tree) == 'CGNSBase_t':
        bases = [tree]
    else:
        bases = I.getBases(tree)

    for base in bases:
        for zone in I.getZones(base):
            zone_name = I.getName(zone)
            FamilyNameNode = I.getNodeFromName1(zone, 'FamilyName')
            if not FamilyNameNode: continue
            zone_family = I.getValue(FamilyNameNode)
            if zone_family == rowFamily:
                if verbose>1: print('  > zone {}'.format(zone_name))
                check = True
                zones2merge = [zone]
                for n in range(nDupli-1):
                    ang = 360./nBlades*(n+1)
                    rot = T.rotate(I.copyNode(zone),(0.,0.,0.), axis, ang, vectors=vectors)
                    I.setName(rot, "{}_{}".format(zone_name, n+2))
                    I._addChild(base, rot)
                    zones2merge.append(rot)
                if merge:
                    for node in zones2merge:
                        I.rmNode(base, node)
                    tree_dist = T.merge(zones2merge, tol=1e-8)
                    for i, node in enumerate(I.getZones(tree_dist)):
                        I._addChild(base, node)
                        disk_block = I.getNodeFromName(base, I.getName(node))
                        disk_block[0] = '{}_{:02d}'.format(zone_name, i)
                        I.createChild(disk_block, 'FamilyName', 'FamilyName_t', value=rowFamily)
    if merge: PRE.autoMergeBCs(tree)

    I.__FlowSolutionCenters__ = OLD_FlowSolutionCenters
    assert check, 'None of the zones was duplicated. Check the name of row family'

def duplicate_flow_solution(t, TurboConfiguration):
    '''
    Duplicated the input PyTree **t**, already initialized.
    This function perform the following operations:

    #. Duplicate the mesh

    #. Initialize the different blade sectors by rotating the ``FlowSolution#Init``
       node available in the original sector(s)

    #. Update connectivities and periodic boundary conditions

    .. warning:: This function does not rotate vectors in BCDataSet nodes.

    Parameters
    ----------

        t : PyTree
            input tree already initialized, but before setting boundary conditions

        TurboConfiguration : dict
            dictionary as provided by :py:func:`getTurboConfiguration`

    Returns
    -------

        t : PyTree
            tree after duplication
    '''
    # Remove connectivities and periodic BCs
    I._rmNodesByType(t, 'GridConnectivity1to1_t')

    angles4ConnectMatchPeriodic = []
    for row, rowParams in TurboConfiguration['Rows'].items():
        nBlades = rowParams['NumberOfBlades']
        nDupli = rowParams['NumberOfBladesSimulated']
        nMesh = rowParams['NumberOfBladesInInitialMesh']
        if nDupli > nMesh:
            duplicate(t, row, nBlades, nDupli=nDupli, axis=(1,0,0))

        angle = 360. / nBlades * nDupli
        if not np.isclose(angle, 360.):
            angles4ConnectMatchPeriodic.append(angle)

    # Connectivities
    X.connectMatch(t, tol=1e-8)
    for angle in angles4ConnectMatchPeriodic:
        # Not full 360 simulation: periodic BC must be restored
        t = X.connectMatchPeriodic(t, rotationAngle=[angle, 0., 0.], tol=1e-8)

    # WARNING: Names of BC_t nodes must be unique to use PyPart on globborders
    for l in [2,3,4]: I._correctPyTree(t, level=l)

    return t

def computeAzimuthalExtensionFromFamily(t, FamilyName):
    '''
    Compute the azimuthal extension in radians of the mesh **t** for the row **FamilyName**.

    .. warning:: This function needs to calculate the surface of the slice in X
                 at Xmin + 5% (Xmax - Xmin). If this surface is crossed by a
                 solid (e.g. a blade) or by the inlet boundary, the function
                 will compute a wrong value of the number of blades inside the
                 mesh.

    Parameters
    ----------

        t : PyTree
            mesh tree

        FamilyName : str
            Name of the row, identified by a ``FamilyName``.

    Returns
    -------

        deltaTheta : float
            Azimuthal extension in radians

    '''
    # Extract zones in family
    zonesInFamily = C.getFamilyZones(t, FamilyName)
    # Slice in x direction at middle range
    xmin = C.getMinValue(zonesInFamily, 'CoordinateX')
    xmax = C.getMaxValue(zonesInFamily, 'CoordinateX')
    sliceX = P.isoSurfMC(zonesInFamily, 'CoordinateX', value=xmin+0.05*(xmax-xmin))
    # Compute Radius
    C._initVars(sliceX, '{Radius}=({CoordinateY}**2+{CoordinateZ}**2)**0.5')
    Rmin = C.getMinValue(sliceX, 'Radius')
    Rmax = C.getMaxValue(sliceX, 'Radius')
    # Compute surface
    SurfaceTree = C.convertArray2Tetra(sliceX)
    SurfaceTree = C.initVars(SurfaceTree, 'ones=1')
    Surface = P.integ(SurfaceTree, var='ones')[0]
    # Compute deltaTheta
    deltaTheta = 2* Surface / (Rmax**2 - Rmin**2)
    return deltaTheta

def getNumberOfBladesInMeshFromFamily(t, FamilyName, NumberOfBlades):
    '''
    Compute the number of blades for the row **FamilyName** in the mesh **t**.

    .. warning:: This function needs to calculate the surface of the slice in X
                 at Xmin + 5% (Xmax - Xmin). If this surface is crossed by a
                 solid (e.g. a blade) or by the inlet boundary, the function
                 will compute a wrong value of the number of blades inside the
                 mesh.

    Parameters
    ----------

        t : PyTree
            mesh tree

        FamilyName : str
            Name of the row, identified by a ``FamilyName``.

        NumberOfBlades : int
            Number of blades of the row **FamilyName** on 360 degrees.

    Returns
    -------

        Nb : int
            Number of blades in **t** for row **FamilyName**

    '''
    # # Extract zones in family
    # zonesInFamily = C.getFamilyZones(t, FamilyName)
    # # Slice in x direction at middle range
    # xmin = C.getMinValue(zonesInFamily, 'CoordinateX')
    # xmax = C.getMaxValue(zonesInFamily, 'CoordinateX')
    # sliceX = P.isoSurfMC(zonesInFamily, 'CoordinateX', value=xmin+0.05*(xmax-xmin))
    # # Compute Radius
    # C._initVars(sliceX, '{Radius}=({CoordinateY}**2+{CoordinateZ}**2)**0.5')
    # Rmin = C.getMinValue(sliceX, 'Radius')
    # Rmax = C.getMaxValue(sliceX, 'Radius')
    # # Compute surface
    # SurfaceTree = C.convertArray2Tetra(sliceX)
    # SurfaceTree = C.initVars(SurfaceTree, 'ones=1')
    # Surface = P.integ(SurfaceTree, var='ones')[0]
    # # Compute deltaTheta
    # deltaTheta = 2* Surface / (Rmax**2 - Rmin**2)
    deltaTheta = computeAzimuthalExtensionFromFamily(t, FamilyName)
    # Compute number of blades in the mesh
    Nb = NumberOfBlades * deltaTheta / (2*np.pi)
    Nb = int(np.round(Nb))
    print('Number of blades in initial mesh for {}: {}'.format(FamilyName, Nb))
    return Nb

def computeFluxCoefByRow(t, ReferenceValues, TurboConfiguration):
    '''
    Compute the parameter **FluxCoef** for boundary conditions (except wall BC)
    and rotor/stator intefaces (``GridConnectivity_t`` nodes).
    **FluxCoef** will be used later to normalize the massflow.

    Modify **ReferenceValues** by adding:

    >>> ReferenceValues['NormalizationCoefficient'][<FamilyName>]['FluxCoef'] = FluxCoef

    for <FamilyName> in the list of BC families, except families of type 'BCWall*'.

    Parameters
    ----------

        t : PyTree
            Mesh tree with boudary conditions families, with a BCType.

        ReferenceValues : dict
            as produced by :py:func:`computeReferenceValues`

        TurboConfiguration : dict
            as produced by :py:func:`getTurboConfiguration`

    '''
    for zone in I.getZones(t):
        FamilyNode = I.getNodeFromType1(zone, 'FamilyName_t')
        if FamilyNode is None:
            continue
        if 'PeriodicTranslation' in TurboConfiguration:
            fluxcoeff = 1.
        else:
            row = I.getValue(FamilyNode)
            try:
                rowParams = TurboConfiguration['Rows'][row]
                fluxcoeff = rowParams['NumberOfBlades'] / float(rowParams['NumberOfBladesSimulated'])
            except KeyError:
                # since a FamilyNode does not necessarily belong to a row
                fluxcoeff = 1.

        for bc in I.getNodesFromType2(zone, 'BC_t')+I.getNodesFromType2(zone, 'GridConnectivity_t'):
            FamilyNameNode = I.getNodeFromType1(bc, 'FamilyName_t')
            if FamilyNameNode is None:
                continue
            FamilyName = I.getValue(FamilyNameNode)
            BCType = PRE.getFamilyBCTypeFromFamilyBCName(t, FamilyName)
            if BCType is None or 'BCWall' in BCType:
                continue
            if not 'NormalizationCoefficient' in ReferenceValues:
                ReferenceValues['NormalizationCoefficient'] = dict()
            ReferenceValues['NormalizationCoefficient'][FamilyName] = dict(FluxCoef=fluxcoeff)

def getReferenceSurface(t, BoundaryConditions, TurboConfiguration):
    '''
    Compute the reference surface (**Surface** parameter in **ReferenceValues**
    :py:class:`dict`) from the inflow family.

    Parameters
    ----------

        t : PyTree
            Input tree

        BoundaryConditions : list
            Boundary conditions to set on the given mesh,
            as given to :py:func:`prepareMainCGNS4ElsA`.

        TurboConfiguration : dict
            Compressor properties, as given to :py:func:`prepareMainCGNS4ElsA`.

    Returns
    -------

        Surface : float
            Reference surface
    '''
    # Get inflow BCs
    InflowBCs = [bc for bc in BoundaryConditions \
        if bc['type'] == 'InflowStagnation' or bc['type'].startswith('inj')]
    # Check unicity
    if len(InflowBCs) != 1:
        MSG = 'Please provide a reference surface as "Surface" in '
        MSG += 'ReferenceValues or provide a unique inflow BC in BoundaryConditions'
        raise Exception(J.FAIL + MSG + J.ENDC)
    # Compute surface of the inflow BC
    InflowFamily = InflowBCs[0]['FamilyName']
    zones = C.extractBCOfName(t, 'FamilySpecified:'+InflowFamily)
    SurfaceTree = C.convertArray2Tetra(zones)
    SurfaceTree = C.initVars(SurfaceTree, 'ones=1')
    Surface = P.integ(SurfaceTree, var='ones')[0]
    if 'PeriodicTranslation' not in TurboConfiguration:
        # Compute normalization coefficient
        zoneName = I.getName(zones[0]).split('/')[0]
        zone = I.getNodeFromName2(t, zoneName)
        row = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
        rowParams = TurboConfiguration['Rows'][row]
        fluxcoeff = rowParams['NumberOfBlades'] / float(rowParams['NumberOfBladesInInitialMesh'])
        # Compute reference surface
        Surface *= fluxcoeff
    print('Reference surface = {} m^2 (computed from family {})'.format(Surface, InflowFamily))

    return Surface

def addMonitoredRowsInExtractions(Extractions, TurboConfiguration):
    '''
    Get the positions of inlet and outlet planes for each row (identified by
    keys **InletPlane** and **OutletPlane** in the row sub-dictionary of
    **TurboConfiguration**) and add them to **Extractions**.

    Parameters
    ----------

        Extractions : list
            List of extractions, each of them being a dictionary.

        TurboConfiguration : dict
            Compressor properties, as given to :py:func:`prepareMainCGNS4ElsA`.

    '''
    # Get the positions of inlet and outlet planes for each row
    # and add them to Extractions
    for row, rowParams in TurboConfiguration['Rows'].items():
        for plane in ['InletPlane', 'OutletPlane']:
            if plane in rowParams:
                planeAlreadyInExtractions = False
                for extraction in Extractions:
                    if extraction['type'] == 'IsoSurface' \
                        and extraction['field'] == 'CoordinateX' \
                        and np.isclose(extraction['value'], rowParams[plane]):
                        planeAlreadyInExtractions = True
                        extraction.update(dict(ReferenceRow=row, tag=plane))
                        break
                if not planeAlreadyInExtractions:
                    Extractions.append(dict(type='IsoSurface', field='CoordinateX', \
                        value=rowParams[plane], ReferenceRow=row, tag=plane))

def computeDistance2Walls(t, WallFamilies=[], verbose=True, wallFilename=None):
    '''
    Identical to :func:`MOLA.Preprocess.computeDistance2Walls`, except that the
    list **WallFamilies** is automatically filled with with the following
    patterns:
    'WALL', 'HUB', 'SHROUD', 'BLADE', 'MOYEU', 'CARTER', 'AUBE'.
    Names are not case-sensitive (automatic conversion to lower, uper and
    capitalized cases). Others patterns might be added with the argument
    **WallFamilies**.
    '''
    WallFamilies += ['WALL', 'HUB', 'SHROUD', 'BLADE', 'MOYEU', 'CARTER', 'AUBE']
    PRE.computeDistance2Walls(t, WallFamilies=WallFamilies, verbose=verbose, wallFilename=wallFilename)


def initialize_flow_solution_with_turbo(t, FluidProperties, ReferenceValues, TurboConfiguration, mask=None):
    '''
    Initialize the flow solution of **t** with the module ``turbo``.
    The initial flow is computed analytically in the 2D-throughflow plane
    based on:

    * radial equilibrium in the radial direction.

    * Euler theorem between rows in the axial direction.

    The values **FlowAngleAtRoot** and **FlowAngleAtTip** (relative angles 'beta')
    must be provided for each row in **TurboConfiguration**.

    .. note::
        See also documentation of the related function in ``turbo`` module
        `<file:///stck/jmarty/TOOLS/turbo/doc/html/initial.html>`_

    .. important::
        Dependency to ``turbo``

    .. danger::
        Works only in Python3, considering that dictionaries conserve order.
        Rows in TurboConfiguration must be list in the downstream order.

    Parameters
    ----------

        t : PyTree
            Tree to initialize

        FluidProperties : dict
            as produced by :py:func:`computeFluidProperties`

        ReferenceValues : dict
            as produced by :py:func:`computeReferenceValues`

        TurboConfiguration : dict

        mask : PyTree

    Returns
    -------

        t : PyTree
            Modified PyTree
    '''
    import turbo.initial as TI

    if not mask:
        mask = C.convertFile2PyTree('mask.cgns')

    def getInletPlane(t, row, rowParams):
        try: 
            return rowParams['InletPlane']
        except KeyError:
            zones = C.getFamilyZones(t, row)
            return C.getMinValue(zones, 'CoordinateX')

    def getOutletPlane(t, row, rowParams):
        try: 
            return rowParams['OutletPlane']
        except KeyError:
            zones = C.getFamilyZones(t, row)
            return C.getMaxValue(zones, 'CoordinateX')

    class RefState(object):
        def __init__(self):
          self.Gamma = FluidProperties['Gamma']
          self.Rgaz  = FluidProperties['IdealGasConstant']
          self.Pio   = ReferenceValues['PressureStagnation']
          self.Tio   = ReferenceValues['TemperatureStagnation']
          self.roio  = self.Pio / self.Tio / self.Rgaz
          self.aio   = (self.Gamma * self.Rgaz * self.Tio)**0.5
          self.Lref  = 1.

    # Get turbulent variables names and values
    turbDict = dict(zip(ReferenceValues['FieldsTurbulence'],  ReferenceValues['ReferenceStateTurbulence']))

    planes_data = []

    row, rowParams = list(TurboConfiguration['Rows'].items())[0]
    xIn = getInletPlane(t, row, rowParams)
    alpha = ReferenceValues['AngleOfAttackDeg']
    planes_data.append(
        dict(
            omega = 0.,
            beta = [alpha, alpha],
            Pt = 1.,
            Tt = 1.,
            massflow = ReferenceValues['MassFlow'],
            plane_points = [[xIn,-999.],[xIn,999.]],
            plane_name = '{}_InletPlane'.format(row)
        )
    )

    for row, rowParams in TurboConfiguration['Rows'].items():
        xOut = getOutletPlane(t, row, rowParams)
        omega = rowParams['RotationSpeed']
        beta1 = rowParams.get('FlowAngleAtRoot', 0.)
        beta2 = rowParams.get('FlowAngleAtTip', 0.)
        for (beta, paramName) in [(beta1, 'FlowAngleAtRoot'), (beta2, 'FlowAngleAtTip')]:
            if beta * omega < 0:
                MSG=f'WARNING: {paramName} ({beta} deg) has not the same sign that the rotation speed in {row} ({omega} rad/s).\n'
                MSG += '        Double check it is not a mistake.'
                print(J.YELLOW + MSG + J.ENDC)
        Csir = 1. if omega == 0 else 0.95  # Total pressure loss is null for a rotor, 5% for a stator
        planes_data.append(
            dict(
                omega = rowParams['RotationSpeed'],
                beta = [beta1, beta2],
                Csir = Csir,
                plane_points = [[xOut,-999.],[xOut,999.]],
                plane_name = '{}_OutletPlane'.format(row)
                )
        )

    # > Initialization
    print(J.CYAN + 'Initialization with turbo...' + J.ENDC)
    silence = J.OutputGrabber()
    with silence:
        t = TI.initialize(t, mask, RefState(), planes_data,
                nbslice=10,
                constant_data=turbDict,
                turbvarsname=list(turbDict),
                velocity='absolute',
                useSI=True,
                keepTmpVars=False,
                keepFS=True  # To conserve other FlowSolution_t nodes, as FlowSolution#Height
                )
    print('..done.')

    return t


def postprocess_turbomachinery(surfaces, stages=[], 
                                var4comp_repart=None, var4comp_perf=None, var2keep=None, 
                                computeRadialProfiles=True, 
                                config='annular', 
                                lin_axis='XY',
                                RowType='compressor'):
    '''
    Perform a series of classical postprocessings for a turbomachinery case : 

    #. Compute extra variables, in relative and absolute frames of reference

    #. Compute averaged values for all iso-X planes (results are in the `.Average` node), and
       compare inlet and outlet planes for each row if available, to get row performance (total 
       pressure ratio, isentropic efficiency, etc) (results are in the `.Average#ComparisonXX` of
       the inlet plane, `XX` being the numerotation starting at `01`)

    #. Compute radial profiles for all iso-X planes (results are in the `.RadialProfile` node), and
       compare inlet and outlet planes for each row if available, to get row performance (total 
       pressure ratio, isentropic efficiency, etc) (results are in the `.RadialProfile#ComparisonXX` of
       the inlet plane, `XX` being the numerotation starting at `01`)

    #. Compute isentropic Mach number on blades, slicing at constant height, for all values of height 
       already extracted as iso-surfaces. Results are in the `.Iso_H_XX` nodes.

    Parameters
    ----------

        surfaces : PyTree
            extracted surfaces

        stages : :py:class:`list` of :py:class:`tuple`, optional
            List of row stages, of the form:

            >>> stages = [('rotor1', 'stator1'), ('rotor2', 'stator2')] 

            For each tuple of rows, the inlet plane of row 1 is compared with the outlet plane of row 2.

        var4comp_repart : :py:class:`list`, optional
            List of variables computed for radial distributions. If not given, all possible variables are computed.

        var4comp_perf : :py:class:`list`, optional
            List of variables computed for row performance (plane to plane comparison). If not given, 
            the same variables as in **var4comp_repart** are computed, plus `Power`.

        var2keep : :py:class:`list`, optional
            List of variables to keep in the saved file. If not given, the following variables are kept:
            
            .. code-block:: python

                var2keep = [
                    'Pressure', 'Temperature', 'PressureStagnation', 'TemperatureStagnation',
                    'StagnationPressureRelDim', 'StagnationTemperatureRelDim',
                    'Entropy',
                    'Viscosity_EddyMolecularRatio',
                    'VelocitySoundDim', 'StagnationEnthalpyAbsDim',
                    'MachNumberAbs', 'MachNumberRel',
                    'AlphaAngleDegree',  'BetaAngleDegree', 'PhiAngleDegree',
                    'VelocityXAbsDim', 'VelocityRadiusAbsDim', 'VelocityThetaAbsDim',
                    'VelocityMeridianDim', 'VelocityRadiusRelDim', 'VelocityThetaRelDim',
                    ]
        
        computeRadialProfiles : bool
            Choose or not to compute radial profiles.
        
        config : str
            see :py:func:`MOLA.PostprocessTurbo.compute1DRadialProfiles`

        lin_axis : str
            see :py:func:`MOLA.PostprocessTurbo.compute1DRadialProfiles`

        RowType : str
            see parameter 'config' of :py:func:`MOLA.PostprocessTurbo.compareRadialProfilesPlane2Plane`
        
    '''
    import Converter.Mpi as Cmpi
    import MOLA.PostprocessTurbo as Post
    import turbo.user as TUS

    Post.setup = J.load_source('setup', 'setup.py')

    #______________________________________________________________________________
    # Variables
    #______________________________________________________________________________
    allVariables = TUS.getFields(config=config)
    if not var4comp_repart:
        var4comp_repart = ['StagnationEnthalpyDelta',
                           'StagnationPressureRatio', 'StagnationTemperatureRatio',
                           'StaticPressureRatio', 'Static2StagnationPressureRatio',
                           'IsentropicEfficiency', 'PolytropicEfficiency',
                           'StaticPressureCoefficient', 'StagnationPressureCoefficient',
                           'StagnationPressureLoss1', 'StagnationPressureLoss2',
                           ]
    if not var4comp_perf:
        var4comp_perf = var4comp_repart + ['Power']
    if not var2keep:
        var2keep = [
            'Pressure', 'Temperature', 'PressureStagnation', 'TemperatureStagnation',
            'StagnationPressureRelDim', 'StagnationTemperatureRelDim',
            'Entropy',
            'Viscosity_EddyMolecularRatio',
            'VelocitySoundDim', 'StagnationEnthalpyAbsDim',
            'MachNumberAbs', 'MachNumberRel',
            'AlphaAngleDegree',  'BetaAngleDegree', 'PhiAngleDegree',
            'VelocityXAbsDim', 'VelocityRadiusAbsDim', 'VelocityThetaAbsDim',
            'VelocityMeridianDim', 'VelocityRadiusRelDim', 'VelocityThetaRelDim',
        ]

    variablesByAverage = Post.sortVariablesByAverage(allVariables)

    #______________________________________________________________________________#
    Post.computeVariablesOnIsosurface(surfaces, allVariables)
    Post.compute0DPerformances(surfaces, variablesByAverage)
    if computeRadialProfiles: 
        Post.compute1DRadialProfiles(
            surfaces, variablesByAverage, config=config, lin_axis=lin_axis)
    # Post.computeVariablesOnBladeProfiles(surfaces, hList='all')
    #______________________________________________________________________________#

    if Cmpi.rank == 0:
        Post.comparePerfoPlane2Plane(surfaces, var4comp_perf, stages)
        if computeRadialProfiles: 
            Post.compareRadialProfilesPlane2Plane(
                surfaces, var4comp_repart, stages, config=RowType)

    Post.cleanSurfaces(surfaces, var2keep=var2keep)
