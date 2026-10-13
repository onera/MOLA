import os
import copy
import traceback
import numpy as np

from .Global import *
from .Kinematics import *
from .AirfoilPolarsInterpolators import *
from .LiftingLine import (_applyPolarOnLiftingLine,
                          computeGeneralLoadsOfLiftingLine,
                          buildPropeller,
                          setConditions)

try:
    from mola.pytree import InternalShortcuts as J
    from mola.mesh.volume import stackSurfacesWithFields
    import mola.mesh.curve as W

    import Converter.Mpi as Cmpi
    import Post.Mpi as Pmpi

except ModuleNotFoundError as e:

    msg = "not all modules requiring Cassiopee are available. Usage of LiftingLine AdvancedOps is limited."
    raise ModuleNotFoundError(msg) from e

def buildBodyForceDisk(Propeller, PolarsInterpolatorsDict, NPtsAzimut,
    RPM=None, Pitch=None, CommandType=None,
    Constraint='Pitch', ConstraintValue=None, ValueTol=1.0,
    AttemptCommandGuess=[],
    PerturbationFields=None,
    StackOptions={}, WeightEqns=[],
    SourceTermScale=1.0, TipLossFactorOptions={}):
    '''
    Macro-function used to generate the ready-to-use BodyForce
    element for interfacing with a CFD solver.

    Parameters
    ----------

        Propeller : base
            A propeller as produced by the function :py:func:`buildPropeller`

        PolarsInterpolatorsDict : dict
            Contains airfoil's polar interpolators. As constructed by the function
            :py:func:`buildPolarsInterpolatorDict`

        NPtsAzimut : int
            Number of points used for sampling a revolution.

        RPM : float
            Propeller's revolutions per minute. If not
            :py:obj:`None`, then this value overrides the possibly existing
            one contained in special CGNS node ``.Kinematics`` associated to
            a *LiftingLine* zone

        Pitch : float
            Propeller's pitch. If not
            :py:obj:`None`, then this value overrides the possibly existing
            one contained in special CGNS node ``.Kinematics`` associated to
            a *LiftingLine* zone

        CommandType : str
            Type of command used for trim. May be ``'Pitch'`` or ``'RPM'``

        Constraint : str
            Used constraint for setting pitch.
            It may be: ``'Pitch'``, ``'Thrust'`` or ``'Power'``

        ConstraintValue : float
            Value to satisfy the constraint.
            Its value is context-dependent (depends on **Constraint**).
            Thus, value may be in degree, Newton or Watt.

        ValueTol : float
            Tolerance to verify for **Constraint** success.
            An acceptable trim is produced if

            :math:`ValueTol \geq | ActualComputedValue - ConstraintValue |`

        AttemptCommandGuess : :py:class:`list` of lists containing 2 :py:class:`float`
            Used as search bounds ``[min, max]`` for the trim procedure.

            .. hint:: use as many different sets of ``[min, max]`` elements as
                the desired number of attempts for trimming.

        PerturbationFields : PyTree
            :py:func:`Post.PyTree.extractMesh`-compatible PyTree
            used as perturbation fields.

            .. important:: **PerturbationFields** must contain the fields:
                ``'VelocityInducedX'``, ``'VelocityInducedY'`` and
                ``'VelocityInducedZ'``

            These values typically comes from CFD.

        StackOptions : dict
            Optional parameters to be passed to the function
            :py:func:`stackBodyForceComponent`

        WeightEqns : :py:class:`list` of :py:class:`str`
            If not empty, applies sequentially the
            equations given by this list after the disk extrusion and before
            source terms computation.

            .. hint:: for use with `Biel Ortun <http://ispserveur.onera/onera/web/upload/dap/DAAA17076.1509005573.pdf>`_ distribution employing Weibull :

                :math:`\\frac{\gamma}{\\alpha} \left(\\frac{x - \mu}{\\alpha}\\right)^{\gamma - 1} e^{-\left(\\frac{x - \mu}{\\alpha}\\right)^{\gamma}}`

                with :math:`\gamma=2`, :math:`\\alpha=0.3`, :math:`\mu=0` and
                 :math:`x=\mathrm{InverseThickwiseCoordinate}`

                Please introduce: ``"weight=2.0/0.3*(({InverseThickwiseCoordinate})/0.3)**(2.0-1.)*exp(-(({InverseThickwiseCoordinate})/0.3)**2.0)"``

        SourceTermScale : float
            Scaling of source terms.

            .. tip:: slightly increase this value for
                compensating dissipation effects during the flow data transfers
                between grids or overset operations

        TipLossFactorOptions : dict
            Use a tip-loss factor function to the aerodynamic coefficients.
            This :py:class:`dict` defines a pair of keyword-arguments of the 
            function :py:func:`applyTipLossFactorToBladeEfforts`.

    Returns
    -------

        BodyForceElement : zone
            A volume mesh containing fields ``'SourceDensity'``, ``'SourceMomentumX'``,
            ``'SourceMomentumY'``, ``'SourceMomentumZ'``, ``'SourceEnergy'``.

            .. hint:: by means of :py:func:`Post.PyTree.extractMesh`,
                **BodyForceElement** can be used to inject source terms into a
                CFD solver.
    '''
    Cmpi.barrier()


    if not Propeller:
        if not PerturbationFields:
            raise ValueError('Must provide Propeller or PerturbationFields')
        addPerturbationFields([], PerturbationFields=PerturbationFields)
        return [] # BEWARE: CANNOT USE BARRIERS IN THIS FUNCTION FROM THIS LINE

    # this guarantees backwards compatibility
    if Constraint == 'Pitch':
        if Pitch is None:
            if ConstraintValue is None:
                raise ValueError('Must provide Pitch value')
            Pitch = ConstraintValue
        else:
            ConstraintValue = Pitch
        CommandType = 'Pitch'


    if CommandType == 'RPM':
        InitialGuessCMD = RPM

    elif CommandType == 'Pitch':
        InitialGuessCMD = Pitch

    makeTrim = True if Constraint in ('Thrust', 'Power') else False

    if makeTrim and len(AttemptCommandGuess) == 0:
        if InitialGuessCMD is None:
            raise ValueError('Must provide initial guess %s'%CommandType)

        AttemptCommandGuess.append([0.95*InitialGuessCMD, 1.05*InitialGuessCMD])
        AttemptCommandGuess.append([0.90*InitialGuessCMD, 1.10*InitialGuessCMD])
        AttemptCommandGuess.append([0.85*InitialGuessCMD, 1.15*InitialGuessCMD])


    isProp = checkComponentKind(Propeller,'Propeller')
    if not isProp: raise AttributeError('A Propeller component is required.')


    # Get some relevant data from Propeller
    LLs = I.getZones(Propeller)
    LLnameInitial = LLs[0]
    NBlades = len(LLs)
    Kin_n = I.getNodeFromName(LLnameInitial,'.Kinematics')
    RotAxis = I.getValue(I.getNodeFromName1(Kin_n,'RotationAxis'))
    RotCenter = I.getValue(I.getNodeFromName1(Kin_n,'RotationCenter'))
    RightHandRuleRotation = I.getValue(I.getNodeFromName1(Kin_n,'RightHandRuleRotation'))
    RPM_n = I.getNodeFromName1(Kin_n,'RPM')
    RPM_n[1] = RPM

    Comp_n = I.getNodeFromName1(Propeller,'.Component#Info')

    # Get the freestream conditions
    Cond_n = I.getNodeFromName1(LLnameInitial,'.Conditions')


    if not Cond_n:
        write4Debug('.Conditions not found in propeller')
        C.convertPyTree2File(Propeller,'debug.cgns')
        os._exit(0)

    fVxyz = I.getNodeFromName1(Cond_n,'VelocityFreestream')[1]

    if np.any(fVxyz!=0.) and PerturbationFields is not None:
        fVxyz *= 0.
        MSG = WARN+'WARNI4NG: found non-zero freestream velocity vector AND PerturbationFields is not None. Setting freestream to zero.'+ENDC
        print(MSG)

    # Initialize the set of LiftingLines used to further
    # sample the BodyForce element
    RotAxis, RotCenter, Dir = getRotationAxisCenterAndDirFromKinematics(LLs[0])
    AllItersLLs = []
    Dpsi = 360./float(NPtsAzimut-1)
    for it in range(NPtsAzimut):
        LiftingLine = I.copyTree( LLs[0] )
        LiftingLine[0] = 'Blade_it%d'%(it)
        T._rotate(LiftingLine,tuple(RotCenter),tuple(RotAxis),it*Dir*Dpsi,
                  vectors=NamesOfChordSpanThickwiseFrame)
        AllItersLLs += [LiftingLine]

    # Put all LLs in a PyTree/Base structure
    tLL = C.newPyTree(['Base',AllItersLLs]) # Tree
    bLL = I.getBases(tLL)[0] # Base
    I.createUniqueChild(bLL,'.Kinematics','UserDefinedData_t',children=Kin_n[2])
    I.createUniqueChild(bLL,'.Conditions','UserDefinedData_t',children=Cond_n[2])
    I.createUniqueChild(bLL,'.Component#Info','UserDefinedData_t',children=Comp_n[2])

    setRPM(tLL, RPM_n[1]) # useless, since done later ?
    computeKinematicVelocity(tLL) # useless, since done later ?

    PerturbationDisk = addPerturbationFields(tLL, PerturbationFields)




    # MOLA LiftingLine solver :
    def singleShotMOLA__(cmd):
        '''
        Single-shot private function. Enters a control command
        (either Pitch or RPM, defined by CommandType) and returns
        the 1-revolution averaged requested load:
        Thrust (default)
        Power (if Constraint=='Power')
        '''
        if CommandType == 'Pitch':
            addPitch(tLL, cmd)
        elif CommandType == 'RPM':
            addPitch(tLL, Pitch)
            RPM_n[1] = cmd

        setRPM(tLL, RPM_n[1])
        computeKinematicVelocity(tLL)
        assembleAndProjectVelocities(tLL)
        _applyPolarOnLiftingLine(tLL,PolarsInterpolatorsDict)

        if TipLossFactorOptions:
            TipLossFactorOptions['NumberOfBlades']=NBlades

        computeGeneralLoadsOfLiftingLine(tLL, TipLossFactorOptions=TipLossFactorOptions)

        if CommandType == 'Pitch':
            addPitch(tLL,-cmd)
        else:
            addPitch(tLL,-Pitch)

        RequestedLoad = 'Power' if Constraint == 'Power' else 'Thrust'
        Loads = [I.getValue(n) for n in I.getNodesFromName(tLL,RequestedLoad)]
        AvrgLoad = np.mean(Loads)

        return AvrgLoad-ConstraintValue/float(NBlades)

    if Constraint == 'Pitch':
        # Just 1 call required
        singleShotMOLA__(Pitch)
        addPitch(tLL,Pitch)

    elif Constraint in ('Power','Thrust'):
        # SEARCH TRIM CONDITION
        AttMat = np.array(AttemptCommandGuess) # AttemptMatrix
        MinBound, MaxBound = AttMat.min(), AttMat.max()
        AttVals, AttCmds = [], []

        for a in AttemptCommandGuess:
            sol = J.secant(singleShotMOLA__, x0=a[0], x1=a[1],
                            ftol=ValueTol/NBlades,
                            bounds=(MinBound, MaxBound), maxiter=20)
            if sol['converged']:
                # Trim found within tolerance
                AttVals += [sol['froot']]
                AttCmds += [sol['root']]
                break
            AttVals += [sol['froot']]
            AttCmds += [sol['root']]

        AttVals = np.hstack(AttVals)
        AttCmds = np.hstack(AttCmds)

        # Apply closest trimmed condition
        iTrim = np.argmin(np.abs(AttVals))
        Trim = AttCmds[iTrim]

        if CommandType == 'Pitch': Pitch = Trim
        elif CommandType == 'RPM': RPM = Trim

        singleShotMOLA__(Trim)
        addPitch(tLL,Pitch)

    else:
        raise AttributeError("Could not recognize Constraint '%s'"%Constraint)

    AvrgThrust= np.mean([n[1] for n in I.getNodesFromName(tLL,'Thrust')]) * NBlades
    AvrgPower = np.mean([n[1] for n in I.getNodesFromName(tLL,'Power')]) * NBlades

    # -------------------------------------------------------------------- #
    # -------------------- FINALIZATION OF COMPUTATION -------------------- #

    LLs = I.getZones(tLL)
    BodyForceSurface = G.stack(LLs) # Stack LLs to surf

    Stacked = stackBodyForceComponent(BodyForceSurface, RotAxis, **StackOptions)

    # TODO: Check for uniqueness of RotorNames in getLocalBodyForceInputData
    RotorName = Propeller[0]
    Stacked[0] = RotorName

    addThickwiseCoordinate2BodyForceDisk(Stacked, RotAxis)
    if Dir == -1: T._reorder( Stacked, (1,2,-3))

    '''
    For use with Biel Ortun distribution employing Weibull :
    output=gamma/alpha*((x-mu)/alpha)**(gamma-1.) * np.exp(-((x-mu)/alpha)**gamma)

    gamma=2, alpha=0.3, mu=0, x=InverseThickwiseCoordinate

    Introduce:
    "weight=2.0/0.3*(({InverseThickwiseCoordinate})/0.3)**(2.0-1.)*exp(-(({InverseThickwiseCoordinate})/0.3)**2.0)"
    '''
    for eqn in WeightEqns: C._initVars(Stacked, eqn)

    CorrVars = ['ForceAxial','ForceTangential','ForceX','ForceY','ForceZ']

    _keepSectionalForces(Stacked)

    Ni, Nj, Nk, dr = getStackedDimensions(Stacked)

    weightNode = I.getNodeFromName2(Stacked, 'weight')
    if weightNode:
        for corrVar in CorrVars+['weight']: C.node2Center__(Stacked, corrVar)
        G._getVolumeMap(Stacked)

        vol_val, weight_val = J.getVars(Stacked, ['vol','weight'], Container='FlowSolution#Centers')

        vol_tot_val=np.zeros_like(vol_val)
        weight_tot_val=np.zeros_like(vol_val)

        # TODO investigate fully vectorial technique, something similar to (this is bugged):
        # vol_tot_val[:,:,:] = np.sum(vol_val[:,1,:],axis=(1,2))
        # weight_tot_val[:,:,:] = np.sum(weight_val[:,1,:]*vol_val[:,1,:],axis=(1,2))/vol_tot_val[:,:,:]
        for i in range(Ni-1):
            vol_tot_val[i,:,:] = np.sum(vol_val[i,1,:])
            weight_tot_val[i,:,:] = np.sum(weight_val[i,1,:]*vol_val[i,1,:])/vol_tot_val[i,:,:]


        fieldsCorrVars_CC = J.getVars(Stacked,CorrVars,Container='FlowSolution#Centers')
        for f in fieldsCorrVars_CC:
            f *= dr * NBlades / (Nj-1) / vol_tot_val * weight_val / weight_tot_val

    else:
        for corrVar in CorrVars: C.node2Center__(Stacked, corrVar)
        G._getVolumeMap(Stacked)

        vol_val = J.getVars(Stacked, ['vol'], Container='FlowSolution#Centers')[0]

        vol_tot_val=np.zeros_like(vol_val)
        for i in range(Ni-1):
            vol_tot_val[i,:,:] = np.sum(vol_val[i,1,:])

        fieldsCorrVars_CC = J.getVars(Stacked,CorrVars,Container='FlowSolution#Centers')
        for f in fieldsCorrVars_CC:
            f *= dr * NBlades / (Nj-1) / vol_tot_val


    AzimutalLoads = dict()
    for ll in LLs:
        LL_loads = J.get(ll,'.Loads')
        for l in LL_loads:
            try: AzimutalLoads[l].append( LL_loads[l] )
            except KeyError: AzimutalLoads[l] = [ LL_loads[l] ]

    for l in AzimutalLoads:
        AzimutalLoads[l] = np.array(AzimutalLoads[l])


    for elt in Propeller, Stacked:
        J.set(elt,'.AzimutalLoads', **AzimutalLoads)

    AzimutallyAveragedLoads = dict()
    for l in AzimutalLoads:
        AzimutallyAveragedLoads[l] =  np.mean( AzimutalLoads[l] ) * NBlades


    for elt in Propeller, Stacked:
        J.set(elt,'.AzimutalAveragedLoads', **AzimutallyAveragedLoads)
        J.set(elt,'.Commands', Pitch=Pitch, RPM=RPM)

    I.createUniqueChild(Stacked,'.Kinematics','UserDefinedData_t',
                                 children=Kin_n[2])

    computeSourceTerms(Stacked, SourceTermScale=SourceTermScale)

    return Stacked

def _keepSectionalForces(t):
    '''
    TODO : change SectionalForces at source (computeGeneralLoads) and propagate
    this is a temporary fix
    '''
    newFields = ['SectionalForceAxial',
                 'SectionalForceTangential',
                 'SectionalForceX',
                 'SectionalForceY',
                 'SectionalForceZ']

    for z in I.getZones(t):
        
        FlSol = I.getNodeFromName1(z,'FlowSolution')
        if not FlSol: continue

        for fn in newFields: I._rmNodesByName1(FlSol, fn)

        for n in FlSol[2][:]:
            if n[0] == 'ForceAxial':
                n_new = I.copyTree(n)
                n_new[0] = 'SectionalForceAxial'
                FlSol[2] += [ n_new ]

            elif n[0] == 'ForceTangential':
                n_new = I.copyTree(n)
                n_new[0] = 'SectionalForceTangential'
                FlSol[2] += [ n_new ]

            elif n[0] == 'ForceX':
                n_new = I.copyTree(n)
                n_new[0] = 'SectionalForceX'
                FlSol[2] += [ n_new ]

            elif n[0] == 'ForceY':
                n_new = I.copyTree(n)
                n_new[0] = 'SectionalForceY'
                FlSol[2] += [ n_new ]

            elif n[0] == 'ForceZ':
                n_new = I.copyTree(n)
                n_new[0] = 'SectionalForceZ'
                FlSol[2] += [ n_new ]

def getStackedDimensions(BF_block):
    Ni, Nj, Nk = I.getZoneDim(BF_block)[1:4]
    span = J.getVars(BF_block, ['Span'])[0]
    dr = span[1:,1:,1:] - span[:-1,:-1,:-1]
    return Ni, Nj, Nk, dr

def stackBodyForceComponent(Component, RotationAxis, StackStrategy='constant',
        StackRelativeChord=1.0, ExtrusionDistance=None, StackDistribution=None):
    '''
    Transform a body-force 2D surface disc into a 3D volume grid suitable for
    transfer data towards CFD grid.

    Parameters
    ----------

        Component : zone
            bodyforce 2D component

        RotationAxis : :py:class:`tuple` of 3 :py:class:`float`
            orientation of the propeller's rotation axis

        StackStrategy : str
            how stacking is performed. Two possibilities:

            * ``'silhouette'``
                following the silhouette of the volume swept during the
                propeller rotation

            * ``'constant'``
                a constant extrusion distance provided by user [m]

                .. see also:: parameter **ExtrusionDistance**

        StackRelativeChord : float
            this parameter controls the relative position
            of the bodyforce block respect to the lifting line actual position.
            A value of ``1`` means that the trailing surface of the bodyforce block
            matches the lifting line (all sources will be put upstream of Lifting
            line). A value of ``0`` means that the entire block is put downstream of
            the liftingline, the leading surface coincides with the lifting line.
            A value of ``0.5`` means that lifting line is in the middle of the block
            (and so on)

        ExtrusionDistance : float
            this parameter controls the bodyforce source
            width.

            .. note:: if :py:obj:`None`, by default it takes:  ``1.5 * Chord.max()``

        StackDistribution : multiple
            desired thickwise distribution
            of the block. It may be polymorphic following acceptable inputs of
            :py:func:`MOLA.InternalShortcuts.getDistributionFromHeterogeneousInput__`.

            .. note:: if :py:obj:`None`, a uniform distribution of 21 points is done.

    Returns
    -------

        VolumeMesh : zone
            the bodyforce volume grid
    '''
    LeadingEdge = I.copyTree(Component)
    xLE,yLE,zLE = J.getxyz(LeadingEdge)
    Chord, Twist = J.getVars(LeadingEdge, ['Chord', 'Twist'])
    sinTwist = np.sin(np.deg2rad(Twist))

    TrailingEdge = I.copyTree(Component)
    xTE, yTE, zTE = J.getxyz(TrailingEdge)


    if StackDistribution is None:
        StackDistribution = np.linspace(0,1,21)
    else:
        StackDistribution = J.getDistributionFromHeterogeneousInput__(StackDistribution)[1]

    if not ExtrusionDistance: ExtrusionDistance = 1.5 * Chord.max()

    if StackStrategy == 'silhouette':
        LeadingEdgeDisplacement =         StackRelativeChord * Chord * sinTwist
        TrailingEdgeDisplacement = - (1.-StackRelativeChord) * Chord * sinTwist


    elif StackStrategy == 'constant':

        LeadingEdgeDisplacement =         StackRelativeChord * ExtrusionDistance
        TrailingEdgeDisplacement = - (1.-StackRelativeChord) * ExtrusionDistance

    else:
        raise AttributeError('StackStrategy %s not recognized'%str(StackStrategy))

    xLE += LeadingEdgeDisplacement * RotationAxis[0]
    yLE += LeadingEdgeDisplacement * RotationAxis[1]
    zLE += LeadingEdgeDisplacement * RotationAxis[2]

    xTE += TrailingEdgeDisplacement * RotationAxis[0]
    yTE += TrailingEdgeDisplacement * RotationAxis[1]
    zTE += TrailingEdgeDisplacement * RotationAxis[2]

    VolumeBodyForce = stackSurfacesWithFields(TrailingEdge, LeadingEdge,
                                                  StackDistribution)

    return VolumeBodyForce

def addThickwiseCoordinate2BodyForceDisk(disk, RotationAxis):
    '''
    This function adds the fields ``ThickwiseCoordinate`` and
    ``InverseThickwiseCoordinate`` to the volume grid of a bodyforce disk.
    These fields can be used for applying distributions of source terms in an
    easy manner. The field ``ThickwiseCoordinate`` yields ``0`` at the trailing edge
    of the disk and ``1`` at the leading edge.

    Parameters
    ----------

        disk : zone
            bodyforce disk as obtained from :py:func:`stackBodyForceComponent`.

            .. note:: zone **disk** is modified *(fields are added)*

        RotationAxis : :py:class:`tuple` of 3 :py:class:`float`
            the rotation axis vector components of the rotor.
    '''
    ThickwiseCoordinate, InverseThickwiseCoordinate = J.invokeFields(disk,
                           ['ThickwiseCoordinate','InverseThickwiseCoordinate'])

    x, y, z = J.getxyz(disk)
    Ni, Nj, Nk = x.shape

    ThickwiseCoordinate[:,:,:] =    x * RotationAxis[0] \
                                  + y * RotationAxis[1] \
                                  + z * RotationAxis[2]


    for i in range(Ni):
        ThickwiseCoordinateSlice = ThickwiseCoordinate[i,:,:]
        InverseThickwiseCoordinateSlice = ThickwiseCoordinateSlice * 1.0

        isReversed = ThickwiseCoordinate[i,0,-1] < ThickwiseCoordinate[i,0,0]

        MinThickness = ThickwiseCoordinateSlice.min()
        MaxThickness = ThickwiseCoordinateSlice.max()

        if isReversed:

            x[i,:,:] = x[i,:,::-1]
            y[i,:,:] = y[i,:,::-1]
            z[i,:,:] = z[i,:,::-1]
            ThickwiseCoordinateSlice[:,:] =  ThickwiseCoordinateSlice[:,::-1]
            InverseThickwiseCoordinateSlice = ThickwiseCoordinateSlice * 1.0

            ThickwiseCoordinateSlice -= MaxThickness
            ThickwiseCoordinateSlice /= ( MinThickness - MaxThickness )
            ThickwiseCoordinate[i,:,:] = ThickwiseCoordinateSlice

            InverseThickwiseCoordinateSlice -= MinThickness
            InverseThickwiseCoordinateSlice /= ( MaxThickness - MinThickness )
            InverseThickwiseCoordinate[i,:,:] = InverseThickwiseCoordinateSlice

        else:

            ThickwiseCoordinateSlice -= MinThickness
            ThickwiseCoordinateSlice /= ( MaxThickness - MinThickness )
            ThickwiseCoordinate[i,:,:] = ThickwiseCoordinateSlice

            InverseThickwiseCoordinateSlice -= MaxThickness
            InverseThickwiseCoordinateSlice /= ( MinThickness - MaxThickness )
            InverseThickwiseCoordinate[i,:,:] = InverseThickwiseCoordinateSlice

def computeSourceTerms(zone, SourceTermScale=1.0):
    '''
    This function computes the source terms of a bodyforce disk using the
    *required fields* ``VelocityTangential``, ``ForceX``, ``ForceY``, ``ForceZ``, ``ForceTangential``.

    Parameters
    ----------

        zone : zone
            BodyForce disk.

            .. important:: It must contain the fields:
                ``ForceX``, ``ForceY``, ``ForceZ``, ``ForceTangential`` located at centers
                (``FlowSolution#Centers`` container); ``VelocityTangential`` 
                located at nodes (``FlowSolution`` container)

            .. note:: **zone** is modified (new cell-centered fields are added in
                ``FlowSolution#SourceTerm`` container)

        SourceTermScale : float
            overall weighting coefficient for source terms.
            User can use values slightly higher than ``1`` in order to compensate
            dissipation effects provoked by the transfer of fields from the disk
            towards the CFD computational grid.
    '''


    ConservativeFields = ['Density', 'MomentumX','MomentumY', 'MomentumZ',
                        'EnergyStagnationDensity']
    I.__FlowSolutionCenters__ = 'FlowSolution#SourceTerm'
    v1= ro, rou, rov, row, roe = J.invokeFields(zone, ConservativeFields,
                                                locationTag='centers:')
    
    for v in v1:
        if np.any(np.logical_not(np.isfinite(v))):
            print("ERROR: NaN were found in conservative quantities !! Ignoring current bodyforce...",color=J.FAIL)
            v[:] = 0

    I.__FlowSolutionCenters__ = 'FlowSolution#Centers'
    C.node2Center__(zone, 'nodes:VelocityTangential')
    PropellerFields = ['VelocityTangential', 'ForceX', 'ForceY', 'ForceZ', 'ForceTangential']
    v2 = VelocityTangential, fx, fy, fz, ft = J.getVars(zone, PropellerFields,
                                                Container='FlowSolution#Centers')

    for v in v2:
        if np.any(np.logical_not(np.isfinite(v))):
            print("ERROR: NaN were found in source terms !! Ignoring current bodyforce...",color=J.FAIL)
            v[:] = 0


    ro[:]  = 0.0
    rou[:] = - fx * SourceTermScale
    rov[:] = - fy * SourceTermScale
    row[:] = - fz * SourceTermScale
    roe[:] = np.abs( ft * VelocityTangential ) * SourceTermScale
    
def migrateSourceTerms2MainPyTree(donor, receiver):
    '''
    Migrate by interpolation the source terms of a donor *(typically, a bodyforce
    disk)* towards a receiver *(typically a CFD grid)*.

    .. note:: this function is designed to be used in a distributed MPI context.

    Parameters
    ----------

        donor : PyTree, base, zone, list of zones
            Element containing the
            source terms to be transfered.

            .. important:: the source terms must be contained in a
                cell-centered ``FlowSolution#SourceTerm`` container, as obtained
                using :py:func:`computeSourceTerms` function.

        receiver : PyTree
            it must be fully distributed. New transfered fields
            will be introduced into a cell-centered ``FlowSolution#SourceTerm``
            container.

    Returns
    -------

        tRec : PyTree
            reference copy of the receiver PyTree
            including the new cell-centered ``FlowSolution#SourceTerm`` container.

            .. note:: if no receiver is present at a given rank, then an empty list is
                returned
    '''
    
    
    Cmpi.barrier()
    BodyForceDisks = I.getZones(donor)
    BodyForceDisksTree = C.newPyTree(['BODYFORCE', BodyForceDisks])
    Cmpi._setProc(BodyForceDisksTree, Cmpi.rank)
    I._adaptZoneNamesForSlash(BodyForceDisksTree)

    donor = I.copyRef(BodyForceDisksTree)
    I._rmNodesByName(donor, 'FlowSolution')
    I._rmNodesByName(donor, '.Info')
    I._rmNodesByName(donor, '.Kinematics')

    Cmpi.barrier()
    tRec = I.copyRef(receiver)

    I._rmNodesByType(tRec, 'FlowSolution_t')

    I.__FlowSolutionCenters__ = 'FlowSolution#SourceTerm'


    # need to make try/except (see Cassiopee #7754)
    Cmpi.barrier()
    try:
        tRec = Pmpi.extractMesh(donor, tRec, mode='accurate',
                                extrapOrder=0, constraint=0.)
    except:
        tRec = []
    Cmpi.barrier()

    I.__FlowSolutionCenters__ = 'FlowSolution#Centers'

    return tRec

def getLocalBodyForceInputData(BodyForceInputData):
    '''
    .. warning:: Private function.

    This function appends the bodyforce input data into a local list if
    the *proc* value corresponds to the local rank.

    Parameters
    ----------

        BodyForceInputData : list
            as defined in ``setup.py`` workflow for bodyforce.
            For more specific details, see documentation of the function
            :py:func:`MOLA.Preprocess.prepareMainCGNS4ElsA`

    Returns
    -------

        LocalBodyForceInputData : list
            only of rotor such that ``proc = rank``. Otherwise
            the list is empty.
    '''
    LocalBodyForceInputData = []
    for Rotor in BodyForceInputData:
        # TODO: first determine proc, then make deepcopy only if condition
        # is fulfilled
        CopiedRotor = copy.deepcopy(Rotor)
        try: proc = CopiedRotor['proc']
        except KeyError: proc = -1

        if proc == Cmpi.rank: LocalBodyForceInputData.append(CopiedRotor)

    return LocalBodyForceInputData

def invokeAndAppendLocalObjectsForBodyForce(LocalBodyForceInputData):
    '''

    .. attention:: this is a private function employed in BODYFORCE technique.

    It builds and append local objects used for bodyforce (propeller,
    lifting-lines, interpolators)

    Parameters
    ----------

        LocalBodyForceInputData : list
            as obtained from the function :py:func:`getLocalBodyForceInputData`
    '''

    def getItemOrRaiseWarning(itemName):
        try:
            item = Rotor[itemName]
        except KeyError:
            try: name = Rotor['name']
            except KeyError: name = '<UndefinedName>'
            MSG = 'WARNING: {} of rotor {} not found at proc {}'.format(
                            itemName,  name,  Cmpi.rank)
            print(MSG)

            item = None
            # CAVEAT default value
            if itemName == 'RightHandRuleRotation': item = True

        return item


    for Rotor in LocalBodyForceInputData:

        RotorName = getItemOrRaiseWarning('name')

        FILE_LiftingLine = getItemOrRaiseWarning('FILE_LiftingLine')
        if not FILE_LiftingLine: continue

        LiftingLine = C.convertFile2PyTree(FILE_LiftingLine)
        LiftingLine, = I.getZones(LiftingLine)
        if not I.getNodeFromName1(LiftingLine,'.Component#Info'):
            J.set(LiftingLine,'.Component#Info', kind='LiftingLine') # related to MOLA #48

        LiftingLine[0] = 'LL.%s.r%d'%(RotorName,Cmpi.rank)
        Rotor['LiftingLine'] = LiftingLine


        FILE_Polars = getItemOrRaiseWarning('FILE_Polars')
        if not FILE_Polars: continue

        PyZonePolars = C.convertFile2PyTree(FILE_Polars)
        PyZonePolars = I.getZones(PyZonePolars)
        PolarsInterpolatorsDict = buildPolarsInterpolatorDict(PyZonePolars)

        Rotor['PolarsInterpolatorsDict'] = PolarsInterpolatorsDict

        NumberOfBlades = getItemOrRaiseWarning('NumberOfBlades')
        RotationCenter = getItemOrRaiseWarning('RotationCenter')
        RotationAxis = getItemOrRaiseWarning('RotationAxis')
        InitialAzimutDirection = getItemOrRaiseWarning('InitialAzimutDirection')
        InitialAzimutPhase = getItemOrRaiseWarning('InitialAzimutPhase')
        RightHandRuleRotation = getItemOrRaiseWarning('RightHandRuleRotation')

        buildPropeller_kwargs = dict(NBlades=NumberOfBlades)
        if InitialAzimutDirection is not None:
            buildPropeller_kwargs['InitialAzimutDirection'] = InitialAzimutDirection
        if InitialAzimutPhase is not None:
            buildPropeller_kwargs['InitialAzimutPhase'] = InitialAzimutPhase



        setKinematicsUsingConstantRotationAndTranslation(LiftingLine,
                                      RotationCenter=RotationCenter,
                                      RotationAxis=RotationAxis,
                                      RPM=0.0,
                                      RightHandRuleRotation=RightHandRuleRotation)

        RequiredVariables = NumberOfBlades,RotationCenter,RotationAxis

        if not all( RequiredVariables ): continue

        Propeller = buildPropeller(LiftingLine, **buildPropeller_kwargs)
        setConditions(Propeller)
        MandatoryFields = ('Density', 'MomentumX', 'MomentumY', 'MomentumZ',
                            'EnergyStagnationDensity', 'Temperature',
                            'AoA', 'Mach', 'Reynolds',
                    'VelocityInducedX', 'VelocityInducedY', 'VelocityInducedZ')
        [C._initVars(Propeller, f, 0.) for f in MandatoryFields]

        Propeller[0] = RotorName
        Rotor['Propeller'] = Propeller

def getNumberOfSerialRuns(BodyForceInputData, NumberOfProcessors):
    '''
    Determine the number of serial runs employed given the employed number
    of procs and the **BodyForceInputData**.

    Parameters
    ----------

        BodyForceInputData : list
            list of data as established in ``setup.py`` (see
            :py:func:`MOLA.Preprocess.prepareMainCGNS4ElsA`)

        NumberOfProcessors : int
            total number of procs employed

    Returns
    -------

        NumberOfSerialRuns : int
            Required number of serial runs
    '''
    NRunsPerProc = np.zeros(NumberOfProcessors, dtype=np.int32)
    for inputData in BodyForceInputData:
        NRunsPerProc[inputData['proc']] += 1
    NumberOfSerialRuns = np.max(NRunsPerProc)
    return NumberOfSerialRuns

def computePropellerBodyForce(to, NumberOfSerialRuns, LocalBodyForceInputData):
    '''
    This is a user-level function called in the BodyForce technique context
    in elsA trigger computation. It is used to construct the bodyforce disks.

    Parameters
    ----------

        to : PyTree
            Distributed CFD PyTree with full Skeleton containing actual fields

        NumberOfSerialRuns : int
            as obtained from :py:func:`getNumberOfSerialRuns`

        LocalBodyForceInputData : list
            as obtained from :py:func:`getLocalBodyForceInputData`

    Returns
    -------

        BodyForceDisks : :py:class:`list` of zone
            list of zones containing fields in ``FlowSolution#SourceTerm``
            container, ready to be migrated into CFD grid
            ( see :py:func:`migrateSourceTerms2MainPyTree` )
    '''
    BodyForceDisks = []
    BodyForcePropellers = []

    try:
        for iBF in range(NumberOfSerialRuns):
            try:
                SerialBFdata = LocalBodyForceInputData[iBF]
                try:
                    Propeller = SerialBFdata['Propeller']
                    PolarsInterpolatorsDict = SerialBFdata['PolarsInterpolatorsDict']
                    NumberOfAzimutalPoints = SerialBFdata['NumberOfAzimutalPoints']
                    buildBodyForceDiskOptions = SerialBFdata['buildBodyForceDiskOptions']
                except KeyError:
                    Propeller = []
                    PolarsInterpolatorsDict = None
                    NumberOfAzimutalPoints = None
                    buildBodyForceDiskOptions = {}

            except IndexError:
                Propeller = []
                PolarsInterpolatorsDict = None
                NumberOfAzimutalPoints = None
                buildBodyForceDiskOptions = {}


            BodyForceOptions = dict(PerturbationFields=to)
            BodyForceOptions.update(buildBodyForceDiskOptions)

            BFdisk = buildBodyForceDisk(Propeller,
                                        PolarsInterpolatorsDict,
                                        NumberOfAzimutalPoints,
                                        **BodyForceOptions)
            if BFdisk: BodyForceDisks.append(BFdisk)

            # TODO:
            # Examine if returning BodyForcePropellers or not: is it really useful?
            if Propeller: BodyForcePropellers.append(Propeller)

    except BaseException:
        print(traceback.format_exc())
        os._exit(0)

    return BodyForceDisks

def write4Debug(MSG):
    with open('LOGS/rank%d.log'%Cmpi.rank,'a') as f: f.write('%s\n'%MSG)

def migratePerturbationsFromAuxiliarDisc2LiftingLines(AuxiliarDisc, LiftingLines):
    '''
    Migrate the perturbation fields :
    ``Density``,
    ``MomentumX``, ``MomentumY``, ``MomentumZ``,
    ``EnergyStagnationDensity``, ``Temperature``,
    ``VelocityInducedX``, ``VelocityInducedY``, ``VelocityInducedZ``
    from **AuxiliarDisc** to the **LiftingLines**.

    .. warning:: the number of **LiftingLines** zones
        *must* coincide with the azimuthal discretization of the **AuxiliarDisc**

    Parameters
    ----------

        AuxiliarDisc : zone
            Auxiliar disc *(bodyforce)* where perturbation fields are contained.

        LiftingLines : :py:class:`list` of zone
            list of lifting lines where perturbation
            fields will be transfered.

            .. important:: **LiftingLines** must be exactly supported on
                the auxiliar disc. This imposes a number of constraints:

                #. the amount of
                   lifting lines **must be** the same as the azimuthal discretization of the
                   auxiliar disc.

                #. the spanwise discretization **must be** the same as the
                   radial discretization of the auxiliar disc (except the root and tip
                   points of auxiliar disc, which lies outside of lifting line).
    '''
    PerturbationFields = ('Density', 'MomentumX', 'MomentumY', 'MomentumZ',
        'EnergyStagnationDensity', 'Temperature',
        'VelocityInducedX', 'VelocityInducedY', 'VelocityInducedZ')

    AuxiliarDisc, = I.getZones(AuxiliarDisc)
    LLs = I.getZones(LiftingLines)

    fieldsDisc = J.getVars(AuxiliarDisc, PerturbationFields)
    for j in range(len(LLs)):
        fieldsLL = J.getVars(LLs[j], PerturbationFields)
        for fieldLL, fieldDisc in zip(fieldsLL, fieldsDisc):
            fieldLL[:] = fieldDisc[1:-1,j]

        Conds = J.get(LLs[j],'.Conditions')
        if Conds:
            AverageFieldNames = ['Temperature', 'Density']
            v = J.getVars2Dict(LLs[j], AverageFieldNames)
            for fn in AverageFieldNames:
                Conds[fn][:] = np.mean(v[fn])

def addPerturbationFields(t, PerturbationFields=None):
    '''
    Sets the existing ``VelocityInducedX``, ``VelocityInducedY``,
    ``VelocityInducedZ`` fields of Lifting Lines contained in **t** the
    perturbation contribution contained in **PerturbationFields**.

    .. note:: if no **PerturbationFields** is given (object :py:obj:`None` is
        provided), then this function simply sets ``VelocityInducedX``,
        ``VelocityInducedY``, ``VelocityInducedZ`` to zero

    Parameters
    ----------

        t : PyTree
            Container with CGNS LiftingLines objects.

            .. note:: lifting-line zones contained in **t** are modified

        PerturbationFields : PyTree, base, zone, list of zones
            donor compatible with :py:func:`Post.PyTree.extractMesh` function.

            .. attention:: **PerturbationFields**  must contain the following
                fields: ``Density`` , ``MomentumX``, ``MomentumY``, ``MomentumZ``

    '''
    Cmpi.barrier()

    if PerturbationFields:

        tPert = I.renameNode(PerturbationFields,
                             'FlowSolution#Init', 'FlowSolution#Centers')
        I._rmNodesByName(tPert,'FlowSolution#EndOfRun#Relative')

        if t:
            LLs = I.getZones(t)
            SpanMax = C.getMaxValue(LLs[0],'Span')
            SpanMin = C.getMinValue(LLs[0],'Span')
            ExtrapolateSpanMax = 0.05 * SpanMax
            ExtrapolateSpanMin = 0.1 * SpanMin
            auxLLs = []
            for auxLL in LLs:
                auxLL = W.extrapolate(auxLL, ExtrapolateSpanMax)
                auxLL = W.extrapolate(auxLL, ExtrapolateSpanMin,
                                      opposedExtremum=True)
                auxLLs.append(auxLL)

            AuxiliarDisc = G.stack(auxLLs)
            AuxiliarDisc[0] = 'AuxDisc.%d'%Cmpi.rank
            AuxiliarDiscs = [AuxiliarDisc]
            Cmpi._setProc(AuxiliarDiscs, Cmpi.rank)
            I._adaptZoneNamesForSlash(AuxiliarDiscs)
        else:
            AuxiliarDiscs = []

        tAux = C.newPyTree(['Base',AuxiliarDiscs])



        I.__FlowSolutionCenters__ = 'FlowSolution#Centers'

        Cmpi.barrier()
        # need to make try/except (see Cassiopee #7754)
        try: tAux = Pmpi.extractMesh(tPert, tAux, constraint=0.)
        except: tAux = None
        Cmpi.barrier()

        if not tAux: return # BEWARE cannot use barriers from this point

        AuxiliarDisc = I.getZones(tAux)[0]
        C._initVars(AuxiliarDisc,'VelocityInducedX={MomentumX}')
        C._initVars(AuxiliarDisc,'VelocityInducedY={MomentumY}')
        C._initVars(AuxiliarDisc,'VelocityInducedZ={MomentumZ}')
        iVx, iVy, iVz, ro, rou, rov, row, Temp = J.getVars(AuxiliarDisc,
                                               ['VelocityInducedX',
                                                'VelocityInducedY',
                                                'VelocityInducedZ',
                                                'Density',
                                                'MomentumX',
                                                'MomentumY',
                                                'MomentumZ',
                                                'Temperature',
                                                ])


        PositiveDensity = ro > 1e-3
        iVx[PositiveDensity] = rou[PositiveDensity]/ro[PositiveDensity]
        iVy[PositiveDensity] = rov[PositiveDensity]/ro[PositiveDensity]
        iVz[PositiveDensity] = row[PositiveDensity]/ro[PositiveDensity]

        for v in [iVx, iVy, iVz]:
            isNotFinite = np.logical_not(np.isfinite(v))
            v[isNotFinite] = 0.

        migratePerturbationsFromAuxiliarDisc2LiftingLines(AuxiliarDisc, t)

        return AuxiliarDisc

    else:
        C._initVars(t,'VelocityInducedX',0.)
        C._initVars(t,'VelocityInducedY',0.)
        C._initVars(t,'VelocityInducedZ',0.)

