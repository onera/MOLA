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
MOLA - LiftingLine.py

This module proposes a set of functions for use with LiftingLine-based methods
including Airfoil Polars data management.

Test modification MOLA-VPM

First creation:
24/11/2019 - L. Bernardos
'''

import MOLA
from . import InternalShortcuts as J
from . import Wireframe as W
from . import GenerativeShapeDesign as GSD
from . import __version__

import sys
import os
import re
import copy
import traceback
import numpy as np
if not MOLA.__ONLY_DOC__:
    from numpy.linalg import norm

    import Converter.PyTree as C
    import Converter.Internal as I
    import Transform.PyTree as T
    import Generator.PyTree as G
    import Connector.PyTree as X
    import Post.PyTree as P
    import Geom.PyTree as D

# Global constants
# -> Fluid constants
Gamma, Rgp = 1.4, 287.058
Mus, Cs, Ts= 1.711e-5, 110.4, 273.0 # Sutherland const.

FAIL  = '\033[91m'
GREEN = '\033[92m'
WARN  = '\033[93m'
MAGE  = '\033[95m'
CYAN  = '\033[96m'
ENDC  = '\033[0m'

NamesOfChordSpanThickwiseFrameNoTangential = [
    ['ChordwiseX','ChordwiseY','ChordwiseZ'],
    ['SpanwiseX','SpanwiseY','SpanwiseZ'],
    ['ThickwiseX','ThickwiseY','ThickwiseZ'],
    ['PitchRelativeCenterX','PitchRelativeCenterY','PitchRelativeCenterZ'],
    ['PitchAxisX','PitchAxisY','PitchAxisZ'],
    ]

NamesOfChordSpanThickwiseFrame = NamesOfChordSpanThickwiseFrameNoTangential + \
    [['TangentialX', 'TangentialY', 'TangentialZ']]

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
    import Converter.Mpi as Cmpi
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
        MSG = WARN+'WARNING: found non-zero freestream velocity vector AND PerturbationFields is not None. Setting freestream to zero.'+ENDC
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
            # LB TODO write more clearly:
            # f *= (dr * NBlades / ((Nj-1) * vol_tot_val)) * (weight_val / weight_tot_val)

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
            # LB TODO write more clearly:
            # f *= (dr * NBlades / ((Nj-1) * vol_tot_val))

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
    from .GenerativeVolumeDesign import stackSurfacesWithFields
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
    from .Coprocess import printCo


    ConservativeFields = ['Density', 'MomentumX','MomentumY', 'MomentumZ',
                        'EnergyStagnationDensity']
    I.__FlowSolutionCenters__ = 'FlowSolution#SourceTerm'
    v1= ro, rou, rov, row, roe = J.invokeFields(zone, ConservativeFields,
                                                locationTag='centers:')
    
    for v in v1:
        if np.any(np.logical_not(np.isfinite(v))):
            printCo("ERROR: NaN were found in conservative quantities !! Ignoring current bodyforce...",color=J.FAIL)
            v[:] = 0

    I.__FlowSolutionCenters__ = 'FlowSolution#Centers'
    C.node2Center__(zone, 'nodes:VelocityTangential')
    PropellerFields = ['VelocityTangential', 'ForceX', 'ForceY', 'ForceZ', 'ForceTangential']
    v2 = VelocityTangential, fx, fy, fz, ft = J.getVars(zone, PropellerFields,
                                                Container='FlowSolution#Centers')

    for v in v2:
        if np.any(np.logical_not(np.isfinite(v))):
            printCo("ERROR: NaN were found in source terms !! Ignoring current bodyforce...",color=J.FAIL)
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
    import Converter.Mpi as Cmpi
    import Post.Mpi as Pmpi
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



def buildPropeller(LiftingLine, NBlades=2, InitialAzimutDirection=[0,1,0],
                                InitialAzimutPhase=0., 
                                MirrorBlade='OnlyIfLeftHandRuleRotation'):
    '''
    Construct a propeller object using a **LiftingLine** with native location, i.e.
    as generated by :py:func:`buildLiftingLine` function.
    Also, **LiftingLine** must contain `.Kinematics` information as provided by,
    e.g., function :py:func:`setKinematicsUsingConstantRotationAndTranslation`.

    A propeller object is a ``CGNSBase_t`` object, in which several LiftingLine
    objects are contained. Special motion nodes may be contained in the base.

    Parameters
    ----------

        LiftingLine : zone
            A Lifting Line object, as generated from
            function :py:func:`buildLiftingLine`, with Kinematics information
            as provided by e.g. :py:func:`setKinematicsUsingConstantRotationAndTranslation` function.

            .. important::
                **LiftingLine** must be placed in native location (as generated
                using :py:func:`buildLiftingLine`)

        NBlades : int
            Number of blades. Will make copies of **LiftingLine**

        InitialAzimutDirection : numpy array of 3 :py:class:`float`
            Direction vector used to
            indicate where the first blade of the propeller shall be
            pointing to. If not contained in the propeller's rotation
            plane, the function will perform a projection.

            .. warning::
                this vector must **NOT** be aligned with rotation axis. In case
                of alignment, InitialAzimutDirection is randomly modified until
                misalignment is obtained

        InitialAzimutPhase : float
            Indicate the phase (in degree) of the initial azimut of the first
            blade. The angle follows the direction of rotation around rotation
            axis and value of RightHandRuleRotation (contained in
            ``.Kinematics`` node of **LiftingLine**).

        MirrorBlade : :py:class:`str` or :py:class:`bool`
            Parameter controlling in which conditions the blade may (or not) be 
            mirrored. Available options are:

            * :py:obj:`False`
                Do not mirror the blade geometry of the provided **LiftingLine**

            * :py:obj:`True`
                Force mirroring the blade geometry of the provided **LiftingLine**

            * `'OnlyIfLeftHandRuleRotation'`
                Only mirror the **LiftingLine** if user selected `RightHandRuleRotation=False`
                when including Kinematics information (using, e.g. :py:func:`setKinematicsUsingConstantRotationAndTranslation`).

                .. warning::
                    option `'OnlyIfLeftHandRuleRotation'` shall be used if the
                    user constructed the **LiftingLine** using parameter `RightHandRuleRotation=True`
                    in :py:func:`buildLiftingLine`. This is the **default** behavior.

            * `'OnlyIfRightHandRuleRotation'`
                Only mirror the **LiftingLine** if user selected `RightHandRuleRotation=True`
                when including Kinematics information (using, e.g. :py:func:`setKinematicsUsingConstantRotationAndTranslation`).

                .. warning::
                    option `'OnlyIfRightHandRuleRotation'` shall be used if the
                    user constructed the **LiftingLine** using parameter `RightHandRuleRotation=False`
                    in :py:func:`buildLiftingLine`.


    Returns
    -------

        Propeller : base
            ``CGNSBase_t`` object with lifting-line zones representing the propeller
    '''
    norm = np.linalg.norm
    LiftingLine, = I.getZones(LiftingLine)
    LiftingLine = I.copyTree(LiftingLine)
    InitialAzimutDirection = np.array(InitialAzimutDirection, order='F', dtype=np.float64)
    InitialAzimutDirection/= np.sqrt(InitialAzimutDirection.dot(InitialAzimutDirection))

    RotAxis, RotCenter,Dir=getRotationAxisCenterAndDirFromKinematics(LiftingLine)

    # Force RotAxis to be unitary
    RotAxis /= np.sqrt(RotAxis.dot(RotAxis))

    def misalignmentInDegrees(a, b):
        return np.abs(np.rad2deg( np.arccos( a.dot(b) / (norm(a)*norm(b)) ) ))

    angle = misalignmentInDegrees(InitialAzimutDirection, RotAxis)
    while angle < 5.:
        InitialAzimutDirection[0] += 0.01
        InitialAzimutDirection[1] += 0.02
        InitialAzimutDirection[2] += 0.03
        InitialAzimutDirection/= norm(InitialAzimutDirection)
        angle = misalignmentInDegrees(InitialAzimutDirection, RotAxis)

    # Force initial azimut direction to be on the Rotation plane
    CoplanarBinormalVector = np.cross(InitialAzimutDirection, RotAxis)
    InitialAzimutDirection = np.cross(RotAxis, CoplanarBinormalVector)
    InitialAzimutDirection /= norm(InitialAzimutDirection)

    # Mirror the Lifting-Line
    if isinstance(MirrorBlade,str):
        if MirrorBlade not in ['OnlyIfLeftHandRuleRotation','OnlyIfRightHandRuleRotation']:
            raise AttributeError("MirrorBlade must be bool or in ['OnlyIfLeftHandRuleRotation','OnlyIfRightHandRuleRotation']")

        if  (Dir == -1 and MirrorBlade=='OnlyIfLeftHandRuleRotation') or \
            (Dir ==  1 and MirrorBlade=='OnlyIfRightHandRuleRotation'):
            mirrorBlade(LiftingLine)
    elif MirrorBlade is True:
        mirrorBlade(LiftingLine)

    # Invoke blades
    LLs = []
    for nb in range(NBlades):
        NewBlade = I.copyTree(LiftingLine)
        NewBlade[0] += '_blade%d'%(nb+1)

        # Apply azimuthal position
        AzPos = nb*(360./float(NBlades))
        T._rotate(NewBlade,(0,0,0),(0,0,1),AzPos+Dir*InitialAzimutPhase,
                  vectors=NamesOfChordSpanThickwiseFrame)
        LLs += [NewBlade]

    # ======= PUT THE PROPELLER IN ITS NEW LOCATION ======= #

    # This is the LiftingLine's reference frame ats its canonical position
    LLFrame = ((1,0,0), # Blade-wise
               (0,1,0), # sweep-wise
               (0,0,1)) # Rotation axis

    # Compute destination propeller's Frame
    sweepwise = np.cross(RotAxis,InitialAzimutDirection)
    PropFrame = (tuple(InitialAzimutDirection),     # Blade-wise
                 tuple(sweepwise), # sweep-wise
                 tuple(RotAxis))   # Rotation axis

    T._rotate(LLs,(0,0,0),LLFrame,arg2=PropFrame,vectors=NamesOfChordSpanThickwiseFrame)
    T._translate(LLs,tuple(RotCenter))

    # ============= INVOKE BASE AND ADD BLADES ============= #
    PropBase = I.newCGNSBase('Propeller',cellDim=1,physDim=3)
    PropBase[2] = LLs # Add Blades

    # Sets component general information
    J.set(PropBase,'.Component#Info',kind='Propeller')

    return PropBase


def buildLiftingLine(Span, RightHandRuleRotation=True, 
        PitchRelativeCenter=[0,0,0], PitchAxis=[1,0,0],
        RotationCenter=[0,0,0], SweepCorrection = True, DihedralCorrection = True, AngleSmoothingLaw = None, **kwargs):
    '''
    Make a PyTree-Line zone defining a Lifting-line. The construction
    procedure of this element is the same as in function
    :py:func:`MOLA.GenerativeShapeDesign.wing`, same logic is used here!


    .. important:: The native canonical lifting line location is set towards:

        :math:`+X` spanwise

        :math:`-Y` sweepwise

        :math:`+Z` dihedralwise

        and centered at :math:`(0,0,0)`


    Parameters
    ----------

        Span : multiple
            This polymorphic input is used to infer the spanwise
            dimensions and discretization that new lifting-line will use.

            For detailed information on possible inputs of **Span**, please see
            :py:func:`MOLA.InternalShortcuts.getDistributionFromHeterogeneousInput__` doc.

            .. tip:: typical use is ``np.linspace(MinimumSpan, MaximumSpan, NbOfSpanwisePoints)``
        
        RightHandRuleRotation : bool
            Determines wether the LiftingLine is taken as a rotating blade 
            following the right-hand-rule rotation or not.

        PitchRelativeCenter : :py:class:`list` of 3 :py:class:`float`
            relative position of the pitch center (in meters).
            It is used by :py:func:`addPitch`.

        PitchAxis : :py:class:`list` of 3 :py:class:`float`
            direction of the positive pitch, passing through **PitchRelativeCenter**.
            It is used by :py:func:`addPitch`.

        RotationCenter : :py:class:`list` of 3 :py:class:`float`
            If the resulting LiftingLine will be used as a rotatory wing
            (propeller, rotor), then this parameter sets the employed 
            rotation center position in meters.
            
            .. note::
                canonical rotation axis is :math:`(0,0,1)`

        AngleSmoothingLaw : str
            Smoothes the SweepAngleDeg and DihedralAngleDeg values to avoid discontinuities.
            Default : None
            Options: 'UnivariateSpline', 'Pchip', 'Akima', 'SavgolFilter'.

            Please refer to the corresponding scipy package for a full description of the functions

        kwargs : pairs of **attribute** = :py:class:`dict`
            This is an arbitrary number of input arguments following the same
            structure as :py:func:`MOLA.GenerativeShapeDesign.wing` function.

            For example, for setting a twist law:

            ::

                Twist = dict(RelativeSpan = [0.2,  0.6,  1.0],
                                    Twist = [30.,  6.0, -7.0],
                             InterpolationLaw = 'akima')

            .. note:: For introduction of Polar data, this shall be done using
                the kwargs argument **Airfoils** and the Tag number of the
                **PyZonePolar**. Example:

                ::

                    Airfoils={'RelativeSpan':[Rmin/Rmax, 1],
                            'PyZonePolarNames' :   [ 'foilA', 'foilB'],
                            'InterpolationLaw':'interp1d_linear'}

                this will make a linear interpolation between
                **PyZonePolar** tagged ``'foilA'`` at Root and **PyZonePolar**
                tagged ``'foilB'`` at tip.

    Returns
    -------

        LiftingLine : zone
            structured curve zone corresponding to the new lifting line

    '''
    import scipy.interpolate as si
    from scipy.signal import savgol_filter

    # ------------ PERFORM SOME VERIFICATIONS ------------ #

    # Verify the Span argument
    Span,s,_ = J.getDistributionFromHeterogeneousInput__(Span)
    Ns = len(Span)

    LiftingLine = D.line((0,0,0),(1,0,0),Ns)
    LLx, LLy, LLz = J.getxyz(LiftingLine)
    NPts = len(LLx)

    SpecialTreatment = ['Airfoils','Span','s']
    Variables2Invoke = [v for v in kwargs if v not in SpecialTreatment]
    LLDict = J.invokeFieldsDict(LiftingLine,Variables2Invoke+['Span','s'])
    LLDict['Span'][:] = Span
    LLDict['s'][:] = s
    RelSpan = Span/Span.max()

    InterpLaws = {}
    for GeomParam in LLDict:
        if GeomParam not in SpecialTreatment:
            InterpLaws[GeomParam+'_law']=kwargs[GeomParam]['InterpolationLaw']
            if 'RelativeSpan' in kwargs[GeomParam]:
                LLDict[GeomParam][:] = J.interpolate__(RelSpan,
                                                kwargs[GeomParam]['RelativeSpan'],
                                                kwargs[GeomParam][GeomParam],
                                                InterpLaws[GeomParam+'_law'],
                                                **kwargs[GeomParam])
            elif 'Abscissa' in kwargs[GeomParam]:
                try:
                    LLDict[GeomParam][:] = J.interpolate__(s,
                                                kwargs[GeomParam]['Abscissa'],
                                                kwargs[GeomParam][GeomParam],
                                                InterpLaws[GeomParam+'_law'],
                                                **kwargs[GeomParam])
                except BaseException as e:
                    raise ValueError(J.FAIL+f'failed for GeomParam={GeomParam} with parameters:{kwargs[GeomParam]}'+J.ENDC) from e
            else:
                raise AttributeError("Attribute %s (dict) must contain 'RelativeSpan' or 'Abscissa' key"%GeomParam)

    # Apply geometrical distribution
    LLx[:] = Span

    if 'Sweep' in LLDict:
        if RightHandRuleRotation:
            LLy[:] = -LLDict['Sweep']
        else:
            LLy[:] =  LLDict['Sweep']
    
    if 'Dihedral' in LLDict: LLz[:] =  LLDict['Dihedral']

    # Add Airfoils node
    D._getCurvilinearAbscissa(LiftingLine)
    if 'RelativeSpan' in kwargs['Airfoils']:
        AbscissaPolar = J.interpolate__(kwargs['Airfoils']['RelativeSpan'], RelSpan, s)
        kwargs['Airfoils']['Abscissa'] = AbscissaPolar
    elif 'Abscissa' in kwargs['Airfoils']:
        AbscissaPolar = kwargs['Airfoils']['Abscissa']
    else:
        raise ValueError("Attribute Polars (dict) must contain 'RelativeSpan' or 'Abscissa' key")

    nSecsPolars = len(AbscissaPolar)
    nSecsNamesPolars = len(kwargs['Airfoils']['PyZonePolarNames'])
    if nSecsPolars != nSecsNamesPolars:
        ErrMsg = 'USER ERROR during LiftingLine construction,\n'
        ErrMsg+= 'Nb of relative span position for airfoils is: %d\n'%nSecsPolars
        ErrMsg+= 'Nb of polar names keys is: %d\n'%nSecsNamesPolars
        ErrMsg+= 'which are not the same.\n'
        ErrMsg+= 'Please check your Polars dictionnary data coherency before calling buildLiftingLine()'
        raise ValueError(ErrMsg)

    # Initialize some variables
    LLfields = ['RelativeSpan', 'AoA', 'Mach', 'Reynolds', 'Cl', 'Cd','Cm',
        'PitchRelativeCenterX','PitchRelativeCenterY','PitchRelativeCenterZ',
        'PitchAxisX','PitchAxisY','PitchAxisZ',
        'TangentialX', 'TangentialY', 'TangentialZ',
        'SweepAngleDeg', 'DihedralAngleDeg', 'ChordVirtualWithSweep']
    addLLfields = ['ChordwiseX','ChordwiseY','ChordwiseZ',
                    'SpanwiseX', 'SpanwiseY', 'SpanwiseZ',
                   'ThickwiseX','ThickwiseY','ThickwiseZ']
    existing_fields = C.getVarNames(LiftingLine)[0]
    for fn in addLLfields:
        if fn not in existing_fields:
            LLfields.append(fn)
    J._invokeFields(LiftingLine,LLfields)
    v = J.getVars2Dict(LiftingLine, LLfields + addLLfields)

    v['RelativeSpan'][:] = RelSpan

    v['PitchRelativeCenterX'][:] = PitchRelativeCenter[0]
    v['PitchRelativeCenterY'][:] = PitchRelativeCenter[1]
    v['PitchRelativeCenterZ'][:] = PitchRelativeCenter[2]

    v['PitchAxisX'][:] = PitchAxis[0]
    v['PitchAxisY'][:] = PitchAxis[1]
    v['PitchAxisZ'][:] = PitchAxis[2]
    
    # infer the airfoil directions if not explicitly provided by user
    if 'ChordwiseX' not in LLDict:
        if 'Twist' not in LLDict: LLDict['Twist'] = 0.0
        if RightHandRuleRotation:
            v['ChordwiseY'][:] = -np.cos(np.deg2rad(LLDict['Twist']))
        else:
            v['ChordwiseY'][:] =  np.cos(np.deg2rad(LLDict['Twist']))
        v['ChordwiseZ'][:] = -np.sin(np.deg2rad(LLDict['Twist']))
    else:
        for j in 'XYZ': v['Chordwise'+j][:] = LLDict['Chordwise'+j]


    # normalize
    direction = 'Chordwise'
    for i in range(NPts):
        norm = np.linalg.norm([v[direction+''+j][i] for j in 'XYZ'])
        v[direction+'X'][i] /= norm
        v[direction+'Y'][i] /= norm
        v[direction+'Z'][i] /= norm

    if 'SpanwiseX' not in LLDict:
        v['SpanwiseX'][:] = 1.0
    else:
        for j in 'XYZ': v['Spanwise'+j][:] = LLDict['Spanwise'+j]

    # normalize
    direction = 'Spanwise'
    for i in range(NPts):
        norm = np.linalg.norm([v[direction+''+j][i] for j in 'XYZ'])
        v[direction+'X'][i] /= norm
        v[direction+'Y'][i] /= norm
        v[direction+'Z'][i] /= norm


    if 'ThickwiseX' not in LLDict:
        chordwise_vector = np.vstack( [v['Chordwise'+i] for i in 'XYZ'] )
        spanwise_vector = np.vstack( [v['Spanwise'+i] for i in 'XYZ'] )
        if RightHandRuleRotation:
            thickwise_vector = np.cross(chordwise_vector, spanwise_vector,axisa=0,axisb=0,axisc=0)
        else:
            thickwise_vector = np.cross(spanwise_vector, chordwise_vector,axisa=0,axisb=0,axisc=0)
        v['ThickwiseX'][:] = thickwise_vector[0,:] 
        v['ThickwiseY'][:] = thickwise_vector[1,:] 
        v['ThickwiseZ'][:] = thickwise_vector[2,:]
    else:
        for j in 'XYZ': v['Thickwise'+j][:] = LLDict['Thickwise'+j]

    # normalize
    direction = 'Thickwise'
    for i in range(NPts):
        norm = np.linalg.norm([v[direction+''+j][i] for j in 'XYZ'])
        v[direction+'X'][i] /= norm
        v[direction+'Y'][i] /= norm
        v[direction+'Z'][i] /= norm

    c = np.array(RotationCenter,dtype=float)
    rx = LLx - c[0]
    ry = LLy - c[1]
    v['TangentialX'][:] = -ry
    v['TangentialY'][:] =  rx
    if not RightHandRuleRotation:
        v['TangentialX'] *= -1
        v['TangentialY'] *= -1

    direction = 'Tangential'
    for i in range(NPts):
        norm = np.linalg.norm([v[direction+''+j][i] for j in 'XYZ'])
        v[direction+'X'][i] /= norm
        v[direction+'Y'][i] /= norm
        v[direction+'Z'][i] /= norm


    # Add Information node
    J.set(LiftingLine,'.Component#Info',kind='LiftingLine',
                                        MOLAversion=__version__,
                                        Corrections3D= dict(Sweep=SweepCorrection,Dihedral=DihedralCorrection),
                                        GeometricalLaws=kwargs)
    LiftingLine[0] = 'LiftingLine'

    # add sweep and dihedral angles
    LeadingEdge = I.getZones(getLeadingEdge(LiftingLine))[0]
    LeadingEdgeTangentCurve = D.getTangent(LeadingEdge)
    LiftingLineTangentCurve = D.getTangent(LiftingLine)
    tLEx, tLEy, tLEz = J.getxyz(LeadingEdgeTangentCurve)
    tLLx, tLLy, tLLz = J.getxyz(LiftingLineTangentCurve)
    for i in range(len(tLEx)):
        LeadingEdgeTangentVector = np.array([tLEx[i], tLEy[i], tLEz[i]])
        LiftingLineTangentVector = np.array([tLLx[i], tLLy[i], tLLz[i]])
        ChordwiseVector = np.array([v['ChordwiseX'][:][i], v['ChordwiseY'][:][i], v['ChordwiseZ'][:][i]])
        ThickwiseVector = np.array([v['ThickwiseX'][:][i], v['ThickwiseY'][:][i], v['ThickwiseZ'][:][i]])
        v['SweepAngleDeg'][i] = 90-np.rad2deg(np.arccos(LeadingEdgeTangentVector.dot(ChordwiseVector)))
        v['DihedralAngleDeg'][i] = 90-np.rad2deg(np.arccos(LiftingLineTangentVector.dot(ThickwiseVector)))

    if AngleSmoothingLaw is None or AngleSmoothingLaw == 'None':
        print(J.WARN+'WARNING: no AngleSmoothingLaw has been prescribed. Please check SweepAngleDeg and DihedralAngleDeg to see if there are any curve discontinuities before launching the simulation.'+J.ENDC)

    elif AngleSmoothingLaw == 'UnivariateSpline':
        spl_sweep = si.UnivariateSpline(Span,v['SweepAngleDeg'][:], k=5)
        spl_dihedral = si.UnivariateSpline(Span,v['DihedralAngleDeg'][:], k=5)
        v['SweepAngleDeg'][:] = spl_sweep(Span)
        v['DihedralAngleDeg'][:] = spl_dihedral(Span)

    elif AngleSmoothingLaw == 'Pchip':
        pchip_sweep = si.PchipInterpolator(Span,v['SweepAngleDeg'][:])
        pchip_dihedral = si.PchipInterpolator(Span,v['DihedralAngleDeg'][:])
        v['SweepAngleDeg'][:] = pchip_sweep(Span)
        v['DihedralAngleDeg'][:] = pchip_dihedral(Span)

    elif AngleSmoothingLaw == 'Akima':
        akima_sweep = si.Akima1DInterpolator(Span,v['SweepAngleDeg'][:])
        akima_dihedral = si.Akima1DInterpolator(Span,v['DihedralAngleDeg'][:])
        v['SweepAngleDeg'][:] = akima_sweep(Span)
        v['DihedralAngleDeg'][:] = akima_dihedral(Span)

    elif AngleSmoothingLaw == 'SavgolFilter':
        sf_sweep = savgol_filter((Span,v['SweepAngleDeg'][:]), window_length=31, polyorder=5)
        sf_dihedral = savgol_filter((Span,v['DihedralAngleDeg'][:]),  window_length=31, polyorder=5)
        v['SweepAngleDeg'][:] = sf_sweep[1]
        v['DihedralAngleDeg'][:] = sf_dihedral[1]

    else:
        raise AttributeError('AngleSmoothingLaw "%s" not supported'%AngleSmoothingLaw)


    v['ChordVirtualWithSweep'][:] = LLDict['Chord'] * np.cos(np.deg2rad(v['SweepAngleDeg']))


    return LiftingLine


def buildLiftingLineFromScan(t, SpanwiseRediscretization=None, resetPitchRelativeSpan=0.75,
        GeometricalLawsInterpolations={}, OverridingKinematics={}):
    '''
    This function takes as input the result of :py:func:`MOLA.GenerativeShapeDesign.scanBlade`
    and builds a LiftingLine in its canonical position. The resulting LiftingLine
    can be used directly (polars file is required) as element for BFM or VPM
    modeling, for instance. 

    Parameters
    ----------

        t : PyTree
            Rigorously, the result of :py:func:`MOLA.GenerativeShapeDesign.scanBlade`

        SpanwiseRediscretization : multiple
            **doc this**

        resetPitchRelativeSpan : float
            This is the relative span (or :math:`r/R`) used to relocate the blade
            such that the section at **resetPitchRelativeSpan** yields zero twist.
            This is important, since the actual sweep and dihedral directions
            depend on the pitch value of the blade. In order to circumvent this
            ambiguity, sweep and dihedral positions are defined using as reference
            a relocation of the blade such that the section located at
            **resetPitchRelativeSpan** has zero twist.


        GeometricalLawsInterpolations : dict
            Each keyword correspond to a GeometricalLaw name that can be passed to
            :py:func:`buildLiftingLine`. Also, such keyword must correspond to 
            a field named contained in **t** zone *BladeLine*. The value 
            associated to each keyword is a :py:class:`str` specifying the 
            type of interpolation law. By default, if not specified all 
            geometrical laws are set using the interpolation law ``'interp1d_linear'``

        OverridingKinematics : dict
            By default (if **OverridingKinematics** is an empty 
            :py:class:`dict`) function :py:func:`setKinematicsUsingConstantRotationAndTranslation`
            will be applied using the scanned RotationCenter, axis and direction.
            If **OverridingKinematics** is provided by user, then such options 
            will override the scanned ones.

    Returns
    -------

        LiftingLine : zone
            The new lifting-line located at canonical position. 

        PyZonePolars : :py:class:`list` of zones
            The airfoil polars (only coordinates, no fields), which can be used
            to make a 3D reconstruction of the blade using :py:func:`postLiftingLine2Surface`
        
        scanPitch : float
            The pitch of the section at **resetPitchRelativeSpan**
    '''
    t = I.copyRef(t)

    BaseBladeLine = I.getNodeFromName1(t,'BaseBladeLine')
    if not BaseBladeLine:
        msg = 'could not find base "BaseBladeLine"\n'
        msg += 'Please make sure the first argument is the output of GSD.scanBlade'
        raise IOError(J.FAIL+msg+J.ENDC)

    BladeLine = I.getNodeFromName1(BaseBladeLine,'BladeLine')
    if not BaseBladeLine:
        msg = 'could not find zone "BladeLine"\n'
        msg += 'Please make sure the first argument is the output of GSD.scanBlade'
        raise IOError(J.FAIL+msg+J.ENDC)

    scan_info = J.get(BladeLine,'ScanInformation')
    
    # PUT BLADELINE IN CANONICAL POSITION ROTATING RELEVANT FIELDS
    # put blade line into rotation center (0,0,0)
    T._translate(BladeLine, -scan_info['RotationCenter'])

    RightHandRuleRotation=bool(scan_info['RightHandRuleRotation'])
    Dir = 1 if RightHandRuleRotation else -1

    # put blade line into canonical orientation
    FinalFrame = [(1,0,0),(0,1,0),(0,0,1)]
    InitialFrame = [tuple(scan_info['PitchAxis']),
        tuple(np.cross(Dir*scan_info['RotationAxis'],scan_info['PitchAxis'])),
        tuple(Dir*scan_info['RotationAxis'])]
    T._rotate(BladeLine, (0,0,0), InitialFrame, FinalFrame,
        vectors=NamesOfChordSpanThickwiseFrameNoTangential)


    # determine the pitch of the blade-line
    Abscissa = W.gets(BladeLine)
    Twist, SpanScan = J.getVars(BladeLine,['Twist','Span'])
    RelativeSpanScan = SpanScan/SpanScan.max()
    pitch = np.interp(resetPitchRelativeSpan, RelativeSpanScan, Twist)
    print(f'scanned pitch at r/R={resetPitchRelativeSpan} is {pitch} deg')

    # substract pitch to put blade line in canonical position
    PitchCtr = J.getVars(BladeLine,
        ['PitchRelativeCenterX','PitchRelativeCenterY','PitchRelativeCenterZ'])
    PitchAxis = J.getVars(BladeLine, ['PitchAxisX','PitchAxisY','PitchAxisZ'])
    for i in range(3): PitchAxis[i][:] *= Dir
    PitchCtr_pt = (PitchCtr[0][0]*1.0, PitchCtr[1][0]*1.0, PitchCtr[2][0]*1.0)
    PitchAxis_vec = (PitchAxis[0][0], PitchAxis[1][0], PitchAxis[2][0])
    T._rotate(BladeLine, PitchCtr_pt, PitchAxis_vec, -pitch, 
            vectors=NamesOfChordSpanThickwiseFrameNoTangential)
    
    
    # INFER SWEEP AND DIHEDRAL FROM COORDINATES
    x, y, z = J.getxyz(BladeLine)
    Span, Sweep, Dihedral = J.invokeFields(BladeLine, ['Span', 'Sweep','Dihedral'])
    Span[:]=x
    Sweep[:] = -y
    Dihedral[:] = z
    
    
    # BUILD GEOMETRICALLAWS 
    airfoils = I.getZones(I.getNodeFromName1(t,'BaseNormalizedAirfoils'))
    GeometricalLawsNamesList = ['Chord', 'Twist', 'Sweep','Dihedral', 'Airfoils',
        'ChordwiseX','ChordwiseY','ChordwiseZ',
         'SpanwiseX', 'SpanwiseY', 'SpanwiseZ',
        'ThickwiseX','ThickwiseY','ThickwiseZ']
    
    GeometricalLawsDict = {}
    for geom in GeometricalLawsNamesList:
        # GeometricalLawsDict[geom] = {'RelativeSpan' : RelativeSpanScan}
        GeometricalLawsDict[geom] = {'Abscissa' : Abscissa} # better this (monotonically increasing)
        if geom == 'Airfoils':
            GeometricalLawsDict[geom]['PyZonePolarNames'] = [a[0] for a in airfoils]
        else:
            GeometricalLawsDict[geom][geom] = J.getVars(BladeLine,[geom])[0]
        try:
            GeometricalLawsDict[geom]['InterpolationLaw'] = GeometricalLawsInterpolations[geom]
        except KeyError:
            GeometricalLawsDict[geom]['InterpolationLaw'] = 'interp1d_linear'       

    # Create the lifting line
    if not RightHandRuleRotation: mirrorBlade(BladeLine)
    PitchCtr = J.getVars(BladeLine,
        ['PitchRelativeCenterX','PitchRelativeCenterY','PitchRelativeCenterZ'])
    PitchAxis = J.getVars(BladeLine, ['PitchAxisX','PitchAxisY','PitchAxisZ'])
    PitchCtr_pt = (PitchCtr[0][0]*1.0, PitchCtr[1][0]*1.0, PitchCtr[2][0]*1.0)
    PitchAxis_vec = (PitchAxis[0][0]*1.0, PitchAxis[1][0]*1.0, PitchAxis[2][0]*1.0)

    # build auxiliar curve from user-defined SpanwiseRediscretization
    if isinstance(SpanwiseRediscretization, int):
        SpanwiseCurve = W.discretize(BladeLine, SpanwiseRediscretization)
    elif W.isStructuredCurve(SpanwiseRediscretization):
        SpanwiseCurve = G.map(BladeLine, D.getDistribution(SpanwiseRediscretization))
    elif SpanwiseRediscretization is None:
        SpanwiseCurve = BladeLine
    else:
        dist = J.getDistributionFromHeterogeneousInput__(SpanwiseRediscretization)[-1]
        SpanwiseCurve = G.map(BladeLine,dist)

    LiftingLine = buildLiftingLine(SpanwiseCurve, RightHandRuleRotation=RightHandRuleRotation,
        PitchRelativeCenter=PitchCtr_pt, PitchAxis=PitchAxis_vec,
        RotationCenter=[0,0,0], **GeometricalLawsDict)

    # propagate kinematics information, but keep canonical position
    scan_info['RotationAxis'] *= Dir # since in kinematics it is actually ThrustAxis
    scan_info.update(OverridingKinematics)
    del scan_info['PitchAxis']
    setKinematicsUsingConstantRotationAndTranslation(LiftingLine, **scan_info)

    # CONSTRUCT COORDINATE-ONLY PYZONE POLARS
    PyZonePolarsGeometryOnly = []
    for zone in I.getZones(I.getNodeFromName1(t,'BaseNormalizedAirfoils')):
        airfoil = I.copyRef(zone)
        node = I.getNodeFromName1(airfoil, 'GridCoordinates')
        node[0] = '.Polar#FoilGeometry'
        node[3] = 'UserDefinedData_t'
        I._rmNodesByName1(node,'CoordinateZ')
        I._rmNodesByType1(airfoil,'FlowSolution_t')
        PyZonePolarsGeometryOnly.append(airfoil)
    

    return LiftingLine, PyZonePolarsGeometryOnly, pitch

def checkComponentKind(component, kind='LiftingLine'):
    '''
    Function to determine whether a component (CGNS Base or zone) is of kind
    given by attribute **kind**.

    Parameters
    ----------

        Component : node
            Component whose kind verification is requested

        kind : str
            Kind to verify. For example : ``'Propeller'``, ``'LiftingLine'``...

    Returns
    -------

        Result : bool
            :py:obj:`True` if node kind corresponds to the requested one
    '''
    ZoneInfo = I.getNodeFromName1(component,'.Component#Info')
    if ZoneInfo is None: return False
    kindC = I.getNodeFromName1(ZoneInfo,'kind')
    kindC = I.getValue(kindC)

    return kind == kindC

def getLiftingLines(t):
    return [c for c in I.getZones(t) if checkComponentKind(c, kind='LiftingLine')]

def buildPolarsInterpolatorDict(PyZonePolars, InterpFields=['Cl', 'Cd','Cm'],
        Nrequest=None):
    """
    Build a Python dictionary of interpolation functions of polars from a list
    of **PyZonePolars**. Each key is the name of the **PyZonePolar**
    (the airfoil's tag) and the value is the interpolation function.

    .. note:: typical usage of the returned dictionary goes like this:

        >>> Cl, Cd, Cm = InterpDict['MyPolar'](AoA, Mach, Reynolds)

        where ``AoA``, ``Mach`` and ``Reynolds`` are :py:class:`float` or
        numpy 1D arrays (all yielding the same length)


    Parameters
    ----------

        PyZonePolars : :py:class:`list` of zone
            list of special zones containing the 2D aerodynamic polars of the
            airfoils.

        InterpFields : :py:class:`list` of :py:class:`str`
            list of names of fields to be interpolated.
            Acceptable names are the field names contained in
            all **PyZonePolars** fields located in ``FlowSolution`` container.

        Nrequest : int
            if provided, set the number of points requested by the
            interpolation.

            .. note:: only relevant if technique is ``'PyZoneExtractMesh'``
                *(which is being deprecated!)*

    Returns
    -------

        InterpDict : dict
            resulting python dictionary containing the interpolation functions
            of the 2D polars.
    """
    InterpDict = {}
    for polar in I.getZones(PyZonePolars):
        PolarInterpNode = I.getNodeFromName1(polar,'.Polar#Interp')
        if PolarInterpNode is None: continue
        mode = I.getValue(I.getNodeFromName1(PolarInterpNode,'Algorithm'))

        if mode == 'RbfInterpolator':
            InterpDict[polar[0]] = RbfInterpFromPyZonePolar(polar, InterpFields=InterpFields)
        elif mode == 'PyZoneExtractMesh':
            InterpDict[polar[0]] = extractorFromPyZonePolar(polar, Nrequest, InterpFields=InterpFields)
        elif mode == 'RectBivariateSpline':
            InterpDict[polar[0]] = interpolatorFromPyZonePolar(polar, InterpFields=InterpFields)
        else:
            raise AttributeError('Mode %s not implemented.'%mode)

    return InterpDict


def buildPolarsAnalyticalDict(Name='MyPolar', CLmin=-1.0, CLmax=1.5, CL0=0.0, CLa=2*np.pi,
        CD0 = 0.011, CD2u = 0.004, CD2l = 0.013, CLCD0 = 0.013, REref = 1.e6,
        REexp = 0.):
    """
    Construct a python dictionary of analytical functions allowing for
    determination of aerodynamic coefficients :math:`(c_l,\, c_d,\, c_m)`.
    The call of the analytical functions is made as follows:

    >>> Cl, Cd, Cm = AnalyticalDict['MyPolar'](AoA, Mach, Reynolds)


    The paramaters of :py:func:`buildPolarsAnalyticalDict` are coefficients
    (:py:class:`float`) that define the mathematical analytical functions like
    follows:

    ::

            # Linear for CL(AoA)
            CL = np.minimum(np.maximum((CL0 + CLa*np.deg2rad(AoA))/np.sqrt(1-Mach**2),CLmin),CLmax)

            # Double parabola for CD(CL)
            CD2 = CL*0
            CD2[CL>CLCD0]  = CD2u
            CD2[CL<=CLCD0] = CD2l
            CD = (CD0+CD2*(CL-CLCD0)**2)*(Reynolds/REref)**REexp

            CM = 0.

    Returns
    -------

        AnalyticalDict : dict
            dictionary containing the analytical functions.
    """
    def analyticalPolar(AoA,Mach,Reynolds):
        # Linear for CL(AoA)
        CL = np.minimum(np.maximum((CL0 + CLa*np.deg2rad(AoA))/np.sqrt(1-Mach**2),CLmin),CLmax)

        # Double parabola for CD(CL)
        CD2 = CL*0
        CD2[CL>CLCD0]  = CD2u
        CD2[CL<=CLCD0] = CD2l
        CD = (CD0+CD2*(CL-CLCD0)**2)*(Reynolds/REref)**REexp

        CM = 0.

        DictOfVals = dict(Cl=CL, Cd=CD, Cm=CM)

        return DictOfVals['Cl'], DictOfVals['Cd'], DictOfVals['Cm']

    InterpDict = dict(Name=analyticalPolar)

    return InterpDict

def buildLiftingLineInterpolator(LiftingLine, InterpFields=['Cl', 'Cd', 'Cm']):
    '''

    .. danger:: this function is to be deprecated (replaced by
        :py:func:`_applyPolarOnLiftingLine`).

    This method employs Cassiopee's Connector interpolation capabilities for
    interpolation of 2D Polar data.

    Parameters
    ----------

        LiftingLine : zone
            Lifting line.

        InterpFields : :py:class:`list` of :py:class:`str`
            name of fields to be interpolated

    Returns
    -------

        DataSurface : zone
            surface zone containing data of polars

        RequestLine : zone
            curve zone yielding the request points
    '''

    # Get curvilinear abscissa of actual LiftingLine
    s, = J.getVars(LiftingLine,['s'])

    # Get Airfoils data
    ComponentInfoNode = I.getNodeFromName1(LiftingLine,'Airfoils')
    PolarInfo= I.getNodeFromName1(LiftingLine,'Airfoils')
    Abscissa= I.getValue(I.getNodeFromName1(PolarInfo,'Abscissa'))
    InterpolationLaw = I.getValue(I.getNodeFromName1(PolarInfo,'InterpolationLaw'))

    # Dimensions of interpolation
    Ns = len(s)         # = number of pts of LiftingLine
    Na = len(Abscissa)  # = Number of PyZonePolars

    # Build DataSurface:
    DataSurface = G.cart((0,0,0), (1,1,1),(Ns,Na,1))
    DSx, DSy = J.getxy(DataSurface)
    DSx[:] = np.broadcast_to(s,(Na,Ns)).T
    DSy[:] = np.broadcast_to(Abscissa,(Ns,Na))
    T._addkplane(DataSurface)
    T._translate(DataSurface,(0,0,-0.5))
    J._invokeFields(DataSurface,InterpFields)

    # Build RequestLine
    RequestLine = D.line((0,0,0),(1,1,0),Ns)
    RLx, RLy = J.getxy(RequestLine)
    RLx[:]= RLy[:] = s
    J._invokeFields(RequestLine,InterpFields)

    # Prepare interpolation data
    C._initVars(RequestLine,'cellN',2)
    C._initVars(DataSurface,'cellN',1)
    X._setInterpData(RequestLine, [DataSurface], order=2,
                     nature=1, loc='nodes', storage='direct', hook=None,
                     method='lagrangian', dim=2)

    LiftingLineInterpolator = DataSurface, RequestLine

    return LiftingLineInterpolator


def interpolatorFromPyZonePolar(PyZonePolar, interpOptions=None,
        InterpFields=['Cl', 'Cd', 'Cm']):
    '''
    This function create the interpolation function of Polar
    data of an airfoil stored as a PyTree Zone.

    It handles out-of-range polar-specified angles of attack.

    Parameters
    ----------

        PyZonePolar : zone
            PyTree Zone containing Polar information,
            as produced by e.g. :py:func:`convertHOSTPolarFile2PyZonePolar`

        interpOptions : options to pass to the interpolator function.

            .. warning:: this will be included as **PyZonePolar** node in future
                versions

        InterpFields : :py:class:`tuple` of :py:class:`str`
            contains the strings of the variables to be interpolated.

    Returns
    -------

        InterpolationFunctions : function
            a function to be employed like this:

            >>> InterpolationFunctions(AoA, Mach, Reynolds, ListOfEquations=[])

    '''
    import scipy.interpolate as si


    # Get the fields to interpolate
    Data = {}
    DataRank = {}
    FS_n = I.getNodeFromName1(PyZonePolar,'FlowSolution')
    FV_n = I.getNodeFromName1(PyZonePolar,'.Polar#FoilValues')
    for IntField in InterpFields:
        Field_n = I.getNodeFromName1(FS_n,IntField)
        if Field_n:
            Data[IntField] = I.getValue(Field_n)
            DataRank[IntField] = len(Data[IntField].shape)
        else:
            Field_n = I.getNodeFromName1(FV_n,IntField)
            if Field_n:
                Data[IntField] = I.getValue(Field_n)
                DataRank[IntField] = len(Data[IntField].shape)

    PR_n = I.getNodeFromName1(PyZonePolar,'.Polar#Range')
    AoARange=I.getValue(I.getNodeFromName1(PR_n,'AngleOfAttack'))
    NAoARange=len(AoARange)
    MachRange=I.getValue(I.getNodeFromName1(PR_n,'Mach'))
    MachRangeMax = MachRange.max()
    MachRangeMin = MachRange.min()
    NMachRange=len(MachRange)

    OutOfRangeValues_ParentNode = I.getNodeFromName1(PyZonePolar,'.Polar#OutOfRangeValues')

    BigAoARange      = {}
    OutOfRangeValues = {}
    for IntField in InterpFields:
        BigAoARangeVar_n = I.getNodeFromName1(PR_n,'BigAngleOfAttack%s'%IntField)
        if BigAoARangeVar_n is None:
            BigAoARangeVar_n = I.getNodeFromName1(PR_n,'BigAngleOfAttackCl')
        BigAoARange[IntField] = I.getValue(BigAoARangeVar_n)

        OutOfRangeValues_n = I.getNodeFromName1(OutOfRangeValues_ParentNode,'BigAngleOfAttack%s'%IntField)
        if OutOfRangeValues_n is not None:
            OutOfRangeValues[IntField] = I.getValue(OutOfRangeValues_n)


    # -------------- BUILD INTERPOLATOR -------------- #
    # Currently, only scipy.interpolator based objects
    # are supported. In future, this shall be extended
    # to e.g: dakota, rncarpio, scattered, sparse...

    # 2D interpolation
    try:
        ReynoldsOverMach = I.getNodeFromName(PyZonePolar,'ReynoldsOverMach')[1][0]
    except TypeError:
        ReynoldsOverMach = None
        ReynoldsRange= I.getNodeFromName(PyZonePolar,'Reynolds')[1]

    if interpOptions is None: interpOptions = dict(kx=1, ky=1)


    # (AoA, Mach) interpolation
    # using si.RectBivariateSpline()
    tableInterpFuns = {}
    for IntField in InterpFields:


        if DataRank[IntField] == 2:
            # Integral quantity: Cl, Cd, Cm, Xtr...

            # Create extended angle-of-attack and data range
            lowAoA  = BigAoARange[IntField] < 0
            highAoA = BigAoARange[IntField] > 0
            ExtAoARange = np.hstack((BigAoARange[IntField][lowAoA],AoARange,BigAoARange[IntField][highAoA]))
            DataLow = np.zeros((np.count_nonzero(lowAoA),NMachRange),dtype=np.float64,order='F')
            DataHigh = np.zeros((np.count_nonzero(highAoA),NMachRange),dtype=np.float64,order='F')
            for m in range(NMachRange):
                if IntField in OutOfRangeValues:
                    DataLow[:,m]  = OutOfRangeValues[IntField][lowAoA]
                    DataHigh[:,m] = OutOfRangeValues[IntField][highAoA]
                else:
                    DataLow[:,m]  = 0
                    DataHigh[:,m] = 0

            ExtData = np.vstack((DataLow,Data[IntField],DataHigh))
            # Create Extended data range
            tableInterpFuns[IntField] = si.RectBivariateSpline(ExtAoARange,MachRange, ExtData, **interpOptions)
            # tableInterpFuns[IntField] is a function

        elif DataRank[IntField] == 3:
            # Foil-distributed quantity: Cp, delta*, theta...
            tableInterpFuns[IntField] = []
            for k in range(Data[IntField].shape[2]):
                interpFun = si.RectBivariateSpline(AoARange,MachRange, Data[IntField][:,:,k], **interpOptions)
                tableInterpFuns[IntField] += [interpFun]
            # tableInterpFuns[IntField] is a list of functions

        else:
            raise ValueError('FATAL ERROR: Rank of data named "%s" to be interpolated is %d, and must be 2 (for integral quantities like Cl, Cd...) or 3 (for foil-distributed quantities like Cp, theta...).\nCheck your PyZonePolar data.'%(IntField,DataRank[IntField]))


    def interpolationFunction(AoA, Mach, Reynolds):
        # This function should be as efficient as possible

        # BEWARE : RectBiVariate ignores Reynolds at
        # interpolation step


        Mach = np.clip(Mach,MachRangeMin,MachRangeMax)

        # Apply RectBiVariate interpolator
        ListOfValues = []
        for IntField in InterpFields:


            if DataRank[IntField] == 2:
                ListOfValues += [tableInterpFuns[IntField](AoA, Mach,grid=False)]
            else:
                # DataRank[IntField] == 3
                FoilValues = []
                for tableIntFun in tableInterpFuns[IntField]:
                    FoilValues += [[tableIntFun(AoA[ir], Mach[ir], grid=False) for ir in range(len(AoA))]]

                ListOfValues += [np.array(FoilValues,dtype=np.float64,order='F')]


        return ListOfValues

    return interpolationFunction

def extractorFromPyZonePolar(PyZonePolar, Nrequest,
        InterpFields=['Cl', 'Cd', 'Cm']):
    '''

    .. danger:: this function is being deprecated

    This function create the interpolation function of Polar
    data of an airfoil stored as a PyTree Zone.

    It handles out-of-range polar-specified angles of attack.

    Parameters
    ----------

         PyZonePolar : zone
            PyTree Zone containing Polar information,
            as produced by e.g. :py:func:`convertHOSTPolarFile2PyZonePolar`

        interpOptions : dict
            options to pass to the interpolator
            function.

            .. note:: this will be included in a node of **PyZonePolar**

        InterpFields - :py:func`tuple` of :py:func`str`
            specify the variables to be interpolated.

    Returns
    -------

        InterpolationFunctions : function
            function with usage:

            >>> InterpolationFunctions(AoA, Mach, Reynolds, ListOfEquations=[])

    '''


    # Get the fields to interpolate
    Data = {}
    DataRank = {}
    for IntField in InterpFields:
        Data[IntField] = I.getNodeFromName(PyZonePolar,IntField)[1]
        DataRank[IntField] = len(Data[IntField].shape)

    AoARange = I.getNodeFromName(PyZonePolar,'AngleOfAttack')[1]
    NAoARange = len(AoARange)
    MachRange= I.getNodeFromName(PyZonePolar,'Mach')[1]
    NMachRange = len(MachRange)

    BigAoARange = {}
    for IntField in InterpFields:
        BigAoARangeVar_n = I.getNodeFromName(PyZonePolar,'BigAngleOfAttack%s'%IntField)
        if BigAoARangeVar_n is None:
            BigAoARangeVar_n = I.getNodeFromName(PyZonePolar,'BigAngleOfAttackCl')
        BigAoARange[IntField] = BigAoARangeVar_n[1]


    # (AoA, Mach) interpolation
    # using P._extractMesh() --> only for DataRank == 2 !

    # Create PolarDataSurface
    PolarDataSurface = G.cart((0,0,-0.5), (1,1,1),(NMachRange,NAoARange,2))
    PolarDataSurface[0] = 'DataSurf_%s'%PyZonePolar[0]
    DSx, DSy = J.getxy(PolarDataSurface)
    DSx[:,:,0] = DSx[:,:,1]  = np.broadcast_to(MachRange,(NAoARange,NMachRange)).T
    DSy[:,:,0] = DSy[:,:,1] = np.broadcast_to(AoARange,(NMachRange,NAoARange))
    PZVarsDict = J.getVars2Dict(PyZonePolar,InterpFields)
    PDSVarsDict = J.invokeFieldsDict(PolarDataSurface,InterpFields)
    for pzvar in InterpFields:
        PDSVarsDict[pzvar][:,:,0] = PDSVarsDict[pzvar][:,:,1] = PZVarsDict[pzvar].T

    # Create PolarRequestLine
    PolarRequestLine = D.line((MachRange[0],AoARange[0],0),(MachRange[-1],AoARange[-1],0),Nrequest)
    PolarRequestLine[0] = 'ReqLine_%s'%PyZonePolar[0]
    J._invokeFields(PolarRequestLine,InterpFields)

    C._initVars(PolarRequestLine,'cellN',2)
    C._initVars(PolarDataSurface,'cellN',1)
    hook = C.createHook(PolarDataSurface,'extractMesh')

    def interpolationFunction(AoA, Mach, Reynolds):

        if isinstance(AoA,list): AoA = np.array(AoA,dtype=np.float64, order='F')
        if isinstance(Mach,list): Mach = np.array(Mach,dtype=np.float64, order='F')
        if isinstance(Reynolds,list): Reynolds = np.array(Reynolds,dtype=np.float64, order='F')


        if all(np.isnan(Reynolds)): raise ValueError('all-NaN Found in Reynolds')
        elif any(np.isnan(Reynolds)): Reynolds[np.isnan(Reynolds)] = 0

        if all(np.isnan(Mach)): raise ValueError('all-NaN Found in Mach')
        elif any(np.isnan(Mach)): Mach[np.isnan(Mach)] = 0


        Npts = len(AoA)

        # Adapt PolarRequestLine
        RLx, RLy = J.getxy(PolarRequestLine)
        RLx[:] = Mach
        RLy[:] = AoA

        # Interpolate in-range AoA values
        # P._extractMesh([PolarDataSurface],PolarRequestLine, order=2, extrapOrder=1, constraint=1e6, tol=1e-4, mode='robust', hook=None)
        X._setInterpData(PolarRequestLine, [PolarDataSurface], order=2, nature=1, loc='nodes', storage='RightHandRuleRotation', hook=[hook],  method='lagrangian',dim=2)
        X._setInterpTransfers(PolarRequestLine, [PolarDataSurface], variables=InterpFields)
        # print('ElapsedTime _setInterpTransfers: %g s'%ElapsedTime)
        # sys.exit()
        Values = J.getVars2Dict(PolarRequestLine,InterpFields)

        # Interpolate in out-of-range AoA values
        InOfRange = np.zeros(Npts,dtype=np.bool)
        for i in range(Npts):
            InOfRange[i] = True if (AoA[i] > AoARange[0]) and (AoA[i] < AoARange[-1]) else False
        OutOfRange = np.logical_not(InOfRange)
        OutOfRangeValues_ParentNode = I.getNodeFromName(PyZonePolar,'.Polar#OutOfRangeValues')

        if np.any(OutOfRange):
            for IntField in InterpFields:
                OutOfRangeValues_n = I.getNodeFromName(OutOfRangeValues_ParentNode,'BigAngleOfAttack%s'%IntField)
                if OutOfRangeValues_n is not None:
                    Values[IntField][OutOfRange] = np.interp(AoA[OutOfRange], BigAoARange[IntField],OutOfRangeValues_n[1])

        ListOfValues = [Values[IntField] for IntField in InterpFields]


        return ListOfValues


    return interpolationFunction

def RbfInterpFromPyZonePolar(PyZonePolar, InterpFields=['Cl', 'Cd', 'Cm']):
    '''
    This function creates the interpolation function of Polar
    data of an airfoil stored as a PyTree Zone, using radial-basis-functions.

    It handles out-of-range polar-specified angles of attack.

    Parameters
    ----------

        PyZonePolar : PyTree Zone containing Polar information,
            as produced by e.g. :py:func:`convertHOSTPolarFile2PyZonePolar`

        interpOptions : dict
            options to pass to the interpolator function.

            .. warning:: this will be include in a node inside **PyZonePolar**

        InterpFields : :py:class:`tuple` of :py:class:`str`
            variables to be interpolated.

    Returns
    -------

        InterpolationFunction : function
            function of interpolation, with expected usage:

            >>> Cl, Cd, Cm = InterpolationFunction(AoA, Mach, Reynolds)
    '''
    from scipy.spatial import Delaunay
    import scipy.interpolate as si

    # Check kind of PyZonePolar
    PolarInterpNode = I.getNodeFromName1(PyZonePolar,'.Polar#Interp')
    PyZonePolarKind = I.getValue(I.getNodeFromName1(PolarInterpNode,'PyZonePolarKind'))
    Algorithm = I.getValue(I.getNodeFromName1(PolarInterpNode,'Algorithm'))
    if PyZonePolarKind != 'Unstr_AoA_Mach_Reynolds':
        raise AttributeError('RbfInterpolator object can only be associated with a PyZonePolar of type "Unstr_AoA_Mach_Reynolds". Check PyZonePolar "%s"'%PyZonePolar[0])
    if Algorithm != 'RbfInterpolator':
        raise ValueError("Attempted to use RbfInterpolator, but Algorithm node in PyZonePolar named '%s' was '%s'"%(PyZonePolar[0], Algorithm))

    # Get the fields to interpolate
    Data       = {}
    DataRank   = {}
    DataShape  = {}
    for IntField in InterpFields:
        Data[IntField] = I.getNodeFromName(PyZonePolar,IntField)[1]
        DataShape[IntField]  = Data[IntField].shape
        DataRank[IntField] = len(DataShape[IntField])

    # Get polar independent variables (AoA, Mach, Reynolds)
    PolarRangeNode = I.getNodeFromName1(PyZonePolar,'.Polar#Range')
    AoARange = I.getNodeFromName1(PolarRangeNode,'AngleOfAttack')[1]
    MachRange = I.getNodeFromName1(PolarRangeNode,'Mach')[1]
    ReRange = I.getNodeFromName1(PolarRangeNode,'Reynolds')[1]

    # Compute bounding box of independent variables
    AoAMin,  AoAMax =  AoARange.min(),  AoARange.max()
    ReMin,    ReMax =   ReRange.min(),   ReRange.max()
    MachMin,MachMax = MachRange.min(), MachRange.max()

    # Compute ranges of big angle-of-attack
    BigAoARange = {}
    OutOfRangeValues_ParentNode = I.getNodeFromName(PyZonePolar,'.Polar#OutOfRangeValues')
    for IntField in InterpFields:
        BigAoARangeVar_n = I.getNodeFromName(PyZonePolar,'BigAngleOfAttack%s'%IntField)
        if BigAoARangeVar_n is None:
            BigAoARangeVar_n = I.getNodeFromName(PyZonePolar,'BigAngleOfAttackCl')
        BigAoARange[IntField] = BigAoARangeVar_n[1]

    # Compute Delaunay triangulation of independent variables
    # (AoA, Mach, Reynolds)
    points = np.vstack((AoARange,MachRange,ReRange)).T
    triDelaunay = Delaunay(points)

    # CONSTRUCT INTERPOLATORS
    # -> inside qhull : use Rbf interpolator
    # -> outside qhull but inside ranges BoundingBox : use
    #       NearestNDInterpolator
    # -> outside ranges BoundingBox : use interp1d_linear
    #       on Big angle-of-attack data, if available

    inQhullFun, outQhullFun, outMaxAoAFun, outMinAoAFun = {}, {}, {}, {}
    def makeNaNFun(dummyArray):
        newArray = dummyArray*0.
        newArray[:] = np.nan
        return newArray

    for IntField in InterpFields:
        if DataRank[IntField] == 1:
            # Integral quantity: Cl, Cd, Cm, Top_Xtr...
            '''
            Rbf functions:
            'multiquadric' # ok
            'inverse'      # bit expensive
            'gaussian'     # expensive (and innacurate?)
            'linear'       # ok
            'cubic'        # expensive
            'quintic'      # expensive
            'thin_plate'   # bit expensive
            '''
            inQhullFun[IntField] = si.Rbf(0.1*AoARange, MachRange,1e-6*ReRange, Data[IntField], function='multiquadric',
                smooth=1, # TODO: control through PyTree node
                )
            outQhullFun[IntField] = si.NearestNDInterpolator(points,Data[IntField])
            outBBRangeValues_n = I.getNodeFromName(OutOfRangeValues_ParentNode,'BigAngleOfAttack%s'%IntField)
            if outBBRangeValues_n is not None:
                MaxAoAIndices = BigAoARange[IntField]>0
                outMaxAoAFun[IntField] = si.interp1d( BigAoARange[IntField][MaxAoAIndices], outBBRangeValues_n[1][MaxAoAIndices], assume_sorted=True, copy=False,fill_value='extrapolate')
                MinAoAIndices = BigAoARange[IntField]<0
                outMinAoAFun[IntField] = si.interp1d( BigAoARange[IntField][MinAoAIndices], outBBRangeValues_n[1][MinAoAIndices], assume_sorted=True, copy=False,fill_value='extrapolate')
            else:
                outMaxAoAFun[IntField] = makeNaNFun
                outMinAoAFun[IntField] = makeNaNFun

        elif DataRank[IntField] == 2:
            # Foil-distributed quantity: Cp, delta1, theta...
            inQhullFun[IntField]  = []
            outQhullFun[IntField] = []
            outBBFun[IntField]    = []

            outBBRangeValues_n = I.getNodeFromName(OutOfRangeValues_ParentNode,'BigAngleOfAttack%s'%IntField)
            for k in range(DataShape[IntField][1]):
                inQhullFun[IntField] += [si.Rbf(0.1*AoARange, MachRange,1e-6*ReRange, Data[IntField][:,k], function='multiquadric',
                smooth=0, # TODO: control through PyTree node
                )]
                outQhullFun[IntField] += [si.NearestNDInterpolator(points,Data[IntField][:,k])]
                if outBBRangeValues_n is not None:
                    outBBFun[IntField] += [si.interp1d( BigAoARange[IntField][:,k], outBBRangeValues_n[1][:,k], assume_sorted=True, copy=False)]
                else:
                    outBBFun[IntField] += [makeNaNFun]

        else:
            raise ValueError('FATAL ERROR: Rank of data named "%s" to be interpolated is %d, and must be 1 (for integral quantities like Cl, Cd...) or 2 (for foil-distributed quantities like Cp, theta...).\nCheck your PyZonePolar data.'%(IntField,DataRank[IntField]))


    def interpolationFunction(AoA, Mach, Reynolds):

        # Check input data structure
        if isinstance(AoA,list): AoA = np.array(AoA,dtype=np.float64, order='F')
        if isinstance(Mach,list): Mach = np.array(Mach,dtype=np.float64, order='F')
        if isinstance(Reynolds,list): Reynolds = np.array(Reynolds,dtype=np.float64, order='F')

        # Replace some NaN in Mach or Reynolds number by 0
        if all(np.isnan(Mach)): raise ValueError('all-NaN Found in Mach')
        elif any(np.isnan(Mach)): Mach[np.isnan(Mach)] = 0

        if all(np.isnan(Reynolds)): raise ValueError('all-NaN Found in Reynolds')
        elif any(np.isnan(Reynolds)): Reynolds[np.isnan(Reynolds)] = 0

        # Find boolean ranges depending on requested data:
        OutAoAMax = AoA > AoAMax
        AnyOutAoAMax = np.any(OutAoAMax)
        OutAoAMin = AoA < AoAMin
        AnyOutAoAMin = np.any(OutAoAMin)
        outBB = OutAoAMax + OutAoAMin
        AllOutBB = np.all(outBB)
        AnyOutBB = np.any(outBB)
        inBB  = np.logical_not(outBB)

        # Interpolate for each requested field "IntField"
        Values = {}
        FirstField = True
        for IntField in InterpFields:

            if DataRank[IntField] == 1:
                Values[IntField] = AoA*0 # Declare array

                if not AllOutBB:
                    # Compute values inside Bounding-Box
                    Values[IntField][inBB] = inQhullFun[IntField](0.1*AoA[inBB], Mach[inBB], 1e-6*Reynolds[inBB])

                    # Determine compute points outside Qhull but
                    # still inside Bounding-Box
                    if FirstField:
                        inBBoutQhull = np.isnan(Values[IntField])
                        someInBBoutQhull = np.any(inBBoutQhull)

                    # Compute outside-Qhull points by nearest
                    # point algorithm
                    if someInBBoutQhull:
                        Values[IntField][inBBoutQhull] = outQhullFun[IntField](AoA[inBBoutQhull], Mach[inBBoutQhull], Reynolds[inBBoutQhull])

                # Compute outside big-angle of attack values
                if AnyOutAoAMax:
                    Values[IntField][OutAoAMax] = outMaxAoAFun[IntField](np.minimum(np.maximum(AoA[OutAoAMax],-180.),+180.))
                if AnyOutAoAMin:
                    Values[IntField][OutAoAMax] = outMinAoAFun[IntField](np.minimum(np.maximum(AoA[OutAoAMax],-180.),+180.))


            else:
                # DataRank[IntField] == 2
                FoilValues = []
                for k in range(DataShape[IntField][1]):
                    CurrentValues = AoA*0 # Declare array

                    if not AllOutBB:
                        # Compute values inside Bounding-Box
                        CurrentValues[inBB] = inQhullFun[IntField](AoA[inBB], Mach[inBB], Reynolds[inBB])

                        # Determine compute points outside Qhull but
                        # still inside Bounding-Box
                        if FirstField:
                            inBBoutQhull = np.isnan(Values[IntField])
                            someInBBoutQhull = np.any(inBBoutQhull)

                        # Compute outside-Qhull points by nearest
                        # point algorithm
                        if someInBBoutQhull:
                            CurrentValues[inBBoutQhull] = outQhullFun[IntField](AoA[inBBoutQhull], Mach[inBBoutQhull], Reynolds[inBBoutQhull])

                    # Compute outside big-angle of attack values
                    if AnyOutBB:
                        CurrentValues[outBB] = outBBFun[IntField](AoA[outBB])

                    FoilValues += [CurrentValues]

                Values[IntField] = np.vstack(FoilValues,dtype=np.float64,order='F')
            FirstField = False
        ListOfValues = [Values[IntField] for IntField in InterpFields]

        return ListOfValues

    return interpolationFunction

def _applyPolarOnLiftingLine(LiftingLines, PolarsInterpolatorDict,
                             InterpFields=['Cl', 'Cd','Cm']):
    """
    This function computes aerodynamic characteristics of each section of the
    LiftingLine using the local conditions defined by ``AoA``, ``Mach`` and
    ``Reynolds`` fields (located in LiftingLine's vertices, at the container
    ``FlowSolution``).

    Parameters
    ----------

        LiftingLines : Tree, Base, Zone or list of Zone
            LiftingLine curves with ``AoA``, ``Mach``, ``Reynolds``
            fields defining the local flow characteristics. New interpolated fields
            will be added into ``FlowSolution`` container.

            .. note:: zone **LiftingLine** is modified

        PolarsInterpolatorDict : dict
            dictionary of interpolator functions of 2D polars, as obtained from
            :py:func:`buildPolarsInterpolatorDict` function.

        InterpFields : :py:class:`list` of :py:class:`str`
            names of aerodynamic characteristics to be interpolated.
            These fields are added to **LiftingLine**.
    """

    # TODO remove starting "_" from function name

    LiftingLines = [z for z in I.getZones(LiftingLines) if checkComponentKind(z,'LiftingLine')]
    for LiftingLine in LiftingLines:

        # Get the required fields
        FlowSolution = I.getNodeFromName1(LiftingLine,'FlowSolution')
        DictOfVars = {}
        for Var in ['AoA', 'Mach', 'Reynolds', 's']+InterpFields:
            DictOfVars[Var] = I.getNodeFromName1(FlowSolution,Var)[1]

        # Get Airfoils data
        PolarInfo= getAirfoilsNodeOfLiftingLine(LiftingLine)
        Abscissa = I.getNodeFromName1(PolarInfo,'Abscissa')[1]
        NodeStr = I.getValue(I.getNodeFromName1(PolarInfo,'PyZonePolarNames'))
        PyZonePolarNames = NodeStr.split(' ')
        InterpolationLaw = I.getValue(I.getNodeFromName1(PolarInfo,'InterpolationLaw'))

        # Interpolates IntField (Cl, Cd,...) from polars to LiftingLine
        NVars   = len(InterpFields)
        VarArrays = {}
        for IntField in InterpFields:
            VarArrays[IntField] = []

        for PolarName in PyZonePolarNames:
            ListOfVals = PolarsInterpolatorDict[PolarName](DictOfVars['AoA'],
                DictOfVars['Mach'],
                DictOfVars['Reynolds'])

            for i in range(NVars):
                try:
                    VarArrays[InterpFields[i]] += [ListOfVals[i]]
                except IndexError as e:
                    C.convertPyTree2File(LiftingLine,'testLL.cgns')
                    raise IndexError(e)

        for IntField in InterpFields:
            VarArrays[IntField] = np.vstack(VarArrays[IntField])
            Res = J.interpolate__(DictOfVars['s'],Abscissa,VarArrays[IntField],Law=InterpolationLaw, axis=0)
            DictOfVars[IntField][:] = np.diag(Res)


        # Bit faster approach:
        # tic = timeit.default_timer()
        # NPts = C.getNPts(LiftingLine)
        # for IntField in InterpFields:
        #     VarArrays[IntField] = np.vstack(VarArrays[IntField])
        #     RBSpline = si.RectBivariateSpline(Abscissa, DictOfVars['s'], VarArrays[IntField], kx=1,ky=1)
        #     for i in range(NPts):
        #         s = DictOfVars['s'][i]
        #         Res = RBSpline(s,s)
        #         DictOfVars[IntField][i] = Res
        # toc = timeit.default_timer()
        # CostApplyPolar[0] = CostApplyPolar[0]+(toc-tic)

def _findOptimumAngleOfAttackOnLiftLine(LiftingLine, PolarsInterpolatorDict,
        Aim='Cl', AimValue=0.5, AoASearchBounds=(-2,6),
        SpecificSections=None, ListOfEquations=[]):
    """
    Update ``AoA`` field with the optimum angle of attack based on a given
    **Aim**, using the provided **PolarsInterpolatorDict** as well as the existing
    ``Reynolds`` and ``Mach`` number values contained in ``FlowSolution`` container.

    Parameters
    ----------

        LiftingLine : zone
            the lifting line where ``AoA`` field will be updated

            .. note:: zone **LiftingLine** is modified

        PolarsInterpolatorDict : dict
            dictionary of
            interpolator functions of 2D polars, as obtained from
            :py:func:`buildPolarsInterpolatorDict` function.

        Aim : str
            can be one of:

            * ``'Cl'``
                aims a requested ``Cl`` (:math:`c_l`) value (provided by argument
                **AimValue**) throughout the entire lifting line

            * ``'minCd'``
                aims the minimum ``Cd`` value, :math:`\min (c_d)`

            * ``'maxClCd'``
                aims the maximum ``Cl/Cd`` value, :math:`\max (c_l / c_d)`

        AimValue : float
            Specifies the aimed value for corresponding relevant
            Aim types.

            .. note:: currently, only relevant for **Aim** = ``'Cl'``

        AoASearchBounds : :py:class:`tuple` of 2 :py:class:`float`
            Since there may exist multiple angle-of-attack (*AoA*) values
            verifying the requested conditions, this argument constraints the
            research interval of angle-of-attack of valid candidates.

        SpecificSections : :py:class:`list` of :py:class:`int`
            If specified (not :py:obj:`None`), only the
            indices corresponding to the user-provided sections are updated.

        ListOfEquations : :py:class:`list` of :py:class:`str`
            list of equations compatible with
            the syntax allowed in :py:func:`Converter.PyTree.initVars`
            (``FlowSolution`` located at vertex), in order to tune or correct
            the aerodynamic coefficients.
    """

    import scipy.optimize as so
    AoA, Cl, Cd, Mach, Reynolds = J.getVars(LiftingLine,['AoA','Cl','Cd','Mach', 'Reynolds'])

    if SpecificSections is None: SpecificSections = range(len(AoA))

    if Aim == 'Cl':
        for i in SpecificSections:
            def searchAoA(x,i):
                AoA[i] = x
                _applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict)
                [C._initVars(LiftingLine, eq) for eq in ListOfEquations]

                Residual = Cl[i]-AimValue
                return Residual

            sol=so.root_scalar(searchAoA, bracket=AoASearchBounds, x0=AoA[i], args=(i),  method='toms748')

            if sol.converged:
                searchAoA(sol.root,i)
            else:
                print ("Not found optimum AoA at section %d"%i)
                print (sol)
                continue
    elif Aim == 'maxClCd':
        for i in SpecificSections:
            def searchAoA(x,i):
                AoA[i] = x
                _applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict)
                [C._initVars(LiftingLine, eq) for eq in ListOfEquations]
                MinimizeThis = -Cl[i]/Cd[i]
                return MinimizeThis

            sol=so.minimize_scalar(searchAoA, bracket=[0,2], args=(i),  method='Golden',
                options=dict(xtol=0.01))


            if not sol.success:
                print ("Not found optimum AoA at section %d"%i)
                print (sol)
                continue
    elif Aim == 'minCd':
        for i in SpecificSections:
            def searchAoA(x,i):
                AoA[i] = x
                _applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict)
                [C._initVars(LiftingLine, eq) for eq in ListOfEquations]

                MinimizeThis = Cd[i]
                return MinimizeThis

            sol=so.minimize_scalar(searchAoA, bracket=AoASearchBounds, args=(i),  method='Golden',
                options=dict(xtol=0.01))

            if not sol.success:
                print ("Not found optimum AoA at section %d"%i)
                print (sol)
                continue

def pyZonePolar2AirfoilZone(pyzonename, PyZonePolars):
    '''
    Conveniently use ``.Polar#FoilGeometry`` coordinates of a **PyZonePolar** in
    order to build a structured curve zone.

    Parameters
    ----------

        pyzonename : :py:class:`str` or :py:class:`list` of :py:class:`str`
            name(s) to be employed in new curve(s) zone(s) defining the airfoil
            geometry.

        PyZonePolars : :py:class:`list` of zone or :py:class:`str`
            list of special **PyZonePolar** zones
            containing 2D polar information as well as the geometry. Specifically,
            the polars must contain the node ``.Polar#FoilGeometry`` with the
            children nodes ``CoordinateX`` and ``CoordinateY``.

            .. hint:: 
                if **PyZonePolars** is a :py:class:`str`, then it will be
                interpreted as a file name, and it will attempt to open it and
                return the zones

    Returns
    -------

        AirfoilGeom : zone or :py:class:`list` of zone
            the 1D curve(s) of the airfoil.
    '''
    if isinstance(PyZonePolars,str):
        PyZonePolars = J.load(PyZonePolars, return_type='zones')

    if isinstance(pyzonename,str):
        pyzonenames = [pyzonename]
    else:
        pyzonenames = pyzonename
    
    PyZonePolars = I.getZones(PyZonePolars)
    if not PyZonePolars: raise ValueError('not PyZonePolars')


    AirfoilGeoms = []
    for pzn in pyzonenames:
        zone = J.getZoneFromListByName(PyZonePolars, pzn)
        if not zone:
            raise ValueError('not zone')
        FoilGeom_n = I.getNodeFromName1(zone,'.Polar#FoilGeometry')
        Xcoord = I.getNodeFromName1(FoilGeom_n,'CoordinateX')[1]
        Ycoord = I.getNodeFromName1(FoilGeom_n,'CoordinateY')[1]

        AirfoilGeoms.append( J.createZone(pzn,
                                [Xcoord,Ycoord,Ycoord*0],
                                ['CoordinateX','CoordinateY','CoordinateZ']) )

    if len(AirfoilGeoms) == 1:
        return AirfoilGeoms[0]
    else:
        return AirfoilGeoms


def resetPitch(LiftingLine, ZeroPitchRelativeSpan=0.75, modifyLiftingLine=True):
    '''
    Given an existing LiftingLine object, reset the pitch taking
    as reference the value in attribute **ZeroPitchRelativeSpan**,
    which modifies in-place the **LiftingLine** object (update of ``Twist``
    field) applying a *DeltaTwist* value such that the resulting Twist
    yields ``0`` degrees at **ZeroPitchRelativeSpan**.
    The value of *DeltaTwist* is returned by the function.

    Parameters
    ----------

        LiftingLine : zone
            the lifting line zone

            .. note:: zone **LiftingLine** is modified if **modifyLiftingLine**
                = :py:obj:`True`

        ZeroPitchRelativeSpan : float
            the relative span location where zero twist must be placed.

        modifyLiftingLine : bool
            if :py:obj:`True`, modify the ``Twist`` field of the
            **LiftingLine** accordingly.

    Returns
    -------

        DeltaTwist : float
            Value required to be added to ``Twist`` field in order
            to verify :math:`Twist=0` at the location requested by **ZeroPitchRelativeSpan**
    '''
    r, Twist = J.getVars(LiftingLine,['Span','Twist'])
    DeltaTwist = J.interpolate__(np.array([0.75]), r/r.max(), Twist)
    if modifyLiftingLine: Twist -= DeltaTwist

    return DeltaTwist


def remapLiftingLine(LiftingLine, NewRelativeDistribution,
                     InterpolationLaw='interp1d_linear'):
    '''
    From an existing **LiftingLine**, this function generates a new
    one with user-defined spanwise discretization. If the
    existing **LiftingLine** had fields in ``FlowSolutions``, those are also
    remapped into the new LiftingLine returned by the function.

    .. note:: special nodes named ``.Component#Info``,
        ``.Loads``,``.Kinematics`` are conserved in new rediscretized LiftingLine

    Parameters
    ----------

        LiftingLine : zone
            The original LiftingLine where remapping will be applied from

        NewRelativeDistribution : multiple
            This polymorphic input is used to infer the spanwise
            discretization that new wing surface will use.
            Typical use is:

            >>> np.linspace(MinimumRelativeSpan, MaximumRelativeSpan, NbOfSpanwisePoints)

            For detailed information on possible inputs, please see
            :py:func:`MOLA.InternalShortcuts getDistributionFromHeterogeneousInput__` doc.

        InterpolationLaw : str
            defines the interpolation law used for remapping the LiftingLine.

    Returns
    -------

        NewLiftingLine : zone
            The newly discretized LiftingLine, including remapped fields
            located at ``FlowSolution``
    '''

    LiftingLine, = I.getZones(LiftingLine)

    # List of variables to remap
    VarsNames = C.getVarNames(LiftingLine)[0]

    # Get the list of arrays, including coordinates
    OldVars = [I.getValue(I.getNodeFromName2(LiftingLine,vn)) for vn in VarsNames]
    if 's' in VarsNames:
        OldAbscissa = J.getVars(LiftingLine,['s'])[0]
    else:
        OldAbscissa = W.gets(LiftingLine)

    # Get the newly user-defined abscissa
    _,NewAbscissa,_ = J.getDistributionFromHeterogeneousInput__(NewRelativeDistribution)

    # Perform remapping (interpolation)
    VarsArrays = [J.interpolate__(NewAbscissa,OldAbscissa,OldVar, Law=InterpolationLaw) for OldVar in OldVars]

    # Invoke newly remapped LiftingLine
    NewLiftingLine = J.createZone(LiftingLine[0],VarsArrays,VarsNames)

    # Migrate additional special nodes
    SpecialNodesNames = ['.Component#Info','.Loads','.Kinematics']
    for snm in SpecialNodesNames:
        SpecialNode = I.getNodeFromName1(LiftingLine,snm)
        if SpecialNode: I.addChild(NewLiftingLine,SpecialNode)

    return NewLiftingLine


def makeBladeSurfaceFromLiftingLineAndAirfoilsPolars(LiftingLine, AirfoilsPolars,
        blade_radial_NPts=100,
        blade_root_cellwidth=0.02, blade_tip_cellwidth=1e-3,
        airfoil_NPts_top=99, airfoil_NPts_bottom=99,
        airfoil_LeadingEdge_width_relative2chord=0.001,
        airfoil_TrailingEdge_width_relative2chord=0.01,
        airfoil_LeadingEdge_abscissa=0.49,
        airfoil_stacking_point_relative2chord=0.25,
        ):

    if isinstance(LiftingLine, str):
        LiftingLine = C.convertFile2PyTree(LiftingLine)
    if isinstance(AirfoilsPolars, str):
        AirfoilsPolars = C.convertFile2PyTree(AirfoilsPolars)

    LiftingLine = I.getZones(LiftingLine)[0]
    AirfoilsPolars = I.getZones(AirfoilsPolars)
    Span, = J.getVars(LiftingLine, ['Span'])
    RadialRelativeDiscretization = dict( N=blade_radial_NPts, kind='tanhTwoSides',
                                 FirstCellHeight=blade_root_cellwidth/Span.max(),
                                 LastCellHeight=blade_tip_cellwidth/Span.max() )

    FoilDistribution=[dict(N=airfoil_NPts_bottom,
                           BreakPoint=airfoil_LeadingEdge_abscissa,
                           kind='tanhTwoSides',
                           FirstCellHeight=airfoil_TrailingEdge_width_relative2chord,
                           LastCellHeight=airfoil_LeadingEdge_width_relative2chord),
                      dict(N=airfoil_NPts_top,
                           BreakPoint=1.0,
                           kind='tanhTwoSides',
                           FirstCellHeight=airfoil_LeadingEdge_width_relative2chord,
                           LastCellHeight=airfoil_TrailingEdge_width_relative2chord),]

    LiftingLine = remapLiftingLine(LiftingLine, RadialRelativeDiscretization)
    blade = postLiftingLine2Surface(LiftingLine, AirfoilsPolars,
                                       ChordRelRef = airfoil_stacking_point_relative2chord,
                                       FoilDistribution=FoilDistribution,
                                       ImposeWingCanonicalPosition=True)
    blade[0] = 'blade'

    return blade


def postLiftingLine2Surface(LiftingLine, PyZonePolars, Variables=[],
                            ChordRelRef=0.25, FoilDistribution=None):
    '''
    Post-process a **LiftingLine** element using enhanced **PyZonePolars** data
    in order to build surface fields (like ``Cp``, ``theta``...) from a BEMT, 
    VPM or BodyForce solution.

    Parameters
    ----------

        LiftingLine : zone
            Result of a BEMT, VPM or BodyForce computation.

        PyZonePolars : :py:func:`list` of :py:func:`zone` or :py:class:`str`
            Enhanced **PyZonePolars** for each airfoil, containing also foilwise
            distributions fields (``Cp``, ``theta``, ``delta1``...).

            .. note::
              if input type is a :py:class:`str`, then **PyZonePolars** is
              interpreted as a CGNS file name containing the airfoil polars data

        Variables : :py:class:`list` of :py:class:`str`
            The variables to be built on the newly created surface.
            For example:

            >>> Variables = ['Cp', 'theta']

        ChordRelRef : float
            Reference chordwise used for stacking the sections.

        FoilDistribution : zone or :py:obj:`None`
            As established in :py:func:`MOLA.Wireframe.useEqualNumberOfPointsOrSameDiscretization`


    Returns
    -------

        Surface : zone
            structured surface containing fields at ``FlowSolution``, where the
            variables requested by the user are interpolated.
    '''
    import scipy.interpolate as si
    def _applyInterpolationFunctionToSpanwiseVariableAtLiftingLine(VariableArray, Var, InterpolationLaw):
        '''
        Perform spanwise interpolation of PyZonePolar data
        contained in AllValues dictionary. For this, use
        the general-purpose macro interpolation function:
        Define a general-purpose macro interpolation function.
        '''
        InterpAxis = 2
        if 'interp1d' in InterpolationLaw.lower():
            ScipyLaw = InterpolationLaw.split('_')[1]
            try:
                interp = si.interp1d( Abscissa, VariableArray, axis=InterpAxis,
                                      kind=ScipyLaw, bounds_error=False,
                                      fill_value='extrapolate', assume_sorted=True)
            except ValueError:
                ErrMsg = 'FATAL ERROR during _applyInterpolationFunctionToSpanwiseVariableAtLiftingLine() call with Var=%s\n'%Var
                ErrMsg+= 'Shapes x and y = %d and %d\n'%(len(x),len(y))
                raise ValueError(ErrMsg)

        elif 'pchip' == InterpolationLaw.lower():
            interp = si.PchipInterpolator(Abscissa, VariableArray, axis=InterpAxis, extrapolate=True)

        elif 'akima' == InterpolationLaw.lower():
            interp = si.Akima1DInterpolator(Abscissa, VariableArray, axis=InterpAxis)

        elif 'cubic' == InterpolationLaw.lower():
            interp = si.CubicSpline(Abscissa, VariableArray, axis=InterpAxis, extrapolate=True)

        else:
            raise AttributeError('applyPolarOnLiftingLine(): InterpolationLaw %s not recognized.'%InterpolationLaw)

        # Prepare data and ship them to SurfVars
        Res = interp(s) # Apply spanwise interpolation function
        ResultSlices = [Res[:,isec,isec] for isec in range(len(s))]
        MyArr = np.vstack(ResultSlices)
        MyArr = MyArr.T
        SurfVars[Var][:] = MyArr

    Surfs = []
    for LiftingLine in getLiftingLines(LiftingLine):
        v = J.getAllVars(LiftingLine)
        x,y,z = J.getxyz(LiftingLine)

        # recover the airfoils at each node of the LiftingLine
        PolarInfoNode = getAirfoilsNodeOfLiftingLine(LiftingLine)
        Abscissa = I.getValue(I.getNodeFromName1(PolarInfoNode, 'Abscissa'))
        PyZonePolarNames = I.getValue(I.getNodeFromName1(PolarInfoNode, 'PyZonePolarNames')).split(' ')
        InterpLaw = I.getValue(I.getNodeFromName1(PolarInfoNode, 'InterpolationLaw'))
        order = J._inferOrderFromInterpLawName(InterpLaw)
        AirfoilsGeom = pyZonePolar2AirfoilZone(PyZonePolarNames,PyZonePolars)
        AirfoilsGeom = W.useEqualNumberOfPointsOrSameDiscretization(AirfoilsGeom, FoilDistribution)
        AirfoilsGeom = W.interpolateAirfoils(AirfoilsGeom, Abscissa, v['s'], order=order)

        # position and resize airfoils
        AirfoilFrame = [(1,0,0),(0,1,0),(0,0,1)]
        for i, foil in enumerate(AirfoilsGeom):
            foil_x = J.getx(foil)
            foil_y = J.gety(foil)
            
            # center at stacking point
            foil_x -= ChordRelRef

            # resize using chord
            foil_x *= v['Chord'][i]
            foil_y *= v['Chord'][i]

            # rotate to match the actual position in the lifting-line
            LLframe = [(v['ChordwiseX'][i],v['ChordwiseY'][i],v['ChordwiseZ'][i]),
                    (v['ThickwiseX'][i],v['ThickwiseY'][i],v['ThickwiseZ'][i]),
                    (-v['SpanwiseX'][i],-v['SpanwiseY'][i],-v['SpanwiseZ'][i])]
            T._rotate(foil, (0,0,0), AirfoilFrame, LLframe)

            # center at the actual Lifting-Line node
            T._translate(foil,(x[i],y[i],z[i]))
        
        # stack all sections
        Surf = G.stack(AirfoilsGeom)
        Surf[0] = LiftingLine[0]+'_surf'
        Surfs.append( Surf )



    if len(Variables) == 0:
        if len(Surfs) == 1: return Surfs[0]
        else: return Surfs

    for Surf in Surfs:
        # Invoke the new variables in surface
        SurfVars = J.invokeFieldsDict(Surf,Variables)

        # Build interpolator functions and store them as dict:
        # usage: InterpDict[<PyZonePolarName>](AoA,Mach,Reynolds,[])
        InterpDict = buildPolarsInterpolatorDict(PyZonePolars,InterpFields=Variables)

        AoA, Mach, Reynolds = J.getVars(LiftingLine,["AoA", "Mach", "Reynolds"])


        # Apply polar interpolations and store them in a dict
        AllFoilNPts = C.getNPts(AirfoilsGeom[0])
        RefCurvAbs = W.gets(AirfoilsGeom[0])
        AllValues = {}
        for pzn in PyZonePolarNames:
            InterpolatedSet = InterpDict[pzn](AoA,Mach,Reynolds)
            # NOTA BENE: InterpolatedSet is a list of arrays.
            # Each element is a (FoilNPts x LLNpts) array of the
            # interpolated variable in the same order as contained
            # in list Variables.

            # Adapt the interpolated data if necessary (adaptedSet)
            adaptedSet = []
            for v in range(len(Variables)):
                InterpolatedArray = InterpolatedSet[v]
                IntArrayShape = InterpolatedArray.shape
                print('Variable %s at polar %s has shape: %s'%(Variables[v],pzn,str(IntArrayShape)))
                if len(IntArrayShape)==2:

                    # Compute the PyZonePolar foilwise abscissa
                    # For that, build an auxiliar foil and
                    # compute its abcissa coordinate
                    AuxFoil = pyZonePolar2AirfoilZone(pzn,PyZonePolars)
                    CurrentCurvAbs = W.gets(AuxFoil)

                    interpFoilwise = si.interp1d(CurrentCurvAbs, InterpolatedArray,
                                        kind='cubic', copy=False, axis=0,
                                        assume_sorted=True)

                    NewInterpArray = interpFoilwise(RefCurvAbs)

                    # TODO: Check orientation of foil and data
                    adaptedSet += [NewInterpArray]

                elif len(IntArrayShape)==1:
                    # Integral data. Simply broadcast.
                    print('BROADCAST')
                    adaptedSet += [np.broadcast_to(InterpolatedArray,(AllFoilNPts,IntArrayShape[0]))]
                else:
                    raise ValueError('Interpolated data for variable %s yields not supported shape %s'%(v,str(IntArrayShape)))

            # Store dimensionally-coherent interpolated data
            AllValues[pzn] = adaptedSet


        for v in range(len(Variables)):
            # Build a 3D matrix containing all data.
            # 1st dimension: Foilwise data
            # 2nd dimension: Spanwise data
            # 3rd dimension: slices corresponding to PyZonePolars
            AllValues3D = np.dstack([AllValues[pzn][v] for pzn in PyZonePolarNames])
            _applyInterpolationFunctionToSpanwiseVariableAtLiftingLine(AllValues3D, Variables[v], 'interp1d_linear')

    if len(Surfs) == 1: return Surfs[0]
    else: return Surfs


def addAccurateSectionArea2LiftingLine(LiftingLine, PyZonePolars):
    '''
    Add a field named ``SectionArea`` info the **LiftingLine** using the airfoil's
    geometry contained in **PyZonePolars**.

    Parameters
    ----------

        LiftingLine : zone
            the lifting line zone where ``SectionArea`` field is added

            .. note:: zone **LiftingLine** is modified

        PyZonePolars : :py:class:`list` of zone
            list of special **PyZonePolar** zones
            containing 2D polar information as well as the geometry. Specifically,
            the polars must contain the node ``.Polar#FoilGeometry`` with the
            children nodes ``CoordinateX`` and ``CoordinateY``.

    Returns
    -------

        LiftingLineSurface : zone
            the corresponding surface of the lifting line, that has been
            generated as an auxiliar item during the function call
    '''
    Surf = postLiftingLine2Surface(LiftingLine, PyZonePolars)
    SectionArea, = J.invokeFields(LiftingLine, ['SectionArea'])
    x = J.getx(Surf)
    NbOfSections = x.shape[1]
    for isec in range(NbOfSections):
        Section = GSD.getBoundary(Surf, 'jmin', layer=isec)
        Section = W.closeCurve(Section,NPts4closingGap=3, tol=1e-10)
        SectSurf = G.T3mesher2D(Section, triangulateOnly=1)
        G._getVolumeMap(SectSurf)
        PatchesAreas, = J.getVars(SectSurf, ['vol'],
                                    Container='FlowSolution#Centers')
        CurrentSectionArea = np.sum(PatchesAreas)
        SectionArea[isec] = CurrentSectionArea

    return SectSurf


def plotStructPyZonePolars(PyZonePolars, addiationalQuantities=[],
        filesuffix='', fileformat='svg'):
    '''
    Convenient matplotlib function employed for plotting 2D airfoil's polar
    characteristics.

    It produces a set of figures in requested format.

    Parameters
    ----------

        PyZonePolars : :py:class:`list` of zone
            list of special PyZonePolar zones
            containing 2D polar information as well as the geometry.

        addiationalQuantities : :py:class:`list` of :py:class:`str`
            quantities to be plotted.
            Allowable names are the field names contained in **PyZonePolars**
            ``FlowSolution`` container.

        filesuffix : str
            suffix to append to new figure files produced by the function

        fileformat : str
            requested file format, that must be compatible with matplotlib. Some
            examples are: ``'svg'``, ``'pdf'``, ``'png'``

    Returns
    -------

        None : None
            A series of files (figures and legends) are written
    '''

    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator


    # if __PLOT_STIX_LATEX__:
    #     plt.rc('text', usetex=True)
    #     plt.rc('text.latex', preamble=r'\usepackage[notextcomp]{stix}')
    #     plt.rc('font',family='stix')


    for pzp in I.getZones(PyZonePolars):

        # Declare figures
        fig1, ax1 = plt.subplots(1,1,figsize=(4.75,4.25)) # CL
        fig2, ax2 = plt.subplots(1,1,figsize=(4.75,4.25)) # CL/CD
        fig3, ax3 = plt.subplots(1,1,figsize=(9.0,2.0)) # CL/CD

        FS_n = I.getNodeFromName1(pzp,'FlowSolution')
        Cl = I.getNodeFromName1(FS_n,'Cl')[1]
        Cd = I.getNodeFromName1(FS_n,'Cd')[1]
        CloCd = Cl/Cd
        PolRange_n = I.getNodeFromName1(pzp,'.Polar#Range')
        AoA = I.getNodeFromName1(PolRange_n,'AngleOfAttack')[1]
        Mach = I.getNodeFromName1(PolRange_n,'Mach')[1]
        Reynolds = I.getNodeFromName1(PolRange_n,'Reynolds')[1]
        FoilID = pzp[0]
        nMach = len(Mach)
        Colors = plt.cm.jet(np.linspace(0,1,nMach))
        for i in range(nMach):
            if nMach > 1:
                ax1.plot(AoA,Cl[:,i],color=Colors[i])
                ax2.plot(AoA,CloCd[:,i],color=Colors[i])
            else:
                ax1.plot(AoA,Cl,color=Colors[i])
                ax2.plot(AoA,CloCd,color=Colors[i])
            ax3.plot([],[],color=Colors[i],label='M=%g, Re=%g'%(Mach[i],Reynolds[i]))

        minLocX = AutoMinorLocator()
        ax1.xaxis.set_minor_locator(minLocX)
        minLocY1 = AutoMinorLocator()
        ax1.yaxis.set_minor_locator(minLocY1)

        minLocX2 = AutoMinorLocator()
        ax2.xaxis.set_minor_locator(minLocX2)
        minLocY2 = AutoMinorLocator()
        ax2.yaxis.set_minor_locator(minLocY2)

        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        ax1.set_xlabel(r'$\alpha$ (deg)')
        ax1.set_ylabel('$c_L$')
        ax1.set_title(FoilID)

        ax2.set_xlabel(r'$\alpha$ (deg)')
        ax2.set_ylabel('$c_L/c_D$')
        ax2.set_title(FoilID)

        plt.sca(ax1)
        plt.tight_layout()
        filename = 'PolarsCL_%s%s.%s'%(FoilID,filesuffix,fileformat)
        print('Saving %s ...'%filename)
        plt.savefig(filename)
        print('ok')

        plt.sca(ax2)
        plt.tight_layout()
        filename = 'PolarsEff_%s%s.%s'%(FoilID,filesuffix,fileformat)
        print('Saving %s ...'%filename)
        plt.savefig(filename)
        print('ok')

        plt.sca(ax3)
        ax3.legend(loc='upper left', ncol=4, bbox_to_anchor=(0.00, 1.00),bbox_transform=fig3.transFigure, title=FoilID, frameon=False)
        plt.axis('off')
        filename = 'PolarsLegend_%s%s.%s'%(FoilID,filesuffix,fileformat)
        print('Saving %s ...'%filename)
        plt.savefig(filename)
        print('ok')
        plt.close('all')

        for addQty in addiationalQuantities:
            fig1, ax1 = plt.subplots(1,1,figsize=(4.75,4.25))
            Field, = J.getVars(pzp,[addQty])
            for i in range(nMach):
                if nMach>1:
                    ax1.plot(AoA,Field[:,i],color=Colors[i])
                else: ax1.plot(AoA,Field,color=Colors[i])
            minLocX = AutoMinorLocator()
            ax1.xaxis.set_minor_locator(minLocX)
            minLocY1 = AutoMinorLocator()
            ax1.yaxis.set_minor_locator(minLocY1)

            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

            ax1.set_xlabel(r'$\alpha$ (deg)')
            ax1.set_ylabel(addQty)
            ax1.set_title(FoilID)

            plt.tight_layout()
            filename = 'Polars%s_%s%s.%s'%(addQty,FoilID,filesuffix,fileformat)
            print('Saving %s ...'%filename)
            plt.savefig(filename)
            print('ok')
            plt.close('all')


def setRPM(LiftingLines, newRPM):
    for LiftingLine in I.getZones(LiftingLines):
        if not checkComponentKind(LiftingLine,'LiftingLine'): continue
        Kin_n = I.getNodeFromName1(LiftingLine,'.Kinematics')
        if Kin_n:
            RPM = I.getNodeFromName1(Kin_n,'RPM')

            if RPM:
                RPMvalue = np.atleast_1d(RPM[1])
                RPMvalue[:] = newRPM
            else:
                I.createNode('RPM','DataArray_t',
                             value=np.atleast_1d(np.array(newRPM,dtype=np.float64)),
                             parent=Kin_n)
        else:
            J.set(LiftingLine,'.Kinematics',RPM=np.atleast_1d(np.array(newRPM,dtype=np.float64)))

def setVPMParameters(LiftingLines, **kwargs):
    '''
    This function is a convenient wrap used for setting the ``.VPM#Parameters`` nodes of
    **LiftingLine** object.

    .. note::
        information contained in ``.VPM#Parameters`` is used by the VPM solver and in
        :py:func:`buildVortexParticleSourcesOnLiftingLine`.

    Parameters
    ----------
    
        LiftingLines : PyTree, base, zone or list of zones
            Container with Lifting lines where ``.Kinematics`` node is to be set.

            .. note:: zones contained in **LiftingLines** are modified.

        kwargs : This is an arbitrary number of input arguments for the VPM solver. Those are
            possible parameters:

            ::

                IntegralLaw : :py:class:`str`
                    Gives the Interpolation law of the particle sources on the Lifting Line(s). This
                    is used in :py:func:`buildVortexParticleSourcesOnLiftingLine`.

                NumberOfParticleSources : :py:class:`int`
                    Gives the number of particle sources on the Lifting Line(s) from where particles
                    are shed.

                ParticleDistribution : :py:class:`dict`
                    Python dictionary specifying distribution instructions.
                    Default value produces a uniform distribution of particles provided by a linear
                    interpolation. Accepted keys are:

                    * kind : :py:class:`str`
                        Can be one of:

                        * ``'uniform'``
                            Makes an uniform spacing.

                        * ``'tanhOneSide'``
                            Specifies the ratio of the first segment.

                        * ``'tanhTwoSides'``
                            Specifies the ratio of the first and last segment.

                        * ``'ratio'``
                            Employs a geometrical-growth type of law

                    * FirstSegmentRatio : :py:class:`float`
                        Specifies the size of the first segment as a ratio of the VPM resolution.

                        .. note::
                            only relevant if **kind** is ``'tanhOneSide'`` , ``'tanhTwoSides'``
                            or ``'ratio'``

                    * LastSegmentRatio : :py:class:`float`
                        Specifies the size of the last segment as a ratio of the VPM resolution.

                        .. note::
                            only relevant if **kind** is ``'tanhOneSide'`` or
                            ``'tanhTwoSides'``

                    * growthRatio : :py:class:`float`
                        Geometrical growth rate ratio regarding the VPM resolution.

                        .. note::
                            only relevant if **kind** is ``'ratio'``

                    * Symmetrical : bool
                        If :py:obj:`True`, the particle distribution becomes symmetrical.
                        :py:obj:`False` otherwise.

                CirculationThreshold : :py:class:`float`
                    Maximum circulation error for the iteration process when shedding particles from
                    the Lifting Line(s).

                CirculationRelaxationFactor : :py:class:`float`
                    Relaxation factor used to update the circulation during the iteration process
                    when shedding particles from the Lifting Line(s).

                MaxLiftingLineSubIterations : :py:class:`int`
                    Gives the maximum number of iteration used during the shedding process.
    '''


    for LiftingLine in I.getZones(LiftingLines):
        J.set(LiftingLine, '.VPM#Parameters', **kwargs)

def setKinematicsUsingConstantRotationAndTranslation(LiftingLines, RotationCenter=[0,0,0],
                                  RotationAxis=[1,0,0], RPM=2500.0, RightHandRuleRotation=True,
                                  VelocityTranslation=[0,0,0], TorqueOrigin=[0,0,0]):
    '''
    This function is a convenient wrap used for setting the ``.Kinematics`` node of **LiftingLine** object.

    .. note:: information contained in ``.Kinematics`` node
        is used by :py:func:`moveLiftingLines` and :py:func:`computeKinematicVelocity`
        functions.

    Parameters
    ----------

        LiftingLines : PyTree, base, zone or list of zones
            Container with Lifting lines where ``.Kinematics`` node is to be set

            .. note:: zones contained in **LiftingLines** are modified

        RotationCenter : :py:class:`list` of 3 :py:class:`float`
            Rotation Center of the motion :math:`(x,y,z)` components

        RotationAxis : :py:class:`list` of 3 :py:class:`float`
            Rotation axis of the motion :math:`(x,y,z)` components

        RPM : float
            revolution per minute. Angular speed of the motion.

        RightHandRuleRotation : bool
            if :py:obj:`True`, the motion is done around
            the **RotationAxis** following the right-hand-rule convention.
            :py:obj:`False` otherwise.

        VelocityTranslation : :py:class:`list` of 3 :py:class:`float`
            Constant velocity translation of the LiftingLine
            along the :math:`(x,y,z)` coordinates in :math:`(m/s)`

    '''

    for LiftingLine in I.getZones( LiftingLines ):
        J.set(LiftingLine,'.Kinematics',
                RotationCenter=np.array(RotationCenter,dtype=np.float64),
                RotationAxis=np.array(RotationAxis,dtype=np.float64),
                RPM=np.atleast_1d(np.array(RPM,dtype=np.float64)),
                RightHandRuleRotation=RightHandRuleRotation,
                VelocityTranslation=np.array(VelocityTranslation,dtype=np.float64),
                TorqueOrigin=np.array(TorqueOrigin,dtype=np.float64))

def setConditions(LiftingLines, VelocityFreestream=[0,0,0], Density=1.225,
                  Temperature=288.15):
    '''
    This function is a convenient wrap used for setting the ``.Conditions``
    node of **LiftingLine** object.

    .. note:: information contained in ``.Conditions`` node
        is used for computation of Reynolds and Mach number, as well as other
        required input of methods such that Vortex Particle Method.

    Parameters
    ----------

        LiftingLines : PyTree, base, zone or list of zones
            Container with Lifting lines where ``.Conditions`` node is to be set

            .. note:: zones contained in **LiftingLines** are modified

        VelocityFreestream : :py:class:`list` of 3 :py:class:`float`
            Components :math:`(x,y,z)` of the freestream velocity, in [m/s].

        Density : float
            air density in [kg/m3]

        Temperature : float
            air temperature in [K]

    '''

    for LiftingLine in I.getZones( LiftingLines ):
        J.set(LiftingLine,'.Conditions',
              VelocityFreestream=np.array(VelocityFreestream,dtype=float),
              Density=np.atleast_1d(float(Density)),
              Temperature=np.atleast_1d(float(Temperature)))



def getRotationAxisCenterAndDirFromKinematics(LiftingLine):
    '''

    .. note:: this is a private function

    Extract **RotationAxis**, **RotationCenter** and rotation direction from
    ``.Kinematics`` node.

    Parameters
    ----------

        LiftingLine : zone
            Lifting-line with ``.Kinematics`` node defined
            using :py:func:`setKinematicsUsingConstantRotationAndTranslation`

            .. note:: zone **LiftingLine** is modified
    '''
    Kinematics_n = I.getNodeFromName(LiftingLine,'.Kinematics')
    if not Kinematics_n:
        raise ValueError('missing ".Kinematics" node')


    RotationAxis_n = I.getNodeFromName1(Kinematics_n,'RotationAxis')
    if not RotationAxis_n:
        raise ValueError('missing "RotationAxis" node in ".Kinematics"')
    RotationAxis = I.getValue( RotationAxis_n )

    RotationCenter_n = I.getNodeFromName1(Kinematics_n,'RotationCenter')
    if not RotationCenter_n:
        raise ValueError('missing "RotationCenter" node in ".Kinematics"')
    RotationCenter = I.getValue( RotationCenter_n )

    Dir_n = I.getNodeFromName1(Kinematics_n,'RightHandRuleRotation')
    if not Dir_n:
        raise ValueError('missing "RightHandRuleRotation" node in ".Kinematics"')
    Dir = I.getValue( Dir_n )
    if not Dir: Dir = -1


    return RotationAxis, RotationCenter, Dir


def computeKinematicVelocity(t):
    '''
    Compute or update ``VelocityKinematicX``, ``VelocityKinematicY`` and
    ``VelocityKinematicZ`` fields of LiftingLines provided to function using
    information contained in ``.Kinematics`` node attached to each LiftingLine.

    Parameters
    ----------

        t : Pytree, base, list of zones, zone
            container with LiftingLines

            .. note:: LiftingLines contained in **t** are modified.
    '''

    RequiredFieldNames = ['VelocityKinematicX',
                          'VelocityKinematicY',
                          'VelocityKinematicZ',]

    LiftingLines = [z for z in I.getZones(t) if checkComponentKind(z,'LiftingLine')]
    for LiftingLine in LiftingLines:
        Kinematics = J.get(LiftingLine,'.Kinematics')
        VelocityTranslation = Kinematics['VelocityTranslation']
        RotationCenter = Kinematics['RotationCenter']
        RotationAxis = Kinematics['RotationAxis']
        RPM = Kinematics['RPM']
        Dir = 1 if Kinematics['RightHandRuleRotation'] else -1
        Omega = RPM*np.pi/30.
        x,y,z = J.getxyz(LiftingLine)
        ExistingFieldNames = C.getVarNames(LiftingLine,excludeXYZ=True)[0]
        v = dict()
        for fieldname in RequiredFieldNames:
            if fieldname in ExistingFieldNames:
                v[fieldname] = J.getVars(LiftingLine,[fieldname])[0]
            else:
                v[fieldname] = J.invokeFields(LiftingLine,[fieldname])[0]

        NPts = len(x)
        # TODO vectorize this
        for i in range(NPts):
            rvec = np.array([x[i] - RotationCenter[0],
                             y[i] - RotationCenter[1],
                             z[i] - RotationCenter[2]],dtype=np.float64)

            VelocityKinematic = np.cross( Dir * Omega * RotationAxis, rvec) + VelocityTranslation

            v['VelocityKinematicX'][i] = VelocityKinematic[0]
            v['VelocityKinematicY'][i] = VelocityKinematic[1]
            v['VelocityKinematicZ'][i] = VelocityKinematic[2]

def assembleAndProjectVelocities(t):
    '''
    This function updates a series of veolicity (and other fields) of
    LiftingLines provided a given kinematic and flow conditions.

    The new or updated fields are the following :

    * ``VelocityX`` ``VelocityY`` ``VelocityZ``
        Three components of the VelocityInduced + VelocityFreestream

    * ``VelocityAxial``
        Relative velocity in -RotationAxis direction

    * ``VelocityTangential``
        Relative velocity in the rotation plane direction

    * ``VelocityNormal2D``
        This is the normal-wise (in ``nx`` ``ny`` ``nz`` direction)
        of the 2D velocity

    * ``VelocityTangential2D``
        This is the tangential (in ``bx`` ``by`` ``bz`` direction)
        of the 2D velocity

    * ``phiRad``
        Angle of the flow with respect to rotation plane as
        ``np.arctan2( VelocityNormal2D, VelocityTangential2D )``

    * ``AoA``
        Local angle-of-attack of the blade section

    * ``VelocityMagnitudeLocal``
        Magnitude of the local velocity neglecting the radial contribution

    * ``Mach``
        Mach number neglecting the radial contribution

    * ``Reynolds``
        Reynolds number neglecting the radial contribution

    .. attention:: please note that this function requires the LiftingLine to
        have the fields: ``VelocityKinematicX``, ``VelocityKinematicY``, ``VelocityKinematicZ``,
        ``VelocityInducedX``,   ``VelocityInducedY``,   ``VelocityInducedZ``,
        if they are not found, then they are created (with zero values).

    Parameters
    ----------

        t : PyTree, base, list of zones, zone
            container with LiftingLines.

            .. note:: Lifting-lines contained in **t** are modified.

    '''
    RequiredFieldNames = ['VelocityKinematicX',
                          'VelocityKinematicY',
                          'VelocityKinematicZ',
                          'VelocityInducedX',
                          'VelocityInducedY',
                          'VelocityInducedZ',
                          'VelocityPerturbationX',
                          'VelocityPerturbationY',
                          'VelocityPerturbationZ',
                          'VelocityX',
                          'VelocityY',
                          'VelocityZ',
                          'Velocity2DX',
                          'Velocity2DY',
                          'Velocity2DZ',
                          'VelocityAxial',
                          'VelocityTangential',
                          'VelocityMagnitudeLocal',
                          'VelocityChordwise',
                          'VelocityThickwise',
                          'Chord','ChordVirtualWithSweep',
                          'Twist','AoA','phiRad','Mach','Reynolds',
                          'ChordwiseX','ChordwiseY','ChordwiseZ',
                          'SpanwiseX','SpanwiseY','SpanwiseZ',
                          'ThickwiseX','ThickwiseY','ThickwiseZ',
                          'TangentialX','TangentialY','TangentialZ',
                          'SweepAngleDeg', 'DihedralAngleDeg'
                          ]

    LiftingLines = [z for z in I.getZones(t) if checkComponentKind(z,'LiftingLine')]
    for LiftingLine in LiftingLines:
        #Collecting prescribed polar corrections
        Correc_n = I.getNodeFromName(LiftingLine,'Corrections3D')
        SweepCorrection = I.getValue(I.getNodeFromName(Correc_n,'Sweep'))
        DihedralCorrection = I.getValue(I.getNodeFromName(Correc_n,'Dihedral'))

        Conditions = J.get(LiftingLine,'.Conditions')
        Temperature = Conditions['Temperature']
        Density = Conditions['Density']
        VelocityFreestream = Conditions['VelocityFreestream']
        Kinematics = J.get(LiftingLine,'.Kinematics')
        RotationAxis = Kinematics['RotationAxis']
        dir = 1 if Kinematics['RightHandRuleRotation'] else -1
        Mu=Mus*((Temperature/Ts)**0.5)*((1.+Cs/Ts)/(1.+Cs/Temperature))
        SoundSpeed = np.sqrt(Gamma * Rgp * Temperature)
        
        ExistingFieldNames = C.getVarNames(LiftingLine,excludeXYZ=True)[0]
        v = dict()
        for fieldname in RequiredFieldNames:
            if fieldname in ExistingFieldNames:
                v[fieldname] = J.getVars(LiftingLine,[fieldname])[0]
            else:
                v[fieldname] = J.invokeFields(LiftingLine,[fieldname])[0]

        VelocityKinematic = np.vstack([v['VelocityKinematic'+i] for i in 'XYZ'])
        VelocityInduced = np.vstack([v['VelocityInduced'+i] for i in 'XYZ'])
        VelocityPerturbation = np.vstack([v['VelocityPerturbation'+i] for i in 'XYZ'])
        TangentialDirection = np.vstack([v['Tangential'+i] for i in 'XYZ'])

        VelocityRelative = (VelocityInduced.T + VelocityPerturbation.T + VelocityFreestream - VelocityKinematic.T).T
        v['VelocityX'][:] = VelocityInduced[0,:] + VelocityPerturbation[0,:] + VelocityFreestream[0]
        v['VelocityY'][:] = VelocityInduced[1,:] + VelocityPerturbation[1,:] + VelocityFreestream[1]
        v['VelocityZ'][:] = VelocityInduced[2,:] + VelocityPerturbation[2,:] + VelocityFreestream[2]
        v['VelocityAxial'][:] = ( VelocityRelative.T.dot(-RotationAxis) ).T
        v['VelocityTangential'][:] = np.diag(VelocityRelative.T.dot(TangentialDirection))


        ChordwiseDirection = np.vstack([v['Chordwise'+i] for i in 'XYZ'])
        ThickwiseDirection = np.vstack([v['Thickwise'+i] for i in 'XYZ'])
        v['VelocityChordwise'][:] = Vchord = Vchord_Base = np.diag( VelocityRelative.T.dot(ChordwiseDirection) )
        v['VelocityThickwise'][:] = Vthick = Vthick_Base = np.diag( VelocityRelative.T.dot(ThickwiseDirection) )
        # note the absence of radial velocity contribution to 2D flow (Spanwise component is cut)
        V2D = np.vstack((Vchord * ChordwiseDirection[0,:] + Vthick * ThickwiseDirection[0,:],
                         Vchord * ChordwiseDirection[1,:] + Vthick * ThickwiseDirection[1,:],
                         Vchord * ChordwiseDirection[2,:] + Vthick * ThickwiseDirection[2,:]))

        v['Velocity2DX'][:] = V2D[0, :] #Used for VPM
        v['Velocity2DY'][:] = V2D[1, :]
        v['Velocity2DZ'][:] = V2D[2, :]


        v['AoA'][:] = np.rad2deg( np.arctan2(Vthick,Vchord) )
        # NOTE the absence of radial velocity contribution to Velocity Magnitude, Mach and Reynolds

        if SweepCorrection:
            v['VelocityChordwise'][:] = Vchord = Vchord_Base * np.cos(np.deg2rad(v['SweepAngleDeg']))
            
        if DihedralCorrection:
            v['VelocityThickwise'][:] = Vthick = Vthick_Base * np.cos(np.deg2rad(v['DihedralAngleDeg']))

        # Updating the Angle of Attack considering the new velocity components.    
        v['AoA'][:] = np.rad2deg( np.arctan2(Vthick,Vchord) )

        v['VelocityMagnitudeLocal'][:] = W = np.sqrt( Vchord**2 + Vthick**2 )
        v['Mach'][:] = W / SoundSpeed
        v['Reynolds'][:] = Density[0] * W * v['Chord'] / Mu


        V2Da = ( V2D.T.dot(-RotationAxis) ).T
        V2Dt = dir * np.diag( V2D.T.dot(TangentialDirection))
        v['phiRad'][:] = np.arctan2( V2Da, V2Dt ) #Used for Tip-Loss corrections



def moveLiftingLines(t, TimeStep):
    '''
    Move the lifting lines following their ``.Kinematics`` law.

    It also updates the local frame quantities of the lifting lines and
    updates the kinematic velocity.

    Parameters
    ----------

        t : PyTree, base, list of zones, zone
            container with LiftingLines.

            .. note:: Lifting-lines contained in **t** are modified.

        TimeStep : float
            time step for the movement of the lifting-lines in [s]
    '''
    LiftingLines = [z for z in I.getZones(t) if checkComponentKind(z,'LiftingLine')]
    for LiftingLine in LiftingLines:
        Kinematics = J.get(LiftingLine,'.Kinematics')
        VelocityTranslation = Kinematics['VelocityTranslation']
        RotationCenter = Kinematics['RotationCenter']
        RotationAxis = Kinematics['RotationAxis']
        RightHandRuleRotation = Kinematics['RightHandRuleRotation']
        RPM = Kinematics['RPM']
        Omega = RPM * np.pi / 30.
        Dpsi = np.rad2deg( Omega * TimeStep )
        try: Dpsi = Dpsi[0]
        except: pass
        if not RightHandRuleRotation: Dpsi *= -1

        if Dpsi: T._rotate(LiftingLine, RotationCenter, RotationAxis, Dpsi,
                           vectors=NamesOfChordSpanThickwiseFrame)

        if VelocityTranslation.any():
            T._translate(LiftingLine, TimeStep*VelocityTranslation)
            RotationCenter += TimeStep*VelocityTranslation

        computeKinematicVelocity(LiftingLine)

def moveObject(t, TimeStep):
    '''
    Move the lifting lines following their ``.Kinematics`` law.

    It also updates the local frame quantities of the lifting lines and
    updates the kinematic velocity.

    Parameters
    ----------

        t : PyTree, base, list of zones, zone
            container with LiftingLines.

            .. note:: Lifting-lines contained in **t** are modified.

        TimeStep : float
            time step for the movement of the lifting-lines in [s]
    '''

    for LiftingLine in I.getZones(t):
        Kinematics = J.get(LiftingLine,'.Kinematics')
        VelocityTranslation = Kinematics['VelocityTranslation']
        RotationCenter = Kinematics['RotationCenter']
        RotationAxis = Kinematics['RotationAxis']
        RightHandRuleRotation = Kinematics['RightHandRuleRotation']
        RPM = Kinematics['RPM']
        Omega = RPM * np.pi / 30.
        Dpsi = np.rad2deg( Omega * TimeStep )
        try: Dpsi = Dpsi[0]
        except: pass
        if not RightHandRuleRotation: Dpsi *= -1

        if Dpsi: T._rotate(LiftingLine, RotationCenter, RotationAxis, Dpsi,
                           vectors=NamesOfChordSpanThickwiseFrame)

        if VelocityTranslation.any():
            T._translate(LiftingLine, TimeStep*VelocityTranslation)
            RotationCenter += TimeStep*VelocityTranslation

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
    import Converter.Mpi as Cmpi
    import Post.Mpi as Pmpi
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


def computeGeneralLoadsOfLiftingLine(t, NBlades=1.0, UnsteadyData={},
        UnsteadyDataIndependentAbscissa='IterationNumber',
        TipLossFactorOptions={}):
    '''
    This function is used to compute local and integral arrays of a lifting line
    with general orientation and shape (including sweep and dihedral).

    .. important:: the flow's contribution to the efforts in the tangential
        direction of the lifting-line is neglected. This means that radial
        contribution on rotors is neglected.

    Parameters
    ----------

        t : PyTree, base, zone, list of zones
            container with Lifting Line zones.
            Each LiftingLine must contain the minimum required fields:

            * ``phiRad``
                Local angle of the flow in radians

            * ``Cl`` ``Cd`` ``Cm``
                Local aerodynamic coefficients of the lifting line sections

            * ``Chord``
                Local chord of the sections

            * ``VelocityMagnitudeLocal``
                velocity magnitude employed for computing the fluxes, moments
                and local bound circulation.

            * ``s``
                curvilinear abscissa

            * ``Span``
                local span, which is the cylindric distance from **RotationAxis**
                (information contained in ``.Kinematics`` node) to each section

            New fields are created as a result of this function call:

            * ``ForceX`` ``ForceY`` ``ForceZ``
                local linear forces at each lifting line's section
                in [N/m]. Each component :math:`(x,y,z)` corresponds to absolute
                coordinate frame (same as ``GridCoordinates``)

            * ``ForceAxial`` ``ForceTangential``
                local linear forces projected onto axial and tangential
                directions. ``ForceAxial`` contributes to Thrust. ``ForceTangential`` contributes to Torque.
                They have dimensions of [N/m]

            * ``TorqueAtAirfoilX`` ``TorqueAtAirfoilY`` ``TorqueAtAirfoilZ``
                local linear moments in :math:`(x,y,z)` frame applied at airfoil sections.
                Dimensions are [N] the moments are applied on 1/4 chord (at LiftingLine's nodes)

            * ``TorqueAtRotationAxisX`` ``TorqueAtRotationAxisY`` ``TorqueAtRotationAxisZ``
                local linear moments in :math:`(x,y,z)` frame applied at
                rotation center of the blade. Dimensions are [N]

            * ``LiftX`` ``LiftY`` ``LiftZ`` ``LiftAxial`` ``LiftTangential``
                Respectively linear Lift contribution following
                the directions :math:`(x,y,z)` axial, tangential [N/m]

            * ``DragX`` ``DragY`` ``DragZ`` ``DragAxial`` ``DragTangential``
                linear Drag contribution following
                the directions :math:`(x,y,z)` axial, tangential [N/m]

            * ``Gamma``
                circulation magnitude of the blade section following the
                Kutta-Joukowski theorem

            * ``GammaX`` ``GammaY`` ``GammaZ``
                circulation vector of the blade section following the
                Kutta-Joukowski theorem

            .. note::
                LiftingLine zones contained in **t** are modified

        NBlades : float
            Multiplication factor of integral arrays

        TipLossFactorOptions : dict
            Use a tip-loss factor function to the aerodynamic coefficients.
            This :py:class:`dict` defines a pair of keyword-arguments of the 
            function :py:func:`applyTipLossFactorToBladeEfforts`.
    '''
    import scipy.integrate as sint

    MinimumRequiredFields = ('Cl','Cd','Cm','Chord','ChordVirtualWithSweep',
        'VelocityMagnitudeLocal','s','Span',
        'AoA',
        'ChordwiseX', 'ChordwiseY', 'ChordwiseZ',
        'ThickwiseX', 'ThickwiseY', 'ThickwiseZ',
        'SpanwiseX', 'SpanwiseY', 'SpanwiseZ',
        'TangentialX', 'TangentialY', 'TangentialZ','SweepAngleDeg', 'DihedralAngleDeg')
    NewFields = (
        'ForceX','ForceY','ForceZ', # previously 'ForceX','ForceY','ForceZ'
        'ForceAxial','ForceTangential', # previously 'ForceAxial','ForceTangential'
        'TorqueAtAirfoilX','TorqueAtAirfoilY','TorqueAtAirfoilZ', # previously 'mx', 'my', 'mz'
        'TorqueAtRotationAxisX','TorqueAtRotationAxisY','TorqueAtRotationAxisZ', # previously 'm0x', 'm0y', 'm0z'
        'TorqueAtOriginCenterX','TorqueAtOriginCenterY','TorqueAtOriginCenterZ',
        'LiftX','LiftY','LiftZ', # previously 'Lx','Ly','Lz'
        'LiftAxial','LiftTangential', # previously 'La', 'Lt'
        'LiftThickwise', 'LiftChordwise', # previsouly 'Lthickwise', 'Lchordwise'
        'DragX','DragY','DragZ', # previously 'Dx', 'Dy', 'Dz'
        'DragAxial','DragTangential', # previously 'Da', 'Dt'
        'DragThickwise', 'DragChordwise', # 'Dthickwise', 'Dchordwise',
        'Gamma',
        'GammaX','GammaY','GammaZ',
        )

    LiftingLines = [z for z in I.getZones(t) if checkComponentKind(z,'LiftingLine')]
    NumberOfLiftingLines = len(LiftingLines)
    AllIntegralData = {}
    for LiftingLine in LiftingLines:
        Correc_n = I.getNodeFromName(LiftingLine,'Corrections3D')
        SweepCorrection = I.getValue(I.getNodeFromName(Correc_n,'Sweep'))
        DihedralCorrection = I.getValue(I.getNodeFromName(Correc_n,'Dihedral'))

        Kinematics = J.get(LiftingLine,'.Kinematics')
        RotationCenter = Kinematics['RotationCenter']
        TorqueOrigin = Kinematics['TorqueOrigin']
        RotationAxis = Kinematics['RotationAxis']
        RightHandRuleRotation = Kinematics['RightHandRuleRotation']
        RPM = Kinematics['RPM']
        Conditions = J.get(LiftingLine,'.Conditions')
        Temperature = Conditions['Temperature']
        Density = Conditions['Density']
        VelocityFreestream = Conditions['VelocityFreestream']

        dir = 1 if RightHandRuleRotation else -1

        # Construct general container v for storing pointers of fields
        FlowSolution_n = I.getNodeFromName1(LiftingLine,'FlowSolution')
        v = {}
        for fn in MinimumRequiredFields:
            try: v[fn] = I.getNodeFromName1(FlowSolution_n,fn)[1]
            except:
                raise ValueError('need %s in FlowSolution of LiftingLine'%fn)

        for fn in NewFields:
            try: v[fn] = I.getNodeFromName1(FlowSolution_n,fn)[1]
            except: v[fn] = J.invokeFields(LiftingLine,[fn])[0]

        x,y,z = J.getxyz(LiftingLine)
        xyz = np.vstack((x,y,z))
        rx = x - RotationCenter[0]
        ry = y - RotationCenter[1]
        rz = z - RotationCenter[2]

        r2x = x - TorqueOrigin[0]
        r2y = y - TorqueOrigin[1]
        r2z = z - TorqueOrigin[2]


        # ----------------------- COMPUTE LINEAR FORCES ----------------------- #
        FluxC = 0.5*Density*v['VelocityMagnitudeLocal']**2*v['Chord']

        if SweepCorrection:
            FluxC = 0.5*Density*v['VelocityMagnitudeLocal']**2*v['ChordVirtualWithSweep']
            SweepCorrectionCoefficient = np.cos(np.deg2rad(v['SweepAngleDeg']))
            Drag = FluxC*v['Cd']*SweepCorrectionCoefficient
        else: Drag= FluxC*v['Cd']

        if DihedralCorrection:
            DihedralCorrectionCoefficient = np.cos(np.deg2rad(v['DihedralAngleDeg']))
            Lift = FluxC*v['Cl']*DihedralCorrectionCoefficient
        else: Lift = FluxC*v['Cl']

        if TipLossFactorOptions:
            applyTipLossFactorToBladeEfforts(LiftingLine, **TipLossFactorOptions)
        Lift = FluxC*v['Cl']
        Drag = FluxC*v['Cd']

        v['LiftChordwise'][:] = -Lift*np.sin(np.deg2rad(v['AoA']))
        v['LiftThickwise'][:] =  Lift*np.cos(np.deg2rad(v['AoA']))

        v['DragChordwise'][:] = Drag*np.cos(np.deg2rad(v['AoA']))
        v['DragThickwise'][:] = Drag*np.sin(np.deg2rad(v['AoA']))

        v['LiftX'][:] = v['LiftChordwise']*v['ChordwiseX'] + v['LiftThickwise']*v['ThickwiseX']
        v['LiftY'][:] = v['LiftChordwise']*v['ChordwiseY'] + v['LiftThickwise']*v['ThickwiseY']
        v['LiftZ'][:] = v['LiftChordwise']*v['ChordwiseZ'] + v['LiftThickwise']*v['ThickwiseZ']
        v['DragX'][:] = v['DragChordwise']*v['ChordwiseX'] + v['DragThickwise']*v['ThickwiseX']
        v['DragY'][:] = v['DragChordwise']*v['ChordwiseY'] + v['DragThickwise']*v['ThickwiseY']
        v['DragZ'][:] = v['DragChordwise']*v['ChordwiseZ'] + v['DragThickwise']*v['ThickwiseZ']


        v['LiftAxial'][:] = v['LiftX']*RotationAxis[0] + \
                            v['LiftY']*RotationAxis[1] + \
                            v['LiftZ']*RotationAxis[2]

        v['DragAxial'][:] = v['DragX']*RotationAxis[0] + \
                            v['DragY']*RotationAxis[1] + \
                            v['DragZ']*RotationAxis[2]

        v['LiftTangential'][:] = v['LiftX']*v['TangentialX'] + \
                                 v['LiftY']*v['TangentialY'] + \
                                 v['LiftZ']*v['TangentialZ']

        v['DragTangential'][:] = v['DragX']*v['TangentialX'] + \
                                 v['DragY']*v['TangentialY'] + \
                                 v['DragZ']*v['TangentialZ']

        v['ForceAxial'][:] = v['LiftAxial'] + v['DragAxial']
        v['ForceTangential'][:] = v['LiftTangential'] + v['DragTangential']

        v['ForceX'][:] = v['LiftX'] + v['DragX']
        v['ForceY'][:] = v['LiftY'] + v['DragY']
        v['ForceZ'][:] = v['LiftZ'] + v['DragZ']

        # ----------------------- COMPUTE LINEAR TORQUE ----------------------- #
        FluxM = FluxC*v['Chord']*v['Cm']

        if SweepCorrection:
            FluxM = FluxC*v['ChordVirtualWithSweep']*v['Cm']*SweepCorrectionCoefficient

        if DihedralCorrection:
            FluxM = FluxM*DihedralCorrectionCoefficient
        
        v['TorqueAtAirfoilX'][:] = dir * FluxM * v['SpanwiseX']
        v['TorqueAtAirfoilY'][:] = dir * FluxM * v['SpanwiseY']
        v['TorqueAtAirfoilZ'][:] = dir * FluxM * v['SpanwiseZ']
        v['TorqueAtRotationAxisX'][:] = v['TorqueAtAirfoilX'] + ry*v['ForceZ'] - rz*v['ForceY']
        v['TorqueAtRotationAxisY'][:] = v['TorqueAtAirfoilY'] + rz*v['ForceX'] - rx*v['ForceZ']
        v['TorqueAtRotationAxisZ'][:] = v['TorqueAtAirfoilZ'] + rx*v['ForceY'] - ry*v['ForceX']
        v['TorqueAtOriginCenterX'][:] = v['TorqueAtAirfoilX'] + r2y*v['ForceZ'] - r2z*v['ForceY']
        v['TorqueAtOriginCenterY'][:] = v['TorqueAtAirfoilY'] + r2z*v['ForceX'] - r2x*v['ForceZ']
        v['TorqueAtOriginCenterZ'][:] = v['TorqueAtAirfoilZ'] + r2x*v['ForceY'] - r2y*v['ForceX']

        # Compute linear bound circulation using Kutta-Joukowski
        # theorem:  Lift = Density * ( Velocity x Gamma )
        w = v['VelocityMagnitudeLocal']
        FluxKJ = Lift/Density
        Flowing = abs(w)>0
        FluxKJ[Flowing] /= w[Flowing]
        FluxKJ[~Flowing] = 0.
        v['GammaX'][:] = dir * FluxKJ * v['SpanwiseX']
        v['GammaY'][:] = dir * FluxKJ * v['SpanwiseY']
        v['GammaZ'][:] = dir * FluxKJ * v['SpanwiseZ']
        v['Gamma'][:] = FluxKJ
        # ------------------------- INTEGRAL LOADS ------------------------- #
        length = norm(np.sum(np.abs(np.diff(xyz,axis=1)),axis=1)) # faster than D.getLength
        DimensionalAbscissa = length * v['s'] # TODO check if v['s'] is updated!

        # Integrate linear axial force <fa> to get Thrust
        FA = Thrust = sint.simps(v['ForceAxial'], DimensionalAbscissa)
        FT =          sint.simps(v['ForceTangential'], DimensionalAbscissa)
        FX =          sint.simps(v['ForceX'], DimensionalAbscissa)
        FY =          sint.simps(v['ForceY'], DimensionalAbscissa)
        FZ =          sint.simps(v['ForceZ'], DimensionalAbscissa)
        MX =         -sint.simps(v['TorqueAtRotationAxisX'], DimensionalAbscissa)
        MY =         -sint.simps(v['TorqueAtRotationAxisY'], DimensionalAbscissa)
        MZ =         -sint.simps(v['TorqueAtRotationAxisZ'], DimensionalAbscissa)
        MoX =         sint.simps(v['TorqueAtOriginCenterX'], DimensionalAbscissa)
        MoY =         sint.simps(v['TorqueAtOriginCenterY'], DimensionalAbscissa)
        MoZ =         sint.simps(v['TorqueAtOriginCenterZ'], DimensionalAbscissa)

        # # Integrate tangential moment <ft>*Span to get Power
        # Torque = sint.simps(v['ForceTangential']*v['Span'],DimensionalAbscissa) # equivalent
        Torque = MX*RotationAxis[0]+MY*RotationAxis[1]+MZ*RotationAxis[2]
        Power  = dir*(RPM*np.pi/30.)*Torque


        # Store computed integral Loads
        Loads = dict(Thrust=NBlades*Thrust,Power=NBlades*Power,
                     Torque=NBlades*Torque, ForceTangential=NBlades*FT,
                     ForceX=NBlades*FX, ForceY=NBlades*FY, ForceZ=NBlades*FZ,
                     TorqueX=NBlades*MX, TorqueY=NBlades*MY, TorqueZ=NBlades*MZ,
                     TorqueAtOriginCenterX=NBlades*MoX,
                     TorqueAtOriginCenterY=NBlades*MoY,
                     TorqueAtOriginCenterZ=NBlades*MoZ)

        IntegralData = J.set(LiftingLine,'.Loads', **Loads)

        if UnsteadyData:
            IntegralData.update(UnsteadyData)
            try:
                IndependentAbscissa = UnsteadyData[UnsteadyDataIndependentAbscissa]
            except KeyError:
                raise KeyError(FAIL+'UnsteadyData dict must contain key "%s"'%UnsteadyDataIndependentAbscissa+ENDC)

            UnsteadyLoads = J.get(LiftingLine,'.UnsteadyLoads')

            if UnsteadyLoads:
                try:
                    PreviousIndependentAbscissa = UnsteadyLoads[UnsteadyDataIndependentAbscissa]
                except KeyError:
                    raise KeyError(FAIL+'UnsteadyLoads must contain"%s"'%UnsteadyDataIndependentAbscissa+ENDC)
                AppendFrom = PreviousIndependentAbscissa > (IndependentAbscissa + 1e-12)
                try: FirstIndex2Update = np.where(AppendFrom)[0][0]
                except IndexError: FirstIndex2Update = len(PreviousIndependentAbscissa)
                for k in IntegralData:
                    PreviousArray = UnsteadyLoads[k][:FirstIndex2Update]
                    AppendArray = IntegralData[k]
                    UnsteadyLoads[k] = np.hstack((PreviousArray, AppendArray))

            else:
                UnsteadyLoads.update(IntegralData)
                UnsteadyLoads.update(UnsteadyData)

            UnsteadyLoads = J.set(LiftingLine,'.UnsteadyLoads',**UnsteadyLoads)

        if NumberOfLiftingLines == 1:  return IntegralData

        AllIntegralData[LiftingLine[0]] = IntegralData

    TotalIntegralData = dict()
    for LiftingLineLoad in AllIntegralData:
        for LoadName in AllIntegralData[LiftingLineLoad]:
            try:
                TotalIntegralData[LoadName] += AllIntegralData[LiftingLineLoad][LoadName]
            except KeyError:
                TotalIntegralData[LoadName] = np.copy(AllIntegralData[LiftingLineLoad][LoadName])

    AllIntegralData['Total'] = TotalIntegralData

    return AllIntegralData

def applyTipLossFactorToBladeEfforts(LiftingLine, kind='Prandtl', NumberOfBlades=3,
        g1_parameter='default', g2_parameter=20.0, composite_factor=0.8):
    '''
    Apply a tip loss factor function to :math:`C_l` and :math:`C_d` quantities 
    of a LiftingLine. 

    .. note:: this function is optionally called used in the context of :py:func:`computeGeneralLoadsOfLiftingLine`

    Parameters
    ----------

        LiftingLine : zone
            lifting-line where the tip-loss factor is being applied

        kind : str
            Kind of tip-loss factor to be used. Can be one of:

            * ``'Shen'``
                Use Shen's function

            * ``'Prandtl'``
                Use Prandtl's function

            * ``'Pantel'``
                Use Pantel's composite function
        
        NumberOfBlades : int
            Number of blades used for modeling the tip loss factor

        g1_parameter : :py:class:`str` or :py:class:`float`
            parameter :math:`g` of tip loss factor function. If ``'default'``,
            and **kind** = ``'Shen'``, then defaults to a kinematic correlation;
            or if **kind** = ``'Pantel'`` then defaults to ``0.75``.

        g2_parameter : float
            second parameter :math:`g` of the second composite function of 
            Pantel's function.

        composite_factor : float
            spanwise portion of application of composite function of Pantel.
    '''

    
    # TODO reuse a call to PropellerAnalysis.TipLossFactor() for obtaining F ?

    required_fields = ['Cl', 'Cd', 'Span', 'phiRad']
    new_fields = ['F', 'Cl_without_F', 'Cd_without_F']
    FlowSolution_n = I.getNodeFromName1(LiftingLine, 'FlowSolution')
    v = {}
    for fn in required_fields: v[fn] = I.getNodeFromName1(FlowSolution_n,fn)[1]
    for fn in new_fields:
        try: v[fn] = I.getNodeFromName1(FlowSolution_n,fn)[1]
        except: v[fn] = J.invokeFields(LiftingLine,[fn])[0]    
    span = v['Span'].max()

    v['Cl_without_F'][:] = v['Cl']
    v['Cd_without_F'][:] = v['Cd']

    Kinematics = J.get(LiftingLine,'.Kinematics')
    omega = Kinematics['RPM'] * np.pi / 30.0
    
    if kind == 'Prandtl':
        F = 2/np.pi*np.arccos(np.exp(  -NumberOfBlades*(span-v['Span'])/  \
                                     (2*v['Span']*np.sin(v['phiRad']))))

    elif kind == 'Pantel':
        if g1_parameter == 'default':
            g1 = 0.75
        else:
            g1 = g1_parameter
        g2 = g2_parameter
        fix1 = span/composite_factor
        fix2 = span
        F1 = 2/np.pi*np.arccos(np.exp(-g1*NumberOfBlades*(fix1-v['Span'])/  \
                                     (2*v['Span']*np.sin(v['phiRad']))))
        F2 = 2/np.pi*np.arccos(np.exp(-g2*NumberOfBlades*(fix2-v['Span'])/  \
                                     (2*v['Span']*np.sin(v['phiRad']))))
        F = F1 * F2

    elif kind == 'Shen':
        if g1_parameter == 'default':
            Conditions = J.get(LiftingLine,'.Conditions')
            VelocityFreestream = Conditions['VelocityFreestream']
            U = np.linalg.norm(VelocityFreestream)
            g = np.exp(-0.125*(NumberOfBlades*span*omega/U-21))+0.1
        else:
            g = g1_parameter
        F = 2/np.pi*np.arccos(np.exp( -g*NumberOfBlades*(span-v['Span'])/  \
                                     (2*v['Span']*np.sin(v['phiRad']))))

    else:
        raise AttributeError('TipLossFactor kind %s not recognized'%kind)

    v['F'][:] = F
    v['Cl'][:] = F*v['Cl']
    v['Cd'][:] = F*v['Cd']

def _makeFormat__(current, other):
    # current and other are axes
    def format_coord(x, y):
        # x, y are data coordinates
        # convert to display coords
        display_coord = current.transData.transform((x,y))
        inv = other.transData.inverted()
        # convert back to data coords with respect to ax
        ax_coord = inv.transform(display_coord)
        coords = [ax_coord, (x, y)]
        return ('Left: {:<40}    Right: {:<}'
                .format(*['({:.3f}, {:.3f})'.format(x, y) for x,y in coords]))
    return format_coord

def _plotChordAndTwist(LiftingLines, savefigname=None):
    '''
    Convenient function for interactively plot chord and twist
    geometrical laws of a list of Lifting Lines
    '''

    # TODO remove "_"

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    Markers = ('o','+','s','.','v','^','x')
    for LiftingLine,m in zip(LiftingLines,Markers):
        r, Chord, Twist =  J.getVars(LiftingLine,['Span','Chord','Twist'])
        ax1.plot(r/r.max(),Chord,'-',marker=m,mfc='None',label=LiftingLine[0])
        ax2.plot(r/r.max(),Twist,'--',marker=m,mfc='None')

    ax1.set_xlabel('$r/R$')
    ax1.set_ylabel('Chord (m) (-)')
    ax2.set_ylabel('Twist (deg) (- - -)')

    ax2.format_coord = _makeFormat__(ax2, ax1)
    ax1.legend(loc='best')
    plt.tight_layout()
    if savefigname is not None: plt.savefig(savefigname)
    plt.show()

def _plotAoAAndCl(LiftingLines, savefigname=None):
    '''
    Convenient function for interactively plot AoA and Cl
    geometrical laws of a list of Lifting Lines
    '''

    # TODO remove "_"

    import matplotlib.pyplot as plt

    LiftingLines = I.getZones(LiftingLines)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    Markers = ('o','+','s','.','v','^','x')
    for LiftingLine in LiftingLines:
        if not checkComponentKind(LiftingLine, kind='LiftingLine'): continue
        r, AoA, Cl =  J.getVars(LiftingLine,['Span','AoA','Cl'])
        ax1.plot(r/r.max(),AoA,'-',marker='.',label=LiftingLine[0])
        ax2.plot(r/r.max(),Cl,'--',marker='.')

    ax1.set_xlabel('$r/R$')
    ax1.set_ylabel('AoA (deg) (-)')
    ax2.set_ylabel('Cl (- - -)')

    ax2.format_coord = _makeFormat__(ax2, ax1)
    ax1.legend(loc='upper left')
    plt.tight_layout()
    if savefigname is not None: plt.savefig(savefigname)
    plt.show()

def convertHOSTPolarFile2Dict(filename):
    '''
    Extract airfoil polar information and convert it to Python
    Dictionnary, given a filename including path of a HOST
    formatted file.

    Parameters
    ----------

        filename : str
            full or relative path towards HOST formatted file

    Returns
    -------

        Result : dict
            Python Dictionnary containing the numerical values
    '''
    def scan(line,OutputType=float, RegExpr=r'[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?'):
        scanned = re.findall(RegExpr,line)
        return [OutputType(item) for item in scanned]

    with open(filename,'r') as f:
        lines = f.readlines()

        Data = {'Cl':{}, 'Cd':{},'Cm':{},}

        AllowedVars = Data.keys()

        LinesQty = len(lines)

        Data['Title']='_'.join(lines[0].split()[1:])

        # Read Allowed Variables:

        for i in range(LinesQty):
            lS = lines[i].split()
            if (len(lS) >= 2) and (lS[1] in AllowedVars):
                Var = lS[1]

                AoAQty, MachQty = scan(lines[i+1],int)

                # Get Angles of Attack
                AoA = []
                j = i+1
                while len(AoA) < AoAQty:
                    j += 1
                    AoA += scan(lines[j],float)
                Data[Var]['AoA'] = np.array(AoA,order='F')

                # Get Mach numbers
                Mach = []
                while len(Mach) < MachQty:
                    j += 1
                    Mach += scan(lines[j],float)
                Data[Var]['Mach'] = np.array(Mach,order='F')

                # Get Variable
                VarNumpy = np.empty((AoAQty,MachQty),order='F')
                VarNumpy[:] = 1
                for a in range(AoAQty):
                    VarLine = []
                    while len(VarLine) < MachQty:
                        j += 1
                        VarLine += scan(lines[j],float)
                    VarNumpy[a,:] = np.array(VarLine,order='F')
                Data[Var]['Array'] = VarNumpy

                # Read big angles
                j+=1
                NextTag = lines[j].split()
                SetOfBigAoA = []
                SetOfBigAoAValues = []
                while len(NextTag) == 1:
                    BigAoA, BigAoAValues = [], []
                    BigAoAQty = int(NextTag[0])
                    while len(BigAoA) < BigAoAQty:
                        j += 1
                        BigAoA += scan(lines[j],float)
                    while len(BigAoAValues) < BigAoAQty:
                        j += 1
                        BigAoAValues += scan(lines[j],float)
                    SetOfBigAoA += BigAoA
                    SetOfBigAoAValues += BigAoAValues
                    j+=1
                    try:
                        NextTag = lines[j].split()
                    except IndexError:
                        break


                SortInd = np.argsort(SetOfBigAoA)
                SetOfBigAoA= np.array([SetOfBigAoA[i] for i in SortInd], order='F')
                SetOfBigAoAValues= np.array([SetOfBigAoAValues[i] for i in SortInd], order='F')

                Data[Var]['BigAoA'] = SetOfBigAoA
                Data[Var]['BigAoAValues'] = SetOfBigAoAValues
            elif '(C*L/NU)I0' in lines[i]:
                j=i
                ReynoldsOverMach = scan(lines[j],float)
                Data['ReynoldsOverMach'] = ReynoldsOverMach[-1]
                Data['Cl']['Reynolds'] = Data['ReynoldsOverMach']*Data['Cl']['Mach']
            elif (len(lS) == 2) and (lS[1] == 'Reynolds'):
                # Get Reynolds
                j = i+1
                ReynoldsQty = scan(lines[j],int)[0]
                if ReynoldsQty != MachQty:
                    raise ValueError('ReynoldsQty (%g) is not equal to MachQty (%g). Check your HOST file.'%(ReynoldsQty,MachQty))
                Reynolds = []
                while len(Reynolds) < ReynoldsQty:
                    j += 1
                    Reynolds += scan(lines[j],float)
                for Var in AllowedVars:
                    Data[Var]['Reynolds'] = np.array(Reynolds,order='F')
    Data['PyZonePolarKind'] = 'Struct_AoA_Mach'

    return Data

def convertDict2PyZonePolar(HostDictionnary):
    """
    Convert the dictionary obtained using :py:func:`convertHOSTPolarFile2Dict`
    to a CGNS format polar zone.

    Parameters
    ----------

        HostDictionnary : dict
            as provided by the function :py:func:`convertHOSTPolarFile2Dict`

    Returns
    -------

        PyZonePolar : zone
            CGNS structured data containing the 2D airfoil
            aerodynamic haracteristics and other relevant data for interpolation
            operations
    """

    # Get the size of the main data array
    Data = HostDictionnary
    DataDims = Data['Cl']['Array'].shape

    if len(DataDims)<3:
        Ni, Nj = DataDims
        Dims2Set = Ni, Nj, 1
    else:
        Dims2Set = Ni, Nj, Nk = DataDims

    # Produce an auxiliar zone where data will be stored
    PyZonePolar = G.cart((0,0,0),(1,1,1),Dims2Set)
    for var in ('Cl', 'Cd', 'Cm'):
        try:
            ArrayValues = Data[var]['Array']
        except KeyError:
            continue
        C._initVars(PyZonePolar,var,0.)
        Array = I.getNodeFromName(PyZonePolar,var)[1]
        Array[:] = ArrayValues

    try:
        Title = Data['Title']
    except KeyError:
        print("WARNING: convertDict2PyZonePolar() ->\n Provided data has no airfoil title.\nThis may produce future interpolation errors.")
        Title = 'Untitled'

    PyZonePolar[0]= Title
    I._rmNodesByName(PyZonePolar,'GridCoordinates')

    # Add information on data range
    children = [
    ['AngleOfAttack', Data['Cl']['AoA']],
    ['Mach', Data['Cl']['Mach']],
    ['BigAngleOfAttackCl', Data['Cl']['BigAoA']],
    ['BigAngleOfAttackCd', Data['Cd']['BigAoA']],
    ['BigAngleOfAttackCm', Data['Cm']['BigAoA']],
    ]

    if 'Reynolds' in Data['Cl']:
        children += [ ['Reynolds', Data['Cl']['Reynolds']], ]
    if 'ReynoldsOverMach' in Data:
        children += [ ['ReynoldsOverMach', Data['ReynoldsOverMach']], ]

    J._addSetOfNodes(PyZonePolar,'.Polar#Range',children)


    # Add out-of-range big Angle Of Attack values
    children = [
    ['BigAngleOfAttackCl', Data['Cl']['BigAoAValues']],
    ['BigAngleOfAttackCd', Data['Cd']['BigAoAValues']],
    ['BigAngleOfAttackCm', Data['Cm']['BigAoAValues']],
    ]
    J._addSetOfNodes(PyZonePolar,'.Polar#OutOfRangeValues',children)

    # Add .Polar#Interp node
    children=[
    ['PyZonePolarKind',Data['PyZonePolarKind']],
    ['Algorithm','RectBivariateSpline'],
    ]
    J._addSetOfNodes(PyZonePolar,'.Polar#Interp',children)

    return PyZonePolar

def convertHOSTPolarFile2PyZonePolar(filename):
    '''
    Convert a HOST-format 2D polar file into a CGNS-structured polar.

    Parameters
    ----------

        filename : str
            full or relative path of HOST polars

    Returns
    -------

        PyZonePolar : zone
            specific zone including 2D polar predictions
    '''
    Data        = convertHOSTPolarFile2Dict(filename)
    PyZonePolar = convertDict2PyZonePolar(Data)
    return PyZonePolar


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
    import Converter.Mpi as Cmpi
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
    import Converter.Mpi as Cmpi
    from .Coprocess import printCo

    def getItemOrRaiseWarning(itemName):
        try:
            item = Rotor[itemName]
        except KeyError:
            try: name = Rotor['name']
            except KeyError: name = '<UndefinedName>'
            MSG = 'WARNING: {} of rotor {} not found at proc {}'.format(
                            itemName,  name,  Cmpi.rank)
            printCo(MSG)

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
    from .Coprocess import printCo
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
        printCo(traceback.format_exc(),color=J.FAIL)
        os._exit(0)

    return BodyForceDisks

def write4Debug(MSG):
    import Converter.Mpi as Cmpi
    with open('LOGS/rank%d.log'%Cmpi.rank,'a') as f: f.write('%s\n'%MSG)

def convertPolarsCGNS2HOSTformat(PyZonePolars,
                                 DIRECTORY_SAVE='POLARS',
                                 OutputFileNamePreffix='HOST_'):
    '''
    This function performs a conversion from CGNS *PyZonePolar* files towards
    HOST ascii format (neglecting special information that cannot be translated
    into HOST format)

    Parameters
    ----------

        PyZonePolars : :py:class:`list` of zone
            zones of 2D polars to be converted

        DIRECTORY_SAVE : str
            The directory where new HOST files are to be writen

        OutputFileNamePreffix : str
            a preffix to append to the name of the HOST files.

    Returns
    -------

        None : None
            HOST files
    '''

    AllowedQuantities = ('Cl','Cd','Cm')

    if not os.path.isdir(DIRECTORY_SAVE): os.makedirs(DIRECTORY_SAVE)

    for FoilZone in I.getZones(PyZonePolars):
        FoilName = I.getName( FoilZone )

        FileFullPath=os.path.join(DIRECTORY_SAVE,OutputFileNamePreffix+FoilName)
        print('writing %s'%FileFullPath)
        with open(FileFullPath,'w+') as f:

            FlowSol_n = I.getNodeFromName1(FoilZone,'FlowSolution')

            PolarRange = I.getNodeFromName1(FoilZone,'.Polar#Range')

            AngleOfAttackRange = I.getNodeFromName1(PolarRange,
                                                    'AngleOfAttack')[1]
            AoAQty = len(AngleOfAttackRange)

            MachRange = I.getNodeFromName1(PolarRange,'Mach')[1]
            MachQty = len(MachRange)

            ReynoldsRange = I.getNodeFromName1(PolarRange,'Reynolds')[1]

            AvrgReOverMach = np.mean(ReynoldsRange/MachRange)

            BigAoAsRange_nodes = I.getNodesFromName(PolarRange,
                                                    'BigAngleOfAttack*')
            BigAoAsRangesDict = {}
            for BigAoAsRange in BigAoAsRange_nodes:
                KeyName = BigAoAsRange[0].replace('BigAngleOfAttack','')
                BigAoAsRangesDict[KeyName] = BigAoAsRange[1]



            OutOfRange = I.getNodeFromName1(FoilZone,'.Polar#OutOfRangeValues')

            BigAoAsValue_nodes = I.getNodesFromName(OutOfRange,
                                                    'BigAngleOfAttack*')
            BigAoAsValuesDict = {}
            for BigAoAsValue in BigAoAsValue_nodes:
                KeyName = BigAoAsValue[0].replace('BigAngleOfAttack','')
                BigAoAsValuesDict[KeyName] = BigAoAsValue[1]

            f.write('      78      %s\n'%FoilName)
            f.write('%5i\n' %MachQty)

            for var in AllowedQuantities:
                var_n = I.getNodeFromName1(FlowSol_n,var)
                varName   = var
                varValues = I.getValue( var_n )

                f.write('    1 %s %s\n'%(varName,FoilName))
                f.write('%5i%5i\n'%(AoAQty, MachQty))
                inc=1
                for i in AngleOfAttackRange: 
                    f.write('%10.5f'%i)
                    if inc%8==0 or inc==len(AngleOfAttackRange) :
                        f.write('\n')
                    inc+=1
                inc=1
                for i in MachRange: 
                    f.write('%10.5f'%i)
                    if inc%8==0 or inc==len(MachRange):
                        f.write('\n')
                    inc+=1
                for row in varValues:
                    inc=1
                    for i in row: 
                        f.write('%10.5f'%i)
                        if inc==len(MachRange) :
                            f.write('\n')
                        inc+=1
                BigAoARange = BigAoAsRangesDict[varName]
                BigAoAValue = BigAoAsValuesDict[varName]
                LowAoABool  = BigAoARange < 0
                HighAoABool = BigAoARange > 0

                for BoolRange in (HighAoABool, LowAoABool):
                    # f.write('  ')
                    f.write('%5i\n'%len(BigAoARange[BoolRange]))
                    inc=1
                    for i in BigAoARange[BoolRange]: 
                        f.write('%10.5f'%i)
                        if inc%8==0 or inc==len(BigAoARange[BoolRange]) :
                            f.write('\n')
                        inc+=1
                    # f.write('  ')
                    inc=1
                    for i in BigAoAValue[BoolRange]: 
                        f.write('%10.5f'%i)
                        if inc%8==0 or inc==len(BigAoAValue[BoolRange]) :
                            f.write('\n')
                        inc+=1

            f.write('COEFFICIENT (C*L/NU)I0 (OU BIEN REYNOLDS/MACH) ............ %10.1f\n' %AvrgReOverMach)
            f.write('CORRECTION DE PRESSION GENERATRICE REYNOLDS/MACH=CSTE. .... SANS\n')
            f.write('EXPOSANT POUR CORRECTION DE REYNOLDS ( EXPREY) ............   -0.16667')
        os.chmod(FileFullPath, 0o755)



def buildVortexParticleSourcesOnLiftingLine(t, AbscissaSegments=[0,0.5,1.],
    IntegralLaw='linear'):
    '''
    Build a set of zones composed of particles with fields:

    ``CoordinateX``, ``CoordinateY``, ``CoordinateZ``, ``Gamma``

    Parameters
    ----------

        t : PyTree, base, zone, list of zones
            container with lifting-line zones

        AbscissaSegments : :py:class:`list` or :py:class:`list` of :py:class:`list`
            it defines the segments that discretizes the lifting line.
            It must be :math:`\in [0,1]`. If several lists are provided (list of
            lists) then each discretization correspond to each LiftingLine found
            on **t**

        IntegralLaw : str
            interpolation law for the interpolation of data contained in the
            lifting line

    Returns
    -------

        AllSourceZones : :py:class:`list` of zone
            list of zones composed of particles element type (*NODE*)
    '''



    FieldsNames2Extract = ['Coordinate' + v for v in 'XYZ'] + \
                                        ['Velocity2D' + v for v in 'XYZ'] + ['Gamma', 'VelocityMagnitudeLocal']
    AllSourceZones = []

    LiftingLines = [z for z in I.getZones(t) if checkComponentKind(z,'LiftingLine')]
    NumberOfLiftingLines = len(LiftingLines)
    NumberOfAbscissaSegments = 0
    for AbscissaSegmentSubList in AbscissaSegments:
        if not (isinstance(AbscissaSegmentSubList, list) or \
                                                   isinstance(AbscissaSegmentSubList, np.ndarray)):
            NewAbscissaSegments = [AbscissaSegments for _ in range(NumberOfLiftingLines)]
            AbscissaSegments = NewAbscissaSegments
            NumberOfAbscissaSegments = len(AbscissaSegments)
            break
        NumberOfAbscissaSegments += 1
    if NumberOfAbscissaSegments != NumberOfLiftingLines:
        MSG = ('abscissa segments sublists number ({}) must be equal to the'
              ' total number of lifting lines contained in "t" ({})').format(
                NumberOfAbscissaSegments, NumberOfLiftingLines)
        raise AttributeError(FAIL+MSG+ENDC)


    for LiftingLine, AbscissaSegment in zip(LiftingLines, AbscissaSegments):
        VPM_Parameters = J.get(LiftingLine,'.VPM#Parameters')

        AbscissaSegment = np.append(2.*AbscissaSegment[0] - AbscissaSegment[1],
                                        AbscissaSegment)
        AbscissaSegment = np.append(AbscissaSegment ,
                                    2*AbscissaSegment[-1] - AbscissaSegment[-2])
        AbscissaSegment = np.array(AbscissaSegment, dtype=np.float64)

        v = J.getVars2Dict(LiftingLine,['s']+FieldsNames2Extract[3:])
        x,y,z = J.getxyz(LiftingLine)
        v['CoordinateX'] = x
        v['CoordinateY'] = y
        v['CoordinateZ'] = z

        sourcefields = {}
        if IntegralLaw.startswith('interp1d'):
            kind = IntegralLaw.replace('interp1d_','')
            for fieldname in FieldsNames2Extract:
                interpolator = si.interp1d(v['s'], v[fieldname],
                                           kind=kind,
                                           bounds_error=False,
                                           fill_value='extrapolate',
                                           assume_sorted=True, copy=False)
                sourcefields[fieldname] = interpolator(AbscissaSegment)

        elif IntegralLaw == 'linear':
            for fieldname in FieldsNames2Extract:
                sourcefields[fieldname] = np.interp(AbscissaSegment,
                                                    v['s'],
                                                    v[fieldname])
                sourcefields[fieldname][0] = 2*sourcefields[fieldname][0]-\
                                                   sourcefields[fieldname][2]

                sourcefields[fieldname][-1] = 2*sourcefields[fieldname][-1]-\
                                                   sourcefields[fieldname][-3]


        elif IntegralLaw == 'pchip':
            for fieldname in FieldsNames2Extract:
                interpolator = si.PchipInterpolator(v['s'], v[fieldname],
                                                    extrapolate=True)
                sourcefields[fieldname] = interpolator(AbscissaSegment)

        elif IntegralLaw == 'akima':
                interpolator = si.PchipInterpolator(v['s'], v[fieldname])
                sourcefields[fieldname] = interpolator(AbscissaSegment,extrapolate=True)

        else:
            raise AttributeError('IntegralLaw "%s" not supported'%IntegralLaw)



        Arrays = [sourcefields[fn] for fn in FieldsNames2Extract]
        ArraysNames = FieldsNames2Extract

        Sources = J.createZone(LiftingLine[0]+'.Sources',Arrays, ArraysNames)
        Sources = C.convertArray2Node(Sources)
        AllSourceZones.append(Sources)

    return AllSourceZones

def getTrailingEdge(t):
    '''
    construct the curve corresponding to the TrailingEdge from a LiftingLine,
    conserving all original fields and data.

    Frenet frame and kinematic velocity are updated with new locations.

    .. warning:: induced velocities are **not** updated

    Parameters
    ----------

        t : PyTree, Base, zone or :py:class:`list` of zones
            the lifting-line (situated at :math:`c/4`)

    Returns
    -------

        TrailingEdgeLines : base
            ``CGNSBase_t`` of zones  corresponding to trailing edge
    '''

    LiftingLines = [z for z in I.getZones(t) if checkComponentKind(z,'LiftingLine')]
    TrailingEdgeLines = []
    for LiftingLine in LiftingLines:
        if not checkComponentKind(LiftingLine,'LiftingLine'):
            raise AttributeError('input must be a LiftingLine')
        TrailingEdge = I.copyTree(LiftingLine)
        x, y, z = J.getxyz(TrailingEdge)
        Chord, ChordwiseX, ChordwiseY, ChordwiseZ = J.getVars(TrailingEdge,
            ['Chord','ChordwiseX', 'ChordwiseY', 'ChordwiseZ'])

        Distance2TrailingEdge = 0.75 * Chord
        x += Distance2TrailingEdge * ChordwiseX
        y += Distance2TrailingEdge * ChordwiseY
        z += Distance2TrailingEdge * ChordwiseZ
        TrailingEdgeLines += [TrailingEdge]

    TrailingEdgeBase = I.newCGNSBase('TrailingEdge',cellDim=1,physDim=3)
    TrailingEdgeBase[2] = TrailingEdgeLines # Add Blades

    # Sets component general information
    J.set(TrailingEdgeBase,'.Component#Info',kind='TrailingEdge')

    return TrailingEdgeBase

def getLeadingEdge(t):
    '''
    construct the curve corresponding to the LeadingEdge from a LiftingLine,
    conserving all original fields and data.

    Frenet frame and kinematic velocity are updated with new locations.

    .. warning:: induced velocities are **not** updated

    Parameters
    ----------

        t : PyTree, Base, zone or :py:class:`list` of zones
            the lifting-line (situated at :math:`c/4`)

    Returns
    -------

        LeadingEdgeLines : base
            ``CGNSBase_t`` of zones  corresponding to leading edge
    '''
    LiftingLines = [z for z in I.getZones(t) if checkComponentKind(z,'LiftingLine')]
    LeadingEdgeLines = []
    for LiftingLine in LiftingLines:
        if not checkComponentKind(LiftingLine,'LiftingLine'):
            raise AttributeError('input must be a LiftingLine')
        LeadingEdge = I.copyTree(LiftingLine)
        x, y, z = J.getxyz(LeadingEdge)

        Chord, ChordwiseX, ChordwiseY, ChordwiseZ = J.getVars(LeadingEdge,
            ['Chord','ChordwiseX', 'ChordwiseY', 'ChordwiseZ'])

        Distance2TrailingEdge = 0.25 * Chord
        x -= Distance2TrailingEdge * ChordwiseX
        y -= Distance2TrailingEdge * ChordwiseY
        z -= Distance2TrailingEdge * ChordwiseZ
        LeadingEdgeLines += [LeadingEdge]

    LeadingEdgeBase = I.newCGNSBase('LeadingEdge',cellDim=1,physDim=3)
    LeadingEdgeBase[2] = LeadingEdgeLines # Add Blades

    # Sets component general information
    J.set(LeadingEdgeBase,'.Component#Info',kind='LeadingEdge')

    return LeadingEdgeBase

def getAirfoilsNodeOfLiftingLine(LiftingLine):
    LiftingLine, = I.getZones(LiftingLine)
    ComponentInfo = I.getNodeFromName1(LiftingLine, '.Component#Info')
    GeometricalLaws = I.getNodeFromName1(ComponentInfo, 'GeometricalLaws')
    if not GeometricalLaws: return I.getNodeFromName1(LiftingLine,'.Polar#Info')
    return I.getNodeFromName1(GeometricalLaws, 'Airfoils')

def loadPolarsInterpolatorDict( filenames, InterpFields=['Cl', 'Cd','Cm'] ):
    if isinstance(filenames, str):
        filenames = [ filenames ]
    elif not isinstance(filenames, list):
        raise TypeError('filenames %s not recognized'%str(type(filenames)))

    PyZonePolars = []
    for filename in filenames:
        PyZonePolars.extend( I.getZones( C.convertFile2PyTree(filename) ) )

    return buildPolarsInterpolatorDict(PyZonePolars, InterpFields=InterpFields)

def mirrorBlade(LiftingLine):
    '''
    Given a Lifting-Line in canonical position (as produced using :py:func:`buildLiftingLine`),
    mirrors its geometry, such that it can be used for an opposite rotation.

    Parameters
    ----------

        LiftingLine : zone
            A Lifting Line object, as generated from
            function :py:func:`buildLiftingLine`
    '''

    C._initVars(LiftingLine,'{CoordinateY}=-{CoordinateY}')
    C._initVars(LiftingLine,'{ChordwiseY}=-{ChordwiseY}')
    C._initVars(LiftingLine,'{SpanwiseY}=-{SpanwiseY}')
    C._initVars(LiftingLine,'{ThickwiseY}=-{ThickwiseY}')
    C._initVars(LiftingLine,'{PitchRelativeCenterY}=-{PitchRelativeCenterY}')
    C._initVars(LiftingLine,'{TangentialY}=-{TangentialY}')

    C._initVars(LiftingLine,'{PitchAxisX}=-{PitchAxisX}')
    C._initVars(LiftingLine,'{PitchAxisY}=-{PitchAxisY}')
    C._initVars(LiftingLine,'{PitchAxisZ}=-{PitchAxisZ}')


def addPitch(LiftingLine, pitch=0.0):
    '''
    Given a Lifting-Line (at any arbitrary position) add a pitch angle (in degrees),
    resulting in a rotation of the LiftingLine and its meaningful fields

    Parameters
    ----------

        LiftingLine : PyTree, base, zone or list of zones
            A Lifting Line object, as generated from
            function :py:func:`buildLiftingLine`.

            .. note:: 
                **LitingLine** is modified in-place

        pitch : float
            angle in degrees of the rotation around `PitchAxis` passing through
            `PitchRelativeCenter`, previously defined in :py:func:`buildLiftingLine`
    '''
    for LL in getLiftingLines(LiftingLine):
        PitchCtr = J.getVars(LL,
                        ['PitchRelativeCenterX','PitchRelativeCenterY','PitchRelativeCenterZ'])

        PitchAxis = J.getVars(LL, ['PitchAxisX','PitchAxisY','PitchAxisZ'])

        PitchCtr_pt = (PitchCtr[0][0]*1.0, PitchCtr[1][0]*1.0, PitchCtr[2][0]*1.0)
        PitchAxis_vec = (PitchAxis[0][0], PitchAxis[1][0], PitchAxis[2][0])
        T._rotate(LL, PitchCtr_pt, PitchAxis_vec, pitch, 
                vectors=NamesOfChordSpanThickwiseFrameNoTangential)
        
def getLocalFrameLines(LiftingLines, Length=0.05):
    '''
    Construct the Chordwise, Thickwise, Spanwise lines at LiftingLines for 
    verification purposes (visualization)

    Parameters
    ----------

        LiftingLines : PyTree, Base, Zone or :py:class:`list` of zone
            variable including LitingLine objects

        Length : float
            dimension of the lines used for visualization

    Returns
    -------

        Lines : :py:class:`list` of zones
            lines of the frame, ready for visualization        
    '''
    Lines = []
    for LiftingLine in getLiftingLines(LiftingLines):
        xyz = np.vstack(J.getxyz(LiftingLine))
        chordwise = np.vstack(J.getVars(LiftingLine,['Chordwise'+i for i in 'XYZ']))
        spanwise  = np.vstack(J.getVars(LiftingLine,[ 'Spanwise'+i for i in 'XYZ']))
        thickwise = np.vstack(J.getVars(LiftingLine,['Thickwise'+i for i in 'XYZ']))

        for i in range(C.getNPts(LiftingLine)):
            chordwise_line = D.line(tuple(xyz[:,i]),tuple(xyz[:,i]+chordwise[:,i]*Length),2)
            chordwise_line[0] = 'Chordwise.%d'%i
            Lines += [chordwise_line]

            spanwise_line = D.line(tuple(xyz[:,i]),tuple(xyz[:,i]+spanwise[:,i]*Length),2)
            spanwise_line[0] = 'Spanwise.%d'%i
            Lines += [spanwise_line]

            thickwise_line = D.line(tuple(xyz[:,i]),tuple(xyz[:,i]+thickwise[:,i]*Length),2)
            thickwise_line[0] = 'Thickwise.%d'%i
            Lines += [chordwise_line, spanwise_line, thickwise_line]

    I._correctPyTree(Lines,level=3)
    return Lines

