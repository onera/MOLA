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
VULCAINS (Viscous Unsteady Lagrangian Code for Aerodynamics with Incompressible Navier-Stokes)

This module enables the coupling of the VPM with the Lifting Line module of MOLA.

Version:
0.5

Author:
Johan VALENTIN
'''

import numpy as np
import Converter.PyTree as C
import Converter.Internal as I
import Transform.PyTree as T

from .. import LiftingLine as LL
from .. import Wireframe as W
from .. import InternalShortcuts as J
from . import Main as V

####################################################################################################
####################################################################################################
########################################### Lifting Lines ##########################################
####################################################################################################
####################################################################################################
def initialiseLiftingLines(tLL = [], Parameters = {}):
    '''
    Initialises the Lifting Lines and updates the parameters.

    Parameters
    ----------
        tLL : Tree
            Lifting Lines.

        Parameters : :py:class:`dict` of :py:class:`dict`
            User-provided VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.User.compute`
    Returns
    -------
        tLL : Tree
            Lifting Lines.
    '''
    if not tLL:
        if 'LiftingLineParameters' in Parameters: Parameters['LiftingLineParameters'].clear()
        return []
        
    if isinstance(tLL, str): tLL = V.load(tLL)
    
    tLL = V.checkTreeStructure(tLL, 'LiftingLines')
    updateLiftingLinesParameters(tLL, Parameters)
    updateParametersFromLiftingLines(tLL, Parameters)
    return tLL

def rotateLiftingLineSections(tLL = []):
    '''
    Rotates the Lifting Lines sections to be perpendicular to the Lifting Lines.

    Parameters
    ----------
        tLL : Tree
            Lifting Lines.
    '''
    _tLL = V.getTrees([tLL], ['LiftingLines'])
    if not _tLL: return

    import scipy
    Rotation = lambda v, theta, axis: \
                               scipy.spatial.transform.Rotation.from_rotvec(theta*axis).apply(v)
    for LiftingLine in I.getZones(_tLL):
        Corrections3D = I.getNodeFromName(LiftingLine, 'Corrections3D')
        try: SweepCorr = I.getValue(I.getNodeFromName(Corrections3D, 'Sweep')) == 0
        except: SweepCorr = 1
        try: DihedralCorr = I.getValue(I.getNodeFromName(Corrections3D, 'Dihedral')) == 0
        except: DihedralCorr = 1
        x, y, z = J.getxyz(LiftingLine)
        xyz = np.vstack((x,y,z))
        tanX, tanY, tanZ, chordX, chordY, chordZ, spanX, spanY, spanZ, thickX, thickY, thickZ, \
            PitchAxisX, PitchAxisY, PitchAxisZ, sweep, dihedral, chord, chordsweep = J.getVars(\
            LiftingLine, ['Tangential' + v for v in 'XYZ'] + ['Chordwise' + v for v in 'XYZ'] +\
                          ['Spanwise' + v for v in 'XYZ'] + ['Thickwise' + v for v in 'XYZ'] + \
                                          ['PitchAxis' + v for v in 'XYZ'] + ['SweepAngleDeg', \
                                          'DihedralAngleDeg', 'Chord', 'ChordVirtualWithSweep'])
        if SweepCorr: chordsweep[:] = chord
        RotationAxis = I.getValue(I.getNodeFromName(LiftingLine, 'RotationAxis'))
        RotationAxis /= np.linalg.norm(RotationAxis, axis = 0)

        PitchAxis = np.array([np.mean(PitchAxisX), np.mean(PitchAxisY), np.mean(PitchAxisZ)])
        PitchAxis /= np.linalg.norm(PitchAxis, axis = 0)

        spanxyz = np.hstack(( (xyz[:, 1] - xyz[:, 0])[np.newaxis].T, 
                            0.5*(np.diff(xyz[:, :-1], axis = 1) + np.diff(xyz[:, 1:], axis = 1)),
                           (xyz[:, -1]-xyz[:, -2])[np.newaxis].T))
        spanxyz /= np.linalg.norm(spanxyz, axis = 0)
        spanX[:] = spanxyz[0,:]
        spanY[:] = spanxyz[1,:]
        spanZ[:] = spanxyz[2,:]
        tanX[:] = spanxyz[1, :]*RotationAxis[2] - spanxyz[2, :]*RotationAxis[1]
        tanY[:] = spanxyz[2, :]*RotationAxis[0] - spanxyz[0, :]*RotationAxis[2]
        tanZ[:] = spanxyz[0, :]*RotationAxis[1] - spanxyz[1, :]*RotationAxis[0]
        for i in range(len(chordX)):
            if SweepCorr:
                chord = Rotation(np.array([chordX[i], chordY[i], chordZ[i]]), sweep[i]*np.pi/180.,
                                                                                      -RotationAxis)
                norm = np.linalg.norm(chord, axis = 0)
                chordX[i] = chord[0]/norm
                chordY[i] = chord[1]/norm
                chordZ[i] = chord[2]/norm
                thick = Rotation(np.array([thickX[i], thickY[i], thickZ[i]]), sweep[i]*np.pi/180.,
                                                                                      -RotationAxis)
                norm = np.linalg.norm(thick, axis = 0)
                thickX[i] = thick[0]/norm
                thickY[i] = thick[1]/norm
                thickZ[i] = thick[2]/norm

            if DihedralCorr:
                RollAxis = np.cross(np.array([PitchAxisX[i], PitchAxisY[i], PitchAxisZ[i]]),
                                                                                       RotationAxis)
                chord = Rotation(np.array([chordX[i], chordY[i], chordZ[i]]),
                                                                   dihedral[i]*np.pi/180., RollAxis)
                norm = np.linalg.norm(chord, axis = 0)
                chordX[i] = chord[0]/norm
                chordY[i] = chord[1]/norm
                chordZ[i] = chord[2]/norm
                thick = Rotation(np.array([thickX[i], thickY[i], thickZ[i]]),
                                                                   dihedral[i]*np.pi/180., RollAxis)
                norm = np.linalg.norm(thick, axis = 0)
                thickX[i] = thick[0]/norm
                thickY[i] = thick[1]/norm
                thickZ[i] = thick[2]/norm

        if DihedralCorr: dihedral[:] = 0.
        if SweepCorr: sweep[:] = 0.

def renameLiftingLinesTree(tLL = []):
    '''
    Checks and updates the types of the nodes of the Lifting Lines.
    
    Parameters
    ----------
        tLL : Tree
            Lifting Lines.
    '''
    _tLL = V.getTrees([tLL], ['LiftingLines'])
    TypeOfInput = I.isStdNode(tLL)
    ERRMSG = J.FAIL + 'LiftingLines must be a tree, a list of bases or a list of zones' + J.ENDC
    if TypeOfInput == -1:# is a standard CGNS node
        if I.isTopTree(_tLL):
            LiftingLineBases = I.getBases(_tLL)
            if len(LiftingLineBases) == 1 and LiftingLineBases[0][0] == 'Base':
                LiftingLineBases[0][0] = 'LiftingLines'
        elif _tLL[3] == 'CGNSBase_t':
            LiftingLineBase = _tLL
            if LiftingLineBase[0] == 'Base': LiftingLineBase[0] = 'LiftingLines'
            _tLL = C.newPyTree([])
            _tLL[2] = [LiftingLineBase]
        elif _tLL[3] == 'Zone_t':
            _tLL = C.newPyTree(['LiftingLines', [_tLL]])
        else:
            raise AttributeError(ERRMSG)
    elif TypeOfInput == 0:# is a list of CGNS nodes
        if _tLL[0][3] == 'CGNSBase_t':
            _tLLBases = I.getBases(_tLL)
            _tLL = C.newPyTree([])
            _tLL[2] = _tLLBases
        elif _tLL[0][3] == 'Zone_t':
            _tLLZones = I.getZones(_tLL)
            _tLL = C.newPyTree(['LiftingLines', _tLLZones])
        else:
            raise AttributeError(ERRMSG)

    else:
        raise AttributeError(ERRMSG)

def updateParametersFromLiftingLines(tLL = [], Parameters = {}):
    '''
    Checks the and updates VPM and Lifting Line parameters.
    
    Parameters
    ----------
        tLL : Tree
            Lifting Lines.

        Parameters : :py:class:`dict` of :py:class:`dict`
            User-provided VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.User.compute`.
    '''
    _tLL = V.getTrees([tLL], ['LiftingLines'])
    LLs = I.getZones(_tLL)
    PP = Parameters['PrivateParameters']
    PP['NumberOfLiftingLines'][0] = len(LLs)
    PP['NumberOfLiftingLineSources'][0] = 0
    for LL in LLs:
        LLParameters = J.get(LL, '.VPM#Parameters')
        PP['NumberOfLiftingLineSources'][0] += LLParameters['NumberOfParticleSources'] \
                                           - 1 + LLParameters['SourcesDistribution']['Symmetrical']
    
    NP = Parameters['NumericalParameters']
    if not(NP['Resolution'][0]) and LLs:
        hmax, hmin = -np.inf, np.inf
        for LL in LLs:
            LLParameters = J.get(LL, '.VPM#Parameters')
            hloc =  LLParameters['LocalResolution']
            hmax = max(hloc, hmax)
            hmin = min(hloc, hmin)
        
        NP['Resolution'] = np.array([hmin, hmax], dtype = np.float64, order = 'F')
        PP['Sigma0'] = NP['Resolution']*Parameters['ModelingParameters']['SmoothingRatio'][0]

    if not(NP['TimeStep']) and LLs:
        setTimeStepFromShedParticles(NP, LLs, NumberParticlesShedAtTip = 1.)

def setLiftingLinesInducedVelocity(tLL = [], InducedVelocity = []):
    '''
    Sets the Lifting Lines induced velocities fields.

    Parameters
    ----------
        tLL : Tree
            Lifting Lines.

        InducedVelocity : numpy.ndarray of 3 numpy.ndarray
            Induced Velocities [Ux, Uy, Uz].
    '''
    _tLL = V.getTrees([tLL], ['LiftingLines'])
    pos = 0
    for LiftingLine in I.getZones(_tLL):
        Ux, Uy, Uz = J.getVars(LiftingLine, ['VelocityInduced' + v for v in 'XYZ'])
        Nll = len(Ux)
        Ux[:] = InducedVelocity[0][pos: pos + Nll]
        Uy[:] = InducedVelocity[1][pos: pos + Nll]
        Uz[:] = InducedVelocity[2][pos: pos + Nll]
        pos += Nll

def resetLiftingLinesInducedVelocity(tLL = []):
    '''
    Resets the Lifting Lines induced velocities fields to zero.

    Parameters
    ----------
        tLL : Tree
            Lifting Lines.
    '''
    _tLL = V.getTrees([tLL], ['LiftingLines'])
    pos = 0
    for LiftingLine in I.getZones(_tLL):
        Ux, Uy, Uz = J.getVars(LiftingLine, ['VelocityInduced' + v for v in 'XYZ'])
        Nll = len(Ux)
        Ux[:] = 0.
        Uy[:] = 0.
        Uz[:] = 0.
        pos += Nll

def getLiftingLinesInducedVelocity(tLL = []):
    '''
    Gets the Lifting Lines induced velocities fields to zero.

    Parameters
    ----------
        tLL : Tree
            Lifting Lines.

    Parameters
    ----------
        InducedVelocity : numpy.ndarray of 3 numpy.ndarray
            Induced Velocities [Ux, Uy, Uz].
    '''
    _tLL = V.getTrees([tLL], ['LiftingLines'])
    UX, UY, UZ = [], [], []
    for LiftingLine in I.getZones(_tLL):
        ux, uy, uz = J.getVars(LiftingLine, ['VelocityInduced' + v for v in 'XYZ'])
        UX.extend(ux)
        UY.extend(uy)
        UZ.extend(uz)

    return np.array([UX, UY, UZ], order = 'F', dtype = np.float64)

def extractWakeInducedVelocityOnLiftingLines(tL = [], tLL = [], Nshed = 0):
    '''
    Gives the velocity induced by the particles in the wake, the hybrid domain and the bem
    particles on Lifting Lines.

    Parameters
    ----------
        tL : Tree
            Lagrangian field.

        tLL : Tree
            Lifting Lines.

        Nshed : :py:class:`int`
            Number of particles shed from the Lifting Lines.
    Returns
    -------
        WakeInducedVelocity : numpy.ndarray of 3 numpy.ndarray
            Induced Velocities [Ux, Uy, Uz].
    '''
    _tL, _tLL = V.getTrees([tL, tLL], ['Particles', 'LiftingLines'])
    if not _tLL: return []

    resetLiftingLinesInducedVelocity(_tLL)
    V.extract_wake_induced_velocity_on_lifting_lines(_tL, _tLL, Nshed)
    return getLiftingLinesInducedVelocity(_tLL)

def extractBoundAndShedVelocityOnLiftingLines(tL = [], tLL = [], Nshed = 0):
    '''
    Gives the velocity induced by the bound and shed particles from the Lifting Lines.

    Parameters
    ----------
        tL : Tree
            Lagrangian field.

        tLL : Tree
            Lifting Lines.

        Nshed : :py:class:`int`
            Number of particles shed from the Lifting Lines.
    Returns
    -------
        BoundAndShedInducedVelocity : numpy.ndarray of 3 numpy.ndarray
            Induced Velocities [Ux, Uy, Uz].
    '''
    _tL, _tLL = V.getTrees([tL, tLL], ['Particles', 'LiftingLines'])
    if not _tLL: return []

    resetLiftingLinesInducedVelocity(_tLL)
    V.extract_bound_and_shed_velocity_on_lifting_lines(_tL, _tLL, Nshed)
    return getLiftingLinesInducedVelocity(_tLL)

def updateLiftingLinesParameters(tLL = [], Parameters = {}):
    '''
    Checks the and updates the parameters in the Lifting Lines.
    
    Parameters
    ----------
        tLL : Tree
            Lifting Lines.

        Parameters : :py:class:`dict` of :py:class:`dict`
            User-provided VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.User.compute`.
    '''
    Zones = I.getZones(V.getTrees([tLL], ['LiftingLines']))
    if not Zones: return []
    U0 = Parameters['FluidParameters']['VelocityFreestream']
    rho = Parameters['FluidParameters']['Density']
    T0 = Parameters['FluidParameters']['Temperature']
    if Parameters['NumericalParameters']['TimeStep'][0]:
        dt = Parameters['NumericalParameters']['TimeStep'][0]
    else: dt = np.inf

    LiftingLineParameters = Parameters['LiftingLineParameters']
    NLLmin = LiftingLineParameters['MinNbShedParticlesPerLiftingLine'][0]
    for LiftingLine in Zones:
        span = W.getLength(LiftingLine)
        LLParameters = J.get(LiftingLine, '.VPM#Parameters')
        if not LLParameters: LLParameters = {}

        if 'IntegralLaw' in LLParameters:
            IntegralLaw = LLParameters['IntegralLaw']
        else:
            IntegralLaw = LiftingLineParameters['IntegralLaw']

        if 'SourcesDistribution' in LLParameters:
            SourcesDistribution = LLParameters['SourcesDistribution']
        elif isinstance(LiftingLineParameters['SourcesDistribution'], dict):
            SourcesDistribution = LiftingLineParameters['SourcesDistribution']
        else:
            ERRMSG = J.FAIL + ('Source particle distribution unspecified for ' + LiftingLine[0]
                                                                                 + '.') + J.ENDC
            raise AttributeError(ERRMSG)

        if 'Symmetrical' not in SourcesDistribution:
            if 'Symmetrical' in LLParameters['SourcesDistribution']:
                SourcesDistribution['Symmetrical'] = \
                                             LLParameters['SourcesDistribution']['Symmetrical']
            elif 'Symmetrical' in LiftingLineParameters['SourcesDistribution']:
                SourcesDistribution['Symmetrical'] = \
                                    LiftingLineParameters['SourcesDistribution']['Symmetrical']
            else:
                ERRMSG = J.FAIL + ('Symmetry of the source particle distribution unspecified '
                                                         'for ' + LiftingLine[0] + '.') + J.ENDC
                raise AttributeError(ERRMSG)

        if 'NumberOfParticleSources' in LLParameters:
            NumberOfParticleSources = max(NLLmin, LLParameters['NumberOfParticleSources'][0])
            LocalResolution = span/NumberOfParticleSources
        elif 'LocalResolution' in LLParameters:
            LocalResolution = LLParameters['LocalResolution'][0]
            NumberOfParticleSources = max(int(round(span/LocalResolution)), NLLmin)
            LocalResolution = span/NumberOfParticleSources
        else:
            NumberOfParticleSources = NLLmin
            LocalResolution = span/NumberOfParticleSources

        if SourcesDistribution['Symmetrical'] and NumberOfParticleSources%2:
            NumberOfParticleSources += 1
            LocalResolution = span/NumberOfParticleSources
            V.show(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))
            V.show(f"{'||':>57}\r" + '|| Odd number of source particles on ' + LiftingLine[0] +  \
                                                                           ' dispite its symmetry.')
            V.show(f"{'||':>57}\r" + '|| Number of particle sources changed to ' + \
                                                             str(NumberOfParticleSources) + '.')
            V.show(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))

        if SourcesDistribution['kind'] == 'ratio' or \
                            SourcesDistribution['kind'] == 'tanhOneSide' or \
                                                 SourcesDistribution['kind'] == 'tanhTwoSides':
            if 'FirstSegmentRatio' in SourcesDistribution:
                SourcesDistribution['FirstCellHeight'] = \
                                       SourcesDistribution['FirstSegmentRatio']*LocalResolution
            elif 'FirstSegmentRatio' in LiftingLineParameters['SourcesDistribution']:
                SourcesDistribution['FirstCellHeight'] = \
                             LiftingLineParameters['SourcesDistribution']['FirstSegmentRatio']\
                                                                                *LocalResolution
            else:
                ERRMSG = J.FAIL + ('FirstSegmentRatio unspecified for ' + LiftingLine[0] + \
                                  ' dispite ' + SourcesDistribution['kind'] + ' law.') + J.ENDC
                raise AttributeError(ERRMSG)

        if SourcesDistribution['kind'] == 'tanhTwoSides':
            if 'LastSegmentRatio' in SourcesDistribution:
                SourcesDistribution['LastCellHeight'] = \
                                        SourcesDistribution['LastSegmentRatio']*LocalResolution
            elif 'LastSegmentRatio' in LiftingLineParameters['SourcesDistribution']:
                SourcesDistribution['LastCellHeight'] = \
                              LiftingLineParameters['SourcesDistribution']['LastSegmentRatio']\
                                                                                *LocalResolution
            else:
                ERRMSG = J.FAIL + ('LastSegmentRatio unspecified for ' + LiftingLine[0] + \
                                  ' dispite ' + SourcesDistribution['kind'] + ' law.') + J.ENDC
                raise AttributeError(ERRMSG)

        if 'CirculationThreshold' in LLParameters:
            CirculationThreshold = LLParameters['CirculationThreshold']
        else:
            CirculationThreshold = LiftingLineParameters['CirculationThreshold']

        if 'CirculationRelaxationFactor' in LLParameters:
            CirculationRelaxationFactor = LLParameters['CirculationRelaxationFactor']
        else:
            CirculationRelaxationFactor = LiftingLineParameters['CirculationRelaxationFactor']

        if 'MaxLiftingLineSubIterations' in LLParameters:
            MaxLiftingLineSubIterations = LLParameters['MaxLiftingLineSubIterations']
        else:
            MaxLiftingLineSubIterations = LiftingLineParameters['MaxLiftingLineSubIterations']

        LL.setVPMParameters(LiftingLine,
            IntegralLaw = IntegralLaw,
            NumberOfParticleSources = np.array([NumberOfParticleSources], order = 'F',
                                                                              dtype = np.int32),
            SourcesDistribution = SourcesDistribution,
            CirculationThreshold = np.array([CirculationThreshold], order = 'F',
                                                                            dtype = np.float64),
            CirculationRelaxationFactor = np.array([CirculationRelaxationFactor], order = 'F',
                                                                            dtype = np.float64),
            LocalResolution = np.array([LocalResolution], order = 'F', dtype = np.float64),
            MaxLiftingLineSubIterations = np.array([MaxLiftingLineSubIterations], order = 'F',
                                                                              dtype = np.int32),
            TimeSinceLastShedding = dt)

        LL.setConditions(LiftingLine, VelocityFreestream = U0,
                                      Density = rho,
                                      Temperature = T0)
    if LiftingLineParameters['RPM']: LL.setRPM(Zones, LiftingLineParameters['RPM'])

    if LiftingLineParameters['VelocityTranslation']:
        for LiftingLine in Zones:
            Kinematics = I.getNodeFromName(LiftingLine, '.Kinematics')
            VelocityTranslation = I.getNodeFromName(Kinematics, 'VelocityTranslation')
            VelocityTranslation[1] = np.array(LiftingLineParameters['VelocityTranslation'],
                                                                    dtype = np.float64, order = 'F')

def initialiseParticlesOnLitingLine(tL = [], tLL = [], Parameters = {}):
    '''
    Initialises the bound particles embedded on the Lifting Lines and the first row of shed
    particles.
    
    Parameters
    ----------
        tL : Tree
            Lagrangian field.

        tLL : Tree
            Lifting Lines.

        Parameters : :py:class:`dict` of :py:class:`dict`
            User-provided VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.User.compute`
    '''
    _tL, _tLL = V.getTrees([tL, tLL], ['Particles', 'LiftingLines'])
    if not _tLL:
        Parameters['PrivateParameters']['NumberOfLiftingLines'][0] = 0
        Parameters['PrivateParameters']['NumberOfLiftingLineSources'][0] = 0
        return

    
    LL.computeKinematicVelocity(_tLL)
    LL.assembleAndProjectVelocities(_tLL)
    LL._applyPolarOnLiftingLine(_tLL, V.PolarsInterpolators[0], ['Cl', 'Cd', 'Cm'])
    LL.computeGeneralLoadsOfLiftingLine(_tLL)

    X0, Y0, Z0, AX0, AY0, AZ0, S0 = [], [], [], [], [], [], []
    X, Y, Z, AX, AY, AZ, S = [], [], [], [], [], [], []
    SmoothingRatio = Parameters['ModelingParameters']['SmoothingRatio'][0]
    U0 = Parameters['FluidParameters']['VelocityFreestream']
    for LiftingLine in I.getZones(_tLL):
        #Gamma, GammaM1 = J.getVars(LiftingLine, ['Gamma', 'GammaM1'])
        #GammaM1[:] = Gamma[:]
        h = I.getValue(I.getNodeFromName(LiftingLine, 'LocalResolution'))
        L = W.getLength(LiftingLine)
        LLParameters = J.get(LiftingLine, '.VPM#Parameters')
        SourcesDistribution = LLParameters['SourcesDistribution']
        if SourcesDistribution['Symmetrical']:
            HalfStations = int(LLParameters['NumberOfParticleSources'][0]/2 + 1)
            SemiWing = W.linelaw(P1 = (0., 0., 0.), P2 = (L/2., 0., 0.), N = HalfStations,
                                                            Distribution = SourcesDistribution)# has to give +1 point because one point is lost with T.symetrize()
            WingDiscretization = J.getx(T.join(T.symetrize(SemiWing, (0, 0, 0), (0, 1, 0), \
                                                                          (0, 0, 1)), SemiWing))
            WingDiscretization += L/2.
            SourcesDistribution = WingDiscretization/L
        else:
            WingDiscretization = J.getx(W.linelaw(P1 = (0., 0., 0.), P2 = (L, 0., 0.),
                                                 N = LLParameters['NumberOfParticleSources'][0],
                                                           Distribution = SourcesDistribution))
            SourcesDistribution = WingDiscretization/L

        LLParameters = J.get(LiftingLine, '.VPM#Parameters')
        LLParameters['SourcesDistribution'] = SourcesDistribution
        J.set(LiftingLine, '.VPM#Parameters', **LLParameters)
        Source = LL.buildVortexParticleSourcesOnLiftingLine(LiftingLine,
                                                      AbscissaSegments = [SourcesDistribution],
                                                      IntegralLaw = LLParameters['IntegralLaw'])

        SourceX = I.getValue(I.getNodeFromName(Source, 'CoordinateX'))
        SourceY = I.getValue(I.getNodeFromName(Source, 'CoordinateY'))
        SourceZ = I.getValue(I.getNodeFromName(Source, 'CoordinateZ'))
        Gamma   = I.getValue(I.getNodeFromName(Source, 'Gamma'))
        dy = np.linalg.norm(np.vstack([SourceX[2:-1] - SourceX[1:-2], SourceY[2:-1] - \
                                       SourceY[1:-2], SourceZ[2:-1] - SourceZ[1:-2]]), axis = 0)
        X0.extend(0.5*(SourceX[2:-1] + SourceX[1:-2]))
        Y0.extend(0.5*(SourceY[2:-1] + SourceY[1:-2]))
        Z0.extend(0.5*(SourceZ[2:-1] + SourceZ[1:-2]))
        AX0.extend(0.5*(Gamma[2:-1] + Gamma[1:-2])*(SourceX[2:-1] - SourceX[1:-2]))
        AY0.extend(0.5*(Gamma[2:-1] + Gamma[1:-2])*(SourceY[2:-1] - SourceY[1:-2]))
        AZ0.extend(0.5*(Gamma[2:-1] + Gamma[1:-2])*(SourceZ[2:-1] - SourceZ[1:-2]))
        S0.extend(dy*SmoothingRatio)
        Kinematics = J.get(LiftingLine, '.Kinematics')
        Urel = U0 - Kinematics['VelocityTranslation']
        Dpsi = np.arctan2(h, W.getLength(LiftingLine))*180./np.pi*(Kinematics['RPM'][0] != 0)
        if not Kinematics['RightHandRuleRotation']: Dpsi *= -1
        T._rotate(Source, Kinematics['RotationCenter'], Kinematics['RotationAxis'], -Dpsi)
        T._translate(Source, Urel/(np.linalg.norm(Urel, axis = 0) + 1e-10)*h)

        SourceX = I.getValue(I.getNodeFromName(Source, 'CoordinateX'))
        SourceY = I.getValue(I.getNodeFromName(Source, 'CoordinateY'))
        SourceZ = I.getValue(I.getNodeFromName(Source, 'CoordinateZ'))
        Gamma   = I.getValue(I.getNodeFromName(Source, 'Gamma'))
        dy = 0.5*np.linalg.norm(np.vstack([SourceX[2:] - SourceX[:-2], SourceY[2:] - \
                                           SourceY[:-2], SourceZ[2:] - SourceZ[:-2]]), axis = 0)
        X.extend(SourceX[1:-1])
        Y.extend(SourceY[1:-1])
        Z.extend(SourceZ[1:-1])
        AX.extend([0.]*(len(SourceX) - 2))
        AY.extend([0.]*(len(SourceX) - 2))
        AZ.extend([0.]*(len(SourceX) - 2))
        S.extend(dy*SmoothingRatio)

    V.addParticlesToTree(_tL, NewX = X0, NewY = Y0, NewZ = Z0, NewAX = AX0, NewAY = AY0,
                                NewAZ = AZ0,  NewSigma = S0, Offset = 0, ExtendAtTheEnd = False)
    V.addParticlesToTree(_tL, NewX = X, NewY = Y, NewZ = Z, NewAX = AX, NewAY = AY,
                            NewAZ = AZ,  NewSigma = S, Offset = len(X0), ExtendAtTheEnd = False)
    Nu, Cvisq = J.getVars(V.getFreeParticles(_tL), ['Nu', 'Cvisq'])
    Nu[:len(X0)] = 0.
    Nu[len(X0):] = Parameters['FluidParameters']['KinematicViscosity']
    Cvisq[:len(X0)] = 0.
    Cvisq[len(X0):] = Parameters['ModelingParameters']['EddyViscosityConstant']
    LL.computeGeneralLoadsOfLiftingLine(_tLL,
                                            UnsteadyData={'IterationNumber'         : 0,
                                                          'Time'                    : 0,
                                                          'CirculationSubiterations': 0,
                                                          'CirculationError'        : 0},
                                            UnsteadyDataIndependentAbscissa = 'IterationNumber')

def setShedParticleStrength(Dir, VeciX, VeciY, VeciZ, Ramp, SheddingDistance, FilamentLength, ax,
    ay, az, Sources, SourcesM1, NumberParticlesShedPerStation, NumberOfLiftingLineSources,
    NumberOfSources, TimeShed, frozenLiftingLine):
    '''
    Updates the strength of the bound, the first row and the shed particles by the Lifting
    Lines.
    
    Parameters
    ----------
        Dir : :py:class:`list` or numpy.ndarray of :py:class:`int`
            Gives the rotation of the Lifting Lines, either positive +1, or negative (-1).

        VeciX : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Tangential vector to the Lifting Lines component along the x axis.

        VeciY : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Tangential vector to the Lifting Lines component along the y axis.

        VeciZ : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Tangential vector to the Lifting Lines component along the z axis.

        Ramp : :py:class:`float`
            Initiale ramp for the particle strength.

        SheddingDistance : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Distance between the source stations on the Lifting Lines and the first row of
            shed particles at the previous TimeStep.

        FilamentLength : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Distance between the source stations on the Lifting Lines.

        ax : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Strength of the particles along the x axis.

        ay : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Strength of the particles along the y axis.

        az : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Strength of the particles along the z axis.

        Sources : :py:class:`list` or numpy.ndarray of Zone
            Sources on the Lifting Lines from where the particles are shed.

        SourcesM1 : :py:class:`list` or numpy.ndarray of Zone
            Sources on the Lifting Lines at the previous TimeStep from where the particles are
            shed.

        NumberParticlesShedPerStation : :py:class:`list` or numpy.ndarray of :py:class:`int`
            Number of particles shed per source station (if any).

        NumberOfLiftingLineSources : :py:class:`int`
            Total number of sources station on the Lifting Lines.

        NumberOfSources : :py:class:`int`
            Total number of embedded bound Lifting Line, BEM and Interface particles.

        TimeShed : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Time from the last particle sheding for each source station.

        frozenLiftingLine : :py:class:`list` or numpy.ndarray of :py:class:`int`
            List of the Lifting Lines not to be updated.
    '''
    if NumberParticlesShedPerStation != []:
        SourcesBase = I.newCGNSBase('Sources', cellDim=1, physDim=3)
        SourcesBase[2] = I.getZones(Sources)
        SourcesBaseM1 = I.newCGNSBase('SourcesM1', cellDim=1, physDim=3)
        SourcesBaseM1[2] = I.getZones(SourcesM1)
        flag = np.array([1]*len(Dir), order = 'F', dtype = np.int32)
        for i in frozenLiftingLine: flag[i] = 0
        V.shed_particles_from_lifting_lines(Dir, VeciX, VeciY, VeciZ, Ramp, SheddingDistance,
              FilamentLength, ax, ay, az, SourcesBase, SourcesBaseM1, NumberParticlesShedPerStation,
                                        NumberOfLiftingLineSources, NumberOfSources, TimeShed, flag)

def relaxCirculationAndGetImbalance(GammaOld = [], GammaRelax = [0.], Sources = [],
    GammaError = [], GammaDampening = []):
    '''
    Relax the circulation and returns the error with the previous circulation.

    Parameters
    ----------
        GammaOld : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Circulation on each source station at the previous iteration step.

        GammaRelax : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Relaxation factor.

        Sources : :py:class:`list` or numpy.ndarray of Zone
            Sources on the stations of the Lifting Lines.
    Returns
    -------
        GammaError : :py:class:`float`
            Maximum relative error of the circulation between the sources stations of the
            previous and current iteration.
    '''
    for i in range(len(Sources)):
        GammaNew, = J.getVars(Sources[i], ['Gamma'])
        err = max(abs(GammaNew - GammaOld[i]))/max(1e-12, np.mean(abs(GammaNew)))
        if GammaError[i] < err:
            GammaDampening[i] = max(GammaDampening[i]*0.85, 0.1)
        GammaError[i] = err
        krelax = GammaDampening[i]*GammaRelax[i]
        GammaOld[i][:] = GammaNew[:] = (1. - krelax)*GammaOld[i] + krelax*GammaNew

    return np.array(GammaError, dtype = np.float64, order = 'F')

def updateLiftingLinesCirculation(tLL):
    for LiftingLine in I.getZones(V.getTrees([tLL], ['LiftingLines'])):
        Correc_n   = J.get(LiftingLine, '.Component#Info')['Corrections3D']
        Kinematics = J.get(LiftingLine, '.Kinematics')
        SweepCorrection    = Correc_n['Sweep']
        DihedralCorrection = Correc_n['Dihedral']
        RotationCenter = Kinematics['RotationCenter']
        TorqueOrigin   = Kinematics['TorqueOrigin']
        RotationAxis   = Kinematics['RotationAxis']
        dir            = 1 if Kinematics['RightHandRuleRotation'] else -1
        RPM            = Kinematics['RPM']

        u, ChordCorr, Chord, Cl, Gamma, DihedralCorr = J.getVars(LiftingLine,
                        ['VelocityMagnitudeLocal', 'ChordVirtualWithSweep', 'Chord', 'Cl', 'Gamma',
                                                                                'DihedralAngleDeg'])
        Flux = 0.5*u*Cl
        if SweepCorrection: Flux *= ChordCorr
        else:               Flux *= Chord
        if DihedralCorrection: Flux *= np.cos(np.deg2rad(DihedralCorr))
        Gamma[:] = Flux

def updateLiftingLinesVelocitiesAndAoA(tLL):
    for LiftingLine in I.getZones(V.getTrees([tLL], ['LiftingLines'])):
        Correc_n = J.get(LiftingLine, '.Component#Info')['Corrections3D']
        Conditions = J.get(LiftingLine, '.Conditions')
        SweepCorrection    = Correc_n['Sweep']
        DihedralCorrection = Correc_n['Dihedral']
        T0  = Conditions['Temperature']
        Rho = Conditions['Density']
        U0  = Conditions['VelocityFreestream']

        Urelx, Urely, Urelz, Uix, Uiy, Uiz, Upx, Upy, Upz, Ukx, Uky, Ukz, chordX, chordY, chordZ, \
        thickX, thickY, thickZ, Mach, Reynolds, u, AoA, Chord, ChordSweep, SweepCorr, DihedralCorr \
                    = J.getVars(LiftingLine, V.vectorise(['VelocityRelative', 'VelocityInduced', \
                           'VelocityPerturbation', 'VelocityKinematic', 'Chordwise', 'Thickwise']) \
                                  + ['Mach', 'Reynolds', 'VelocityMagnitudeLocal', 'AoA', 'Chord', \
                                      'ChordVirtualWithSweep', 'SweepAngleDeg', 'DihedralAngleDeg'])

        Urelx[:] = Uix + Upx + U0[0] - Ukx
        Urely[:] = Uiy + Upy + U0[1] - Uky
        Urelz[:] = Uiz + Upz + U0[2] - Ukz
        Uchord = Urelx*chordX + Urely*chordY + Urelz*chordZ
        Uthick = Urelx*thickX + Urely*thickY + Urelz*thickZ

        if SweepCorrection:    Uchord *= np.cos(np.deg2rad(SweepCorr))
        if DihedralCorrection: Uthick *= np.cos(np.deg2rad(DihedralCorr))
        # Updating the Angle of Attack considering the new velocity components.
        AoA[:] = np.rad2deg(np.arctan2(Uthick, Uchord))
        u[:] = np.linalg.norm(np.vstack([Uthick, Uchord]), axis = 0)
        Mach[:] = u/np.sqrt(1.4*287.058*T0)
        Mu = 1.711e-5*np.sqrt(T0/273.)*(1. + 110.4/273.)/(1. + 110.4/T0)
        if SweepCorrection: Reynolds[:] = Rho*u*Chord/Mu    
        else:               Reynolds[:] = Rho*u*ChordSweep/Mu

def updateLiftingLinesVelocities(tLL):
    for LiftingLine in I.getZones(V.getTrees([tLL], ['LiftingLines'])):
        Conditions = J.get(LiftingLine, '.Conditions')
        U0  = Conditions['VelocityFreestream']

        Urelx, Urely, Urelz, Uix, Uiy, Uiz, Upx, Upy, Upz, Ukx, Uky, Ukz = J.getVars(\
                                LiftingLine, V.vectorise(['VelocityRelative', 'VelocityInduced', \
                                                      'VelocityPerturbation', 'VelocityKinematic']))

        Urelx[:] = Uix + Upx + U0[0] - Ukx
        Urely[:] = Uiy + Upy + U0[1] - Uky
        Urelz[:] = Uiz + Upz + U0[2] - Ukz

def moveAndUpdateLiftingLines(tL = [], tLL = [], tP = [], dt = 0.):
    '''
    Moves the Lifting Lines with their kinematic velocity and updates their local velocity
    accordingly.

    Parameters
    ----------
        t : Tree
            Containes a zone of particles named 'Particles'.

        tLL : Tree
            Lifting Lines.

        dt : :py:class:`float`
            TimeStep.
    '''
    _tL, _tLL, _tP = V.getTrees([tL, tLL, tP], ['Particles', 'LiftingLines', 'Perturbation'])
    if not _tLL: return
    LL.computeKinematicVelocity(_tLL)
    LL.moveLiftingLines(_tLL, dt)
    if _tP: V.extractperturbationField(Targets = _tLL, tL = _tL, tP = _tP)
    updateLiftingLinesVelocities(_tLL)

def initialiseShedParticles(tL = [], tLL = [], Sources = [], Ramp = 1.,
    SmoothingRatio = 2., NumberOfLLSources = 0., NumberOfSources = 0., it = []):
    '''
    Initialises the particles to update and shed from the Lifting Lines.

    Parameters
    ----------
        tL : Tree
            Lagrangian field.

        tLL : Tree
            Lifting Lines.

        Sources : :py:class:`list` or numpy.ndarray of Zone
            Sources on the Lifting Lines from where the particles are shed.

        Ramp : :py:class:`float`
            Initial Ramp to decrease the strength of the particles.

        SmoothingRatio : :py:class:`float`
            Overlapping ratio of the VPM particles.

        NumberOfLLSources : :py:class:`int`
            Total number of source stations on the Lifting Lines.

        NumberOfSources : :py:class:`int`
            Total number of embedded bound Lifting Line, BEM and Interface particles.
    '''
    _tL, _tLL = V.getTrees([tL, tLL], ['Particles', 'LiftingLines'])
    if not _tL or not _tLL: return [None]*9

    frozenLiftingLines = []
    Particles = V.getFreeParticles(_tL)
    LiftingLines = V.getLiftingLines(_tLL)
    px, py, pz = J.getxyz(Particles)
    apx, apy, apz, nup = J.getVars(Particles, ['Alpha' + v for v in 'XYZ'] + ['Nu'])
    pos = NumberOfSources
    index = 0
    for Source, LiftingLine in zip(Sources, LiftingLines):#checks if their is enough space for particles to be shed for this LL
        h = I.getValue(I.getNodeFromName(LiftingLine, 'LocalResolution'))
        sx, sy, sz = J.getxyz(Source)
        dy = 0
        for i in range(1, len(sx) - 1):
            dy += np.linalg.norm(np.array([sx[i]  , sy[i]  , sz[i]]) - 
                                               np.array([px[pos], py[pos],  pz[pos]]), axis = 0)
            pos += 1

        if dy/(len(sx) - 2) < h*0.85: frozenLiftingLines += [index]#on average there is not enough space to shed particles

        index += 1

    NewX, NewY, NewZ = [], [], []
    NewAX, NewAY, NewAZ = [], [], []
    NewS, NewNu = [], []
    Dir = []
    pos = 0
    index = 0
    deleteFlag = np.array([False]*len(apx))
    for Source, LiftingLine in zip(Sources, LiftingLines):#bound particles
        sx, sy, sz = J.getxyz(Source)
        NewX.extend(0.5*(sx[2:-1] + sx[1:-2]))
        NewY.extend(0.5*(sy[2:-1] + sy[1:-2]))
        NewZ.extend(0.5*(sz[2:-1] + sz[1:-2]))
        NewS.extend(SmoothingRatio*np.linalg.norm(np.vstack([sx[2:-1] - sx[1:-2], sy[2:-1] - \
                                                     sy[1:-2], sz[2:-1] - sz[1:-2]]), axis = 0))
        NewNu.extend([0]*(len(sx) - 3))
        if index in frozenLiftingLines:
            NewAX.extend(apx[pos: pos + len(sx) - 3])
            NewAY.extend(apy[pos: pos + len(sx) - 3])
            NewAZ.extend(apz[pos: pos + len(sx) - 3])
        else:
            NewAX.extend([0]*(len(sx) - 3))
            NewAY.extend([0]*(len(sx) - 3))
            NewAZ.extend([0]*(len(sx) - 3))

        deleteFlag[pos: pos + len(sx) - 3] = True
        pos += len(sx) - 3
        index += 1
        Dir += [1 if I.getValue(I.getNodeFromName(LiftingLine,'RightHandRuleRotation')) else -1]

    pos = NumberOfSources
    index = 0
    for Source, LiftingLine in zip(Sources, LiftingLines):#first row of shed particles
        sx, sy, sz = J.getxyz(Source)
        for i in range(1, len(sx) - 1):
            NewS += [0.5*SmoothingRatio*np.linalg.norm([sx[i + 1] - sx[i - 1],\
                                           sy[i + 1] - sy[i - 1], sz[i + 1] - sz[i - 1]], axis = 0)]
        
        NewNu.extend(nup[pos: pos + len(sx) - 2])
        if index in frozenLiftingLines:
            NewX.extend(px[pos: pos + len(sx) - 2])
            NewY.extend(py[pos: pos + len(sx) - 2])
            NewZ.extend(pz[pos: pos + len(sx) - 2])
            NewAX.extend(apx[pos: pos + len(sx) - 2])
            NewAY.extend(apy[pos: pos + len(sx) - 2])
            NewAZ.extend(apz[pos: pos + len(sx) - 2])
            deleteFlag[pos: pos + len(sx) - 2] = True
        else:
            NewX.extend(sx[1: len(sx) - 1])
            NewY.extend(sy[1: len(sx) - 1])
            NewZ.extend(sz[1: len(sx) - 1])
            NewAX.extend([0.]*(len(sx) - 2))
            NewAY.extend([0.]*(len(sx) - 2))
            NewAZ.extend([0.]*(len(sx) - 2))

        pos += len(sx) - 2
        index += 1

    ParticlesShedPerStation = []
    VeciX, VeciY, VeciZ, = [], [], []
    SheddingDistance, FilamentLength = [], []
    index = 0
    pos = NumberOfSources
    for Source, LiftingLine in zip(Sources, LiftingLines):#remaining particles shed in the wake
        h = I.getValue(I.getNodeFromName(LiftingLine, 'LocalResolution'))
        sx, sy, sz = J.getxyz(Source)
        if index not in frozenLiftingLines:
            for i in range(1, len(sx) - 1):
                xm = np.array([sx[i], sy[i], sz[i]])
                vecj = np.array([px[pos], py[pos], pz[pos]]) - xm
                VeciX += [0.5*(sx[i + 1] - sx[i - 1])]
                VeciY += [0.5*(sy[i + 1] - sy[i - 1])]
                VeciZ += [0.5*(sz[i + 1] - sz[i - 1])]
                dy = np.linalg.norm(vecj, axis = 0)
                dx = np.linalg.norm([VeciX[-1], VeciY[-1], VeciZ[-1]], axis = 0)
                SheddingDistance += [dy]
                FilamentLength += [dx]
                Nshed = max(int(round(dy/h - 0.95)), 0)
                for j in range(Nshed):
                    NewX += [xm[0] + (j + 1)/(Nshed + 1)*vecj[0]]
                    NewY += [xm[1] + (j + 1)/(Nshed + 1)*vecj[1]]
                    NewZ += [xm[2] + (j + 1)/(Nshed + 1)*vecj[2]]

                NewAX += [0.]*Nshed
                NewAY += [0.]*Nshed
                NewAZ += [0.]*Nshed
                NewS += [SmoothingRatio*dx]*Nshed
                NewNu += [nup[pos]]*Nshed

                ParticlesShedPerStation += [Nshed]
                pos += 1
        else:
            pos += len(sx) - 2

        index += 1

    ParticlesShedPerStation = np.array(ParticlesShedPerStation, dtype=np.int32, order = 'F')
    Dir = np.array(Dir, dtype = np.int32, order = 'F')
    frozenLiftingLines = np.array(frozenLiftingLines, dtype = np.int32, order = 'F')
    SheddingDistance = np.array(SheddingDistance, dtype = np.float64, order = 'F')
    FilamentLength = np.array(FilamentLength, dtype = np.float64, order = 'F')
    VeciX = np.array(VeciX, dtype = np.float64, order = 'F')/FilamentLength
    VeciY = np.array(VeciY, dtype = np.float64, order = 'F')/FilamentLength
    VeciZ = np.array(VeciZ, dtype = np.float64, order = 'F')/FilamentLength
    V.delete(Particles, deleteFlag)
    V.addParticlesToTree(Particles, NewX = NewX[:NumberOfLLSources],
        NewY = NewY[:NumberOfLLSources], NewZ = NewZ[:NumberOfLLSources],
        NewAX = NewAX[:NumberOfLLSources], NewAY = NewAY[:NumberOfLLSources],
        NewAZ = NewAZ[:NumberOfLLSources], 
        NewSigma = NewS[:NumberOfLLSources], Offset = 0, ExtendAtTheEnd = False)
    V.addParticlesToTree(Particles, NewX = NewX[NumberOfLLSources:],
        NewY = NewY[NumberOfLLSources:], NewZ = NewZ[NumberOfLLSources:],
        NewAX = NewAX[NumberOfLLSources:], NewAY = NewAY[NumberOfLLSources:],
        NewAZ = NewAZ[NumberOfLLSources:], 
        NewSigma = NewS[NumberOfLLSources:], Offset = NumberOfSources, ExtendAtTheEnd = False)

    nup = J.getVars(Particles, ['Nu'])[0]
    nup[:NumberOfLLSources] = NewNu[:NumberOfLLSources]
    nup[NumberOfSources:NumberOfSources + len(NewNu) -NumberOfLLSources] = NewNu[NumberOfLLSources:]

    return frozenLiftingLines, ParticlesShedPerStation, Dir, VeciX, VeciY, VeciZ, \
                                     SheddingDistance, FilamentLength, len(NewX[NumberOfLLSources:])

def shedVorticitySourcesFromLiftingLines(tL = [], tLL = [], tP = []):
    '''
    Updates the bound and first row of particles and shed particles from the Lifting Lines.

    Parameters
    ----------
        t : Tree
            Containes a zone of particles named 'Particles'.
    Returns
    -------
        IterationInfo : :py:class:`dict`
            VPM solver information on the current iteration.
    '''
    timeLL = J.tic()
    _tL, _tLL, _tP = V.getTrees([tL, tLL, tP], ['Particles', 'LiftingLines', 'Perturbation'])
    if not _tLL or not _tL: return {}
    #for LiftingLine in I.getZones(tLL):
    #    Gamma, GammaM1 = J.getVars(LiftingLine, ['Gamma', 'GammaM1'])
    #    GammaM1[:] = Gamma[:].copy()

    Particles = V.getFreeParticles(_tL)
    Np0 = V.getParticlesNumber(Particles)
    SmoothingRatio, dt, time, it, Ramp, KinematicViscosity, EddyViscosityConstant, \
    NumberOfLLSources, NumberOfBEMSources, NumberOfCFDSources = V.getParameters(_tL, \
             ['SmoothingRatio', 'TimeStep', 'Time', 'CurrentIteration', 'StrengthRampAtbeginning', \
                      'KinematicViscosity', 'EddyViscosityConstant', 'NumberOfLiftingLineSources', \
                                                        'NumberOfBEMSources', 'NumberOfCFDSources'])
    NumberOfSources = NumberOfLLSources[0] + NumberOfCFDSources[0] + NumberOfBEMSources[0]
    NumberOfLLSources = NumberOfLLSources[0]
    Ramp = np.sin(min((it[0] + 1)/Ramp[0], 1.)*np.pi/2.)
    moveAndUpdateLiftingLines(_tL, _tLL, _tP, dt[0])

    SourcesDistribution = [I.getNodeFromName(LiftingLine, 'SourcesDistribution')[1] for \
                                                                    LiftingLine in I.getZones(_tLL)]
    Sources = LL.buildVortexParticleSourcesOnLiftingLine(_tLL, AbscissaSegments = \
                                                       SourcesDistribution, IntegralLaw = 'linear')
    TimeShed, GammaThreshold, GammaRelax, MaxIte = [], [], [], 0
    for LiftingLine in I.getZones(_tLL):
        LLParameters = J.get(LiftingLine, '.VPM#Parameters')
        TimeShed += [LLParameters['TimeSinceLastShedding'][0]]
        GammaThreshold += [LLParameters['CirculationThreshold'][0][0]]
        GammaRelax += [LLParameters['CirculationRelaxationFactor'][0][0]]
        MaxIte = max(MaxIte, LLParameters['MaxLiftingLineSubIterations'][0][0])

    TimeShed = np.array(TimeShed, dtype = np.float64, order = 'F')
    SourcesM1 = [I.copyTree(Source) for Source in Sources]
    GammaOld = [I.getNodeFromName3(Source, 'Gamma')[1] for Source in Sources]


    frozenLiftingLines, ParticlesShedPerStation, Dir, VeciX, VeciY, VeciZ, SheddingDistance, \
                     FilamentLength, Nshed = initialiseShedParticles(_tL, _tLL, Sources, Ramp,
                                       SmoothingRatio[0], NumberOfLLSources, NumberOfSources, it[0])
    SheddingLiftingLines = I.getZones(I.copyRef(_tLL))
    for index in frozenLiftingLines[::-1]:
        SheddingLiftingLines.pop(index)
        SourcesDistribution.pop(index)
        GammaThreshold.pop(index)
        GammaRelax.pop(index)
        GammaOld.pop(index)

    GammaError = [np.inf]*len(SheddingLiftingLines)
    GammaDampening = [1.]*len(SheddingLiftingLines)
    tLL_shed = C.newPyTree(['LiftingLines', SheddingLiftingLines])

    GammaThreshold = np.array(GammaThreshold, dtype = np.float64, order = 'F')
    GammaRelax = np.array(GammaRelax, dtype = np.float64, order = 'F')

    ax, ay, az, s = J.getVars(Particles, ['Alpha' + v for v in 'XYZ'] + ['Sigma'])
    WakeInducedVelocity = extractWakeInducedVelocityOnLiftingLines(_tL, tLL_shed, Nshed)
    ni = 0
    for _ in range(MaxIte):
        setShedParticleStrength(Dir, VeciX, VeciY, VeciZ, Ramp, SheddingDistance, FilamentLength,
                         ax, ay, az, Sources, SourcesM1, ParticlesShedPerStation, NumberOfLLSources,
                                                      NumberOfSources, TimeShed, frozenLiftingLines)
        BoundAndShedInducedVelocity = extractBoundAndShedVelocityOnLiftingLines(_tL,
                                                                        tLL_shed, Nshed)
        
        setLiftingLinesInducedVelocity(tLL_shed,
                                                  WakeInducedVelocity + BoundAndShedInducedVelocity)
        updateLiftingLinesVelocitiesAndAoA(tLL_shed)
        LL._applyPolarOnLiftingLine(tLL_shed, V.PolarsInterpolators[0], ['Cl'])
        updateLiftingLinesCirculation(tLL_shed)
        Sources = LL.buildVortexParticleSourcesOnLiftingLine(tLL_shed,
                                    AbscissaSegments = SourcesDistribution, IntegralLaw = 'linear')
        GammaError = relaxCirculationAndGetImbalance(GammaOld, GammaRelax, Sources, GammaError,
                                                                                     GammaDampening)
        ni += 1
        if (GammaError < GammaThreshold).all(): break

        for index in frozenLiftingLines: Sources.insert(index, SourcesM1[index])

    #if (GammaError < GammaThreshold).any():
    #    safeLiftingLinesIterations(_tL, tLL_shed, frozenLiftingLines, SourcesDistribution, GammaThreshold, GammaRelax, Dir, VeciX, VeciY, VeciZ, SheddingDistance, ax, ay, az)

    Nu, Cvisq = J.getVars(Particles, ['Nu', 'Cvisq'])
    offset = NumberOfSources + Nshed
    Nu[:NumberOfLLSources] = 0.
    Cvisq[:NumberOfLLSources] = 0.
    Cvisq[NumberOfSources: NumberOfSources + Nshed] = EddyViscosityConstant

    for LiftingLine in I.getZones(tLL_shed):
        TimeShed = I.getNodeFromName(LiftingLine, 'TimeSinceLastShedding')
        TimeShed[1][0] = 0
    #for LiftingLine in I.getZones(_tLL):
    #    Gamma, GammaM1, dGammadt = J.getVars(LiftingLine, ['Gamma', 'GammaM1', 'dGammadt'])
    #    dGammadt[:] = (Gamma[:] - GammaM1[:])/dt

    LL.assembleAndProjectVelocities(tLL_shed)
    LL._applyPolarOnLiftingLine(tLL_shed, V.PolarsInterpolators[0], ['Cl', 'Cd', 'Cm'])
    if len(GammaError) == 0: GammaError = np.array([0])
    LL.computeGeneralLoadsOfLiftingLine(_tLL,
            UnsteadyData = {'IterationNumber': it[0],
                            'Time': time[0],
                            'CirculationSubiterations': ni,
                            'CirculationError': np.max(GammaError)},
                            UnsteadyDataIndependentAbscissa = 'IterationNumber')
    IterationInfo = {'Circulation error' : np.max(GammaError),
                     'Number of sub-iterations (LL)' : ni,
                     'Number of shed particles' : V.getParticlesNumber(Particles) - Np0,
                     'Lifting Line time' : J.tic() - timeLL}
    return IterationInfo
