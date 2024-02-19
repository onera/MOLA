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

import sys
import os
import scipy
import numpy as np
import VULCAINS.vulcains as V
import Converter.PyTree as C
import Geom.PyTree as D
import Converter.Internal as I
import Generator.PyTree as G
import Transform.PyTree as T
import Connector.PyTree as CX
import Post.PyTree as P
import CPlot.PyTree as CPlot

from . import FreeWakeParticles as VPM

from .. import LiftingLine as LL
from .. import Wireframe as W
from .. import InternalShortcuts as J
from .. import ExtractSurfacesProcessor as ESP


####################################################################################################
####################################################################################################
########################################### Lifting Lines ##########################################
####################################################################################################
####################################################################################################
def setTimeStepFromShedParticles(t = [], LiftingLines = [], NumberParticlesShedAtTip = 5.):
    '''
    Sets the VPM TimeStep so that the user-given number of particles are shed at the tip of the
    fastest moving Lifting Line.

    Parameters
    ----------
        t: Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        LiftingLines : Zone, :py:class:`list` or numpy.ndarray of Zone
            Containes the Lifting Lines.

        NumberParticlesShedAtTip : :py:class:`int`
            Number of particles to shed per TimeStep.
    '''
    if not LiftingLines: raise AttributeError('The time step is not given and can not be \
                 computed without a Lifting Line. Specify the time step or give a Lifting Line')
    LL.computeKinematicVelocity(LiftingLines)
    LL.assembleAndProjectVelocities(LiftingLines)

    if type(t) == dict:
        Resolution = t['Resolution']
        U0         = t['VelocityFreestream']
    else:
        Particles  = VPM.pickParticlesZone(t)
        Resolution = I.getNodeFromName(Particles, 'Resolution')[1][0]
        U0         = I.getValue(I.getNodeFromName(Particles, 'VelocityFreestream'))

    Urelmax = 0.
    for LiftingLine in LiftingLines:
        Ukin = np.vstack(J.getVars(LiftingLine, ['VelocityKinematic' + i for i in 'XYZ']))
        ui   = np.vstack(J.getVars(LiftingLine, ['VelocityInduced' + i for i in 'XYZ']))
        Urel = U0 + ui.T - Ukin.T
        Urel = max([np.linalg.norm(urel, axis = 0) for urel in Urel])
        if (Urelmax < Urel): Urelmax = Urel

    if Urelmax == 0:
        raise ValueError('Maximum velocity is zero. Set non-zero kinematic or freestream \
                                                                                     velocity.')

    if type(t) == dict:
        t['TimeStep'] = NumberParticlesShedAtTip*Resolution/Urel
    else:
        VPMParameters = VPM.getVPMParameters(t)
        VPMParameters['TimeStep'] = NumberParticlesShedAtTip*Resolution/Urel

def setTimeStepFromBladeRotationAngle(t = [], LiftingLines = [], BladeRotationAngle = 5.):
    '''
    Sets the VPM TimeStep so that the fastest moving Lifting Line rotates by the user-given
    angle per TimeStep.
    .
    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        LiftingLines : Zone, :py:class:`list` or numpy.ndarray of Zone
            Containes the Lifting Lines.

        BladeRotationAngle : :py:class:`float`
            Blade rotation angle per TimeStep.
    '''
    if not LiftingLines: raise AttributeError('The time step is not given and can not be \
                 computed without a Lifting Line. Specify the time step or give a Lifting Line')

    RPM = 0.
    for LiftingLine in LiftingLines:
        RPM = max(RPM, I.getValue(I.getNodeFromName(LiftingLine, 'RPM')))
    
    if type(t) == dict:
        t['TimeStep'] = 1./6.*BladeRotationAngle/RPM
    else:
        VPMParameters = VPM.getVPMParameters(t)
        VPMParameters['TimeStep'] = 1./6.*BladeRotationAngle/RPM

TimeStepFunction_str2int = {'setTimeStepFromBladeRotationAngle':
setTimeStepFromBladeRotationAngle, 'shed': setTimeStepFromShedParticles,
'BladeRotationAngle': setTimeStepFromBladeRotationAngle, 'setTimeStepFromShedParticles':
setTimeStepFromShedParticles, 'ShedParticles': setTimeStepFromShedParticles, 'Angle':
setTimeStepFromBladeRotationAngle, 'angle': setTimeStepFromBladeRotationAngle, 'Shed':
setTimeStepFromShedParticles}

def setMinNbShedParticlesPerLiftingLine(LiftingLines = [], Parameters = {},
    NumberParticlesShedAtTip = 5):
    '''
    Sets the minimum number of shedding station on the Lifting Line(s) so that the fastest
    moving Lifting Line sheds the user-given number of particles at its tip at each TimeStep.
    .
    Parameters
    ----------
        LiftingLines : Zone, :py:class:`list` or numpy.ndarray of Zone
            Containes the Lifting Lines.

        Parameters : :py:class:`dict` of :py:class:`float`
            Parameters containing the VPM parameters.

        NumberParticlesShedAtTip : :py:class:`int`
            Blade rotation angle per TimeStep.
    '''
    LL.computeKinematicVelocity(LiftingLines)
    LL.assembleAndProjectVelocities(LiftingLines)
    Urelmax = 0.
    L = 0.
    flag = False
    if 'VelocityFreestream' not in Parameters: flag = True
    else: U0 = Parameters['VelocityFreestream']
    for LiftingLine in LiftingLines:
        Ukin = np.array(J.getVars(LiftingLine, ['VelocityKinematicX',
                                                   'VelocityKinematicY', 'VelocityKinematicZ']))
        ui = np.array(J.getVars(LiftingLine, ['VelocityInducedX',
                                                       'VelocityInducedY', 'VelocityInducedZ']))
        if flag:
            U0 = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityFreestream'))
        Urel= np.linalg.norm(U0 + ui - Ukin, axis = 0)
        if (Urelmax < Urel):
            Urelmax = Urel
            L = np.minimum(L, W.getLength(LiftingLine))
    Parameters['MinNbShedParticlesPerLiftingLine'] = int(round(2. + NumberParticlesShedAtTip*L/\
                                                                   Urel/Parameters['TimeStep']))

def rotateLiftingLineSections(LiftingLines):
    Rotation = lambda v, theta, axis: \
                               scipy.spatial.transform.Rotation.from_rotvec(theta*axis).apply(v)
    for LiftingLine in I.getZones(LiftingLines):
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

def renameLiftingLineTree(LiftingLineTree = []):
    '''
    Checks the and updates the types of the nodes of the Lifting Line(s).
    .
    Parameters
    ----------
        LiftingLineTree : Tree
            Containes the Lifting Lines.
    '''
    if not LiftingLineTree: return

    TypeOfInput = I.isStdNode(LiftingLineTree)
    ERRMSG = J.FAIL + 'LiftingLines must be a tree, a list of bases or a list of zones' + J.ENDC
    if TypeOfInput == -1:# is a standard CGNS node
        if I.isTopTree(LiftingLineTree):
            LiftingLineBases = I.getBases(LiftingLineTree)
            if len(LiftingLineBases) == 1 and LiftingLineBases[0][0] == 'Base':
                LiftingLineBases[0][0] = 'LiftingLines'
        elif LiftingLineTree[3] == 'CGNSBase_t':
            LiftingLineBase = LiftingLineTree
            if LiftingLineBase[0] == 'Base': LiftingLineBase[0] = 'LiftingLines'
            LiftingLineTree = C.newPyTree([])
            LiftingLineTree[2] = [LiftingLineBase]
        elif LiftingLineTree[3] == 'Zone_t':
            LiftingLineTree = C.newPyTree(['LiftingLines', [LiftingLineTree]])
        else:
            raise AttributeError(ERRMSG)
    elif TypeOfInput == 0:# is a list of CGNS nodes
        if LiftingLineTree[0][3] == 'CGNSBase_t':
            LiftingLineTreeBases = I.getBases(LiftingLineTree)
            LiftingLineTree = C.newPyTree([])
            LiftingLineTree[2] = LiftingLineTreeBases
        elif LiftingLineTree[0][3] == 'Zone_t':
            LiftingLineTreeZones = I.getZones(LiftingLineTree)
            LiftingLineTree = C.newPyTree(['LiftingLines', LiftingLineTreeZones])
        else:
            raise AttributeError(ERRMSG)

    else:
        raise AttributeError(ERRMSG)

def updateParametersFromLiftingLines(LiftingLineTree = [], VPMParameters = {}):
    '''
    Checks the and updates VPM and Lifting Line parameters.
    .
    Parameters
    ----------
        LiftingLineTree : Tree
            Containes the Lifting Lines.

        VPMParameters : :py:class:`dict` of :py:class:`float`
            Containes VPM parameters for the VPM solver.
    '''
    LiftingLines = I.getZones(LiftingLineTree)
    VPMParameters['NumberOfLiftingLines'] = np.array([len(LiftingLines)], \
                                                                  dtype = np.int32, order = 'F')
    VPMParameters['NumberOfLiftingLineSources'] = np.zeros(1, dtype = np.int32, order = 'F')
    for LiftingLine in LiftingLines:
        LLParameters = J.get(LiftingLine, '.VPM#Parameters')
        VPMParameters['NumberOfLiftingLineSources'] += LLParameters['NumberOfParticleSources'] \
                                       - 1 + LLParameters['ParticleDistribution']['Symmetrical']

    if 'Resolution' not in VPMParameters and LiftingLines:
        hmax, hmin = -np.inf, np.inf
        for LiftingLine in LiftingLines:
            LLParameters = J.get(LiftingLine, '.VPM#Parameters')
            Resolution =  LLParameters['LocalResolution']
            hmax = max(Resolution, hmax)
            hmin = min(Resolution, hmin)
        
        VPMParameters['Resolution'] = np.array([hmin, hmax], dtype = np.float64, order = 'F')
    elif 'Resolution' in VPMParameters:
        VPMParameters['Resolution'] = np.array([min(VPMParameters['Resolution']),
                             max(VPMParameters['Resolution'])], dtype = np.float64, order = 'F')
    elif 'Resolution' not in VPMParameters: raise ValueError('The Resolution can not be \
                                  computed. The Resolution or a Lifting Line must be specified')

    if 'VelocityFreestream' not in VPMParameters: VPMParameters['VelocityFreestream'] = \
                                               np.array([0.]*3, dtype = np.float64, order = 'F')

    if 'TimeStep' not in VPMParameters and LiftingLines:
        setTimeStepFromShedParticles(VPMParameters, LiftingLines, NumberParticlesShedAtTip = 1.)

    elif 'TimeStep' not in VPMParameters: raise ValueError('The TimeStep can not be computed. \
                                              The TimeStep or a Lifting Line must be specified')

    VPMParameters['Sigma0'] = np.array(VPMParameters['Resolution']*\
                               VPMParameters['SmoothingRatio'], dtype = np.float64, order = 'F')
    VPMParameters['IterationCounter'] = np.array([0], dtype = np.int32, order = 'F')
    VPMParameters['StrengthRampAtbeginning'][0] = max(VPMParameters['StrengthRampAtbeginning'],\
                                                                                              1)
    VPMParameters['MinimumVorticityFactor'][0] = max(0.,VPMParameters['MinimumVorticityFactor'])

def setLiftingLinesInducedVelocity(LiftingLines, InducedVelocity):
    '''
    Sets the Lifting Line(s) induced velocities fields.

    Parameters
    ----------
        LiftingLines : Zone, :py:class:`list` or numpy.ndarray of Zone
            Containes the Lifting Lines.

        InducedVelocity : numpy.ndarray of 3 numpy.ndarray
            Induced Velocities [Ux, Uy, Uz].
    '''
    pos = 0
    for LiftingLine in I.getZones(LiftingLines):
        Ux, Uy, Uz = J.getVars(LiftingLine, ['VelocityInduced' + v for v in 'XYZ'])
        Nll = len(Ux)
        Ux[:] = InducedVelocity[0][pos: pos + Nll]
        Uy[:] = InducedVelocity[1][pos: pos + Nll]
        Uz[:] = InducedVelocity[2][pos: pos + Nll]
        pos += Nll

def resetLiftingLinesInducedVelocity(LiftingLines):
    '''
    Resets the Lifting Line(s) induced velocities fields to zero.

    Parameters
    ----------
        LiftingLines : Zone, :py:class:`list` or numpy.ndarray of Zone
            Containes the Lifting Lines.
    '''
    pos = 0
    for LiftingLine in I.getZones(LiftingLines):
        Ux, Uy, Uz = J.getVars(LiftingLine, ['VelocityInduced' + v for v in 'XYZ'])
        Nll = len(Ux)
        Ux[:] = 0.
        Uy[:] = 0.
        Uz[:] = 0.
        pos += Nll

def getLiftingLinesInducedVelocity(LiftingLines):
    '''
    Gets the Lifting Line(s) induced velocities fields to zero.

    Parameters
    ----------
        LiftingLines : Zone, :py:class:`list` or numpy.ndarray of Zone
            Containes the Lifting Lines.

    Parameters
    ----------
        InducedVelocity : numpy.ndarray of 3 numpy.ndarray
            Induced Velocities [Ux, Uy, Uz].
    '''
    UX, UY, UZ = [], [], []
    for LiftingLine in I.getZones(LiftingLines):
        ux, uy, uz = J.getVars(LiftingLine, ['VelocityInduced' + v for v in 'XYZ'])
        UX.extend(ux)
        UY.extend(uy)
        UZ.extend(uz)

    return np.array([UX, UY, UZ], order = 'F', dtype = np.float64)

def extractWakeInducedVelocityOnLiftingLines(t = [], LiftingLines = [], Nshed = 0):
    '''
    Gives the velocity induced by the particles in the wake, the hybrid domain and the bem
    particles on Lifting Line(s).

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        LiftingLines : Zone, :py:class:`list` or numpy.ndarray of Zone
            Containes the Lifting Lines.

    Returns
    -------
        WakeInducedVelocity : numpy.ndarray of 3 numpy.ndarray
            Induced Velocities [Ux, Uy, Uz].
    '''
    resetLiftingLinesInducedVelocity(LiftingLines)
    V.extract_wake_induced_velocity_on_lifting_lines(t, Nshed)
    return getLiftingLinesInducedVelocity(LiftingLines)

def extractBoundAndShedVelocityOnLiftingLines(t = [], LiftingLines = [], Nshed = 0):
    '''
    Gives the velocity induced by the bound and shed particles from the Lifting Line(s).

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        LiftingLines : Zone, :py:class:`list` or numpy.ndarray of Zone
            Containes the Lifting Lines.

        Nshed : :py:class:`int`
            Number of particles shed from the Lifting Line(s).

    Returns
    -------
        BoundAndShedInducedVelocity : numpy.ndarray of 3 numpy.ndarray
            Induced Velocities [Ux, Uy, Uz].
    '''
    resetLiftingLinesInducedVelocity(LiftingLines)
    V.extract_bound_and_shed_velocity_on_lifting_lines(t, Nshed)
    return getLiftingLinesInducedVelocity(LiftingLines)

def updateLiftingLines(LiftingLineTree = [], VPMParameters = {}, LiftingLineParameters = {}):
    '''
    Checks the and updates the parameters in the Lifting Line(s).
    .
    Parameters
    ----------
        LiftingLineTree : Tree
            Containes the Lifting Lines.

        VPMParameters : :py:class:`dict` of :py:class:`float`, :py:class:`int`, :py:class:`bool`
            and :py:class:`str`
            Containes VPM parameters for the VPM solver.

        LiftingLineParameters : :py:class:`dict` of :py:class:`float`, :py:class:`int` and
            :py:class:`str`
            Containes Lifting Line parameters for the Lifting Line(s).
    '''
    if not LiftingLineTree: return

    if 'TimeStep' in VPMParameters: dt = np.copy(VPMParameters['TimeStep'])
    else: dt = np.array([np.inf], order = 'F', dtype = np.float64)

    NLLmin = LiftingLineParameters['MinNbShedParticlesPerLiftingLine'][0]
    for LiftingLine in I.getZones(LiftingLineTree):
        span = W.getLength(LiftingLine)
        LLParameters = J.get(LiftingLine, '.VPM#Parameters')
        if not LLParameters: LLParameters = {}

        if 'IntegralLaw' in LLParameters:
            IntegralLaw = LLParameters['IntegralLaw']
        else:
            IntegralLaw = LiftingLineParameters['IntegralLaw']

        if 'ParticleDistribution' in LLParameters:
            ParticleDistribution = LLParameters['ParticleDistribution']
        elif 'ParticleDistribution' in LiftingLineParameters:
            ParticleDistribution = LiftingLineParameters['ParticleDistribution']
        else:
            ERRMSG = J.FAIL + ('Source particle distribution unspecified for ' + LiftingLine[0]
                                                                                 + '.') + J.ENDC
            raise AttributeError(ERRMSG)

        if 'Symmetrical' not in ParticleDistribution:
            if 'Symmetrical' in LLParameters['ParticleDistribution']:
                ParticleDistribution['Symmetrical'] = \
                                             LLParameters['ParticleDistribution']['Symmetrical']
            elif 'Symmetrical' in LiftingLineParameters['ParticleDistribution']:
                ParticleDistribution['Symmetrical'] = \
                                    LiftingLineParameters['ParticleDistribution']['Symmetrical']
            else:
                ERRMSG = J.FAIL + ('Symmetry of the source particle distribution unspecified '
                                                         'for ' + LiftingLine[0] + '.') + J.ENDC
                raise AttributeError(ERRMSG)

        if 'NumberOfParticleSources' in LLParameters:
            NumberOfParticleSources = max(LLParameters['NumberOfParticleSources'][0], NLLmin)
            LocalResolution = span/NumberOfParticleSources
        elif 'LocalResolution' in LLParameters:
            LocalResolution = LLParameters['LocalResolution'][0]
            NumberOfParticleSources = max(int(round(span/LocalResolution)), NLLmin)
            LocalResolution = span/NumberOfParticleSources
        else:
            NumberOfParticleSources = NLLmin
            LocalResolution = span/NumberOfParticleSources

        if ParticleDistribution['Symmetrical'] and NumberOfParticleSources%2:
            NumberOfParticleSources += 1
            LocalResolution = span/NumberOfParticleSources
            print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))
            print(f"{'||':>57}\r" + '|| Odd number of source particles on ' + LiftingLine[0] + ' dispite \
                                                                                 its symmetry.')
            print(f"{'||':>57}\r" + '|| Number of particle sources changed to ' + \
                                                             str(NumberOfParticleSources) + '.')
            print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))

        if ParticleDistribution['kind'] == 'ratio' or \
                            ParticleDistribution['kind'] == 'tanhOneSide' or \
                                                 ParticleDistribution['kind'] == 'tanhTwoSides':
            if 'FirstSegmentRatio' in ParticleDistribution:
                ParticleDistribution['FirstCellHeight'] = \
                                       ParticleDistribution['FirstSegmentRatio']*LocalResolution
            elif 'FirstSegmentRatio' in LiftingLineParameters['ParticleDistribution']:
                ParticleDistribution['FirstCellHeight'] = \
                             LiftingLineParameters['ParticleDistribution']['FirstSegmentRatio']\
                                                                                *LocalResolution
            else:
                ERRMSG = J.FAIL + ('FirstSegmentRatio unspecified for ' + LiftingLine[0] + \
                                  ' dispite ' + ParticleDistribution['kind'] + ' law.') + J.ENDC
                raise AttributeError(ERRMSG)

        if ParticleDistribution['kind'] == 'tanhTwoSides':
            if 'LastSegmentRatio' in ParticleDistribution:
                ParticleDistribution['LastCellHeight'] = \
                                        ParticleDistribution['LastSegmentRatio']*LocalResolution
            elif 'LastSegmentRatio' in LiftingLineParameters['ParticleDistribution']:
                ParticleDistribution['LastCellHeight'] = \
                              LiftingLineParameters['ParticleDistribution']['LastSegmentRatio']\
                                                                                *LocalResolution
            else:
                ERRMSG = J.FAIL + ('LastSegmentRatio unspecified for ' + LiftingLine[0] + \
                                  ' dispite ' + ParticleDistribution['kind'] + ' law.') + J.ENDC
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
            ParticleDistribution = ParticleDistribution,
            CirculationThreshold = np.array([CirculationThreshold], order = 'F',
                                                                            dtype = np.float64),
            CirculationRelaxationFactor = np.array([CirculationRelaxationFactor], order = 'F',
                                                                            dtype = np.float64),
            LocalResolution = np.array([LocalResolution], order = 'F', dtype = np.float64),
            MaxLiftingLineSubIterations = np.array([MaxLiftingLineSubIterations], order = 'F',
                                                                              dtype = np.int32),
            TimeSinceLastShedding = dt)

    LL.setConditions(LiftingLineTree, VelocityFreestream = VPMParameters['VelocityFreestream'],
                                      Density = VPMParameters['Density'],
                                      Temperature = VPMParameters['Temperature'])
    if 'RPM' in LiftingLineParameters: LL.setRPM(LiftingLineTree, LiftingLineParameters['RPM'])

    if 'VelocityTranslation' in LiftingLineParameters:
        for LiftingLine in I.getZones(LiftingLineTree):
            Kinematics = I.getNodeFromName(LiftingLine, '.Kinematics')
            VelocityTranslation = I.getNodeFromName(Kinematics, 'VelocityTranslation')
            VelocityTranslation[1] = np.array(LiftingLineParameters['VelocityTranslation'],
                                                                dtype = np.float64, order = 'F')

def initialiseParticlesOnLitingLine(t = [], LiftingLines = [], PolarsInterpolators = {},
    VPMParameters = {}):
    '''
    Initialises the bound particles embedded on the Lifting Line(s) and the first row of shed
    particles.
    
    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        LiftingLines : Zone, :py:class:`list` or numpy.ndarray of Zone
            Containes the Lifting Lines.

        PolarsInterpolators : Base or Zone or :py:class:`list` or numpy.ndarray of Base
            or Zone
            Containes the Polars for the sections of the Lifting Line(s).

        VPMParameters : :py:class:`dict` of :py:class:`float`, :py:class:`int`, :py:class:`bool` and :py:class:`str`
            Containes VPM parameters for the VPM solver.
    '''
    if not LiftingLines:
        VPMParameters['NumberOfLiftingLineSources'] = np.zeros(1, order = 'F', dtype = np.int32)
        return

    LL.computeKinematicVelocity(LiftingLines)
    LL.assembleAndProjectVelocities(LiftingLines)
    LL._applyPolarOnLiftingLine(LiftingLines, PolarsInterpolators, ['Cl', 'Cd', 'Cm'])
    LL.computeGeneralLoadsOfLiftingLine(LiftingLines)

    X0, Y0, Z0, AX0, AY0, AZ0, S0 = [], [], [], [], [], [], []
    X, Y, Z, AX, AY, AZ, S = [], [], [], [], [], [], []
    for LiftingLine in LiftingLines:
        #Gamma, GammaM1 = J.getVars(LiftingLine, ['Gamma', 'GammaM1'])
        #GammaM1[:] = Gamma[:]
        h = I.getValue(I.getNodeFromName(LiftingLine, 'LocalResolution'))
        L = W.getLength(LiftingLine)
        LLParameters = J.get(LiftingLine, '.VPM#Parameters')
        ParticleDistribution = LLParameters['ParticleDistribution']
        if ParticleDistribution['Symmetrical']:
            HalfStations = int(LLParameters['NumberOfParticleSources'][0]/2 + 1)
            SemiWing = W.linelaw(P1 = (0., 0., 0.), P2 = (L/2., 0., 0.), N = HalfStations,
                                                            Distribution = ParticleDistribution)# has to give +1 point because one point is lost with T.symetrize()
            WingDiscretization = J.getx(T.join(T.symetrize(SemiWing, (0, 0, 0), (0, 1, 0), \
                                                                          (0, 0, 1)), SemiWing))
            WingDiscretization += L/2.
            ParticleDistribution = WingDiscretization/L
        else:
            WingDiscretization = J.getx(W.linelaw(P1 = (0., 0., 0.), P2 = (L, 0., 0.),
                                                 N = LLParameters['NumberOfParticleSources'][0],
                                                           Distribution = ParticleDistribution))
            ParticleDistribution = WingDiscretization/L

        LLParameters = J.get(LiftingLine, '.VPM#Parameters')
        LLParameters['ParticleDistribution'] = ParticleDistribution
        J.set(LiftingLine, '.VPM#Parameters', **LLParameters)
        Source = LL.buildVortexParticleSourcesOnLiftingLine(LiftingLine,
                                                      AbscissaSegments = [ParticleDistribution],
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
        S0.extend(dy*VPMParameters['SmoothingRatio'][0])
        Kinematics = J.get(LiftingLine, '.Kinematics')
        Urel = VPMParameters['VelocityFreestream']-Kinematics['VelocityTranslation']
        Dpsi = np.arctan2(h, W.getLength(LiftingLine))*180./np.pi*(Kinematics['RPM'][0] != 0)
        #if (Dpsi == 0. and Urel == 0.): Urel = np.array(VPMParameters['Resolution'][0], VPMParameters['Resolution'][0], VPMParameters['Resolution'][0])
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
        S.extend(dy*VPMParameters['SmoothingRatio'][0])

    VPM.addParticlesToTree(t, NewX = X0, NewY = Y0, NewZ = Z0, NewAX = AX0, NewAY = AY0,
                                NewAZ = AZ0,  NewSigma = S0, Offset = 0, ExtendAtTheEnd = False)
    VPM.addParticlesToTree(t, NewX = X, NewY = Y, NewZ = Z, NewAX = AX, NewAY = AY,
                            NewAZ = AZ,  NewSigma = S, Offset = len(X0), ExtendAtTheEnd = False)
    Nu, Cvisq = J.getVars(VPM.pickParticlesZone(t), ['Nu', 'Cvisq'])
    Nu[:len(X0)] = 0.
    Nu[len(X0):] = VPMParameters['KinematicViscosity']
    Cvisq[:len(X0)] = 0.
    Cvisq[len(X0):] = VPMParameters['EddyViscosityConstant']
    LL.computeGeneralLoadsOfLiftingLine(LiftingLines,
                                            UnsteadyData={'IterationNumber'         : 0,
                                                          'Time'                    : 0,
                                                          'CirculationSubiterations': 0,
                                                          'CirculationError'        : 0},
                                            UnsteadyDataIndependentAbscissa = 'IterationNumber')

def setShedParticleStrength(Dir, VeciX, VeciY, VeciZ, SheddingDistance, ax, ay, az, Sources,
    SourcesM1, NumberParticlesShedPerStation, NumberOfLiftingLineSources, NumberOfSources,
    TimeShed, frozenLiftingLine):
    '''
    Updates the strength of the bound, the first row and the shed particles by the Lifting
    Line(s).
    
    Parameters
    ----------
        Dir : :py:class:`list` or numpy.ndarray of :py:class:`int`
            Gives the rotation of the Lifting Line(s), either positive +1, or negative (-1).

        VeciX : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Tangential vector to the Lifting Line(s) component along the x axis.

        VeciY : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Tangential vector to the Lifting Line(s) component along the y axis.

        VeciZ : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Tangential vector to the Lifting Line(s) component along the z axis.

        SheddingDistance : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Distance between the source stations on the Lifting Line(s) and the first row of
            shed particles at the previous TimeStep.

        ax : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Strength of the particles along the x axis.

        ay : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Strength of the particles along the y axis.

        az : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Strength of the particles along the z axis.

        Sources : :py:class:`list` or numpy.ndarray of Zone
            Sources on the Lifting Line(s) from where the particles are shed.

        SourcesM1 : :py:class:`list` or numpy.ndarray of Zone
            Sources on the Lifting Line(s) at the previous TimeStep from where the particles are
            shed.

        NumberParticlesShedPerStation : :py:class:`list` or numpy.ndarray of :py:class:`int`
            Number of particles shed per source station (if any).

        NumberOfLiftingLineSources : :py:class:`int`
            Total number of sources station on the Lifting Line(s).

        NumberOfSources : :py:class:`int`
            Total number of embedded bound Lifting Line, BEM and Interface particles.

        TimeShed : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Time from the last particle sheding for each source station.

        frozenLiftingLine : :py:class:`list` or numpy.ndarray of :py:class:`int`
            List of the Lifting Line(s) not to be updated.
    '''
    if NumberParticlesShedPerStation != []:
        SourcesBase = I.newCGNSBase('Sources', cellDim=1, physDim=3)
        SourcesBase[2] = I.getZones(Sources)
        SourcesBaseM1 = I.newCGNSBase('SourcesM1', cellDim=1, physDim=3)
        SourcesBaseM1[2] = I.getZones(SourcesM1)
        flag = np.array([1]*len(Dir), order = 'F', dtype = np.int32)
        for i in frozenLiftingLine: flag[i] = 0
        return V.shed_particles_from_lifting_lines(Dir, VeciX, VeciY, VeciZ,
                                       SheddingDistance, ax, ay, az, SourcesBase, SourcesBaseM1,
                                      NumberParticlesShedPerStation, NumberOfLiftingLineSources,
                                                                NumberOfSources, TimeShed, flag)
    else: return None

def getLiftingLineParameters(t = []):
    '''
    Gets the parameters regarding the Lifting Line(s).

    Parameters
    ----------
        t : Tree, Base, Zone
            Containes a node of parameters named '.LiftingLine#Parameters'.

    Returns
    -------
        LiftingLineParameters : :py:class:`dict` of numpy.ndarray
            Dictionnary of parameters. The parameters are pointers inside numpy ndarrays.
    '''
    return J.get(VPM.pickParticlesZone(t), '.LiftingLine#Parameters')

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
            Sources on the stations of the Lifting Line(s).

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

def updateLiftingLinesCirculation(LiftingLines):
    for LiftingLine in I.getZones(LiftingLines):
        Correc_n   = J.get(LiftingLine, '.Component#Info')['Corrections3D']
        Kinematics = J.get(LiftingLine, '.Kinematics')
        SweepCorrection    = Correc_n['Sweep']
        DihedralCorrection = Correc_n['Dihedral']
        RotationCenter = Kinematics['RotationCenter']
        TorqueOrigin   = Kinematics['TorqueOrigin']
        RotationAxis   = Kinematics['RotationAxis']
        dir            = 1 if Kinematics['RightHandRuleRotation'] else -1
        RPM            = Kinematics['RPM']

        U, ChordCorr, Chord, Cl, Gamma = J.getVars(LiftingLine, ['VelocityMagnitudeLocal',
                                                   'ChordVirtualWithSweep', 'Chord', 'Cl', 'Gamma'])

        Flux = 0.5*U*Cl
        if SweepCorrection: Flux *= ChordCorr
        else:               Flux *= Chord
        Gamma[:] = Flux


def projectLiftingLinesVelocities(LiftingLines):
    for LiftingLine in I.getZones(LiftingLines):
        Correc_n = J.get(LiftingLine, '.Component#Info')['Corrections3D']
        Conditions = J.get(LiftingLine, '.Conditions')
        SweepCorrection    = Correc_n['Sweep']
        DihedralCorrection = Correc_n['Dihedral']
        T0  = Conditions['Temperature']
        Rho = Conditions['Density']
        U0  = Conditions['VelocityFreestream']

        Ux, Uy, Uz, Uix, Uiy, Uiz, Upx, Upy, Upz, Ukx, Uky, Ukz, U2Dx, U2Dy, U2Dz, \
        chordX, chordY, chordZ, thickX, thickY, thickZ, U, Mach, Reynolds, AoA, Chord, SweepCorr, \
        DihedralCorr = J.getVars(LiftingLine, VPM.vectorise(['Velocity', 'VelocityInduced', \
            'VelocityPerturbation', 'VelocityKinematic', 'Velocity2D', 'Chordwise', 'Thickwise']) +\
            ['VelocityMagnitudeLocal', 'Mach', 'Reynolds', 'AoA', 'Chord', 'SweepAngleDeg', \
                                                                            'DihedralAngleDeg'])

        Ux[:] = Uix + Upx + U0[0]
        Uy[:] = Uiy + Upy + U0[1]
        Uz[:] = Uiz + Upz + U0[2]
        Urelx = Ux - Ukx
        Urely = Uy - Uky
        Urelz = Uz - Ukz

        Uchord = Urelx*chordX + Urely*chordY + Urelz*chordZ
        Uthick = Urelx*thickX + Urely*thickY + Urelz*thickZ

        if SweepCorrection:    Uchord *= np.cos(np.deg2rad(SweepCorr))
        if DihedralCorrection: Uthick *= np.cos(np.deg2rad(DihedralCorr))

        U2Dx[:] = Uchord*chordX + Uthick*thickX
        U2Dy[:] = Uchord*chordY + Uthick*thickY
        U2Dz[:] = Uchord*chordZ + Uthick*thickZ
        # Updating the Angle of Attack considering the new velocity components.
        AoA[:] = np.rad2deg(np.arctan2(Uthick,Uchord))

        U[:] = np.linalg.norm(np.vstack([Uchord, Uthick]), axis = 0)
        Mach[:] = U/np.sqrt(1.4*287.058*T0)
        Mu = 1.711e-5*np.sqrt(T0/273.)*(1. + 110.4/273.)/(1. + 110.4/T0)
        Reynolds[:] = Conditions['Density']*U*Chord/Mu

def moveAndUpdateLiftingLines(t = [], LiftingLines = [], dt = 0.,
    PerturbationFieldCapsule = []):
    '''
    Moves the Lifting Line(s) with their kinematic velocity and updates their local velocity
    accordingly.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        LiftingLines : Zone, :py:class:`list` or numpy.ndarray of Zone
            Containes the Lifting Lines.

        dt : :py:class:`float`
            TimeStep.

        PerturbationFieldCapsule : :py:class:`capsule`
            Stores the FMM octree used to interpolate the Perturbation Mesh onto the particles.
    '''
    LL.computeKinematicVelocity(LiftingLines)
    LL.moveLiftingLines(LiftingLines, dt)
    VPM.extractperturbationField(t = t, Targets = LiftingLines,
                                            PerturbationFieldCapsule = PerturbationFieldCapsule)
    projectLiftingLinesVelocities(LiftingLines)

def initialiseShedParticles(t = [], LiftingLines = [], Sources = [], Ramp = 1.,
    SmoothingRatio = 2., NumberOfLLSources = 0., NumberOfSources = 0., it = []):
    '''
    Initialises the particles to update and shed from the Lifting Line(s).

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        LiftingLines : Zone, :py:class:`list` or numpy.ndarray of Zone
            Containes the Lifting Lines.

        Sources : :py:class:`list` or numpy.ndarray of Zone
            Sources on the Lifting Line(s) from where the particles are shed.

        Ramp : :py:class:`float`
            Initial Ramp to decrease the strength of the particles.

        SmoothingRatio : :py:class:`float`
            Overlapping ratio of the VPM particles.

        NumberOfLLSources : :py:class:`int`
            Total number of source stations on the Lifting Line(s).

        NumberOfSources : :py:class:`int`
            Total number of embedded bound Lifting Line, BEM and Interface particles.
    '''
    frozenLiftingLines = []
    Particles = VPM.pickParticlesZone(t)
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

        if dy/(len(sx) - 2) < h*0.95: frozenLiftingLines += [index]#on average there is not enough space to shed particles

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
            NewS += [0.5*SmoothingRatio*np.linalg.norm([sx[i + 1] - sx[i - 1], sy[i + 1] - sy[i - 1], \
                                                              sz[i + 1] - sz[i - 1]], axis = 0)]
        
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
    SheddingDistance = []
    index = 0
    pos = NumberOfSources
    for Source, LiftingLine in zip(Sources, LiftingLines):#remaining particles shed in the wake
        h = I.getValue(I.getNodeFromName(LiftingLine, 'LocalResolution'))
        sx, sy, sz = J.getxyz(Source)
        if index not in frozenLiftingLines:
            for i in range(1, len(sx) - 1):
                xm = np.array([sx[i], sy[i], sz[i]])
                vecj = np.array([px[pos], py[pos], pz[pos]]) - xm
                dy = np.linalg.norm(vecj, axis = 0)
                VeciX += [0.5*(sx[i + 1] - sx[i - 1])]
                VeciY += [0.5*(sy[i + 1] - sy[i - 1])]
                VeciZ += [0.5*(sz[i + 1] - sz[i - 1])]
                SheddingDistance += [dy]
                Nshed = max(int(round(dy/h - 0.95)), 0)
                for j in range(Nshed):
                    NewX += [xm[0] + (j + 1)/(Nshed + 1)*vecj[0]]
                    NewY += [xm[1] + (j + 1)/(Nshed + 1)*vecj[1]]
                    NewZ += [xm[2] + (j + 1)/(Nshed + 1)*vecj[2]]

                NewAX += [0.]*Nshed
                NewAY += [0.]*Nshed
                NewAZ += [0.]*Nshed
                NewS += [SmoothingRatio*np.linalg.norm([VeciX[-1], VeciY[-1], VeciZ[-1]], axis = 0)]*Nshed
                NewNu += [nup[pos]]*Nshed

                ParticlesShedPerStation += [Nshed]
                pos += 1
        else:
            pos += len(sx) - 2

        index += 1

    ParticlesShedPerStation = np.array(ParticlesShedPerStation, dtype=np.int32, order = 'F')
    Ramp = Ramp/(ParticlesShedPerStation + 1)
    Dir = np.array(Dir, dtype = np.int32, order = 'F')
    frozenLiftingLines = np.array(frozenLiftingLines, dtype = np.int32, order = 'F')
    VeciX = np.array(VeciX, dtype = np.float64, order = 'F')*Ramp
    VeciY = np.array(VeciY, dtype = np.float64, order = 'F')*Ramp
    VeciZ = np.array(VeciZ, dtype = np.float64, order = 'F')*Ramp
    SheddingDistance = np.array(SheddingDistance, dtype = np.float64, order = 'F')*Ramp + 1e-12
    VPM.delete(Particles, deleteFlag)
    VPM.addParticlesToTree(Particles, NewX = NewX[:NumberOfLLSources],
        NewY = NewY[:NumberOfLLSources], NewZ = NewZ[:NumberOfLLSources],
        NewAX = NewAX[:NumberOfLLSources], NewAY = NewAY[:NumberOfLLSources],
        NewAZ = NewAZ[:NumberOfLLSources], 
        NewSigma = NewS[:NumberOfLLSources], Offset = 0, ExtendAtTheEnd = False)
    VPM.addParticlesToTree(Particles, NewX = NewX[NumberOfLLSources:],
        NewY = NewY[NumberOfLLSources:], NewZ = NewZ[NumberOfLLSources:],
        NewAX = NewAX[NumberOfLLSources:], NewAY = NewAY[NumberOfLLSources:],
        NewAZ = NewAZ[NumberOfLLSources:], 
        NewSigma = NewS[NumberOfLLSources:], Offset = NumberOfSources, ExtendAtTheEnd = False)

    nup = J.getVars(Particles, ['Nu'])[0]
    nup[:NumberOfLLSources] = NewNu[:NumberOfLLSources]
    nup[NumberOfSources:NumberOfSources + len(NewNu) - NumberOfLLSources] = NewNu[NumberOfLLSources:]

    return frozenLiftingLines, ParticlesShedPerStation, Dir, VeciX, VeciY, VeciZ, \
                                                 SheddingDistance, len(NewX[NumberOfLLSources:])

def ShedVorticitySourcesFromLiftingLines(t = [], PolarsInterpolators = {},
    IterationInfo = {}, PerturbationFieldCapsule = []):
    '''
    Updates the bound and first row of particles and shed particles from the Lifting Line(s).

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        PolarsInterpolators : Base or Zone or :py:class:`list` or numpy.ndarray of Base
            or Zone
            Containes the Polars for the sections of the Lifting Line(s).

        IterationInfo : :py:class:`dict` of :py:class:`str`
            Hybrid solver information on the current iteration.

        PerturbationFieldCapsule : :py:class:`capsule`
            Stores the FMM octree used to interpolate the Perturbation Mesh onto the particles.
    '''
    timeLL = J.tic()
    LiftingLines = LL.getLiftingLines(t)
    if not LiftingLines: return IterationInfo

    #for LiftingLine in LiftingLines:
    #    Gamma, GammaM1 = J.getVars(LiftingLine, ['Gamma', 'GammaM1'])
    #    GammaM1[:] = Gamma[:].copy()

    Particles = VPM.pickParticlesZone(t)
    Np0 = Particles[1][0][0]
    SmoothingRatio, dt, time, it, Ramp, \
    KinematicViscosity, EddyViscosityConstant, NumberOfLLSources, NumberOfBEMSources, \
    NumberOfCFDSources = VPM.getParameters(t, ['SmoothingRatio', 'TimeStep', 'Time', \
                 'CurrentIteration', 'StrengthRampAtbeginning', 'KinematicViscosity', \
                 'EddyViscosityConstant', 'NumberOfLiftingLineSources', 'NumberOfBEMSources', \
                 'NumberOfCFDSources'])
    if not NumberOfBEMSources: NumberOfBEMSources = [0]
    if not NumberOfCFDSources: NumberOfCFDSources = [0]
    NumberOfSources = NumberOfLLSources[0] + NumberOfCFDSources[0] + NumberOfBEMSources[0]
    NumberOfLLSources = NumberOfLLSources[0]
    Ramp = np.sin(min((it[0] + 1)/Ramp[0], 1.)*np.pi/2.)
    moveAndUpdateLiftingLines(t, LiftingLines, dt[0], PerturbationFieldCapsule)

    ParticleDistribution = [I.getNodeFromName(LiftingLine, 'ParticleDistribution')[1] for \
                                                                    LiftingLine in LiftingLines]
    Sources = LL.buildVortexParticleSourcesOnLiftingLine(LiftingLines, AbscissaSegments = \
                                                   ParticleDistribution, IntegralLaw = 'linear')
    TimeShed, GammaThreshold, GammaRelax, MaxIte = [], [], [], 0
    for LiftingLine in LiftingLines:
        LLParameters = J.get(LiftingLine, '.VPM#Parameters')
        TimeShed += [LLParameters['TimeSinceLastShedding'][0]]
        GammaThreshold += [LLParameters['CirculationThreshold'][0][0]]
        GammaRelax += [LLParameters['CirculationRelaxationFactor'][0][0]]
        MaxIte = max(MaxIte, LLParameters['MaxLiftingLineSubIterations'][0][0])

    TimeShed = np.array(TimeShed, dtype = np.float64, order = 'F')
    SourcesM1 = [I.copyTree(Source) for Source in Sources]
    GammaOld = [I.getNodeFromName3(Source, 'Gamma')[1] for Source in Sources]


    frozenLiftingLines, ParticlesShedPerStation, Dir, VeciX, VeciY, VeciZ, SheddingDistance, \
              Nshed = initialiseShedParticles(t, LiftingLines, Sources, Ramp, SmoothingRatio[0],
                                                             NumberOfLLSources, NumberOfSources, it[0])
    SheddingLiftingLines = I.getZones(I.copyRef(LiftingLines))
    for index in frozenLiftingLines[::-1]:
        SheddingLiftingLines.pop(index)
        ParticleDistribution.pop(index)
        GammaThreshold.pop(index)
        GammaRelax.pop(index)
        GammaOld.pop(index)

    GammaError = [np.inf]*len(SheddingLiftingLines)
    GammaDampening = [1.]*len(SheddingLiftingLines)

    GammaThreshold = np.array(GammaThreshold, dtype = np.float64, order = 'F')
    GammaRelax = np.array(GammaRelax, dtype = np.float64, order = 'F')

    ax, ay, az, s = J.getVars(Particles, ['Alpha' + v for v in 'XYZ'] + ['Sigma'])
    WakeInducedVelocity = extractWakeInducedVelocityOnLiftingLines(t, SheddingLiftingLines,
                                                                                          Nshed)
    ni = 0
    for _ in range(MaxIte):
        setShedParticleStrength(Dir, VeciX, VeciY, VeciZ, SheddingDistance, ax, ay, az, \
                                 Sources, SourcesM1, ParticlesShedPerStation, NumberOfLLSources,
                                                  NumberOfSources, TimeShed, frozenLiftingLines)
        BoundAndShedInducedVelocity = extractBoundAndShedVelocityOnLiftingLines(t,
                                                                    SheddingLiftingLines, Nshed)
        
        setLiftingLinesInducedVelocity(SheddingLiftingLines,
                                              WakeInducedVelocity + BoundAndShedInducedVelocity)
        projectLiftingLinesVelocities(SheddingLiftingLines)
        LL._applyPolarOnLiftingLine(SheddingLiftingLines, PolarsInterpolators, ['Cl'])
        updateLiftingLinesCirculation(SheddingLiftingLines)
        Sources = LL.buildVortexParticleSourcesOnLiftingLine(SheddingLiftingLines,
                                AbscissaSegments = ParticleDistribution, IntegralLaw = 'linear')
        GammaError = relaxCirculationAndGetImbalance(GammaOld, GammaRelax, Sources, GammaError,
                                                                                 GammaDampening)
        ni += 1
        if (GammaError < GammaThreshold).all(): break

        for index in frozenLiftingLines: Sources.insert(index, SourcesM1[index])

    #if (GammaError < GammaThreshold).any():
    #    safeLiftingLinesIterations(t, SheddingLiftingLines, frozenLiftingLines, ParticleDistribution, GammaThreshold, GammaRelax, Dir, VeciX, VeciY, VeciZ, SheddingDistance, ax, ay, az)

    Nu, Cvisq = J.getVars(Particles, ['Nu', 'Cvisq'])
    offset = NumberOfSources + Nshed
    Nu[:NumberOfLLSources] = 0.
    Cvisq[:NumberOfLLSources] = 0.
    Cvisq[NumberOfSources: NumberOfSources + Nshed] = EddyViscosityConstant

    for LiftingLine in SheddingLiftingLines:
        TimeShed = I.getNodeFromName(LiftingLine, 'TimeSinceLastShedding')
        TimeShed[1][0] = 0
    #for LiftingLine in LiftingLines:
    #    Gamma, GammaM1, dGammadt = J.getVars(LiftingLine, ['Gamma', 'GammaM1', 'dGammadt'])
    #    dGammadt[:] = (Gamma[:] - GammaM1[:])/dt

    LL.assembleAndProjectVelocities(SheddingLiftingLines)
    LL._applyPolarOnLiftingLine(SheddingLiftingLines, PolarsInterpolators, ['Cl', 'Cd', 'Cm'])
    if len(GammaError) == 0: GammaError = np.array([0])
    LL.computeGeneralLoadsOfLiftingLine(LiftingLines,
            UnsteadyData={'IterationNumber':it[0],
                          'Time':time[0],
                          'CirculationSubiterations':ni,
                          'CirculationError':np.max(GammaError)},
                            UnsteadyDataIndependentAbscissa='IterationNumber')

    IterationInfo['Circulation error'] = np.max(GammaError)
    IterationInfo['Number of sub-iterations (LL)'] = ni
    IterationInfo['Number of shed particles'] = Particles[1][0][0] - Np0
    IterationInfo['Lifting Line time'] = J.tic() - timeLL
    return IterationInfo

####################################################################################################
####################################################################################################
######################################### Coeff/Loads Aero #########################################
####################################################################################################
####################################################################################################
def getAerodynamicCoefficientsOnLiftingLine(LiftingLines = [], StdDeviationSample = 50,
    IterationInfo = {}, Freestream = True, Wings = False, Surface = 0.):
    '''
    Gets the aerodynamic coefficients on the Lifting Line(s).

    Parameters
    ----------

        LiftingLines : Zone, :py:class:`list` or numpy.ndarray of Zone
            Containes the Lifting Lines.

        StdDeviationSample : :py:class:`int`
            Number of samples for the standard deviation.

        IterationInfo : :py:class:`dict` of :py:class:`str`
            Hybrid solver information on the current iteration.

        Freestream : :py:class:`bool`
            States whether their is a freestream velocity.

        Wings : :py:class:`bool`
            States whether the Lifting Line is a wing.

        Surface : :py:class:`float`
            Surface of the wing Lifting Line (if any).
    '''
    if LiftingLines:
        if Wings: IterationInfo = getAerodynamicCoefficientsOnWing(LiftingLines, Surface,
                  StdDeviationSample = StdDeviationSample, IterationInfo = IterationInfo)
        else:
            if Freestream: IterationInfo = getAerodynamicCoefficientsOnPropeller(LiftingLines,
                  StdDeviationSample = StdDeviationSample, IterationInfo = IterationInfo)
            else: IterationInfo = getAerodynamicCoefficientsOnRotor(LiftingLines,
                  StdDeviationSample = StdDeviationSample, IterationInfo = IterationInfo)
    return IterationInfo

def getAerodynamicCoefficientsOnPropeller(LiftingLines = [], StdDeviationSample = 50,
    IterationInfo = {}):
    '''
    Gets the aerodynamic coefficients on propeller Lifting Line(s).

    Parameters
    ----------

        LiftingLines : Zone, :py:class:`list` or numpy.ndarray of Zone
            Containes the Lifting Lines.

        StdDeviationSample : :py:class:`int`
            Number of samples for the standard deviation.

        IterationInfo : :py:class:`dict` of :py:class:`str`
            Hybrid solver information on the current iteration.
    '''
    LiftingLine = I.getZones(LiftingLines)[0]
    RotationCenter = I.getValue(I.getNodeFromName(LiftingLine, 'RotationCenter'))
    RPM = I.getValue(I.getNodeFromName(LiftingLine, 'RPM'))
    Rho = I.getValue(I.getNodeFromName(LiftingLine, 'Density'))
    U0 = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityFreestream'))
    V = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityTranslation'))
    x, y, z = J.getxyz(LiftingLine)
    D = 2*max(np.linalg.norm(np.vstack([x - RotationCenter[0], y - RotationCenter[1], \
                                                             z - RotationCenter[2]]), axis = 0))
    IntegralLoads = LL.computeGeneralLoadsOfLiftingLine(LiftingLines)
    n   = RPM/60.
    T = IntegralLoads['Total']['Thrust'][0]
    P = IntegralLoads['Total']['Power'][0]
    P = np.sign(P)*np.maximum(1e-12,np.abs(P))
    cT = T/(Rho*np.square(n*D*D))
    cP = P/(Rho*n*D*np.square(n*D*D))
    Uinf = np.linalg.norm(U0 - V, axis = 0)
    Eff = Uinf*T/P
    std_Thrust, std_Power = getStandardDeviationBlade(LiftingLines = LiftingLines,
                                                        StdDeviationSample = StdDeviationSample)
    IterationInfo['Thrust'] = T
    IterationInfo['Thrust Standard Deviation'] = std_Thrust/(T + np.sign(T)*1e-12)*100.
    IterationInfo['Power'] = P
    IterationInfo['Power Standard Deviation'] = std_Power/(P + np.sign(P)*1e-12)*100.
    IterationInfo['cT'] = cT
    IterationInfo['cP'] = cP
    IterationInfo['Eff'] = Eff
    return IterationInfo

def getAerodynamicCoefficientsOnRotor(LiftingLines = [], StdDeviationSample = 50,
    IterationInfo = {}):
    '''
    Gets the aerodynamic coefficients on rotor Lifting Line(s).

    Parameters
    ----------

        LiftingLines : Zone, :py:class:`list` or numpy.ndarray of Zone
            Containes the Lifting Lines.

        StdDeviationSample : :py:class:`int`
            Number of samples for the standard deviation.

        IterationInfo : :py:class:`dict` of :py:class:`str`
            Hybrid solver information on the current iteration.
    '''
    LiftingLine = I.getZones(LiftingLines)[0]
    RotationCenter = I.getValue(I.getNodeFromName(LiftingLine, 'RotationCenter'))
    RPM = I.getValue(I.getNodeFromName(LiftingLine, 'RPM'))
    Rho = I.getValue(I.getNodeFromName(LiftingLine, 'Density'))
    x, y, z = J.getxyz(LiftingLine)
    D = 2*max(np.linalg.norm(np.vstack([x - RotationCenter[0], y - RotationCenter[1], \
                                                             z - RotationCenter[2]]), axis = 0))
    IntegralLoads = LL.computeGeneralLoadsOfLiftingLine(LiftingLines)
    n   = RPM/60.
    T = IntegralLoads['Total']['Thrust'][0]
    P = IntegralLoads['Total']['Power'][0]
    cT = T/(Rho*np.square(n*D*D))
    cP = P/(Rho*n*D*np.square(n*D*D))
    cP = np.sign(cP)*np.maximum(1e-12,np.abs(cP))
    Eff = np.sqrt(2./np.pi)*np.abs(cT)*np.sqrt(np.abs(cT))/cP

    std_Thrust, std_Power = getStandardDeviationBlade(LiftingLines = LiftingLines,
                                                        StdDeviationSample = StdDeviationSample)
    IterationInfo['Thrust'] = T
    IterationInfo['Thrust Standard Deviation'] = std_Thrust/(T + np.sign(T)*1e-12)*100.
    IterationInfo['Power'] = P
    IterationInfo['Power Standard Deviation'] = std_Power/(P + np.sign(P)*1e-12)*100.
    IterationInfo['cT'] = cT
    IterationInfo['cP'] = cP
    IterationInfo['Eff'] = Eff
    return IterationInfo

def getAerodynamicCoefficientsOnWing(LiftingLines = [], Surface = 0., StdDeviationSample = 50,
    IterationInfo = {}):
    '''
    Gets the aerodynamic coefficients on wing Lifting Line(s).

    Parameters
    ----------

        LiftingLines : Zone, :py:class:`list` or numpy.ndarray of Zone
            Containes the Lifting Lines.

        Surface : :py:class:`float`
            Surface of the wing Lifting Line.

        StdDeviationSample : :py:class:`int`
            Number of samples for the standard deviation.

        IterationInfo : :py:class:`dict` of :py:class:`str`
            Hybrid solver information on the current iteration.
    '''
    Drag, Lift, cL, cD = 0., 0., 0., 0.
    for LiftingLine in LiftingLines:
        Rho = I.getValue(I.getNodeFromName(LiftingLine, 'Density'))
        U0 = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityFreestream'))
        V = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityTranslation'))
        Axis = I.getValue(I.getNodeFromName(LiftingLine, 'RotationAxis'))
        IntegralLoads = LL.computeGeneralLoadsOfLiftingLine(LiftingLine)
        F = np.array([IntegralLoads['Force' + v][0] for v in 'XYZ'])
        Lift0 = np.dot(F, Axis)
        Drag0 = np.sqrt(np.sum(np.square(F)) - np.square(Lift0))
        Lift += Lift0
        Drag += Drag0
        q0 = 0.5*Rho*Surface*np.dot(U0 - V, (U0 - V).T)
        cL += Lift0/(q0 + 1e-12)
        cD += Drag0/(q0 + 1e-12)

    std_Thrust, std_Drag = getStandardDeviationWing(LiftingLines = LiftingLines,
                                                        StdDeviationSample = StdDeviationSample)

    IterationInfo['Lift'] = Lift
    IterationInfo['Lift Standard Deviation'] = std_Thrust/(Lift + np.sign(Lift)*1e-12)*100.
    IterationInfo['Drag'] = Drag
    IterationInfo['Drag Standard Deviation'] = std_Drag/(Drag + np.sign(Drag)*1e-12)*100.
    IterationInfo['cL'] = cL if 1e-12 < q0 else 0.
    IterationInfo['cD'] = cD if 1e-12 < q0 else 0.
    IterationInfo['f'] = Lift/Drag
    return IterationInfo

def getStandardDeviationWing(LiftingLines = [], StdDeviationSample = 50):
    '''
    Gets the standard deviation on the aerodynamic coefficients on wing Lifting Line(s).

    Parameters
    ----------

        LiftingLines : Zone, :py:class:`list` or numpy.ndarray of Zone
            Containes the Lifting Lines.

        StdDeviationSample : :py:class:`int`
            Number of samples for the standard deviation.
    '''
    LiftingLine = I.getZones(LiftingLines)[0]
    UnsteadyLoads = I.getNodeFromName(LiftingLine, '.UnsteadyLoads')
    Thrust = I.getValue(I.getNodeFromName(UnsteadyLoads, 'Thrust'))
    if type(Thrust) == np.ndarray or type(Thrust) == list:
        StdDeviationSample = max(min(StdDeviationSample,len(Thrust)), 1)
    else: return 0., 0.

    Thrust = np.array([0.]*StdDeviationSample)
    Drag = np.array([0.]*StdDeviationSample)
    for LiftingLine in LiftingLines:
        UnsteadyLoads = I.getNodeFromName(LiftingLine, '.UnsteadyLoads')
        Thrust += I.getNodeFromName(UnsteadyLoads, 'ForceZ')[1][-StdDeviationSample:]
        Drag += I.getNodeFromName(UnsteadyLoads, 'ForceX')[1][-StdDeviationSample:]
    meanThrust = sum(Thrust)/StdDeviationSample
    meanDrag = sum(Drag)/StdDeviationSample

    std_Thrust = np.sqrt(sum(np.square(Thrust - meanThrust))/StdDeviationSample)
    std_Drag = np.sqrt(sum(np.square(Drag - meanDrag))/StdDeviationSample)
    return std_Thrust, std_Drag

def getStandardDeviationBlade(LiftingLines = [], StdDeviationSample = 50):
    '''
    Gets the standard deviation on the aerodynamic coefficients on blade Lifting Line(s).

    Parameters
    ----------

        LiftingLines : Zone, :py:class:`list` or numpy.ndarray of Zone
            Containes the Lifting Lines.

        StdDeviationSample : :py:class:`int`
            Number of samples for the standard deviation.
    '''
    LiftingLine = I.getZones(LiftingLines)[0]
    UnsteadyLoads = I.getNodeFromName(LiftingLine, '.UnsteadyLoads')
    Thrust = I.getValue(I.getNodeFromName(UnsteadyLoads, 'Thrust'))
    if type(Thrust) == np.ndarray or type(Thrust) == list:
        StdDeviationSample = max(min(StdDeviationSample,len(Thrust)), 1)
    else: return 0., 0.

    Thrust = np.array([0.]*StdDeviationSample)
    Power = np.array([0.]*StdDeviationSample)
    for LiftingLine in LiftingLines:
        UnsteadyLoads = I.getNodeFromName(LiftingLine, '.UnsteadyLoads')
        Thrust += I.getNodeFromName(UnsteadyLoads, 'Thrust')[1][-StdDeviationSample:]
        Power += I.getNodeFromName(UnsteadyLoads, 'Power')[1][-StdDeviationSample:]
    meanThrust = sum(Thrust)/StdDeviationSample
    meanPower = sum(Power)/StdDeviationSample

    std_Thrust = np.sqrt(sum(np.square(Thrust - meanThrust))/StdDeviationSample)
    std_Power = np.sqrt(sum(np.square(Power - meanPower))/StdDeviationSample)
    return std_Thrust, std_Power
