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

from . import LiftingLineCoupling as VPMLL
from . import EulerianCoupling as H

from .. import LiftingLine as LL
from .. import Wireframe as W
from .. import InternalShortcuts as J
from .. import ExtractSurfacesProcessor as ESP

from VULCAINS.__init__ import __version__, __author__

####################################################################################################
####################################################################################################
######################################## CGNS tree control #########################################
####################################################################################################
####################################################################################################
def vectorise(names = '', maj = True):
    coord = 'XYZ' if maj else 'xyz'
    if type(names) == str: names = [names]

    vector = []
    for name in names: vector += [name + v for v in coord]

    return vector

Scheme_str2int = {'Transpose': 0, 'T': 0, 'Mixed': 1, 'M': 1, 'Classical': 2, 'C': 2}
EddyViscosityModel_str2int = {'Vreman': 1, 'Mansour': 2, 'Mansour2': 3, 'Smagorinsky': 4,
                                                                             None: 0, 'None': 0}
RedistributionKernel_str2int = {'M4Prime': 5, 'M4': 4, 'M3': 3, 'M2': 2, 'M1': 1, None: 0,
                                                                                      'None': 0}
DiffusionScheme_str2int = {'PSE': 1, 'ParticleStrengthExchange': 1, 'pse': 1, 'DVM': 2,'dvm': 2,
    'DiffusionVelocityMethod': 2, 'CSM': 3, 'CS': 3, 'csm': 3, 'cs': 3, 'CoreSpreading': 3,
    'CoreSpreadingMethod': 3, 'None': 0, None: 0}

Vector_VPM_FlowSolution = vectorise(['VelocityInduced', 'VelocityPerturbation', \
    'VelocityBEM', 'VelocityInterface', 'VelocityDiffusion', 'gradxVelocity', 'gradyVelocity', \
                                    'gradzVelocity', 'PSE', 'Vorticity', 'Alpha', 'Stretching'])
Scalar_VPM_FlowSolution = ['Age', 'Sigma', 'Cvisq', 'Nu', 'divUd', 'Enstrophyf', 'Enstrophy',
                                                                                  'EnstrophyM1']
VPM_FlowSolution = Vector_VPM_FlowSolution + Scalar_VPM_FlowSolution

def checkParametersTypes(ParametersList = [], int_Params = [], float_Params = [],
    bool_Params = []):
    '''
    Sets the types of parameters according to a given list of parameter types

    Parameters
    ----------
        ParametersList : :py:class:`list` or numpy.ndarray of :py:class:`dict`
            List of the parameters to check. The parameters must be integer, loats or booleans
            and must be contained within int_Params, float_Params or bool_Params to be checked.

        int_Params : :py:class:`list` or numpy.ndarray
            List of the integer parameters.

        float_Params : :py:class:`list` or numpy.ndarray
            List of the float parameters.

        bool_Params : :py:class:`list` or numpy.ndarray
            List of the boolean parameters.
    '''
    for Parameters in ParametersList:
        for key in Parameters:
            if key in int_Params:
                Parameters[key] = np.atleast_1d(np.array(Parameters[key],
                                                        dtype = np.int32))
            elif key in float_Params:
                Parameters[key] = np.atleast_1d(np.array(Parameters[key],
                                                        dtype = np.float64))
            elif key in bool_Params:
                Parameters[key] = np.atleast_1d(np.array(Parameters[key],
                                                        dtype = np.int32))

def delete(t = [], mask = []):
    '''
    Deletes the particles inside t that are flagged by mask.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        mask : :py:class:`list` or numpy.ndarray
            List of booleans of the same size as the particle zone. A true flag will delete a 
            particle, a false flag will leave it.
    '''
    Particles = pickParticlesZone(t)
    GridCoordinatesNode = I.getNodeFromName1(Particles, 'GridCoordinates')
    FlowSolutionNode = I.getNodeFromName1(Particles, 'FlowSolution')
    Np = Particles[1].ravel(order = 'F')[0]
    if Np != len(mask): raise ValueError('The length of the mask (%d) must be the same as the \
                                                      number of particles (%d)'%(len(mask), Np))
    mask = np.logical_not(mask)
    for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
        if node[3] == 'DataArray_t':
            node[1] = node[1][mask]

    Particles[1].ravel(order = 'F')[0] = len(node[1])

def extend(t = [], ExtendSize = 0, Offset = 0, ExtendAtTheEnd = True):
    '''
    Add empty particles in t.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        ExtendSize : :py:class:`int`
            Number of particles to add.

        Offset : :py:class:`int`
            Position where the particles are added.

        ExtendAtTheEnd : :py:class:`bool`
            If True the particles are added at the end of t, by an offset of Offset, if False
            the particles are added at the beginning of t at the position Offset.
    '''
    Particles = pickParticlesZone(t)
    Cvisq = getParameter(Particles, 'EddyViscosityConstant')
    if not Cvisq: Cvisq = 0.
    GridCoordinatesNode = I.getNodeFromName1(Particles, 'GridCoordinates')
    FlowSolutionNode = I.getNodeFromName1(Particles, 'FlowSolution')
    Np = len(J.getx(Particles))
    if Np < abs(Offset): raise ValueError('Offset (%d) cannot be greater than existing number \
                                                                of particles (%d)'%(Offset, Np))
    if ExtendAtTheEnd:
        for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
            if node[3] == 'DataArray_t':
                zeros = np.array(np.zeros(ExtendSize) + Cvisq*(node[0] \
                                                == 'Cvisq'), dtype = node[1].dtype, order = 'F')
                node[1] = np.append(np.append(node[1][:Np -Offset], zeros), node[1][Np-Offset:])

    else:
        for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
            if node[3] == 'DataArray_t':
                zeros = np.array(np.zeros(ExtendSize) + Cvisq*(node[0] \
                                                == 'Cvisq'), dtype = node[1].dtype, order = 'F')
                node[1] = np.append(np.append(node[1][:Offset], zeros), node[1][Offset:])

    Particles[1].ravel(order = 'F')[0] = len(node[1])

def addParticlesToTree(t, NewX = [], NewY = [], NewZ = [], NewAX = [], NewAY = [], NewAZ = [],
    NewSigma = [], Offset = 0, ExtendAtTheEnd = False):
    '''
    Add particles in t.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        NewX : :py:class:`list` or numpy.ndarray
            Positions along the x axis of the particles to add.

        NewY : :py:class:`list` or numpy.ndarray
            Positions along the y axis of the particles to add.

        NewZ : :py:class:`list` or numpy.ndarray
            Positions along the z axis of the particles to add.

        NewAX : :py:class:`list` or numpy.ndarray
            Strength along the x axis of the particles to add.

        NewAY : :py:class:`list` or numpy.ndarray
            Strength along the y axis of the particles to add.

        NewAZ : :py:class:`list` or numpy.ndarray
            Strength along the z axis of the particles to add.

        NewSigma : :py:class:`list` or numpy.ndarray
            Size of the particles to add.

        Offset : :py:class:`int`
            Position where the particles are added.

        ExtendAtTheEnd : :py:class:`bool`
            If True the particles are added at the end of t, by an offset of Offset, if False
            the particles are added at the beginning of t at the position Offset.
    '''
    Nnew = len(NewX)
    extend(t, ExtendSize = Nnew, Offset = Offset, ExtendAtTheEnd = ExtendAtTheEnd)
    Particles = pickParticlesZone(t)
    Nnew += Offset
    px, py, pz = J.getxyz(Particles)
    px[Offset: Nnew] = NewX
    py[Offset: Nnew] = NewY
    pz[Offset: Nnew] = NewZ
    AX, AY, AZ, WX, WY, WZ, Nu, Sigma = J.getVars(Particles, vectorise(['Alpha', \
                                                                'Vorticity']) + ['Nu', 'Sigma'])
    AX[Offset: Nnew] = NewAX
    AY[Offset: Nnew] = NewAY
    AZ[Offset: Nnew] = NewAZ
    Sigma[Offset: Nnew] = NewSigma

def trim(t = [], NumberToTrim = 0, Offset = 0, TrimAtTheEnd = True):
    '''
    Removes a set of particles in t.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        NumberToTrim : :py:class:`int`
            Number of particles to remove.

        Offset : :py:class:`int`
            Position from where the particles are removed.

        ExtendAtTheEnd : :py:class:`bool`
            If True the particles are removed from the end of t, by an offset of Offset, if
            False the particles are removed from the beginning of t from the position Offset.
    '''
    Particles = pickParticlesZone(t)
    GridCoordinatesNode = I.getNodeFromName1(Particles, 'GridCoordinates')
    FlowSolutionNode = I.getNodeFromName1(Particles, 'FlowSolution')
    Np = len(J.getx(Particles))
    if Np < abs(Offset): raise ValueError('Offset (%d) cannot be greater than existing number \
                                                                of particles (%d)'%(Offset, Np))
    if TrimAtTheEnd:
        for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
            if node[3] == 'DataArray_t':
                node[1] = np.append(node[1][:Np - Offset - NumberToTrim], node[1][Np - Offset:])
    else:
        for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
            if node[3] == 'DataArray_t':
                node[1] = np.append(node[1][:Offset], node[1][Offset + NumberToTrim:])

    Particles[1].ravel(order = 'F')[0] = len(node[1])

def adjustTreeSize(t = [], NewSize = 0, OldSize = -1, Offset = 0, AtTheEnd = True):
    '''
    Adds (Offset < NewSize) or removes (NewSize < Offset) a set of particles in t to adjust its
    size. If OldSize is not given, the current size of t is used.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        NewSize : :py:class:`int`
            New number of particles.

        OldSize : :py:class:`int`
            Old number of particles.

        Offset : :py:class:`int`
            Position from where the particles are added or removed.

        ExtendAtTheEnd : :py:class:`bool`
            If True the particles are added or removed from the end of t, by an offset of
            Offset, if False the particles are added or removed from the beginning of t from
            the position Offset.
    '''
    Particles = pickParticlesZone(t)
    if OldSize == -1: OldSize = len(J.getx(Particles))
    SizeDiff = NewSize - OldSize

    if 0 < SizeDiff:extend(t, ExtendSize = SizeDiff, Offset = Offset, ExtendAtTheEnd = AtTheEnd)
    else: trim(t, NumberToTrim = -SizeDiff, Offset = Offset, TrimAtTheEnd = AtTheEnd)

def roll(t = [], PivotNumber = 0):
    '''
    Moves a set of particles at the end of t

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        PivotNumber : :py:class:`int`
            Position of the new first particle
    '''
    Particles = pickParticlesZone(t)
    Np = Particles[1].ravel(order = 'F')[0]
    if PivotNumber == 0 or PivotNumber == Np: return
    if PivotNumber > Np:
        raise ValueError('PivotNumber (%d) cannot be greater than existing number of particles\
                                                                        (%d)'%(PivotNumber, Np))
    GridCoordinatesNode = I.getNodeFromName1(Particles, 'GridCoordinates')
    FlowSolutionNode = I.getNodeFromName1(Particles, 'FlowSolution')
    for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
        if node[3] == 'DataArray_t':
            node[1] = np.roll(node[1], PivotNumber)

def getPerturbationFieldParameters(t = []):
    '''
    Gets the parameters regarding the perturbation field.

    Parameters
    ----------
        t : Tree, Base, Zone
            Containes a node of parameters named '.PerturbationField#Parameters'.

    Returns
    -------
        PerturbationFieldParameters : :py:class:`dict` of numpy.ndarray
            Dictionnary of parameters. The parameters are pointers inside numpy ndarrays.
    '''
    return J.get(pickParticlesZone(t), '.PerturbationField#Parameters')

def getParameter(t = [], Name = ''):
    '''
    Get a parameter.

    Parameters
    ----------
        t : Tree, Base, Zone
            Containes a node of parameters where one of them is named Name.

        Name : :py:class:`str`
            Name of the parameter to get.

    Returns
    -------
        ParameterNode : :py:class:`dict` of numpy.ndarray
            The parameter is a pointer inside a numpy ndarray.
    '''
    Particles = pickParticlesZone(t)
    Node = getVPMParameters(Particles)
    if Name in Node:
        ParameterNode = Node[Name]
        if (isinstance(ParameterNode, list) or isinstance(ParameterNode, np.ndarray)):
            return ParameterNode
    else :ParameterNode = None
    if ParameterNode == None:
        Node = H.getHybridParameters(Particles)
        if Node and Name in Node: ParameterNode = Node[Name]
    if ParameterNode == None:
        Node = getPerturbationFieldParameters(Particles)
        if Node and Name in Node: ParameterNode = Node[Name]
    return ParameterNode

def getParameters(t = [], Names = []):
    '''
    Get a list of parameters.

    Parameters
    ----------
        t : Tree, Base, Zone
            Containes a node of parameters with their names in Names.

        Names : :py:class:`list` or numpy.ndarray of :py:class:`str`
            List of parameter names

    Returns
    -------
        ParameterNode : :py:class:`dict` of numpy.ndarray
            The parameter is a pointer inside a numpy ndarray.
    '''
    Particles = pickParticlesZone(t)
    return [getParameter(Particles, Name) for Name in Names]

####################################################################################################
####################################################################################################
############################################### VPM ################################################
####################################################################################################
####################################################################################################
def buildEmptyVPMTree():
    '''
    Build an empty particle tree with all the particle fields.

    Returns
    -------
        t : Tree
            CGNS Tree containing an empty base of particles.
    '''
    Particles = C.convertArray2Node(D.line((0., 0., 0.), (0., 0., 0.), 2))
    Particles[0] = 'Particles'
    J.invokeFieldsDict(Particles, VPM_FlowSolution)
    IntegerFieldNames = ['Age']
    GridCoordinatesNode = I.getNodeFromName1(Particles, 'GridCoordinates')
    FlowSolutionNode = I.getNodeFromName1(Particles, 'FlowSolution')
    for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
        if node[3] == 'DataArray_t':
            if node[0] in IntegerFieldNames:
                node[1] = np.array([], dtype = np.int32, order = 'F')
            else:
                node[1] = np.array([], dtype = np.float64, order = 'F')

    Particles[1].ravel(order = 'F')[0] = 0
    I._sortByName(I.getNodeFromName1(Particles, 'FlowSolution'))
    return C.newPyTree(['ParticlesBase', Particles])

def initialiseVPM(EulerianMesh = [], LiftingLineTree = [], PerturbationField = [],
    PolarsInterpolators = [], VPMParameters = {}, HybridParameters = {},
    LiftingLineParameters = {}, PerturbationFieldParameters = {}):
    '''
    Creates a Tree initialised with the given checked and updated parameters and initialise the
    particles according to the given Lifting Lines or Eulerian Mesh. Also generate Hybrid
    Domains and/or Lifting Lines in the Tree.

    Parameters
    ----------
        EulerianMesh : Tree, Base or :py:class:`str`
            Containes an Eulerian Mesh or the adress where it is located.

        LiftingLineTree : Tree, Base or Zone or :py:class:`list` or numpy.ndarray of Tree, Base
            or Zone
            Containes Lifting Line(s).

        PerturbationField : Tree, Base or Zone
            Containes a mesh of perturbation velocities or the adress where it is located.

        PolarsInterpolators : Base or Zone or :py:class:`list` or numpy.ndarray of Base
            or Zone
            Containes the Polars for the sections of the Lifting Line(s).

        VPMParameters : :py:class:`dict` of :py:class:`float`, :py:class:`int`, :py:class:`bool` and :py:class:`str`
            Containes user given parameters for the VPM solver. They will be updated with a set
            of preregistered parameters.

        HybridParameters : :py:class:`dict` of :py:class:`float` and :py:class:`int`
            Containes user given parameters for the Hybrid solver. They will be updated with a
            set of preregistered parameters.

        LiftingLineParameters : :py:class:`dict` of :py:class:`float`, :py:class:`int` and
            :py:class:`str`
            Containes user given parameters for the Lifting Line(s). They will be updated with a
            set of preregistered parameters.

        PerturbationFieldParameters : :py:class:`dict` of :py:class:`float` and :py:class:`int`
            Containes user given parameters for the Perturbation Velocity Field. They will be
            updated with a set of preregistered parameters.

    Returns
    -------
        ParameterNode : :py:class:`dict` of numpy.ndarray
            The parameter is a pointer inside a numpy ndarray.

        t : Tree
            Tree of updated parameters, initialised particles and generated Hybrid Domain
            (if any).

        tE : Tree
            Eulerian Tree for the CFD solver.

        PerturbationFieldCapsule : :py:class:`capsule`
            Stores the FMM octree used to interpolate the Perturbation Mesh onto the particles.
    '''
    int_Params = ['MaxLiftingLineSubIterations', 'MaximumSubIteration','StrengthRampAtbeginning', 
        'MinNbShedParticlesPerLiftingLine', 'CurrentIteration', 'NumberOfHybridInterfaces',
        'MaximumAgeAllowed', 'RedistributionPeriod', 'NumberOfThreads', 'IntegrationOrder',
        'IterationTuningFMM', 'IterationCounter', 'NumberOfNodes', 'EnstrophyControlRamp',
        'NumberOfParticlesPerInterface', 'ClusterSize', 'NumberOfLiftingLines'
        'FarFieldApproximationOrder', 'NumberOfLiftingLineSources', 'NumberOfHybridSources',
        'NumberOfBEMSources', 'NumberOfCFDSources']

    float_Params = ['Density', 'EddyViscosityConstant', 'Temperature', 'ResizeParticleFactor',
        'Time', 'CutoffXmin', 'CutoffZmin', 'MaximumMergingVorticityFactor', 
        'AntiDiffusion', 'SmoothingRatio', 'CirculationThreshold', 'RPM','KinematicViscosity',
        'CirculationRelaxationFactor', 'CutoffXmax', 'CutoffYmin', 'CutoffYmax', 'Sigma0',
        'CutoffZmax', 'ForcedDissipation','MaximumAngleForMerging', 'MinimumVorticityFactor', 
        'RelaxationFactor', 'MinimumOverlapForMerging', 'VelocityFreestream', 'AntiStretching',
        'RelaxationThreshold', 'RedistributeParticlesBeyond', 'RedistributeParticleSizeFactor',
        'TimeStep', 'Resolution', 'VelocityTranslation', 'NearFieldOverlappingRatio', 'TimeFMM',
        'RemoveWeakParticlesBeyond', 'OuterDomainToWallDistance', 'InnerDomainToWallDistance',
        'MagnitudeRelaxationFactor', 'EddyViscosityRelaxationFactor', 'TimeVelPert', 'Limitor',
        'FactorOfCFDVorticityShed', 'RealignmentRelaxationFactor', 'MachLimitor']

    bool_Params = ['LowStorageIntegration']

    defaultParameters = {
        ############################################################################################
        ################################### Simulation Conditions ##################################
        ############################################################################################
            'Density'                       : 1.225,          #]0., +inf[, in kg.m^-3
            'EddyViscosityConstant'         : 0.15,           #[0., +inf[, constant for the eddy viscosity model, Cm(Mansour) around 0.1, Cs(Smagorinsky) around 0.15, Cr(Vreman) around 0.07
            'EddyViscosityModel'            : 'Vreman',       #Mansour, Mansour2, Smagorinsky, Vreman or None, select a LES model to compute the eddy viscosity
            'KinematicViscosity'            : 1.46e-5,        #[0., +inf[, in m^2.s^-1
            'Temperature'                   : 288.15,         #]0., +inf[, in K
            'Time'                          : 0.,             #in s, keep track of the physical time
        ############################################################################################
        ###################################### VPM Parameters ######################################
        ############################################################################################
            'AntiStretching'                : 0.5,             #between 0 and 1, 0 means particle strength fully takes vortex stretching, 1 means the particle size fully takes the vortex stretching
            'DiffusionScheme'               : 'DVM',          #PSE, CSM or None. gives the scheme used to compute the diffusion term of the vorticity equation
            'RegularisationKernel'          : 'Gaussian',     #The available smoothing kernels are Gaussian, HOA, LOA, Gaussian3 and SuperGaussian
            'AntiDiffusion'                 : 0.5,             #between 0 and 1, the closer to 0, the more the viscosity affects the particle strength, the closer to 1, the more it affects the particle size
            'SmoothingRatio'                : 2.,             #in m, anywhere between 1.5 and 2.5, the higher the NumberSource, the smaller the Resolution and the higher the SmoothingRatio should be to avoid blowups, the HOA kernel requires a higher smoothing
            'VorticityEquationScheme'       : 'Transpose',    #Classical, Transpose or Mixed, The schemes used to compute the vorticity equation are the classical scheme, the transpose scheme (conserves total vorticity) and the mixed scheme (a fusion of the previous two)
            'MachLimitor'                   : 0.5,             #[0, +inf[, sets the maximum/minimun induced velocity a particle can have
            'StrengthVariationLimitor'      : 2.,             #[1, +inf[, gives the maximum variation the strength of a particle can have during an iteration
            'ParticleSizeVariationLimitor'  : 1.1,             #[1, +inf[, gives the maximum a particle can grow/shrink during an iteration
        ############################################################################################
        ################################### Numerical Parameters ###################################
        ############################################################################################
            'CurrentIteration'              : 0,              #follows the current iteration
            'IntegrationOrder'              : 1,              #[|1, 4|]1st, 2nd, 3rd or 4th order Runge Kutta. In the hybrid case, there must be at least as much Interfaces in the hybrid domain as the IntegrationOrder of the time integration scheme
            'LowStorageIntegration'         : True,           #[|0, 1|], states if the classical or the low-storage Runge Kutta is used
            'NumberOfLiftingLines'          : 0,              #[0, +inf[, number of LiftingLines
            'NumberOfLiftingLineSources'    : 0,              #[0, +inf[, total number of embedded source particles on the LiftingLines
            'NumberOfBEMSources'            : 0,              #[0, +inf[, total number of embedded Boundary Element Method particles on the solid boundaries
            'NumberOfCFDSources'            : 0,              #[0, +inf[, total number of embedded cfd particles on the Hybrid Inner Interface
            'NumberOfHybridSources'         : 0,              #[0, +inf[, total number of hybrid particles generated in the hybrid Domain
            'NumberOfNodes'                 : 0,              #[0, +inf[, total number of nodes in the velocity perturbation field grid
        ############################################################################################
        ##################################### Particles Control ####################################
        ############################################################################################
            'CutoffXmin'                    : -np.inf,        #in m, spatial Cutoff
            'CutoffXmax'                    : +np.inf,        #in m, spatial Cutoff
            'CutoffYmin'                    : -np.inf,        #in m, spatial Cutoff
            'CutoffYmax'                    : +np.inf,        #in m, spatial Cutoff
            'CutoffZmin'                    : -np.inf,        #in m, spatial Cutoff
            'CutoffZmax'                    : +np.inf,        #in m, spatial Cutoff
            'ForcedDissipation'             : 0.,             #in %/s, gives the % of strength particles looses per sec, usefull to kill unnecessary particles without affecting the LLs
            'MaximumAgeAllowed'             : 0,              #0 <=,  particles are eliminated after MaximumAgeAllowed iterations, if MaximumAgeAllowed == 0, they are not deleted
            'MaximumAngleForMerging'        : 0.,             #[0., 180.[ in deg,   maximum angle   allowed between two particles to be merged
            'MaximumMergingVorticityFactor' : 0.,             #in %, particles can be merged if their combined strength is below the given poucentage of the maximum strength on the blades
            'MinimumOverlapForMerging'      : 0.,             #[0., +inf[, if two particles have at least an overlap of MinimumOverlapForMerging*SigmaRatio, they are considered for merging
            'MinimumVorticityFactor'        : 0.,             #in %, sets the minimum strength kept as a percentage of the maximum strength on the blades
            'RedistributeParticlesBeyond'   : np.inf,         #do not redistribute particles if closer than RedistributeParticlesBeyond*Resolution from a LL
            'RedistributionKernel'          : None,           #M4Prime, M4, M3, M2, M1 or None, redistribution kernel used. the number gives the order preserved by the kernel, if None local splitting/merging is used
            'RedistributionPeriod'          : 0,              #frequency at which particles are redistributed, if 0 the particles are never redistributed
            'RealignmentRelaxationFactor'   : 0.05,             #[0., 1.[, is used during the relaxation process to realign the particles with their voticity and avoid having a non null divergence of the vorticity field
            'MagnitudeRelaxationFactor'     : 1.,             #[0., 1.[, is used during the relaxation process to change the magnitude of the particles to avoid having a non null divergence of the vorticity field
            'EddyViscosityRelaxationFactor' : 0.,             #[0., 1.[, is used during the relaxation process when updating the eddy viscosity constant to satisfy the transfert of enstrophy to the kinetic energy
            'RemoveWeakParticlesBeyond'     : np.inf,         #do not remove weak particles if closer than RemoveWeakParticlesBeyond*Resolution from a LL
            'ResizeParticleFactor'          : 3.,             #[0, +inf[, resize particles that grow/shrink RedistributeParticleSizeFactor * Sigma0 (i.e. Resolution*SmoothingRatio), if 0 then no resizing is done
            'StrengthRampAtbeginning'       : 50,             #[|0, +inf[|, limit the vorticity shed for the StrengthRampAtbeginning first iterations for the wake to stabilise
            'EnstrophyControlRamp'          : 100,            #[|0, +inf[|, number of iteration before the enstrophy relaxation is fully applied. If propeller -> nbr of iteration to make 1 rotation (60/(dt*rpm)). If wing -> nbr of iteration for the freestream to travel one wingspan (L/(Uinf*dt)).
        ############################################################################################
        ###################################### FMM Parameters ######################################
        ############################################################################################
            'FarFieldApproximationOrder'    : 8,              #[|6, 12|], order of the polynomial which approximates the far field interactions, the higher the more accurate and the more costly
            'IterationTuningFMM'            : 50,             #frequency at which the FMM is compared to the direct computation, gives the relative L2 error
            'NearFieldOverlappingRatio'     : 0.5,            #[0., 1.], Direct computation of the interactions between clusters that overlap by NearFieldOverlappingRatio, the smaller the more accurate and the more costly
            'NumberOfThreads'               : 'auto',         #number of threads of the machine used. If 'auto', the highest number of threads is set
            'TimeFMM'                       : 0.,             #in s, keep track of the CPU time spent for the FMM
            'ClusterSize'                   : 2**9,           #[|0, +inf[|, maximum number of particles per FMM cluster, better as a power of 2
        ############################################################################################
        ############################## Perturbation Field Parameters ###############################
        ############################################################################################
            'FMMPerturbationOverlappingRatio' : 0.5,
            'TimeVelocityPerturbation'        : 0.,
    }
    defaultHybridParameters = {
        ############################################################################################
        ################################ Hybrid Domain Parameters ################################
        ############################################################################################
            #'BCFarFieldName'                   : 'farfield',#the name of the farfield boundary condition from which the Outer Interface is searched
            'MaximumSubIteration'              : 100,       #[|0, +inf[|, gives the maximum number of sub-iterations when computing the strength of the particles generated from the vorticity on the Interfaces
            'NumberOfHybridInterfaces'         : 4.,         #|]0, +inf[|, number of interfaces in the Hybrid Domain from which hybrid particles are generated
            'NumberOfParticlesPerInterface'    : 500,        #[|0, +inf[|, number of particles generated per hybrid interface
            'OuterDomainToWallDistance'        : 0.3,        #]0, +inf[ in m, distance between the wall and the end of the Hybrid Domain
            'FactorOfCFDVorticityShed'         : 75.,        #[0, 100] in %, gives the percentage of vorticity from the Hybrid Domain that is converted into hybrid particles
            'RelaxationFactor'                 : 0.5,       #[0, +inf[, gives the relaxtion factor used for the relaxation process when computing the strength of the particles generated from the vorticity on the Interface
            'RelaxationThreshold'              : 1e-6,      #[0, +inf[ in m^3.s^-1, gives the convergence criteria for the relaxtion process when computing the strength of the particles generated from the vorticity on the Interface
    }
    defaultLiftingLineParameters = {
        ############################################################################################
        ################################# Lifting Lines Parameters #################################
        ############################################################################################
            'CirculationThreshold'             : 1e-4,                  #]0., +inf[, convergence criteria for the circulation sub-iteration process, somewhere between 1e-3 and 1e-6 is ok
            'CirculationRelaxationFactor'      : 1./3.,                 #]0., 1.], relaxation parameter of the circulation sub-iterations, somwhere between 0.1 and 1 is good, the more unstable the simulation, the lower it should be
            'IntegralLaw'                      : 'linear',              #uniform, tanhOneSide, tanhTwoSides or ratio, gives the type of interpolation of the circulation on the lifting lines
            'MaxLiftingLineSubIterations'      : 100,                   #[|0, +inf[|, max number of sub iteration when computing the LL circulations
            'MinNbShedParticlesPerLiftingLine' : 26,                    #[|10, +inf[|, minimum number of station for every LL from which particles are shed
            'ParticleDistribution'             : dict(kind = 'uniform', #uniform, tanhOneSide, tanhTwoSides or ratio, repatition law of the particles on the Lifting Lines
                                                  FirstSegmentRatio=2., #]0., +inf[, size of the particles at the root of the blades relative to Sigma0 (i.e. Resolution*SmoothingRatio)
                                                  LastSegmentRatio=0.5, #]0., +inf[, size of the particles at the tip  of the blades relative to Sigma0 (i.e. Resolution*SmoothingRatio)
                                                  Symmetrical = False), #[|0, 1|], gives a symmetrical repartition of particles along the blades or not, if symmetrical, MinNbShedParticlesPerLiftingLine should be even
    }
    defaultParameters.update(VPMParameters)
    defaultParameters.update(PerturbationFieldParameters)
    if defaultParameters['NumberOfThreads'] == 'auto':
        defaultParameters['NumberOfThreads'] = int(os.getenv('OMP_NUM_THREADS', \
                                                                  len(os.sched_getaffinity(0))))
    else:
        defaultParameters['NumberOfThreads'] = min(defaultParameters['NumberOfThreads'], \
                                int(os.getenv('OMP_NUM_THREADS', len(os.sched_getaffinity(0)))))
    if EulerianMesh: defaultHybridParameters.update(HybridParameters)
    else: defaultHybridParameters = {}
    if LiftingLineTree: defaultLiftingLineParameters.update(LiftingLineParameters)
    else: defaultLiftingLineParameters = {}

    checkParametersTypes([defaultParameters, defaultHybridParameters,
                           defaultLiftingLineParameters], int_Params, float_Params, bool_Params)
    architecture = V.mpi_init(defaultParameters['NumberOfThreads'][0])
    print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))
    print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))
    print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))
    print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(' Launching VULCAINS ' + __version__ + ' '))
    print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))
    print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))
    print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))
    print(f"{'||':>57}\r" + '||' + '{:-^53}'.format(' CPU Architecture '))
    print(f"{'||':>57}\r" + '|| ' + '{:32}'.format('Number of threads') + ': ' + '{:d}'.format(architecture[0]))
    if architecture[1] == 2:
        print(f"{'||':>57}\r" + '|| ' + '{:32}'.format('SIMD') + ': ' + '{:d}'.format(architecture[1]) + ' (SSE)')
    elif architecture[1] == 4:
        print(f"{'||':>57}\r" + '|| ' + '{:32}'.format('SIMD') + ': ' + '{:d}'.format(architecture[1]) + ' (AVX)')
    elif architecture[1] == 8:
        print(f"{'||':>57}\r" + '|| ' + '{:32}'.format('SIMD') + ': ' + '{:d}'.format(architecture[1]) + ' (AVX512)')
    else: print(f"{'||':>57}\r" + '|| ' + '{:32}'.format('') + ': ' + '{:d}'.format(architecture[1]))
    if architecture[2]: print(f"{'||':>57}\r" + '|| ' + '{:32}'.format('Precison') + ': ' + 'double (64 bits)')
    else: print(f"{'||':>57}\r" + '|| ' + '{:32}'.format('Precison') + ': ' + 'single (32 bits)')
    print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))
    
    VPMLL.rotateLiftingLineSections(LiftingLineTree)
    VPMLL.renameLiftingLineTree(LiftingLineTree)
    VPMLL.updateLiftingLines(LiftingLineTree, defaultParameters, defaultLiftingLineParameters)
    VPMLL.updateParametersFromLiftingLines(LiftingLineTree, defaultParameters)
    tE = []
    t = buildEmptyVPMTree()
    Particles = pickParticlesZone(t)
    if LiftingLineTree:
        print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(' Initialisation of Lifting Lines '))
        LiftingLines = I.getZones(LiftingLineTree)
        VPMLL.initialiseParticlesOnLitingLine(t, LiftingLines, PolarsInterpolators, defaultParameters)
        print(f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Done '))
    if EulerianMesh:
        print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(' Initialisation of Hybrid Domain '))
        tE = H.checkEulerianField(EulerianMesh, defaultParameters, defaultHybridParameters)
        #tE = H.generateMirrorWing(EulerianMesh, defaultParameters, defaultHybridParameters)
        HybridDomain = H.generateHybridDomain(tE, defaultParameters, defaultHybridParameters)
        H.initialiseHybridParticles(t, tE, defaultParameters, defaultHybridParameters)
        J.set(Particles, '.Hybrid#Parameters', **defaultHybridParameters)
        I._sortByName(I.getNodeFromName1(Particles, '.Hybrid#Parameters'))
        print(f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Done '))

    J.set(Particles, '.VPM#Parameters', **defaultParameters)
    I._sortByName(I.getNodeFromName1(Particles, '.VPM#Parameters'))

    if PerturbationField:
        print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(' Initialisation of Perturbation Field '))
        t = I.merge([t, PerturbationField])
        NumberOfNodes = getParameter(t, 'NumberOfNodes')
        PerturbationFieldCapsule = V.build_perturbation_velocity_capsule(PerturbationField, NumberOfNodes)
        print(f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Done '))
    else: PerturbationFieldCapsule = None
    if LiftingLineTree:
        print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(' Generate Lifting Lines Particles '))
        t = I.merge([t, LiftingLineTree])
        VPMLL.ShedVorticitySourcesFromLiftingLines(t, PolarsInterpolators, 
                                        PerturbationFieldCapsule = PerturbationFieldCapsule)
        print(f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Done '))
    
    if EulerianMesh:
        print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(' Generate Hybrid Particles '))
        t = I.merge([t, HybridDomain])
        Nh = getParameter(t, 'NumberOfHybridSources')
        Nhcurrent = splitHybridParticles(t)
        Nh[0] = Nhcurrent
        print(f"{'||':>57}\r" + '||' + '{:27}'.format('Number of Hybrid particles') + ': ' + '{:d}'.format(Nh[0]))
        print(f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Inverse BEM Matrix '))
        H.updateBEMMatrix(t)
        print(f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Enforce BC on Solid '))
        H.updateBEMSources(t)
        print(f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Done '))

    induceVPMField(t, PerturbationFieldCapsule = PerturbationFieldCapsule)
    IterationCounter = I.getNodeFromName(t, 'IterationCounter')
    IterationCounter[1][0] = defaultParameters['IterationTuningFMM']*\
                                                           defaultParameters['IntegrationOrder']
    return t, tE, PerturbationFieldCapsule

def pickParticlesZone(t = []):
    '''
    gives the Zone containing the VPM particles.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

    Returns
    ----------
        Particles : Zone
            Particle Zone (if any).
    '''
    for z in I.getZones(t):
        if z[0] == 'Particles':
            return z
    return []

def pickPerturbationFieldZone(t = []):
    '''
    gives the Zone containing the Perturbation Field.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'PerturbationField'.

    Returns
    ----------
        PerturbationField : Zone
            Perturbation Field Zone (if any).
    '''
    for z in I.getZones(t):
        if z[0] == 'PerturbationField':
            return [z]
    return []

def getVPMParameters(t = []):
    '''
    Get a the VPM parameters.

    Parameters
    ----------
        t : Tree, Base, Zone.
            Containes the VPM parameters named '.VPM#Parameters'.

    Returns
    -------
        VPMParameter : :py:class:`dict`
            VPM parameters.
    '''
    return J.get(pickParticlesZone(t), '.VPM#Parameters')


def extractperturbationField(t = [], Targets = [], PerturbationFieldCapsule = []):
    '''
    Extract the Perturbation field velocities onto given nodes.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes the Perturbation Field.

        Targets : Tree, Base, Zone(s)
            Liftinglines.

        PerturbationFieldCapsule : :py:class:`capsule`
            Stores the FMM octree used to interpolate the Perturbation Mesh onto the particles.
    '''
    if PerturbationFieldCapsule:
        Theta, NumberOfNodes, TimeVelPert = getParameters(t, ['FMMPerturbationOverlappingRatio',
                                                       'NumberOfNodes', 'TimeVelocityPerturbation'])
        TimeVelPert[0] += V.extract_perturbation_velocity_field(C.newPyTree(I.getZones(Targets)),
                                        C.newPyTree(I.getZones(pickPerturbationFieldZone(t))),
                                              PerturbationFieldCapsule, NumberOfNodes[0], Theta[0])

def induceVPMField(t = [], IterationInfo = {}, PerturbationFieldCapsule = []):
    '''
    Computes the current velocity, velocity gradients, vorticity, diffusion and stretching of
    the VPM particles.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        IterationInfo : :py:class:`dict` of :py:class:`str`
            VPM solver information on the current iteration.

        PerturbationFieldCapsule : :py:class:`capsule`
            Stores the FMM octree used to interpolate the Perturbation Mesh onto the particles.

    Returns
    -------
        IterationInfo : :py:class:`dict` of :py:class:`str`
            VPM solver information on the current iteration.
    '''
    Scheme = Scheme_str2int[getParameter(t, 'VorticityEquationScheme')]
    Diffusion = DiffusionScheme_str2int[getParameter(t, 'DiffusionScheme')]
    solveVorticityEquationInfo = V.induce_vpm_field(t, C.newPyTree(I.getZones(pickPerturbationFieldZone(t))),
                                            PerturbationFieldCapsule, Scheme, Diffusion)
    IterationInfo['FMM time'] = solveVorticityEquationInfo[0]
    if PerturbationFieldCapsule:
        IterationInfo['Perturbation time'] = solveVorticityEquationInfo[-1]
        solveVorticityEquationInfo = solveVorticityEquationInfo[:-1]

    if len(solveVorticityEquationInfo) != 1:
        IterationInfo['Rel. err. of Velocity'] = solveVorticityEquationInfo[1]
        IterationInfo['Rel. err. of Vorticity'] = solveVorticityEquationInfo[2]
        IterationInfo['Rel. err. of Velocity Gradient'] = solveVorticityEquationInfo[3]
        if len(solveVorticityEquationInfo) == 5: 
            IterationInfo['Rel. err. of PSE'] = solveVorticityEquationInfo[4]
        if len(solveVorticityEquationInfo) == 6: 
            IterationInfo['Rel. err. of PSE'] = solveVorticityEquationInfo[4]
            IterationInfo['Rel. err. of Diffusion Velocity'] = solveVorticityEquationInfo[5]

    return IterationInfo

def updateSmagorinskyConstantAndComputeTurbulentViscosity(t = []):
    '''
    Updates the Smagorinski constant according to the variation of the Enstrophy and computes the turbulent viscosity model used by the diffusion term of the vorticity equation.
    This requires that the vorticity, velocity and its gradients have already been induced on
    the Lagrangian flow field. The Enstrophy of the previous time step is also needed.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.
    '''
    V.update_smagorinsky_constant_and_compute_turbulent_viscosity(t, EddyViscosityModel_str2int[getParameter(t,
                                                                         'EddyViscosityModel')])

def computeTurbulentViscosity(t = []):
    '''
    Compute the turbulent viscosity model used by the diffusion term of the vorticity equation.
    This requires that the vorticity, velocity and its gradients have already been induced on
    the Lagrangian flow field.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.
    '''
    V.compute_turbulent_viscosity(t, EddyViscosityModel_str2int[getParameter(t,
                                                                         'EddyViscosityModel')])

def computeNextTimeStep(t = [], PerturbationFieldCapsule = []):
    '''
    Advances the VPM field one time step forward.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        PerturbationFieldCapsule : :py:class:`capsule`
            Stores the FMM octree used to interpolate the Perturbation Mesh onto the particles.
    '''
    Particles = pickParticlesZone(t)
    LiftingLines = LL.getLiftingLines(t)
    HybridInterface = H.pickHybridDomainOuterInterface(t)
    
    time, dt, it, IntegOrder, lowstorage = getParameters(t,
            ['Time','TimeStep', 'CurrentIteration', 'IntegrationOrder','LowStorageIntegration'])
    if lowstorage:
        if IntegOrder == 1:
            a = np.array([0.], dtype = np.float64)
            b = np.array([1.], dtype = np.float64)
            #c = np.array([1.], dtype = np.float64)
        elif IntegOrder == 2:
            a = np.array([0., -0.5], dtype = np.float64)
            b = np.array([0.5, 1.], dtype = np.float64)
            #c = np.array([1./2., 1.], dtype = np.float64)
        elif IntegOrder == 3:
            a = np.array([0., -5./9., -153./128.], dtype = np.float64)
            b = np.array([1./3., 15./16., 8./15.], dtype = np.float64)
            #c = np.array([1./3., 3./4., 1.], dtype = np.float64)
        elif IntegOrder == 4:
            a = np.array([0., -1., -0.5, -4.], dtype = np.float64)
            b = np.array([1./2., 1./2., 1., 1./6.], dtype = np.float64)
            #c = np.array([1./2., 1./2., 3./2., 1.], dtype = np.float64)
        else:
            raise AttributeError('This Integration Scheme Has Not Been Implemented Yet')

        c = [0., b[0]]
        for i in range(1, len(b)):
            c += [b[i]*(a[i]/b[i - 1]*(c[-1] - c[-2]) + 1) + c[-1]]
        
        c = np.array(c, dtype = np.float64)

    else:
        if IntegOrder == 1:
            a = np.array([], dtype = np.float64)
            b = np.array([1.], dtype = np.float64)
        elif IntegOrder == 2:
            a = np.array([[0.5]], dtype = np.float64)
            b = np.array([0, 1.], dtype = np.float64)
        elif IntegOrder == 3:
            a = np.array([[0.5, 0.], [-1., 2.]], dtype = np.float64)
            b = np.array([1./6., 2./3., 1./6.], dtype = np.float64)
        elif IntegOrder == 4:
            a = np.array([[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 1.]], dtype = np.float64)
            b = np.array([1./6., 1./3., 1./3., 1./6.], dtype = np.float64)
        else:
            raise AttributeError('This Integration Scheme Has Not Been Implemented Yet')

        c = np.array([np.sum(ai) for ai in a] + [1.], dtype = np.float64)

    Scheme = Scheme_str2int[getParameter(t, 'VorticityEquationScheme')]
    Diffusion = DiffusionScheme_str2int[getParameter(t, 'DiffusionScheme')]
    EddyViscosityModel = EddyViscosityModel_str2int[getParameter(t, 'EddyViscosityModel')]
    if lowstorage:
        V.runge_kutta_low_storage(t, C.newPyTree(I.getZones(pickPerturbationFieldZone(t))), PerturbationFieldCapsule, a, b,
                                               c, Scheme, Diffusion, EddyViscosityModel)
    else:
        V.runge_kutta(t, C.newPyTree(I.getZones(pickPerturbationFieldZone(t))), PerturbationFieldCapsule, a, b, c,
                                                           Scheme, Diffusion,EddyViscosityModel)
    
    for LiftingLine in LiftingLines:
        TimeShed = I.getNodeFromName(LiftingLine, 'TimeSinceLastShedding')
        TimeShed[1][0] += dt[0]

    time += dt
    it += 1

def populationControl(t = [], IterationInfo = {}, NoRedistributionZones = []):
    '''
    Split, merge, resize and erase particles when necessary.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        IterationInfo : :py:class:`dict` of :py:class:`str`
            VPM solver information on the current iteration.
    '''
    IterationInfo['Population Control time'] = J.tic()
    LiftingLines = LL.getLiftingLines(t)
    Particles = pickParticlesZone(t)
    HybridInterface = H.pickHybridDomainOuterInterface(t)
    Np = Particles[1][0]
    AABB = []
    for LiftingLine in I.getZones(LiftingLines):
        x, y, z = J.getxyz(LiftingLine)
        AABB += [[min(x), min(y), min(z), max(x), max(y), max(z)]]

    for BC in I.getZones(HybridInterface):
        x, y, z = J.getxyz(BC)
        AABB += [[min(x), min(y), min(z), max(x), max(y), max(z)]]


    for Zone in I.getZones(NoRedistributionZones):
        x, y, z = J.getxyz(Zone)
        AABB += [[np.min(x), np.min(y), np.min(z), np.max(x), np.max(y), np.max(z)]]

    AABB = np.array(AABB, dtype = np.float64)
    RedistributionKernel = RedistributionKernel_str2int[getParameter(t, 'RedistributionKernel')]
    N0 = Np[0]
    populationControlInfo = np.array([0, 0, 0, 0, 0], dtype = np.int32)
    RedistributedParticles = V.population_control(t, AABB, RedistributionKernel,
                                                                          populationControlInfo)
    if RedistributedParticles.any():
        adjustTreeSize(t, NewSize = len(RedistributedParticles[0]), OldSize = N0)
        X, Y, Z = J.getxyz(Particles)
        AX, AY, AZ, S, Cvisq, Nu, Enstrophy, Age = J.getVars(Particles, vectorise('Alpha') + \
                                                         ['Sigma', 'Cvisq', 'Nu', 'Enstrophy', 'Age'])
        X[:]         = RedistributedParticles[0][:]
        Y[:]         = RedistributedParticles[1][:]
        Z[:]         = RedistributedParticles[2][:]
        AX[:]        = RedistributedParticles[3][:]
        AY[:]        = RedistributedParticles[4][:]
        AZ[:]        = RedistributedParticles[5][:]
        S[:]         = RedistributedParticles[6][:]
        Cvisq[:]     = RedistributedParticles[7][:]
        Nu[:]        = RedistributedParticles[8][:]
        Enstrophy[:] = RedistributedParticles[9][:]
        Age[:]       = np.array([int(a) for a in RedistributedParticles[10]], dtype = np.int32)
    else:
       adjustTreeSize(t, NewSize = Np[0], OldSize = N0)

    IterationInfo['Number of depleted particles'] = populationControlInfo[0]
    IterationInfo['Number of particles beyond cutoff'] = populationControlInfo[1]
    IterationInfo['Number of resized particles'] = populationControlInfo[2]
    IterationInfo['Number of split particles'] = populationControlInfo[3]
    IterationInfo['Number of merged particles'] = populationControlInfo[4]
    IterationInfo['Population Control time'] = J.tic() -IterationInfo['Population Control time']
    return IterationInfo

####################################################################################################
####################################################################################################
########################################### Vortex Rings ###########################################
####################################################################################################
####################################################################################################
def createLambOseenVortexRing(t, VPMParameters, VortexParameters):
    Gamma = VortexParameters['Intensity'][0]
    sigma = VPMParameters['Sigma0'][0]
    lmbd_s = VPMParameters['SmoothingRatio'][0]
    nu = VPMParameters['KinematicViscosity'][0]
    R = VortexParameters['RingRadius'][0]
    nc = [0]
    h = VPMParameters['Resolution'][0]
    
    if 'CoreRadius' in VortexParameters:
        a = VortexParameters['CoreRadius'][0]
        if nu != 0.: tau = a*a*2/4./nu
        else: tau = 0.
        VortexParameters['Tau'] = np.array([tau], dtype = np.float64, order = 'F')
    elif 'Tau' in VortexParameters:
        tau = VortexParameters['Tau'][0]
        a = np.sqrt(4.*nu*tau)
        VortexParameters['CoreRadius'] = np.array([a], dtype = np.float64, order = 'F')

    a = VortexParameters['CoreRadius'][0]
    w = lambda r : Gamma/(np.pi*a*a)*np.exp(-r*r/(a*a))

    if 'MinimumVorticityFraction' in VortexParameters:
        r = a*np.sqrt(-np.log(VortexParameters['MinimumVorticityFraction'][0]))
        nc = int(r/h)
        VortexParameters['NumberLayers'] = np.array([nc], dtype = np.int32, order = 'F')
    elif 'NumberLayers' in VortexParameters:
        nc = VortexParameters['NumberLayers'][0]
        frac = w(nc*h)
        VortexParameters['MinimumVorticityFraction'] = \
                                              np.array([frac], dtype = np.float64, order = 'F')

    N_s = 1 + 3*nc*(nc + 1)
    N_phi = int(2.*np.pi*R/h)
    N_phi += N_phi%4
    Np = N_s*N_phi
    extend(t, Np)
    Particles = pickParticlesZone(t)
    px, py, pz = J.getxyz(Particles)
    AlphaX, AlphaY, AlphaZ, VorticityX, VorticityY, VorticityZ, Sigma, Nu = \
                       J.getVars(Particles, vectorise(['Alpha', 'Vorticity']) + ['Sigma', 'Nu'])
    Nu[:] = nu
    
    r0 = h/2.
    rc = r0*(2*nc + 1)
    if (Np != N_phi*N_s): print("Achtung Bicyclette")
    if (R - rc < 0): print("Beware of the initial ring radius " , R , " < " , rc)
    else: print("R=", R, ", rc=", rc, ", a=", a, ", sigma=", sigma, ", nc=", nc, ", N_phi=",
                                                               N_phi, ", N_s=", N_s, ", N=", Np)

    X = [R]
    Z = [0.]
    W = [Gamma/(np.pi*a*a)]
    # V = [2.*np.pi*np.pi*R/N_phi*r0*r0]
    for n in range(1, nc + 1):
        for j in range(6*n):
            theta = np.pi*(2.*j + 1.)/6./n
            r = r0*(1. + 12.*n*n)/6./n
            X.append(R + r*np.cos(theta))
            Z.append(r*np.sin(theta))
            # V.append(4./3.*4.*np.pi*r0*r0/N_phi*(np.pi*R/2. + (np.sin(np.pi*(j + 1)/4./n) - \
            #                                        np.sin(np.pi*j/4./n))*(4.*n*n + 1./3.)*r0))
            W.append(W[0]*np.exp(-r*r/(a*a)))

    print(W[0])
    S = [lmbd_s*h]*len(X)
    for i in range(N_phi):
        phi = 2.*np.pi*i/N_phi
        for j in range(N_s):
            px[i*N_s + j] = X[j]*np.cos(phi)
            py[i*N_s + j] = X[j]*np.sin(phi)
            pz[i*N_s + j] = Z[j]
            Sigma[i*N_s + j] = S[j]
            VorticityX[i*N_s + j] = -W[j]*np.sin(phi)
            VorticityY[i*N_s + j] = W[j]*np.cos(phi)
            VorticityZ[i*N_s + j] = 0.
            AlphaX[i*N_s + j] = V[j]*VorticityX[i*N_s + j]
            AlphaY[i*N_s + j] = V[j]*VorticityY[i*N_s + j]
            AlphaZ[i*N_s + j] = V[j]*VorticityZ[i*N_s + j]

    VortexParameters['Nphi'] = N_phi
    VortexParameters['Ns'] = N_s
    VortexParameters['NumberLayers'] = nc
    V.adjust_vortex_ring(t, N_s, N_phi, Gamma, np.pi*r0*r0, np.pi*r0*r0*4./3., nc)

def createLambOseenVortexBlob(t, VPMParameters, VortexParameters):
    Gamma = VortexParameters['Intensity'][0]
    sigma = VPMParameters['Sigma0'][0]
    lmbd_s = VPMParameters['SmoothingRatio'][0]
    nu = VPMParameters['KinematicViscosity'][0]
    L = VortexParameters['Length'][0]
    frac = VortexParameters['MinimumVorticityFraction'][0]
    h = VPMParameters['Resolution'][0]
    if 'CoreRadius' in VortexParameters:
        a = VortexParameters['CoreRadius'][0]
        tau = a*a/4./nu
        VortexParameters['Tau'] = np.array([tau], dtype = np.float64, order = 'F')
    elif 'Tau' in VortexParameters:
        tau = VortexParameters['Tau'][0]
        a = np.sqrt(4.*nu*tau)
        VortexParameters['CoreRadius'] = np.array([a], dtype = np.float64, order = 'F')

    w = lambda r : Gamma/(4.*np.pi*nu*tau)*np.exp(-r*r/(4.*nu*tau))
    minw = w(0)*frac
    l = a*np.sqrt(-np.log(frac))
    l = int(l/h)*h
    L = int(L/h)*h
    NL = 2*int(L/h) + 1
    nc = int(l/h + 0.5) + 1
    Ns = 1 + 3*nc*(nc + 1)

    l = nc*h
    L = NL*h
    VortexParameters['RingRadius'] = l
    VortexParameters['Length'] = L
    VortexParameters['Nphi'] = NL
    VortexParameters['Ns'] = Ns
    VortexParameters['NumberLayers'] = nc
    Np = NL*Ns
    extend(t, Np)
    Particles = pickParticlesZone(t)
    px, py, pz = J.getxyz(Particles)
    AlphaX, AlphaY, AlphaZ, VorticityX, VorticityY, VorticityZ, Sigma, Nu = \
                       J.getVars(Particles, vectorise(['Alpha', 'Vorticity']) + ['Sigma', 'Nu'])
    Nu[:] = nu
    
    r0 = h/2.
    rc = r0*(2*nc + 1)
    print("L=", L, ", tau=", tau, ", rc=", l, ", a=", a, ", sigma=", sigma, ", NL=", NL, \
                                                                        ", Ns=", Ns, ", N=", Np)

    X = [0]
    Z = [0.]
    W = [Gamma/np.pi/a*a]
    V = [h*np.pi*r0*r0]
    for n in range(1, nc + 1):
        for j in range(6*n):
            theta = np.pi*(2.*j + 1.)/6./n
            r = r0*(1. + 12.*n*n)/6./n
            X.append(r*np.cos(theta))
            Z.append(r*np.sin(theta))
            V.append(V[0]*4./3.)
            W.append(W[0]*np.exp(-r*r/(a*a)))
    
    print("W in", w(0), np.min(W))
    S = [lmbd_s*h]*len(X)
    px[:Ns] = X[:]
    py[:Ns] = 0.
    pz[:Ns] = Z[:]
    Sigma[:Ns] = S[:]
    VorticityX[:Ns] = 0.
    VorticityY[:Ns] = W[:]
    VorticityZ[:Ns] = 0.
    AlphaX[:Ns] = 0.
    AlphaY[:Ns] = np.array(V[:])*np.array(W[:])
    AlphaZ[:Ns] = 0.
    for i in range(1, NL, 2):
        for j in range(Ns):
            px[i*Ns + j] = X[j]
            py[i*Ns + j] = (i//2 + 1)*h
            pz[i*Ns + j] = Z[j]
            Sigma[i*Ns + j] = S[j]
            VorticityX[i*Ns + j] = 0.
            VorticityY[i*Ns + j] = W[j]
            VorticityZ[i*Ns + j] = 0.
            AlphaX[i*Ns + j] = 0.
            AlphaY[i*Ns + j] = V[j]*VorticityY[i*Ns + j]
            AlphaZ[i*Ns + j] = 0.
        for j in range(Ns):
            px[(i + 1)*Ns + j] = X[j]
            py[(i + 1)*Ns + j] = -(i//2 + 1)*h
            pz[(i + 1)*Ns + j] = Z[j]
            Sigma[(i + 1)*Ns + j] = S[j]
            VorticityX[(i + 1)*Ns + j] = 0.
            VorticityY[(i + 1)*Ns + j] = W[j]
            VorticityZ[(i + 1)*Ns + j] = 0.
            AlphaX[(i + 1)*Ns + j] = 0.
            AlphaY[(i + 1)*Ns + j] = V[j]*VorticityY[(i + 1)*Ns + j]
            AlphaZ[(i + 1)*Ns + j] = 0.
    
    V.adjust_vortex_tube(t, Ns, NL, Gamma, np.pi*r0*r0, np.pi*r0*r0*4./3., nc)
