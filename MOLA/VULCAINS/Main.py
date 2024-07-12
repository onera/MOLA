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

This Vortex Particle Method solver can be used to simulate isolated vortex structures or 3D solids
with the Lifting Line module of MOLA or the FAST CFD solver.

Version:
0.5

Author:
Johan VALENTIN
'''

####################################################################################################
####################################### Import Python Modules ######################################
####################################################################################################
import numpy as np
import os
import sys

####################################################################################################
##################################### Import CASSIOPEE Modules #####################################
####################################################################################################
import Converter.PyTree as C
import Geom.PyTree as D
import Converter.Internal as I
import Generator.PyTree as G
import Transform.PyTree as T
import Connector.PyTree as CX
import CPlot.PyTree as CPlot

####################################################################################################
######################################## Import FAST Modules #######################################
####################################################################################################
try:
    import Fast.PyTree as Fast
    import FastS.PyTree as FastS
    import FastC.PyTree as FastC
except:
    pass

####################################################################################################
######################################## Import MOLA Modules #######################################
####################################################################################################
from .. import LiftingLine as LL
from .. import Wireframe as W
from .. import InternalShortcuts as J

####################################################################################################
######################################## VULCAINS Parameters #######################################
####################################################################################################
float_Params = ['Density', 'EddyViscosityConstant', 'KinematicViscosity', 'Temperature', 'Time',
    'VelocityFreestream', 'AntiStretching', 'AntiDiffusion', 'Resolution', 'Sigma0',
    'SmoothingRatio', 'TimeStep', 'MachLimitor',
    'StrengthVariationLimitor', 'ParticleSizeVariationLimitor', 'CutoffXmin', 'CutoffXmax',
    'CutoffYmin', 'CutoffYmax', 'CutoffZmin', 'CutoffZmax', 'ForcedDissipation',
    'MaximumAngleForMerging', 'MaximumMergingVorticityFactor', 'MinimumOverlapForMerging',
    'MinimumVorticityFactor', 'RedistributeParticlesBeyond', 'RealignmentRelaxationFactor',
    'MagnitudeRelaxationFactor', 'EddyViscosityRelaxationFactor', 'RemoveWeakParticlesBeyond',
    'ResizeParticleFactor', 'ClusterSizeFactor', 'NearFieldOverlapingFactor',
    'NearFieldSmoothingFactor', 'TimeFMM', 'PerturbationOverlappingFactor',
    'TimeVelocityPerturbation', 'CirculationThreshold', 'CirculationRelaxationFactor', 
    'LocalResolution', 'RPM', 'VelocityTranslation', 'EulerianTimeStep', 'GenerationZones',
    'HybridDomainSize', 'MinimumSplitStrengthFactor', 'RelaxationRatio', 'RelaxationThreshold',
                                                                                        'Intensity']
                                                                                        
int_Params = ['CurrentIteration', 'IntegrationOrder', 'LowStorageIntegration',
    'NumberOfLiftingLines', 'NumberOfLiftingLineSources', 'NumberOfBEMSources',
    'NumberOfCFDSources', 'NumberOfHybridSources', 'NumberOfNodes', 'MaximumAgeAllowed',
    'RedistributionPeriod', 'StrengthRampAtbeginning', 'EnstrophyControlRamp',
    'FarFieldPolynomialOrder', 'IterationCounter', 'IterationTuningFMM', 'MaxParticlesPerCluster',
    'MaxLiftingLineSubIterations', 'MinNbShedParticlesPerLiftingLine', 'NumberOfParticleSources',
    'EulerianSubIterations', 'HybridRedistributionOrder', 'InnerDomainCellLayer',
    'MaxHybridGenerationIteration', 'MaximumSourcesPerLayer', 'NumberOfBCCells',
              'NumberOfBEMUnknown', 'NumberOfHybridLayers', 'OuterDomainCellOffset', 'NumberLayers']

str_Params = ['EddyViscosityModel', 'DiffusionScheme', 'VorticityEquationScheme', 'IntegralLaw',
                                                                         'ParticleGenerationMethod']

def setType(val, dtype):
    '''
    Impose the type of the given variable and returns a 1D numpy array pointer.

    Parameters
    ----------

        val : :py:class:`float`, :py:class:`int`, :py:class:`dict` or :py:class:`str`
            Value on wich to impose the type.

        dtype : :py:class:`float`, :py:class:`int`, :py:class:`dict` or :py:class:`str`
            Type to impose.
    Returns
    -------
        array : :py:class:`numpy.ndarray`
            Pointer of the given val with the imposed type.
    '''
    return np.atleast_1d(np.array(val, order = 'F', dtype = object if (np.array(val) == None).any()
                                                                                        else dtype))

def checkRange(Parameters = {}, Range = {}):
    '''
    Resest the values in ``Parameters`` to be within ``Range``.

    Parameters
    ----------
        Parameters : :py:class:`dict`
            Contains the ``key`` parameters.

        Range : :py:class:`dict`
            Operable range of the parameters in ``Parameters``. Each range must be a list of 2
            :py:class:`float` or :py:class:`int`, or a list of :py:class:`str`.
    '''
    for key in Parameters:
        if key in str_Params:
            if Parameters[key] not in Range[key]:
                print(J.WARN + Parameters[key] + ' does not exist. Set to ' + str(Range[key][0])\
                                                                                           + J.ENDC)
                Parameters[key] = Range[key][0]
        elif type(Parameters[key]) != dict and (np.array(Parameters[key]) != None).all():
            if np.min(Parameters[key]) < Range[key][0]:
                print(J.WARN + key + ' = ' + str(np.min(Parameters[key])) + ' outside range. ' + \
                                                            'Set to ' + str(Range[key][0]) + J.ENDC)
                Parameters[key][Parameters[key] < Range[key][0]] = Range[key][0]
            if Range[key][-1] < np.max(Parameters[key]):
                print(J.WARN + key + ' = ' + str(np.max(Parameters[key])) + ' outside range. ' + \
                                                            'Set to ' + str(Range[key][1]) + J.ENDC)
                Parameters[key][Range[key][1] < Parameters[key]] = Range[key][1]

def getDefaultFluidParameters(Density               = 1.225,
                              KinematicViscosity    = 1.46e-5,
                              Temperature           = 288.15,
                              VelocityFreestream    = np.array([0., 0., 0.]),
                              **kwargs):
    '''
        Get a :py:class:`dict` containing all the relevant fluid parameters.

        Parameters
        ----------
            Density : :py:class:`float`
                :math:`\in [0, +\infty[` fluid density at infinity :math:`(kg \cdot m^{-3})`.
            
            KinematicViscosity : :py:class:`float`
                :math:`\in [0, +\infty[` fluid kinematic viscosity at freestream
                :math:`(m^2 \cdot s^{-1})`
            
            Temperature : :py:class:`float`
                :math:`\in [0, +\infty[` fluid temperature at freestream :math:`(K)`.
                    
            VelocityFreestream : :py:class:`numpy.ndarray` of 3 :py:class:`float`
                :math:`\in [-\infty, +\infty[`, fluid velocity at freestream in :math:`(m \cdot s^{-1})`
        Returns
        -------
            FluidParameters : :py:class:`dict`
                Free flow parameters
    '''
    for key in kwargs: print(J.WARN + key + ' does not exist. Parameter deleted.' + J.ENDC)
    
    FluidParametersRange = {
        'Density'               : [0., +np.inf],
        'KinematicViscosity'    : [0., +np.inf],
        'Temperature'           : [0., +np.inf],
        'VelocityFreestream'    : [-np.inf, +np.inf],
    }
    FluidParameters = dict(
        Density               = setType(Density, np.float64),
        KinematicViscosity    = setType(KinematicViscosity, np.float64),
        Temperature           = setType(Temperature, np.float64),
        VelocityFreestream    = setType(VelocityFreestream, np.float64),
    )
    checkRange(FluidParameters, FluidParametersRange)
    return FluidParameters

def getDefaultPrivateParameters(CurrentIteration           = 0,
                                IterationCounter           = 0,
                                NumberOfBEMSources         = 0,
                                NumberOfCFDSources         = 0,
                                NumberOfHybridSources      = 0,
                                NumberOfLiftingLines       = 0,
                                NumberOfLiftingLineSources = 0,
                                NumberOfNodes              = 0,
                                Sigma0                     = [None, None],
                                Time                       = 0.,
                                TimeFMM                    = 0.,
                                TimeVelocityPerturbation   = 0.,
                                **kwargs):
    '''
        Get a :py:class:`dict` containing all the relevant VPM numerical parameters.

        Parameters
        ----------
            CurrentIteration : :py:class:`int`
                :math:`\in [0, +\infty[` the current iteration (at restart)
            
            IterationCounter : :py:class:`int`
                :math:`\in [0, +\infty[` keeps track of how many iteration past since the last IterationTuningFMM.
            
            NumberOfBEMSources : :py:class:`int`
                :math:`\in [0, +\infty[` total number of embedded Boundary Element Method particles on the
                solid boundaries.
            
            NumberOfCFDSources : :py:class:`int`
                :math:`\in [0, +\infty[` total number of embedded Eulerian Immersed particles on the Hybrid
                Inner Interface.
            
            NumberOfHybridSources : :py:class:`int`
                :math:`\in [0, +\infty[` total number of hybrid particles generated in the Hybrid Domain.
            
            NumberOfLiftingLines : :py:class:`int`
                :math:`\in [0, +\infty[` number of LiftingLines.
            
            NumberOfLiftingLineSources : :py:class:`int`
                :math:`\in [0, +\infty[` total number of embedded source particles on the LiftingLines.
            
            NumberOfNodes : :py:class:`int`
                :math:`\in [0, +\infty[` total number of nodes in the velocity perturbation field grid.
            
            Sigma0 : :py:class:`numpy.ndarray` of 2 :py:class:`float` (or :py:obj:`None`)
                :math:`\in [0, +\infty[` initial minimum and maximum size of the
                particles, :math:`\sigma_0`
                    
            Time : :py:class:`float`
                :math:`\in [0, +\infty[` physical time :math:`(s)`.
            
            TimeFMM : :py:class:`float`
                :math:`\in [0, +\infty[`, keeps track of the CPU time spent by the FMM for the computation of the
                particle interactions, in s.
            
            TimeVelocityPerturbation : :py:class:`float`
                :math:`\in [0, +\infty[`, keeps track of the CPU time spent by the FMM for the interpolation of the
                perturbation mesh, in s.
        Returns
        -------
            NumericalParameters : :py:class:`dict`
                Numerical Parameters
    '''
    for key in kwargs: print(J.WARN + key + ' does not exist. Parameter deleted.' + J.ENDC)

    NumericalParametersRange = {
        'CurrentIteration'           : [0, +np.inf],
        'IterationCounter'           : [0, +np.inf],
        'NumberOfBEMSources'         : [0, +np.inf],
        'NumberOfCFDSources'         : [0, +np.inf],
        'NumberOfHybridSources'      : [0, +np.inf],
        'NumberOfLiftingLines'       : [0, +np.inf],
        'NumberOfLiftingLineSources' : [0, +np.inf],
        'NumberOfNodes'              : [0, +np.inf],
        'Sigma0'                     : [0., +np.inf],
        'Time'                       : [0., +np.inf],
        'TimeFMM'                    : [0., +np.inf],
        'TimeVelocityPerturbation'   : [0., +np.inf],
    }
    NumericalParameters = dict(
        CurrentIteration           = setType(CurrentIteration, np.int32),
        IterationCounter           = setType(IterationCounter, np.int32),
        NumberOfBEMSources         = setType(NumberOfBEMSources, np.int32),
        NumberOfCFDSources         = setType(NumberOfCFDSources, np.int32),
        NumberOfHybridSources      = setType(NumberOfHybridSources, np.int32),
        NumberOfLiftingLines       = setType(NumberOfLiftingLines, np.int32),
        NumberOfLiftingLineSources = setType(NumberOfLiftingLineSources, np.int32),
        NumberOfNodes              = setType(NumberOfNodes, np.int32),
        Sigma0                     = setType(Sigma0, np.float64),
        Time                       = setType(Time, np.float64),
        TimeFMM                    = setType(TimeFMM, np.float64),
        TimeVelocityPerturbation   = setType(TimeVelocityPerturbation, np.float64),
    )
    checkRange(NumericalParameters, NumericalParametersRange)
    return NumericalParameters

def getDefaultModelingParameters(AntiDiffusion                 = 0,
                                 AntiStretching                = 0,
                                 DiffusionScheme               = 'DVM',
                                 EddyViscosityConstant         = 0.15,
                                 EddyViscosityModel            = 'Vreman',
                                 EddyViscosityRelaxationFactor = 0.005,
                                 IntegrationOrder              = 1,
                                 LowStorageIntegration         = 1,
                                 MagnitudeRelaxationFactor     = 0.,
                                 RealignmentRelaxationFactor   = 0.,
                                 SmoothingRatio                = 2.,
                                 VorticityEquationScheme       = 'Transpose',
                                 **kwargs):
    '''
        Get a :py:class:`dict` containing all the relevant VPM modeling parameters.

        Parameters
        ----------
            AntiDiffusion : :py:class:`float`
                :math:`\in [0, 1]` vortex diffusion either modifying only the particle strength
                (AntiDiffusion = 0) or the particle size (AntiDiffusion = 1)
            
            AntiStretching : :py:class:`float`
                :math:`\in [0, 1]` vortex stretching either modifying only the particle strength
                (AntiStretching = 0) or the particle size (AntiStretching = 1)
            
            DiffusionScheme : :py:class:`str`
                Provides the scheme used to compute the diffusion term of the vorticity equation.
                ``'DVM'``
                ``'PSE'``
                ``'CSM'``
                :py:obj:`None`
            
            EddyViscosityConstant : :py:class:`float`
                :math:`\in [0, +\infty[` constant for the eddy viscosity model.
            
            EddyViscosityModel : :py:class:`str` or :py:obj:`None`
                Selects the eddy viscosity model. Possible options:
                ``'Mansour'``
                ``'Mansour2'``
                ``'Smagorinsky'``
                ``'Vreman'``
                :py:obj:`None`
            
            EddyViscosityRelaxationFactor : :py:class:`float`
                :math:`\in [0, 1)` modifies the EddyViscosityConstant of every particle by
                EddyViscosityRelaxationFactor at each iteration according to the local loss of
                Enstrophy of the particles.
            
            IntegrationOrder : :py:class:`int`
                :math:`\in [1, 4]` 1st, 2nd, 3rd or 4th order Runge Kutta for time-marching precision
            
            LowStorageIntegration : :py:class:`int`
                states if the classical (:py:obj:`False`) or the low-storage :py:obj:`True` Runge Kutta is used.
            
            MagnitudeRelaxationFactor : :py:class:`float`
                :math:`\in [0, +\infty[` filters the particles strength magnitude to have divergence-free
                vorticity field.
            
            RealignmentRelaxationFactor : :py:class:`float`
                :math:`\in [0, +\infty[` filters the particles direction to realign the particles with their
                vorticity to have divergence-free vorticity field.
            
            SmoothingRatio : :py:class:`float`
                :math:`\in [0, 5]` sets the ratio between Resolution and core size 
                (:math:`\sigma_0/h`). Big values smooth the particle interactions, avoiding
                singularities at induction, but deteriorates precision
            
            VorticityEquationScheme : :py:class:`str`
                The schemes used to compute the vortex stretching term of
                the vorticity equation. May be one of:
                ``'Transpose'``
                ``'Clasical'``
                ``'Mixed'``
        Returns
        -------
            ModelingParameters : :py:class:`dict`
                VPM Modeling Parameters
    '''
    for key in kwargs: print(J.WARN + key + ' does not exist. Parameter deleted.' + J.ENDC)
    
    ModelingParametersRange = {
        'AntiDiffusion'                 : [0., 1.],
        'AntiStretching'                : [0., 1.],
        'DiffusionScheme'               : ['DVM', 'PSE', 'CSM', None],
        'EddyViscosityConstant'         : [0., 1.],
        'EddyViscosityModel'            : ['Vreman', 'Mansour', 'Mansour2', 'Smagorinsky', None],
        'EddyViscosityRelaxationFactor' : [0., 1.],
        'IntegrationOrder'              : [1, 4],
        'LowStorageIntegration'         : [0, 1],
        'MagnitudeRelaxationFactor'     : [0., +np.inf],
        'RealignmentRelaxationFactor'   : [0., +np.inf],
        'SmoothingRatio'                : [1., +np.inf],
        'VorticityEquationScheme'       : ['Transpose', 'Mixed', 'Classical'],
    }
    ModelingParameters = dict(
        AntiDiffusion                 = setType(AntiDiffusion, np.float64),
        AntiStretching                = setType(AntiStretching, np.float64),
        DiffusionScheme               = np.str_(DiffusionScheme),
        EddyViscosityConstant         = setType(EddyViscosityConstant, np.float64),
        EddyViscosityModel            = np.str_(EddyViscosityModel),
        EddyViscosityRelaxationFactor = setType(EddyViscosityRelaxationFactor, np.float64),
        IntegrationOrder              = setType(IntegrationOrder, np.int32),
        LowStorageIntegration         = setType(LowStorageIntegration, np.int32),
        MagnitudeRelaxationFactor     = setType(MagnitudeRelaxationFactor, np.float64),
        RealignmentRelaxationFactor   = setType(RealignmentRelaxationFactor, np.float64),
        SmoothingRatio                = setType(SmoothingRatio, np.float64),
        VorticityEquationScheme       = np.str_(VorticityEquationScheme),
    )
    checkRange(ModelingParameters, ModelingParametersRange)
    return ModelingParameters

def getDefaultNumericalParameters(EnstrophyControlRamp          = 100,
                                  ForcedDissipation             = 0,
                                  MachLimitor                   = 0.5,
                                  NumberOfThreads               = 'auto',
                                  ParticleSizeVariationLimitor  = 1.1,
                                  Resolution                    = [None, None],
                                  StrengthRampAtbeginning       = 50,
                                  StrengthVariationLimitor      = 2,
                                  TimeStep                      = None,
                                  ParticleControlParameters     = dict(
                                         CutoffXmin                    = -np.inf,
                                         CutoffXmax                    = +np.inf,
                                         CutoffYmin                    = -np.inf,
                                         CutoffYmax                    = +np.inf,
                                         CutoffZmin                    = -np.inf,
                                         CutoffZmax                    = +np.inf,
                                         MaximumAgeAllowed             = 0,
                                         MaximumAngleForMerging        = 90,
                                         MaximumMergingVorticityFactor = 100,
                                         MinimumOverlapForMerging      = 3,
                                         MinimumVorticityFactor        = 0.001,
                                         RedistributeParticlesBeyond   = 0,
                                         RedistributionPeriod          = 1,
                                         RemoveWeakParticlesBeyond     = 0,
                                         ResizeParticleFactor          = 3, 
                                         ),
                                  FMMParameters                 = dict(
                                             ClusterSizeFactor             = 10.,
                                             FarFieldPolynomialOrder       = 8,
                                             IterationTuningFMM            = 50,
                                             MaxParticlesPerCluster        = 2**8,
                                             NearFieldOverlapingFactor     = 2,
                                             NearFieldSmoothingFactor      = 1,
                                             PerturbationOverlappingFactor = 1
                                             ),
                                  **kwargs):
    '''
        Get a :py:class:`dict` containing all the relevant VULCAINS numerical parameters.

        Parameters
        ----------
            EnstrophyControlRamp : :py:class:`int`
                :math:`\in [0, +\infty[` put a sinusoidal ramp on the Enstrophy filter applied to the particles
                for the EnstrophyControlRamp first iterations after the shedding of each particles.
            
            ForcedDissipation : :py:class:`float`
                :math:`\in [0, +\infty[` sets the percentage of strength the particles loose per second :math:`(%/s)`
            
            MachLimitor : :py:class:`float`
                :math:`\in [0, +\infty[` sets the maximum induced velocity a particle can have.
                Does not take into account the VelocityFreestream.
            
            NumberOfThreads : :py:class:`int` or ``'auto'``
                :math:`\geq 1`, number of threads of the machine used. If ``'auto'``, the
                highest number of threads is set.
            
            ParticleSizeVariationLimitor : :py:class:`float`
                :math:`\in [1, +\infty[` gives the maximum a particle can grow/shrink during an iteration.
            
            Resolution : :py:class:`numpy.ndarray` of 2 :py:class:`float` (or :py:obj:`None`)
                :math:`\in [0, +\infty[` Respectively minimum and maximum resolution scale of the VPM.
                It is usually noted :math:`h`.
            
            StrengthRampAtbeginning : :py:class:`int`
                :math:`\in [0, +\infty[` put a sinusoidal ramp on the magnitude of the vorticity shed for the
                StrengthRampAtbeginning first iterations of the simulation.
            
            StrengthVariationLimitor : :py:class:`float`
                :math:`\in [1, +\infty[` gives the maximum variation the strength of a 
                particle can have during an iteration (:math:`\max(\Delta ||\\alpha|| / \Delta t)`)
                (:math:`\max(||\\alpha||^{t+\Delta t} - ||\\alpha||^{t} / ||\\alpha||^{t})`)
            
            TimeStep : :py:class:`float`
                :math:`\in [0, +\infty[` time step :math:`\Delta t` of the VPM :math:`(s)`

                .. hint::
                    To help you in deciding the timestep, you may use 
                    :py:func:`setTimeStepFromBladeRotationAngle` or
                    :py:func:`setTimeStepFromShedParticles`

            ParticleControlParameters : :py:class:`dict`
                Contains all the parameters relevent to the redistribution (merging and splitting),
                and kill criteria of the particles, may it based on their age, strength, position,
                size ...

                CutoffXmin : :py:class:`float`
                :math:`\in ]-\infty, +\infty[` particles below this spatial cutoff are deleted :math:`(m)`.
            
                CutoffXmax : :py:class:`float`
                    :math:`\in ]-\infty, +\infty[` particles beyond this spatial cutoff are deleted :math:`(m)`.
                
                CutoffYmin : :py:class:`float`
                    :math:`\in ]-\infty, +\infty[` particles below this spatial cutoff are deleted :math:`(m)`.
                
                CutoffYmax : :py:class:`float`
                    :math:`\in ]-\infty, +\infty[` particles beyond this spatial cutoff are deleted :math:`(m)`.
                
                CutoffZmin : :py:class:`float`
                    :math:`\in ]-\infty, +\infty[` particles below this spatial cutoff are deleted :math:`(m)`.
                
                CutoffZmax : :py:class:`float`
                    :math:`\in ]-\infty, +\infty[` particles beyond this spatial cutoff are deleted :math:`(m)`.
                    
                MaximumAgeAllowed : :py:class:`int`
                    :math:`\in [0, +\infty[` particles older than MaximumAgeAllowed iterations are deleted. If
                    ``MaximumAgeAllowed == 0``, they are not deleted (disabling this feature).
                
                MaximumAngleForMerging : :py:class:`float`
                    :math:`\in [0, 180)` maximum angle allowed between two particles to be merged, in deg.
                
                MaximumMergingVorticityFactor : :py:class:`float`
                    :math:`\in [0, +\infty[` particles that have their strength above (resp. below)
                    MaximumMergingVorticityFactor times the maximum particle strength of the Lifting
                    Line embedded particles and the hybrid particles combined are split (resp. merged),
                    in %.
                
                MinimumOverlapForMerging : :py:class:`float`
                    :math:`\in [0, +\infty[` particles are merged if their distance is below their size (:math:`\sigma`) times
                    **MinimumOverlapForMerging**.
                
                MinimumVorticityFactor : :py:class:`float`
                    :math:`\in [0, +\infty[` particles that have their strength below MinimumVorticityFactor times
                    the maximum particle strength of the Lifting Line embedded particles and the hybrid
                    particles combined are deleted, in %.
                    
                RedistributeParticlesBeyond : :py:class:`float`
                    :math:`\in [0, +\infty[` do not split/merge particles if closer than 
                    ``RedistributeParticlesBeyond*Resolution`` from any Lifting Line or Hybrid Domain.
                
                RedistributionPeriod : :py:class:`int`
                    :math:`\in [0, +\infty[` iteration frequency at which particles are tested for
                    splitting/merging. If 0 the particles are never split/merged.
                
                RemoveWeakParticlesBeyond : :py:class:`float`
                    :math:`\in [0, +\infty[` do not remove weak particles if closer than 
                    RedistributeParticlesBeyond*Resolution from any Lifting Line or Hybrid Domain.
                
                ResizeParticleFactor : :py:class:`float`
                    :math:`\in [0, +\infty[` resize particles that grow/shrink past ResizeParticleFactor their
                    original size (given by Sigma0). If ``ResizeParticleFactor == 0``, no resizing is done.


            FMMParameters : :py:class:`dict`
                Contains all the parameters relevent to the Fast Multipole Method for the
                acceleration of the inter-particular interactions.

                ClusterSizeFactor : :py:class:`float`
                    :math:`\in [0, +\infty[` FMM clusters smaller than Resolution*ClusterSizeFactor cannot be
                    divided into smaller clusters.
                
                FarFieldPolynomialOrder : :py:class:`int`
                    :math:`\in [4, 12)` order of the polynomial which approximates the long distance particle
                    interactions by the FMM, the higher the more accurate and the more costly.
                
                IterationTuningFMM : :py:class:`int`
                    :math:`\in [0, +\infty[` period at which the FMM is compared to the direct computation,
                    shows the relative L2 error made by the FMM approximation.
                
                MaxParticlesPerCluster : :py:class:`int`
                    :math:`\in [1, +\infty[` FMM clusters with less than MaxParticlesPerCluster particles cannot be
                    divided into smaller clusters.
                
                NearFieldOverlapingFactor : :py:class:`float`
                    :math:`\in [1, +\infty[` particle interactions are approximated by the FMM as soon as two
                    clusters of particles are separated by at least NearFieldOverlapingFactor the size
                    of the particles in the cluster, the higher the more accurate and the more costly.
                
                NearFieldSmoothingFactor : :py:class:`float`
                    :math:`\in [1, +\mathrm{NearFieldOverlapingFactor}]` particle interactions are smoothed as soon as two
                    clusters of particles are separated by at most NearFieldSmoothingFactor the size of
                    the particles in the cluster, the higher the more accurate and the more costly.

                PerturbationOverlappingFactor : :py:class:`float`
                    :math:`\in [1, +\infty[` perturbation grid interpolations are approximated by the FMM as soon as
                    two clusters of cells are separated by at least NearFieldOverlapingFactor the size
                    of the cluster, the higher the more accurate and the more costly.
        Returns
        -------
            NumericalParameters : :py:class:`dict`
                VULCAINS Numerical Parameters
    '''
    if 'RECURSIVE' in kwargs: return dict(ParticleControlParameters = ParticleControlParameters,
                                                                      FMMParameters = FMMParameters)
    
    defaultNumericalParameters = getDefaultNumericalParameters(**dict(RECURSIVE=True))
    PCP = defaultNumericalParameters['ParticleControlParameters']
    FMMP = defaultNumericalParameters['FMMParameters']
    for key in kwargs: print(J.WARN + key + ' does not exist. Parameter deleted.' + J.ENDC)

    NumericalParametersRange = {
        'EnstrophyControlRamp'          : [0, +np.inf],
        'ForcedDissipation'             : [0, +np.inf],
        'MachLimitor'                   : [0., 1.],
        'NumberOfThreads'               : [1, +np.inf],
        'ParticleSizeVariationLimitor'  : [1, +np.inf],
        'Resolution'                    : [0., +np.inf],
        'StrengthRampAtbeginning'       : [0, +np.inf],
        'StrengthVariationLimitor'      : [1, +np.inf],
        'TimeStep'                      : [0., +np.inf],
    }
    ParticleControlParametersRange = {
        'CutoffXmin'                    : [-np.inf, +np.inf],
        'CutoffXmax'                    : [-np.inf, +np.inf],
        'CutoffYmin'                    : [-np.inf, +np.inf],
        'CutoffYmax'                    : [-np.inf, +np.inf],
        'CutoffZmin'                    : [-np.inf, +np.inf],
        'CutoffZmax'                    : [-np.inf, +np.inf],
        'MaximumAgeAllowed'             : [0, +np.inf],
        'MaximumAngleForMerging'        : [0., 180.],
        'MaximumMergingVorticityFactor' : [0., +np.inf],
        'MinimumOverlapForMerging'      : [0., +np.inf],
        'MinimumVorticityFactor'        : [0., +np.inf],
        'RedistributeParticlesBeyond'   : [0., +np.inf],
        'RedistributionPeriod'          : [0, +np.inf],
        'RemoveWeakParticlesBeyond'     : [0., +np.inf],
        'ResizeParticleFactor'          : [1., +np.inf],
    }
    FMMParametersRange = {
        'ClusterSizeFactor'            : [0, +np.inf],
        'FarFieldPolynomialOrder'      : [4, 12],
        'IterationTuningFMM'           : [1, +np.inf],
        'MaxParticlesPerCluster'       : [1, +np.inf],
        'NearFieldOverlapingFactor'    : [0., +np.inf],
        'NearFieldSmoothingFactor'     : [0., +np.inf],
        'PerturbationOverlappingFactor' : [1., +np.inf],
    }
    NumericalParameters = dict(
        EnstrophyControlRamp          = setType(EnstrophyControlRamp, np.int32),
        ForcedDissipation             = setType(ForcedDissipation, np.float64),
        MachLimitor                   = setType(MachLimitor, np.float64),
        NumberOfThreads               = setType(NumberOfThreads, np.int32),
        ParticleSizeVariationLimitor  = setType(ParticleSizeVariationLimitor, np.float64),
        Resolution                    = setType(Resolution, np.float64),
        StrengthRampAtbeginning       = setType(StrengthRampAtbeginning, np.int32),
        StrengthVariationLimitor      = setType(StrengthVariationLimitor, np.float64),
        TimeStep                      = setType(TimeStep, np.float64),
    )
    PCP.update(ParticleControlParameters)
    ParticleControlParameters = dict(
        CutoffXmin                    = setType(PCP['CutoffXmin'], np.float64),
        CutoffXmax                    = setType(PCP['CutoffXmax'], np.float64),
        CutoffYmin                    = setType(PCP['CutoffYmin'], np.float64),
        CutoffYmax                    = setType(PCP['CutoffYmax'], np.float64),
        CutoffZmin                    = setType(PCP['CutoffZmin'], np.float64),
        CutoffZmax                    = setType(PCP['CutoffZmax'], np.float64),
        MaximumAgeAllowed             = setType(PCP['MaximumAgeAllowed'], np.int32),
        MaximumAngleForMerging        = setType(PCP['MaximumAngleForMerging'], np.float64),
        MaximumMergingVorticityFactor = setType(PCP['MaximumMergingVorticityFactor'], np.float64),
        MinimumOverlapForMerging      = setType(PCP['MinimumOverlapForMerging'], np.float64),
        MinimumVorticityFactor        = setType(PCP['MinimumVorticityFactor'], np.float64),
        RedistributeParticlesBeyond   = setType(PCP['RedistributeParticlesBeyond'], np.float64),
        RedistributionPeriod          = setType(PCP['RedistributionPeriod'], np.int32),
        RemoveWeakParticlesBeyond     = setType(PCP['RemoveWeakParticlesBeyond'], np.float64),
        ResizeParticleFactor          = setType(PCP['ResizeParticleFactor'], np.float64),
    )
    FMMP.update(FMMParameters)
    FMMParameters = dict(
        ClusterSizeFactor             = setType(FMMP['ClusterSizeFactor'], np.float64),
        FarFieldPolynomialOrder       = setType(FMMP['FarFieldPolynomialOrder'], np.int32),
        IterationTuningFMM            = setType(FMMP['IterationTuningFMM'], np.int32),
        MaxParticlesPerCluster        = setType(FMMP['MaxParticlesPerCluster'], np.int32),
        NearFieldOverlapingFactor     = setType(FMMP['NearFieldOverlapingFactor'], np.float64),
        NearFieldSmoothingFactor      = setType(FMMP['NearFieldSmoothingFactor'], np.float64),
        PerturbationOverlappingFactor = setType(FMMP['PerturbationOverlappingFactor'], np.float64),
    )
    checkRange(NumericalParameters, NumericalParametersRange)
    checkRange(ParticleControlParameters, ParticleControlParametersRange)
    checkRange(FMMParameters, FMMParametersRange)
    NumericalParameters['ParticleControlParameters'] = ParticleControlParameters
    NumericalParameters['FMMParameters'] = FMMParameters
    return NumericalParameters

def getDefaultLiftingLineParameters(CirculationThreshold             = 1e-4,
                                    CirculationRelaxationFactor      = 1./3.,
                                    IntegralLaw                      = 'linear',
                                    MaxLiftingLineSubIterations      = 100,
                                    MinNbShedParticlesPerLiftingLine = 26,
                                    NumberOfParticleSources          = 50,
                                    SourcesDistribution              = dict(
                                                          kind              = 'tanhTwoSides',
                                                          FirstSegmentRatio = 2.,
                                                          LastSegmentRatio  = 0.5,
                                                          Symmetrical       = False),
                                    LocalResolution                  = None,
                                    RPM                              = None,
                                    VelocityTranslation              = None,
                                    **kwargs):
    '''
        Get a :py:class:`dict` containing all the relevant Lifting Lines parameters.

        .. hint:: These parameters are only imposed if they are not already present in the
                  .VPM#Parameters of the Lifting Lines. Each Lifting Line can have its own set of parameters.

        Parameters
        ----------
            CirculationThreshold : :py:class:`float`
                :math:`\in ]0., 1]`, convergence criteria for the circulation sub-iteration process to shed the
                particles from the Lifting Lines.
                
            CirculationRelaxationFactor : :py:class:`float`
                :math:`\in ]0., 1.]`, relaxation parameter of the circulation sub-iterations, the more unstable
                the simulation, the lower it should be.
                
            IntegralLaw : :py:class:`str`
                linear, uniform, tanhOneSide, tanhTwoSides or ratio`, gives the type of interpolation
                of the circulation from the Lifting Lines sections onto the particles sources
                embedded on the Lifting Lines.
                
            MaxLiftingLineSubIterations : :py:class:`int`
                :math:`\in [0, +\infty[`, max number of sub-iteration when sheding the particles from the Lifting
                Lines.
                
            MinNbShedParticlesPerLiftingLine : :py:class:`int`
                :math:`\in [0, +\infty[`, Lifting Lines cannot have less than MinNbShedParticlesPerLiftingLine
                particle sources. This parameter is imposed indiscriminately on all the Lifting
                Lines.
                
            NumberOfParticleSources : :py:class:`int`
                :math:`\in [26, +\infty[`, number of particle sources on the Lifting Lines. Will impose
                :py:class:`LocalResolution` as:
                :: LocalResolution = MOLA.Wireframe.getLength(LiftingLine)/NumberOfParticleSources
                
            SourcesDistribution : :py:class:`dict`
                Provides with the repartition of the particle sources on the Lifting Lines.

                    kind : :py:class:`str`
                        uniform, tanhOneSide, tanhTwoSides or ratio, repatition law of the particle.
                        
                    FirstSegmentRatio : :py:class:`float`
                        :math:`\in ]0., +\infty[`, particles at the root of the Lifting Line are spaced by
                        FirstSegmentRatio times the LocalResolution.
                        
                    LastSegmentRatio : :py:class:`float`
                        :math:`\in ]0., +\infty[`, particles at the tip of the Lifting Line are spaced by
                        LastSegmentRatio times the LocalResolution.
                        
                    Symmetrical : :py:class:`bool`
                        :math:`\in [0, 1]`, forces or not the symmetry of the particle sources on the Lifting
                        Lines.

            LocalResolution : :py:class:`float`
                :math:`\in ]0, +\infty[`, resolution of the Lifting Line, i.e., mean distance between the
                particle sources. If undefined, :py:class:`NumberOfParticleSources` is imposed as:
                :: NumberOfParticleSources = int(round(MOLA.Wireframe.getLength(LiftingLine)/LocalResolution))

            RPM : :py:class:`float`
                :math:`\in [0, +\infty[`, revolution per minute of the Lifting Lines, rev.min-1

            VelocityTranslation : numpy.ndarray of :py:class:`float`
                :math:`\in ]-\infty, +\infty[^3`, translation velocity of the Lifting Lines.
        Returns
        -------
            LiftingLinesParameters : :py:class:`dict`
                 Lifting Lines Parameters
    '''
    for key in kwargs: print(J.WARN + key + ' does not exist. Parameter deleted.' + J.ENDC)

    LiftingLineParametersRange = {
        'CirculationThreshold'             : [0., 1.],
        'CirculationRelaxationFactor'      : [0., 1.],
        'IntegralLaw'                      : ['linear', 'uniform', 'tanhOneSide', 'tanhTwoSides',
                                                                                           'ratio'],
        'MaxLiftingLineSubIterations'      : [0, +np.inf],
        'MinNbShedParticlesPerLiftingLine' : [0, +np.inf],
        'NumberOfParticleSources'          : [26, +np.inf],
        'LocalResolution'                  : [0, +np.inf],
        'RPM'                              : [-np.inf, +np.inf],
        'VelocityTranslation'              : [-np.inf, +np.inf],
    }
    LiftingLineParameters = dict(
        CirculationThreshold             = setType(CirculationThreshold, np.float64),
        CirculationRelaxationFactor      = setType(CirculationRelaxationFactor, np.float64),
        IntegralLaw                      = np.str_(IntegralLaw),
        MaxLiftingLineSubIterations      = setType(MaxLiftingLineSubIterations, np.int32),
        MinNbShedParticlesPerLiftingLine = setType(MinNbShedParticlesPerLiftingLine, np.int32),
        NumberOfParticleSources          = setType(NumberOfParticleSources, np.int32),
        SourcesDistribution              = dict(SourcesDistribution),
        LocalResolution                  = setType(LocalResolution, np.float64),
        RPM                              = setType(RPM, np.float64),
        VelocityTranslation              = setType(VelocityTranslation, np.float64),
    )
    checkRange(LiftingLineParameters, LiftingLineParametersRange)
    return LiftingLineParameters

def getDefaultHybridParameters(EulerianSubIterations        = 30,
                               EulerianTimeStep             = None,
                               GenerationZones              = np.array([[-1.]*3 + [1.]*3])*np.inf,
                               HybridDomainSize             = None,
                               HybridRedistributionOrder    = 2,
                               InnerDomainCellLayer         = 0,
                               MaxHybridGenerationIteration = 50,
                               MaximumSourcesPerLayer       = 1000,
                               MinimumSplitStrengthFactor   = 1.,
                               NumberOfBCCells              = 1,
                               NumberOfBEMUnknown           = 0,
                               NumberOfHybridLayers         = 5,
                               OuterDomainCellOffset        = 2,
                               ParticleGenerationMethod     = 'BiCGSTAB',
                               RelaxationRatio              = 1,
                               RelaxationThreshold          = 1e-3,
                               **kwargs):
    '''
        Get a :py:class:`dict` containing all the relevant Hybrid parameters.

        Parameters
        ----------
            EulerianSubIterations : :py:class:`int`
                :math: `\in [0, +\infty[`, number of sub-iterations for the Eulerian solver
            
            EulerianTimeStep : :py:class:`float`
                :math: `\in ]0., +\infty[`, timestep for the Eulerian solver, in s.

            GenerationZones : list of :py:class:`list` or list of numpy.ndarray of :py:class:`float`
                The Eulerian vorticity sources are only considered if within GenerationZones, in
                :math:`m`. Each list sets the limits of given box
                ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
            
            HybridDomainSize : :py:class:`float`
                :math: `\in ]0., +\infty[`, size of the Hybrid Domain contained between the Outer and Inner
            
            HybridRedistributionOrder : :py:class:`int`
                :math: `\in [1, 5]`, order of the polynomial used to redistribute the generated particles on a
                regular cartesian grid.
            
            InnerDomainCellLayer : :py:class:`int`
                :math: `\in ]0, +\infty[`, gives the position of the beginning of the Hybrid Domain, i.e., the
                position of the Inner Interface. The Hybrid Domain is thus starts 2 ghost cells +
                NumberOfBCCells + OuterDomainCellOffset + InnerDomainCellLayer layers of cells from
                the exterior boundary of the Eulerian mesh.
            
            MaxHybridGenerationIteration : :py:class:`int`
                :math: `\in [0, +\infty[`, max number of sub-iterations for the iterative particle generation
                method.
            
            MaximumSourcesPerLayer : :py:class:`int`
                :math: `\in [0, +\infty[`, max number of vorticity sources in each layer of the Hybrid Domain.
            
            MinimumSplitStrengthFactor : :py:class:`float`
                :math: `\in [0., +\infty[`, in %, sets the minimum particle strength kept per layer after generation
                of the hybrid particles. The strength threshold is set as a percentage of the
                maximum strength in the hybrid domain times MinimumSplitStrengthFactor.
            
            NumberOfBCCells : :py:class:`int`
                :math: `\in ]0, +\infty[`, number of layers cells on which the BC farfield is imposed by the VPM.
            
            NumberOfBEMUnknown : :py:class:`int`
                :math: `\in [0, 3]`, number of unknown for the BEM. If NumberOfBEMUnknown == 0: sources and
                vortex sheets are given with an initial guess but not solved. If NumberOfBEMUnknown
                == 1: only sources are solved. If NumberOfBEMUnknown == 2: only vortex sheets are
                solved. If NumberOfBEMUnknown == 3: both sources and vortex sheets are solved.
            
            NumberOfHybridLayers : :py:class:`int`
                :math: `\in ]0, +\infty[`, number of layers dividing the Hybrid Domain.
            
            OuterDomainCellOffset : :py:class:`int`
                :math: `\in ]0, +\infty[`, offsets the position of the Hybrid Domain by OuterDomainCellOffset from
                the far field BC imposed by the VPM. The Hybrid Domain thus ends 2 ghost cells +
                NumberOfBCCells + OuterDomainCellOffset layers of cells from the exterior boundary
                of the Eulerian mesh.
            
            ParticleGenerationMethod : :py:class:`str`
                Gives the iterative methode to compute the strength of the hybrid particles fom the Eulerian
                vorticity sources. Selects the Generalized Minimal Residual, Bi-Conjugate Gradient
                Stabilised, Conjugate Gradient or Direct Resolution from LAPACKE. May be one of:
                ``'GMRES'``
                ``'BiCGSTAB'``
                ``'CG'``
                ``'Direct'``
            
            RelaxationRatio : :py:class:`float`
                :math: `\in [0, +\infty[`, dynamically updates the iterative method convergence criteria for the
                relative error of the vorticity induced by the generated particles to be as close as
                possible to RelaxationThreshold.
            
            RelaxationThreshold : :py:class:`float`
                :math: `\in [0, +\infty[` in m^3.s^-1, gives the convergence criteria for the iterative particle
                generation method.
        Returns
        -------
            HybridParameters : :py:class:`dict`
                Hybrid parameters
    '''
    for key in kwargs: print(J.WARN + key + ' does not exist. Parameter deleted.' + J.ENDC)

    HybridParametersRange = {
        'EulerianSubIterations'        : [0, +np.inf],
        'EulerianTimeStep'             : [0., +np.inf],
        'GenerationZones'              : [-np.inf, +np.inf],
        'HybridDomainSize'             : [0., +np.inf],
        'HybridRedistributionOrder'    : [1, 5],
        'InnerDomainCellLayer'         : [0, +np.inf],
        'MaxHybridGenerationIteration' : [0, +np.inf],
        'MaximumSourcesPerLayer'       : [0, +np.inf],
        'MinimumSplitStrengthFactor'   : [0., +np.inf],
        'NumberOfBCCells'              : [0, +np.inf],
        'NumberOfBEMUnknown'           : [0, 3],
        'NumberOfHybridLayers'         : [0, +np.inf],
        'OuterDomainCellOffset'        : [0, +np.inf],
        'ParticleGenerationMethod'     : ['BiCGSTAB', 'GMRES', 'CG', 'Direct'],
        'RelaxationRatio'              : [0, +np.inf],
        'RelaxationThreshold'          : [0, +np.inf],
    }
    HybridParameters = dict(
        EulerianSubIterations        = setType(EulerianSubIterations, np.int32),
        EulerianTimeStep             = setType(EulerianTimeStep, np.float64),
        GenerationZones              = setType(GenerationZones, np.float64),
        HybridDomainSize             = setType(HybridDomainSize, np.float64),
        HybridRedistributionOrder    = setType(HybridRedistributionOrder, np.int32),
        InnerDomainCellLayer         = setType(InnerDomainCellLayer, np.int32),
        MaxHybridGenerationIteration = setType(MaxHybridGenerationIteration, np.int32),
        MaximumSourcesPerLayer       = setType(MaximumSourcesPerLayer, np.int32),
        MinimumSplitStrengthFactor   = setType(MinimumSplitStrengthFactor, np.float64),
        NumberOfBCCells              = setType(NumberOfBCCells, np.int32),
        NumberOfBEMUnknown           = setType(NumberOfBEMUnknown, np.int32),
        NumberOfHybridLayers         = setType(NumberOfHybridLayers, np.int32),
        OuterDomainCellOffset        = setType(OuterDomainCellOffset, np.int32),
        ParticleGenerationMethod     = np.str_(ParticleGenerationMethod),
        RelaxationRatio              = setType(RelaxationRatio, np.float64),
        RelaxationThreshold          = setType(RelaxationThreshold, np.float64),
    )
    checkRange(HybridParameters, HybridParametersRange)
    return HybridParameters

####################################################################################################
########################################### Miscellaneous ##########################################
####################################################################################################
Scheme_str2int = {'Transpose': 0, 'Mixed': 1, 'Classical': 2}
EddyViscosityModel_str2int = {'Vreman': 1, 'Mansour': 2, 'Mansour2': 3, 'Smagorinsky': 4, None: 0}
RedistributionKernel_str2int = {'M4Prime': 5, 'M4': 4, 'M3': 3, 'M2': 2, 'M1': 1, None: 0}
DiffusionScheme_str2int = {'PSE': 1, 'DVM': 2, 'CSM': 3, None: 0}

Vector_VPM_FlowSolution = []
for var in ['VelocityInduced', 'VelocityPerturbation', 'VelocityDiffusion', 'gradxVelocity',
                       'gradyVelocity', 'gradzVelocity', 'PSE', 'Vorticity', 'Alpha', 'Stretching']:
    Vector_VPM_FlowSolution += [var + v for v in 'XYZ']

Scalar_VPM_FlowSolution = ['Age', 'Sigma', 'Cvisq', 'Nu', 'divUd', 'Enstrophyf', 'Enstrophy',
                                                                                  'EnstrophyM1']
VPM_FlowSolution = Vector_VPM_FlowSolution + Scalar_VPM_FlowSolution
Method_str2int = {'Direct': 0, 'direct': 0, 'CG': 1, 'cg': 1, 'Conjugate Gradient': 1,
                  'conjugate gradient': 1, 'ConjugateGradient': 1, 'BiCGSTAB': 2, 'bicgstab': 2,
                  'Bi-Conjugate Gradient Stabilised': 2, 'bi-conjugate gradient stabilised': 2,
                  'GMRES': 3, 'gmres': 3, 'Generalized Minimal Residual Method': 3,
                  'generalized minimal residual method': 3}
# TimeStepFunction_str2int = {'setTimeStepFromBladeRotationAngle':
# setTimeStepFromBladeRotationAngle, 'shed': setTimeStepFromShedParticles,
# 'BladeRotationAngle': setTimeStepFromBladeRotationAngle, 'setTimeStepFromShedParticles':
# setTimeStepFromShedParticles, 'ShedParticles': setTimeStepFromShedParticles, 'Angle':
# setTimeStepFromBladeRotationAngle, 'angle': setTimeStepFromBladeRotationAngle, 'Shed':
# setTimeStepFromShedParticles}

union = lambda t: T.join(C.convertArray2Hexa(G.close(CX.connectMatch(t))))

####################################################################################################
######################################### Global Variables #########################################
####################################################################################################
logo = '\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mM\x1b[37mB\x1b[37mK\x1b[37m8\x1b[37mA\x1b[37mk\x1b[37mb\x1b[37mb\x1b[37mY\x1b[37mk\x1b[37m&\x1b[37m$\x1b[37mH\x1b[37mW\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mB\x1b[37mA\x1b[37m4\x1b[35mF\x1b[35m#\x1b[35mo\x1b[34m[\x1b[34mI\x1b[34m{\x1b[34mr\x1b[34mc\x1b[34mc\x1b[34mc\x1b[34mc\x1b[34mc\x1b[34mc\x1b[34ml\x1b[34mr\x1b[34m*\x1b[34m?\x1b[34mt\x1b[35mj\x1b[35mf\x1b[37mS\x1b[37mG\x1b[37m$\x1b[37mM\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mK\x1b[37mX\x1b[35m3\x1b[34mo\x1b[34m}\x1b[34ml\x1b[34m%\x1b[34m%\x1b[34m%\x1b[34mx\x1b[34mx\x1b[34mc\x1b[34mc\x1b[34mc\x1b[34ml\x1b[34ml\x1b[34ml\x1b[34ml\x1b[34ml\x1b[34ml\x1b[34mc\x1b[34mc\x1b[34mc\x1b[34mc\x1b[34mx\x1b[34mx\x1b[34m%\x1b[34m%\x1b[34mx\x1b[34mr\x1b[34m]\x1b[35mn\x1b[37mq\x1b[37mO\x1b[37mN\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mU\x1b[37mS\x1b[34mj\x1b[34m}\x1b[34mc\x1b[34mc\x1b[34ml\x1b[34ml\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34mr\x1b[34ml\x1b[34mc\x1b[34mc\x1b[34mr\x1b[34m[\x1b[35mw\x1b[37mP\x1b[37mW\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mW\x1b[37mE\x1b[34mL\x1b[34m*\x1b[34ml\x1b[34mr\x1b[34ms\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34m{\x1b[34ms\x1b[34ml\x1b[34mr\x1b[34m[\x1b[35m2\x1b[37m&\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mB\x1b[37mS\x1b[34mt\x1b[34m{\x1b[34m{\x1b[34m*\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m}\x1b[34m*\x1b[34ms\x1b[34m}\x1b[34mT\x1b[37mb\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mM\x1b[37mh\x1b[34mt\x1b[34m*\x1b[34m}\x1b[34mI\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34m?\x1b[34mI\x1b[34m*\x1b[34m}\x1b[34m#\x1b[37m&\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mO\x1b[34mL\x1b[34mI\x1b[34m?\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34m!\x1b[34mI\x1b[34m!\x1b[36m6\x1b[37mW\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m0\x1b[37mg\x1b[34m[\x1b[34m!\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m]\x1b[34m?\x1b[34mu\x1b[37m8\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mW\x1b[36my\x1b[34m?\x1b[34m]\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34mt\x1b[34mt\x1b[34mt\x1b[34mt\x1b[34mt\x1b[34m1\x1b[34m1\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m]\x1b[34m]\x1b[34m]\x1b[34m[\x1b[34m[\x1b[34m[\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m1\x1b[34m[\x1b[34me\x1b[37mY\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mP\x1b[36m2\x1b[36mq\x1b[36mg\x1b[37md\x1b[37m4\x1b[37mE\x1b[37mE\x1b[37mg\x1b[37mg\x1b[37mg\x1b[37mP\x1b[37mP\x1b[37mX\x1b[37mX\x1b[37mX\x1b[37mX\x1b[37mg\x1b[37mE\x1b[37m4\x1b[36mh\x1b[36mS\x1b[36mp\x1b[36m3\x1b[36mJ\x1b[34mu\x1b[34m7\x1b[34me\x1b[34mt\x1b[34m1\x1b[34m1\x1b[34mt\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34ma\x1b[34mt\x1b[34ma\x1b[37mY\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[37mW\x1b[37m@\x1b[37mA\x1b[37mG\x1b[36mV\x1b[36mq\x1b[36mF\x1b[36my\x1b[36mJ\x1b[36mT\x1b[36mu\x1b[36mL\x1b[36mj\x1b[34mz\x1b[34m7\x1b[34m7\x1b[34m7\x1b[34m7\x1b[34m7\x1b[34mz\x1b[36mj\x1b[36mL\x1b[36mu\x1b[36m#\x1b[36mC\x1b[36my\x1b[36mF\x1b[36mm\x1b[36mh\x1b[36md\x1b[37m4\x1b[36m4\x1b[36mg\x1b[36mp\x1b[36mf\x1b[36mn\x1b[34mz\x1b[34mo\x1b[34me\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34mo\x1b[34me\x1b[34m7\x1b[37m8\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[37mD\x1b[37m8\x1b[37mb\x1b[37mP\x1b[37mX\x1b[37mP\x1b[36mp\x1b[34ma\x1b[36me\x1b[36me\x1b[36ma\x1b[36ma\x1b[36ma\x1b[36ma\x1b[36me\x1b[36me\x1b[36mo\x1b[36m7\x1b[36mz\x1b[36mj\x1b[36mu\x1b[36mn\x1b[36mT\x1b[36m#\x1b[36m#\x1b[36mJ\x1b[36mJ\x1b[36m#\x1b[36m#\x1b[36mT\x1b[36mu\x1b[36mL\x1b[36mz\x1b[36m7\x1b[36m7\x1b[36mz\x1b[36mL\x1b[36mT\x1b[36mJ\x1b[36mf\x1b[36mw\x1b[36mT\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36mo\x1b[36mJ\x1b[37mM\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mM\x1b[37mW\x1b[37mN\x1b[37mR\x1b[37mQ\x1b[37mQ\x1b[37mM\x1b[36m#\x1b[36mo\x1b[36mj\x1b[36mn\x1b[36mw\x1b[36m5\x1b[36mS\x1b[36m4\x1b[37mG\x1b[37mO\x1b[37m8\x1b[37mK\x1b[37mD\x1b[37mW\x1b[37mN\x1b[37mM\x1b[37mM\x1b[37mM\x1b[37mM\x1b[37mM\x1b[37mM\x1b[37mW\x1b[37mW\x1b[37mW\x1b[37mW\x1b[37mB\x1b[37mB\x1b[37mD\x1b[37m@\x1b[37mU\x1b[37mO\x1b[37mG\x1b[36md\x1b[36mm\x1b[36m2\x1b[36mJ\x1b[36mu\x1b[36mz\x1b[36m7\x1b[36m7\x1b[36mz\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mj\x1b[36mo\x1b[36mg\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m$\x1b[36m4\x1b[37mk\x1b[37mK\x1b[37mN\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[37mW\x1b[37mK\x1b[37mU\x1b[36mA\x1b[36mb\x1b[36mP\x1b[36mg\x1b[36mE\x1b[36md\x1b[36mV\x1b[36mh\x1b[36mg\x1b[36mg\x1b[36mh\x1b[36md\x1b[36md\x1b[36m4\x1b[36mE\x1b[36mE\x1b[36mX\x1b[36mG\x1b[36mY\x1b[36m&\x1b[37m@\x1b[37mB\x1b[37mM\x1b[37mM\x1b[37mB\x1b[37m8\x1b[37mP\x1b[36mm\x1b[36mw\x1b[36mL\x1b[36mz\x1b[36mj\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mL\x1b[36mj\x1b[36mw\x1b[37mM\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m0\x1b[37mW\x1b[37m@\x1b[36mA\x1b[36mG\x1b[36mE\x1b[36mh\x1b[36mS\x1b[36mm\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36mS\x1b[36mZ\x1b[36m&\x1b[37mA\x1b[37mA\x1b[37mA\x1b[37mA\x1b[37m&\x1b[36mO\x1b[36mA\x1b[36mA\x1b[36mZ\x1b[36mg\x1b[36mh\x1b[36mS\x1b[36mq\x1b[36mh\x1b[36mX\x1b[36mA\x1b[37mH\x1b[37mR\x1b[37mR\x1b[37m@\x1b[36mX\x1b[36m2\x1b[36mn\x1b[36mL\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mn\x1b[36mu\x1b[37m8\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[37mB\x1b[37m@\x1b[37mU\x1b[37m$\x1b[37mB\x1b[37m0\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[37mD\x1b[37m8\x1b[36mb\x1b[36m4\x1b[36mg\x1b[36mm\x1b[36mp\x1b[36mp\x1b[36mp\x1b[36mp\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36mm\x1b[37mW\x1b[37m0\x1b[36mp\x1b[36m#\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mw\x1b[37mZ\x1b[37mW\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m0\x1b[37mN\x1b[37m@\x1b[36mY\x1b[36md\x1b[36m6\x1b[36mF\x1b[36mq\x1b[36mg\x1b[36m8\x1b[37mM\x1b[37m0\x1b[37m$\x1b[36mS\x1b[36mT\x1b[36mT\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36mn\x1b[36mP\x1b[37mN\x1b[37mB\x1b[37m@\x1b[37m&\x1b[37mb\x1b[36mg\x1b[36mh\x1b[36mm\x1b[36mq\x1b[36m4\x1b[37mY\x1b[37mK\x1b[37mM\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m0\x1b[37mW\x1b[37m$\x1b[36mb\x1b[36md\x1b[36mm\x1b[36mF\x1b[36m5\x1b[36mF\x1b[36mF\x1b[36mp\x1b[36mp\x1b[36mp\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36mp\x1b[36m6\x1b[37m$\x1b[37mQ\x1b[37m&\x1b[36m5\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36my\x1b[37mK\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m0\x1b[37mW\x1b[36mO\x1b[36mg\x1b[36mF\x1b[36mF\x1b[36mm\x1b[36mP\x1b[37mB\x1b[37mQ\x1b[37mA\x1b[36mf\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mw\x1b[36mf\x1b[36mJ\x1b[36m#\x1b[36mJ\x1b[36m3\x1b[36mS\x1b[36mG\x1b[37m$\x1b[37mN\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m0\x1b[37mB\x1b[37m8\x1b[36mG\x1b[36mV\x1b[36mq\x1b[36mq\x1b[36mg\x1b[36md\x1b[36mX\x1b[36mb\x1b[36mk\x1b[36m6\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mp\x1b[36mY\x1b[37mR\x1b[37m0\x1b[37m&\x1b[36mg\x1b[36my\x1b[36mJ\x1b[36mC\x1b[36mq\x1b[36mY\x1b[37mH\x1b[37mR\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m0\x1b[36mY\x1b[36m2\x1b[36m5\x1b[36m5\x1b[36m2\x1b[36mF\x1b[37m8\x1b[37mQ\x1b[36mG\x1b[36mJ\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mC\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mw\x1b[36m2\x1b[36mq\x1b[36mg\x1b[37mA\x1b[37mD\x1b[37mR\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37m0\x1b[37mB\x1b[37mU\x1b[36mk\x1b[36mY\x1b[36mk\x1b[37m8\x1b[37mK\x1b[37mW\x1b[37mR\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[36mg\x1b[36m5\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36mF\x1b[36m5\x1b[36m2\x1b[36mm\x1b[36mG\x1b[37mD\x1b[37m0\x1b[37mN\x1b[37m&\x1b[36mE\x1b[36mp\x1b[36mf\x1b[36mf\x1b[36mF\x1b[36mV\x1b[36mZ\x1b[37m8\x1b[37mD\x1b[37mN\x1b[37mM\x1b[37m$\x1b[36mb\x1b[36mG\x1b[36mP\x1b[36mP\x1b[36mP\x1b[36mZ\x1b[37m@\x1b[37m@\x1b[36mm\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36my\x1b[36m2\x1b[36m6\x1b[36mV\x1b[36mG\x1b[37m&\x1b[37mD\x1b[37mM\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mR\x1b[37mM\x1b[37mM\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[36mP\x1b[36m2\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36m2\x1b[36m5\x1b[36mS\x1b[36mG\x1b[37m$\x1b[37mN\x1b[37mR\x1b[37mW\x1b[37m@\x1b[36mk\x1b[36mg\x1b[36mg\x1b[36mm\x1b[36mp\x1b[36m6\x1b[36mq\x1b[36mg\x1b[36md\x1b[36m4\x1b[36mE\x1b[36mE\x1b[36m4\x1b[36mh\x1b[36mp\x1b[36my\x1b[36mf\x1b[36my\x1b[36my\x1b[36my\x1b[36m3\x1b[36m2\x1b[36mF\x1b[36m6\x1b[36mS\x1b[36md\x1b[36mX\x1b[36mY\x1b[37mU\x1b[37mD\x1b[37mM\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m0\x1b[37mW\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mK\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m3\x1b[36m3\x1b[36mF\x1b[36mS\x1b[36mE\x1b[36mb\x1b[37m8\x1b[37mH\x1b[37mW\x1b[37mN\x1b[37mW\x1b[37mB\x1b[37mH\x1b[37m@\x1b[37mU\x1b[37m&\x1b[36mO\x1b[36mk\x1b[36mY\x1b[36mY\x1b[36mk\x1b[36mA\x1b[36mA\x1b[37m8\x1b[37m$\x1b[37mK\x1b[37mD\x1b[37mW\x1b[37mM\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[37mN\x1b[37mH\x1b[37mU\x1b[36mk\x1b[36mX\x1b[36mV\x1b[36m6\x1b[36m4\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[36m4\x1b[36my\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m2\x1b[36m3\x1b[36m3\x1b[36my\x1b[36m3\x1b[36m5\x1b[36m5\x1b[36m5\x1b[36mF\x1b[36mm\x1b[36mS\x1b[36md\x1b[36mE\x1b[36mP\x1b[36mZ\x1b[36mY\x1b[37mk\x1b[37mA\x1b[37mA\x1b[37m&\x1b[37m&\x1b[37m&\x1b[37mA\x1b[37mO\x1b[37mk\x1b[36mb\x1b[36mZ\x1b[36mX\x1b[36mE\x1b[36md\x1b[36mg\x1b[36mm\x1b[36mp\x1b[36m5\x1b[36m3\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[37m8\x1b[37mW\x1b[37mH\x1b[37m@\x1b[37m$\x1b[37m@\x1b[37mD\x1b[37mR\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mW\x1b[36mF\x1b[36my\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m3\x1b[36m2\x1b[36mp\x1b[36mg\x1b[36md\x1b[36mE\x1b[36m4\x1b[36md\x1b[36mg\x1b[36mm\x1b[36mF\x1b[36m2\x1b[36my\x1b[36mf\x1b[36mf\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mf\x1b[36mf\x1b[36my\x1b[36my\x1b[36m2\x1b[36m5\x1b[36mp\x1b[36mS\x1b[36mP\x1b[36mb\x1b[37m&\x1b[37m@\x1b[37mW\x1b[37mR\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m$\x1b[36m3\x1b[36mf\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36my\x1b[36mf\x1b[36mf\x1b[36my\x1b[36m2\x1b[36mF\x1b[36mS\x1b[36md\x1b[36mg\x1b[36mG\x1b[37mY\x1b[37mk\x1b[37mk\x1b[37mY\x1b[37mY\x1b[36mZ\x1b[36mG\x1b[36mP\x1b[36mX\x1b[36mg\x1b[36mE\x1b[36m4\x1b[36m4\x1b[36m4\x1b[36mE\x1b[36mE\x1b[36mX\x1b[36mX\x1b[36mG\x1b[36mZ\x1b[37mY\x1b[37mk\x1b[37mO\x1b[37mk\x1b[37mH\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m&\x1b[36my\x1b[36mw\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mf\x1b[36my\x1b[36m3\x1b[36m5\x1b[36mF\x1b[36m6\x1b[36mq\x1b[36mS\x1b[36mg\x1b[36mh\x1b[36mh\x1b[36mh\x1b[36mh\x1b[36mh\x1b[36mh\x1b[36mg\x1b[36mS\x1b[36mq\x1b[36mm\x1b[36mp\x1b[36mF\x1b[36m2\x1b[36mf\x1b[36mg\x1b[37mN\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m$\x1b[36m5\x1b[36mC\x1b[36mw\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mf\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mJ\x1b[36md\x1b[37mM\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mB\x1b[36mg\x1b[36mJ\x1b[36mC\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mJ\x1b[36mf\x1b[37mb\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m0\x1b[37mY\x1b[36m2\x1b[36m#\x1b[36mJ\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mC\x1b[36mJ\x1b[36mJ\x1b[36mg\x1b[37mH\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mW\x1b[37mX\x1b[36mf\x1b[36mT\x1b[36m#\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36m#\x1b[36m#\x1b[36m6\x1b[37m8\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mW\x1b[37mG\x1b[36m2\x1b[36mT\x1b[36mT\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36m#\x1b[36mT\x1b[36mT\x1b[36mC\x1b[36mS\x1b[37m8\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[37m&\x1b[36mh\x1b[36mf\x1b[34mT\x1b[34mn\x1b[34mT\x1b[34mT\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34m#\x1b[34mT\x1b[34mn\x1b[34mn\x1b[34m#\x1b[36m5\x1b[37mP\x1b[37mH\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mN\x1b[37m&\x1b[36md\x1b[36m2\x1b[34m#\x1b[34mu\x1b[34mu\x1b[34mu\x1b[34mn\x1b[34mn\x1b[34mT\x1b[34mT\x1b[34mT\x1b[34mT\x1b[34mT\x1b[34mT\x1b[34mT\x1b[34mT\x1b[34mT\x1b[34mT\x1b[34mT\x1b[34mT\x1b[34mT\x1b[34mT\x1b[34mT\x1b[34mT\x1b[34mT\x1b[34mT\x1b[34mT\x1b[34mn\x1b[34mu\x1b[34mu\x1b[34mu\x1b[34mT\x1b[36mw\x1b[36mm\x1b[37mG\x1b[37mK\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m0\x1b[37mD\x1b[37mO\x1b[37mE\x1b[36mm\x1b[36my\x1b[34mJ\x1b[34mT\x1b[34mu\x1b[34mL\x1b[34mL\x1b[34mL\x1b[34mL\x1b[34mL\x1b[34mu\x1b[34mu\x1b[34mu\x1b[34mu\x1b[34mL\x1b[34mL\x1b[34mL\x1b[34mL\x1b[34mL\x1b[34mu\x1b[34mu\x1b[34mT\x1b[34mw\x1b[36m5\x1b[36mh\x1b[37mG\x1b[37m$\x1b[37mN\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[37mW\x1b[37m@\x1b[37m&\x1b[37mb\x1b[37mX\x1b[37m4\x1b[36mh\x1b[36mS\x1b[36mm\x1b[36m6\x1b[36m6\x1b[36m6\x1b[36mm\x1b[36mS\x1b[37mh\x1b[37mE\x1b[37mP\x1b[37mk\x1b[37mU\x1b[37mD\x1b[37mM\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\n\x1b[37mN\x1b[37mW\x1b[37mW\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mN\x1b[37mW\x1b[37mN\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m0\x1b[37mW\x1b[37mW\x1b[37mR\x1b[37mQ\x1b[37mQ\x1b[37mM\x1b[37mW\x1b[37mN\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mN\x1b[37mW\x1b[37mM\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mW\x1b[37mB\x1b[37mM\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mN\x1b[37mW\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mM\x1b[37mW\x1b[37mN\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mM\x1b[37mW\x1b[37mM\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[37mW\x1b[37mW\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m0\x1b[37mD\x1b[37mK\x1b[37mD\x1b[37m0\x1b[37mQ\x1b[37mQ\n\x1b[37mk\x1b[34m1\x1b[34m[\x1b[37mG\x1b[37mQ\x1b[37mQ\x1b[37mb\x1b[34m[\x1b[34m[\x1b[37mk\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mW\x1b[34m7\x1b[34m!\x1b[37md\x1b[37mQ\x1b[37mQ\x1b[37m5\x1b[34m!\x1b[34mn\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[34mu\x1b[34m!\x1b[37mq\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m&\x1b[34my\x1b[34mt\x1b[34m!\x1b[34m!\x1b[34m[\x1b[34mL\x1b[37mh\x1b[37mB\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m2\x1b[34m]\x1b[34m]\x1b[34m3\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mp\x1b[34m?\x1b[34m#\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[34mw\x1b[34m]\x1b[34mt\x1b[37mX\x1b[37mQ\x1b[37mA\x1b[34m[\x1b[34m1\x1b[37mU\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mY\x1b[34mj\x1b[34m[\x1b[34m[\x1b[34m!\x1b[34m7\x1b[37mG\x1b[37m0\n\x1b[37m0\x1b[36m5\x1b[34m*\x1b[34m7\x1b[37mW\x1b[37mW\x1b[34m7\x1b[34m*\x1b[36m2\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mW\x1b[34ma\x1b[34m}\x1b[37mh\x1b[37mQ\x1b[37mQ\x1b[36my\x1b[34m*\x1b[34mj\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[34mj\x1b[34m*\x1b[36m6\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mP\x1b[34m!\x1b[34m!\x1b[36mF\x1b[37mY\x1b[37mA\x1b[37mg\x1b[34mT\x1b[36mf\x1b[37m8\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mX\x1b[34m?\x1b[34m!\x1b[34m]\x1b[34m?\x1b[37mg\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[36m5\x1b[34m*\x1b[34mu\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[34m#\x1b[34mI\x1b[34m?\x1b[34m?\x1b[37mg\x1b[37mY\x1b[34m?\x1b[34m!\x1b[37m8\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[34mJ\x1b[34m{\x1b[34mo\x1b[37m4\x1b[37mS\x1b[37mq\x1b[37mU\x1b[37mQ\n\x1b[37mQ\x1b[37mW\x1b[34mL\x1b[34m[\x1b[36mm\x1b[36mm\x1b[34m[\x1b[34mL\x1b[37mB\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mW\x1b[34mz\x1b[34m[\x1b[37mE\x1b[37mQ\x1b[37mQ\x1b[36mF\x1b[34m[\x1b[36mT\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[36mT\x1b[34m[\x1b[36mg\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[36mC\x1b[34m]\x1b[36my\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mK\x1b[34m7\x1b[34m1\x1b[36m6\x1b[36mm\x1b[34m1\x1b[34mo\x1b[37m@\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[36m6\x1b[34m[\x1b[36mJ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[36mf\x1b[34m[\x1b[36mJ\x1b[36mC\x1b[34m[\x1b[34mu\x1b[34ma\x1b[34ma\x1b[37mU\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mH\x1b[36mm\x1b[36m#\x1b[34mL\x1b[34mu\x1b[36mf\x1b[37mP\x1b[37mR\n\x1b[37mQ\x1b[37mQ\x1b[37mk\x1b[36m7\x1b[36m7\x1b[36m7\x1b[36m7\x1b[37mb\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mM\x1b[36mJ\x1b[36ma\x1b[36m5\x1b[37m8\x1b[37m8\x1b[36mw\x1b[36ma\x1b[36m2\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[36mJ\x1b[36me\x1b[36m6\x1b[37mU\x1b[37mU\x1b[37m8\x1b[37mN\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mZ\x1b[36mz\x1b[36mz\x1b[36mg\x1b[37m&\x1b[37mU\x1b[37mZ\x1b[36my\x1b[36mF\x1b[37mU\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mM\x1b[36my\x1b[36me\x1b[36mj\x1b[36mT\x1b[36mT\x1b[36mj\x1b[36mo\x1b[36mf\x1b[37mM\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[36mq\x1b[36ma\x1b[36mf\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[36m3\x1b[36ma\x1b[36m6\x1b[37mM\x1b[36mm\x1b[36mo\x1b[36m7\x1b[36m7\x1b[37m$\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mM\x1b[37mZ\x1b[36mm\x1b[37mG\x1b[37mO\x1b[36mm\x1b[36me\x1b[36mo\x1b[37m$\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[36mh\x1b[36mC\x1b[36mC\x1b[36mg\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m@\x1b[36mq\x1b[36mw\x1b[36mJ\x1b[36mJ\x1b[36mf\x1b[36mh\x1b[37mD\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[36mF\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mw\x1b[36mJ\x1b[37m&\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m$\x1b[36mg\x1b[36mw\x1b[36mJ\x1b[36mJ\x1b[36mJ\x1b[36m5\x1b[36mX\x1b[37mW\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mG\x1b[36mJ\x1b[36my\x1b[37m8\x1b[37mD\x1b[37mD\x1b[37mU\x1b[36my\x1b[36mJ\x1b[36mP\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[36m4\x1b[36mJ\x1b[36mm\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[36mS\x1b[36m#\x1b[36mh\x1b[37mQ\x1b[37mQ\x1b[36mP\x1b[36mw\x1b[36mw\x1b[37mK\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[37mG\x1b[36m3\x1b[36mJ\x1b[36mC\x1b[36mC\x1b[36m3\x1b[36mX\x1b[37mR\n\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37m0\x1b[37mM\x1b[37mM\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mM\x1b[37mB\x1b[37mB\x1b[37mR\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mM\x1b[37mM\x1b[37mM\x1b[37mM\x1b[37mM\x1b[37mN\x1b[37mR\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[37mW\x1b[37mB\x1b[37mN\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mN\x1b[37mM\x1b[37mR\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[37mM\x1b[37mN\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mR\x1b[37mM\x1b[37mM\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mM\x1b[37mM\x1b[37mR\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mM\x1b[37mM\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mQ\x1b[37mN\x1b[37mB\x1b[37mW\x1b[37m0\x1b[37mQ\x1b[37mQ\x1b[39m'
printedlogo = np.array([False])
printBlocked = np.array([False])
tP_Capsule = [[]]
PolarsInterpolators = [[]]
SafeZones = [[]]
FastMetrics = [[]]

####################################################################################################
######################################### VULCAINS Modules #########################################
####################################################################################################
from . User import *
from . FreeWakeParticles import *
from . LiftingLineCoupling import *
from . EulerianCoupling import *

from VULCAINS.vulcains import *
from VULCAINS.__init__ import __version__, __author__

####################################################################################################
########################################## MAIN Functions ##########################################
####################################################################################################
def vectorise(names = '', capital = True):
    '''
    Repeats all the input strings with  the axis coordinates ``'X'``, ``'Y'``
    and ``'Z'`` added at the end.

    Parameters
    ----------
        names : :py:class:`str` or list of :py:class:`str`
            contains strings to vectorise.
        capital : :py:class:`bool`
            States whether this axis coordinates letters are in capitals or not.
    Returns
    -------
        vectors : list of :py:class:`str`
            List of all the names with the axis coordinates added.
    '''
    coord = 'XYZ' if capital else 'xyz'
    if not (isinstance(names, list) or isinstance(names, np.ndarray)): names = [names]

    vectors = []
    for name in names: vectors += [name + v for v in coord]

    return vectors

def buildEmptyVPMTree(Np = 0, FieldsNames = None):
    '''
    Builds a particle tree.

    Parameters
    ----------
        Np : :py:class:`int`
            Size of the tree.
        FieldsNames : list of :py:class:`str`
            List of flow fields to initiate. By , all the VPM flow fields are initialised.
    Returns
    -------
        t : Tree
            CGNS Tree or Base of particles.
    '''
    if FieldsNames == None: FieldsNames = VPM_FlowSolution
    Particles = C.convertArray2Node(D.line((0., 0., 0.), (0., 0., 0.), 2))
    Particles[0] = 'FreeParticles'
    J.invokeFieldsDict(Particles, FieldsNames)
    IntegerFieldNames = ['Age']
    GridCoordinatesNode = I.getNodeFromName1(Particles, 'GridCoordinates')
    FlowSolutionNode = I.getNodeFromName1(Particles, 'FlowSolution')
    for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
        if node[3] == 'DataArray_t':
            if node[0] in IntegerFieldNames:
                node[1] = np.zeros(Np, dtype = np.int32, order = 'F')
            else:
                node[1] = np.zeros(Np, dtype = np.float64, order = 'F')

    getParticlesNumber(Particles, pointer = True)[0] = Np
    I._sortByName(I.getNodeFromName1(Particles, 'FlowSolution'))
    I._rmNodesByName(Particles, 'GridElements')
    return C.newPyTree(['Particles', Particles])

def getParticles(t = []):
    '''
    Gets the base containing all the VPM particles.

    Parameters
    ----------
        t : Tree
            contains a Base of particles named ``Particles``.
    Returns
    -------
        Particles : Base
            Particle Base (if any).
    '''
    if t and t[0] == 'Particles': return t
    return I.getNodeFromName1(t, 'Particles')

def getFreeParticles(t = []):
    '''
    Gets the zone containing the free wake particles.

    Parameters
    ----------
        t : Tree
            contains a Zone of particles named ``FreeParticles``.
    Returns
    -------
        Particles : Zone
            Free Particle Zone (if any).
    '''
    if t and t[0] == 'FreeParticles': return t
    Particles = getParticles(t)
    if Particles: return I.getNodeFromName1(Particles, 'FreeParticles')

    Particles = I.getNodeFromName1(t, 'FreeParticles')
    if Particles: return Particles

    for z in I.getZones(t):
        if z[0] == 'FreeParticles':
            return z

def getBEMParticles(t = []):
    '''
    Gets the zone containing the BEM particles.

    Parameters
    ----------
        t : Tree
            contains a Zone of particles named ``BEMParticles``.
    Returns
    -------
        Particles : Zone
            BEM Particle Zone (if any).
    '''
    if t and t[0] == 'BEMParticles': return t
    Particles = getParticles(t)
    if Particles: return I.getNodeFromName1(Particles, 'BEMParticles')

    Particles = I.getNodeFromName1(t, 'BEMParticles')
    if Particles: return Particles

    for z in I.getZones(t):
        if z[0] == 'BEMParticles':
            return z

def getImmersedParticles(t = []):
    '''
    Gets the zone containing the Eulerian Immersed particles.

    Parameters
    ----------
        t : Tree
            contains a Zone of particles named ``ImmersedParticles``.
    Returns
    -------
        Particles : Zone
            Eulerian Immersed Particle Zone (if any).
    '''
    if t and t[0] == 'ImmersedParticles': return t
    Particles = getParticles(t)
    if Particles: return I.getNodeFromName1(Particles, 'ImmersedParticles')

    Particles = I.getNodeFromName1(t, 'ImmersedParticles')
    if Particles: return Particles

    for z in I.getZones(t):
        if z[0] == 'ImmersedParticles':
            return z

def getParticlesTree(t = []):
    '''
    Gets the tree containing the all the VPM particles.

    Parameters
    ----------
        t : Tree
            contains a Base named ``Particles``.
    Returns
    -------
        Particles : Tree
            Particle Tree containg all the particles (if any).
    '''
    Particles = I.getZones(getParticles(t))
    if Particles: return C.newPyTree(['Particles', Particles])

def getPerturbationField(t = []):
    '''
    Gets the Base containing the Perturbation field.

    Parameters
    ----------
        t : Tree
            contains a Base named 'PerturbationField'.
    Returns
    -------
        PerturbationField : Base
            PerturbationField Base (if any).
    '''
    if t and t[0] == 'PerturbationField': return t
    return I.getNodeFromName1(t, 'PerturbationField')

def getPerturbationFieldTree(t = []):
    '''
    Gets the tree containing the Perturbation field.

    Parameters
    ----------
        t : Tree
            contains a Base named ``PerturbationField``.
    Returns
    -------
        PerturbationField : Tree
            PerturbationField Tree (if any).
    '''
    PerturbationField = I.getZones(getPerturbationField(t))
    if PerturbationField: return C.newPyTree(['PerturbationField', PerturbationField])

def getAllParameters(t = []):
    '''
    Get all the VULCAINS parameters.

    Parameters
    ----------
        t : Tree
            contains the VULCAINS parameters.
            ``'.Fluid#Parameters'``
            ``'.Hybrid#Parameters'``
            ``'.Modeling#Parameters'``
            ``'.Numerical#Parameters'``
            ``'.Private#Parameters'``
    Returns
    -------
        Parameters : :py:class:`dict`
            VULCAINS parameters.
    '''
    Parameters = {}
    names = ['Fluid', 'Hybrid', 'Modeling', 'Numerical', 'Private']
    for name in names:
        dico = J.get(getFreeParticles(t), '.' + name + '#Parameters')
        if dico: Parameters[name + 'Parameters'] = dico

    return Parameters

def getFluidParameters(t = []):
    '''
    Get the Fluid parameters.

    Parameters
    ----------
        t : Tree
            contains the Fluid parameters ``'.Fluid#Parameters'``.
    Returns
    -------
        FluidParameters : :py:class:`dict`
            Fluid parameters.
    '''
    return J.get(getFreeParticles(t), '.Fluid#Parameters')

def getNumericalParameters(t = []):
    '''
    Get the Numerical parameters.

    Parameters
    ----------
        t : Tree
            contains the Numerical parameters ``'.Numerical#Parameters'``.
    Returns
    -------
        NumericalParameters : :py:class:`dict`
            Numerical parameters.
    '''
    return J.get(getFreeParticles(t), '.Numerical#Parameters')

def getParticleControlParameters(t = []):
    '''
    Get the Particle Control parameters.

    Parameters
    ----------
        t : Tree
            contains the Particle Control parameters ``'ParticleControlParameters'``.
    Returns
    -------
        ParticleControlParameters : :py:class:`dict`
            Particles Control parameters.
    '''
    return getNumericalParameters(t)['ParticleControlParameters']

def getFMMParameters(t = []):
    '''
    Get FMM parameters.

    Parameters
    ----------
        t : Tree
            contains the FMM parameters ``'FMMParameters'``.
    Returns
    -------
        FMMParameters : :py:class:`dict`
            FMM parameters.
    '''
    return getNumericalParameters(t)['FMMParameters']

def getModelingParameters(t = []):
    '''
    Get the Modeling parameters.

    Parameters
    ----------
        t : Tree
            contains the Modeling parameters ``'.Modeling#Parameters'``.
    Returns
    -------
        ModelingParameters : :py:class:`dict`
            Modeling parameters.
    '''
    return J.get(getFreeParticles(t), '.Modeling#Parameters')

def getPrivateParameters(t = []):
    '''
    Get the Private parameters.

    Parameters
    ----------
        t : Tree
            contains the Private parameters ``'.Private#Parameters'``.
    Returns
    -------
        PrivateParameters : :py:class:`dict`
            Private parameters.
    '''
    return J.get(getFreeParticles(t), '.Private#Parameters')

def getHybridParameters(t = []):
    '''
    Get the Hybrid parameters.

    Parameters
    ----------
        t : Tree
            contains the Hybrid parameters ``'.Hybrid#Parameters'``.
    Returns
    -------
        HybridParameters : :py:class:`dict`
            Hybrid parameters.
    '''
    return J.get(getFreeParticles(t), '.Hybrid#Parameters')

def getParticlesNumber(t = [], pointer = False):
    '''
    Get a the free VPM particles number.

    Parameters
    ----------
        t : Tree
            contains the free VPM parameters named 'FreeParticles'.

        pointer : :py:class:`bool`
            States whether the pointer or the value of the size of the zone is returned.

    Returns
    -------
        ParticleNumber : :py:class:`int` or numpy.ndarray
            Returns the size of the free particles zone.
    '''
    if pointer: return getFreeParticles(t)[1][0]
    return getFreeParticles(t)[1][0][0]

def getBEMParticlesNumber(t = [], pointer = False):
    '''
    Get a the BEM particles number.

    Parameters
    ----------
        t : Tree
            contains the free VPM parameters named BEMParticles.
        pointer : :py:class:`bool`
            States whether the pointer or the value of the size of the zone is returned.

    Returns
    -------
        ParticleNumber : :py:class:`dict`
            Returns the size of the BEM particles zone.
    '''
    if pointer: return getBEMParticles(t)[1][0]
    return getBEMParticles(t)[1][0][0]

def getImmersedParticlesNumber(t = [], pointer = False):
    '''
    Get a the Eulerian Immersed particles number.

    Parameters
    ----------
        t : Tree
            contains the free VPM parameters named 'ImmersedParticles'.
        pointer : :py:class:`bool`
            States whether the pointer or the value of the size of the zone is returned.

    Returns
    -------
        ParticleNumber : :py:class:`dict`
            Returns the size of the Immersed particles zone.
    '''
    if pointer: return getImmersedParticles(t)[1][0]
    return getImmersedParticles(t)[1][0][0]

def addSafeZones(Zones):
    '''
    Adds zones to the global variable SafeZones. Particles within SafeZones are not redistributed
    nor deleted.

    Parameters
    ----------
        Zones : list of Zones.
            contains Zones of Coordinates.
    '''
    SafeZones[0] += I.getZones(Zones)

def emptySafeZones():
    '''
    Empties the Zones within the global variable SafeZones. Particles within SafeZones are not
    redistributed nor deleted.
    '''
    SafeZones[0] = []

def extractperturbationField(Targets = [], tL = [], tP = []):
    '''
    Extract the Perturbation field velocities onto the Coordinates in Targets.

    Parameters
    ----------
        Targets : Tree
            contains the Coordinates onto wich the Perturbation velocity fields are interpolated.

        tL : Tree
            contains the VPM parameters.

        tP : Tree
            contains the PerturbationFields.
    '''
    for var in vectorise('VelocityPerturbation'): C._initVars(Targets, var, 0.)
    _tL, _tP = getTrees([tL, tP], ['Particles', 'Perturbation'])
    if tP_Capsule[0] and _tP:
        Theta, NumberOfNodes, TimeVelPert = getParameters(_tL, ['PerturbationOverlappingFactor',
                                                       'NumberOfNodes', 'TimeVelocityPerturbation'])
        TimeVelPert[0] += extract_perturbation_velocity_field(C.newPyTree(I.getZones(Targets)),
                                                     _tP, tP_Capsule[0], NumberOfNodes[0], Theta[0])

####################################################################################################
####################################################################################################
######################################## CGNS tree control #########################################
####################################################################################################
####################################################################################################
def checkTrees(t = [], Parameters = {}):
    '''
    Checks if the minimum requirements are met within the Bases in t to launch VULCAINS
    computations. The necessary parameters and fields are initialised if missing. Allready existing 
    parameters are updated with the default parameters and the input parameters.

    Parameters
    ----------
        t : Tree
            containers to check.
        Parameters : :py:class:`dict` of :py:class:`dict`
            User-provided VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.Main.compute`
    '''
    tL, tLL, tE, tH, tP = getTrees([t], ['Particles', 'LiftingLines', 'Eulerian', 'Hybrid',
                                                                                    'Perturbation'])
    Particles = getFreeParticles(tL)
    FlowSolution = I.getNodeFromName1(Particles, 'FlowSolution')
    RequiredFlowSolution = VPM_FlowSolution.copy()
    ExcessFlowSolution = []
    for Field in FlowSolution[2]:
        if Field[0] in VPM_FlowSolution: RequiredFlowSolution.remove(Field[0])
        else: ExcessFlowSolution += [Field[0]]

    for Field in ExcessFlowSolution: I._rmNodesByName(Particles, Field)
    J.invokeFieldsDict(Particles, RequiredFlowSolution)
    I._sortByName(Particles)

    newParameters = getAllParameters(tL)
    for field in [f + 'Parameters' for f in ['Fluid', 'Modeling', 'Numerical', 'Private']]:
        if field not in newParameters: newParameters[field] = dict()
    for key in Parameters:
        newParameters[key].update(Parameters[key])
    
    checkParameters(newParameters)
    for field in ['Fluid', 'Hybrid', 'Modeling', 'Numerical', 'Private']:
        name = field + 'Parameters'
        if name in newParameters:
            J.set(Particles, '.' + field + '#Parameters', **newParameters[name])
            I._sortByName(I.getNodeFromName1(Particles, '.' + field + '#Parameters'))

    t[2] = [I.getNodeFromName1(t, 'CGNSLibraryVersion')] + I.getBases(tL) + I.getBases(tE) + \
                                                   I.getBases(tH) + I.getBases(tLL) + I.getBases(tP)

def getTrees(Trees = [], TreeNames = [], fillEmptyTrees = False):
    '''
    Checks if the minimum requirements are met within the Bases in t to launch VULCAINS
    computations. The necessary parameters and fields are initialised if missing.

    Parameters
    ----------
        Targets : Tree
            contains the Coordinates onto which the Perturbation velocity fields are interpolated.
        tL : Tree
            contains the VPM parameters.
        tP : Tree
            contains the PerturbationFields.
    '''

    def getTree(Tree = [], function = getParticlesTree(), BackUpTree = [], fillEmptyTrees = False):
        _Tree = function(Tree)
        if not _Tree and BackUpTree: _Tree = function(BackUpTree)
        if _Tree: return _Tree
        if fillEmptyTrees: return C.newPyTree()
        else: return []

    if not (isinstance(TreeNames, list) or isinstance(TreeNames, np.ndarray)):
        TreeNames = [TreeNames]
        Trees = [Trees]

    MainTree = Trees[0]
    while len(Trees) != len(TreeNames): Trees = [MainTree] + Trees

    newTrees = []
    for Tree, TreeName in zip(Trees, TreeNames):
        if TreeName[:9] == 'Particles':
            newTrees += [getTree(Tree, getParticlesTree, MainTree, fillEmptyTrees)]
        elif TreeName[:12] == 'LiftingLines':
            newTrees += [getTree(Tree, getLiftingLinesTree, MainTree, fillEmptyTrees)]
        elif TreeName[:8] == 'Eulerian':
            newTrees += [getTree(Tree, getEulerianTree, MainTree, fillEmptyTrees)]
        elif TreeName[:6] == 'Hybrid':
            newTrees += [getTree(Tree, getHybridDomainTree, MainTree, fillEmptyTrees)]
        elif TreeName[:12] == 'Perturbation':
            newTrees += [getTree(Tree, getPerturbationFieldTree, MainTree, fillEmptyTrees)]
        else:
            raise AttributeError(J.FAIL + 'You ask for a non-existing Tree (%s).\n'%(TreeName) + \
                 '                The available Trees are Particles, LiftingLines,\n' + \
                 '                Eulerian, Hybrid or Perturbation' + J.ENDC)

    if len(newTrees) == 1: return newTrees[0]
    return newTrees

def checkParameters(Parameters = {}):
    '''
    Imposes the types of the parameters in each dictionnary used by VULCAINS. If a parameter is not
    provided, a  value will be prescribed. Parameters are modified (if necessary) to fit in
    their operability range.

    Parameters
    ----------
        Parameters : :py:class:`dict` of :py:class:`dict`
            User-provided VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.Main.compute`
    '''
    fields = [f + 'Parameters' for f in ['Fluid', 'Hybrid', 'LiftingLine', 'Modeling', 'Numerical',
                                                                                         'Private']]
    funcs = [getDefaultFluidParameters, getDefaultHybridParameters, getDefaultLiftingLineParameters,
            getDefaultModelingParameters, getDefaultNumericalParameters, getDefaultPrivateParameters]
    for field, func in zip(fields, funcs):
        if field in Parameters:
            Parameters[field].update(func(**Parameters[field]))

    if 'NumericalParameters' in Parameters:
        NP = Parameters['NumericalParameters']
        if NP['Resolution'][0]:
            NP['Resolution'][0], NP['Resolution'][1] = np.min(NP['Resolution']), \
                                                                            np.max(NP['Resolution'])
        if NP['Resolution'][0] and 'PrivateParameters' in Parameters and \
                                                                 'ModelingParameters' in Parameters:
            Parameters['PrivateParameters']['Sigma0'] = NP['Resolution']*\
                                               Parameters['ModelingParameters']['SmoothingRatio'][0]
        if 'ParticleControlParameters' in NP:
            PCP = NP['ParticleControlParameters']
            PCP['CutoffXmin'][0], PCP['CutoffXmax'][0] = min(PCP['CutoffXmin'][0],
                              PCP['CutoffXmax'][0]), max(PCP['CutoffXmin'][0], PCP['CutoffXmax'][0])
            PCP['CutoffYmin'][0], PCP['CutoffYmax'][0] = min(PCP['CutoffYmin'][0],
                              PCP['CutoffYmax'][0]), max(PCP['CutoffYmin'][0], PCP['CutoffYmax'][0])
            PCP['CutoffZmin'][0], PCP['CutoffZmax'][0] = min(PCP['CutoffZmin'][0],
                              PCP['CutoffZmax'][0]), max(PCP['CutoffZmin'][0], PCP['CutoffZmax'][0])

def delete(t = [], mask = []):
    '''
    Deletes the free particles inside t that are flagged by mask.

    Parameters
    ----------
        t : Tree
            contains a zone of particles named 'FreeParticles'.

        mask : :py:class:`list` or numpy.ndarray of :py:class:`bool`
            List of booleans of the same size as the particle zone. A true flag will delete a 
            particle, a false flag will leave it.
    '''
    Particles = getFreeParticles(t)
    GridCoordinatesNode = I.getNodeFromName1(Particles, 'GridCoordinates')
    FlowSolutionNode = I.getNodeFromName1(Particles, 'FlowSolution')
    Np = getParticlesNumber(t, pointer = True)
    if Np[0] != len(mask): raise ValueError('The length of the mask (%d) must be the same as the \
                                                      number of particles (%d)'%(len(mask), Np[0]))
    mask = np.logical_not(mask)
    for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
        if node[3] == 'DataArray_t':
            node[1] = node[1][mask]

    Np[0] = len(node[1])

def extend(t = [], ExtendSize = 0, Offset = 0, ExtendAtTheEnd = True):
    '''
    Add empty free particles.

    Parameters
    ----------
        t : Tree
            contains a zone of particles named 'FreeParticles'.

        ExtendSize : :py:class:`int`
            Number of particles to add.

        Offset : :py:class:`int`
            Position where the particles are added.

        ExtendAtTheEnd : :py:class:`bool`
            If True the particles are added at the end of t, by an offset of Offset, if False
            the particles are added at the beginning of t at the position Offset.
    '''
    Particles = getFreeParticles(t)
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

    getParticlesNumber(t, pointer = True)[0] = len(node[1])

def addParticlesToTree(t, NewX = [], NewY = [], NewZ = [], NewAX = [], NewAY = [], NewAZ = [],
    NewSigma = [], Offset = 0, ExtendAtTheEnd = False):
    '''
    Add free particles.

    Parameters
    ----------
        t : Tree
            contains a zone of particles named 'FreeParticles'.

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
    Particles = getFreeParticles(t)
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
        t : Tree
            contains a zone of particles named 'FreeParticles'.

        NumberToTrim : :py:class:`int`
            Number of particles to remove.

        Offset : :py:class:`int`
            Position from where the particles are removed.

        ExtendAtTheEnd : :py:class:`bool`
            If True the particles are removed from the end of t, by an offset of Offset, if
            False the particles are removed from the beginning of t from the position Offset.
    '''
    Particles = getFreeParticles(t)
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

    getParticlesNumber(t, pointer = True)[0] = len(node[1])

def adjustTreeSize(t = [], NewSize = 0, OldSize = -1, Offset = 0, AtTheEnd = True):
    '''
    Adds (Offset < NewSize) or removes (NewSize < Offset) a set of particles in t to adjust its
    size. If OldSize is not given, the current size of t is used.

    Parameters
    ----------
        t : Tree
            contains a zone of particles named 'FreeParticles'.

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
    Particles = getFreeParticles(t)
    if OldSize == -1: OldSize = len(J.getx(Particles))
    SizeDiff = NewSize - OldSize

    if 0 < SizeDiff:extend(t, ExtendSize = SizeDiff, Offset = Offset, ExtendAtTheEnd = AtTheEnd)
    else: trim(t, NumberToTrim = -SizeDiff, Offset = Offset, TrimAtTheEnd = AtTheEnd)

def roll(t = [], PivotNumber = 0):
    '''
    Moves a set of particles at the end of t

    Parameters
    ----------
        t : Tree
            contains a zone of particles named 'FreeParticles'.

        PivotNumber : :py:class:`int`
            Position of the new first particle
    '''
    Particles = getFreeParticles(t)
    Np = getParticlesNumber(t)
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
        t : Tree
            contains a node of parameters named '.PerturbationField#Parameters'.

    Returns
    -------
        PerturbationFieldParameters : :py:class:`dict`
            Dictionnary of parameters. The parameters are pointers inside numpy ndarrays.
    '''
    return J.get(getFreeParticles(t), '.PerturbationField#Parameters')

def getParameter(t = [], Name = '', Field = ['Numerical', 'Private', 'Modeling', 'Hybrid',
                                                                                          'Fluid']):
    '''
    Recursively searches for a parameter.

    Parameters
    ----------
        t : Tree
            Contains the parameter nodes in ``Field`` as ``.Field#Parameters``.

        Name : :py:class:`str`
            Name of the parameter to get.

        Field : list of :py:class:`str`
            Containers in which to look for.
            ``'Numerical'``
            ``'Private'``
            ``'Modeling'``
            ``'Hybrid'``
            ``'Fluid'``

    Returns
    -------
        Parameter : :py:class:`float`, :py:class:`int`, :py:class:`str` or :py:class:`dict`
            Pointer of the parameter.
    '''
    if Field:
        Parameters = J.get(getFreeParticles(t), '.' + Field[0] + '#Parameters')    
        if Name in Parameters: return Parameters[Name]
        elif Field[0] == 'Numerical':
            if 'ParticleControlParameters' in Parameters and \
                                                    Name in Parameters['ParticleControlParameters']:
                return Parameters['ParticleControlParameters'][Name]
            elif 'FMMParameters' in Parameters and Name in Parameters['FMMParameters']:
                return Parameters['FMMParameters'][Name]
            else: return getParameter(t, Name, Field[1:])
        else: return getParameter(t, Name, Field[1:])
    return None

def getParameters(t = [], Names = []):
    '''
    Get a list of parameters.

    Parameters
    ----------
        t : Tree
            contains a node of parameters with their names in Names.

        Names : :py:class:`list` or numpy.ndarray of :py:class:`str`
            List of parameter names

    Returns
    -------
        ParameterNode : :py:class:`dict`
            The parameter is a pointer inside a numpy ndarray.
    '''
    Names = np.ravel(Names)
    Parameters = [0]*len(Names)
    FreeParticles = getFreeParticles(t)
    FieldNames = ['Numerical', 'Private', 'Modeling', 'Hybrid', 'Fluid']
    Index = np.array([False]*len(Names))
    for FieldName in FieldNames:
        Field = J.get(FreeParticles, '.' + FieldName + '#Parameters')
        DeleteList = []
        for pos, Name in enumerate(Names):
            if not(Index[pos]) and Name in Field:
                Parameters[pos] = Field[Name]
                Index[pos] = True

        if Index.all(): return Parameters
        if FieldName == 'Numerical':
            subField = Field['ParticleControlParameters']
            for pos, Name in enumerate(Names):
                if not(Index[pos]) and Name in subField:
                    Parameters[pos] = subField[Name]
                    Index[pos] = True

            if Index.all(): return Parameters
            subField = Field['FMMParameters']
            for pos, Name in enumerate(Names):
                if not(Index[pos]) and Name in subField:
                    Parameters[pos] = subField[Name]
                    Index[pos] = True


            if Index.all(): return Parameters

    print(Parameters, Index)
    return Parameters

####################################################################################################
####################################################################################################
########################################### Lifting Lines ##########################################
####################################################################################################
####################################################################################################
def buildPolarsInterpolator(Polars = []):
    '''
    Computes the 2D polars interpolators inside the global variable PolarsInterpolators.

    Parameters
    ----------
        Polars : Tree
            2D Polars.
    '''
    PolarsInterpolators[0] = LL.buildPolarsInterpolatorDict(Polars, ['Cl', 'Cd', 'Cm'])

def getLiftingLines(t = []):
    '''
    gives the Zones containing the LiftingLines.

    Parameters
    ----------
        t : Tree
            contains a zone of particles named 'FreeParticles'.
    Returns
    -------
        LiftingLines : list of Zones
            LiftingLines Zones (if any).
    '''
    LiftingLines = I.getNodeFromName1(t, 'LiftingLines')
    if LiftingLines: return I.getZones(LiftingLines)

    LiftingLines = LL.getLiftingLines(t)
    if LiftingLines: return LiftingLines

    return []

def getLiftingLinesTree(t = []):
    '''
    Gets the tree containing the all the VPM particles.

    Parameters
    ----------
        t : Tree
            contains a Base named 'LiftingLines'.
    Returns
    -------
        LiftingLines : Tree
            LiftingLines Tree containg all the Lifting Lines (if any).
    '''
    LiftingLines = getLiftingLines(t)
    if LiftingLines: return C.newPyTree(['LiftingLines', LiftingLines])

def setTimeStepFromShedParticles(tL = [], tLL = [], NumberParticlesShedAtTip = 5.):
    '''
    Sets the VPM TimeStep so that the user-given number of particles are shed at the tip of the
    fastest moving Lifting Line.

    Parameters
    ----------
        tL: Tree or :py:class:`dict`
            Lagrangian field or dictionary containing VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.Main.compute`

        tLL : Tree
            Lifting Lines.

        NumberParticlesShedAtTip : :py:class:`int`
            Number of particles to shed per TimeStep.
    Returns
    -------
        dt : numpy.ndarray of :py:class:`float`
            Time step.
    '''
    _tLL = getLiftingLinesTree(tLL)
    if not _tLL and type(tL) != dict: _tLL = getLiftingLinesTree(tL)
    if not _tLL: raise AttributeError('The time step is not given and can not be \
                 computed without a Lifting Line. Specify the time step or give a Lifting Line')
    LL.computeKinematicVelocity(_tLL)
    LL.assembleAndProjectVelocities(_tLL)

    if type(tL) == dict: Parameters = tL
    else: Parameters = getAllParameters(tL)

    Resolution = Parameters['NumericalParameters']['Resolution'][0]
    U0         = Parameters['FluidParameters']['VelocityFreestream']
    Urelmax = 0.
    for LiftingLine in I.getZones(_tLL):
        Ukin = np.vstack(J.getVars(LiftingLine, ['VelocityKinematic' + i for i in 'XYZ']))
        Urel = np.max(np.linalg.norm(np.vstack(U0 - Ukin.T), axis = 0))
        if (Urelmax < Urel): Urelmax = Urel

    if Urelmax == 0:
        raise ValueError('Maximum velocity is zero. Set non-zero kinematic or freestream \
                                                                                     velocity.')
    Parameters['NumericalParameters']['TimeStep'] = np.array([NumberParticlesShedAtTip*Resolution/\
                                                             Urel], dtype = np.float64, order = 'F')
    return Parameters['NumericalParameters']['TimeStep']

def setTimeStepFromBladeRotationAngle(tL = [], tLL = [], BladeRotationAngle = 5.):
    '''
    Sets the VPM TimeStep so that the fastest moving Lifting Line rotates by the user-given
    angle per TimeStep.
    
    Parameters
    ----------
        tL: Tree or :py:class:`dict`
            Lagrangian field or dictionary containing VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.Main.compute`

        tLL : Tree
            Lifting Lines.

        BladeRotationAngle : :py:class:`float`
            Blade rotation angle per TimeStep.
    Returns
    -------
        dt : numpy.ndarray of :py:class:`float`
            Time step.
    '''
    _tLL = getLiftingLinesTree(tLL)
    if not _tLL and type(tL) != dict: _tLL = getLiftingLinesTree(tL)
    if not _tLL: raise AttributeError('The time step is not given and can not be \
                 computed without a Lifting Line. Specify the time step or give a Lifting Line')

    RPM = 0.
    for LiftingLine in I.getZones(_tLL):
        RPM = max(RPM, np.abs(I.getValue(I.getNodeFromName(LiftingLine, 'RPM'))))
    
    if RPM == 0: raise ValueError('Maximum RPM is zero. Set non-zero RPM.')
    if type(tL) == dict: Parameters = tL
    else: Parameters = getAllParameters(tL)

    if Urelmax == 0:
        raise ValueError('Maximum velocity is zero. Set non-zero kinematic or freestream \
                                                                                     velocity.')
    Parameters['NumericalParameters']['TimeStep'] = np.array([BladeRotationAngle/(6.*RPM)], \
                                                                    dtype = np.float64, order = 'F')
    return Parameters['NumericalParameters']['TimeStep']

def getAerodynamicCoefficientsOnLiftingLine(tLL = [], StdDeviationSample = 50, Freestream = True,
    Wings = False, Surface = 0.):
    '''
    Gets the aerodynamic coefficients on the Lifting Line(s).

    Parameters
    ----------
        tLL : Tree
            Lifting Lines.

        StdDeviationSample : :py:class:`int`
            Number of samples for the standard deviation.

        Freestream : :py:class:`bool`
            States whether their is a freestream velocity.

        Wings : :py:class:`bool`
            States whether the Lifting Line is a wing.

        Surface : :py:class:`float`
            Surface of the wing Lifting Line (if any).
    Returns
    -------
        Loads : :py:class:`dict`
            Aerodynamic coefficients, loads and standard deviation.
    '''
    _tLL = getLiftingLinesTree(tLL)
    if _tLL:
        if Wings: return getAerodynamicCoefficientsOnWing(_tLL, Surface,
                                                            StdDeviationSample = StdDeviationSample)
        else:
            if Freestream: return getAerodynamicCoefficientsOnPropeller(_tLL,
                                                            StdDeviationSample = StdDeviationSample)
            else: return getAerodynamicCoefficientsOnRotor(_tLL,
                                                            StdDeviationSample = StdDeviationSample)
    return {}

def getAerodynamicCoefficientsOnPropeller(tLL = [], StdDeviationSample = 50):
    '''
    Gets the aerodynamic coefficients on propeller Lifting Line(s).

    Parameters
    ----------
        tLL : Tree
            Lifting Lines.

        StdDeviationSample : int
            Number of samples for the standard deviation.
    Returns
    -------
        Loads : :py:class:`dict`
            Aerodynamic coefficients, loads and standard deviation.
    '''
    if not tLL: tLL = getLiftingLinesTree(tL)
    Loads = {}
    if not tLL: return Loads
    LiftingLine = I.getZones(tLL)[0]
    RotationCenter = I.getValue(I.getNodeFromName(LiftingLine, 'RotationCenter'))
    RPM = I.getValue(I.getNodeFromName(LiftingLine, 'RPM'))
    n = RPM/60.
    Rho = I.getValue(I.getNodeFromName(LiftingLine, 'Density'))
    U0 = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityFreestream'))
    V = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityTranslation'))
    x, y, z = J.getxyz(LiftingLine)
    D = 2*max(np.linalg.norm(np.vstack([x - RotationCenter[0], y - RotationCenter[1], \
                                                             z - RotationCenter[2]]), axis = 0))
    IntegralLoads = LL.computeGeneralLoadsOfLiftingLine(tLL)
    if 'Total' in IntegralLoads: IntegralLoads = IntegralLoads['Total']
    T = IntegralLoads['Thrust'][0]
    P = IntegralLoads['Power'][0]
    q0 = Rho*np.square(n*D*D)
    cT = T/q0       if 1e-6 < q0 else 0.
    cP = P/(q0*n*D) if 1e-6 < q0 else 0.
    Eff = np.linalg.norm(U0 - V, axis = 0)*T/P if 1e-3 < P else 0.
    std_Thrust, std_Power = getStandardDeviationBlade(tLL = tLL,
                                                        StdDeviationSample = StdDeviationSample)
    Loads['Thrust'] = T
    Loads['Thrust Standard Deviation'] = std_Thrust/T*100. if 1e-6 < np.abs(T) else 0.
    Loads['Power'] = P
    Loads['Power Standard Deviation']  = std_Power/P*100.  if 1e-6 < np.abs(P) else 0.
    Loads['cT'] = cT
    Loads['cP'] = cP
    Loads['Eff'] = Eff
    return Loads

def getAerodynamicCoefficientsOnRotor(tLL = [], StdDeviationSample = 50):
    '''
    Gets the aerodynamic coefficients on rotor Lifting Line(s).

    Parameters
    ----------
        tLL : Tree
            Lifting Lines.

        StdDeviationSample : :py:class:`int`
            Number of samples for the standard deviation.
    Returns
    -------
        Loads : :py:class:`dict`
            Aerodynamic coefficients, loads and standard deviation.
    '''
    if not tLL: tLL = getLiftingLinesTree(tL)
    Loads = {}
    if not tLL: return Loads
    LiftingLine = I.getZones(tLL)[0]
    RotationCenter = I.getValue(I.getNodeFromName(LiftingLine, 'RotationCenter'))
    RPM = I.getValue(I.getNodeFromName(LiftingLine, 'RPM'))
    Rho = I.getValue(I.getNodeFromName(LiftingLine, 'Density'))
    x, y, z = J.getxyz(LiftingLine)
    R = max(np.linalg.norm(np.vstack([x - RotationCenter[0], y - RotationCenter[1], \
                                                                 z - RotationCenter[2]]), axis = 0))
    IntegralLoads = LL.computeGeneralLoadsOfLiftingLine(tLL)
    if 'Total' in IntegralLoads: IntegralLoads = IntegralLoads['Total']
    T = IntegralLoads['Thrust'][0]
    P = IntegralLoads['Power'][0]
    U = RPM*np.pi/30.*R
    q0 = Rho*np.square(U)*np.pi*R**2
    cT = T/q0     if 1e-6 < q0 else 0.
    cP = P/(q0*U) if 1e-6 < q0 else 0.
    Eff = np.sqrt(np.abs(cT))*cT/(np.sqrt(2.)*cP) if 1e-12 < np.abs(cP) else 0.

    std_Thrust, std_Power = getStandardDeviationBlade(tLL = tLL,
                                                        StdDeviationSample = StdDeviationSample)
    Loads['Thrust'] = T
    Loads['Thrust Standard Deviation'] = std_Thrust/T*100. if 1e-6 < np.abs(T) else 0.
    Loads['Power'] = P
    Loads['Power Standard Deviation']  = std_Power/P*100.  if 1e-6 < np.abs(P) else 0.
    Loads['cT'] = cT
    Loads['cP'] = cP
    Loads['Eff'] = Eff
    return Loads

def getAerodynamicCoefficientsOnWing(tLL = [], Surface = 0., StdDeviationSample = 50):
    '''
    Gets the aerodynamic coefficients on wing Lifting Line(s).

    Parameters
    ----------
        tLL : Tree
            Lifting Lines.

        Surface : :py:class:`float`
            Surface of the wing Lifting Line.

        StdDeviationSample : :py:class:`int`
            Number of samples for the standard deviation.
    Returns
    -------
        Loads : :py:class:`dict`
            Aerodynamic coefficients, loads and standard deviation.
    '''
    if not tLL: tLL = getLiftingLinesTree(tL)
    Loads = {}
    if not tLL: return Loads
    LiftingLine = I.getZones(tLL)[0]
    Rho = I.getValue(I.getNodeFromName(LiftingLine, 'Density'))
    U0 = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityFreestream'))
    V = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityTranslation'))
    Axis = I.getValue(I.getNodeFromName(LiftingLine, 'RotationAxis'))
    IntegralLoads = LL.computeGeneralLoadsOfLiftingLine(tLL)
    if 'Total' in IntegralLoads: IntegralLoads = IntegralLoads['Total']
    F = np.array([IntegralLoads['Force' + v][0] for v in 'XYZ'])
    Lift = np.dot(F, Axis)
    Drag = np.sqrt(np.sum(np.square(F)) - np.square(Lift))
    q0 = 0.5*Rho*Surface*np.dot(U0 - V, (U0 - V).T)
    cL = Lift/q0 if 1e-6 < q0 else 0.
    cD = Drag/q0 if 1e-6 < q0 else 0.

    std_Thrust, std_Drag = getStandardDeviationWing(tLL = tLL,
                                                        StdDeviationSample = StdDeviationSample)

    Loads['Lift'] = Lift
    Loads['Lift Standard Deviation'] = std_Thrust/Lift*100. if 1e-6 < np.abs(Lift) else 0.
    Loads['Drag'] = Drag
    Loads['Drag Standard Deviation'] = std_Drag/Drag*100.   if 1e-6 < np.abs(Drag) else 0.
    Loads['cL'] = cL if 1e-12 < q0 else 0.
    Loads['cD'] = cD if 1e-12 < q0 else 0.
    Loads['f'] = Lift/Drag
    return Loads

def getStandardDeviationWing(tLL = [], StdDeviationSample = 50):
    '''
    Gets the standard deviation on the aerodynamic coefficients on wing Lifting Line(s).

    Parameters
    ----------
        tLL : Tree
            Lifting Lines.

        StdDeviationSample : :py:class:`int`
            Number of samples for the standard deviation.
    Returns
    -------
        std_Lift : :py:class:`float`
            Lift standard deviation.
        std_Drag : :py:class:`float`
            Drag standard deviation.
    '''
    if not tLL: tLL = getLiftingLinesTree(tL)
    if not tLL: return 0., 0.
    LiftingLine = I.getZones(tLL)[0]
    UnsteadyLoads = I.getNodeFromName(LiftingLine, '.UnsteadyLoads')
    Lift = I.getValue(I.getNodeFromName(UnsteadyLoads, 'Thrust'))
    if type(Lift) == np.ndarray or type(Lift) == list:
        StdDeviationSample = max(min(StdDeviationSample,len(Lift)), 1)
    else: return 0., 0.
    

    Lift = np.array([0.]*StdDeviationSample)
    Drag = np.array([0.]*StdDeviationSample)
    for LiftingLine in I.getZones(tLL):
        UnsteadyLoads = I.getNodeFromName(LiftingLine, '.UnsteadyLoads')
        Lift += I.getNodeFromName(UnsteadyLoads, 'ForceZ')[1][-StdDeviationSample:]
        Drag += I.getNodeFromName(UnsteadyLoads, 'ForceX')[1][-StdDeviationSample:]
    meanLift = sum(Lift)/StdDeviationSample
    meanDrag = sum(Drag)/StdDeviationSample

    std_Lift = np.sqrt(sum(np.square(Lift - meanLift))/StdDeviationSample)
    std_Drag = np.sqrt(sum(np.square(Drag - meanDrag))/StdDeviationSample)
    return std_Lift, std_Drag

def getStandardDeviationBlade(tLL = [], StdDeviationSample = 50):
    '''
    Gets the standard deviation on the aerodynamic coefficients on blade Lifting Line(s).

    Parameters
    ----------
        tLL : Tree
            Lifting Lines.

        StdDeviationSample : :py:class:`int`
            Number of samples for the standard deviation.
    Returns
    -------
        std_Thrust : :py:class:`float`
            Thrust standard deviation.
        std_Drag : :py:class:`float`
            Drag standard deviation.
    '''
    if not tLL: tLL = getLiftingLinesTree(tL)
    if not tLL: return 0., 0.

    LiftingLine = I.getZones(tLL)[0]
    UnsteadyLoads = I.getNodeFromName(LiftingLine, '.UnsteadyLoads')
    Thrust = I.getValue(I.getNodeFromName(UnsteadyLoads, 'Thrust'))
    if type(Thrust) == np.ndarray or type(Thrust) == list:
        StdDeviationSample = max(min(StdDeviationSample,len(Thrust)), 1)
    else: return 0., 0.

    Thrust = np.array([0.]*StdDeviationSample)
    Power = np.array([0.]*StdDeviationSample)
    for LiftingLine in I.getZones(tLL):
        UnsteadyLoads = I.getNodeFromName(LiftingLine, '.UnsteadyLoads')
        Thrust += I.getNodeFromName(UnsteadyLoads, 'Thrust')[1][-StdDeviationSample:]
        Power += I.getNodeFromName(UnsteadyLoads, 'Power')[1][-StdDeviationSample:]
    meanThrust = sum(Thrust)/StdDeviationSample
    meanPower = sum(Power)/StdDeviationSample

    std_Thrust = np.sqrt(sum(np.square(Thrust - meanThrust))/StdDeviationSample)
    std_Power = np.sqrt(sum(np.square(Power - meanPower))/StdDeviationSample)
    return std_Thrust, std_Power

####################################################################################################
####################################################################################################
############################################## Hybrid ##############################################
####################################################################################################
####################################################################################################
def getHybridDomain(t = []):
    '''
    Gets the Hybrid Domain Tree.

    Parameters
    ----------
        t : Tree
            contains a zone named 'HybridDomain'.
    Returns
    -------
        HybridDomain : Base
            BEM, Inner and Outer Interfaces of the Hybrid Domain, Hybrid vorticity sources and
                                                                              Eulerian far fiedl BC.
    '''
    return I.getNodeFromName1(t, 'HybridDomain')

def getHybridDomainTree(t = []):
    '''
    Gets the Hybrid Domain Tree.

    Parameters
    ----------
        t : Tree
            contains a zone named 'HybridDomain'.
    Returns
    -------
        HybridDomain : Tree
            BEM, Inner and Outer Interfaces of the Hybrid Domain, Hybrid vorticity sources and
                                                                              Eulerian far fiedl BC.
    '''
    HybridDomain = I.getNodeFromName1(t, 'HybridDomain')
    if HybridDomain: return C.newPyTree(['HybridDomain', I.getZones(HybridDomain)])

def getHybridDomainOuterInterface(t = []):
    '''
    Gets the Outer Hybrid Domain Interface.

    Parameters
    ----------
        t : Tree
            contains a zone named 'HybridDomain'.
    Returns
    -------
        OuterInterface : Zone
            Outer Interface of the Hybrid Domain.
    '''
    HybridDomain = getHybridDomain(t)
    if HybridDomain: return I.getNodeFromName1(HybridDomain, 'OuterInterface')

    HybridDomain = I.getNodeFromName1(t, 'OuterInterface')
    if HybridDomain: return HybridDomain

    for z in I.getZones(t):
        if z[0] == 'OuterInterface':
            return z

    #return []

def getHybridDomainInnerInterface(t = []):
    '''
    Gets the Inner Hybrid Domain Interface.

    Parameters
    ----------
        t : Tree
            contains a zone named 'HybridDomain'.
    Returns
    -------
        InnerInterface : Zone
            Inner Interface of the Hybrid Domain.
    '''
    HybridDomain = getHybridDomain(t)
    if HybridDomain: return I.getNodeFromName1(HybridDomain, 'InnerInterface')

    HybridDomain = I.getNodeFromName1(t, 'InnerInterface')
    if HybridDomain: return HybridDomain

    for z in I.getZones(t):
        if z[0] == 'InnerInterface':
            return z

    #return []

def getHybridDomainBEMInterface(t = []):
    '''
    Gets the Inner Hybrid Domain Interface.

    Parameters
    ----------
        t : Tree
            contains a zone named 'HybridDomain'.
    Returns
    -------
        BEMInterface : Zone
            BEM Interface of the Hybrid Domain.
    '''
    HybridDomain = getHybridDomain(t)
    if HybridDomain: return I.getNodeFromName1(HybridDomain, 'BEMInterface')

    HybridDomain = I.getNodeFromName1(t, 'BEMInterface')
    if HybridDomain: return HybridDomain

    for z in I.getZones(t):
        if z[0] == 'BEMInterface':
            return z

    #return []

def getHybridSources(t = []):
    '''
    Gets the Hybrid Hybrid Domain Interface.

    Parameters
    ----------
        t : Tree
            contains a zone named 'HybridDomain'.
    Returns
    -------
        HybridInterface : Zone
            Hybrid vorticity sources of the Hybrid Domain.
    '''
    HybridDomain = getHybridDomain(t)
    if HybridDomain: return I.getNodeFromName1(HybridDomain, 'HybridSources')

    HybridDomain = I.getNodeFromName1(t, 'HybridSources')
    if HybridDomain: return HybridDomain

    for z in I.getZones(t):
        if z[0] == 'HybridSources':
            return z

    #return []

def getEulerianTree(t = []):
    '''
    Gets the tree containing the node and cell-centers Eulerian bases.

    Parameters
    ----------
        t : Tree
            contains the Eulerian Bases.
    Returns
    -------
        tE : Tree
            Eulerian field.
    '''
    base, basec = getEulerianBases(t)
    if base and basec: return C.newPyTree([base, basec])

def getEulerianBase(tE = []):
    '''
    Gets the base containing the node-centers Eulerian bases.

    Parameters
    ----------
        tE : Tree
            Eulerian field.
    Returns
    -------
        base : Base
            Eulerian Base (if any).
    '''
    return I.getNodeFromName1(tE, 'EulerianBase')

def getEulerianBaseCenter(tE = []):
    '''
    Gets the base containing the cell-centers Eulerian bases.

    Parameters
    ----------
        tE : Tree
            Eulerian field.
    Returns
    -------
        base : Base
            Eulerian Base (if any).
    '''
    return I.getNodeFromName1(tE, 'EulerianBaseCenter')

def getEulerianBases(tE = []):
    '''
    Gets the node and cell-centers Eulerian bases.

    Parameters
    ----------
        tÈ : Tree
            contains the Eulerian Bases.
    Returns
    -------
        tE : Tree
            Eulerian field.
    '''
    return getEulerianBase(tE), getEulerianBaseCenter(tE)

def getHybridParameters(t = []):
    '''
    Get a the Hybrid parameters.

    Parameters
    ----------
        t : Tree
            contains the Hybrid parameters named '.Hybrid#Parameters'.

    Returns
    -------
        HybridParameter : :py:class:`dict`
            Hybrid parameters.
    '''
    return J.get(getFreeParticles(t), '.Hybrid#Parameters')

####################################################################################################
####################################################################################################
######################################### IO/Visualisation #########################################
####################################################################################################
####################################################################################################
def setVisualization(t = [], ParticlesColorField = 'VorticityMagnitude',
    ParticlesRadius = '{Sigma}/5', addLiftingLineSurfaces = True, AirfoilPolars = []):
    '''
    Set the visualisation options for CPlot when the CGNS are saved.

    Parameters
    ----------
        t : Tree
            contains a zone of particles named 'FreeParticles'.

        ParticlesColorField : :py:class:`list`
            VPM field to color.

        ParticlesRadius : :py:class:`float`
            Radius of the visualised particles.

        addLiftingLineSurfaces : :py:class:`bool`
            States whether Lifting Line(s) surfaces are added for visualisation. Requires a
            valid AirfoilPolars.

        Polars : :py:func:`list` of :py:func:`zone` or :py:class:`str`
            Enhanced **Polars** for each airfoil, containing also foilwise
            distributions fields (``Cp``, ``theta``, ``delta1``...).

            .. note::
              if input type is a :py:class:`str`, then **Polars** is
              interpreted as a CGNS file name containing the airfoil polars data
    '''
    Particles = getFreeParticles(t)
    Sigma = I.getValue(I.getNodeFromName(Particles, 'Sigma'))
    C._initVars(Particles, 'radius=' + ParticlesRadius)
    FlowSolution = I.getNodeFromName1(Particles, 'FlowSolution')
    if not ParticlesColorField: ParticlesColorField = 'VorticityMagnitude'

    flag = True
    for Field in FlowSolution[2]:
        if ParticlesColorField == Field[0]: flag = False

    if flag:
        if ParticlesColorField == 'VorticityMagnitude': C._initVars(Particles,
                    'VorticityMagnitude=({VorticityX}**2 + {VorticityY}**2 + {VorticityZ}**2)**0.5')
        elif ParticlesColorField == 'StrengthMagnitude': C._initVars(Particles,
                                 'StrengthMagnitude=({AlphaX}**2 + {AlphaY}**2 + {AlphaZ}**2)**0.5')
        elif ParticlesColorField == 'VelocityMagnitude':
            U0 = I.getNodeFromName(Particles, 'VelocityFreestream')[1]
            C._initVars(Particles, 'VelocityMagnitude=(\
             (%g+{VelocityInducedX}+{VelocityPerturbationX}+{VelocityBEMX}+{VelocityInterfaceX})**2\
            +(%g+{VelocityInducedY}+{VelocityPerturbationY}+{VelocityBEMY}+{VelocityInterfaceY})**2\
            +(%g+{VelocityInducedZ}+{VelocityPerturbationZ}+{VelocityBEMZ}+{VelocityInterfaceZ})**2\
                                                                                            )**0.5'%
                                                                              (U0[0], U0[1], U0[2]))
        elif ParticlesColorField == 'rotU':
            C._initVars(Particles, 'rotU=(({gradyVelocityZ} - {gradzVelocityY})**2 + \
                ({gradzVelocityX} - {gradxVelocityZ})**2 + ({gradxVelocityY} - {gradyVelocityX})**2\
                                                                                            )**0.5')
    CPlot._addRender2Zone(Particles, material = 'Sphere',
             color = 'Iso:' + ParticlesColorField, blending = 0.6, shaderParameters = [0.04, 0])
    LiftingLines = getLiftingLines(t)
    for zone in LiftingLines:
        CPlot._addRender2Zone(zone, material = 'Flat', color = 'White', blending = 0.2)

    if addLiftingLineSurfaces and AirfoilPolars:
        LiftingLineSurfaces = []
        for ll in LiftingLines:
            surface = LL.postLiftingLine2Surface(ll, AirfoilPolars)
            surface[0] = ll[0] + '.surf'
            CPlot._addRender2Zone(surface, material = 'Solid', color = 'Grey',
                                                      meshOverlay = 1, shaderParameters=[1.,1.])
            LiftingLineSurfaces += [surface]
        I.createUniqueChild(t, 'LiftingLineSurfaces', 'CGNSBase_t',
            value = np.array([2, 3], order = 'F'), children = LiftingLineSurfaces)

    for zone in I.getZones(getHybridDomain(t)):
        CPlot._addRender2Zone(zone, material = 'Glass', color = 'White', blending = 0.6,
                                                      meshOverlay = 1, shaderParameters=[1.,1.])
    for zone in I.getZones(getPerturbationField(t)):
        CPlot._addRender2Zone(zone, material = 'Glass', color = 'White', blending = 0.6,
                                                      meshOverlay = 1, shaderParameters=[1.,1.])
    CPlot._addRender2PyTree(t, mode = 'Render', colormap = 'Blue2Red', isoLegend=1,
                                                              scalarField = ParticlesColorField)

def saveImage(t = [], ShowInScreen = False, ImagesDirectory = 'FRAMES', **DisplayOptions):
    '''
    Saves an image from the t tree.

    Parameters
    ----------
        t : Tree
            contains a zone of particles named 'FreeParticles'.

        ShowInScreen : :py:class:`bool`
            CPlot option.

        ImagesDirectory : :py:class:`str`
            Location where the image is written.

        DisplayOptions : :py:class:`dict`
                mode : :py:class:`dict`
                    CPlot Display mode (Render, Scalar, Vector, ...).

                displayInfo : :py:class:`int`
                
                colormap : :py:class:`str`
                    Color used for the visulation of the particles.
                
                win : :py:class:`tuple`
                    Image resolution.

                ... other CPlot options
    '''
    if 'mode' not in DisplayOptions: DisplayOptions['mode'] = 'Render'
    if 'displayInfo' not in DisplayOptions: DisplayOptions['displayInfo'] = 0
    if 'colormap' not in DisplayOptions: DisplayOptions['colormap'] = 0
    if 'win' not in DisplayOptions: DisplayOptions['win'] = (700, 700)
    DisplayOptions['exportResolution'] = '%gx%g'%DisplayOptions['win']

    try: os.makedirs(ImagesDirectory)
    except: pass

    DisplayOptions['export'] = os.path.join(ImagesDirectory,
        'frame%05d.png'%V.getParameter(t, 'CurrentIteration'))

    if ShowInScreen:
        DisplayOptions['offscreen'] = 0
    else:
        DisplayOptions['offscreen'] = 1

    CPlot.display(t, **DisplayOptions)
    if DisplayOptions['offscreen']:
        CPlot.finalizeExport(DisplayOptions['offscreen'])

def load(filename = ''):
    '''
    Opens the CGNS file designated by the user. If the CGNS contains particles, the VPM field
    is updated.

    Parameters
    ----------
        filename : :py:class:`str`
            Location of the CGNS to open.
    Returns
    -------
        t : Tree
    '''
    print(f'{"||":>57}\r' + '|| ', end='')
    t = C.convertFile2PyTree(filename)
    deletePrintedLines()
    return t

def checkSaveFields(SaveFields = ['all']):
    '''
    Updates the VPM fields to conserve when the particle zone is saved.

    Parameters
    ----------
        SaveFields : :py:class:`list` or numpy.ndarray of :py:class:`str`
            Fields to save. if 'all', then they are all saved.
    Returns
    -------
        FieldNames : :py:class:`list` or numpy.ndarray of :py:class:`str`
            Fields to save. if 'all', then they are all saved.
    ''' 
    VectorFieldNames = Vector_VPM_FlowSolution + vectorise(['rotU']) + vectorise(['Velocity'])
    ScalarFieldNames = Scalar_VPM_FlowSolution + ['StrengthMagnitude', 'rotU', \
                                                      'VelocityMagnitude', 'VorticityMagnitude']

    if not (isinstance(SaveFields, list) or isinstance(SaveFields, np.ndarray)):
        SaveFields = [SaveFields]

    SaveFields = np.array(SaveFields)
    FieldNames = []
    if 'all' in SaveFields:
        for VectorFieldName in VectorFieldNames:
            FieldNames += [VectorFieldName]

        for ScalarFieldName in ScalarFieldNames:
            FieldNames += [ScalarFieldName]
    else:
        for VectorFieldName in VectorFieldNames:
            if (VectorFieldName[:-1] in SaveFields) or (VectorFieldName in SaveFields):
                FieldNames += [VectorFieldName]

        for ScalarFieldName in ScalarFieldNames:
            if ScalarFieldName in SaveFields:
                FieldNames += [ScalarFieldName]

    FieldNames += vectorise('Alpha') + ['Age', 'Sigma', 'Nu', 'Cvisq', 'EnstrophyM1']
    return np.unique(FieldNames)

def checkTreeStructure(t = [], name = ''):
    '''
    Checks and updates the types of the nodes of the given entry.
    
    Parameters
    ----------
        t : Tree, base(s), Zone(s)
            Container to check and update.
    Returns
    -------
        t : Tree
            Checked tree.
    '''
    TypeOfInput = I.isStdNode(t)
    ERRMSG = J.FAIL + 't must be a tree, a list of bases or a list of zones' + J.ENDC
    if TypeOfInput == -1:# is a standard CGNS node
        if I.isTopTree(t):
            Bases = I.getBases(t)
            if len(Bases) == 1 and name: Bases[0][0] = name
        elif t[3] == 'CGNSBase_t':
            LiftingLineBase = t
            if name: LiftingLineBase[0] = name
            t = C.newPyTree([])
            t[2] = [LiftingLineBase]
        elif t[3] == 'Zone_t':
            if name: t = C.newPyTree([name, [t]])
            else: t = C.newPyTree([t])
        else:
            raise AttributeError(ERRMSG)
    elif TypeOfInput == 0:# is a list of CGNS nodes
        if t[0][3] == 'CGNSBase_t':
            Bases = I.getBases(t)
            t = C.newPyTree([])
            t[2] = Bases
        elif t[0][3] == 'Zone_t':
            Zones = I.getZones(t)
            if name: t = C.newPyTree([name, Zones])
            else: t = C.newPyTree([Zones])
        else:
            raise AttributeError(ERRMSG)
    else:
        raise AttributeError(ERRMSG)

    return t

def save(t = [], filename = '', VisualisationOptions = {}, SaveFields = checkSaveFields()):
    '''
    Saves the CGNS file designated by the user. If the CGNS contains particles, the VPM field
    saved are the one given by the user.

    Parameters
    ----------
        t : Tree

        filename : :py:class:`str`
            Location of the where t is saved.

        VisualisationOptions : :py:class:`dict`
            CPlot visualisation options.

        SaveFields : :py:class:`list` or numpy.ndarray of :py:class:`str`
            Particles fields to save (if any). if 'all', then they are all saved.
    '''
    tref = checkTreeStructure(I.copyRef(t))
    if VisualisationOptions:
        setVisualization(tref, **VisualisationOptions)
        SaveFields = np.append(SaveFields, ['radius'])

    Particles = getFreeParticles(tref)
    if I.getZones(Particles):
        I._rmNodesByName(Particles, 'BEMMatrix')
        if 'VelocityX' in SaveFields:
            u0 = np.array(I.getNodeFromName(Particles, 'VelocityFreestream')[1], dtype = str)
            C._initVars(Particles, 'VelocityX='+u0[0]+'+{VelocityInducedX}+{VelocityPerturbationX}+\
                                                                              {VelocityDiffusionX}')
            C._initVars(Particles, 'VelocityY='+u0[1]+'+{VelocityInducedY}+{VelocityPerturbationY}+\
                                                                              {VelocityDiffusionY}')
            C._initVars(Particles, 'VelocityZ='+u0[2]+'+{VelocityInducedZ}+{VelocityPerturbationZ}+\
                                                                              {VelocityDiffusionZ}')

        if 'VelocityMagnitude' in SaveFields:
            u0 = np.array(I.getNodeFromName(Particles, 'VelocityFreestream')[1], dtype = str)
            C._initVars(Particles, 'VelocityMagnitude=(\
              ('+u0[0]+'+{VelocityInducedX}+{VelocityPerturbationX}+{VelocityDiffusionX})**2 + \
              ('+u0[1]+'+{VelocityInducedY}+{VelocityPerturbationY}+{VelocityDiffusionY})**2 + \
              ('+u0[2]+'+{VelocityInducedZ}+{VelocityPerturbationZ}+{VelocityDiffusionZ})**2)**0.5')

        if 'rotUX' in SaveFields:
            C._initVars(Particles, 'rotUX={gradyVelocityZ} - {gradzVelocityY}')
            C._initVars(Particles, 'rotUY={gradzVelocityX} - {gradxVelocityZ}')
            C._initVars(Particles, 'rotUZ={gradxVelocityY} - {gradyVelocityX}')

        if 'rotU' in SaveFields:
            C._initVars(Particles, 'rotU=(({gradyVelocityZ} - {gradzVelocityY})**2 + \
                                          ({gradzVelocityX} - {gradxVelocityZ})**2 + \
                                          ({gradxVelocityY} - {gradyVelocityX})**2)**0.5')
        if 'VorticityMagnitude' in SaveFields:
            C._initVars(Particles, 'VorticityMagnitude=({VorticityX}**2 + {VorticityY}**2 + \
                                                                             {VorticityZ}**2)**0.5')
        if 'StrengthMagnitude' in SaveFields:
            C._initVars(Particles, 'StrengthMagnitude=({AlphaX}**2 + {AlphaY}**2 + {AlphaZ}**2)\
                                                                                             **0.5')

        FlowSolution = I.getNodeFromName(Particles, 'FlowSolution')
        rmNodes = []
        for Field in FlowSolution[2]:
            if Field[0] not in SaveFields: rmNodes += [Field[0]]

        for Node in rmNodes: I._rmNodesByName(FlowSolution, Node)

        # C._initVars(Particles, 'Theta={Enstrophy}/({StrengthMagnitude}*{rotU})')
        # Theta = I.getNodeFromName(Particles, 'Theta')
        # Theta[1] = 180./np.pi*np.arccos(Theta[1])
        I._sortByName(Particles)

    tE = getEulerianTree(tref)
    if tE:
        t, tc = getEulerianBases(tE)
        I._rmNodesByName(t, 'Density_M1')
        I._rmNodesByName(t, 'Density_P1')
        I._rmNodesByName(t, 'Temperature_M1')
        I._rmNodesByName(t, 'Temperature_P1')
        I._rmNodesByName(t, 'TurbulentSANuTilde_M1')
        I._rmNodesByName(t, 'TurbulentSANuTilde_P1')
        I._rmNodesByName(t, 'VelocityX_M1')
        I._rmNodesByName(t, 'VelocityY_M1')
        I._rmNodesByName(t, 'VelocityZ_M1')
        I._rmNodesByName(t, 'VelocityX_P1')
        I._rmNodesByName(t, 'VelocityY_P1')
        I._rmNodesByName(t, 'VelocityZ_P1')
        tE[2] = [I.getNodeFromName1(tE, 'CGNSLibraryVersion')] + [t]

    try:
        if os.path.islink(filename):
            os.unlink(filename)
        else:
            os.remove(filename)
    except:
        pass
        
    C.convertPyTree2File(tref, filename)
    deletePrintedLines()

def deletePrintedLines(NumberOfLineToDelete = 1):
    '''
    Deletes the last printed lines on the teminal.

    Parameters
    ----------
        NumberOfLineToDelete : :py:class:`int`
            Number of lines to delete.
    '''
    for i in range(NumberOfLineToDelete):
       sys.stdout.write('\x1b[1A')
       sys.stdout.write('\x1b[2K')

def blockPrint():
    '''
    Sets the global variable printBlocked to prevent from printing anything.
    '''
    sys.stdout = open(os.devnull, 'w')
    printBlocked[0] = True

def enablePrint():
    '''
    Sets the global variable printBlocked to allow from printing.
    '''
    sys.stdout = sys.__stdout__
    printBlocked[0] = False

def show(*msg, end = '\n'):
    '''
    Overloads the print function and bypasses the global variable printBlocked.
    '''
    blocked = printBlocked[0]
    enablePrint()
    print(*msg, end = end)#, sep=', ')
    if blocked: blockPrint()

def initialiseThreads(NumberOfThreads = 'auto'):
    '''
    Sets the number of threads.

    Parameters
    ----------
        NumberOfThreads : :py:class:`int` or :py:class:`str`
            Number of threads to set. If 'auto', the maximum available number of threads is set.
    '''
    if isinstance(NumberOfThreads, list) or isinstance(NumberOfThreads, np.ndarray):
        OMP_NUM_THREADS = NumberOfThreads[0]
    else: OMP_NUM_THREADS = NumberOfThreads
    if OMP_NUM_THREADS == 'auto':
        OMP_NUM_THREADS = int(os.getenv('OMP_NUM_THREADS', len(os.sched_getaffinity(0))))
    else:
        OMP_NUM_THREADS = int(min(OMP_NUM_THREADS, \
                                   int(os.getenv('OMP_NUM_THREADS', len(os.sched_getaffinity(0))))))

    architecture = mpi_init(OMP_NUM_THREADS)
    show(f'{"||":>57}\r' + '||' + '{:=^53}'.format(''))
    show(f'{"||":>57}\r' + '||' + '{:=^53}'.format(''))
    show(f'{"||":>57}\r' + '||' + '{:=^53}'.format(''))
    show(f'{"||":>57}\r' + '||' + '{:=^53}'.format(' Launching VULCAINS ' + __version__ + ' '))
    show(f'{"||":>57}\r' + '||' + '{:=^53}'.format(''))
    show(f'{"||":>57}\r' + '||' + '{:=^53}'.format(''))
    show(f'{"||":>57}\r' + '||' + '{:=^53}'.format(''))
    show(f'{"||":>57}\r' + '||' + '{:-^53}'.format(' CPU Architecture '))
    show(f'{"||":>57}\r' + '|| ' + '{:32}'.format('Number of threads') + ': ' + \
                                                                     '{:d}'.format(architecture[0]))
    if architecture[1] == 2:
        show(f'{"||":>57}\r' + '|| ' + '{:32}'.format('SIMD') + ': ' + \
                                                          '{:d}'.format(architecture[1]) + ' (SSE)')
    elif architecture[1] == 4:
        show(f'{"||":>57}\r' + '|| ' + '{:32}'.format('SIMD') + ': ' + \
                                                          '{:d}'.format(architecture[1]) + ' (AVX)')
    elif architecture[1] == 8:
        show(f'{"||":>57}\r' + '|| ' + '{:32}'.format('SIMD') + ': ' + \
                                                       '{:d}'.format(architecture[1]) + ' (AVX512)')
    else: show(f'{"||":>57}\r' + '|| ' + '{:32}'.format('') + ': ' + '{:d}'.format(architecture[1]))
    if architecture[2]: show(f'{"||":>57}\r' + '|| ' + '{:32}'.format('Precison') + ': ' + \
                                                                                 'double (64 bits)')
    else: show(f'{"||":>57}\r' + '|| ' + '{:32}'.format('Precison') + ': ' + 'single (32 bits)')
    show(f'{"||":>57}\r' + '||' + '{:=^53}'.format(''))

    return np.array([OMP_NUM_THREADS], dtype = np.int32, order = 'F')
