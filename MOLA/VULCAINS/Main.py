####################################################################################################
####################################### Import Python Modules ######################################
####################################################################################################
import sys
import os
import numpy as np
import importlib.util

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
if importlib.util.find_spec('Fast'):
    import Fast.PyTree as Fast
    import FastS.PyTree as FastS
    import FastC.PyTree as FastC

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
    'NearFieldSmoothingFactor', 'TimeFMM', 'FMMPerturbationOverlappingRatio',
    'TimeVelocityPerturbation', 'CirculationThreshold', 'CirculationRelaxationFactor', 'RPM',
    'VelocityTranslation', 'EulerianTimeStep', 'GenerationZones', 'HybridDomainSize',
                'MinimumSplitStrengthFactor', 'RelaxationRatio', 'RelaxationThreshold', 'Intensity']
                                                                                        
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
defaultVPMParameters = {
    ################################################################################################
    ######################################## VPM Parameters ########################################
    ################################################################################################
    ####################################### Fluid Parameters #######################################
        'Density'                      : 1.225,
        'EddyViscosityConstant'        : 0.15,
        'EddyViscosityModel'           : 'Vreman',
        'KinematicViscosity'           : 1.46e-5,
        'Temperature'                  : 288.15,
        'Time'                         : 0,
        'VelocityFreestream'           : np.array([0., 0., 0.]),
    ##################################### Numerical Parameters #####################################
        'AntiDiffusion'                : 0,
        'AntiStretching'               : 0,
        'CurrentIteration'             : 0,
        'DiffusionScheme'              : 'DVM',
        'IntegrationOrder'             : 1,
        'LowStorageIntegration'        : 1,
        'MachLimitor'                  : 0.5,
        'NumberOfBEMSources'           : 0,
        'NumberOfCFDSources'           : 0,
        'NumberOfHybridSources'        : 0,
        'NumberOfLiftingLines'         : 0,
        'NumberOfLiftingLineSources'   : 0,
        'NumberOfNodes'                : 0,
        'ParticleSizeVariationLimitor' : 1.1,
        'Resolution'                   : [None, None],
        'Sigma0'                       : [None, None],
        'SmoothingRatio'               : 2.,
        'StrengthVariationLimitor'     : 2,
        'TimeStep'                     : None,
        'VorticityEquationScheme'      : 'Transpose',
    ####################################### Particles Control ######################################
        'CutoffXmin'                    : -np.inf,
        'CutoffXmax'                    : +np.inf,
        'CutoffYmin'                    : -np.inf,
        'CutoffYmax'                    : +np.inf,
        'CutoffZmin'                    : -np.inf,
        'CutoffZmax'                    : +np.inf,
        'ForcedDissipation'             : 0,
        'MaximumAgeAllowed'             : 0,
        'MaximumAngleForMerging'        : 90,
        'MaximumMergingVorticityFactor' : 100,
        'MinimumOverlapForMerging'      : 3,
        'MinimumVorticityFactor'        : 0.001,
        'RedistributeParticlesBeyond'   : 0,
        'RedistributionPeriod'          : 1,
        'RealignmentRelaxationFactor'   : 0.,
        'MagnitudeRelaxationFactor'     : 0.,
        'EddyViscosityRelaxationFactor' : 0.005,
        'RemoveWeakParticlesBeyond'     : 0,
        'ResizeParticleFactor'          : 3,
        'StrengthRampAtbeginning'       : 50,
        'EnstrophyControlRamp'          : 100,
    ############################### Fast Multipole Method Parameters ###############################
        'ClusterSizeFactor'         : 10.,
        'FarFieldPolynomialOrder'   : 8,
        'IterationCounter'          : 0,
        'IterationTuningFMM'        : 50,
        'MaxParticlesPerCluster'    : 2**8,
        'NearFieldOverlapingFactor' : 3,
        'NearFieldSmoothingFactor'  : 2,
        'NumberOfThreads'           : 'auto',
        'TimeFMM'                   : 0,
    ################################ Perturbation Field Parameters #################################
        'FMMPerturbationOverlappingRatio' : 3,
        'TimeVelocityPerturbation'        : 0,
}

VPMParametersRange = {
    ################################################################################################
    ######################################## VPM Parameters ########################################
    ################################################################################################
    ####################################### Fluid Parameters #######################################
        'Density'                         : [0., +np.inf],
        'EddyViscosityConstant'           : [0., 1.],
        'EddyViscosityModel'              : [None, 'Vreman', 'Mansour', 'Mansour2', 'Smagorinsky'],
        'KinematicViscosity'              : [0., +np.inf],
        'Temperature'                     : [0., +np.inf],
        'Time'                            : [0., +np.inf],
        'VelocityFreestream'              : [-np.inf, +np.inf],
    ##################################### Numerical Parameters #####################################
        'AntiDiffusion'                   : [0., 1.],
        'AntiStretching'                  : [0., 1.],
        'CurrentIteration'                : [0, +np.inf],
        'DiffusionScheme'                 : [None, 'DVM', 'PSE', 'CSM'],
        'IntegrationOrder'                : [1, 4],
        'LowStorageIntegration'           : [0, 1],
        'MachLimitor'                     : [0., 1.],
        'NumberOfBEMSources'              : [0, +np.inf],
        'NumberOfCFDSources'              : [0, +np.inf],
        'NumberOfHybridSources'           : [0, +np.inf],
        'NumberOfLiftingLines'            : [0, +np.inf],
        'NumberOfLiftingLineSources'      : [0, +np.inf],
        'NumberOfNodes'                   : [0, +np.inf],
        'ParticleSizeVariationLimitor'    : [1, +np.inf],
        'Resolution'                      : [0., +np.inf],
        'Sigma0'                          : [0., +np.inf],
        'SmoothingRatio'                  : [1., +np.inf],
        'StrengthVariationLimitor'        : [1, +np.inf],
        'TimeStep'                        : [0., +np.inf],
        'VorticityEquationScheme'         : ['Transpose', 'Mixed', 'Classical'],
    ####################################### Particles Control ######################################
        'CutoffXmin'                      : [-np.inf, +np.inf],
        'CutoffXmax'                      : [-np.inf, +np.inf],
        'CutoffYmin'                      : [-np.inf, +np.inf],
        'CutoffYmax'                      : [-np.inf, +np.inf],
        'CutoffZmin'                      : [-np.inf, +np.inf],
        'CutoffZmax'                      : [-np.inf, +np.inf],
        'ForcedDissipation'               : [0, +np.inf],
        'MaximumAgeAllowed'               : [0, +np.inf],
        'MaximumAngleForMerging'          : [0., 180.],
        'MaximumMergingVorticityFactor'   : [0., +np.inf],
        'MinimumOverlapForMerging'        : [0., +np.inf],
        'MinimumVorticityFactor'          : [0., +np.inf],
        'RedistributeParticlesBeyond'     : [0., +np.inf],
        'RedistributionPeriod'            : [0, +np.inf],
        'RealignmentRelaxationFactor'     : [0., +np.inf],
        'MagnitudeRelaxationFactor'       : [0., +np.inf],
        'EddyViscosityRelaxationFactor'   : [0., 1.],
        'RemoveWeakParticlesBeyond'       : [0., +np.inf],
        'ResizeParticleFactor'            : [1., +np.inf],
        'StrengthRampAtbeginning'         : [0, +np.inf],
        'EnstrophyControlRamp'            : [0, +np.inf],
    ############################### Fast Multipole Method Parameters ###############################
        'ClusterSizeFactor'               : [0, +np.inf],
        'FarFieldPolynomialOrder'         : [4, 12],
        'IterationCounter'                : [0, +np.inf],
        'IterationTuningFMM'              : [1, +np.inf],
        'MaxParticlesPerCluster'          : [1, +np.inf],
        'NearFieldOverlapingFactor'       : [0., +np.inf],
        'NearFieldSmoothingFactor'        : [0., +np.inf],
        'NumberOfThreads'                 : [1, +np.inf],
        'TimeFMM'                         : [0., +np.inf],
    ################################ Perturbation Field Parameters #################################
        'FMMPerturbationOverlappingRatio' : [1., +np.inf],
        'TimeVelocityPerturbation'        : [0., +np.inf],
}

defaultLiftingLineParameters = {
    ################################################################################################
    ################################### Lifting Lines Parameters ###################################
    ################################################################################################
        'CirculationThreshold'             : 1e-4,
        'CirculationRelaxationFactor'      : 1./3.,
        'IntegralLaw'                      : 'linear',
        'MaxLiftingLineSubIterations'      : 100,
        'MinNbShedParticlesPerLiftingLine' : 26,
        'NumberOfParticleSources'          : 100,
        'ParticleDistribution'             : dict(kind = 'tanhTwoSides', FirstSegmentRatio = 2.,
                                                       LastSegmentRatio = 0.5, Symmetrical = False),
        'RPM'                              : 0.,
        'VelocityTranslation'              : [0., 0., 0.],
}

LiftingLineParametersRange = {
    ################################################################################################
    ################################### Lifting Lines Parameters ###################################
    ################################################################################################
        'CirculationThreshold'             : [0., 1.],
        'CirculationRelaxationFactor'      : [0., 1.],
        'IntegralLaw'                      : ['linear', 'uniform', 'tanhOneSide', 'tanhTwoSides',
                                                                                           'ratio'],
        'MaxLiftingLineSubIterations'      : [0, +np.inf],
        'MinNbShedParticlesPerLiftingLine' : [0, +np.inf],
        'NumberOfParticleSources'          : [0, +np.inf],
        'RPM'                              : [-np.inf, +np.inf],
        'VelocityTranslation'              : [-np.inf, +np.inf],
}

defaultHybridParameters = {
    ################################################################################################
    ####################################### Hybrid Parameters ######################################
    ################################################################################################
        'EulerianSubIterations'        : 30,
        'EulerianTimeStep'             : None,
        'GenerationZones'              : np.array([[-1., -1., -1., 1., 1., 1.]])*np.inf,
        'HybridDomainSize'             : None,
        'HybridRedistributionOrder'    : 2,
        'InnerDomainCellLayer'         : 0,
        'MaxHybridGenerationIteration' : 50,
        'MaximumSourcesPerLayer'       : 1000,
        'MinimumSplitStrengthFactor'   : 1.,
        'NumberOfBCCells'              : 1,
        'NumberOfBEMUnknown'           : 0,
        'NumberOfHybridLayers'         : 5,
        'OuterDomainCellOffset'        : 2,
        'ParticleGenerationMethod'     : 'BiCGSTAB',
        'RelaxationRatio'              : 1,
        'RelaxationThreshold'          : 1e-3,
}

HybridParametersRange = {
    ################################################################################################
    ####################################### Hybrid Parameters ######################################
    ################################################################################################
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
        'ParticleGenerationMethod'     : ['GMRES', 'BiCGSTAB', 'CG', 'Direct'],
        'RelaxationRatio'              : [0, +np.inf],
        'RelaxationThreshold'          : [0, +np.inf],
}

defaultVortexParameters = {
    ################################################################################################
    #################################### Vortex Rings parameters ###################################
    ################################################################################################
            'Intensity'    : 1.,
            'NumberLayers' : 6,
}

VortexParametersRange = {
    ################################################################################################
    #################################### Vortex Rings parameters ###################################
    ################################################################################################
            'Intensity'    : [-np.inf, +np.inf],
            'NumberLayers' : [0, +np.inf],
}

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
    Repeats all the input strings with  the axis coordinates 'X', 'Y' and 'Z' added at the end.

    Parameters
    ----------
        names : :py:class:`str` or list of :py:class:`str`
            Containes strings to vectorise.
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
    Build a particle tree.

    Parameters
    ----------
        Np : :py:class:`int`
            Size of the tree.
        FieldsNames : list of :py:class:`str`
            List of flow fields to initiate. By default, all the VPM flow fields are initialised.
    Returns
    -------
        t : Tree
            CGNS Tree a base of particles.
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
            Containes a Base of particles named 'Particles'.
    Returns
    -------
        Particles : Base
            Particle Base (if any).
    '''
    return I.getNodeFromName1(t, 'Particles')

def getFreeParticles(t = []):
    '''
    Gets the zone containing the free wake particles.

    Parameters
    ----------
        t : Tree
            Containes a Zone of particles named 'FreeParticles'.
    Returns
    -------
        Particles : Zone
            Free Particle Zone (if any).
    '''
    Particles = getParticles(t)
    if Particles: return I.getNodeFromName1(Particles, 'FreeParticles')

    Particles = I.getNodeFromName1(t, 'FreeParticles')
    if Particles: return Particles

    for z in I.getZones(t):
        if z[0] == 'FreeParticles':
            return [z]

def getBEMParticles(t = []):
    '''
    Gets the zone containing the BEM particles.

    Parameters
    ----------
        t : Tree
            Containes a Zone of particles named 'BEMParticles'.
    Returns
    -------
        Particles : Zone
            BEM Particle Zone (if any).
    '''
    Particles = getParticles(t)
    if Particles: return I.getNodeFromName1(Particles, 'BEMParticles')

    Particles = I.getNodeFromName1(t, 'BEMParticles')
    if Particles: return Particles

    for z in I.getZones(t):
        if z[0] == 'BEMParticles':
            return [z]

def getImmersedParticles(t = []):
    '''
    Gets the zone containing the Eulerian Immersed particles.

    Parameters
    ----------
        t : Tree
            Containes a Zone of particles named 'ImmersedParticles'.
    Returns
    -------
        Particles : Zone
            Eulerian Immersed Particle Zone (if any).
    '''
    Particles = getParticles(t)
    if Particles: return I.getNodeFromName1(Particles, 'ImmersedParticles')

    Particles = I.getNodeFromName1(t, 'ImmersedParticles')
    if Particles: return Particles

    for z in I.getZones(t):
        if z[0] == 'ImmersedParticles':
            return [z]

def getParticlesTree(t = []):
    '''
    Gets the tree containing the all the VPM particles.

    Parameters
    ----------
        t : Tree
            Containes a Base named 'Particles'.
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
            Containes a Base named 'PerturbationField'.
    Returns
    -------
        PerturbationField : Base
            PerturbationField Base (if any).
    '''
    return I.getNodeFromName1(t, 'PerturbationField')

def getPerturbationFieldTree(t = []):
    '''
    Gets the tree containing the Perturbation field.

    Parameters
    ----------
        t : Tree
            Containes a Base named 'PerturbationField'.
    Returns
    -------
        PerturbationField : Tree
            PerturbationField Tree (if any).
    '''
    PerturbationField = I.getZones(getPerturbationField(t))
    if PerturbationField: return C.newPyTree(['PerturbationField', PerturbationField])

def getVPMParameters(t = []):
    '''
    Get a the VPM parameters.

    Parameters
    ----------
        t : Tree
            Containes the VPM parameters named '.VPM#Parameters'.

    Returns
    -------
        VPMParameter : :py:class:`dict`
            VPM parameters.
    '''
    return J.get(getFreeParticles(t), '.VPM#Parameters')

def getParticlesNumber(t = [], pointer = False):
    '''
    Get a the free VPM particles number.

    Parameters
    ----------
        t : Tree
            Containes the free VPM parameters named 'FreeParticles'.
        pointer : :py:class:`bool`
            States whether the pointer or the value of the size of the zone is returned.

    Returns
    -------
        VPMParameter : :py:class:`dict`
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
            Containes the free VPM parameters named 'BEMParticles'.
        pointer : :py:class:`bool`
            States whether the pointer or the value of the size of the zone is returned.

    Returns
    -------
        VPMParameter : :py:class:`dict`
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
            Containes the free VPM parameters named 'ImmersedParticles'.
        pointer : :py:class:`bool`
            States whether the pointer or the value of the size of the zone is returned.

    Returns
    -------
        VPMParameter : :py:class:`dict`
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
            Containes Zones of Coordinates.
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
            Containes the Coordinates onto wich the Perturbation velocity fields are interpolated.

        tL : Tree
            Containes the VPM parameters.

        tP : Tree
            Containes the PerturbationFields.
    '''
    for var in vectorise('VelocityPerturbation'): C._initVars(Targets, var, 0.)
    _tL, _tP = getTrees([tL, tP], ['Particles', 'Perturbation'])
    if tP_Capsule[0] and _tP:
        Theta, NumberOfNodes, TimeVelPert = getParameters(_tL, ['FMMPerturbationOverlappingRatio',
                                                       'NumberOfNodes', 'TimeVelocityPerturbation'])
        TimeVelPert[0] += extract_perturbation_velocity_field(C.newPyTree(I.getZones(Targets)),
                                                     _tP, tP_Capsule[0], NumberOfNodes[0], Theta[0])

####################################################################################################
####################################################################################################
######################################## CGNS tree control #########################################
####################################################################################################
####################################################################################################
def checkTrees(t = [], VPMParameters = {}, HybridParameters = {}):
    '''
    Checks if the minimum recquirements are met within the Bases in t to launch VULCAINS
    computations. The necessary parameters and fields are initialised if missing. Allready existing 
    parameters are updated with the default parameters and the parameters inputs.

    Parameters
    ----------
        t : Tree
            Countainers to check.
        VPMParameters : :py:class:`dict`
            Containes the VPM parameters to put in the checked countainers.
        HybridParameters : :py:class:`dict`
            Containes the Hybrid parameters to put in the checked countainers.
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

    newVPMParameters = defaultVPMParameters.copy()
    newVPMParameters.update(getVPMParameters(tL))
    newVPMParameters.update(VPMParameters)
    checkParameters(VPMParameters = newVPMParameters)
    J.set(Particles, '.VPM#Parameters', **newVPMParameters)
    I._sortByName(I.getNodeFromName1(Particles, '.VPM#Parameters'))

    OldHybridParameters = getHybridParameters(tL)
    if OldHybridParameters:
        newHybridParameters = defaultHybridParameters.copy()
        newHybridParameters.update(OldHybridParameters)
        newHybridParameters.update(HybridParameters)
        checkParameters(HybridParameters = newHybridParameters)
        J.set(Particles, '.Hybrid#Parameters', **newHybridParameters)
        I._sortByName(I.getNodeFromName1(Particles, '.Hybrid#Parameters'))

    t[2] = [I.getNodeFromName1(t, 'CGNSLibraryVersion')] + I.getBases(tL) + I.getBases(tE) + \
                                                   I.getBases(tH) + I.getBases(tLL) + I.getBases(tP)

def getTrees(Trees = [], TreeNames = [], fillEmptyTrees = False):
    '''
    Checks if the minimum recquirements are met within the Bases in t to launch VULCAINS
    computations. The necessary parameters and fields are initialised if missing.

    Parameters
    ----------
        Targets : Tree
            Containes the Coordinates onto wich the Perturbation velocity fields are interpolated.
        tL : Tree
            Containes the VPM parameters.
        tP : Tree
            Containes the PerturbationFields.
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

def checkParameters(VPMParameters = {}, LiftingLineParameters = {}, HybridParameters = {},
    VortexParameters = {}):
    '''
    Imposes the types of the parameters in each dictionnary used by VULCAINS. If a parameter is not
    provided, a default value will be prescribed. Parameters are modified (if necessary) to fit in
                                                                            their operability range.

    Parameters
    ----------
        VPMParameters : :py:class:`dict`
            List of the parameters linked to the VPM solver or the perturbation field. These
            parameters are:
        ############################################################################################
        ###################################### VPM Parameters ######################################
        ############################################################################################

        ##################################### Fluid Parameters #####################################
            Density : :py:class:`float`
                ]0., +inf[, fluid density at infinity, in kg.m^-3.
                Default value: 1.225
            EddyViscosityConstant : :py:class:`float`
                [0., +inf[, constant for the eddy viscosity model.
                Default value: 0.15
            EddyViscosityModel : :py:class:`str`
                Mansour, Mansour2, Smagorinsky, Vreman or None, select a LES model to compute the
                                                                                     eddy viscosity.
                Default value: 'Vreman'
            KinematicViscosity : :py:class:`float`
                [0., +inf[, fluid kinematic viscosity at infinity, in m^2.s^-1.
                Default value: 1.46e-5
            Temperature : :py:class:`float`
                ]0., +inf[, fluid temperature at infinity, in K.
                Default value: 288.15
            Time : :py:class:`float`
                [0., +inf[, physical time, in s.
                Default value: 0
            VelocityFreestream : numpy.ndarray of :py:class:`float`
                ]-inf, +inf[^3, fluid velocity at infinity, in m.s-1.
                Default value: [0., 0., 0.]

        ################################### Numerical Parameters ###################################
            AntiDiffusion : :py:class:`float`
                [0. 1.], vortex diffusion either modifies only the particle strength.
                (AntiStretching = 0), or the particle size (AntiStretching = 1)
                Default value: 0
            AntiStretching : :py:class:`float`
                [0. 1.], vortex stretching either modifies only the particle strength.
                (AntiStretching = 0), or the particle size (AntiStretching = 1)
                Default value: 0
            CurrentIteration : :py:class:`int`
                [|0, +inf[|, follows the current iteration.
                Default value: 0
            DiffusionScheme : :py:class:`str`
                DVM, PSE, CSM or None, gives the scheme used to compute the diffusion term of the
                                                                                 vorticity equation.
                Default value: 'DVM'
            IntegrationOrder : :py:class:`int`
                [|1, 4|], 1st, 2nd, 3rd or 4th order Runge Kutta.
                Default value: 1
            LowStorageIntegration : :py:class:`int`
                [|0, 1|], states if the classical or the low-storage Runge Kutta is used.
                Default value: 1
            MachLimitor : :py:class:`float`
                [0, +inf[, sets the maximum induced velocity a particle can have. Does not take into
                                                                     account the VelocityFreestream.
                Default value: 0.5
            NumberOfBEMSources : :py:class:`int`
                [|0, +inf[|, total number of embedded Boundary Element Method particles on the
                                                                                   solid boundaries.
                Default value: 0
            NumberOfCFDSources : :py:class:`int`
                [|0, +inf[|, total number of embedded Eulerian Immersed particles on the Hybrid
                                                                                    Inner Interface.
                Default value: 0
            NumberOfHybridSources : :py:class:`int`
                [|0, +inf[|, total number of hybrid particles generated in the Hybrid Domain.
                Default value: 0
            NumberOfLiftingLines : :py:class:`int`
                [|0, +inf[|, number of LiftingLines.
                Default value: 0
            NumberOfLiftingLineSources : :py:class:`int`
                [|0, +inf[|, total number of embedded source particles on the LiftingLines.
                Default value: 0
            NumberOfNodes : :py:class:`int`
                [|0, +inf[|, total number of nodes in the velocity perturbation field grid.
                Default value: 0
            ParticleSizeVariationLimitor : :py:class:`float`
                [1, +inf[, gives the maximum a particle can grow/shrink during an iteration.
                Default value: 1.1
            Resolution : numpy.ndarray of :py:class:`float`
                ]0., +inf[^2, minimum and maximum resolution scale of the VPM.
                Default value: np.array([TimeStep*np.linalg.norm(VelocityFreestream)]*2)
            Sigma0 : numpy.ndarray of :py:class:`float`
                ]0., +inf[^2, initial minimum and maximum size of the particles.
                Default value: Resolution*SmoothingRatio
            SmoothingRatio : :py:class:`float`
                [0., 5.], smoothes the particle interactions to avoid inducing singularities.
                Default value: 2.
            StrengthVariationLimitor : :py:class:`float`
                [1, +inf[, gives the maximum variation the strength of a particle can have during an
                                                                                          iteration.
                Default value: 2
            TimeStep : :py:class:`float`
                ]0., +inf[, time step of the VPM, ins s.
                Default value: np.min(Resolution)/np.linalg.norm(VelocityFreestream)
            VorticityEquationScheme : :py:class:`str`
                Transpose, Classical or Mixed, The schemes used to compute the vstretching term of
                                                                             the vorticity equation.
                Default value: 'Transpose'

        ##################################### Particles Control ####################################
            CutoffXmin : :py:class:`float`
                ]-inf, +inf[, particles beyond this spatial Cutoff are deleted, in m.
                Default value: -np.inf
            CutoffXmax : :py:class:`float`
                ]-inf, +inf[, particles beyond this spatial Cutoff are deleted, in m.
                Default value: +np.inf
            CutoffYmin : :py:class:`float`
                ]-inf, +inf[, particles beyond this spatial Cutoff are deleted, in m.
                Default value: -np.inf
            CutoffYmax : :py:class:`float`
                ]-inf, +inf[, particles beyond this spatial Cutoff are deleted, in m.
                Default value: +np.inf
            CutoffZmin : :py:class:`float`
                ]-inf, +inf[, particles beyond this spatial Cutoff are deleted, in m.
                Default value: -np.inf
            CutoffZmax : :py:class:`float`
                ]-inf, +inf[, particles beyond this spatial Cutoff are deleted, in m.
                Default value: +np.inf
            ForcedDissipation : :py:class:`float`
                [0, +inf[, gives the % of strength the particles loose per sec, in %/s
                Default value: 0
            MaximumAgeAllowed : :py:class:`int`
                [|0, +inf[|, particles older than MaximumAgeAllowed iterations are deleted. If
                                                       MaximumAgeAllowed == 0, they are not deleted.
                Default value: 0
            MaximumAngleForMerging : :py:class:`float`
                [0., 180.[, maximum angle allowed between two particles to be merged, in deg.
                Default value: 90
            MaximumMergingVorticityFactor : :py:class:`float`
                [0, +inf[, particles that have their strength above (resp. below)
                MaximumMergingVorticityFactor times the maximum particle strength of the Lifting
                Line embedded particles and the hybrid particles combined are split (resp. merged),
                                                                                               in %.
                Default value: 100
            MinimumOverlapForMerging : :py:class:`float`
                [0., +inf[, particles are merged if their distance is below their size (Sigma) times
                                                                           MinimumOverlapForMerging.
                Default value: 3
            MinimumVorticityFactor : :py:class:`float`
                [0., +inf[, particles that have their strength below MinimumVorticityFactor times
                the maximum particle strength of the Lifting Line embedded particles and the hybrid
                                                               particles combined are deleted, in %.
                Default value: 0.001
            RedistributeParticlesBeyond : :py:class:`float`
                [0., +inf[, do not split/merge particles if closer than 
                      RedistributeParticlesBeyond*Resolution from any Lifting Line or Hybrid Domain.
                Default value: 0
            RedistributionPeriod : :py:class:`int`
                [|0, +inf[|, iteration frequency at which particles are tested for
                                       splitting/merging. If 0 the particles are never split/merged.
                Default value: 1
            RealignmentRelaxationFactor : :py:class:`float`
                [0., +inf[, filters the particles direction to realign the particles with their
                                                  vorticity to have divergence-free vorticity field.
                Default value: 0.
            MagnitudeRelaxationFactor : :py:class:`float`
                [0., +inf[, filters the particles strength magnitude to have divergence-free
                                                                                    vorticity field.
                Default value: 0.
            EddyViscosityRelaxationFactor : :py:class:`float`
                [0., 1.[, modifies the EddyViscosityConstant of every particles by
                EddyViscosityRelaxationFactor at each iteration according to the local loss of
                                                                         Enstrophy of the particles.
                Default value: 0.005
            RemoveWeakParticlesBeyond : :py:class:`float`
                [0., +inf[, do not remove weak particles if closer than 
                      RedistributeParticlesBeyond*Resolution from any Lifting Line or Hybrid Domain.
                Default value: 0
            ResizeParticleFactor : :py:class:`float`
                [0, +inf[, resize particles that grow/shrink past ResizeParticleFactor their
                 original size (given by Sigma0). If ResizeParticleFactor == 0, no resizing is done.
                Default value: 3
            StrengthRampAtbeginning : :py:class:`int`
                [|0, +inf[|, put a sinusoidal ramp on the magnitude of the vorticity shed for the
                                         StrengthRampAtbeginning first iterations of the simulation.
                Default value: 50
            EnstrophyControlRamp : :py:class:`int`
                [|0, +inf[|, put a sinusoidal ramp on the Enstrophy filter applied to the particles
                 for the EnstrophyControlRamp first iterations after the shedding of each particles.
                Default value: 100

        ############################# Fast Multipole Method Parameters #############################
            ClusterSizeFactor : :py:class:`float`
                [|0, +inf[|, FMM clusters smaller than Resolution*ClusterSizeFactor cannot be
                                                                      divided into smaller clusters.
                Default value: 10
            FarFieldPolynomialOrder : :py:class:`int`
                [|4, 12|], order of the polynomial which approximates the long distance particle
                          interactions by the FMM, the higher the more accurate and the more costly.
                Default value: 8
            IterationCounter : :py:class:`int`
            [|0, +inf[|, keeps track of how many iteration past since the last IterationTuningFMM.
                Default value: 0
            IterationTuningFMM : :py:class:`int`
                [|0, +inf[|, frequency at which the FMM is compared to the direct computation,
                                          shows the relative L2 error made by the FMM approximation.
                Default value: 50
            MaxParticlesPerCluster : :py:class:`int`
                [1, +inf[, FMM clusters with less than MaxParticlesPerCluster particles cannot be
                                                                      divided into smaller clusters.
                Default value: 2**8
            NearFieldOverlapingFactor : :py:class:`float`
                [1., +inf[, particle interactions are approximated by the FMM as soon as two
                clusters of particles are separated by at least NearFieldOverlapingFactor the size
                  of the particles in the cluster, the higher the more accurate and the more costly.
                Default value: 3
            NearFieldSmoothingFactor : :py:class:`float`
                [1., NearFieldOverlapingFactor], particle interactions are smoothed as soon as two
                clusters of particles are separated by at most NearFieldSmoothingFactor the size of
                     the particles in the cluster, the higher the more accurate and the more costly.
                Default value: 2
            NumberOfThreads : :py:class:`int`
                [|1, OMP_NUM_THREADS|], number of threads of the machine used. If 'auto', the
                                                                    highest number of threads is set
                Default value: 'auto'
            TimeFMM : :py:class:`float`
                [0, +inf[, keeps track of the CPU time spent by the FMM for the computation of the
                particle interactions, in s.
                Default value: 0

        ############################## Perturbation Field Parameters ###############################
            FMMPerturbationOverlappingRatio : :py:class:`float`
                [1., +inf[, perturbation grid interpolations are approximated by the FMM as soon as
                two clusters of cells are separated by at least NearFieldOverlapingFactor the size
                                   of the cluster, the higher the more accurate and the more costly.
                Default value: 3
            TimeVelocityPerturbation : :py:class:`float`
                [0, +inf[, keeps track of the CPU time spent by the FMM for the interpolation of the
                perturbation mesh, in s.
                Default value: 0

        LiftingLineParameters : :py:class:`dict`
            List of the parameters linked to the Lifting Lines. These parameters are only imposed if
            they not already present in the .VPM#Parameters of the Lifting Lines. Each Lifting Line
            can have its own set of parameters. These parameters are:
        ############################################################################################
        ################################# Lifting Lines Parameters #################################
        ############################################################################################
            CirculationThreshold : :py:class:`float`
                ]0., 1], convergence criteria for the circulation sub-iteration process to shed the
                                                                   particles from the Lifting Lines.
                Default value: 1e-4
            CirculationRelaxationFactor : :py:class:`float`
                ]0., 1.], relaxation parameter of the circulation sub-iterations, the more unstable
                                                             the simulation, the lower it should be.
                Default value: 1./3.
            IntegralLaw : :py:class:`str`
                linear, uniform, tanhOneSide, tanhTwoSides or ratio, gives the type of interpolation
                of the circulation from the Lifting Lines sections onto the particles sources
                                                                      embedded on the Lifting Lines.
                Default value: 'linear`
            MaxLiftingLineSubIterations : :py:class:`int`
                [|0, +inf[|, max number of sub-iteration when sheding the particles from the Lifting
                                                                                              Lines.
                Default value: 100
            MinNbShedParticlesPerLiftingLine : :py:class:`int`
                [|0, +inf[|, Lifting Lines cannot have less than MinNbShedParticlesPerLiftingLine
                particle sources. This parameter is imposed indiscriminately on all the Lifting
                                                                                              Lines.
                Default value: 26
            NumberOfParticleSources : :py:class:`int`
                [|0, +inf[|, number of particle sources on the Lifting Lines.
                Default value: 100
            ParticleDistribution : :py:class:`dict`
                Provides with the repartition of the particle sources on the Lifting Lines.
                Default value: dict(kind = 'tanhTwoSides', FirstSegmentRatio = 2.,
                                                        LastSegmentRatio = 0.5, Symmetrical = False)
                    kind : :py:class:`str`
                        uniform, tanhOneSide, tanhTwoSides or ratio, repatition law of the particle.
                        Default value: 'tanhTwoSides'
                    FirstSegmentRatio : :py:class:`float`
                        ]0., +inf[, particles at the root of the Lifting Line are spaced by
                                                   FirstSegmentRatio times their size, i.e., Sigma0.
                        Default value: 2.
                    LastSegmentRatio : :py:class:`float`
                        ]0., +inf[, particles at the tip of the Lifting Line are spaced by
                                                   LastSegmentRatio times their size, i.e., Sigma0.
                        Default value: 0.5
                    Symmetrical : :py:class:`bool`
                        [|0, 1|], forces or not the symmetry of the particle sources on the Lifting
                                                                                              Lines.
            RPM : :py:class:`float`
                [0, +inf], revolution per minute of the Lifting Lines, rev.min-1
                Default value: 0.
            VelocityTranslation : numpy.ndarray of :py:class:`float`
            ]-inf, +inf[^3, translation velocity of the Lifting Lines.
                Default value: [0., 0., 0.]

        HybridParameters : :py:class:`dict`
            List of the parameters linked to the Eulerian-Lagrangian hybridisation. These parameters
                                                                                                are:
        ############################################################################################
        ##################################### Hybrid Parameters ####################################
        ############################################################################################
            EulerianSubIterations : :py:class:`int`
                [|0, +inf[|, number of sub-iterations for the Eulerian solver
                Default value: 30
            EulerianTimeStep : :py:class:`float`
                ]0., +inf[, timestep for the Eulerian solver, in s.
                Default value: TimeStep
            GenerationZones : list of :py:class:`list` or list of numpy.ndarray of :py:class:`float`
                The Eulerian vorticity sources are only considered if within GenerationZones, in m.
                Default value: np.array([[-np.inf, -np.inf, -np.inf, np.inf, np.inf, np.inf]])
            HybridDomainSize : :py:class:`float`
                ]0., +inf[, size of the Hybrid Domain contained between the Outer and Inner
                                                                                         Interfaces.
                Default value: 0                                                                                         
            HybridRedistributionOrder : :py:class:`int`
            [|1, 5|], order of the polynomial used to redistribute the generated particles on a
                                                                             regular cartesian grid.
                Default value: 2
            InnerDomainCellLayer : :py:class:`int`
                |]0, +inf[|, gives the position of the beginning of the Hybrid Domain, i.e., the
                position of the Inner Interface. The Hybrid Domain is thus starts 2 ghost cells +
                NumberOfBCCells + OuterDomainCellOffset + InnerDomainCellLayer layers of cells from
                the exterior boundary of the Eulerian mesh.
                Default value: 0
            MaxHybridGenerationIteration : :py:class:`int`
                [|0, +inf[|, max number of sub-iterations for the iterative particle generation
                                                                                             method.
                Default value: 50
            MaximumSourcesPerLayer : :py:class:`int`
                [|0, +inf[|, max number of vorticity sources in each layer of the Hybrid Domain.
                Default value: 1000
            MinimumSplitStrengthFactor : :py:class:`float`
                [0., +inf[, in %, sets the minimum particle strength kept per layer after generation
                of the hybrid particles. The strength threshold is set as a percentage of the
                             maximum strength in the hybrid domain times MinimumSplitStrengthFactor.
                Default value: 1.
            NumberOfBCCells : :py:class:`int`
                |]0, +inf[|, number of layers cells on which the BC farfield is imposed by the VPM.
                Default value: 1
            NumberOfBEMUnknown : :py:class:`int`
                [|0, 3|], number of unknown for the BEM. If NumberOfBEMUnknown == 0: sources and
                vortex sheets are given with an initial guess but not solved. If NumberOfBEMUnknown
                == 1: only sources are solved. If NumberOfBEMUnknown == 2: only vortex sheets are
                      solved. If NumberOfBEMUnknown == 3: both sources and vortex sheets are solved.
                Default value: 0
            NumberOfHybridLayers : :py:class:`int`
                |]0, +inf[|, number of layers dividing the Hybrid Domain.
                Default value: 5
            OuterDomainCellOffset : :py:class:`int`
                |]0, +inf[|, offsets the position of the Hybrid Domain by OuterDomainCellOffset from
                the far field BC imposed by the VPM. The Hybrid Domain thus ends 2 ghost cells +
                NumberOfBCCells + OuterDomainCellOffset layers of cells from the exterior boundary
                                                                               of the Eulerian mesh.
                Default value: 2
            ParticleGenerationMethod : :py:class:`str`
                GMRES, BiCGSTAB, CG or Direct, gives the iterative methode to compute the strength
                of the hybrid particles fom the Euerian vorticity sources. Selects the Generalized
                Minimal Residual, Bi-Conjugate Gradient Stabilised, Conjugate Gradient or Direct
                                                                            Resolution from LAPACKE.
                Default value: 'BiCGSTAB'
            RelaxationRatio : :py:class:`float`
                [|0, +inf[|, dynamically updates the iterative method convergence criteria for the
                relative error of the vorticity induced by the generated particles to be as close as
                                                                    possible to RelaxationThreshold.
                Default value: 1
            RelaxationThreshold : :py:class:`float`
                [0, +inf[ in m^3.s^-1, gives the convergence criteria for the iterative particle
                                                                                  generation method.
                Default value: 1e-3

        VortexParameters : :py:class:`dict`
            List of the parameters linked to the free flow configurations. These parameters are:
        ############################################################################################
        ################################## Vortex Rings parameters #################################
        ############################################################################################
            Intensity : :py:class:`float`
                [0., +inf[, vortex intensity, in m^2.s-1.
                Default value: 1.
            NumberLayers : :py:class:`int`
                [|1., +inf[|, number of layers of particles composing the vortex structure.
                Default value: 6
    '''
    defaultParameters = [defaultVPMParameters, defaultLiftingLineParameters,
                                                   defaultHybridParameters, defaultVortexParameters]
    Parameters = [VPMParameters, LiftingLineParameters, HybridParameters, VortexParameters]
    Ranges = [VPMParametersRange, LiftingLineParametersRange, HybridParametersRange,
                                                                              VortexParametersRange]
    def checkRange(Parameter = {}, Range = {}, key = ''):
        if np.min(Parameter[key]) < Range[key][0]:
            Parameter[key][Parameter[key] < Range[key][0]] = Range[key][0]
            print(J.WARN + key + ' outside range. Set to ' + str(Parameter[key]) + J.ENDC)
        if Range[key][-1] < np.max(Parameter[key]):
            Parameter[key][Range[key][1] < Parameter[key]] = Range[key][1]
            print(J.WARN + key + ' outside range. Set to ' + str(Parameter[key]) + J.ENDC)

    for defaultParameter, Parameter, Range in zip(defaultParameters, Parameters, Ranges):
        tmp = defaultParameter.copy()
        tmp.update(Parameter)
        Parameter.update(tmp)
        deleteKeys = []
        for key in Parameter:
            if key in float_Params:
                Parameter[key] = np.atleast_1d(np.array(Parameter[key], order = 'F',
                                                                                dtype = np.float64))
                if np.isnan(Parameter[key]).any():
                    Parameter[key] = np.array([None]*len(Parameter[key]), order = 'F')
                else: checkRange(Parameter, Range, key)
            elif key in int_Params:
                Parameter[key] = np.atleast_1d(np.array(Parameter[key], order = 'F',
                                                                                  dtype = np.int32))
                checkRange(Parameter, Range, key)
            elif key in str_Params:
                if not(isinstance(Parameter[key], str) or isinstance(Parameter[key], bytes) or \
                    isinstance(Parameter[key], np.str_) or isinstance(Parameter[key], np.bytes_) \
                                                        or isinstance(Parameter[key], type(None))):
                    print(J.WARN + key + ' should be a string, not a ' + str(type(Parameter[key]))\
                                                     + '. Set to ' + defaultParameter[key] + J.ENDC)
                    Parameter[key] = defaultParameter[key]
                if Parameter[key] not in Range[key]:
                    print(J.WARN + 'Choose another ' + key +' parameter, ' + str(Parameter[key]) + \
                                                ' does not exist. Set to ' + Range[key][0] + J.ENDC)
                    Parameter[key] = Range[key]

                Parameter[key] = np.str_(Parameter[key])
            elif type(Parameter[key]) != dict : deleteKeys += [key]

        if 'NumberOfThreads' in deleteKeys: deleteKeys.remove('NumberOfThreads')
        for key in deleteKeys:
            print(J.WARN + key + ' does not exist. Parameter deleted.' + J.ENDC)
            del Parameter[key]

    if VPMParameters['Resolution'].all():
        VPMParameters['Resolution'] = np.array([np.min(VPMParameters['Resolution']),
                              np.max(VPMParameters['Resolution'])], dtype = np.float64, order = 'F')

    VPMParameters['CutoffXmin'][0], VPMParameters['CutoffXmax'][0] = \
                               min(VPMParameters['CutoffXmin'][0], VPMParameters['CutoffXmax'][0]),\
                               max(VPMParameters['CutoffXmin'][0], VPMParameters['CutoffXmax'][0])
    VPMParameters['CutoffYmin'][0], VPMParameters['CutoffYmax'][0] = \
                               min(VPMParameters['CutoffYmin'][0], VPMParameters['CutoffYmax'][0]),\
                               max(VPMParameters['CutoffYmin'][0], VPMParameters['CutoffYmax'][0])
    VPMParameters['CutoffZmin'][0], VPMParameters['CutoffZmax'][0] = \
                               min(VPMParameters['CutoffZmin'][0], VPMParameters['CutoffZmax'][0]),\
                               max(VPMParameters['CutoffZmin'][0], VPMParameters['CutoffZmax'][0])
    if VPMParameters['Resolution'][0]:
        VPMParameters['Sigma0'] = VPMParameters['Resolution']*VPMParameters['SmoothingRatio'][0]

def delete(t = [], mask = []):
    '''
    Deletes the free particles inside t that are flagged by mask.

    Parameters
    ----------
        t : Tree
            Containes a zone of particles named 'FreeParticles'.

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
            Containes a zone of particles named 'FreeParticles'.

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
            Containes a zone of particles named 'FreeParticles'.

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
            Containes a zone of particles named 'FreeParticles'.

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
            Containes a zone of particles named 'FreeParticles'.

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
            Containes a zone of particles named 'FreeParticles'.

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
            Containes a node of parameters named '.PerturbationField#Parameters'.

    Returns
    -------
        PerturbationFieldParameters : :py:class:`dict`
            Dictionnary of parameters. The parameters are pointers inside numpy ndarrays.
    '''
    return J.get(getFreeParticles(t), '.PerturbationField#Parameters')

def getParameter(t = [], Name = ''):
    '''
    Get a parameter.

    Parameters
    ----------
        t : Tree
            Containes a node of parameters where one of them is named Name.

        Name : :py:class:`str`
            Name of the parameter to get.

    Returns
    -------
        ParameterNode : :py:class:`dict`
            The parameter is a pointer inside a numpy ndarray.
    '''
    Particles = getFreeParticles(t)
    Node = getVPMParameters(Particles)
    if Name in Node: ParameterNode = Node[Name]
    else:
        Node = getHybridParameters(Particles)
        if Node and Name in Node: ParameterNode = Node[Name]
        else: ParameterNode = None
    return ParameterNode

def getParameters(t = [], Names = []):
    '''
    Get a list of parameters.

    Parameters
    ----------
        t : Tree
            Containes a node of parameters with their names in Names.

        Names : :py:class:`list` or numpy.ndarray of :py:class:`str`
            List of parameter names

    Returns
    -------
        ParameterNode : :py:class:`dict`
            The parameter is a pointer inside a numpy ndarray.
    '''
    Particles = getFreeParticles(t)
    return [getParameter(Particles, Name) for Name in Names]

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
            Containes a zone of particles named 'FreeParticles'.
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
            Containes a Base named 'LiftingLines'.
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
        tL: Tree
            Lagrangian field.

        tLL : Tree
            Lifting Lines.

        NumberParticlesShedAtTip : :py:class:`int`
            Number of particles to shed per TimeStep.
    '''
    if not tLL: raise AttributeError('The time step is not given and can not be \
                 computed without a Lifting Line. Specify the time step or give a Lifting Line')
    LL.computeKinematicVelocity(tLL)
    LL.assembleAndProjectVelocities(tLL)

    if type(tL) == dict:
        Resolution = tL['Resolution']
        U0         = tL['VelocityFreestream']
    else:
        Particles  = getFreeParticles(tL)
        Resolution = I.getNodeFromName(Particles, 'Resolution')[1][0]
        U0         = I.getValue(I.getNodeFromName(Particles, 'VelocityFreestream'))

    Urelmax = 0.
    for LiftingLine in I.getZones(tLL):
        Ukin = np.vstack(J.getVars(LiftingLine, ['VelocityKinematic' + i for i in 'XYZ']))
        ui   = np.vstack(J.getVars(LiftingLine, ['VelocityInduced' + i for i in 'XYZ']))
        Urel = U0 + ui.T - Ukin.T
        Urel = max([np.linalg.norm(urel, axis = 0) for urel in Urel])
        if (Urelmax < Urel): Urelmax = Urel

    if Urelmax == 0:
        raise ValueError('Maximum velocity is zero. Set non-zero kinematic or freestream \
                                                                                     velocity.')

    if type(tL) == dict:
        tL['TimeStep'] = NumberParticlesShedAtTip*Resolution/Urel
    else:
        VPMParameters = getVPMParameters(tL)
        VPMParameters['TimeStep'] = NumberParticlesShedAtTip*Resolution/Urel

def setTimeStepFromBladeRotationAngle(tL = [], tLL = [], BladeRotationAngle = 5.):
    '''
    Sets the VPM TimeStep so that the fastest moving Lifting Line rotates by the user-given
    angle per TimeStep.
    .
    Parameters
    ----------
        tL : Tree
            Containes a zone of particles named 'FreeParticles'.

        tLL : Tree
            Lifting Lines.

        BladeRotationAngle : :py:class:`float`
            Blade rotation angle per TimeStep.
    '''
    if not tLL: tLL = getLiftingLinesTree(tL)
    if not tLL: raise AttributeError('The time step is not given and can not be \
                 computed without a Lifting Line. Specify the time step or give a Lifting Line')

    RPM = 0.
    for LiftingLine in I.getZones(tLL):
        RPM = max(RPM, I.getValue(I.getNodeFromName(LiftingLine, 'RPM')))
    
    if type(tL) == dict:
        tL['TimeStep'] = 1./6.*BladeRotationAngle/RPM
    else:
        VPMParameters = getVPMParameters(tL)
        VPMParameters['TimeStep'] = 1./6.*BladeRotationAngle/RPM

def setMinNbShedParticlesPerLiftingLine(tLL = [], Parameters = {}, NumberParticlesShedAtTip = 5):
    '''
    Sets the minimum number of shedding station on the Lifting Line(s) so that the fastest
    moving Lifting Line sheds the user-given number of particles at its tip at each TimeStep.
    .
    Parameters
    ----------
        tLL : Tree
            Lifting Lines.

        Parameters : :py:class:`dict`
            Parameters containing the VPM parameters.

        NumberParticlesShedAtTip : :py:class:`int`
            Blade rotation angle per TimeStep.
    '''
    LL.computeKinematicVelocity(tLL)
    LL.assembleAndProjectVelocities(tLL)
    Urelmax = 0.
    L = 0.
    flag = False
    if Parameters['VelocityFreestream']: flag = True
    else: U0 = Parameters['VelocityFreestream']
    for LiftingLine in I.getZones(tLL):
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

def getLiftingLineParameters(t = []):
    '''
    Gets the parameters regarding the Lifting Line(s).

    Parameters
    ----------
        t : Tree
            Containes a node of parameters named '.LiftingLine#Parameters'.

    Returns
    -------
        LiftingLineParameters : :py:class:`dict`
            Dictionnary of parameters. The parameters are pointers inside numpy ndarrays.
    '''
    return J.get(getFreeParticles(t), '.LiftingLine#Parameters')

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
            Containes a zone named 'HybridDomain'.
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
            Containes a zone named 'HybridDomain'.
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
            Containes a zone named 'HybridDomain'.
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
            return [z]

    #return []

def getHybridDomainInnerInterface(t = []):
    '''
    Gets the Inner Hybrid Domain Interface.

    Parameters
    ----------
        t : Tree
            Containes a zone named 'HybridDomain'.
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
            return [z]

    #return []

def getHybridDomainBEMInterface(t = []):
    '''
    Gets the Inner Hybrid Domain Interface.

    Parameters
    ----------
        t : Tree
            Containes a zone named 'HybridDomain'.
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
            return [z]

    #return []

def getHybridSources(t = []):
    '''
    Gets the Hybrid Hybrid Domain Interface.

    Parameters
    ----------
        t : Tree
            Containes a zone named 'HybridDomain'.
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
            return [z]

    #return []

def getEulerianTree(t = []):
    '''
    Gets the tree containing the node and cell-centers Eulerian bases.

    Parameters
    ----------
        t : Tree
            Containes the Eulerian Bases.
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
            Containes the Eulerian Bases.
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
            Containes the Hybrid parameters named '.Hybrid#Parameters'.

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
            Containes a zone of particles named 'FreeParticles'.

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
            U0 = getParameter(Particles, 'VelocityFreestream')
            C._initVars(Particles, 'U0X', U0[0])
            C._initVars(Particles, 'U0Y', U0[1])
            C._initVars(Particles, 'U0Z', U0[2])
            C._initVars(Particles, 'VelocityMagnitude=(\
          ({UX0}+{VelocityInducedX}+{VelocityPerturbationX}+{VelocityBEMX}+{VelocityInterfaceX})**2\
         +({UY0}+{VelocityInducedY}+{VelocityPerturbationY}+{VelocityBEMY}+{VelocityInterfaceY})**2\
         +({UZ0}+{VelocityInducedZ}+{VelocityPerturbationZ}+{VelocityBEMZ}+{VelocityInterfaceZ})**2\
                                                                                            )**0.5')
        elif ParticlesColorField == 'rotU':
            C._initVars(Particles, 'rotU=(({gradyVelocityZ} - {gradzVelocityY})**2 + \
                ({gradzVelocityX} - {gradxVelocityZ})**2 + ({gradxVelocityY} - {gradyVelocityX})**2\
                                                                                            )**0.5')
    CPlot._addRender2Zone(Particles, material = 'Sphere',
             color = 'Iso:' + ParticlesColorField, blending = 0.6, shaderParameters = [0.04, 0])
    LiftingLines = getLiftingLines(t)
    for zone in LiftingLines:
        CPlot._addRender2Zone(zone, material = 'Flat', color = 'White', blending = 0.2)

    if addLiftingLineSurfaces:
        if not AirfoilPolars:
            ERRMSG = J.FAIL + ('production of surfaces from lifting-line requires'
                ' attribute AirfoilPolars') + J.ENDC
            raise AttributeError(ERRMSG)
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
            Containes a zone of particles named 'FreeParticles'.

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

    sp = getVPMParameters(t)

    DisplayOptions['export'] = os.path.join(ImagesDirectory,
        'frame%05d.png'%sp['CurrentIteration'])

    if ShowInScreen:
        DisplayOptions['offscreen'] = 0
    else:
        DisplayOptions['offscreen'] = 1

    CPlot.display(t, **DisplayOptions)
    if DisplayOptions['offscreen']:
        CPlot.finalizeExport(DisplayOptions['offscreen'])

def load(filename = ''):
    '''
    Opens the CGNS file designated by the user. If the CGNS containes particles, the VPM field
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

def save(t = [], filename = '', VisualisationOptions = {}, SaveFields = checkSaveFields()):
    '''
    Saves the CGNS file designated by the user. If the CGNS containes particles, the VPM field
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
    tref = I.copyRef(t)
    if VisualisationOptions:
        setVisualization(tref, **VisualisationOptions)
        SaveFields = np.append(SaveFields, ['radius'])

    Particles = getFreeParticles(tref)
    # I.printTree(Particles)
    if I.getZones(Particles):
        I._rmNodesByName(Particles, 'BEMMatrix')
        if 'VelocityX' in SaveFields:
            u0 = np.array(getParameter(Particles, 'VelocityFreestream'), dtype = str)
            C._initVars(Particles, 'VelocityX='+u0[0]+'+{VelocityInducedX}+{VelocityPerturbationX}+\
                                                                              {VelocityDiffusionX}')
            C._initVars(Particles, 'VelocityY='+u0[1]+'+{VelocityInducedY}+{VelocityPerturbationY}+\
                                                                              {VelocityDiffusionY}')
            C._initVars(Particles, 'VelocityZ='+u0[2]+'+{VelocityInducedZ}+{VelocityPerturbationZ}+\
                                                                              {VelocityDiffusionZ}')

        if 'VelocityMagnitude' in SaveFields:
            u0 = np.array(getParameter(Particles, 'VelocityFreestream'), dtype = str)
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

def show(*msg):
    '''
    Overloads the print function and bypasses the global variable printBlocked.
    '''
    blocked = printBlocked[0]
    enablePrint()
    print(*msg)#, sep=', ')
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
