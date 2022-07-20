'''
prepareRestart.py template for use with ORAS WORKFLOW.

Produces the case main CGNS file (main.cgns)

MOLA 1.14 - 19/07/2022 - M. Balmaseda - creation
'''

import sys, os
import numpy as np
from timeit import default_timer as tic

import Converter.PyTree as C
import Converter.Internal as I


import MOLA.WorkflowORAS as WO

toc = tic() # auxiliary variable used to log the script execution time

RPM = 850.

WO.prepareMainCGNS4ElsA('mesh.cgns',
    ReferenceValuesParams=dict(
        Density= 0.379597,       # Cruise: 0.379597       // TakeOff: 1.16439
        Temperature= 218.808,    # Cruise: 218.808        // TakeOff: 303.150
        Velocity= 0.75*296.535,  # Cruise: 0.75*296.535   // TakeOff: 0.25*349.039
        AngleOfAttackDeg=0.0,
        
        TurbulenceModel='Wilcox2006-klim',
        
        FieldsAdditionalExtractions=['q_criterion', 'Mach', 'Pressure'],
        CoprocessOptions=dict(
            RequestedStatistics=['std-MomentumXFlux','avg-MomentumXFlux'],

            ConvergenceCriteria = [dict(Family    = 'Rotor_Blade',
                                        Variable  = 'std-MomentumXFlux',
                                        Threshold = 1.,),
                                   dict(Family    = 'Stator_Blade',
                                        Variable  = 'std-MomentumXFlux',
                                        Threshold = 1.,),
                                   ],
            
            AveragingIterations = 1000,
            ItersMinEvenIfConverged = 1000,

            BodyForceInitialIteration = 1,
            BodyForceComputeFrequency = 50,
            BodyForceSaveFrequency    = 100,

            UpdateArraysFrequency     = 100,
            UpdateSurfacesFrequency   = 500,
            UpdateFieldsFrequency     = 2000,
            ),),

    NumericalParams = dict(NumericalScheme='jameson',
                           CFLparams=dict(vali=1.,valf=10.,iteri=1,iterf=300,function_type='linear'),
                           ),

    TurboConfiguration=dict(
                    # Shaft speed in rad/s
                    # BEWARE: only for single shaft configuration
                    ShaftRotationSpeed = RPM * np.pi / 30.,
                
                    # Hub rotation speed
                    # List of tuples. Each tuple (xmin, xmax) corresponds to a CoordinateX
                    # interval where the speed at hub wall is ShaftRotationSpeed. It is zero
                    # outsides these intervals.
                    HubRotationSpeed = [(-1e20, -0.43)],
                
                    # This dictionary has one entry for each row domain.
                    # The key names must be the family names in the CGNS Tree.
                    Rows = dict(
                        Rotor  = dict(RotationSpeed = RPM * np.pi / 30.,
                                     NumberOfBlades = 12,
                                     ),
                        Stator = dict(RotationSpeed = 0,
                                      NumberOfBlades = 10,
                                      ),
                        )
                    ),
    
    Extractions=[
        dict(type='BCWall'),
        dict(type='IsoSurface',
             field='q_criterion',
             value=5000),
        dict(type='IsoSurface',
             field='q_criterion',
             value=50000),
        dict(type='IsoSurface',
             field='Mach',
             value=1.),
        dict(type='IsoSurface',
             field='Mach',
             value=1.2),
        ],

    BoundaryConditions= [dict(type='Farfield', FamilyName='Farfield'),
                         dict(type='MixingPlane', left='MixingPlaneUpstream', right='MixingPlaneDownstream'),
                         dict(type='WallInviscid', FamilyName = 'Rotor_Hub' ),
                         ],

    writeOutputFields=True,)

print('Elaped time: %g minutes'%((tic()-toc)/60.))
