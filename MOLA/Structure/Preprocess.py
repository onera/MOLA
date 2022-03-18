'''
MOLA - StructuralPreprocess.py 

This module defines some handy shortcuts of the Cassiopee's
Converter.Internal module.

First creation:
31/05/2021 - M. Balmaseda
'''


FAIL  = '\033[91m'
GREEN = '\033[92m' 
WARN  = '\033[93m'
MAGE  = '\033[95m'
CYAN  = '\033[96m'
ENDC  = '\033[0m'

# System modules
import os
from . import Analysis as SA
from . import ShortCuts as SJ

from .. import InternalShortcuts as J
from .. import LiftingLine  as LL

import Converter.PyTree as C
import Converter.Internal as I



def PreprocessCGNS4StructuralParameters(t, DictMaterialProperties, DictROMProperties, DictMeshProperties):
    ''' Sets the structural parameters Node from the input dictionaries'''

    # Set the Structural Parameters node:
    
    J.set(t,'.StructuralParameters', **dict(MaterialProperties = DictMaterialProperties, 
                                            ROMProperties = DictROMProperties, 
                                            MeshProperties = DictMeshProperties,
                                            )
          )

def PreprocessCGNS4SimulationParameters(t, DictLoadingProperties, DictIntegrationProperties, DictRotatingParameters):
    ''' Sets the structural parameters Node from the input dictionaries'''

    # Set the Structural Parameters node:
    
    J.set(t,'.SimulationParameters', **dict(LoadingProperties = DictLoadingProperties, 
                                            IntegrationProperties = DictIntegrationProperties, 
                                            RotatingProperties = DictRotatingParameters,
                                            )
          )

def PreprocessCGNS4BEMTAelCoupling(t,DictBladeParameters, DictFlightConditions, DictBEMTParameters):
    ''' Sets the aerodynamic parameters Node from the input dictionaries'''

    # Set the Structural Parameters node:
    
    J.set(t,'.AerodynamicProperties', **dict(BladeParameters = DictBladeParameters, 
                                             FlightConditions = DictFlightConditions, 
                                             BEMTParameters = DictBEMTParameters,
                                            )
          )


def InitialLiftinfLine(t):

    print(GREEN+'Evaluate the initial LL'+ENDC)
    DictAerodynamicProperties = J.get(t, '.AerodynamicProperties')

    # Now we have everything ready for construction of the 
    # LiftingLine object:
    
    DictAerodynamicProperties['BladeParameters']['BladeDiscretization']['P1'] = SJ.totuple(DictAerodynamicProperties['BladeParameters']['BladeDiscretization']['P1'])
    DictAerodynamicProperties['BladeParameters']['BladeDiscretization']['P2'] = SJ.totuple(DictAerodynamicProperties['BladeParameters']['BladeDiscretization']['P2'])
    
    LiftingLine = LL.buildLiftingLine(DictAerodynamicProperties['BladeParameters']['BladeDiscretization'],
                                      Polars=DictAerodynamicProperties['BladeParameters']['PolarsDict'],
                                      Chord =DictAerodynamicProperties['BladeParameters']['ChordDict'],
                                      Twist =DictAerodynamicProperties['BladeParameters']['TwistDict'],
                                      Dihedral=DictAerodynamicProperties['BladeParameters']['DihedralDict'],
                                      Sweep=DictAerodynamicProperties['BladeParameters']['SweepDict']
                                      )
    
    # [ Beware that LiftingLine is a 1D PyTree Zone ! ] We can, for example,
    # give it a name, like this:
    LiftingLine[0] = 'LiftingLine'
    #LL.resetPitch(LiftingLine, ZeroPitchRelativeSpan=0.75)
    if DictAerodynamicProperties['BladeParameters']['Polars']['PolarsType'] == 'HOST':
        PyZonePolars = [LL.convertHOSTPolarFile2PyZonePolar(os.getcwd()+'/InputData/'+fn) for fn in DictAerodynamicProperties['BladeParameters']['Polars']['filenames']]
    elif DictAerodynamicProperties['BladeParameters']['Polars']['PolarsType'][0] == 'cgns':
        PyZonePolars = C.convertFile2PyTree(os.getcwd()+'/InputData/POLARS/'+DictAerodynamicProperties['BladeParameters']['Polars']['filenames'][0])
        PyZonePolars = I.getZones(PyZonePolars)
    else:
        print(FAIL +'ERROR: PyZonePolars not defined!')
        XXX


    PolarsInterpFuns = LL.buildPolarsInterpolatorDict(PyZonePolars,
                                            InterpFields=['Cl', 'Cd', 'Cm'])

    
    for fn in ['AoA','phiRad','Cl','Cd','Cm','VelocityMagnitudeLocal']:
        C._initVars(LiftingLine,fn,0.)
    LL.setConditions(LiftingLine)
    LL.setKinematicsUsingConstantRPM(LiftingLine)
    LL.computeGeneralLoadsOfLiftingLine(LiftingLine)

    I.addChild(t, LiftingLine)

    print(CYAN + 'LL Initialised...'+ENDC)

    return t, PolarsInterpFuns