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
import numpy as np
import pprint

#import pickle

from . import Analysis as SA
from . import ShortCuts as SJ

from .. import InternalShortcuts as J
from .. import LiftingLine  as LL
from .. import Wireframe as W
from .. import GenerativeShapeDesign as GSD

import Converter.PyTree as C
import Converter.Internal as I
import Generator.PyTree as G
import Transform.PyTree as T


def buildBladeCGNS(BladeName,ParametersDict,path,PitchRotationDict):
   
    '''
    Create the geometry files (.tp and .cngs) associated to the input 
    blade geometry that is given as a parameter

    Parameters
    ----------

        BladeNumber: str
            To identify a specific blade
            
           
        ParametersDict : nested dict
            Contains the blade geometrical data

            .. important:: **ParametersDict** must be composed of:

                * Pitch0 : float
                
                * Rmin : float

                * Rmax: float

                * NPts: int

                * NPtsSpanwise : int

                * ChordDict : dict
                
                * TwistDict : dict
                
                * DihedralDict : dict

                * SweepDict : dict


        path : str
            General woking directory

        PitchRotation : bool
            If True, a rotation of the complete blade is computed

    Returns
    -------
        None : None
            **CGNS** file is saved
     '''
   
      
    #Getting every dictionary/variable
    Pitch0=ParametersDict['Pitch0']
    Rmin=ParametersDict['Rmin']
    Rmax=ParametersDict['Rmax']
    NPts=ParametersDict['NPts']
    NPtsSpanwise=ParametersDict['NPtsSpanwise']
    ChordDict=ParametersDict['ChordDict']
    TwistDict=ParametersDict['TwistDict']
    DihedralDict=ParametersDict['DihedralDict']
    SweepDict=ParametersDict['SweepDict']

    
    ############################# Begin of main program ###########################################

    #Profile section
    #NACA4412 = W.airfoil('NACA4412', ClosedTolerance=0)
    NACA4416 = W.airfoil('NACA4416', ClosedTolerance=0)

    T._oneovern(NACA4416, (4,1,1))


    AirfoilsDict = dict(RelativeSpan     = [  0.2,     1.000],
                        Airfoil          = [NACA4416,  NACA4416],
                        InterpolationLaw = 'interp1d_linear',)


    BladeDiscretization = np.linspace(Rmin,Rmax,NPtsSpanwise)
    NPtsTrailingEdge = ParametersDict['NPtsTrailingEdge'] # MUST BE ODD !!!
    IndexEdge = int((NPtsTrailingEdge+1)/2)
    Sections, WingWall,_=GSD.wing(BladeDiscretization,ChordRelRef=0.25,
                                NPtsTrailingEdge=NPtsTrailingEdge,
                                Airfoil=AirfoilsDict,
                                Chord=ChordDict,
                                Twist=TwistDict,
                                Dihedral=DihedralDict,
                                Sweep=SweepDict,)

    ClosedSections = []
    for s in Sections:
        ClosedSections.extend( GSD.closeAirfoil(s,Topology='ThickTE_simple',
            options=dict(NPtsUnion=NPtsTrailingEdge,
                        TFITriAbscissa=0.1,
                        TEdetectionAngleThreshold=30.)) )

    WingSolidStructured = G.stack(ClosedSections,None) # Structured
    _,Ni,Nj,Nk,_=I.getZoneDim(WingSolidStructured)



    # Spanwise towards X positive:

    WingSolidStructured = T.rotate(WingSolidStructured, (0.,0.,0.),(0.,1.,0.),-90.)
    WingSolidStructured = T.rotate(WingSolidStructured, (0.,0.,0.),(1.,0.,0.),90.)


    C._addBC2Zone(WingSolidStructured,
                'Node_Encastrement',
                'FamilySpecified:Noeud_Encastrement',
                'kmin')


    C._addBC2Zone(WingSolidStructured,
                'LeadingEdge',
                'FamilySpecified:LeadingEdge',
                wrange=[IndexEdge,IndexEdge,Nj,Nj,1,Nk])

    C._addBC2Zone(WingSolidStructured,
                'TrailingEdge',
                'FamilySpecified:TrailingEdge',
                wrange=[IndexEdge,IndexEdge,1,1,1,Nk])

    #C._fillEmptyBCWith(WingSolidStructured,
    #                   'External_Forces',
    #                   'FamilySpecified:External_Forces')

    # Creating saving folder
    if not os.path.exists(path + '/InputData/Geometry/BladeCGNS'):
     os.makedirs(path + '/InputData/Geometry/BladeCGNS')
    os.chdir(path+'/InputData/Geometry/BladeCGNS')

    
    t = C.newPyTree(['SOLID',WingSolidStructured])

    #Pitch rotation (The problem associated to local pitch section rotation is avoided)
    if PitchRotationDict['PitchRotation']== True:
        t = T.rotate(t,PitchRotationDict['PitchPoint'],PitchRotationDict['PitchAxis'], PitchRotationDict['PitchAngle'])



    #Saving the .cgns and .tp files
    C.convertPyTree2File(t,'%s.cgns'%BladeName,'bin_adf')
    C.convertPyTree2File(t,'%s.tp'%BladeName)



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
    DictSimulationParameters  = J.get(t, '.SimulationParameters')

    # Now we have everything ready for construction of the 
    # LiftingLine object:
    
    DictAerodynamicProperties['BladeParameters']['BladeDiscretization']['P1'] = SJ.totuple(DictAerodynamicProperties['BladeParameters']['BladeDiscretization']['P1'])
    DictAerodynamicProperties['BladeParameters']['BladeDiscretization']['P2'] = SJ.totuple(DictAerodynamicProperties['BladeParameters']['BladeDiscretization']['P2'])
    
    DictAerodynamicProperties['BladeParameters']['PolarsDict']['PyZonePolarNames'] = DictAerodynamicProperties['BladeParameters']['PolarsDict']['PyZonePolarNames'].split(' ') 
    LiftingLine = LL.buildLiftingLine(DictAerodynamicProperties['BladeParameters']['BladeDiscretization'],
                                      Airfoils=DictAerodynamicProperties['BladeParameters']['PolarsDict'],
                                      Chord =DictAerodynamicProperties['BladeParameters']['ChordDict'],
                                      Twist =DictAerodynamicProperties['BladeParameters']['TwistDict'],
                                      Dihedral=DictAerodynamicProperties['BladeParameters']['DihedralDict'],
                                      Sweep=DictAerodynamicProperties['BladeParameters']['SweepDict']
                                      )
    
    # [ Beware that LiftingLine is a 1D PyTree Zone ! ] We can, for example,
    # give it a name, like this:
    LiftingLine[0] = 'LiftingLine'
    #LL.resetPitch(LiftingLine, ZeroPitchRelativeSpan=0.75)

    print('PolarsType== '+DictAerodynamicProperties['BladeParameters']['Polars']['PolarsType'])

    if DictAerodynamicProperties['BladeParameters']['Polars']['PolarsType'] == 'HOST':
        PyZonePolars = [LL.convertHOSTPolarFile2PyZonePolar(os.getcwd()+'/InputData/'+fn) for fn in DictAerodynamicProperties['BladeParameters']['Polars']['filenames']]
    elif DictAerodynamicProperties['BladeParameters']['Polars']['PolarsType'] == 'cgns':
        PyZonePolars = C.convertFile2PyTree(os.getcwd()+'/InputData/POLARS/'+DictAerodynamicProperties['BladeParameters']['Polars']['filenames'])
        PyZonePolars = I.getZones(PyZonePolars)
    else:
        print(FAIL +'ERROR: PyZonePolars not defined!')
        stop


    PolarsInterpFuns = LL.buildPolarsInterpolatorDict(PyZonePolars,
                                            InterpFields=['Cl', 'Cd', 'Cm'])

    
    for fn in ['AoA','phiRad','Cl','Cd','Cm','VelocityMagnitudeLocal']:
        C._initVars(LiftingLine,fn,0.)
    LL.setConditions(LiftingLine)
    #LL.setKinematicsUsingConstantRPM(LiftingLine)

    LL.setKinematicsUsingConstantRotationAndTranslation(LiftingLine,
                                      RotationCenter=DictSimulationParameters['RotatingProperties']['RotationCenter'],
                                      RotationAxis=DictSimulationParameters['RotatingProperties']['AxeRotation'],
                                      RPM=0.0,
                                      RightHandRuleRotation=DictSimulationParameters['RotatingProperties']['RightHandRuleRotation'])

    LL.computeGeneralLoadsOfLiftingLine(LiftingLine)

    I.addChild(t, LiftingLine)

    print(CYAN + 'LL Initialised...'+ENDC)

    return t, PolarsInterpFuns