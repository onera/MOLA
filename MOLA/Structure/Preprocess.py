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


def buildBladeCGNS(BladeName,ParametersDict):
   
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



    Returns
    -------
        None : None
            **CGNS** file is saved
     '''
   
      
    #Getting every dictionary
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


    # MUST BE ODD !!!NPtsSpanwise
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

    return t

def BladeDictBuilder(GeoId,ParametersDict,path):
    
    '''
    Create a dictionary associated to the input blade geometry
    that is given as a parameter
    """


    Parameters
    ----------

        GeoId : 4 digits long str 
            must contain 4 digits (identifiers) used to indicate the Chord law, Pitch law, Twist law, Sweep law
            
           
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

                * PolarsType: str

                * filesname: array of str


        path : str
            general woking directory



    Returns
    -------
        None : None
            **Dict** is saved
     '''
    
    
    #Getting every dictionary
    Pitch0=ParametersDict['Pitch0']
    Rmin=ParametersDict['Rmin']
    Rmax=ParametersDict['Rmax']
    NPts=ParametersDict['NPts']
    NPtsSpanwise=ParametersDict['NPtsSpanwise']
    ChordDict=ParametersDict['GeomLaws']['ChordDict']
    TwistDict=ParametersDict['GeomLaws']['TwistDict']
    DihedralDict=ParametersDict['GeomLaws']['DihedralDict']
    SweepDict=ParametersDict['GeomLaws']['SweepDict']

    #Composing the blade name (unique for a given geometry)
    BladeName = 'Blade_'+str(int(Rmin*100))+'cm_'+str(int(Rmax*100))+\
    'cm_pitch'+str(int(Pitch0))+'_ChLaw'+ GeoId[0]+\
    '_PLaw'+GeoId[1]+'_TLaw'+GeoId[2]+'_SLaw'+GeoId[3]+'_NPts'+str(int(NPts))


    ############################# Begin of main program ###########################################

    AirfoilName = 'NACA4416'
    AIRFOIL = W.airfoil('NACA4416', ClosedTolerance=0)
    AirfoilsDict = dict(RelativeSpan     = [  Rmin/Rmax,     1.000],
                        Airfoil          = [AIRFOIL,  AIRFOIL],
                        InterpolationLaw = 'interp1d_linear',)

    PolarsDict = dict(RelativeSpan     = [  Rmin/Rmax,  1.000],
                    PyZonePolarNames = [AirfoilName,AirfoilName],
                    InterpolationLaw = 'interp1d_linear',)

    RootSegmentLength = 0.0500 * Rmax
    TipSegmentLength  = 0.0016 * Rmax

    # We list the HOST files in absolute path:
    PolarsType = ParametersDict['Polars']['PolarsType'] # 'cgns', # HOST/cgns
    filenames = ParametersDict['Polars']['filenames'] #['PolarsCorrected.cgns',
    #'POLARS/HOST_Profil_OH312', 
    #'POLARS/HOST_Profil_OH310',
    #'POLARS/HOST_Profil_OH309',
    #]

    #Dictionary composing
    DictBladeParameters = dict(ChordDict = ChordDict,     
                            TwistDict = TwistDict,       
                            DihedralDict = DihedralDict,  
                            PolarsDict = PolarsDict,  
                            SweepDict = SweepDict,   
                            
                            NBlades=ParametersDict['NBlades'], 
                            Constraint='Pitch', 
                            ConstraintValue=0.0,
                            Rmin = Rmin,
                            Rmax = Rmax,
                            NPts = NPts,   
                            BladeDiscretization = dict(P1=[Rmin,0,0],P2=[Rmax,0,0],
                                                    N=NPts, kind='tanhTwoSides',
                                                    FirstCellHeight=RootSegmentLength,
                                                    LastCellHeight=TipSegmentLength),
                            Polars = dict(PolarsType = PolarsType,
                                            filenames = filenames),
                            Pitch0 = Pitch0,
                            )

 
    if not os.path.exists(path + '/InputData/Geometry/BladeDicts'):
        os.makedirs(path + '/InputData/Geometry/BladeDicts')
    #os.chdir(path+'/InputData/Geometry/BladeDicts')


    Lines = ['#!/usr/bin/python\n']
    Lines = ['from numpy import array\n']
    Lines+= ['DictBladeParameters = '+pprint.pformat(DictBladeParameters)+"\n"]

    AllLines = '\n'.join(Lines)


    with open(path + '/InputData/Geometry/BladeDicts/%s.py'%BladeName, "w") as a_file:
        #pickle.dump(DictBladeParameters, a_file)
        a_file.write(AllLines)
        a_file.close()

    
    

    print('Writing ./InputData/Geometry/BladeDicts/%s.py'%BladeName)



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
    print(DictAerodynamicProperties['BladeParameters']['Polars']['PolarsType'][0] == 'cgns')
    print(DictAerodynamicProperties['BladeParameters']['Polars']['PolarsType'][0])

    if DictAerodynamicProperties['BladeParameters']['Polars']['PolarsType'] == 'HOST':
        PyZonePolars = [LL.convertHOSTPolarFile2PyZonePolar(os.getcwd()+'/InputData/'+fn) for fn in DictAerodynamicProperties['BladeParameters']['Polars']['filenames']]
    elif DictAerodynamicProperties['BladeParameters']['Polars']['PolarsType'] == 'cgns':
        PyZonePolars = C.convertFile2PyTree(os.getcwd()+'/InputData/POLARS/'+DictAerodynamicProperties['BladeParameters']['Polars']['filenames'])
        PyZonePolars = I.getZones(PyZonePolars)
    else:
        print(FAIL +'ERROR: PyZonePolars not defined!')
        XXX


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