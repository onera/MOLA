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

import pickle
import MOLA

try:
	from . import Analysis as SA
	from . import ShortCuts as SJ
	
	
	from .. import InternalShortcuts as J
	from .. import LiftingLine  as LL
	from .. import Wireframe as W
	from .. import GenerativeShapeDesign as GSD
	from .. import PropellerAnalysis as PA
	from .. import Data as M
	from ..Data import BEMT 
	from ..Data.LiftingLine import LiftingLine
	
	import Converter.PyTree as C
	import Converter.Internal as I
	import Generator.PyTree as G
	import Transform.PyTree as T
except:
	print(WARN+'MOLA modules not loaded!!'+ENDC)

import matplotlib.pyplot as plt

def AddToBladesDict(NewBlade,NewBladeNumber, BladesDict):
    '''
    Add a new blade geometry to the big dictionary of blades, called "BladesBible.py". 
    Firstly, it makes sure this geometry hasn't been added yet to inform the user that
    the blade information will be modified on the dictionary.
    """

    Parameters
    ----------

        NewBladeNumber: int
            Number to identify the blade geometry
           
        NewBlade : dict
            Contains the geometric information of a new blade to add or an existing blade
            to be modified
        
        BladesDict: dict
            Contains the BladeDictionary
    Returns
    -------
        None : None
            **BladesDict** is saved
            
    '''
    #Add a new key to the big blades dictionary
    BladesDict['Blade%s'%NewBladeNumber]=NewBlade
    Lines = ['#!/usr/bin/python\n']
    Lines = ['from numpy import array\n']
    Lines+= ['BladesBible = '+pprint.pformat(BladesDict)+"\n"]
    AllLines = '\n'.join(Lines)
    with open('INPUT/Geometry/BladesBible.py', "w") as a_file:
        a_file.write(AllLines)
        a_file.close()


def getSpanDistribution(**kwargs):

    # Constant distribution:
    if kwargs['SpanDiscretisationLaw'] == 'Constant':
        SpanDiscretisation = np.linspace(0.,1., kwargs['NPtsSpanwise'])

    elif kwargs['SpanDiscretisationLaw'] == 'Tanh2Sides':

        BladeDiscretizationStructure = dict(P1=[0,0,0],P2=[0.99,0,0],
                                       N=kwargs['NPtsSpanwise'], kind='tanhTwoSides',
                                       FirstCellHeight=kwargs['RootHeight'],
                                       LastCellHeight=kwargs['TipHeight'])
        
        SpanDiscretisation,s,_ = J.getDistributionFromHeterogeneousInput__(BladeDiscretizationStructure)

    elif kwargs['SpanDiscretisationLaw'] == 'ConstantRate':

        n = np.arange(0,kwargs['NPtsSpanwise'] - 1)
        factors = kwargs['Rate']**n
        initialStep = (1.-0.)/np.sum(factors)

        SpanDiscretisation = [np.sum(initialStep * factors[:Element]) for Element in range(kwargs['NPtsSpanwise'])]

    elif kwargs['SpanDiscretisationLaw'] == 'MixedConstantRootTanhTip':
        # TODO!!!! Impose a tanh on the tip
        # Arguments: RateRoot, TransitionElementsFromTip, NPtsSpanwise, TipHeight
        nRoot = np.arange(0,kwargs['NPtsSpanwise']-kwargs['TransitionElementsFromTip'] - 1)
        factorsRoot = kwargs['RateRoot']**nRoot
        initialStepRoot = (1-(kwargs['TransitionElementsFromTip']/kwargs['NPtsSpanwise']))/np.sum(factorsRoot)

        RootBladeDiscretization = [np.sum(initialStepRoot * factorsRoot[:Element]) for Element in range(kwargs['NPtsSpanwise']-kwargs['TransitionElementsFromTip'])]

        BladeDiscretizationStructure = dict(P1=[RootBladeDiscretization[-1],0,0],P2=[1.,0,0],
                                       N=kwargs['TransitionElementsFromTip'], kind='tanhTwoSides',
                                       FirstCellHeight=initialStepRoot * factorsRoot[-1],
                                       LastCellHeight=kwargs['TipHeight'])
        
        TipBladeDiscretization,s,_ = J.getDistributionFromHeterogeneousInput__(BladeDiscretizationStructure)
        SpanDiscretisation = RootBladeDiscretization[:-1] + list(TipBladeDiscretization)
        


    elif kwargs['SpanDiscretisationLaw'] == 'TwoDifferentConstantRates':
        # Arguments: RateRoot, RateTip, TransitionElementsFromTip

        nRoot = np.arange(0,kwargs['NPtsSpanwise'] - 1)
        factorsRoot = kwargs['RateRoot']**nRoot
        initialStepRoot = (1.-0.)/np.sum(factorsRoot)

        RootBladeDiscretization = [np.sum(initialStepRoot * factorsRoot[:Element]) for Element in range(kwargs['NPtsSpanwise'])][:-kwargs['TransitionElementsFromTip']]

        nTip = np.arange(0,kwargs['TransitionElementsFromTip']-1)
        factorsTip = kwargs['RateTip']**nTip
        initialStepTip = (1.-RootBladeDiscretization[-1])/np.sum(factorsTip)

        InitialBladeDiscretizationTip = [RootBladeDiscretization[-1] + np.sum(initialStepTip * factorsTip[:Element]) for Element in range(kwargs['TransitionElementsFromTip']+1)]

        SpanDiscretisation = RootBladeDiscretization[:-1] + InitialBladeDiscretizationTip

    return SpanDiscretisation

def buildCSMmeshFromCFDsurfaceMesh(t, **kwargs):

    zones = I.getZones(t)
    NumberOfSections = I.getZoneDim( zones[0] )[1]

    SpanDiscretisation = getSpanDistribution(**kwargs)
    

    ScannedBlade = GSD.scanBlade(t, SpanDiscretisation, [0., 0., 0.], [1., 0. , 0.], [0., 0., 1.])
    Sections = I.getZones(I.getNodeFromName(ScannedBlade, 'Sections'))
    
    
    for s in Sections:
        if C.getNPts(s)%2 != 0:
            SectionDistribution = W.copyDistribution(s)
            break
    ClosedSections = []
    closedCurves = []
    for s in I.getZones(Sections):
    
        Curve = W.discretize(s, Distribution=SectionDistribution)
        closed = GSD.closeAirfoil(Curve, Topology='ThickTE_simple',
            options=dict(NPtsUnion=kwargs['NPtsTrailingEdge']+2,TFITriAbscissa=0.1,TEdetectionAngleThreshold=None))
        T._reorder(closed,(-1,2,3))
        ClosedSections.extend(I.getZones(closed))
    
    RootAndTipSections = []
    for i in [0,NumberOfSections-1]:
        curves = []
        for zone in zones:
            x, y, z = J.getxyz( zone )
            x = x[i,:]
            y = y[i,:]
            z = z[i,:]
            curves += [ J.createZone( 'curve', [x, y, z], ['x','y','z']) ]
        contour = W.joinSequentially(curves, reorder=True, sort=True)
        contour = W.discretize(contour, Distribution=SectionDistribution)
        closed = GSD.closeAirfoil(contour, Topology='ThickTE_simple',
            options=dict(NPtsUnion=kwargs['NPtsTrailingEdge']+2,TFITriAbscissa=0.1,TEdetectionAngleThreshold=None))
        T._reorder(closed,(-1,2,3))
        RootAndTipSections.extend( closed )
    
    ClosedSections = [RootAndTipSections[0]] + ClosedSections + [RootAndTipSections[-1]]
    
    #C.convertPyTree2File(ClosedSections, 'sections.cgns')
    
    WingSolidStructured = G.stack(ClosedSections)
    #C.convertPyTree2File(WingSolidStructured, 'solid.cgns')
    
    
    # Spanwise towards X positive:
    WingSolidStructured = T.rotate(WingSolidStructured, (0.,0.,0.),(0.,1.,0.),-90.)
    WingSolidStructured = T.rotate(WingSolidStructured, (0.,0.,0.),(1.,0.,0.),90.)
    IndexEdge = int((kwargs['NPtsTrailingEdge']+1)/2)
    C._addBC2Zone(WingSolidStructured,
                'Node_Encastrement',
                'FamilySpecified:Noeud_Encastrement',
                'kmin')
    C._addBC2Zone(WingSolidStructured,
                'LeadingEdge',
                'FamilySpecified:LeadingEdge',
                'jmin')
    C._addBC2Zone(WingSolidStructured,
                'TrailingEdge',
                'FamilySpecified:TrailingEdge',
                'jmax')
    
    return WingSolidStructured, Sections

def buildAirfoilDictFromPolars(polarName, ClosedProfile = False, ModifyPolars = False, DictOfPolyLines = None):
    Polars = C.convertFile2PyTree(polarName)
    AirfoilDict = {}
    for zone in I.getZones(Polars):
        FoilGeom = J.get(zone, '.Polar#FoilGeometry')
        
        if ClosedProfile:
            AirfoilDict[zone[0]] = J.createZone(zone[0], [FoilGeom['CoordinateX'][:-1], FoilGeom['CoordinateY'][:-1],np.zeros(len(FoilGeom['CoordinateX'][:-1]))], ['CoordinateX', 'CoordinateY', 'CoordinateZ'])
        else:
            AirfoilDict[zone[0]] = J.createZone(zone[0], [FoilGeom['CoordinateX'], FoilGeom['CoordinateY'][::4], np.zeros(len(FoilGeom['CoordinateX'], ))], ['CoordinateX', 'CoordinateY', 'CoordinateZ'])

        if ModifyPolars:
            if ClosedProfile:
                FoilGeom['CoordinateX'],FoilGeom['CoordinateY'] = J.getxy(AirfoilDict[zone[0]])  # FoilGeom['CoordinateX'][1:-1], FoilGeom['CoordinateY'][1:-1]
            
            J.set(zone, '.Polar#FoilGeometry', **FoilGeom)

    if ModifyPolars:
        J.save(Polars, 'PolarsOTE.cgns')          

    return AirfoilDict


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
    
    Rmin=ParametersDict['Rmin']
    Rmax=ParametersDict['Rmax']
    #NPts=ParametersDict['NPts']
    #NPtsSpanwise=ParametersDict['NPtsSpanwise']

    ChordDict=ParametersDict['GeomLaws']['ChordDict']
    TwistDict=ParametersDict['GeomLaws']['TwistDict']
    DihedralDict=ParametersDict['GeomLaws']['DihedralDict']
    SweepDict=ParametersDict['GeomLaws']['SweepDict']
    AirfoilsDict = ParametersDict['GeomLaws']['Airfoil']
    
    ############################# Begin of main program ###########################################  

    SpanDiscretisation = getSpanDistribution(**ParametersDict['SpanDiscretisation'])
    BladeDiscretization = Rmin + (Rmax - Rmin) * np.array(SpanDiscretisation)
    #print(SpanDiscretisation)
    #print(BladeDiscretization)

    NPtsTrailingEdge = ParametersDict['NPtsTrailingEdge'] # MUST BE ODD !!!
    IndexEdge = int((NPtsTrailingEdge+1)/2)

    Sections, WingWall,_=GSD.wing(BladeDiscretization,ChordRelRef=0.25,
                                NPtsTrailingEdge=NPtsTrailingEdge,
                                Airfoil=AirfoilsDict,
                                Chord=ChordDict,
                                Twist=TwistDict,
                                Dihedral=DihedralDict,
                                Sweep=SweepDict)
    
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

    
    
    t = C.newPyTree(['SOLID',WingSolidStructured])
    #J.save(t,'InitialTree.cgns')
    #Pitch rotation (The problem associated to local pitch section rotation is avoided)
    #if ParametersDict['Pitch']['PitchAngle'] != 0.:
    #    t = T.rotate(t,ParametersDict['Pitch']['PitchCenter'],ParametersDict['Pitch']['PitchAxis'], ParametersDict['Pitch']['PitchAngle'])

    
    return t



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

def PreprocessCGNS4BEMTAelCoupling(t,DictBladeParameters = {}, DictFlightConditions = {}, DictBEMTParameters = {}):
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
    
    DictAerodynamicProperties['BladeParameters']['BladeLLDiscretization']['P1'] = SJ.totuple(DictAerodynamicProperties['BladeParameters']['BladeLLDiscretization']['P1'])
    DictAerodynamicProperties['BladeParameters']['BladeLLDiscretization']['P2'] = SJ.totuple(DictAerodynamicProperties['BladeParameters']['BladeLLDiscretization']['P2'])
    
    DictAerodynamicProperties['BladeParameters']['PolarsDict']['PyZonePolarNames'] = DictAerodynamicProperties['BladeParameters']['Polars']['PyZonePolarNames'].split(' ') 
    LiftingLine = LL.buildLiftingLine(DictAerodynamicProperties['BladeParameters']['BladeLLDiscretization'],
                                      Airfoils=DictAerodynamicProperties['BladeParameters']['GeomLaws']['PolarsDict'],
                                      Chord =DictAerodynamicProperties['BladeParameters']['GeomLaws']['ChordDict'],
                                      Twist =DictAerodynamicProperties['BladeParameters']['GeomLaws']['TwistDict'],
                                      Dihedral=DictAerodynamicProperties['BladeParameters']['GeomLaws']['DihedralDict'],
                                      Sweep=DictAerodynamicProperties['BladeParameters']['GeomLaws']['SweepDict']
                                      )
    
    # [ Beware that LiftingLine is a 1D PyTree Zone ! ] We can, for example,
    # give it a name, like this:
    LiftingLine[0] = 'LiftingLine'
    #LL.resetPitch(LiftingLine, ZeroPitchRelativeSpan=0.75)

    print('PolarsType== '+DictAerodynamicProperties['BladeParameters']['GeomLaws']['Polars']['PolarsType'])

    if DictAerodynamicProperties['BladeParameters']['Polars']['PolarsType'] == 'HOST':
        PyZonePolars = [LL.convertHOSTPolarFile2PyZonePolar(os.getcwd()+'/INPUT/'+fn) for fn in DictAerodynamicProperties['BladeParameters']['Polars']['filenames']]
    elif DictAerodynamicProperties['BladeParameters']['Polars']['PolarsType'] == 'cgns':
        PyZonePolars = C.convertFile2PyTree(os.getcwd()+'/INPUT/POLARS/'+DictAerodynamicProperties['BladeParameters']['Polars']['filenames'])
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




def adaptCommAster(DictAuxiliarInformation, commName):  
    fw = open('%s.comm'%commName, 'w')
    with open('./INPUT/Templates/ComputeMODELS.comm', 'r') as f0:
        for line in f0:
            line = line.replace("PATH2REPLACE", DictAuxiliarInformation['path'])
            #line = line.replace("SETUP2REPLACE","%s.py"%DictAuxiliarInformation['SetupName'])
            fw.write(line)
    fw.close()

def adaptLaunchAster(DictAuxiliarInformation ,commName, typeAsterLaunch):

    fw = open('LaunchAster%s'%(typeAsterLaunch), 'w')

    with open('./INPUT/Templates/LaunchAster', 'r') as file:
        filedata = file.read()
        #file.close()
        print('%s/%s.comm'%(DictAuxiliarInformation['path'], commName))
        filedata = filedata.replace("'COMM2REPLACE'", "%s/%s.comm"%(DictAuxiliarInformation['path'], commName))
        filedata = filedata.replace("'PATH2REPLACE'", '%s'%DictAuxiliarInformation['pathAster'])
    fw.write(filedata)
    fw.close()
    
def prepareFiles4AsterFromTemplate(DictAuxiliarInformation, typeAsterLaunch = 'Models', commName = None ):
    
    if typeAsterLaunch in ['Models']:
        if commName == None:
            commName = '4_ComputeMODELS'
        
        adaptCommAster(DictAuxiliarInformation, commName)
        adaptLaunchAster(DictAuxiliarInformation ,commName, typeAsterLaunch)



def computeBladeOptimizationMIL(ParametersDict, AoAImposed = None):
    RootSegmentLength = 0.0500 * ParametersDict['Rmax']
    TipSegmentLength  = 0.0016 * ParametersDict['Rmax']
    BladeDiscretization = dict(P1=(ParametersDict['Rmin'],0,0),P2=(ParametersDict['Rmax'],0,0),N=ParametersDict['NPts'], kind='tanhTwoSides',FirstCellHeight=RootSegmentLength,LastCellHeight=TipSegmentLength)
    LiftingLine = LL.buildLiftingLine(BladeDiscretization,
        Airfoils=ParametersDict['GeomLaws']['PolarsDict'],
        Chord =ParametersDict['GeomLaws']['ChordDict'],
        Twist =ParametersDict['GeomLaws']['TwistDict'], 
        Sweep =ParametersDict['GeomLaws']['SweepDict'], 
        Dihedral=ParametersDict['GeomLaws']['DihedralDict'])
    
    PyZonePolars = J.load(ParametersDict['PolarName'])
    PolarsInterpFuns = LL.buildPolarsInterpolatorDict(PyZonePolars)
    
    print ('Launching (2.a) design function...')
    ResultsDict = PA.designPropellerAdkins(
    LiftingLine, # Initial guess -> It will be modified !
    PolarsInterpFuns,
    # These design parameters were defined in Step 0
    NBlades=ParametersDict['NBlades'],Velocity=ParametersDict['Velocity'],RPM=ParametersDict['RPM'],Temperature=ParametersDict['Temperature'],Density=ParametersDict['Density'],
    Constraint=ParametersDict['Constraint'],ConstraintValue=ParametersDict['ConstraintValue'],
    AirfoilAim='maxClCd', # This means automatically look for
                          # max(Cl/Cd) condition
    # Number of iterations where AoA search is done
    # (few are usually enough)
    itMaxAoAsearch=3
    )
    print('Launching (2.a) design function... COMPLETED\n')
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   #
    # (2.b) - Educated-guess imposition of angle-of-attack
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   #
    # The resulting blade of (2.a) is noisy, as one can check:
    Span, AoA, Chord = J.getVars(LiftingLine,['Span','AoA','Chord'])
    #Chord[:] = np.minimum(ChordMax,Chord)
    
    if type(AoAImposed) == type(None):
    
        plt.plot(Span/np.max(Span), AoA)
        plt.show()
        exit()
    else:
        print('Imposing the given distribution of AoA...')
        plt.plot(Span/np.max(Span), AoA*1.)
        AoA[:] = AoAImposed
        plt.plot(Span/np.max(Span), AoA)
        plt.savefig('Images/AoA.png')
        #plt.close()
    # Now, the design call goes like this:
    print ('Launching (2.b) design function...')
    ResultsDict = PA.designPropellerAdkins(
    LiftingLine, # Initial guess -> It will be modified !
    PolarsInterpFuns,
    # These design parameters were defined in Step 0
    NBlades=ParametersDict['NBlades'],Velocity=ParametersDict['Velocity'],RPM=ParametersDict['RPM'],Temperature=ParametersDict['Temperature'],Density=ParametersDict['Density'],
    Constraint=ParametersDict['Constraint'],ConstraintValue=ParametersDict['ConstraintValue'],
    AirfoilAim='AoA'
    )

    return LiftingLine, ResultsDict, PyZonePolars, PolarsInterpFuns



def computeBEMTfromLiftingLineWithMOLAData(LiftingLineTest, ParametersDict, Pitch = 0.):

    Span, Chord, Twist, Sweep, Dihedral = J.getVars(LiftingLineTest, ['Span', 'Chord','Twist', 'Sweep','Dihedral'])
    SpanTotal = ParametersDict['Rmax']-ParametersDict['Rmin']
    #print(ParametersDict['PolarName'])
    myLL = LiftingLine(Name='MyPropeller', SpanMin=ParametersDict['Rmin'], SpanMax=ParametersDict['Rmax'], N=ParametersDict['NPts'],
             SpanwiseDistribution=dict(kind='bitanh',first=0.05*SpanTotal, last=0.0016*SpanTotal),
             GeometricalLaws=dict(
                Chord=dict(RelativeSpan=Span/Span.max(),
                           Chord=Chord,
                           InterpolationLaw='interp1d_linear'),
                Twist=dict(RelativeSpan=Span/Span.max(),
                           Twist=Twist,
                           InterpolationLaw='interp1d_linear'),
                Sweep=dict(RelativeSpan=Span/Span.max(),
                           Sweep=Sweep,
                           InterpolationLaw='interp1d_linear'),
                Dihedral=dict(RelativeSpan=Span/Span.max(),
                           Dihedral=Dihedral,
                           InterpolationLaw='interp1d_linear'),
                Airfoils=ParametersDict['GeomLaws']['PolarsDict'],
                ),
             AirfoilPolarsFilenames=ParametersDict['PolarName'],
             )
    Loads = J.get(LiftingLineTest, '.Loads')
    Predictions = M.BEMT.compute(myLL, model='Heene',
                            AxialVelocity=ParametersDict['Velocity'], RPM=ParametersDict['RPM'], 
                            Pitch=Loads['Pitch']+Pitch,
                            NumberOfBlades=ParametersDict['NBlades'],
                            Density=ParametersDict['Density'], Temperature=ParametersDict['Temperature'])
    return myLL, Predictions


def writeSetup(AllSetupDictionaries, setupFilename='setup.py'): 
    '''
    # SAME As MOLA PRE
    Write ``setup.py`` file using a dictionary of dictionaries containing setup
    information

    Parameters
    ----------

        AllSetupDictionaries : dict
            contains all dictionaries to be included in ``setup.py``

        setupFilename : str
            name of setup file

    '''

    Lines = '#!/usr/bin/python\n'
    Lines+= "'''\nMOLA %s setup.py file automatically generated in PREPROCESS\n"%MOLA.__version__
    Lines+= "Path to MOLA: %s\n"%MOLA.__MOLA_PATH__
    Lines+= "Commit SHA: %s\n'''\n\n"%MOLA.__SHA__

    for SetupDict in AllSetupDictionaries:
        Lines+=SetupDict+"="+pprint.pformat(AllSetupDictionaries[SetupDict])+"\n\n"

    with open(setupFilename,'w') as f: f.write(Lines)

    try: os.remove(setupFilename+'c')
    except: pass
