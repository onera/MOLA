#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

'''
MOLA - PostProcess.py

This module provides handly tools to post-treat the structural results.

First creation:
16/03/2022 - M. Balmaseda
'''

FAIL  = '\033[91m'
GREEN = '\033[92m'
WARN  = '\033[93m'
MAGE  = '\033[95m'
CYAN  = '\033[96m'
ENDC  = '\033[0m'


import os, sys
import numpy as np

import Converter.PyTree as C
import Converter.Internal as I 

import MOLA.InternalShortcuts as J

import matplotlib.pyplot as plt
import time
import imp
import pprint

def PlotVarFromCGNSForAllIterationsAtNodeRPMFcoeff(t, Var, Node, RPM, Fcoeff):
    """Plots a Variable over all the SavedIterations for a single Node, RPM and Fcoeff"""

    Varfunc = []
    Iterfunc = []

    DictStruct = J.get(t, '.StructuralParameters')
    TypeElemInMesh = DictStruct['MeshProperties']['DictElements']['GroupOfElements'].keys()

    for TypeElem in TypeElemInMesh:
        ZoneName = '%sRPM_Fcoeff%s'
        if  Node in DictStruct['MeshProperties']['DictElements']['GroupOfElements'][TypeElem]['Nodes']:

            for Base in I.getNodesFromName(t, 'Iteration*'):
                Iterfunc.append(int(Base[0][9:]))

                ZoneName = '%sRPM_FCoeff%s_%s'%(np.round(RPM,2), Fcoeff, TypeElem)
                print(ZoneName)
                Zone = I.getNodeByName(Base, ZoneName)
                
                Varfunc.append(J.getVars(Zone, [Var])[0][Node])
    
    print(Iterfunc)
    print(Varfunc)

    return Iterfunc, Varfunc         
    






def PlotAerodynamicParametersFromBEMT(LiftingLine, Iter = None, Blade = None):

    if (Iter != None) and (Blade == None):
        NameVars = 'Iter:'
    elif (Iter == None) and (Blade != None):
        NameVars = 'Blade:'
    else:
        print(FAIL + 'Error: The plot variable is not defined.')
        stop


    #Plot some interesting results
    ##############################   
    v = J.getVars2Dict(LiftingLine,['Mach', 'Span', 'AoA' , 'Chord', 'Cl', 'Cd','VelocityAxial', 'Reynolds'])
    

    #Subplot creation
    
    fig, ax = plt.subplots(2,2, num=1)

    fig.suptitle('Main aerodynamics performances')
   
    ax[0,0].plot(v['Span'],v['Mach'], label = '%s %s'%(NameVars, Iter))
    ax[0,0].set(xlabel='Relative span position [-]', ylabel='Mach [-]')

    ax[0,1].plot(v['Span'],v['AoA'], label = '%s %s'%(NameVars, Iter))
    ax[0,1].set(xlabel='Relative span position [-]', ylabel='Angle of attack [deg]')

    ax[1,0].plot(v['Span'],v['Reynolds'], label = '%s %s'%(NameVars, Iter))
    ax[1,0].set(xlabel='Relative span position [-]', ylabel='Number of Reynolds [-]')

    
    ax[1,1].plot(v['Span'],v['VelocityAxial'], label = '%s %s'%(NameVars, Iter))
    ax[1,1].set(xlabel='Relative span position [-]', ylabel='Axial velocity [m/s]')

    
    plt.show(block=False)
    plt.pause(0.5)
   


            
def GetAndSaveResults(pathOut,ModelDir,ModelName,FullResultsDict,LiftingLine, Up ,us, Residue,RPM):


    if not os.path.exists(pathOut + 'ResultsDicts'):
        os.makedirs(pathOut + 'ResultsDicts')

    #Dictionary is imported if it does exist
    try:
        import imp
        ResultsFile = imp.load_source('ResultsDic',pathOut + 'ResultsDicts/'+str(ModelDir)+str(ModelName)+'.py')
        ResultsDict = ResultsFile.ResultsDict
    except:
        ResultsDict = {}
        print('No dictionary found')
    ResultsDict[str(np.round(RPM,2))+'RPM']={}

   
    v = J.getVars2Dict(LiftingLine,['Mach', 'Span', 'AoA' , 'Chord', 'Cl', 'Cd','VelocityAxial', 'Reynolds'])
       

    
    #Results Loading

    #Propulsif global performances
    ResultsDict[str(np.round(RPM,2))+'RPM']['PropPerformances']={}
    ResultsDict[str(np.round(RPM,2))+'RPM']['PropPerformances']['Thrust']= FullResultsDict['Thrust']   
    ResultsDict[str(np.round(RPM,2))+'RPM']['PropPerformances']['FigureOfMeritPropeller']=FullResultsDict['FigureOfMeritPropeller']
    ResultsDict[str(np.round(RPM,2))+'RPM']['PropPerformances']['Power']=FullResultsDict['Power']
    ResultsDict[str(np.round(RPM,2))+'RPM']['PropPerformances']['Efficiency']=FullResultsDict['PropulsiveEfficiency']

    #Aero performances
    ResultsDict[str(np.round(RPM,2))+'RPM']['AeroPerformances']={}
    indexMaxList=[idx for idx, element in enumerate(v['Cl']) if element==max(v['Cl'])]
    indexMax=indexMaxList[0]
    ResultsDict[str(np.round(RPM,2))+'RPM']['AeroPerformances']['SeparationPosition']=v['Chord'][indexMax]
    ResultsDict[str(np.round(RPM,2))+'RPM']['AeroPerformances']['SeparationPercent']=(v['Chord'][indexMax])/(v['Chord'][-1])
    ResultsDict[str(np.round(RPM,2))+'RPM']['AeroPerformances']['MaxMach']=max(v['Mach'])
    ResultsDict[str(np.round(RPM,2))+'RPM']['AeroPerformances']['MaxAoA']=max(v['AoA'])
    ResultsDict[str(np.round(RPM,2))+'RPM']['AeroPerformances']['ClEvolution']=v['Cl']

    #Struct performances
    ResultsDict[str(np.round(RPM,2))+'RPM']['StructPerformances']={}
    ResultsDict[str(np.round(RPM,2))+'RPM']['StructPerformances']['MaxStress']=0
    ResultsDict[str(np.round(RPM,2))+'RPM']['StructPerformances']['MaxDeformation']=0
    ResultsDict[str(np.round(RPM,2))+'RPM']['StructPerformances']['MaxDisplacement']=max(Up)
    ResultsDict[str(np.round(RPM,2))+'RPM']['StructPerformances']['OverStressedAreaPercent']=0
    ResultsDict[str(np.round(RPM,2))+'RPM']['StructPerformances']['IsYealdingTrue']=0
    ResultsDict[str(np.round(RPM,2))+'RPM']['StructPerformances']['StressEvolution']=0
    ResultsDict[str(np.round(RPM,2))+'RPM']['StructPerformances']['DefEvolution']=0

    #Aeroelastic perfomances
    ResultsDict[str(np.round(RPM,2))+'RPM']['AELPerformances']={}
    ResultsDict[str(np.round(RPM,2))+'RPM']['AELPerformances']['MaxRelativeDeltaDisplacement']=(max(Up)-max(us))/max(us)
    #Convergence
    ResultsDict[str(np.round(RPM,2))+'RPM']['Convergence']={}
    ResultsDict[str(np.round(RPM,2))+'RPM']['Convergence']['Residue']=Residue
    

    Lines = ['#!/usr/bin/python\n']
    Lines = ['from numpy import array\n']
    Lines+= ['ResultsDict = '+pprint.pformat(ResultsDict)+"\n"]
    AllLines = '\n'.join(Lines)
    with open(pathOut + 'ResultsDicts/'+str(ModelDir)+str(ModelName)+'.py', "w") as a_file:
        #pickle.dump(DictBladeParameters, a_file)
        a_file.write(AllLines)
        a_file.close()

    
    print('\n')

    return ResultsDict



def CheckCriteria(ModelDir,ModelName,pathOut,RefModelDir,RPM):

    CaseDir=str(ModelDir)+str(ModelName)
    RefCaseDir=str(RefModelDir)+str(ModelName)
    
    if not os.path.exists(pathOut + 'ResultsDicts'):
        os.makedirs(pathOut + 'ResultsDicts')
    
    #Dictionary is imported if it does exist
    WeightingPath=pathOut
    try:
        WeightingDictC = imp.load_source('WeightingDict',WeightingPath+'Weightings.py')
        WeightingDict = WeightingDictC.WeightingDict
        try:
            WeightingDict[str(CaseDir)]    #If this key does not exit, it's created as a dictionary
        except:
            WeightingDict[str(CaseDir)]={}
    except: 
        WeightingDict = {}
        WeightingDict[str(CaseDir)]={}
        print('No weighting dictionary found')

    WeightingDict[str(CaseDir)][str(np.round(RPM,2))+'RPM']={}
    
    #Reading RefResultsDict
    try:
        RefResultsDictC = imp.load_source('RefResultsDict',pathOut+'ResultsDicts/'+str(RefCaseDir)+'.py')
        RefResultsDict = RefResultsDictC.ResultsDict
    except:
        RefResultsDict = {}
        print('No RefResultDict found')
   
    #Reading ResultsDict
    try:
        ResultsDictC = imp.load_source('ResultsDict',pathOut+'ResultsDicts/'+str(CaseDir)+'.py')
        ResultsDict = ResultsDictC.ResultsDict
    except:
        ResultsDict = {}
        print('No ResultDict found')
    

    #ResultsDict contains the performance parameters of a new blade
    #RefResultsDict contains the performance parameters of a reference/original blade
 
    #This weighting values could be decided to depend on the difference between current and ref blade
    CriteriaWeighting={}
    #Propulsif global performances
    CriteriaWeighting['PropPerformances']={}
    CriteriaWeighting['PropPerformances']['Thrust']=1
    CriteriaWeighting['PropPerformances']['Power']=1
    CriteriaWeighting['PropPerformances']['Efficiency']=2
    #Aero performances
    CriteriaWeighting['AeroPerformances']={}
    CriteriaWeighting['AeroPerformances']['SeparationPosition']=1
    CriteriaWeighting['AeroPerformances']['SeparationPercent']=-1
    CriteriaWeighting['AeroPerformances']['MaxMach']=-1
    CriteriaWeighting['AeroPerformances']['MaxAoA']=-1
    #Struct performances
    CriteriaWeighting['StructPerformances']={}
    CriteriaWeighting['StructPerformances']['MaxStress']=-1
    CriteriaWeighting['StructPerformances']['MaxDeformation']=-1
    CriteriaWeighting['StructPerformances']['MaxDisplacement']=-1
    CriteriaWeighting['StructPerformances']['OverStressedAreaPercent']=-1
    CriteriaWeighting['StructPerformances']['IsYealdingTrue']=-1
    #Aeroelastic perfomances ?

    
    #Every value/performance is compared to the reference blade value and previous 
    #weighting criteria are used to set a numerical value
    WeightingValue=0
    BladeDict=ResultsDict[str(np.round(RPM,2))+'RPM']
    try:
        RefBladeDict=RefResultsDict[str(np.round(RPM,2))+'RPM']
        for i in BladeDict:  #Group of parameters/peformances
            for j in BladeDict[i]: #Specific parameter
                try:
                    print('Calculating  '+str(i)+' and '+str(j))
                    if BladeDict[i][j]-RefBladeDict[i][j] !=0 :
                        WeightingValueStep=(CriteriaWeighting[i][j])*(BladeDict[i][j]-RefBladeDict[i][j])/abs(BladeDict[i][j]-RefBladeDict[i][j])
                        print('No zero: ' +str(WeightingValueStep))
                    else:
                        WeightingValueStep=0
                        print('Zero')
                    try:
                        WeightingValue+=WeightingValueStep[0]
                        WeightingDict[str(CaseDir)][str(np.round(RPM,2))+'RPM'][j]=WeightingValueStep[0]
                    except:
                        WeightingValue+=WeightingValueStep
                        WeightingDict[str(CaseDir)][str(np.round(RPM,2))+'RPM'][j]=WeightingValueStep

                except:
                    print('No parameter found')  


        WeightingDict[str(CaseDir)][str(np.round(RPM,2))+'RPM']['Total']=WeightingValue
        WeightingDict[str(CaseDir)][str(np.round(RPM,2))+'RPM']['WRT']=RefModelDir

        Lines = ['#!/usr/bin/python\n']
        Lines = ['from numpy import array\n']
        Lines+= ['WeightingDict = '+pprint.pformat(WeightingDict)+"\n"]
        AllLines = '\n'.join(Lines)
        with open(pathOut + '/Weightings.py', "w") as a_file:
            #pickle.dump(DictBladeParameters, a_file)
            a_file.write(AllLines)
            a_file.close()
        print('\n')
    except:
        print(WARN+'Reference blade has no '+str(RPM)+'RPM information'+ENDC)
    return WeightingValue

