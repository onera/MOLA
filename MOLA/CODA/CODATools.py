from telnetlib import NAOL
from typing import Dict
import numpy as np
from mpi4py import MPI

import Converter.Internal as I
import Converter.Mpi as Cmpi
import Converter.PyTree as C
import Geom.PyTree as D
import Post.PyTree as P
import Post.Mpi as Pmpi
import MOLA.Coprocess as CO
import MOLA.LiftingLine as LL
import MOLA.InternalShortcuts as J
import MOLA.Structure.Models as SM
import os

import time

from FSDataManager import FSMeshEnums

import timeit

LaunchTime = timeit.default_timer()
from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
NProcs = comm.Get_size()


FAIL  = '\033[91m'
GREEN = '\033[92m'
WARN  = '\033[93m'
MAGE  = '\033[95m'
CYAN  = '\033[96m'
ENDC  = '\033[0m'




comm = MPI.COMM_WORLD
rank = comm.Get_rank()


#----------------------------- SETTINGS ------------------------------ #
FULL_CGNS_MODE   = False
FILE_SETUP       = 'setup.py'
FILE_CGNS        = 'main.cgns'
FILE_SURFACES    = 'surfaces.cgns'
FILE_ARRAYS       = 'arrays.cgns'
FILE_FIELDS      = 'tmp-fields.cgns' # BEWARE of tmp- suffix
FILE_COLOG       = 'coprocess.log'
FILE_BODYFORCESRC= 'bodyforce.cgns'
DIRECTORY_OUTPUT = 'OUTPUT'
DIRECTORY_LOGS   = 'LOGS'
# Load Setup:
# ------------------ IMPORT AND SET CURRENT SETUP DATA ------------------ #
setup = J.load_source('setup',FILE_SETUP)
# Load and appropriately set variables of coprocess module
CO.FULL_CGNS_MODE   = FULL_CGNS_MODE
CO.FILE_SETUP       = FILE_SETUP
CO.FILE_CGNS        = FILE_CGNS
CO.FILE_SURFACES    = FILE_SURFACES
CO.FILE_ARRAYS       = FILE_ARRAYS
CO.FILE_FIELDS      = FILE_FIELDS
CO.FILE_COLOG       = FILE_COLOG
CO.FILE_BODYFORCESRC= FILE_BODYFORCESRC
CO.DIRECTORY_OUTPUT = DIRECTORY_OUTPUT
CO.setup            = setup
if rank==0:
    try: os.makedirs(DIRECTORY_OUTPUT)
    except: pass
    try: os.makedirs(DIRECTORY_LOGS)
    except: pass

# --------------------------- END OF IMPORTS --------------------------- #





def FSArray2Numpy(a):    
    return np.array(a.Buffer(), copy = False)

def cellType2CODACellObject(name):
    #print('Name: %s'%name)
    if name == 'Hexa8':
        CODACellObject = FSMeshEnums.CT_Hexa8
    elif name == 'Tetra4':
        CODACellObject = FSMeshEnums.CT_Tetra4
    elif name == 'Pyra5':
        CODACellObject = FSMeshEnums.CT_Pyra5
    elif name == 'Prism6':
        CODACellObject = FSMeshEnums.CT_Prism6
    else:
        CO.printCo('%s not recognized elementType!'%name, color = CO.FAIL)

    return CODACellObject

def AddFSCellTypeAndConnectivity2ElementDict(DictElements, fsmesh):

    cellTypes = (FSMeshEnums.CT_Hexa8,
                 FSMeshEnums.CT_Tetra4,
                 FSMeshEnums.CT_Pyra5,
                 FSMeshEnums.CT_Prism6,
                 #FSMeshEnums.CT_Quad4,
                 #FSMeshEnums.CT_Edge2,             
                 #FSMeshEnums.CT_Node
                 )
    ListOfAcceptedElements = ['Hexa8', 'Tetra4', 'Pyra5', 'Prism6']

    #DictCellTypes = {}
    first_index_of_elt_type = []
    counter = 0
    #print('%s'%fsmesh.GetCellTypeArray())
    for ct in fsmesh.GetCellTypeArray():
         
        
        Ncells = fsmesh.GetNCells(ct)
        
        if (Ncells > 0) and ('%s'%FSMeshEnums.CellTypeToString(ct) in ListOfAcceptedElements):
            first_index_of_elt_type.append(counter)
               
            #CO.printCo('%s'%('%s'%FSMeshEnums.CellTypeToString(ct) in ListOfAcceptedElements), color = CO.WARN)
            #CO.printCo('%s'%FSMeshEnums.CellTypeToString(ct), color = CO.WARN)



            DictElements['GroupOfElements']['%s'%FSMeshEnums.CellTypeToString(ct)] = {}
            DictElements['GroupOfElements']['%s'%FSMeshEnums.CellTypeToString(ct)]['IndexNElements'] = [counter, Ncells]
            DictElements['GroupOfElements']['%s'%FSMeshEnums.CellTypeToString(ct)]['CODATypeE'] = cellType2CODACellObject('%s'%FSMeshEnums.CellTypeToString(ct))
            
            #CO.printCo('%s :: %s'%('%s'%FSMeshEnums.CellTypeToString(ct),DictElements['GroupOfElements']['%s'%FSMeshEnums.CellTypeToString(ct)]['IndexNElements']), proc=0)
            #print(help(fsmesh.GetCell2Proc()))
            #CO.printCo('%s_%s'%(Cmpi.rank, FSMeshEnums.CellTypeToString(ct)))
            #if Cmpi.rank == 0:
            #    print(FSArray2Numpy(fsmesh.GetCell2Proc(ct)))
            #    print(FSArray2Numpy(fsmesh.GetCellTypeArray()))
                
        #if  FSMeshEnums.CellTypeToString(ct) != 'Node':
            # Connectivity:
            Conectivity = FSArray2Numpy(fsmesh.GetCell2Node(ct))
            DictElements['GroupOfElements']['%s'%FSMeshEnums.CellTypeToString(ct)]['Connectivity'] = Conectivity + 1
            
            # Add Nodes to celltype:
            #DictElements['%s'%FSMeshEnums.CellTypeToString(ct)] = {}
            DictElements['GroupOfElements']['%s'%FSMeshEnums.CellTypeToString(ct)]['Nodes'] = np.unique(Conectivity)
            
            if np.amax(DictElements['GroupOfElements']['%s'%FSMeshEnums.CellTypeToString(ct)]['Connectivity']) > len(DictElements['GroupOfElements']['%s'%FSMeshEnums.CellTypeToString(ct)]['Nodes']):
                #CO.printCo('The %s conectivity should be initialized...'%FSMeshEnums.CellTypeToString(ct), proc=0, color = CO.WARN)
                InitialShape = np.shape(DictElements['GroupOfElements']['%s'%FSMeshEnums.CellTypeToString(ct)]['Connectivity'])
                ary = DictElements['GroupOfElements']['%s'%FSMeshEnums.CellTypeToString(ct)]['Connectivity'].ravel()
                #you don't need a defaultdict, but I like to use them just in case, instead of a dictionary, as a general practice
                rank = {}
                for x in ary: #first, we get all the values of the array and put them in the dict
                    rank[x] = 0
                count = 1
                for x in sorted(rank): #now we go from lowest to highest in the array, adding 1 to the dict-value
                    rank[x] = count
                    count += 1
                ary = [rank[x] for x in ary]
                DictElements['GroupOfElements']['%s'%FSMeshEnums.CellTypeToString(ct)]['Connectivity'] = np.reshape(ary, InitialShape)
            
            counter += fsmesh.GetNCells(ct)
        #else:
            #CO.printCo('%s not added to DictElements'%FSMeshEnums.CellTypeToString(ct))
            
def AddCGNSNameAndType2DictElements(DictElements):
    
    for ElemName in DictElements['GroupOfElements']:#.keys():
        
        DictElements['GroupOfElements'][ElemName]['CGNSType'], _,DictElements['GroupOfElements'][ElemName]['CellDimension'] = SM.CGNSElementType(ElemName)


def AddCoordinates2DictElements(DictElements, fsmesh):

    CoordinatesValueFS = fsmesh.GetUnstructDataset("Coordinates").GetValues()
    CoordinatesArray   = FSArray2Numpy(CoordinatesValueFS)
        
    
    for ElemName in DictElements['GroupOfElements'].keys():
        #if (ElemName != 'Node') and (ElemName != 'Quad4'):

        DictElements['GroupOfElements'][ElemName]['Coordinates'] = np.array(CoordinatesArray[DictElements['GroupOfElements'][ElemName]['Nodes'],:], copy = False)
            

    #return DictElements

def getCODAStatefromFSDM(fsmesh):

    StateFS = fsmesh.GetUnstructDataset("State")
    StateValueFS = StateFS.GetValues()
    ValuesArray  = FSArray2Numpy(StateValueFS)
    VarNumber = np.shape(ValuesArray)[1]
    NameOfVars  = StateFS.GetNames()
     
    DictAugState = {}
    for name, pos in zip(NameOfVars, range(len(NameOfVars))):

        DictAugState['%s'%name] = np.array(ValuesArray[:,pos], copy = False)
    
    #DictAugState['Density'][:] = 150.
    return ValuesArray, NameOfVars, VarNumber, DictAugState

def getCODASourceTermsfromFSDM(fsmesh):

    StateFS = fsmesh.GetUnstructDataset("Source")
    StateValueFS = StateFS.GetValues()
    ValuesArray  = FSArray2Numpy(StateValueFS)
    #CO.printCo('%s %s'%ValuesArray.size)
    
    VarNumber = np.shape(ValuesArray)[1]
     
    NameOfVars  = StateFS.GetNames()

    DictSourceTerm = {}
    for name, pos in zip(NameOfVars, range(len(NameOfVars))):

        DictSourceTerm['%s'%name] = np.array(ValuesArray[:,pos], copy = False)
    
    return DictSourceTerm, NameOfVars


def transferStateVisu2State(fsmesh):

    StateFS = fsmesh.GetUnstructDataset("State")
    StateValueFS = StateFS.GetValues()
    ValuesArrayState  = FSArray2Numpy(StateValueFS)
    #CO.printCo('%s %s'%ValuesArray.size)

    StateVisuFS = fsmesh.GetUnstructDataset("StateVisu")
    StateVisuValueFS = StateVisuFS.GetValues()
    ValuesArrayVisu  = FSArray2Numpy(StateVisuValueFS)

    
    VarNumber = np.shape(ValuesArrayState)[1]
     
    NameOfVars  = StateFS.GetNames()

    DictState = {}
    DictStateVisu = {}

    for name, pos in zip(NameOfVars, range(len(NameOfVars))):

        DictState['%s'%name] = np.array(ValuesArrayState[:,pos], copy = False)    
        DictState['%s'%name][:] = np.array(ValuesArrayVisu[:,pos], copy = True)
    
    return fsmesh



def AddFlowSolution2DictElements(DictElements, fsmesh):
    
    try: 
        ValuesArray, NameOfVars, VarNumber, DictAugState = getCODAStatefromFSDM(fsmesh)
    except:
        print(WARN+"CodaAugState not found in the fsmesh"+ENDC)
        return DictElements

    #if rank == 0:
    #    print(np.shape(ValuesArray))#, proc = 0)
    
    #fsmesh.ExportMeshHDF5(Filename='StateMesh.hdf') or FSError.PrintAndExit()
    
    it = 0
    for ElemName in DictElements['GroupOfElements']:
        it += 1 
        #if rank == 0:
        #    print(np.shape(DictElements[ElemName]['Nodes']))#, proc = 0)
        #    print(np.max(np.array(DictElements[ElemName]['Nodes'])))#, proc = 0)
    
        #if (ElemName != 'Node') and (ElemName != 'Quad4'):
    
        DictElements['GroupOfElements'][ElemName]['FlowSolution'] = {}
        
        for varName, varNumber in zip(NameOfVars, range(VarNumber)):

            #CO.printCo('%s'%DictAugState['%s'%varName].flags, proc =0, color = CO.MAGE)

            #CO.printCo('%s %s'%(ElemName, varName), color = CO.FAIL, proc = 0)
            DictElements['GroupOfElements'][ElemName]['FlowSolution']['%s'%varName] = np.array(DictAugState['%s'%varName],copy = False)
            
            #DictElements['GroupOfElements'][ElemName]['FlowSolution']['%s'%varName][np.arange(DictElements['GroupOfElements'][ElemName]['IndexNElements'][0],DictElements['GroupOfElements'][ElemName]['IndexNElements'][0]+50)[:]] = 260.
            
            
            #NelemProc = DictElements['GroupOfElements']['Tetra4']['IndexNElements'][1] + DictElements['GroupOfElements']['Hexa8']['IndexNElements'][1] + DictElements['GroupOfElements']['Pyra5']['IndexNElements'][1]
            #CO.printCo('%s :: %s :: %s :: %s !! %s # %s'%(ElemName, varName, DictElements['GroupOfElements'][ElemName]['IndexNElements'][1], len(DictAugState['%s'%varName]), NelemProc, DictElements['GroupOfElements'][ElemName]['IndexNElements'][0]), color = CO.CYAN, proc = 0)
            #CO.printCo('%s'%DictElements['GroupOfElements'][ElemName]['FlowSolution']['%s'%varName].flags, proc =0, color = CO.MAGE)
            #DictElements['GroupOfElements'][ElemName]['FlowSolution']['%s'%varName][DictElements['GroupOfElements'][ElemName]['IndexNElements'][0]:DictElements['GroupOfElements'][ElemName]['IndexNElements'][0]+DictElements['GroupOfElements'][ElemName]['IndexNElements'][1]] = it * 5000
    
    #Cmpi.barrier()
    #fsmesh.ExportMeshTECPLOT(Filename='MeshTest.dat',PrefixDatasetName=True) or FSError.PrintAndExit()
    
    #return DictElements

#def initializeFlowSolution2DictElements(DictElements, fsmesh):
#    
#    try: 
#        _, NameOfVars, _, _ = getCODAStatefromFSDM(fsmesh)
#    except:
#        print(WARN+"CodaAugState not found in the fsmesh"+ENDC)
#        return DictElements
#
#    for ElemName in DictElements['GroupOfElements']:
#
#        DictElements['GroupOfElements'][ElemName]['FlowSolution'] = {}
#        for varName in NameOfVars:
#            DictElements['GroupOfElements'][ElemName]['FlowSolution']['%s'%varName] = np.zeros(DictElements['GroupOfElements'][ElemName]['NElements'],)



def AddSolutionFields2CGNSFromDictElement(Base, DictElements, ElemName):
    #print(DictElements['GroupOfElements'][ElemName]['FlowSolution'])
    VariableNames = [ '%s'%x for x in DictElements['GroupOfElements'][ElemName]['FlowSolution']]
    
    
    zone = I.getNodeFromNameAndType(Base, 'Element_%s.%s'%(ElemName,Cmpi.rank), 'Zone_t')
    
    FlowNode = I.createNode('FlowSolution#Centers', 'FlowSolution_t', parent=zone)
    I.createNode('GridLocation', 'GridLocation_t', value ='CellCenter', parent=FlowNode)
        
    for var in VariableNames:
        FSarray = DictElements['GroupOfElements'][ElemName]['FlowSolution'][var][DictElements['GroupOfElements'][ElemName]['IndexNElements'][0]:DictElements['GroupOfElements'][ElemName]['IndexNElements'][0]+DictElements['GroupOfElements'][ElemName]['IndexNElements'][1]]
        
        #CO.printCo('%s %s'%(np.shape(FSarray), DictElements['GroupOfElements'][ElemName]['IndexNElements'][1]))
        
        I.createNode('%s'%var, 'DataArray_t', value=np.array(FSarray, dtype = np.float64,order='F'), parent=FlowNode)
    #CO.save(Base, 'Base%s.cgns'%ElemName)
    #CO.printCo('%s'%np.max(abs(FSarray)), color = CO.MAGE)
    #errors = []
    #errors += I.checkPyTree(Base)
    #CO.printCo('%s'%errors)
    #Base = I.correctPyTree(Base)
    

    #CO.save(t, 'NodesUpdate.cgns')
    
#def initializeZeroSolutionFields2CGNSFromDictElement(Base, DictElements, ElemName):
#    #print(DictElements['GroupOfElements'][ElemName]['FlowSolution'])
#    VariableNames = [ '%s'%x for x in DictElements['GroupOfElements'][ElemName]['FlowSolution']]
#    
#    zone = I.getNodeFromNameAndType(Base, 'Element_%s.%s'%(ElemName,Cmpi.rank), 'Zone_t')
#    
#    # L'arbre FSDM sauvegarde les valeurs aux centres:
#    
#    #FlowNode = I.createNode('FlowSolution#Centers', 'FlowSolution_t', parent=zone)
#    #I.createNode('GridLocation', 'GridLocation_t', value ='CellCenter', parent=FlowNode)
#    #    
#    #for var in VariableNames:
#    #    FSarray = DictElements['GroupOfElements'][ElemName]['FlowSolution'][var]
#    #    I.createNode(var, 'DataArray_t', value=np.array(FSarray, order='F'), parent=FlowNode)
#    
#    for name in VariableNames:
#        J.invokeFields(zone, ['%s'%name], locationTag = 'centers:')
#    
#    if rank == 0:
#        C.convertPyTree2File(Base, 'Base0.cgns')
    


#def buildIntegrationCGNS(fsmesh, DictElements):
#    # Zone at nodes for FSDM:
#    CoordinatesValueFS = fsmesh.GetUnstructDataset("Coordinates").GetValues()
#    CoordinatesArray   = FSArray2Numpy(CoordinatesValueFS)
#
#    nodesIntDict = {}
#
#    for ElemName in DictElements['GroupOfElements']:
#
#        x = np.array(CoordinatesArray[DictElements['GroupOfElements'][ElemName]['Nodes'],0], copy=False)#True, order='F')
#        y = np.array(CoordinatesArray[DictElements['GroupOfElements'][ElemName]['Nodes'],1], copy=False)#True, order='F')
#        z = np.array(CoordinatesArray[DictElements['GroupOfElements'][ElemName]['Nodes'],2], copy=False)#True, order='F')
#        #import Geom.PyTree as D
#        nodesIntDict[ElemName] = D.line((0,0,0),(1,0,0),len(x))
#        xl, yl, zl = J.getxyz(nodesIntDict[ElemName])
#        xl[:] = x
#        yl[:] = y
#        zl[:] = z
#
#    return nodesIntDict

def buildIntegrationCGNS(fsmesh):
    # Zone at nodes for FSDM:
    CoordinatesValueFS = fsmesh.GetUnstructDataset("Coordinates").GetValues()
    CoordinatesArray   = FSArray2Numpy(CoordinatesValueFS)

    
    x = np.array(CoordinatesArray[:,0], copy=False)#True, order='F')
    y = np.array(CoordinatesArray[:,1], copy=False)#True, order='F')
    z = np.array(CoordinatesArray[:,2], copy=False)#True, order='F')
    #import Geom.PyTree as D
    nodesIntDict = D.line((0,0,0),(1,0,0),len(x))
    xl, yl, zl = J.getxyz(nodesIntDict)
    xl[:] = x
    yl[:] = y
    zl[:] = z

    return nodesIntDict
def getNonDimensionalCoefficientsFromSetup(setup):

    DictOfNonDimensionalCoeff = {}
    DictOfNonDimensionalCoeff['NonDimensionalDensityCoef'] = setup.ReferenceValues['NonDimensionalDensityCoef']
    DictOfNonDimensionalCoeff['NonDimensionalMomentumCoef'] = setup.ReferenceValues['NonDimensionalMomentumCoef']
    DictOfNonDimensionalCoeff['NonDimensionalMomentumXCoef'] = DictOfNonDimensionalCoeff['NonDimensionalMomentumCoef'] 
    DictOfNonDimensionalCoeff['NonDimensionalMomentumYCoef'] = DictOfNonDimensionalCoeff['NonDimensionalMomentumCoef']
    DictOfNonDimensionalCoeff['NonDimensionalMomentumZCoef'] = DictOfNonDimensionalCoeff['NonDimensionalMomentumCoef']
    
    DictOfNonDimensionalCoeff['NonDimensionalEnergyStagnationDensityCoef'] =  setup.ReferenceValues['NonDimensionalEnergyStagnationDensityCoef']

    DictOfNonDimensionalCoeff['NonDimensionalSourceCoef'] =  setup.ReferenceValues['NonDimensionalPressureCoef'] * setup.ReferenceValues['Length']


    return DictOfNonDimensionalCoeff

def dimensionalizeFlowSolutionDict(FlowSolutionDict, DictOfNonDimensionalCoeff):

    dimensionalFlowSolutionDict = {}
    for Var in FlowSolutionDict:
        #CO.printCo(Var)
        if not 'SANuTilde' in Var:
            dimensionalFlowSolutionDict[Var] = FlowSolutionDict[Var] / DictOfNonDimensionalCoeff['NonDimensional%sCoef'%Var]
        else:
            dimensionalFlowSolutionDict[Var] = FlowSolutionDict[Var] #* 0.
            
    return dimensionalFlowSolutionDict



def adimensionalizeFlowSolutionDict(FlowSolutionDict, DictOfNonDimensionalCoeff, SourceTerm = False):

    adimensionalFlowSolutionDict = {}
    for Var in FlowSolutionDict:
        if not SourceTerm:
            adimensionalFlowSolutionDict[Var] = FlowSolutionDict[Var] * DictOfNonDimensionalCoeff['NonDimensional%sCoef'%Var]
        else:
            #if not 'SANuTilde' in Var:
            adimensionalFlowSolutionDict[Var] = FlowSolutionDict[Var] * DictOfNonDimensionalCoeff['NonDimensionalSourceCoef']
            #else:
            #    adimensionalFlowSolutionDict[Var] = FlowSolutionDict[Var]

    return adimensionalFlowSolutionDict


def setFlowData2FSDMAugState(flowDict, fsmesh):
    
    ValuesArray, NameOfVars, VarNumber, DictAugState = getCODAStatefromFSDM(fsmesh)
    #CO.printCo('%s'%NameOfVars)
    for VarName, Number in zip(NameOfVars, range(VarNumber)):
    		
        try:
            #CO.printCo('%s %s'%(np.max(DictAugState['%s'%VarName]), np.max(flowDict['%s'%VarName])))
            DictAugState['%s'%VarName][:] = flowDict['%s'%VarName]
        except:
            CO.printCo('%s does not exist in the dictionary.'%VarName)

def transferVars(varName):
    if 'BodyForce' in '%s'%varName:
        var1 = '%s'%varName
        varTransfer = 'Momentum%s'%var1[-1]
    else:
        varTransfer = '%s'%varName
    return varTransfer



def buildListIndexDictionnaryCGNS2FSDM(tDonor, tRec):
    DictOfListIndex = {}
    for zone in I.getZones(C.node2Center(tDonor)):
        DictOfListIndex[zone[0]] = getListIndexForDataTransfer(zone, tRec)

    return DictOfListIndex

def setSourceTerms2FSDM(t, nodesInt,fsmesh, DictOfListIndex, NonDimensionalCoefficientsDict):
     """t: arbre cgns sortie du couplage body force
        fsmesh: integration mesh"""
   
     
     #nodesInt = buildIntegrationCGNS(fsmesh)
     # FSmesh dict: 
     DictSourceTerm, NameOfVars = getCODASourceTermsfromFSDM(fsmesh)
     
     #CO.printCo('%s'%(time.time()-t1), proc=0, color = CO.WARN)
     DimDictSourceTerm = {}
     for zone in I.getZones(t):
         #C.convertPyTree2File(zone, 'Zone0%s.cgns'%rank)
         #zone = I.getZones(C.node2Center(zone))
         #CO.printCo('Zone:\n %s'%zone)
         #C.convertPyTree2File(zone, 'Zone%s.cgns'%rank)
         t1 = time.time()
         ListIndex = DictOfListIndex[zone[0]] #  getListIndexForDataTransfer(zone, nodesInt)
         #CO.printCo(str(ListIndex))
       
         CO.printCo('%s'%(time.time()-t1), proc=0, color = CO.FAIL)
         
         try:
             for varName in NameOfVars:
                 #if 'Momentum' in varName:    
                 CO.printCo('%s'%varName)
                 varTransfer = transferVars(varName)
                 CO.printCo('AfterTransfer')
                 DimDictSourceTerm['%s'%varTransfer] = transferCGNSDataFromSourceZone2RecZone(zone, nodesInt, '%s'%varTransfer, ListIndex)
                 CO.printCo('AfterTransferCGNS')
                 adimensionalVarDict = adimensionalizeFlowSolutionDict(DimDictSourceTerm, NonDimensionalCoefficientsDict, SourceTerm = True ) 
                 CO.printCo('AfterDim')
             for varName in NameOfVars:
                 #if 'Momentum' in varName:
                 if 'Turbulent' not in '%s'%varName:
                     varTransfer = transferVars(varName)
                     CO.printCo('BeforerAdim')
                     DictSourceTerm['%s'%varName][:] = adimensionalVarDict[varTransfer]
                     CO.printCo('Source Term adim %s: %s'%(varName, np.max(abs(DictSourceTerm['%s'%varName])))) 
         except: 
             #pass
             CO.printCo('Source not found in rank %s'%rank)
     return fsmesh

#def transferCGNSDataFromSourceZone2RecZone(sourceZone, recZone,varName, ListIndex,  Container = 'FlowSolution#SourceTerm'):


# def setSourceTerms2FSDM(t, nodesInt,fsmesh, setup):
#     """t: arbre cgns sortie du couplage body force
#        fsmesh: integration mesh"""
    
#     NonDimensionalCoefficientsDict =  getNonDimensionalCoefficientsFromSetup(setup)
    

#     t1 = time.time()
#     #nodesInt = buildIntegrationCGNS(fsmesh)
#     # FSmesh dict: 
#     DictSourceTerm, NameOfVars = getCODASourceTermsfromFSDM(fsmesh)
#     CO.printCo('%s'%(time.time()-t1), proc=0, color = CO.WARN)

#     I.__FlowSolutionNodes__ = 'FlowSolution#SourceTerm'
    
    
#     for zone in I.getZones(t):
#     #for zone in I.getZones(C.node2Center(t)):
#         #C.convertPyTree2File(zone, 'Zone0%s.cgns'%rank)
#         #zone = I.getZones(C.node2Center(zone))
#         #CO.printCo('Zone:\n %s'%zone)
#         #C.convertPyTree2File(zone, 'Zone%s.cgns'%rank)
#         t1 = time.time()
#         ##ListIndex = getListIndexForDataTransfer(zone, nodesInt)
#         #CO.printCo(str(ListIndex))
#         C.convertPyTree2File(zone, 'zone_%s.cgns'%zone[0])
#         P._extractMesh(zone, nodesInt, constraint = 0)
#         C.convertPyTree2File(nodesInt, 'Extract_%s.cgns'%zone[0])
#         print(len(DictSourceTerm['BodyForceX']))
#         CO.printCo('::%s'%(time.time()-t1), proc=0, color = CO.FAIL)

#     DimDictSourceTerm = {}
#     #try:
#         #for varName in NameOfVars:
#         #    varTransfer = transferVars(varName)
#         #    DimDictSourceTerm['%s'%varTransfer] = transferCGNSDataFromSourceZone2RecZone(zone, nodesInt, '%s'%varTransfer, ListIndex)
        
#     for varName in NameOfVars:
#         varTransfer = transferVars(varName)
#         DimDictSourceTerm['%s'%varTransfer] = J.getVars(nodesInt, ['%s'%varTransfer], Container = I.__FlowSolutionNodes__)[0]
#     adimensionalVarDict = adimensionalizeFlowSolutionDict(DimDictSourceTerm, NonDimensionalCoefficientsDict, SourceTerm = True ) 
#     #CO.printCo('%s'%adimensionalVarDict)
#     for varName in NameOfVars:
#         CO.printCo('!! %s !! %s'%(len(DictSourceTerm['%s'%varName]), len(adimensionalVarDict[varTransfer])), proc = 0, color = CO.CYAN)
#         CO.printCo('!! %s !! %s'%(np.min(DictSourceTerm['%s'%varName]), np.min(adimensionalVarDict[varTransfer])), proc = 0, color = CO.CYAN)
#         varTransfer = transferVars(varName)
#         DictSourceTerm['%s'%varName][:] = adimensionalVarDict[varTransfer]
#         CO.printCo('Source Term adim %s: %s'%(varName, np.max(abs(DictSourceTerm['%s'%varName])))) 
#     #except: 
        
#     #    CO.printCo('Source not found in rank %s'%rank)
#     #    pass
#     I.__FlowSolutionNodes__ = 'FlowSolution'
    
#     return fsmesh



def getListIndexForDataTransfer(sourceZone, recZone):
    #varSource[:] = range(len(varSource))
    #CO.printCo('%s'%varSource)
    hook = C.createHook(recZone, function = 'nodes')
    ListIndex = C.identifyNodes(hook, sourceZone) -1
    return ListIndex

    #CO.printCo('%s \n \n %s \n \n %s '%(hook, sourceZone, recZone))
    #CO.printCo('%s '%(ListIndex))

def transferCGNSDataFromSourceZone2RecZone(sourceZone, recZone,varName, ListIndex, ContainerSource = 'FlowSolution#SourceTerm'):
    varSource = J.getVars(sourceZone, [varName], Container = ContainerSource)[0]
    
    varHook = J.getVars(recZone, [varName])[0]
    
    try:
        if varHook == None:
            J.invokeFields(recZone, [varName])
            varHook = J.getVars(recZone, [varName])[0]
    except:
        pass
    #CO.printCo('Hook %s \n \n SOurce %s'%(varHook, varSource))
    if varName != 'TurbulentSANuTildeDensity':
        varHook[ListIndex[np.where(ListIndex != -2)]] = varSource[np.where(ListIndex != -2)]
    
    return varHook

def rmGhostCellsFromZone(zone, ListIndex, varName):
    x,y,z = J.getxyz(zone)
    x,y,z = np.delete(x, ListIndex[np.where(ListIndex == -2)]), np.delete(y, ListIndex[np.where(ListIndex == -2)]), np.delete(z, ListIndex[np.where(ListIndex == -2)])
    

def updateCGNSfsMeshTreeWithIntegrationData(t,nodesInt, integrationfsmesh):

    # Get the integration values:
    #nodesInt = buildIntegrationCGNS(integrationfsmesh)
    _, NameOfVars, _, DictAugState = getCODAStatefromFSDM(integrationfsmesh)
    
    I.__FlowSolutionNodes__ = 'FlowSolution#Init'
    #for ElemName in nodesInt:
    for name in NameOfVars:
        J.invokeFields(nodesInt, ['%s'%name])
        var = J.getVars(nodesInt, ['%s'%name], Container = I.__FlowSolutionNodes__)[0]                         # Modificar esta funcion y pensar ext?
        var[:] = DictAugState['%s'%name]
    
    
    for zone in I.getZones(t):
        for name in NameOfVars:
            J.invokeFields(zone, ['%s'%name]) 
    #CO.save(nodesInt, 'updateNodesInt.cgns')
    # Dimensionalize the AugState:

    
    # Transfer the data from IntFSCGNS to the NodeCenter of the fsmeshCGNS:
    #CO.printCo('1', proc = 0)

    I._rmNodesByName(t, 'FlowSolution#Init')
    
    #CO.save(t, 'TreeAvant.cgns')
    hook = C.createGlobalHook(nodesInt, 'nodes')
    for zone in I.getZones(t):
        #zoneTree = C.node2Center(zone)
        #CO.printCo('2', proc = 0)
        #t1 = time.time()
        #ListIndex = getListIndexForDataTransfer(nodesInt, zoneTree)
        #C.convertPyTree2File(nodesInt, 'NodesInt%s.cgns'%zone[0])
        #C.convertPyTree2File(zone, 'toto%s.cgns'%zone[0])
        t1 = time.time()
        C._identifySolutions(zone, nodesInt, hookN = hook)
        #P._extractMesh(nodesInt, zone, constraint = 0)
        
        #CO.printCo('TransferData:: %s'%(time.time()-t1))
        # C.convertPyTree2File(nodesInt, 'NodesInt%s.cgns'%zone[0])
        # C.convertPyTree2File(zone, 'toto%s.cgns'%zone[0])
        #for varName in NameOfVars:
        #    CO.printCo('%s'%varName, proc =0)
        #    
        #    #varHook = transferCGNSDataFromSourceZone2RecZone(nodesInt, zoneTree, '%s'%varName,ListIndex, ContainerSource = 'FlowSolution')
        #    
        #    var = J.getVars(zone, ['%s'%varName], Container = 'FlowSolution#Centers')[0]
        #    #var[:] = varHook
    
        #rmGhostCellsFromZone(zone, ListIndex, NameOfVars)
    
    # Transfer data to "t" tree:
    
    #CO.save(t, 'CentersUpdate.cgns')
    
    for zone in I.getZones(t):
        for varName in NameOfVars:
            C.node2Center__(zone, '%s'%varName)
#    I._rmNodesByName(t, 'FlowSolution#Centers')
    I._rmNodesByName(t, 'FlowSolution#Init')
    #CO.save(t, 'NodesUpdate.cgns')
    I.__FlowSolutionNodes__ = 'FlowSolution'
    

#CO.printCo(FAIL+'%s'%np.max(abs(DictAugState['MomentumZ']))+ENDC)
#
#        CO.printCo(WARN+'%s'%np.max(abs(dimensionalAugState['Density']))+ENDC)
#        CO.printCo(WARN+'%s'%np.max(abs(dimensionalAugState['MomentumZ']))+ENDC)
#
#    pass


# def updateFsMeshFromCGNSTree(t, fsmesh):

#     _, NameOfVars, _, DictAugState = getCODAStatefromFSDM(fsmesh)

        
#     for zone in I.getZones(t):
#         for varName in DictAugState:

#             #var = J.getVars(zone, ['%s'%varName], Container = 'FlowSolution#Init')[0]
#             #CO.printCo('%s %s'%(np.min(var), np.max((var))))
#             C.node2Center__(zone, '%s'%varName, cellNType=0)
#             #CO.printCo('var: %s'%var)
#             DictAugState['%s'%varName][:] = var


#     return fsmesh
    



def computeDictElementsFromfsMesh(fsmesh):

    DictElements = {}
    DictElements['GroupOfElements'] = {}

    AddFSCellTypeAndConnectivity2ElementDict(DictElements, fsmesh)
    AddCGNSNameAndType2DictElements(DictElements)
    AddCoordinates2DictElements(DictElements, fsmesh)
    AddFlowSolution2DictElements(DictElements, fsmesh)

    return DictElements





def names(nodeA, nodeB):
    equal = nodeA[0] == nodeB[0]
    if not equal:
        print('node name %s is not equal to node name %s'%(nodeA[0],nodeB[0]))
    
def types(nodeA, nodeB):
    equal = nodeA[3] == nodeB[3]
    if not equal:
        print('nodeA name %s , nodeB name %s'%(nodeA[0],nodeB[0]))
        print('nodeA type %s is not equal to nodeB type %s'%(nodeA[0],nodeB[0]))


def values(nodeA, nodeB, data_comparison=None):
    valueA = nodeA[1]
    valueB = nodeB[1]

    same_type = type(valueA) == type(valueB)
    if not same_type:
        print('nodeA name %s , nodeB name %s'%(nodeA[0],nodeB[0]))
        print('not same type %s != %s'%(str(type(valueA)), str(type(valueB))))
        return


    if isinstance(valueA, np.ndarray):
        same_shape = valueA.shape == valueB.shape
        if not same_shape:
            print('nodeA name %s , nodeB name %s'%(nodeA[0],nodeB[0]))
            print('not same shape %s != %s'%(str(valueA.shape), str(valueB.shape)))
            return

        same_dtype = valueA.dtype == valueB.dtype
        if not same_dtype:
            print('nodeA name %s , nodeB name %s'%(nodeA[0],nodeB[0]))
            print('nodeA dtype %s , nodeB dtype %s'%(str(valueA.dtype), str(valueB.dtype)))
            return

        same_fortran_type = valueA.flags['F_CONTIGUOUS'] == valueB.flags['F_CONTIGUOUS']
        if not same_fortran_type:
            print('nodeA name %s , nodeB name %s'%(nodeA[0],nodeB[0]))
            print('nodeA fortran %s , nodeB fortran %s'%(str(valueA.flags['F_CONTIGUOUS']), str(valueB.flags['F_CONTIGUOUS'])))
            return

        same_C_type = valueA.flags['C_CONTIGUOUS'] == valueB.flags['C_CONTIGUOUS']
        if not same_C_type:
            print('nodeA name %s , nodeB name %s'%(nodeA[0],nodeB[0]))
            print('nodeA C %s , nodeB C %s'%(str(valueA.flags['C_CONTIGUOUS']), str(valueB.flags['C_CONTIGUOUS'])))
            return

        if data_comparison == 'allclose':
            same_data = np.allclose(a,b)
            if not same_data:
                print('nodeA name %s , nodeB name %s do not have same data'%(nodeA[0],nodeB[0]))
                return

        if data_comparison == 'strict':
            same_data = all(a == b)
            if not same_data:
                print('nodeA name %s , nodeB name %s do not have same data'%(nodeA[0],nodeB[0]))
                return

    else:
        same_data = valueA == valueB
        if not same_data:
            print('nodeA name %s , nodeB name %s do not have same data'%(nodeA[0],nodeB[0]))
            return
        

def nodes(nodeA, nodeB, data_comparison=None):
    names(nodeA, nodeB)
    types(nodeA, nodeB)
    values(nodeA, nodeB)
    nbChildrenA = len(nodeA[2])
    nbChildrenB = len(nodeB[2])
    if nbChildrenA != nbChildrenB:
        print('nodeA name %s , nodeB name %s has different child number'%(nodeA[0],nodeB[0]))
        exit()

    for i in range(nbChildrenA):
        childA = nodeA[2][i]
        childB = nodeB[2][i]
        nodes(childA, childB, data_comparison)




def BuildCGNSTreeFromDictElements(DictElements):
    t = I.newCGNSTree()
    Base = I.newCGNSBase('FSDMmesh', parent=t)
    for ElemName in DictElements['GroupOfElements']:
            
        SM.CreateUnstructuredZone4ElemenType(Base, DictElements, ElemName)
        AddSolutionFields2CGNSFromDictElement(Base, DictElements, ElemName)
    
    t = I.renameNode(t,'FlowSolution#Init', 'FlowSolution#Centers')
#    I.__FlowSolutionNodes__ = 'FlowSolution#Init'
#
#    for zone in I.getZones(t):
#
#        for varName in [ '%s'%x for x in DictElements['GroupOfElements'][ElemName]['FlowSolution']]:
#            C.center2Node__(zone, 'centers:'+varName, cellNType=0)
#        I._rmNodesByName(zone, 'FlowSolution#Centers')
#
#    I.__FlowSolutionNodes__ = 'FlowSolution'
    #CO.save(t, 'TestCGNS.cgns')   

    return t

#def initializeCGNSTreeFromDictElements(DictElements):
#    t = I.newCGNSTree()
#    Base = I.newCGNSBase('FSDMmesh', parent=t)
#    for ElemName in DictElements['GroupOfElements']:
#        #if (ElemName != 'Node') and (ElemName != 'Quad4'):
#            
#        SM.CreateUnstructuredZone4ElemenType(Base, DictElements, ElemName)
#        initializeZeroSolutionFields2CGNSFromDictElement(Base, DictElements, ElemName)
#        
#    return t


def initializeCGNSFromFSMesh(fsmesh):
    

    DictElements = computeDictElementsFromfsMesh(fsmesh)
    #initializeFlowSolution2DictElements(DictElements, fsmesh)
    
    #t = initializeCGNSTreeFromDictElements(DictElements)
    t = BuildCGNSTreeFromDictElements(DictElements)

    #print(DictElements)
    return t, DictElements





def TransformFSMeshData2CGNS(fsmesh):

    DictElements = computeDictElementsFromfsMesh(fsmesh)
    AddFlowSolution2DictElements(DictElements, fsmesh)

    t = BuildCGNSTreeFromDictElements(DictElements)


    return t, DictElements



def convertTree2PartialTreeWithSkeleton(t):

    t = I.copyRef(t) if I.isTopTree(t) else C.newPyTree(['Base', J.getZones(t)])
    Cmpi._convert2PartialTree(t)
    I._adaptZoneNamesForSlash(t)
    for z in I.getZones(t):
        SolverParam = I.getNodeFromName(z,'.Solver#Param')
        if not SolverParam or not I.getNodeFromName(SolverParam,'proc'):
            Cmpi._setProc(z, Cmpi.rank)
    
    Skeleton = J.getStructure(t)

    UseMerge = False
    
    trees = comm.allgather( Skeleton )
    trees.insert( 0, t )
    tWithSkel = I.merge( trees )
    CO.renameTooLongZones(tWithSkel)
    for l in 2,3: I._correctPyTree(tWithSkel,l) # unique base and zone names

    return tWithSkel




def setNonDimensionalValues2Setup(setup, AdimensionalType = 'DPT', imposeAdimensionalisation = False):
    """Computes the values of the non dimmensional quantities based on the dimmensional quantities that are present in the setup.py file """
    #print(setup.ReferenceValues)

    if AdimensionalType == 'DPT': # P, T, U adimensionalisation
            
        if (not 'NonDimensionalPressure' in setup.ReferenceValues) or (imposeAdimensionalisation):
            # NonDimensionalPressure = 1.
            setup.ReferenceValues['NonDimensionalPressureCoef'] = 1./(setup.ReferenceValues['Pressure'])
            setup.ReferenceValues['NonDimensionalPressure'] = setup.ReferenceValues['Pressure']*setup.ReferenceValues['NonDimensionalPressureCoef']
            CO.printCo(CYAN+'Computed NonDimensionalPressure'+ENDC)

        if (not 'NonDimensionalDensity' in setup.ReferenceValues) or (imposeAdimensionalisation):
            # NonDimensional Density = 1.
            setup.ReferenceValues['NonDimensionalDensityCoef'] = 1./setup.ReferenceValues['Density']
            setup.ReferenceValues['NonDimensionalDensity'] = setup.ReferenceValues['Density']*setup.ReferenceValues['NonDimensionalDensityCoef']
            CO.printCo(CYAN+'Computed NonDimensionalDensity'+ENDC)

        if (not 'NonDimensionalTemperature' in setup.ReferenceValues) or (imposeAdimensionalisation):
            # NonDimensional Temperature = 1.
            setup.ReferenceValues['NonDimensionalTemperatureCoef'] = 1./setup.ReferenceValues['Temperature']
            setup.ReferenceValues['NonDimensionalTemperature'] = setup.ReferenceValues['Temperature']*setup.ReferenceValues['NonDimensionalDensityCoef']
            CO.printCo(CYAN+'Computed NonDimensionalDensity'+ENDC)

        if (not 'NonDimensionalEnergyStagnationDensity' in setup.ReferenceValues) or (imposeAdimensionalisation):
        #    # NonDimensionalEnergyStagnationDensityCoef = rho*Et = Rho * (cv + 0.5*V**2)                                                                          
            setup.ReferenceValues['NonDimensionalEnergyStagnationDensityCoef'] = 1./(setup.ReferenceValues['Density']*setup.FluidProperties['Gamma']*setup.FluidProperties['IdealGasConstant']*setup.ReferenceValues['Temperature'])
            setup.ReferenceValues['NonDimensionalEnergyStagnationDensity'] = setup.ReferenceValues['EnergyStagnationDensity']*setup.ReferenceValues['NonDimensionalEnergyStagnationDensityCoef']
            CO.printCo(CYAN+'Computed NonDimensionalEnergyStagnationDensity'+ENDC)

        if (not 'NonDimensionalMomentumCoef' in setup.ReferenceValues) or (imposeAdimensionalisation):
            setup.ReferenceValues['NonDimensionalMomentumCoef'] = 1./(setup.ReferenceValues['Density']*np.sqrt(setup.FluidProperties['IdealGasConstant']*setup.ReferenceValues['Temperature']))
            setup.ReferenceValues['NonDimensionalMomentumX'] = setup.ReferenceValues['MomentumX']*setup.ReferenceValues['NonDimensionalMomentumCoef']
            setup.ReferenceValues['NonDimensionalMomentumY'] = setup.ReferenceValues['MomentumY']*setup.ReferenceValues['NonDimensionalMomentumCoef']
            setup.ReferenceValues['NonDimensionalMomentumZ'] = setup.ReferenceValues['MomentumZ']*setup.ReferenceValues['NonDimensionalMomentumCoef']
            CO.printCo(CYAN+'Computed NonDimensionalMomentum'+ENDC)

        #if (not 'NonDimensionalVelocityCoef' in setup.ReferenceValues) or (imposeAdimensionalisation):
        #    # NonDimensionalVelocity = Minf * sqrt(Gamma) = Vinf / sqrt(rm * Tinf)
        #    setup.ReferenceValues['NonDimensionalVelocityCoef'] = 1./(np.sqrt(setup.FluidProperties['IdealGasConstant']*setup.ReferenceValues['Temperature']))
        #    setup.ReferenceValues['NonDimensionalVelocity'] = setup.ReferenceValues['NonDimensionalVelocity']*setup.ReferenceValues['NonDimensionalVelocity']
        #    CO.printCo(CYAN+'Computed NonDimensionalEnergyStagnationDensity'+ENDC)

    # Sutherland Law in CODA:    

    if not 'SutherlandLawConstant' in setup.FluidProperties:
        setup.FluidProperties['SutherlandLawConstant'] = 110.4  # Sutherland constant for Air
        CO.printCo(CYAN+'Imposing the SutherlandLawConstant or Air: 110.4. '+ENDC)

    # Reynolds pour CODA:

    if (not 'Reynolds' in setup.ReferenceValues) or (imposeAdimensionalisation):

        setup.ReferenceValues['Reynolds'] = setup.ReferenceValues['Density'] * setup.ReferenceValues['Mach'] * np.sqrt(setup.FluidProperties['Gamma']*setup.FluidProperties['IdealGasConstant']*setup.ReferenceValues['Temperature']) * setup.ReferenceValues['Length'] / setup.ReferenceValues['ViscosityMolecular']
        print(setup.ReferenceValues['Reynolds'])
        CO.printCo(CYAN+'Computed Reynolds'+ENDC)

def dimensionalizeCGNSTree(t, DictionaryOfNonDimensionalCoefficients):
    # Dimensionalize all the values in the field: 
    for zone in I.getZones(t):
        
        NamesOfFields = [ n[0] for n in I.getNodeFromName(zone,'FlowSolution#Centers')[2] if n[3] == 'DataArray_t' ]
        for fieldName in NamesOfFields:
            var = J.getVars(zone, [fieldName], Container = 'FlowSolution#Centers')[0]
            FlowSolutionDict = {}
            FlowSolutionDict[fieldName] = var
            DimensionaVarDict = dimensionalizeFlowSolutionDict(FlowSolutionDict, DictionaryOfNonDimensionalCoefficients)
            #CO.printCo('MaxVar: %s'%np.max(abs(DimensionaVarDict[fieldName])))
            var[:] = DimensionaVarDict[fieldName]




def oneWayBodyForce(fileName):
    setup = LL.setup
    try: BodyForceInputData = setup.BodyForceInputData
    except: BodyForceInputData = None

def solveBodyForce(t, nodesInt, integrationPointsFSMesh, DictionaryOfNonDimensionalCoefficients, arrays, DictOfListIndex,discUpdatableParaDict, clac, disc, BodyForcePeriod):
    if (CO.CurrentIteration > 0) and ((CO.CurrentIteration+1)%BodyForcePeriod == 0) or (CO.CurrentIteration == 0):

        CO.printCo('Entering BODYFORCE coupling module...', proc = 0, color = CO.MAGE)
        t1 = time.time()
        
        updateCGNSfsMeshTreeWithIntegrationData(t, nodesInt, integrationPointsFSMesh)
        CO.printCo('updateCGNS time: %s s'%(time.time()-t1), proc = 0)
        
        dimensionalizeCGNSTree(t, DictionaryOfNonDimensionalCoefficients)
        t1 = time.time()
        toWithSourceTerms = computeCouplingBodyForce(t, arrays)
        CO.printCo('ComputeBF: %s s'%(time.time()-t1), proc = 0)
        

        
        CO.save(toWithSourceTerms,'treeWithSourceTerms.cgns')
        

        CO.addMemoryUsage2Arrays(arrays)
        arraysTree = CO.arraysDict2PyTree(arrays)
        CO.save(arraysTree, 'OUTPUT/arrays.cgns')
        t1 = time.time()
        integrationPointsFSMesh = setSourceTerms2FSDM(toWithSourceTerms, nodesInt,integrationPointsFSMesh, DictOfListIndex, DictionaryOfNonDimensionalCoefficients)
        CO.printCo('updateSource2FSDM time: %s s'%(time.time()-t1), proc = 0)
        t1 = time.time()
        disc.Update(discUpdatableParaDict, integrationPointsFSMesh, clac)
        CO.printCo('updateFSDMCoda time: %s s'%(time.time()-t1), proc = 0)
    
def computeCouplingBodyForce(t, arrays):
    setup = LL.setup
    # ----------------- DECLARE ADDITIONAL GLOBAL VARIABLES ----------------- #
    try: BodyForceInputData = setup.BodyForceInputData
    except: BodyForceInputData = None
    #CO.invokeCoprocessLogFile()
    #arrays = CO.invokeArrays()
    #niter    = setup.elsAkeysNumerics['niter']
    #inititer = setup.elsAkeysNumerics['inititer']
    tWithSkel = convertTree2PartialTreeWithSkeleton(t)
# ------------------------- INITIALIZE BODYFORCE -------------------------- #
    toWithSourceTerms = []
    BodyForceDisks = []
    BODYFORCE_INITIATED = False
    
    if BodyForceInputData:
        LocalBodyForceInputData = LL.getLocalBodyForceInputData(BodyForceInputData)
        LL.invokeAndAppendLocalObjectsForBodyForce(LocalBodyForceInputData)
        NumberOfSerialRuns = LL.getNumberOfSerialRuns(BodyForceInputData, NProcs)
# ------------------------------------------------------------------------- #
    
    # Control Flags for interactive control using command 'touch <flag>'
    if CO.getSignal('QUIT'): os._exit(0)
    SAVE_BODYFORCE    = CO.getSignal('SAVE_BODYFORCE')
    COMPUTE_BODYFORCE = CO.getSignal('COMPUTE_BODYFORCE')
    
    
    if CO.getSignal('RELOAD_SETUP'):
        if setup and setup.__name__ != "__main__": J.reload_source(setup)
        CO.setup = setup
        #niter    = setup.elsAkeysNumerics['niter']
        #inititer = setup.elsAkeysNumerics['inititer']
        #itmax    = inititer+niter-1 # BEWARE last iteration accessible trigger-state-16
    
        try: BodyForceInputData = setup.BodyForceInputData
        except: BodyForceInputData = None
    
        if BodyForceInputData:
            LocalBodyForceInputData = LL.getLocalBodyForceInputData(BodyForceInputData)
            LL.invokeAndAppendLocalObjectsForBodyForce(LocalBodyForceInputData)
            NumberOfSerialRuns = LL.getNumberOfSerialRuns(BodyForceInputData, NProcs)


    BodyForceSaveFrequency  = CO.getOption('BodyForceSaveFrequency', default=500)
    BodyForceComputeFrequency = CO.getOption('BodyForceComputeFrequency', default=500)
    BodyForceInitialIteration = CO.getOption('BodyForceInitialIteration', default=1000)

    # TODO!! Body force imput? 
    COMPUTE_BODYFORCE = True

    if COMPUTE_BODYFORCE:
        BODYFORCE_INITIATED = True
        Cmpi.barrier()
        CO.printCo('COMPUTING BODYFORCE', proc=0, color=CO.MAGE)
        BodyForceDisks = LL.computePropellerBodyForce(tWithSkel,
                                                      NumberOfSerialRuns,
                                                      LocalBodyForceInputData)
        
            
        CO.addBodyForcePropeller2Arrays(arrays, BodyForceDisks)
        #elsAxdt.free('xdt-runtime-tree')
        #del toWithSourceTerms
        Cmpi.barrier()
        
        toWithSourceTerms = LL.migrateSourceTerms2MainPyTree(BodyForceDisks, tWithSkel)
        CO.save(BodyForceDisks,os.path.join(DIRECTORY_OUTPUT,FILE_BODYFORCESRC))
        #SAVE_BODYFORCE = False

        CO.printCo('migrating computed source terms...', proc=0, color=CO.MAGE)
        #CO.save(toWithSourceTerms, 'tWithSource.cgns')
        

#    if SAVE_BODYFORCE:
#        CO.save(BodyForceDisks,os.path.join(DIRECTORY_OUTPUT,FILE_BODYFORCESRC))
#    if BODYFORCE_INITIATED:
#        Cmpi.barrier()
#        CO.printCo('sending source terms to elsA...', proc=0)
#        #elsAxdt.xdt(elsAxdt.PYTHON, ('xdt-runtime-tree', toWithSourceTerms, 1) )
    return toWithSourceTerms
