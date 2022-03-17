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
    

            
    