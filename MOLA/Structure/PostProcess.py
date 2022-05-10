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
        XX


    #Plot some interesting results
    
    
    v = J.getVars2Dict(LiftingLine,['VelocityMagnitudeLocal','Mach', 'Span', 'AoA' , 'Chord', 'Cl', 'Cd', 'dFx', 'dMx','VelocityAxial'])
    # this is the call to matplotlib that allows dynamic plotting
    #plt.figure(1)
    #plt.axis([0, max(v['Span'])*1.1 ,0 ,1.5])
    ax[0,0].plot(v['Span'],v['Mach'], label = '%s %s'%(NameVars, Iter))
    #plt.pause(0.05)
    ax[0,0].xlabel('Span position [m]')
    ax[0,0].ylabel('Mach [-]')
    #ax[0,0].title('Mach along the span (Iter : %s)'%Iter)
    #plt.figure(2)
    #plt.axis([0, max(v['Span'])*1.1 ,-10,10 ])
    ax[0,1].plot(v['Span'],v['AoA'], label = '%s %s'%(NameVars, Iter))
    #plt.pause(0.05)
    ax[0,1].xlabel('Span position [m]')
    ax[0,1].ylabel('Angle of attack [deg]')
    #ax[0,0].title('AoA along the span (Iter : %s)'%Iter)
    
    plt.figure(3)
    nu=1.48e-5
    Re=np.multiply(v['VelocityMagnitudeLocal'],v['Chord'])/nu
    plt.axis([0, max(v['Span'])*1.1 ,0,5e6 ])
    plt.plot(v['Span'],Re)
    plt.pause(0.05)
    plt.xlabel('Span position [m]')
    plt.ylabel('Re [-]')
    plt.title('Local Reynolds number along the span (Iter : %s)'%Iter)
    plt.figure(4)
    plt.axis([0, max(v['Span'])*1.1 ,0,100])
    plt.plot(v['Span'],v['VelocityAxial'])
    plt.pause(0.05)
    plt.xlabel('Span position [m]')
    plt.ylabel('Axial velocity [m/s]')
    plt.title('Axial velocity along the span (Iter : %s)'%Iter)
    plt.show()

            
