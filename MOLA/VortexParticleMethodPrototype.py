'''
MOLA - VortexParticleMethodPrototype.py 

This module proposes a set of functions for use with LiftingLine-based methods
including Airfoil Polars data management.

First creation:
24/11/2019 - L. Bernardos
'''

import sys
import os
import re
import numpy as np
from timeit import default_timer as tic

import Converter.PyTree as C
import Converter.Internal as I

from . import InternalShortcuts as J
from . import LiftingLine as LL

def _nextTimeStep(t, dt, PolarsInterpolatorsDict,InducedVelocityModel=None, PerturbationFields=None):

    '''
            GENERAL-PURPOSE ADVANCE TIMESTEP METHOD
    Updates the simulation for the next timestep, including
    motion and eventually FreestreamVelocity, InducedVelocity,
    and PerturbationVelocity.

    INPUTS
    t - (PyTree of Simulation) - Must include .Conditions node
        stating the FreestreamVelocity, air Density, etc. For
        example, <t> may be composed of one or several Propeller
        objects

    dt - (float) - current timestep length (not necessarily the
        same during the whole simulation)

    PolarsInterpolatorDict - (Python dictionary) - Dictionary 
        of polar's interpolating functions to be used by the 
        LiftingLines objects. This dictionary is generated using
        the function buildPolarsInterpolatorDict().

    InducedVelocityModel - (string) - Choose one (or none) of
        the available velocity induced models. This will be 
        implemented in future and accept: ('VPM', 'BEMT')

    PerturbationFields - (PyTree, zone, list of zones) - 
        Indicates the perturbation fields that will be added
        to the simulation at this time-step. 

    OUTPUS
    None (in-place function) - <t> is updated at next timestep
    '''
    _moveLiftingLines(t, dt) # Updates SolidVelocity
    _addPerturbationFields(t,PerturbationFields)
    if InducedVelocityModel == 'VPM':
        _computeVPM(t,dt)
    elif InducedVelocityModel == 'BEMT':
        _computeBEMTunsteady()
    _computeLocalVelocity(t)
    _updateLiftingLines(t,PolarsInterpolatorsDict) # _applyPolars, _computeLoads...

    return None

def _setVPMinfo2LiftingLine(LiftingLine, Abscissa, Law='interp1d_linear'):
    '''
    Set a .VPM#Info node into a LiftingLine. This function
    will create all required nodes in LiftingLine to allow for
    emission of vorticity particles. 

    INPUTS
    LiftingLine (PyTree zone) - Must be a LiftingLine.

    Abscissa (1D numpy array) - A monotonically increasing
        vector between 0 (root) and 1 (tip), which specifies
        the number and position of vorticity emission sources.

    InterpolationLaw (string) - A InternalShortcuts.interpolate__
        compatible Law used for mapping the vorticity sources
        from LiftingLine to VPM sources locations.

    OUTPUTS
    None (in-place function) - Updates LiftingLine with newly
        created .VPM#Info node.
    '''

    # ATTRIBUTE CHECKS
    isLL = checkComponentKind(LiftingLine,kind='LiftingLine')
    if not isLL: raise AttributeError('1st argument must be a LiftingLine object.')

    if isinstance(Abscissa,list) or isinstance(Abscissa,tuple):
        Abscissa = np.array(Abscissa,dtype=np.float64,order='F')
    elif type(Abscissa) != np.ndarray:
        raise AttributeError('2nd argument must be a vector: 1D numpy array, Python list or Python tuple')

    if np.any(np.diff(Abscissa)<0): raise AttributeError('2nd argument must be a monotonically increasing vector. Decreasing elements were detected.')

    # .VPM#Info node construction
    ZeroVec = Abscissa*0.
    J.set(LiftingLine,'.VPM#Info',Abscissa=Abscissa,gammaSourceX=ZeroVec,gammaSourceY=ZeroVec,gammaSourceZ=ZeroVec,InterpolationLaw=Law)

    return None

def buildParticlesFromLiftingLines(t,N,zonename='VortexParticles'):
    '''
    Invoke <N> particles as a single NODE PyTree zone. 
    Include .Component#Info node, with general information.

    INPUTS
    t (PyTree) - Includes LiftingLine objects where
        _setVPMinfo2LiftingLine() was applied (i.e. with existing
        '.VPM#Info' node)
    N (integer) - Maximum number of particles associated with
        all the sources. It must yield no rest on the division
        <N>/<Nsources>. If not, N is adapted.

    OUTPUTS
    ParticlesZone (PyTree Zone) - A NODE Zone with <N> invoked
        particles with all required FlowSolution fields invoked.
    '''

    # ATTRIBUTE CHECKS
    Nsources = 0
    for LiftingLine in I.getZones(t):
        # Ignore non-LiftingLine zones
        isLL = checkComponentKind(LiftingLine,kind='LiftingLine')
        if not isLL: continue

        VPMinfoDic = J.get(LiftingLine,'.VPM#Info')
        if VPMinfoDic == {}: continue
        Nsources += len(VPMinfoDic['Abscissa'])

    if Nsources == 0: raise AttributeError("No LiftingLine contained '.VPM#Info' node. Please make sure you use _setVPMinfo2LiftingLine() beforehand.")

    # Invoke particles
    NPerSrc = int(N/Nsources)
    QtyRest = int(N%Nsources)
    N -= QtyRest # Adapt for exact uniform distribution

    x = np.zeros((N),dtype=np.float64,order='F')
    y = np.zeros((N),dtype=np.float64,order='F')
    z = np.zeros((N),dtype=np.float64,order='F')

    ParticleZone = I.newZone(zonename, zsize=[[N,0,0]], ztype='Unstructured')
    GC_n = I.newGridCoordinates(parent=ParticleZone)
    I.createNode('CoordinateX','DataArray_t',value=x,parent=GC_n)
    I.createNode('CoordinateY','DataArray_t',value=y,parent=GC_n)
    I.createNode('CoordinateZ','DataArray_t',value=z,parent=GC_n)

    # Invoke required fields for ParticleZone
    RequiredFields = (
    # Specific vorticity
    'alphaX','alphaY','alphaZ',       # current timestep
    'alphaXm1','alphaYm1','alphaZm1', # previous timestep
    # Particle's lagrangian velocity
    'VelocityX','VelocityY','VelocityZ',       # current 
    'VelocityXm1','VelocityYm1','VelocityZm1', # previous
    # Stretching term
    'STX','STY','STZ',       # current 
    'STXm1','STYm1','STZm1', # previous
    # Viscous diffusion term
    'VDX','VDY','VDZ',       # current 
    'VDXm1','VDYm1','VDZm1', # previous
    # Miscellaneous
    'cellN',
    )
    SaveMemory = ('cellN')
    SmallMemory= np.int32 # smallest memory field currently supported by CGNS
    FS_n = I.newFlowSolution(parent=ParticleZone)
    for v in RequiredFields:
        dtype = np.float64 if v not in SaveMemory else SmallMemory
        I.createNode(v,'DataArray_t',value=np.zeros(N,dtype=dtype,order='F'),parent=FS_n)

    # Add .Component#Info node
    J.set(ParticleZone,'.Component#Info',kind='VortexParticles',Nsources=Nsources)

    return ParticleZone

def _activateNewParticles(t):
    '''
    Activate new set of particles at source locations of 
    LiftingLines.

    INPUTS
    t (PyTree) - Must include:
        LiftingLines (with .VPM#Info) and ParticleZones.
        A node .VPM#Params at root.

    OUTPUTS
    None (in-place function) - <t> is modified. A new set of
        particles is placed at LiftingLine's sources locations.
    '''

    # Get VPM parameters
    VPMParam = J.get(t,'.VPM#Params')
    if VPMParam == {}: raise AttributeError("Node '.VPM#Params' is missing in simulation set-up. Make sure you included VPMdata when calling prepareUnsteadyLiftingLine() function.")

    # Get ParticleZone
    AllZones = I.getZones(t)
    ParticleZone = [zn for zn in AllZones if checkComponentKind(zn,'VortexParticles')][0]
    ParticleInfoDict = J.get(ParticleZone,'.Component#Info')
    Nsources = ParticleInfoDict['Nsources']

    # Get relevant ParticlesZone pointers 
    x,y,z = J.getxyz(ParticleZone)
    cellN,aX,aY,aZ,aXm,aYm,aZm = J.getVars(ParticleZone,['cellN','alphaX','alphaY','alphaZ','alphaXm1','alphaYm1','alphaZm1']) 
    
    # Compute relevant data
    InactivePrtcIndices = np.where((cellN==0))[0]
    Ninactive = len(InactivePrtcIndices)
    if Ninactive<Nsources:
        C.convertPyTree2File(t,'bug.cgns')
        raise ValueError("FATAL ERROR: Insufficient inactive particles were detected (%d) compared to the requested number of sources (%d).\nPlease report bug using the produced 'bug.cgns' file to the developer team."%(Ninactive,Nsources))

    # Get Lifting Lines
    LLs = [zn for zn in AllZones if checkComponentKind(zn,'LiftingLine')]
    nLL = len(LLs)
    iLL    = -1 # Auxiliary counter (LiftingLine)
    isrcLL =  -1 # Auxiliary counter (source of a LiftingLine)
    NsourcesLL = 0 # is further updated

    # Activate a set of non-active particles (1 per source)
    for isrc in xrange(Nsources):
        iprt = InactivePrtcIndices[isrc]
        
        # Get current VPM - Lifting Line
        while isrcLL==NsourcesLL-1:
            iLL+=1
            LiftingLine = LLs[iLL]
            VPMinfoDic = J.get(LiftingLine,'.VPM#Info')
            if VPMinfoDic != {}:
                s,gSX,gSY,gSZ = J.getVars(LiftingLine,['s','gammaSourceX','gammaSourceY','gammaSourceZ'])
                xLL,yLL,zLL = J.getxyz(LiftingLine)

                # Compute source coordinates and vorticity sources
                NsourcesLL = len(VPMinfoDic['Abscissa'])
                xSource = J.interpolate__(VPMinfoDic['Abscissa'],s,xLL,Law=VPMinfoDic['InterpolationLaw'])
                ySource = J.interpolate__(VPMinfoDic['Abscissa'],s,yLL,Law=VPMinfoDic['InterpolationLaw'])
                zSource = J.interpolate__(VPMinfoDic['Abscissa'],s,zLL,Law=VPMinfoDic['InterpolationLaw'])
                VPMinfoDic['gammaSourceX'][:] = J.interpolate__(VPMinfoDic['Abscissa'],s,gSX,Law=VPMinfoDic['InterpolationLaw'])
                VPMinfoDic['gammaSourceY'][:] = J.interpolate__(VPMinfoDic['Abscissa'],s,gSY,Law=VPMinfoDic['InterpolationLaw'])
                VPMinfoDic['gammaSourceZ'][:] = J.interpolate__(VPMinfoDic['Abscissa'],s,gSZ,Law=VPMinfoDic['InterpolationLaw'])
                isrcLL = -1
        isrcLL+=1
        # Activate particle
        cellN[iprt] = 1
        # Put particle in source location
        x[iprt] = xSource[isrcLL]
        y[iprt] = ySource[isrcLL]
        z[iprt] = zSource[isrcLL]
        # Assign vorticity to particle
        aXm[iprt] = aX[iprt] = VPMinfoDic['gammaSourceX'][isrcLL]/VPMParam['ParticleVolume']
        aYm[iprt] = aY[iprt] = VPMinfoDic['gammaSourceY'][isrcLL]/VPMParam['ParticleVolume']
        aZm[iprt] = aZ[iprt] = VPMinfoDic['gammaSourceZ'][isrcLL]/VPMParam['ParticleVolume']

    return None

def _induceParticles(t):
    '''
    Prototype (to be extensively optimized) of the induced
    computation of quantities for the Vortex Particle Method.
    This function computes the induced quantities such as 
    specific vorticity and velocities.

    INPUTS
    t (PyTree) - Must include:
        LiftingLines (with .VPM#Info) and ParticleZones.
        A node .VPM#Params at root.

    OUTPUTS
    None (in-place function) - <t> is modified. Fields of each
        VortexParticle zone are updated
    '''
    
    # Get VPM parameters
    VPMParam = J.get(t,'.VPM#Params')
    if VPMParam == {}: raise AttributeError("Node '.VPM#Params' is missing in simulation set-up. Make sure you included VPMdata when calling prepareUnsteadyLiftingLine() function.")
    # get OverlapFactor
    try: OverlapFactor = VPMParam['OverlapFactor']
    except: raise AttributeError("Could not find 'OverlapFactor' in .VPM#Params. Check your call of prepareUnsteadyLiftingLine()")
    # get Resolution
    try: Resolution = VPMParam['Resolution']
    except: raise AttributeError("Could not find 'Resolution' in .VPM#Params. Check your call of prepareUnsteadyLiftingLine()")
    # get Volume
    try: Vol = VPMParam['ParticleVolume']
    except: raise AttributeError("Could not find 'ParticleVolume' in .VPM#Params. Check your call of prepareUnsteadyLiftingLine()")


    # Get ParticleZone
    AllZones = I.getZones(t)
    ParticleZone = [zn for zn in AllZones if checkComponentKind(zn,'VortexParticles')][0]

    # Get relevant pointers
    x,y,z = J.getxyz(ParticleZone)
    RequiredFields = (
    # Specific vorticity
    'alphaX','alphaY','alphaZ',       # current timestep
    'alphaXm1','alphaYm1','alphaZm1', # previous timestep
    # Particle's lagrangian velocity
    'VelocityX','VelocityY','VelocityZ',       # current 
    'VelocityXm1','VelocityYm1','VelocityZm1', # previous
    # Stretching term
    'STX','STY','STZ',       # current 
    'STXm1','STYm1','STZm1', # previous
    # Viscous diffusion term
    'VDX','VDY','VDZ',       # current 
    'VDXm1','VDYm1','VDZm1', # previous
    # Miscellaneous
    'cellN',
    )
    v = J.getVars2Dict(ParticleZone,RequiredFields)

    # Store previous timestep
    PrevVars = ('alphaX','alphaY','alphaZ','VelocityX','VelocityY','VelocityZ','STX','STY','STZ','VDX','VDY','VDZ')
    for var in PrevVars: v[var+'m1']=v[var]

    # Create some convenient pointers
    aX,aY,aZ,u,v,w = v['alphaX'],v['alphaY'],v['alphaZ'],v['VelocityX'],v['VelocityY'],v['VelocityZ']

    # TODO:
    '''
    magic happens here
    '''

    return None

def _computeVPM(t,dt):
    '''
    '''
    _computeLiftingLineSources(t,dt)
    _activateNewParticles(t)
    _induceParticles(t) # VPM's heart... 
    # _transportParticles(t)
    # _recycleParticles(t)

    return None