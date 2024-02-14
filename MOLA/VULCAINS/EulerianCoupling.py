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

import sys
import os
import scipy
import numpy as np
import VULCAINS.vulcains as V
import Converter.PyTree as C
import Geom.PyTree as D
import Converter.Internal as I
import Generator.PyTree as G
import Transform.PyTree as T
import Connector.PyTree as CX
import Post.PyTree as P
import CPlot.PyTree as CPlot

from . import FreeWakeParticles as VPM

from .. import LiftingLine as LL
from .. import Wireframe as W
from .. import InternalShortcuts as J
from .. import ExtractSurfacesProcessor as ESP

####################################################################################################
####################################################################################################
############################################## Hybrid ##############################################
####################################################################################################
####################################################################################################
def generateMirrorWing(EulerianMesh = [], VPMParameters = {}, HybridParameters = {}):
    if type(EulerianMesh) == str: tE = load(EulerianMesh)
    Zones = I.getZones(tE)
    rmNodes = ['Momentum' + v for v in 'XYZ'] + ['Density', 'TurbulentEnergyKineticDensity',
           'TurbulentDissipationRateDensity', 'ViscosityMolecular', 'Pressure', 'Mach', 'cellN',
           'EnergyStagnationDensity', 'q_criterion', 'Viscosity_EddyMolecularRatio', 'indicm']
    
    Zones_m = []
    reverse = ['VorticityX', 'VorticityZ', 'VelocityY', 'CenterY']
    for i, Zone in enumerate(Zones):
        FlowSolutionNode = I.getNodeFromName1(Zone, 'FlowSolution#Init')
        FlowSolutionNode[0] = 'FlowSolution#Centers'

        x, y, z = J.getxyz(C.node2Center(Zone))
        xc, yc, zc = J.invokeFields(Zone, ['CenterX', 'CenterY', 'CenterZ'],
          locationTag = 'centers:')
        xc[:], yc[:], zc[:] = x, y, z

        C._initVars(Zone, 'centers:Zone', i)
        C._initVars(Zone, 'centers:Index', 0)
        C._initVars(Zone, 'centers:VelocityX={MomentumX}/{Density}')
        C._initVars(Zone, 'centers:VelocityY={MomentumY}/{Density}')
        C._initVars(Zone, 'centers:VelocityZ={MomentumZ}/{Density}')
        Index = I.getNodeFromName(Zone, 'Index')
        cmpt = 0
        for l in range(len(Index[1][0][0])):
            for m in range(len(Index[1][0])):
                for n in range(len(Index[1])):
                    Index[1][n][m][l] = cmpt
                    cmpt += 1

        for name in rmNodes: I._rmNodesByName(Zone, name)
        
        FlowSolutionNode = I.getNodeFromName1(Zone, 'FlowSolution#Centers')
        x, y, z = J.getxyz(Zone)
        Zone_m = J.createZone(Zone[0] + '_m', [x, -y, z], 'xyz')
        for FlowSolution in FlowSolutionNode[2][1:]:
            FlowSolution_m = J.invokeFields(Zone_m, [FlowSolution[0]],
                                                                    locationTag = 'centers:')[0]
            if FlowSolution[0] in reverse: FlowSolution_m[:] = -FlowSolution[1]
            else: FlowSolution_m[:] = FlowSolution[1]
            if FlowSolution[0] == 'Zone': FlowSolution_m[:] += len(Zones)
        Zones_m += [Zone_m]

    MeshRadius = -1.
    for Zone in Zones:
        TurbDist = I.getNodeFromName(Zone, 'TurbulentDistance')[1]
        MeshRadius = max(MeshRadius, np.max(TurbDist))

    NumberOfHybridInterfaces = HybridParameters['NumberOfHybridInterfaces'][0]
    OuterDomainToWallDistance = HybridParameters['OuterDomainToWallDistance'][0]
    Sigma = VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]
    InnerDomain = OuterDomainToWallDistance - NumberOfHybridInterfaces*Sigma
    if InnerDomain < Sigma:
        ERRMSG = J.FAIL + ('The Hybrid Domain radius (NumberOfHybridInterfaces*Sigma = %.5f m) \
            is too close to the solid (InnerDomainToWallDistance = %.5f m < Sigma = %.5f) \
            for the selected OuterDomainToWallDistance = %.5f m. Either reduce the \
            NumberOfHybridInterfaces, the Resolution or the SmoothingRatio, or increase the \
            OuterDomainToWallDistance.')%(NumberOfHybridInterfaces*Sigma, InnerDomain, Sigma,
                                                             OuterDomainToWallDistance) + J.ENDC
        raise ValueError(ERRMSG)
    if MeshRadius <= OuterDomainToWallDistance:
        ERRMSG = J.FAIL +('The Hybrid Domain ends beyond the mesh (OuterDomainToWallDistance = \
            %.5f m). The furthest cell is %.5f m from the wall.'%(OuterDomainToWallDistance, \
                                                                           MeshRadius)) + J.ENDC
        raise ValueError(ERRMSG)

    if Zones_m: tE = C.newPyTree([Zones + Zones_m])
    else: tE = C.newPyTree([Zones])
    return I.correctPyTree(tE)

def checkEulerianField(EulerianMesh = [], VPMParameters = {}, HybridParameters = {}):
    '''
    Updates the Eulerian Tree.

    Parameters
    ----------
        t : Tree, Base
            Containes the Eulerian Field.

        VPMParameters : :py:class:`dict` of :py:class:`float`, :py:class:`int`, :py:class:`bool` and :py:class:`str`
            Containes VPM parameters for the VPM solver.

        HybridParameters : :py:class:`dict` of :py:class:`float` and :py:class:`int`
            Containes Hybrid parameters for the Hybrid solver.
    '''
    if type(EulerianMesh) == str: tE = load(EulerianMesh)
    Zones = I.getZones(tE)
    rmNodes = ['Momentum' + v for v in 'XYZ'] + ['Density', 'TurbulentEnergyKineticDensity',
           'TurbulentDissipationRateDensity', 'ViscosityMolecular', 'Pressure', 'Mach', 'cellN',
           'EnergyStagnationDensity', 'q_criterion', 'Viscosity_EddyMolecularRatio', 'indicm']
    
    reverse = ['VorticityX', 'VorticityZ', 'VelocityY', 'CenterY']
    for i, Zone in enumerate(Zones):
        FlowSolutionNode = I.getNodeFromName1(Zone, 'FlowSolution#Init')
        FlowSolutionNode[0] = 'FlowSolution#Centers'

        x, y, z = J.getxyz(C.node2Center(Zone))
        xc, yc, zc = J.invokeFields(Zone, ['CenterX', 'CenterY', 'CenterZ'],
          locationTag = 'centers:')
        xc[:], yc[:], zc[:] = x, y, z

        C._initVars(Zone, 'centers:VelocityX={MomentumX}/{Density}')
        C._initVars(Zone, 'centers:VelocityY={MomentumY}/{Density}')
        C._initVars(Zone, 'centers:VelocityZ={MomentumZ}/{Density}')

        for name in rmNodes: I._rmNodesByName(Zone, name)

    MeshRadius = -1.
    for Zone in Zones:
        TurbDist = I.getNodeFromName(Zone, 'TurbulentDistance')[1]
        MeshRadius = max(MeshRadius, np.max(TurbDist))

    NumberOfHybridInterfaces = HybridParameters['NumberOfHybridInterfaces'][0]
    OuterDomainToWallDistance = HybridParameters['OuterDomainToWallDistance'][0]
    Sigma = VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]
    InnerDomain = OuterDomainToWallDistance - NumberOfHybridInterfaces*Sigma
    if InnerDomain < Sigma:
        ERRMSG = J.FAIL + ('The Hybrid Domain radius (NumberOfHybridInterfaces*Sigma = %.5f m) \
            is too close to the solid (InnerDomainToWallDistance = %.5f m < Sigma = %.5f) \
            for the selected OuterDomainToWallDistance = %.5f m. Either reduce the \
            NumberOfHybridInterfaces, the Resolution or the SmoothingRatio, or increase the \
            OuterDomainToWallDistance.')%(NumberOfHybridInterfaces*Sigma, InnerDomain, Sigma,
                                                             OuterDomainToWallDistance) + J.ENDC
        raise ValueError(ERRMSG)
    if MeshRadius <= OuterDomainToWallDistance:
        ERRMSG = J.FAIL +('The Hybrid Domain ends beyond the mesh (OuterDomainToWallDistance = \
            %.5f m). The furthest cell is %.5f m from the wall.'%(OuterDomainToWallDistance, \
                                                                           MeshRadius)) + J.ENDC
        raise ValueError(ERRMSG)

    return I.correctPyTree(C.newPyTree([Zones]))   

def generateHybridDomainInterfaces(tE = [], VPMParameters = {}, HybridParameters = {}):
    '''
    Gives the solid boundary for the BEM and the Inner and Outer Interface of the Hybrid Domain.

    Parameters
    ----------
        t : Tree, Base
            Containes the Eulerian Field.

        VPMParameters : :py:class:`dict` of :py:class:`float`, :py:class:`int`, :py:class:`bool` and :py:class:`str`
            Containes VPM parameters for the VPM solver.

        HybridParameters : :py:class:`dict` of :py:class:`float` and :py:class:`int`
            Containes Hybrid parameters for the Hybrid solver.
    '''
    print(f"{'||':>57}\r" + '||'+'{:-^53}'.format(' Generate Hybrid Interfaces '))
    OuterDomainToWallDistance = HybridParameters['OuterDomainToWallDistance'][0]
    h = VPMParameters['Resolution'][0]
    InterfaceGap = h*VPMParameters['SmoothingRatio'][0]
    NumberOfHybridInterfaces = HybridParameters['NumberOfHybridInterfaces'][0]
    InnerDomainToWallDistance = OuterDomainToWallDistance - InterfaceGap*\
                                                                        NumberOfHybridInterfaces
    HybridParameters['InnerDomainToWallDistance'] = np.array([InnerDomainToWallDistance],
                                                                order = 'F', dtype = np.float64)
    tE_joint = T.join(C.convertArray2Hexa(tE))
    remove = ['GridLocation', '.Solver#Param', '.MOLA#Offset', '.MOLA#Trim', 'FamilyName',
                         'AdditionalFamilyName', 'ZoneGridConnectivity', 'FlowSolution#Centers']
    Interfaces = []
    names = ['OuterInterface', 'InnerInterface', 'BEMInterface']
    for d in [OuterDomainToWallDistance, InnerDomainToWallDistance, h/100.]:
        Zones = I.getZones(P.isoSurfMC(tE_joint, 'TurbulentDistance', d))
        for Zone in Zones:
            for rm in remove: I._rmNodesByName(Zone, rm)

        Interface = T.join(C.convertArray2Hexa(C.newPyTree([Zones])))
        Interface[0] = names.pop(0)
        G._getNormalMap(Interface)
        Interface = C.center2Node(Interface, ['centers:sx', 'centers:sy', 'centers:sz'])
        I._rmNodesByName(Interface, 'FlowSolution#Centers')
        Interfaces += [Interface]

    msg =  f"{'||':>57}\r" + '|| ' + '{:27}'.format('Outer Interface distance') + ': ' + \
                                               '{:.4f}'.format(OuterDomainToWallDistance) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:27}'.format('Inner Interface distance')     + ': ' + \
                                               '{:.4f}'.format(InnerDomainToWallDistance) + '\n'
    msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Done ')
    print(msg)
    return Interfaces

def getRegularGridInHybridDomain(OuterInterface = [], InnerInterface = [], Resolution = 0.):
    '''
    Creates a cartesian grid contained between two closed surfaces.

    Parameters
    ----------
        OuterInterface : Zone
            Outer surface.

        InnerInterface : Zone
            Inner surface.

        Resolution : :py:class:`float`
            Grid resolution.

    Returns
    ----------
        Grid : Zone
            Nodes of the cartesian grid.
    '''
    bbox = np.array(G.bbox(OuterInterface))
    Ni = int(np.ceil((bbox[3] - bbox[0])/Resolution)) + 4
    Nj = int(np.ceil((bbox[4] - bbox[1])/Resolution)) + 4
    Nk = int(np.ceil((bbox[5] - bbox[2])/Resolution)) + 4
    cart = G.cart(np.ceil(bbox[:3]/Resolution)*Resolution - 2*Resolution, (Resolution,
                                                          Resolution, Resolution), (Ni, Nj, Nk))
    t_cart = C.newPyTree(['Base', cart])
    maskInnerSurface = CX.blankCells(t_cart, [[InnerInterface]], np.array([[1]]),
                        blankingType = 'center_in', delta = 0, dim = 3, tol = 0.)
    maskOuterSurface = CX.blankCells(t_cart, [[OuterInterface]], np.array([[1]]),
                        blankingType = 'center_in', delta = Resolution, dim = 3, tol = 0.)
    maskInnerSurface  = C.node2Center(I.getZones(maskInnerSurface)[0])
    maskOuterSurface = C.node2Center(I.getZones(maskOuterSurface)[0])
    cellInnerSurface = J.getVars(maskInnerSurface, ['cellN'], 'FlowSolution')[0]
    cellOuterSurface = J.getVars(maskOuterSurface, ['cellN'], 'FlowSolution')[0]
    inside = (cellOuterSurface == 0)*(cellInnerSurface == 1)
    x, y, z = J.getxyz(maskOuterSurface)
    return C.convertArray2Node(J.createZone('Grid', [x[inside], y[inside], z[inside]], 'xyz'))

def findDonorsIndex(Mesh = [], Target = []):
    '''
    Gives the indexes of the closests cell-centers of a mesh from a set of user-given node
    targets.

    Parameters
    ----------
        Mesh : Tree, Base or Zone
            Donor mesh.

        Target : Base, Zone or list of Zone
            Target nodes.

    Returns
    ----------
        donors : numpy.ndarray
            Cell-centers closest neighbours indexes.

        receivers : numpy.ndarray
            Node targets indexes.

        unique : numpy.ndarray
            Flag list of the unique donors.
    '''
    x, y, z = J.getxyz(Target)
    t_centers = C.node2Center(Mesh)
    zones_centers = I.getZones(t_centers)
    hook, indir = C.createGlobalHook(t_centers, function = 'nodes', indir = 1)
    nodes, dist = C.nearestNodes(hook, J.createZone('Zone', [x, y, z], 'xyz'))
    nodes, unique = np.unique(nodes, return_index = True)
    x = x[unique]

    cumul = 0
    cumulated = []
    for z in zones_centers:
        cumulated += [cumul]
        cumul += C.getNPts(z)

    receivers = []
    donors = []
    for p in range(len(x)):
        ind = nodes[p] - 1
        closest_index = ind - cumulated[indir[ind]]
        receivers += [indir[ind]]
        donors += [closest_index]

    donors = np.array(donors, order = 'F', dtype = np.int32)
    receivers = np.array(receivers, order = 'F', dtype = np.int32)
    return donors, receivers, unique

def findDonorFields(Mesh = [], donors = [], receivers = [], FieldNames = []):
    '''
    Gets the user-given fields from the donor mesh onto the receiver nodes.

    Parameters
    ----------
        Mesh : Tree, Base or Zone
            Donor mesh.

        donors : numpy.ndarray
            Cell-centers donor indexes.

        receivers : numpy.ndarray
            Node targets indexes.

        FieldNames : :py:class:`list` or numpy.ndarray of :py:class:`str`
            Names of the fields to retreive.

    Returns
    ----------
        Fields : :py:class:`dict` of numpy.ndarray
            Extracted fields from the Mesh.
    '''
    Nh = len(donors)
    Fields = {}
    for name in FieldNames:
        Fields[name] = np.zeros(Nh, dtype = np.float64, order = 'F')

    for iz, zone in enumerate(I.getZones(Mesh)):
        receiver_slice = receivers == iz
        DonorFields = J.getVars(zone, FieldNames, 'FlowSolution#Centers')

        donor_slice = donors[receiver_slice]
        for n, name in enumerate(FieldNames):
            Fields[name][receiver_slice] = np.ravel(DonorFields[n], order = 'F')[donor_slice]

    return Fields

def setHybridDonors(Mesh = [], Interfaces = [], VPMParameters = {}, HybridParameters = {}):
    '''
    Sets the donors from the Eulerian Mesh from where the hybrid particles are generated.

    Parameters
    ----------
        Mesh : Tree, Base or Zone
            Donor mesh.

        Interfaces : :py:class:`list` or numpy.ndarray of Zone
            Containes the Inner and Outer Interfaces delimiting the Hybrid Domain.

        VPMParameters : :py:class:`dict` of :py:class:`float`, :py:class:`int`, :py:class:`bool` and :py:class:`str`
            Containes VPM parameters for the VPM solver.

        HybridParameters : :py:class:`dict` of :py:class:`float` and :py:class:`int`
            Containes Hybrid parameters for the Hybrid solver.
    '''
    print(f"{'||':>57}\r" + '||'+'{:-^53}'.format(' Generate Hybrid Sources '))
    Sigma0 = VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]
    Grid = getRegularGridInHybridDomain(Interfaces[0], Interfaces[-1], Sigma0)
    donors, receivers, unique = findDonorsIndex(Mesh, Grid)
    Fields = findDonorFields(Mesh, donors, receivers, ['Center' + v for v in 'XYZ'] + \
                                                                          ['TurbulentDistance'])
    OuterDomainToWallDistance = HybridParameters['OuterDomainToWallDistance'][0]
    NumberOfHybridInterfaces = HybridParameters['NumberOfHybridInterfaces'][0]
    InnerDomainToWallDistance = OuterDomainToWallDistance - Sigma0*NumberOfHybridInterfaces
    flags = []
    for n in range(NumberOfHybridInterfaces + 1):
        d = OuterDomainToWallDistance - n*Sigma0
        flags += [Fields['TurbulentDistance'] <= d]#flags everything inside current interface

    Domain_Flag = flags[0]*np.logical_not(flags[-1])#Particles inside OuterInterface and outside InnerInterface
    Resolution = findMinimumDistanceBetweenParticles(Fields['CenterX'][Domain_Flag],
                                 Fields['CenterY'][Domain_Flag], Fields['CenterZ'][Domain_Flag])
    while np.min(Resolution) < Sigma0/2.:
        CurrentFlag = Domain_Flag[Domain_Flag]
        hmin = np.unique(Resolution[Resolution < Sigma0/2.])#get one sample of each h that is too small
        for i in range(len(hmin)):
            CurrentFlag[np.min(np.where(Resolution == hmin[i]))] = False#get rid of one particle for each h that is too small. that is because if one h is too small, then at least one other has the same value. when the 1st h is taken out, the others will increase.

        Domain_Flag[Domain_Flag] = CurrentFlag
        Resolution = findMinimumDistanceBetweenParticles(Fields['CenterX'][Domain_Flag],
                                 Fields['CenterY'][Domain_Flag], Fields['CenterZ'][Domain_Flag])

    InterfacesFlags = {}
    for i in range(len(flags) - 1):
        InterfacesFlags['Interface_' + str(i)] = np.array(flags[i][Domain_Flag]*\
                       np.logical_not(flags[i + 1][Domain_Flag]), dtype = np.int32, order = 'F')#seperates the particles for each interface

    VPMParameters['NumberOfHybridSources']    = np.array([0], order = 'F', dtype = np.int32)
    HybridParameters['HybridDonors']          = np.array(donors[Domain_Flag],    order = 'F',
                                                                               dtype = np.int32)
    HybridParameters['HybridReceivers']       = np.array(receivers[Domain_Flag], order = 'F',
                                                                               dtype = np.int32)
    HybridParameters['ParticleSeparationPerInterface'] = InterfacesFlags
    HybridParameters['HybridSigma'] = np.array(Resolution, order = 'F', dtype = np.float64)

    msg = f"{'||':>57}\r" + '|| ' + '{:27}'.format('Number of Hybrid sources') + ': ' + \
                                                           '{:d}'.format(len(Resolution)) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:27}'.format('Targeted Particle spacing') + ': ' + '{:.4f}'.format(\
                     VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:27}'.format('Mean Particle spacing')     + ': ' + \
                                                   '{:.4f}'.format(np.mean(Resolution)) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' +'{:27}'.format('Particle spacing deviation') + ': ' + \
                                                    '{:.4f}'.format(np.std(Resolution)) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:27}'.format('Maximum Particle spacing')  + ': ' + \
                                                    '{:.4f}'.format(np.max(Resolution)) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:27}'.format('Minimum Particle spacing')  + ': ' + \
                                                    '{:.4f}'.format(np.min(Resolution)) + ' m\n'
    msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Done ')
    print(msg)

def setBEMDonors(Mesh = [], Interface = [], VPMParameters = {}, HybridParameters = {}):
    '''
    Sets the donors from the Eulerian Mesh from where the bem particles are generated.

    Parameters
    ----------
        Mesh : Tree, Base or Zone
            Donor mesh.

        Interfaces : Zone
            Containes the BEM Interface surrounding the solid.

        VPMParameters : :py:class:`dict` of :py:class:`float`, :py:class:`int`, :py:class:`bool` and :py:class:`str`
            Containes VPM parameters for the VPM solver.

        HybridParameters : :py:class:`dict` of :py:class:`float` and :py:class:`int`
            Containes Hybrid parameters for the Hybrid solver.
    '''
    print(f"{'||':>57}\r" + '||'+'{:-^53}'.format(' Generate BEM Panels '))
    Sigma0 = VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]
    donors, receivers, unique = findDonorsIndex(Mesh, Interface)
    Fields = findDonorFields(Mesh, donors, receivers, ['Center' + v for v in 'XYZ'])
    sx, sy, sz = J.getVars(Interface, ['s' + v for v in 'xyz'])
    sx = sx[unique]
    sy = sy[unique]
    sz = sz[unique]
    Zone = C.convertArray2Node(J.createZone('Zone', [Fields['CenterX'], Fields['CenterY'], \
                                                                     Fields['CenterZ']], 'xyz'))
    surf = V.find_panel_clusters(Zone, VPMParameters['Resolution'][0])
    flag = surf != 0.

    Resolution = findMinimumDistanceBetweenParticles(Fields['CenterX'][flag], \
                                               Fields['CenterY'][flag], Fields['CenterZ'][flag])
    while np.min(Resolution) < Sigma0/2.:
        CurrentFlag = flag[flag]
        hmin = np.unique(Resolution[Resolution < Sigma0/2.])#get one sample of each h that is too small
        for i in range(len(hmin)):
            CurrentFlag[np.min(np.where(Resolution == hmin[i]))] = False#get rid of one particle for each h that is too small. that is because if one h is too small, then at least one other has the same value. when the 1st h is taken out, the others will increase.
        
        flag[flag] = CurrentFlag
        Resolution = findMinimumDistanceBetweenParticles(Fields['CenterX'][flag], \
                                               Fields['CenterY'][flag], Fields['CenterZ'][flag])

    sx = sx[flag]
    sy = sy[flag]
    sz = sz[flag]
    s = np.linalg.norm(np.vstack([sx, sy, sz]), axis = 0)
    nx = -sx/s
    ny = -sy/s
    nz = -sz/s
    #t1 = ez vec n 
    t1x = -ny
    t1y = nx
    t1z = 0.*nz
    t1 = np.linalg.norm(np.vstack([t1x, t1y, t1z]), axis = 0)
    #t2 = n vec t1
    t2x = ny*t1z - nz*t1y
    t2y = nz*t1x - nx*t1z
    t2z = nx*t1y - ny*t1x
    t2 = np.linalg.norm(np.vstack([t2x, t2y, t2z]), axis = 0)

    HybridParameters['NormalBEMX'] = np.array(nx, dtype = np.float64, order = 'F')
    HybridParameters['NormalBEMY'] = np.array(ny, dtype = np.float64, order = 'F')
    HybridParameters['NormalBEMZ'] = np.array(nz, dtype = np.float64, order = 'F')
    HybridParameters['Tangential1BEMX'] = np.array(t1x/t1, dtype = np.float64, order = 'F')
    HybridParameters['Tangential1BEMY'] = np.array(t1y/t1, dtype = np.float64, order = 'F')
    HybridParameters['Tangential1BEMZ'] = np.array(t1z/t1, dtype = np.float64, order = 'F')
    HybridParameters['Tangential2BEMX'] = np.array(t2x/t2, dtype = np.float64, order = 'F')
    HybridParameters['Tangential2BEMY'] = np.array(t2y/t2, dtype = np.float64, order = 'F')
    HybridParameters['Tangential2BEMZ'] = np.array(t2z/t2, dtype = np.float64, order = 'F')
    VPMParameters['NumberOfBEMSources'] = np.array([len(nx)], dtype = np.int32, order = 'F')
    HybridParameters['BEMDonors']       = np.array(donors[flag], order = 'F', dtype = np.int32)
    HybridParameters['BEMReceivers']  = np.array(receivers[flag], order = 'F', dtype = np.int32)
    HybridParameters['SurfaceBEM']    = np.array(Resolution*Resolution, order = 'F',
                                                                             dtype = np.float64)
    msg =  f"{'||':>57}\r" + '|| ' + '{:27}'.format('Number of BEM panels')     + ': '+ '{:d}'.format(len(nx))+'\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:27}'.format('Targeted Particle spacing') + ': '+'{:.4f}'.format(\
                     VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:27}'.format('Mean Particle spacing')     + ': ' + \
                                                   '{:.4f}'.format(np.mean(Resolution)) + ' m\n'
    msg += f"{'||':>57}\r" + '||' +'{:27}'.format('Particle spacing deviation') + ': ' + \
                                                    '{:.4f}'.format(np.std(Resolution)) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:27}'.format('Maximum Particle spacing')  + ': ' + \
                                                    '{:.4f}'.format(np.max(Resolution)) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:27}'.format('Minimum Particle spacing')  + ': ' + \
                                                    '{:.4f}'.format(np.min(Resolution)) + ' m\n'
    msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Done ')
    print(msg)

def setCFDDonors(Mesh = [], Interface = [], VPMParameters = {}, HybridParameters = {}):
    '''
    Sets the donors from the Eulerian Mesh from where the interface particles are generated.

    Parameters
    ----------
        Mesh : Tree, Base or Zone
            Donor mesh.

        Interfaces : Zone
            Containes the Inner Interface delimiting the Hybrid Domain.

        VPMParameters : :py:class:`dict` of :py:class:`float`, :py:class:`int`, :py:class:`bool` and :py:class:`str`
            Containes VPM parameters for the VPM solver.

        HybridParameters : :py:class:`dict` of :py:class:`float` and :py:class:`int`
            Containes Hybrid parameters for the Hybrid solver.
    '''
    print(f"{'||':>57}\r" + '||'+'{:-^53}'.format(' Generate Eulerian Panels '))
    Sigma0 = VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]
    donors, receivers, unique = findDonorsIndex(Mesh, Interface)
    Fields = findDonorFields(Mesh, donors, receivers, ['Center' + v for v in 'XYZ'])
    sx, sy, sz = J.getVars(Interface, ['s' + v for v in 'xyz'])
    sx = sx[unique]
    sy = sy[unique]
    sz = sz[unique]
    Zone = C.convertArray2Node(J.createZone('Zone', [Fields['CenterX'], Fields['CenterY'], \
                                                                     Fields['CenterZ']], 'xyz'))
    surf = V.find_panel_clusters(Zone, VPMParameters['Resolution'][0])
    flag = surf != 0.

    Resolution = findMinimumDistanceBetweenParticles(Fields['CenterX'][flag],
                                               Fields['CenterY'][flag], Fields['CenterZ'][flag])
    while np.min(Resolution) < Sigma0/2.:
        CurrentFlag = flag[flag]
        hmin = np.unique(Resolution[Resolution < Sigma0/2.])#get one sample of each h that is too small
        for i in range(len(hmin)):
            CurrentFlag[np.min(np.where(Resolution == hmin[i]))] = False#get rid of one particle for each h that is too small. that is because if one h is too small, then at least one other has the same value. when the 1st h is taken out, the others will increase.
        
        flag[flag] = CurrentFlag
        Resolution = findMinimumDistanceBetweenParticles(Fields['CenterX'][flag], \
                                               Fields['CenterY'][flag], Fields['CenterZ'][flag])

    sx = -sx[flag]
    sy = -sy[flag]
    sz = -sz[flag]

    surf = Resolution*Resolution/np.linalg.norm(np.vstack([sx, sy, sz]), axis = 0)
    HybridParameters['SurfaceCFDX']  = np.array(sx*surf, dtype = np.float64, order = 'F')
    HybridParameters['SurfaceCFDY']  = np.array(sy*surf, dtype = np.float64, order = 'F')
    HybridParameters['SurfaceCFDZ']  = np.array(sz*surf, dtype = np.float64, order = 'F')
    HybridParameters['CFDDonors']    = np.array(donors[flag], order = 'F', dtype = np.int32)
    HybridParameters['CFDReceivers'] = np.array(receivers[flag], order = 'F', dtype = np.int32)
    VPMParameters['NumberOfCFDSources'] = np.array([len(sx)], dtype = np.int32, order = 'F')
    msg  = f"{'||':>57}\r" + '|| ' + '{:27}'.format('Number of CFD panels') + ': '+ '{:d}'.format(len(sx)) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:27}'.format('Targeted Particle spacing') + ': '+'{:.4f}'.format(\
                     VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:27}'.format('Mean Particle spacing')     + ': ' + \
                                                   '{:.4f}'.format(np.mean(Resolution)) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' +'{:27}'.format('Particle spacing deviation') + ': ' + \
                                                    '{:.4f}'.format(np.std(Resolution)) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:27}'.format('Maximum Particle spacing')  + ': ' + \
                                                    '{:.4f}'.format(np.max(Resolution)) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:27}'.format('Minimum Particle spacing')  + ': ' + \
                                                    '{:.4f}'.format(np.min(Resolution)) + ' m\n'
    msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Done ')
    print(msg)

def generateHybridDomain(tE = [], VPMParameters = {}, HybridParameters = {}):
    '''
    Sets the donors from the Eulerian Mesh and generates the BEM, Inner and Outer Interfaces of
    the Hybrid Domain.

    Parameters
    ----------
        tE : Tree, Base or Zone
            Eulerian Field.

        VPMParameters : :py:class:`dict` of :py:class:`float`, :py:class:`int`, :py:class:`bool` and :py:class:`str`
            Containes VPM parameters for the VPM solver.

        HybridParameters : :py:class:`dict` of :py:class:`float` and :py:class:`int`
            Containes Hybrid parameters for the Hybrid solver.

    Returns
    ----------
        HybridDomain : Tree
            BEM, Inner and Outer Interfaces of the Hybrid Domain.
    '''
    Interfaces = generateHybridDomainInterfaces(tE, VPMParameters, HybridParameters)
    setBEMDonors(tE, Interfaces[-1], VPMParameters, HybridParameters)
    setCFDDonors(tE, Interfaces[-2], VPMParameters, HybridParameters)
    setHybridDonors(tE, Interfaces, VPMParameters, HybridParameters)
    HybridParameters['BEMMatrix'] = np.array([0.]*HybridParameters['NumberOfBEMUnknown'][0]*\
                  VPMParameters['NumberOfBEMSources'][0]*VPMParameters['NumberOfBEMSources'][0],
                                                                dtype = np.float64, order = 'F')
    return C.newPyTree(['HybridDomain', Interfaces])

def pickHybridDomain(t = []):
    '''
    Gets the Hybrid Domain Tree.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone named 'HybridDomain'.

    Returns
    ----------
        HybridDomain : Tree
            BEM, Inner and Outer Interfaces of the Hybrid Domain.
    '''
    return I.getNodeFromName1(t, 'HybridDomain')

def pickHybridDomainOuterInterface(t = []):
    '''
    Gets the Outer Hybrid Domain Interface.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone named 'HybridDomain'.

    Returns
    ----------
        OuterInterface : Zone
            Outer Interface of the Hybrid Domain.
    '''
    HybridDomain = pickHybridDomain(t)
    if HybridDomain: return I.getZones(HybridDomain)[0]

    return []

def pickHybridDomainInnerInterface(t = []):
    '''
    Gets the Inner Hybrid Domain Interface.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone named 'HybridDomain'.

    Returns
    ----------
        InnerInterface : Zone
            Inner Interface of the Hybrid Domain.
    '''
    HybridDomain = pickHybridDomain(t)
    if HybridDomain: return I.getZones(HybridDomain)[-2]

    return []

def pickBEMInterface(t = []):
    '''
    Gets the BEM Hybrid Domain Interface.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone named 'HybridDomain'.

    Returns
    ----------
        BEMInterface : Zone
            BEM Interface of the Hybrid Domain.
    '''
    HybridDomain = pickHybridDomain(t)
    if HybridDomain: return I.getZones(HybridDomain)[-1]

    return []

def getHybridParameters(t = []):
    '''
    Get a the Hybrid parameters.

    Parameters
    ----------
        t : Tree, Base, Zone.
            Containes the Hybrid parameters named '.Hybrid#Parameters'.

    Returns
    -------
        HybridParameter : :py:class:`dict`
            Hybrid parameters.
    '''
    return J.get(VPM.pickParticlesZone(t), '.Hybrid#Parameters')

def findMinHybridParticleVorticityToShed(Vorticity = [], VorticityPercentageToShed = 0.):
    '''
    Get the minimum vorticity to generate so that the total generated vorticity is the
    user-given percentage of the total vorticity.

    Parameters
    ----------
        Vorticity : :py:class:`list` or numpy.ndarray
            List of the vorticity at each node.

        VorticityPercentageToShed : :py:class:`float`
            Percentage of vorticity to generate.

    Returns
    -------
        MinVorticity : :py:class:`float`
            Minimum vorticity to generate.
    '''
    Vorticity = np.sort(Vorticity)
    Vorticitysum = [Vorticity[-1]]
    for i in range(len(Vorticity) - 2, -1, -1):Vorticitysum += [Vorticity[i] + Vorticitysum[-1]]

    Nshed = np.argmin(np.abs(np.array(Vorticitysum) - \
                                               Vorticitysum[-1]*VorticityPercentageToShed/100.))#keep the Nshed strongest
    return Vorticity[-min(Nshed + 2, len(Vorticity))]

def initialiseHybridParticles(tL = [], tE = [], VPMParameters = {}, HybridParameters = {},
    Offset = 0):
    '''
    Updates the Hybrid Domain to generate bem, interface and hybrid particles and initialise
    them.

    Parameters
    ----------
        tL : Tree, Base, Zone or list of Zone
            Lagrangian Field.

        tE : Tree, Base or Zone
            Eulerian Field.

        VPMParameters : :py:class:`dict` of :py:class:`float`, :py:class:`int`, :py:class:`bool` and :py:class:`str`
            Containes VPM parameters for the VPM solver.

        HybridParameters : :py:class:`dict` of :py:class:`float` and :py:class:`int`
            Containes Hybrid parameters for the Hybrid solver.

        Offset : :py:class:`int`
            Position from where the particles are generated.
    '''
    Ramp = np.sin(min(1./VPMParameters['StrengthRampAtbeginning'][0], 1.)*np.pi/2.)
    #First BEM particles
    Fields = findDonorFields(tE, HybridParameters['BEMDonors'],
                          HybridParameters['BEMReceivers'], ['Center' + v for v in 'XYZ'] + \
                                                                ['Velocity' + v for v in 'XYZ'])
    s  = HybridParameters['SurfaceBEM']
    nx = HybridParameters['NormalBEMX']
    ny = HybridParameters['NormalBEMY']
    nz = HybridParameters['NormalBEMZ']
    tx = HybridParameters['Tangential1BEMX'] + HybridParameters['Tangential2BEMX']
    ty = HybridParameters['Tangential1BEMY'] + HybridParameters['Tangential2BEMY']
    tz = HybridParameters['Tangential1BEMZ'] + HybridParameters['Tangential2BEMZ']        
    at = (Fields['VelocityX']*tx + Fields['VelocityY']*ty + Fields['VelocityZ']*tz)*s*Ramp
    addParticlesToTree(tL, Fields['CenterX'], Fields['CenterY'], Fields['CenterZ'], at*tx, \
                                                               at*ty, at*tz, np.sqrt(s), Offset)
    HybridParameters['AlphaBEMN'] = (nx*Fields['VelocityX'] + ny*Fields['VelocityY'] + \
                                                               nz*Fields['VelocityZ'])*s*Ramp
    Nbem = len(Fields['CenterX'])
    #Second CFD particles
    Fields = findDonorFields(tE, HybridParameters['CFDDonors'],
                          HybridParameters['CFDReceivers'], ['Center' + v for v in 'XYZ'] + \
                                                                ['Velocity' + v for v in 'XYZ'])
    sx = HybridParameters['SurfaceCFDX']
    sy = HybridParameters['SurfaceCFDY']
    sz = HybridParameters['SurfaceCFDZ']
    s  = np.linalg.norm(np.vstack([sx, sy, sz]), axis = 0)
    addParticlesToTree(tL, Fields['CenterX'], Fields['CenterY'], Fields['CenterZ'], \
              (Fields['VelocityY']*sz - Fields['VelocityZ']*sy)*Ramp, \
              (Fields['VelocityZ']*sx - Fields['VelocityX']*sz)*Ramp, \
              (Fields['VelocityX']*sy - Fields['VelocityY']*sx)*Ramp, np.sqrt(s), Offset + Nbem)
    HybridParameters['AlphaCFDN'] = (Fields['VelocityX']*sx + Fields['VelocityY']*sy + \
                                                                    Fields['VelocityZ']*sz)*Ramp
    Ncfd = len(Fields['CenterX'])
    #Last Hybrid particles
    Fields = findDonorFields(tE, HybridParameters['HybridDonors'],
                          HybridParameters['HybridReceivers'], ['Center' + v for v in 'XYZ'] + \
                                                               ['Vorticity' + v for v in 'XYZ'])
    wp = np.linalg.norm(np.vstack([Fields['VorticityX'], Fields['VorticityY'], \
                                                               Fields['VorticityZ']]), axis = 0)
    wtot = np.sum(wp)#total particle vorticity in the Hybrid Domain
    ParticleSeparationPerInterface = HybridParameters['ParticleSeparationPerInterface']
    VorticityFactor = HybridParameters['FactorOfCFDVorticityShed'][0]
    hybrid = np.array([False]*len(ParticleSeparationPerInterface['Interface_0']))
    if 0. < VorticityFactor:
        for i, InterfaceFlag in enumerate(ParticleSeparationPerInterface):
            flag = (ParticleSeparationPerInterface[InterfaceFlag] == 1)
            wi = wp[flag]
            if len(wi):
                wmin = findMinHybridParticleVorticityToShed(wi, VorticityFactor)
                flag[flag] = wmin < wi#only the strongest will survive
                hybrid += np.array(flag)
    else:
        Ni = HybridParameters['NumberOfParticlesPerInterface'][0]
        for i, InterfaceFlag in enumerate(ParticleSeparationPerInterface):
            flag = (ParticleSeparationPerInterface[InterfaceFlag] == 1)
            wi = wp[flag]
            Ncurrent = min(max(Ni, 1), len(wi))
            wmin = np.sort(wi)[-Ncurrent]#vorticity cutoff
            flagi = wmin < wi
            Nflagi = np.sum(flagi)
            if Ncurrent != Nflagi:#if several cells have the same vort and it happens to be wmin, Ncurrent will be below Ni, so i need to add somme cells until Ncurrent = Ni
                allwmin = np.where(wmin == wi)[0]
                flagi[allwmin[: Ncurrent - Nflagi]] = True

            flag[flag] = flagi#only the strongest will survive
            hybrid += np.array(flag)

    Nh = np.sum(hybrid)
    sigma = HybridParameters['HybridSigma']
    s3 = sigma*sigma*sigma
    apx = Fields['VorticityX']*s3*Ramp
    apy = Fields['VorticityY']*s3*Ramp
    apz = Fields['VorticityZ']*s3*Ramp
    addParticlesToTree(tL, Fields['CenterX'][hybrid], Fields['CenterY'][hybrid], \
                                            Fields['CenterZ'][hybrid], apx[hybrid], apy[hybrid],
                                               apz[hybrid], sigma[hybrid], Offset + Nbem + Ncfd)
    HybridParameters['AlphaHybridX'] = apx
    HybridParameters['AlphaHybridY'] = apy
    HybridParameters['AlphaHybridZ'] = apz
    VPMParameters['NumberOfHybridSources'][0] = Nh

def flagNodesInsideSurface(X = [], Y = [], Z = [], Surface = []):
    '''
    Gives the particles inside the user-given Surface of the Hybrid DOmain.

    Parameters
    ----------
        X : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Node position along the x axis.

        Y : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Node position along the y axis.

        Z : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Node position along the z axis.

        Surface : Zone
            Cutoff closed surface.

    Returns
    ----------
        inside : numpy.ndarray
            Flag of the particles inside the Surface.
    '''
    if np.array(Surface).any() and np.array(X).any() and np.array(Y).any() and np.array(Z).any():
        box = [np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf]
        for BC in I.getZones(Surface):
            x, y, z = J.getxyz(BC)
            box = [min(box[0], np.min(x)), min(box[1], np.min(y)), min(box[2], np.min(z)),
                         max(box[3], np.max(x)), max(box[4], np.max(y)), max(box[5], np.max(z))]

        inside = (box[0] < X)*(box[1] < Y)*(box[2] < Z)*(X < box[3])*(Y < box[4])*(Z < box[5])#does a first cleansing to avoid checking far away particles
        x, y, z = X[inside], Y[inside], Z[inside]
        if x and y and z:
            mask = C.convertArray2Node(J.createZone('Zone', [x, y, z], 'xyz'))
            mask = I.getZones(CX.blankCells(C.newPyTree(['Base', mask]), [[Surface]],
                      np.array([[1]]),  blankingType = 'node_in', delta = 0., dim = 3, tol = 0.))[0]
            cellN = J.getVars(mask, ['cellN'], 'FlowSolution')[0]
            inside[inside] = (cellN == 0)
    else:
        inside = [False]*len(X)

    return np.array(inside, order = 'F', dtype = np.int32)

def eraseParticlesInHybridDomain(t = []):
    '''
    Erases the particles inside the Inner Interface of the Hybrid Domain.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Lagrangian Field.
    '''
    x, y, z = J.getxyz(VPM.pickParticlesZone(t))
    flag = flagParticlesInsideSurface(x, y, z, pickHybridDomainOuterInterface(t))
    Nll, Nbem, Ncfd = getParameters(VPM.pickParticlesZone(t), ['NumberOfLiftingLineSources',
                                                    'NumberOfBEMSources', 'NumberOfCFDSources'])
    flag[:Nll[0] + Nbem[0] + Ncfd[0]] = False
    VPM.delete(t, flag)
    return np.sum(flag)

def splitHybridParticles(t = []):
    '''
    Redistributes the particles inside the Hybrid Domain onto a finer grid.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Lagrangian Field.
    '''
    Particles = VPM.pickParticlesZone(t)
    splitParticles = V.split_hybrid_particles(t)
    Nll, Nbem, Ncfd, Nh = getParameters(Particles, ['NumberOfLiftingLineSources',
                           'NumberOfBEMSources', 'NumberOfCFDSources', 'NumberOfHybridSources'])
    Offset = Nll[0] + Nbem[0] + Ncfd[0]
    if splitParticles.any():
        Nsplit = len(splitParticles[0])
        adjustTreeSize(t, NewSize = Nsplit, OldSize =  Nh, AtTheEnd = False, Offset = Offset)
        X, Y, Z = J.getxyz(Particles)
        AX, AY, AZ, S = J.getVars(Particles, ['Alpha' + v for v in 'XYZ'] + ['Sigma'])
        X[Offset: Offset + Nsplit]          = splitParticles[0][:]
        Y[Offset: Offset + Nsplit]          = splitParticles[1][:]
        Z[Offset: Offset + Nsplit]          = splitParticles[2][:]
        AX[Offset: Offset + Nsplit]         = splitParticles[3][:]
        AY[Offset: Offset + Nsplit]         = splitParticles[4][:]
        AZ[Offset: Offset + Nsplit]         = splitParticles[5][:]
        S[Offset: Offset + Nsplit]          = splitParticles[6][:]
    else :Nsplit = Nh#case where Resolution = Sigma

    return Nsplit

def solveParticleStrength(t = [], VorticityX = [], VorticityY = [], VorticityZ = [],Offset = 0):
    '''
    Initialise the strength of Particles so that they induce the vorticity given be the user.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        VorticityX : :py:class:`list` or numpy.ndarray of :py:class:`str`
            Target Vorticity along the x axis.

        VorticityY : :py:class:`list` or numpy.ndarray of :py:class:`str`
            Target Vorticity along the y axis.

        VorticityZ : :py:class:`list` or numpy.ndarray of :py:class:`str`
            Target Vorticity along the z axis.

        Offset : :py:class:`int`
            Offset position from where the particle strength will be initialised.
    '''
    Particles = pickParticlesZone(t)
    x, y, z = J.getxyz(Particles)
    ax, ay, az = J.getVars(Particles, ['Alpha' + v for v in 'XYZ'])
    ax[:] = 0.
    ay[:] = 0.
    az[:] = 0.
    Zone = C.convertArray2Node(J.createZone('Grid', [x, y, z], 'xyz'))
    wx, wy, wz = J.invokeFields(Zone, ['VorticityX', 'VorticityY', 'VorticityZ'])
    wx[Offset: Offset + len(VorticityX)] = VorticityX[:]
    wy[Offset: Offset + len(VorticityY)] = VorticityY[:]
    wz[Offset: Offset + len(VorticityZ)] = VorticityZ[:]
    C.convertPyTree2File(Zone, 'tref.cgns')
    aaa = V.solve_particle_strength(t, VorticityX, VorticityY, VorticityZ, Offset)
    VPM.induceVPMField(t)
    C.convertPyTree2File(t, 't.cgns')
    sys.exit()
    return aaa

def findMinimumDistanceBetweenParticles(X = [], Y = [], Z = []):
    '''
    Gives the distance between the nodes and their closest neighbour.

    Parameters
    ----------
        X : :py:class:`list` or numpy.ndarray of :py:class:`str`
            Positions along the x axis.

        Y : :py:class:`list` or numpy.ndarray of :py:class:`str`
            Positions along the y axis.

        Z : :py:class:`list` or numpy.ndarray of :py:class:`str`
            Positions along the z axis.

    Returns
    -------
        MinimumDistance : numpy.ndarray
            Closest neighbours distance.
    '''
    return V.find_minimum_distance_between_particles(X, Y, Z)

def updateBEMMatrix(t = []):
    '''
    Creates and inverse the BEM matrix used to impose the boundary condition on the solid.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Lagrangian Field.
    '''
    HybridParameters = getHybridParameters(t)
    HybridParameters['BEMMatrix'][:] = V.inverse_bem_matrix(t)

def updateBEMSources(tL = []):
    '''
    Impose the boundary condition on the solid by solving the BEM equation and updating the 
    strength of the solid bound particles.

    Parameters
    ----------
        tL : Tree, Base, Zone or list of Zone
            Lagrangian Field.
    '''
    return V.update_bem_strength(tL)

def updateCFDSources(tL = [], tE = []):
    '''
    Updates the particles embedded on the Inner Interface of the Hybrid Domain from the Eulerian
    Field.

    Parameters
    ----------
        tL : Tree, Base, Zone or list of Zone
            Lagrangian Field.

        tE : Tree, Base, Zone or list of Zone
            Eulerian Field.
    '''
    Particles = VPM.pickParticlesZone(tL)
    HybridParameters = getHybridParameters(Particles)
    it, Ramp = getParameters(tL, ['CurrentIteration', 'StrengthRampAtbeginning'])
    Ramp = np.sin(min((it[0] + 1)/Ramp[0], 1.)*np.pi/2.)
    Nll, Nbem, Ncfd, U0 = getParameters(Particles, ['NumberOfLiftingLineSources',
                              'NumberOfBEMSources', 'NumberOfCFDSources', 'VelocityFreestream'])
    Offset = Nll[0] + Nbem[0]

    Fields = findDonorFields(tE, HybridParameters['CFDDonors'],
                              HybridParameters['CFDReceivers'], ['Velocity' + v for v in 'XYZ'])
    Fields['VelocityX'] -= U0[0]
    Fields['VelocityY'] -= U0[1]
    Fields['VelocityZ'] -= U0[2]
    SX = HybridParameters['SurfaceCFDX']
    SY = HybridParameters['SurfaceCFDY']
    SZ = HybridParameters['SurfaceCFDZ']
    AX, AY, AZ = J.getVars(Particles, vectorise('Alpha'))
    Ncfd = Offset + Ncfd[0]
    AX[Offset: Ncfd] = (SY*Fields['VelocityZ'] - SZ*Fields['VelocityY'])*Ramp
    AY[Offset: Ncfd] = (SZ*Fields['VelocityX'] - SX*Fields['VelocityZ'])*Ramp
    AZ[Offset: Ncfd] = (SX*Fields['VelocityY'] - SY*Fields['VelocityX'])*Ramp
    HybridParameters['AlphaCFDN'][:] = Ramp*(SX*Fields['VelocityX'] + SY*Fields['VelocityY'] + \
                                                                         SZ*Fields['VelocityZ'])

def updateHybridSources(tL = [], tE = [], IterationInfo = {}):
    '''
    Generates hybrid particles inside the Hybrid Domain.

    Parameters
    ----------
        tL : Tree, Base, Zone or list of Zone
            Lagrangian Field.

        tE : Tree, Base, Zone or list of Zone
            Eulerian Field.

        IterationInfo : :py:class:`dict` of :py:class:`str`
            Hybrid solver information on the current iteration.
    '''
    Particles = VPM.pickParticlesZone(tL)
    HybridParameters = getHybridParameters(Particles)
    VPMParameters = VPM.getVPMParameters(Particles)
    Ramp = np.sin(min((VPMParameters['CurrentIteration'][0] + 1)/\
                                      VPMParameters['StrengthRampAtbeginning'][0], 1.)*np.pi/2.)
    Nll, Nbem, Ncfd = getParameters(Particles, ['NumberOfLiftingLineSources',
                                                    'NumberOfBEMSources', 'NumberOfCFDSources'])
    Nbem = Nbem[0]
    Ncfd = Ncfd[0]
    Offset = Nll[0]
    Fields = findDonorFields(tE, HybridParameters['HybridDonors'],
                          HybridParameters['HybridReceivers'], ['Center' + v for v in 'XYZ'] + \
                                                               ['Vorticity' + v for v in 'XYZ'])
    wp = np.linalg.norm(np.vstack([Fields['VorticityX'], Fields['VorticityY'],
                                                               Fields['VorticityZ']]), axis = 0)
    wtot = np.sum(wp)#total particle vorticity in the Hybrid Domain

    ParticleSeparationPerInterface = HybridParameters['ParticleSeparationPerInterface']
    VorticityFactor = HybridParameters['FactorOfCFDVorticityShed'][0]
    hybrid = np.array([False]*len(ParticleSeparationPerInterface['Interface_0']))
    if 0. < VorticityFactor:
        for i, InterfaceFlag in enumerate(ParticleSeparationPerInterface):
            flag = (ParticleSeparationPerInterface[InterfaceFlag] == 1)
            wi = wp[flag]
            if len(wi):
                wmin = findMinHybridParticleVorticityToShed(wi, VorticityFactor)
                flag[flag] = wmin < wi#only the strongest will survive
                hybrid += np.array(flag)
    else:
        Ni = HybridParameters['NumberOfParticlesPerInterface'][0]
        for i, InterfaceFlag in enumerate(ParticleSeparationPerInterface):
            flag = (ParticleSeparationPerInterface[InterfaceFlag] == 1)
            wi = wp[flag]
            Ncurrent = min(max(Ni, 1), len(wi))
            wmin = np.sort(wi)[-Ncurrent]#vorticity cutoff
            flagi = wmin < wi
            Nflagi = np.sum(flagi)
            if Ncurrent != Nflagi:#if several cells have the same vort and it happens to be wmin, Ncurrent will be below Ni, so i need to add somme cells until Ncurrent = Ni
                allwmin = np.where(wmin == wi)[0]
                flagi[allwmin[: Ncurrent - Nflagi]] = True

            flag[flag] = flagi#only the strongest will survive
            hybrid += np.array(flag)

    Nh = np.sum(hybrid)
    sigma = VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]
    apx = HybridParameters['AlphaHybridX']
    apy = HybridParameters['AlphaHybridY']
    apz = HybridParameters['AlphaHybridZ']
    IterationInfo['Number of Hybrids Generated'] = -eraseParticlesInHybridDomain(tL)

    addParticlesToTree(tL, Fields['CenterX'][hybrid], Fields['CenterY'][hybrid],
                               Fields['CenterZ'][hybrid], apx[hybrid], apy[hybrid], apz[hybrid],
                                  HybridParameters['HybridSigma'][hybrid], Offset + Nbem + Ncfd)

    solveParticleStrength(tL, Fields['VorticityX'][hybrid]*Ramp, Fields['VorticityY'][hybrid]*\
                                  Ramp, Fields['VorticityZ'][hybrid]*Ramp, Offset + Nbem + Ncfd)











    Particles = VPM.pickParticlesZone(tL)
    x, y, z = J.getxyz(Particles)
    Zone = C.convertArray2Node(J.createZone('Grid', [x, y, z], 'xyz'))
    wx, wy, wz = J.invokeFields(Zone, ['VorticityX', 'VorticityY', 'VorticityZ'])
    wx[Offset + Nbem + Ncfd: Offset + Nbem + Ncfd + np.sum(hybrid)] = Fields['VorticityX'][hybrid]*Ramp
    wy[Offset + Nbem + Ncfd: Offset + Nbem + Ncfd + np.sum(hybrid)] = Fields['VorticityY'][hybrid]*Ramp
    wz[Offset + Nbem + Ncfd: Offset + Nbem + Ncfd + np.sum(hybrid)] = Fields['VorticityZ'][hybrid]*Ramp

    C.convertPyTree2File(Zone, 'tref.cgns')
    tref = C.newPyTree(['tref', [Zone]])
    extract(tL, tref)
    C.convertPyTree2File(tref, 't.cgns')


    wx, wy, wz = J.getVars(I.getZones(tref)[0], ['Vorticity' + v for v in 'XYZ'])
    err = np.linalg.norm(np.vstack([wx[Offset + Nbem + Ncfd: Offset + Nbem + Ncfd + np.sum(hybrid)] - Fields['VorticityX'][hybrid]*Ramp,
                                    wy[Offset + Nbem + Ncfd: Offset + Nbem + Ncfd + np.sum(hybrid)] - Fields['VorticityY'][hybrid]*Ramp,
                                    wz[Offset + Nbem + Ncfd: Offset + Nbem + Ncfd + np.sum(hybrid)] - Fields['VorticityZ'][hybrid]*Ramp]), axis = 0)
    print(Offset + Nbem + Ncfd, len(x), np.sum(err)/len(err), np.max(err), np.min(err))
    err /= np.linalg.norm(np.vstack([Fields['VorticityX'][hybrid]*Ramp,
                                     Fields['VorticityY'][hybrid]*Ramp,
                                     Fields['VorticityZ'][hybrid]*Ramp]), axis = 0)
    print(np.sum(err)/len(err), np.max(err), np.min(err))

    sys.exit()











    
    ax, ay, az = J.getVars(Particles, ['Alpha' + v for v in 'XYZ'])
    Offset += Nbem + Ncfd
    HybridParameters['AlphaHybridX'][hybrid] = ax[Offset: Offset + Nh]
    HybridParameters['AlphaHybridY'][hybrid] = ay[Offset: Offset + Nh]
    HybridParameters['AlphaHybridZ'][hybrid] = az[Offset: Offset + Nh]

    HybridParameters['NumberOfHybridSources'][0] = Nh
    HybridParameters['NumberOfHybridSources'][0] = splitHybridParticles(tL)
    IterationInfo['Number of Hybrids Generated'] += HybridParameters['NumberOfHybridSources'][0]

    wtot = np.sum(wp)
    wp = wp[hybrid]
    wkept = np.sum(wp)#total particle voticity kept
    IterationInfo['Minimum Eulerian Vorticity'] = np.min(wp)
    IterationInfo['Eulerian Vorticity lost'] = wtot - wkept
    IterationInfo['Eulerian Vorticity lost per'] = (wtot - wkept)/wtot*100
    return IterationInfo

def updateHybridDomainAndSources(tL = [], tE = [], IterationInfo = {}):
    '''
    Updates the Hybrid Domain from the Eulerian Field.

    Parameters
    ----------
        tL : Tree, Base, Zone or list of Zone
            Lagrangian Field.

        tE : Tree, Base, Zone or list of Zone
            Eulerian Field.

        IterationInfo : :py:class:`dict` of :py:class:`str`
            Hybrid solver information on the current iteration.
    '''
    if not tE: return IterationInfo
    IterationInfo['Hybrid Computation time'] = J.tic()

    updateCFDSources(tL, tE)
    updateBEMSources(tL)
    IterationInfo = updateHybridSources(tL, tE, IterationInfo)

    IterationInfo['Hybrid Computation time'] = J.tic() -IterationInfo['Hybrid Computation time']
    return IterationInfo

def inducePanelVelocity(t = []):
    '''
    Add the induced self velocity of the panels upon themselves.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Lagrangian Field.
    '''
    Particles = VPM.pickParticlesZone(t)
    Nll, Nbem, Ncfd, ratio = getParameters(Particles, ['NumberOfLiftingLineSources', \
                                  'NumberOfBEMSources', 'NumberOfCFDSources', 'SmoothingRatio'])
    Offset = Nll[0]
    Np = Nbem[0] + Ncfd[0]
    HybridParameters = getHybridParameters(Particles)
    s = HybridParameters['SurfaceCFD']
    nx = np.append(HybridParameters['NormalBEMX'], HybridParameters['SurfaceCFDX']/s)
    ny = np.append(HybridParameters['NormalBEMY'], HybridParameters['SurfaceCFDY']/s)
    nz = np.append(HybridParameters['NormalBEMZ'], HybridParameters['SurfaceCFDZ']/s)
    an = np.append(HybridParameters['AlphaBEMN'], HybridParameters['AlphaCFDN'])
    ax, ay, az, s, ux, uy, uz = J.getVars(Particles, ['Alpha' + v for v in 'XYZ'] + ['Sigma'] +\
                                                         ['VelocityInduced' + v for v in 'XYZ'])
    S = 0.5/(s[Offset: Np]*s[Offset: Np])
    ux[Offset: Np] += (nx*an + ny*az[Offset: Np] - nz*ay[Offset: Np])*S
    uy[Offset: Np] += (ny*an + nz*ax[Offset: Np] - nx*az[Offset: Np])*S
    uz[Offset: Np] += (nz*an + nx*ay[Offset: Np] - ny*ax[Offset: Np])*S
