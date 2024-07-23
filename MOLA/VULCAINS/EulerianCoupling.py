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
#    along with MOLA.  If not, see <http://www.gnV.org/licenses/>.

'''
VULCAINS (Viscous Unsteady Lagrangian Code for Aerodynamics with Incompressible Navier-Stokes)

This module enables the coupling of the VPM with the FAST CFD solver.

Version:
0.5

Author:
Johan VALENTIN
'''

import numpy as np

import Converter.PyTree as C
import Converter.Internal as I
import Generator.PyTree as G
import Transform.PyTree as T
import Connector.PyTree as CX

try:
    import Fast.PyTree as Fast
    import FastS.PyTree as FastS
    import FastC.PyTree as FastC
except:
    pass

from .. import InternalShortcuts as J
from . import Main as V

####################################################################################################
####################################################################################################
############################################## Hybrid ##############################################
####################################################################################################
####################################################################################################
def initialiseEulerianDomain(Mesh = [], Parameters = {}):
    '''
    Creates the Eulerian tree used by VULCAINS and FAST.

    Parameters
    ----------
        Mesh : Tree
            Containes the Eulerian mesh.

        Parameters : :py:class:`dict` of :py:class:`dict`
            User-provided VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.User.compute`
    Returns
    -------
        tE : Tree
            Eulerian field.
    '''
    if not Mesh:
        if 'HybridParameters' in Parameters: Parameters['HybridParameters'].clear()
        return []
    HybridParameters = Parameters['HybridParameters']
    if isinstance(Mesh, str): Mesh = V.load(Mesh)

    import MOLA.Preprocess as PRE
    import Apps.Fast.MB as AppFastMB
    FluidParameters = Parameters['FluidParameters']
    ReferenceValuesParams = dict(Density = FluidParameters['Density'],
                                 Temperature = FluidParameters['Temperature'],
                                 Velocity = np.linalg.norm(FluidParameters['VelocityFreestream']))

    FluidProperties = PRE.computeFluidProperties(SutherlandViscosity = \
                                   FluidParameters['KinematicViscosity']*FluidParameters['Density'])
    ReferenceValues = PRE.computeReferenceValues(FluidProperties,
                                                 **ReferenceValuesParams)
    RefState = [
            ['Cv',   FluidProperties['cv']],
            ['Gamma',FluidProperties['Gamma']],
            ['Mus',  FluidProperties['SutherlandViscosity']],
            ['Cs',   FluidProperties['SutherlandConstant']],
            ['Ts',   FluidProperties['SutherlandTemperature']],
            ['Pr',   FluidProperties['Prandtl']],
            ['Density', ReferenceValues['Density']],
            ['MomentumX', FluidParameters['VelocityFreestream'][0]*FluidParameters['Density'][0]],
            ['MomentumY', FluidParameters['VelocityFreestream'][1]*FluidParameters['Density'][0]],
            ['MomentumZ', FluidParameters['VelocityFreestream'][2]*FluidParameters['Density'][0]],
            ['EnergyStagnationDensity', ReferenceValues['EnergyStagnationDensity']],
            ['TurbulentEnergyKineticDensity', ReferenceValues['TurbulentEnergyKineticDensity']],
            ['TurbulentDissipationRateDensity', ReferenceValues['TurbulentDissipationRateDensity']],
            ['TurbulentEnergyKineticDensity', ReferenceValues['TurbulentEnergyKineticDensity']],
            ['TurbulentDissipationRateDensity', ReferenceValues['TurbulentDissipationRateDensity']],
            ['ViscosityMolecular', ReferenceValues['ViscosityMolecular']],
            ['ViscosityEddy', ReferenceValues['ViscosityEddy']],
            ['Mach', ReferenceValues['Mach']],
            ['Reynolds', ReferenceValues['Reynolds']],
            ['Pressure', ReferenceValues['Pressure']],
            ['Temperature', ReferenceValues['Temperature']],
            ['Rok', 1e-9],
            ['RoOmega', 0.0675],
            ['TurbulentSANuTildeDensity', FluidProperties['SutherlandViscosity']],
                    ]

    t = C.newPyTree(['BASE'])
    t[2][1][2].append(I.createNode('ReferenceState', 'ReferenceState_t', 
        children = [I.createNode(state[0],'DataArray_t', value = state[1]) for state in RefState]))
    for z in I.getZones(Mesh):    
        C._addBC2Zone(z, 'wall', 'BCWall','kmin')
        C._addBC2Zone(z, 'overlap', 'BCOverlap', 'kmax')
        t[2][1][2].append(z)

    t = CX.connectMatch(t, tol = 1e-8)
    t, tc = AppFastMB.prepare(t, 0, 0, NP = 0)
    tc0 = C.node2Center(t)
    for var in V.vectorise('Coordinate'):
        C._initVars(tc, var, 0)
        C._cpVars(tc0, var, tc, var)

    I._rmNodesFromType(t, 'FlowEquationSet_t')
    C.addState2Node__(t[2][1], 'EquationDimension', 3)
    C.addState2Node__(t[2][1], 'GoverningEquations', 'NSTurbulent')

    # vars = V.vectorise('Coordinate') + ['Density', 'EnergyStagnationDensity']
    vars = V.vectorise('Momentum') + ['Density', 'EnergyStagnationDensity']
    Model = I.getNodeFromName(t, 'GoverningEquations')
    if Model is not None and I.getValue(Model) == 'NSTurbulent':
        vars += ['TurbulentSANuTildeDensity']

    state = I.getNodeFromType(t, 'ReferenceState_t')
    for base in I.getBases(t):
        for var in vars:
            node = I.getNodeFromName(state, var)
            if node is not None: C._initVars(base, 'centers:' + var, float(node[1][0]))
    dtL = Parameters['NumericalParameters']['TimeStep'][0]
    dtE = HybridParameters['EulerianTimeStep'][0]
    if not dtL or not dtE:
        if dtL and (not dtE): dtE = dtL
        elif dtE and (not dtL): dtL = HybridParameters['EulerianTimeStep'][0]
        else: raise ValueError(J.FAIL + 'The Lagrangian or Eulerian TimeStep must be specified ' + \
                                          'in .Numerical#Parameters or .Hybrid#Parameters' + J.ENDC)

    Parameters['NumericalParameters']['TimeStep'][0] = dtL
    HybridParameters['EulerianTimeStep'][0] = np.array([dtL/max(1, int(round(dtL/dtE)))],
                                                                    order = 'F', dtype = np.float64)
    # #RANS
    # numb = {
    #             "temporal_scheme": "implicit",
    #             "ss_iteration":HybridParameters['EulerianSubIterations'][0],
    #             "modulo_verif":100,
    #         }
    # numz = {
    #             "time_step": HybridParameters['EulerianTimeStep'][0],
    #             "scheme":"ausmpred",
    #             "time_step_nature":"local",
    #             "ssdom_IJK":[10000,10000,10000], # pas de decoupage dans l'implicite local
    #             "psiroe": 0.01,
    #             "cfl":5.,
    #             "nb_relax":1, # nbre de passages de newton
    #             "epsi_newton":0.01, # residu a atteindre
    #         }
    #URANS
    numb = {
                "temporal_scheme": "implicit_local",
                "ss_iteration":HybridParameters['EulerianSubIterations'][0],
                "modulo_verif":100,
            }
    numz = {
                "time_step": HybridParameters['EulerianTimeStep'][0],
                "scheme":"ausmpred",
                "time_step_nature":"global",
                "ssdom_IJK":[10000,10000,10000], # pas de decoupage dans l'implicite local
                "psiroe": 0.01,
                "cfl":1.,
                "nb_relax":1, # nbre de passages de newton
                "epsi_newton":0.01, # residu a atteindre
            }
    t = V.load('tE.cgns')
    base = I.getBases(t)[0]
    base[0] = 'EulerianBase'
    basec = I.getBases(t)[1]
    basec[0] = 'EulerianBaseCenter'
    Fast._setNum2Base(base, numb)
    Fast._setNum2Zones(base, numz)
    tE = C.newPyTree([base, basec])
    computeFastMetrics(tE)

    V.show(f"{'||':>57}\r" + '||'+'{:-^53}'.format(' Fast Warmup (0%) '))
    n_warmup = 1
    for ite in range(n_warmup):
        computeFast(tE)
        V.deletePrintedLines()
        V.show(f"{'||':>57}\r" + '||'+'{:-^53}'.format(' Fast Warmup (' + \
                                                '{:d}'.format(int((ite + 1)/n_warmup*100)) + '%) '))

        # if ite%1000 == 0:
        #     print(np.max(I.getNodeFromName(tE, 'Temperature')[1]))
        #     Fast.save(V.getEulerianBase(tE), 'tE01.cgns')

    V.show(f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Done '))
    # save(tE, 'tE0.cgns')
    # exit()
    flagEulerianCells(tE, Parameters)
    computeEulerianVorticity(tE, removeVelocityGradients = True)
    I.createUniqueChild(base, 'Iteration', 'DataArray_t', value = 0)
    I.createUniqueChild(base, 'Time', 'DataArray_t', value = 0)
    
    # t, tc = V.getEulerianBases(tE)
    # Fast.save(t, 't.cgns')
    # Fast.save(tc, 'tc.cgns')
    # exit()
    return tE

def getMeanTurbulentDistance(t = [], cellN = 2, cellNName = ''):
    '''
    Computes the mean turbulent distance in t where the cellNName field egals cellN.

    Parameters
    ----------
        t : Tree
            Containes the TurbulentDistance and cellNName.

        cellN : :py:class:`float`
            cellNName value.

        cellNName : :py:class:`str`
            cellNName field name.
    Returns
    -------
        d : :py:class:`float`
            Mean TurbulentDistance.
    '''
    d = []
    for zone in I.getZones(t):
        FlowSolution = I.getNodeFromName1(zone, 'FlowSolution#Centers')
        TurbulentDistance = I.getNodeFromName1(FlowSolution, 'TurbulentDistance')[1]
        if cellNName: flag = I.getNodeFromName1(FlowSolution, cellNName)[1] == cellN
        else: flag = np.ones(np.shape(TurbulentDistance), dtype = np.bool_)
        d += np.ravel(TurbulentDistance[flag]).tolist()
    return np.mean(d)

def flagEulerianCells(tE = [], Parameters = {}):
    '''
    Initialises the far field BC, ghost cells, Inner, Outer and BEM cells indices and the
    corresponding parameters.

    Parameters
    ----------
        tE : Tree
            Eulerian field.

        Parameters : :py:class:`dict` of :py:class:`dict`
            User-provided VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.User.compute`
    '''
    V.show(f"{'||':>57}\r" + '||'+'{:-^53}'.format(' Flagging Cells '))
    HybridParameters = Parameters['HybridParameters']
    t, tc = V.getEulerianBases(tE)
    Nvoid = 2#for the ghost cells
    Nbc = HybridParameters['NumberOfBCCells'][0]
    NOuter = HybridParameters['OuterDomainCellOffset'][0]
    C._initVars(t,'centers:cellN', 1)
    CX._applyBCOverlaps(t, depth = Nvoid      , loc = 'centers', val = 0, cellNName = 'cellN')
    CX._applyBCOverlaps(t, depth = Nvoid + Nbc, loc = 'centers', val = 2, cellNName = 'cellN')
    C._cpVars(t, 'centers:cellN', tc, 'cellN')
    Nvoid += Nbc

    C._initVars(t,'centers:Index', 0)
    for zone in I.getZones(t):
        Index = np.ravel(I.getNodeFromName2(zone, 'Index')[1], order = 'F')
        Index[:] = np.arange(len(Index))

    t_Ghostless = rmGhostCells(t)
    C._initVars(t,'centers:Flag', 1)
    for zone, zone_Ghostless in zip(I.getZones(t), I.getZones(t_Ghostless)):
        Flag = np.ravel(I.getNodeFromName2(zone, 'Flag')[1], order = 'F')
        Index = np.ravel(I.getNodeFromName2(zone_Ghostless, 'Index')[1], order = 'F')
        Flag[:] = 0
        Flag[Index.astype(np.int32)] = 1

    I._rmNodesByName(t, 'Index')
    NBEM = 0
    for zone in I.getZones(t_Ghostless):
        overlaps = I.getNodeFromName(zone, 'ZoneGridConnectivity')
        for o in overlaps[2]:
            NBEM = max(NBEM, I.getNodeFromName1(o, 'PointRange')[1][-1][-1])

    NBEM -= 1
    C._initVars(t, 'centers:BEM={centers:Flag}')
    CX._applyBCOverlaps(t, depth = NBEM - 1, loc = 'centers', val = 0, cellNName = 'BEM')
    CX._applyBCOverlaps(t, depth = NBEM, loc = 'centers', val = 2, cellNName = 'BEM')
    dBEM = getMeanTurbulentDistance(t, cellN = 2, cellNName = 'BEM')

    C._initVars(t, 'centers:Outer={centers:Flag}')
    Nvoid += NOuter
    CX._applyBCOverlaps(t, depth = Nvoid, loc = 'centers', val = 0, cellNName = 'Outer')
    CX._applyBCOverlaps(t, depth = Nvoid + 1, loc = 'centers', val = 2, cellNName = 'Outer')
    dOuter = getMeanTurbulentDistance(t, cellN = 2, cellNName = 'Outer')
    NOuter = Nvoid + 1

    C._initVars(t, 'centers:Inner={centers:Outer}')
    Nvoid += 1
    dInner = dOuter - HybridParameters['HybridDomainSize'][0]
    if dInner <= dBEM:
        raise ValueError(J.FAIL + 'The Hybrid Domain is too big to fit in the Eulerian Mesh.' +
               '(DomainSize=%g, OuterInterfaceRadius=%g)'%(HybridParameters['HybridDomainSize'][0],\
                                                                                   dOuter) + J.ENDC)
    d0 = d1 = dOuter
    while dInner < d1:
        d0 = d1
        Nvoid += 1
        CX._applyBCOverlaps(t, depth = Nvoid, loc = 'centers', val = 0, cellNName = 'Inner')
        CX._applyBCOverlaps(t, depth = Nvoid + 1, loc = 'centers', val = 2, cellNName = 'Inner')
        d1 = getMeanTurbulentDistance(t, cellN = 2, cellNName = 'Inner')

    NInner = Nvoid + 1
    I._rmNodesByName(t, '.Solver#VPM')



    import Dist2Walls.PyTree as Dist2Walls
    Outers = T.subzone(t, (1, 1, -2), (-1, -1, -2), type='elements')
    Outers = Dist2Walls.distance2Walls(t, Outers, signed = 1, loc = 'nodes')
    I._rmNodesByName(Outers, 'FlowSolution#Centers')
    Outers = C.node2Center(Outers, 'FlowSolution')
    I._rmNodesByName(Outers, 'FlowSolution')
    Inners = T.subzone(t, (1, 1, -(2 + Nbc)), (-1, -1, -(2 + Nbc)), type='elements')
    Inners = Dist2Walls.distance2Walls(t, Inners, signed = 1, loc = 'nodes')
    I._rmNodesByName(Inners, 'FlowSolution#Centers')
    Inners = C.node2Center(Inners, 'FlowSolution')
    I._rmNodesByName(Inners, 'FlowSolution')
    for zone, Inner, Outer in zip(I.getZones(t), I.getZones(Inners), I.getZones(Outers)):
        parent = I.createUniqueChild(zone, '.Solver#VPM','UserDefined_t')
        BC = np.array(np.where(I.getNodeFromName(zone, 'cellN')[1] == 2), dtype = np.int32,
                                                                                        order = 'F')
        children = [I.createNode('BCFarFieldIndices', 'Tuple_t', value = BC),
                    I.createNode('GhostCellsIndices', 'Tuple_t',
                                 value = np.array(np.where(I.getNodeFromName(zone, 'Flag')[1] == 0),
                                                                    dtype = np.int32, order = 'F')),
                    I.createNode('BEMInterfaceIndices', 'Tuple_t',
                                  value = np.array(np.where(I.getNodeFromName(zone, 'BEM')[1] == 2),
                                                                    dtype = np.int32, order = 'F')),
                    I.createNode('InnerInterfaceIndices', 'Tuple_t',
                                value = np.array(np.where(I.getNodeFromName(zone, 'Inner')[1] == 2),
                                                                    dtype = np.int32, order = 'F')),
                    I.createNode('OuterInterfaceIndices', 'Tuple_t',
                                value = np.array(np.where(I.getNodeFromName(zone, 'Outer')[1] == 2),
                                                                    dtype = np.int32, order = 'F'))]

        I.addChild(parent, children)

    I._rmNodesByName(t, 'Outer')
    I._rmNodesByName(t, 'Inner')
    I._rmNodesByName(t, 'BEM')
    I._rmNodesByName(t, 'Flag')
    HybridParameters['HybridDomainSize'] = dH = np.array([dOuter - dInner], dtype = np.float64, \
                                                                                        order = 'F')
    SmoothingRatio = Parameters['ModelingParameters']['SmoothingRatio']
    NP = Parameters['NumericalParameters']
    Sigma0 = Parameters['PrivateParameters']['Sigma0']
    h = dH[0]/(HybridParameters['NumberOfHybridLayers'][0]*SmoothingRatio[0])
    parent = I.createUniqueChild(t, '.Solver#VPM','UserDefined_t')
    children = [I.createNode('ResolutionVPM', 'DataArray_t', value = h),
                I.createNode('BEMInterfaceIndex', 'DataArray_t', value = NBEM),
                I.createNode('BEMInterfaceDistance', 'DataArray_t', value = dBEM),
                I.createNode('InnerInterfaceIndex', 'DataArray_t', value = NInner),
                I.createNode('InnerInterfaceDistance', 'DataArray_t', value = dInner),
                I.createNode('OuterInterfaceIndex', 'DataArray_t', value = NOuter),
                I.createNode('OuterInterfaceDistance', 'DataArray_t', value = dOuter)]
    I.addChild(parent, children)
    if NP['Resolution'][0]:
        NP['Resolution'][0] = min(h, NP['Resolution'][0])
        NP['Resolution'][1] = max(h, NP['Resolution'][1])
    else:
        NP['Resolution'] = np.array([h]*2, order = 'F', dtype = np.float64)

    Parameters['PrivateParameters']['Sigma0'] = NP['Resolution']*SmoothingRatio

    msg =  f"{'||':>57}\r" + '|| ' + '{:32}'.format('Outer Interface cell') + ': ' + \
                                                                        '{:d}'.format(NOuter) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:32}'.format('Inner Interface cell') + ': ' + \
                                                                        '{:d}'.format(NInner) + '\n'
    msg +=  f"{'||':>57}\r" + '|| ' + '{:32}'.format('Outer Interface distance') + ': ' + \
                                                                    '{:.4g}'.format(dOuter) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:32}'.format('Inner Interface distance') + ': ' + \
                                                                    '{:.4g}'.format(dInner) + ' m\n'
    msg +=  f"{'||':>57}\r" + '|| ' + '{:32}'.format('Hybrid Domain size') + ': ' + \
                                                                     '{:.4g}'.format(dH[0]) + ' m\n'
    msg +=  f"{'||':>57}\r" + '|| ' + '{:32}'.format('Hybrid VPM resolution') + ': ' + \
                                                                         '{:.4g}'.format(h) + ' m\n'
    msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Done ')
    V.show(msg)

def generateHybridDomainInterfaces(tE = []):
    '''
    Gives the solid boundary for the BEM and the Inner and Outer Interface of the Hybrid Domain.

    Parameters
    ----------
        tE : Tree
            Eulerian Field.
    Returns
    -------
        Interfaces : Zones
            Containes the BEM, Inner and Outer Interfaces.
    '''
    t, tc = V.getEulerianBases(tE)
    NBEM = I.getNodeFromName(t, 'BEMInterfaceIndex')[1][0] - 1
    NInner = I.getNodeFromName(t, 'InnerInterfaceIndex')[1][0] - 1
    NOuter = I.getNodeFromName(t, 'OuterInterfaceIndex')[1][0] - 1
    BEMInterface = V.union(T.subzone(tc, (1, 1, -NBEM), (-1, -1, -NBEM), type='elements'))
    InnerInterface = V.union(T.subzone(tc, (1, 1, -NInner), (-1, -1, -NInner), type='elements'))
    OuterInterface = V.union(T.subzone(tc, (1, 1, -NOuter), (-1, -1, -NOuter), type='elements'))
    BEMInterface[0] = 'BEMInterface'
    InnerInterface[0] = 'InnerInterface'
    OuterInterface[0] = 'OuterInterface'
    G._getNormalMap(BEMInterface)
    G._getNormalMap(InnerInterface)
    G._getNormalMap(OuterInterface)
    for f in ['centers:s = 1e-16 + ({centers:sx}**2 + {centers:sy}**2 + {centers:sz}**2)**0.5', 
              'centers:nx = {centers:sx}/{centers:s}',
              'centers:ny = {centers:sy}/{centers:s}',
              'centers:nz = {centers:sz}/{centers:s}']:
        C._initVars(BEMInterface, f)
        C._initVars(InnerInterface, f)
        C._initVars(OuterInterface, f)

    for val in V.vectorise('s', False) + ['s']:
        I._rmNodesByName(BEMInterface, val)
        I._rmNodesByName(InnerInterface, val)
        I._rmNodesByName(OuterInterface, val)

    BEMInterface = C.center2Node(BEMInterface, 'FlowSolution#Centers')
    I._rmNodesByName(BEMInterface, 'FlowSolution#Centers')
    InnerInterface = C.center2Node(InnerInterface, 'FlowSolution#Centers')
    I._rmNodesByName(InnerInterface, 'FlowSolution#Centers')
    OuterInterface = C.center2Node(OuterInterface, 'FlowSolution#Centers')
    I._rmNodesByName(OuterInterface, 'FlowSolution#Centers')
    return [BEMInterface, InnerInterface, OuterInterface]

def setDonorsIndex(Target = [], tE = [], DonorsName = 'Donors'):
    '''
    Initialises the indices of the closests cell-centers of a mesh from a set of user-given node
    targets.

    Parameters
    ----------
        Target : Base, Zone or list of Zone
            Target nodes.

        tE : Tree
            Eulerian field.

        DonorsName : :py:class:`str`
            Name of the initialised indeces.
    Returns
    -------
        unique : numpy.ndarray
            Flag list of the unique donors.
    '''
    t = V.getEulerianBase(tE)
    zones = I.getZones(t)
    for zone in zones:
        C._initVars(zone, 'centers:' + DonorsName, -1)
        C._initVars(zone, 'centers:Index', 0)
        Index = np.ravel(I.getNodeFromName(zone, 'Index')[1], order = 'F')
        Index[:] = np.arange(len(Index))

    t_Ghostless, tc_Ghostless = V.getEulerianBases(rmGhostCells(V.getTrees(tE, 'Eulerian')))
    zones_Ghostless = I.getZones(t_Ghostless)
    hook, indir = C.createGlobalHook(tc_Ghostless, function = 'nodes', indir = 1)
    nodes, dist = C.nearestNodes(hook, J.createZone('Zone', J.getxyz(Target), 'xyz'))
    nodes, unique = np.unique(nodes, return_index = True)
    N = len(unique)
    cumul = 0
    cumulated = []
    for z in I.getZones(tc_Ghostless):
        cumulated += [cumul]
        cumul += C.getNPts(z)

    GhostlessDonors = [np.ravel(I.getNodeFromName(zone, DonorsName)[1], order = 'F') for zone in \
                                                                                    zones_Ghostless]
    pos = 0
    for p in range(N):
        ind = nodes[p] - 1
        closest_index = ind - cumulated[indir[ind]]
        GhostlessDonors[indir[ind]][closest_index] = pos
        pos += 1

    for zone, zone_Ghostless in zip(zones, zones_Ghostless):
        Donor          = I.getNodeFromName(zone, DonorsName)
        GhostlessDonor = np.ravel(I.getNodeFromName(zone_Ghostless, DonorsName)[1])
        GhostlessIndex = np.ravel(I.getNodeFromName(zone_Ghostless, 'Index')[1])
        np.ravel(Donor[1], order = 'F')[GhostlessIndex.astype(np.int32)] = GhostlessDonor

    I._rmNodesByName(t, 'Index')
    return unique

def getDonorsFields(tE = [], DonorsName = 'Donors', FieldNames = []):
    '''
    Gets the user-given fields from the donor mesh onto the receiver nodes.

    Parameters
    ----------
        tE : Tree
            Eulerian field.

        DonorsName : :py:class:`str`
            Name of the indices donors.

        FieldNames : :py:class:`list` or numpy.ndarray of :py:class:`str`
            Names of the fields to retreive.
    Returns
    -------
        Fields : :py:class:`dict`
            Extracted fields from the donor mesh.
    '''
    t, tc = V.getEulerianBases(tE)
    Fields = {}
    Nh = np.sum([np.sum(-1 < I.getValue(I.getNodeFromName(zone, DonorsName))) for zone in
                                                                                     I.getZones(t)])
    for name in FieldNames + V.vectorise('Center'):
        Fields[name] = np.zeros(Nh, dtype = np.float64, order = 'F')

    for zone  in zip(I.getZones(t), I.getZones(tc)):
        xc, yc, zc = J.getxyz(zone[1])
        Index = np.array(np.ravel(I.getValue(I.getNodeFromName(zone[0], DonorsName)),
                                                                     order = 'F'), dtype = np.int32)
        Donors = -1 < Index
        Receivers = Index[Donors]
        Fields['CenterX'][Receivers] = np.ravel(xc, order = 'F')[Donors]
        Fields['CenterY'][Receivers] = np.ravel(yc, order = 'F')[Donors]
        Fields['CenterZ'][Receivers] = np.ravel(zc, order = 'F')[Donors]
        DonorFields = J.getVars(zone[0], FieldNames, 'FlowSolution#Centers')
        for n, name in enumerate(FieldNames):
            Fields[name][Receivers] = np.ravel(DonorFields[n], order = 'F')[Donors]

    return Fields

def filterHybridSources(tL = [], tE = [], tH = []):
    '''
    Filters the Eulerian vorticity sources and selects the .

    Parameters
    ----------
        tL : Tree
            Lagrangian field.

        tE : Tree
            Eulerian field.

        tH : Tree
            Hybrid Domain.
    Returns
    -------
        Sources : Zone
            Hybrid particles sources.
    '''
    _tL, _tE, _tH = V.getTrees([tL, tE, tH], ['Particles', 'Eulerian', 'Hybrid'])
    if not _tE or not _tH or not _tL: return
    h, lmbd, Nl, Nhl, it, Ramp = V.getParameters(_tL, ['Resolution', 'SmoothingRatio',
                               'NumberOfHybridLayers', 'MaximumSourcesPerLayer', 'CurrentIteration',
                                                                         'StrengthRampAtbeginning'])
    Ramp = np.sin(min((it[0] + 1)/Ramp[0], 1.)*np.pi/2.)
    Sigma0 = h[0]*lmbd[0]
    Fields = getDonorsFields(_tE, 'HybridDonorsIndices', V.vectorise('Vorticity'))
    Layers = J.getVars(V.getHybridSources(_tH), ['Layers'])[0]

    w = np.linalg.norm(np.vstack([Fields['VorticityX'], Fields['VorticityY'],
                                                                   Fields['VorticityZ']]), axis = 0)
    flag = 0.01*np.max(w) < w
    Cells = C.convertArray2Node(J.createZone('Zone', [Fields['CenterX'][flag],
                                          Fields['CenterY'][flag], Fields['CenterZ'][flag]], 'xyz'))
    J.invokeFields(Cells, ['VorticityX'])[0][:] = Fields['VorticityX'][flag]
    J.invokeFields(Cells, ['VorticityY'])[0][:] = Fields['VorticityY'][flag]
    J.invokeFields(Cells, ['VorticityZ'])[0][:] = Fields['VorticityZ'][flag]
    J.invokeFields(Cells, ['Layers'])[0][:] = Layers[flag]

    Clusters = V.find_cell_clusters(Cells, Sigma0, 45., Nl[0], Nhl[0])
    Sources = C.convertArray2Node(J.createZone('Zone', [Clusters[0], Clusters[1],
                                                                               Clusters[2]], 'xyz'))
    J.invokeFields(Sources, ['AlphaX'])[0][:] = Clusters[3]*Ramp
    J.invokeFields(Sources, ['AlphaY'])[0][:] = Clusters[4]*Ramp
    J.invokeFields(Sources, ['AlphaZ'])[0][:] = Clusters[5]*Ramp
    J.invokeFields(Sources, ['VorticityX'])[0][:] = Clusters[6]*Ramp
    J.invokeFields(Sources, ['VorticityY'])[0][:] = Clusters[7]*Ramp
    J.invokeFields(Sources, ['VorticityZ'])[0][:] = Clusters[8]*Ramp
    J.invokeFields(Sources, ['Sigma'])[0][:] = Clusters[9]
    J.invokeFields(Sources, ['Layers'])[0][:] = Clusters[10]
    return Sources

def generateHybridSources(tE = [], Parameters = {}):
    '''
    Initialises the vorticity sources from the Eulerian Mesh.

    Parameters
    ----------
        tE : Tree
            Eulerian field.

        Parameters : :py:class:`dict` of :py:class:`dict`
            User-provided VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.User.compute`
    Returns
    -------
        HybridSources : Zone
            Vorticity sources.
    '''
    V.show(f"{'||':>57}\r" + '||'+'{:-^53}'.format(' Generate Hybrid Sources '))
    Parameters['PrivateParameters']['NumberOfHybridSources'][0] = 0
    Sigma0 = Parameters['PrivateParameters']['Sigma0'][0]
    Nl = Parameters['HybridParameters']['NumberOfHybridLayers'][0]
    Nhl = Parameters['HybridParameters']['MaximumSourcesPerLayer'][0]
    GenZones = Parameters['HybridParameters']['GenerationZones']
    t, tc = V.getEulerianBases(tE)
    CFDParam = I.getNodeFromName1(t, '.Solver#VPM')
    NInner = I.getValue(I.getNodeFromName1(CFDParam, 'InnerInterfaceIndex')) + 2#because of the ghost cells
    NOuter = I.getValue(I.getNodeFromName1(CFDParam, 'OuterInterfaceIndex')) - 2
    dO = I.getValue(I.getNodeFromName1(CFDParam, 'OuterInterfaceDistance'))
    dI = I.getValue(I.getNodeFromName1(CFDParam, 'InnerInterfaceDistance'))
    dL = (dO - dI)/Nl
    Mesh = rmGhostCells(T.subzone(tc, (1, 1, -NInner + 1), (-1, -1, -NOuter + 1)))#because the centers are shifted by one compared to the nodes
    x, y, z = [], [], []
    for zone in I.getZones(Mesh):
        xc, yc, zc = J.getxyz(zone)
        x = np.concatenate((x, np.ravel(xc)))
        y = np.concatenate((y, np.ravel(yc)))
        z = np.concatenate((z, np.ravel(zc)))

    flag = np.array([False]*len(x))
    for GenZone in GenZones:
        flag += (GenZone[0] < x)*(x < GenZone[3])*\
                (GenZone[1] < y)*(y < GenZone[4])*\
                (GenZone[2] < z)*(z < GenZone[5])

    Zone = J.createZone('Zone', [x[flag], y[flag], z[flag]], 'xyz')
    setDonorsIndex(Zone, tE, DonorsName = 'HybridDonorsIndices')
    Fields = getDonorsFields(tE, 'HybridDonorsIndices', ['TurbulentDistance'])

    Layers = np.array([Nl]*len(Fields['CenterX']))
    dH = Fields['TurbulentDistance'] - dI - (np.max(Fields['TurbulentDistance']) - dO)
    Layers[dH < 0] = 1
    for i in range(Nl): Layers[(dL*i < dH)*(dH <= dL*(i + 1))] = i + 1

    HybridSources = C.convertArray2Node(J.createZone('HybridSources', [Fields['CenterX'],
                                                      Fields['CenterY'], Fields['CenterZ']], 'xyz'))
    J.invokeFields(HybridSources, ['Layers'])[0][:] = Layers
    HybridSources[0] = 'HybridSources'
    I._sortByName(I.getNodeFromName(HybridSources, 'FlowSolution'))
    # I._rmNodesByName(HybridSources, 'GridCoordinates')
    msg = f"{'||':>57}\r" + '|| ' + '{:32}'.format('Number of Hybrid sources') + ': ' + \
                                                                   '{:d}'.format(len(Layers)) + '\n'
    msg = f"{'||':>57}\r" + '|| ' + '{:32}'.format('Max shed sources per layer') + ': ' + \
                                                                           '{:d}'.format(Nhl) + '\n'
    for i in range(1, Nl + 1):
        msg += f"{'||':>57}\r" + '|| ' + '{:32}'.format('Nodes at layer ' + str(i)) + ': ' + \
                                                           '{:d}'.format(np.sum(Layers == i)) + '\n'
    msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Done ')
    V.show(msg)
    return [HybridSources]

def generateBEMParticles(tE = [], tH = [], Parameters = {}):
    '''
    Initialises the BEM particles from the Eulerian field.

    Parameters
    ----------
        tE : Tree
            Eulerian field.

        tH : Tree
            Hybrid Domain.

        Parameters : :py:class:`dict` of :py:class:`dict`
            User-provided VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.User.compute`
    Returns
    -------
        BEM particles : Zone
            BEM particles.
    '''
    _tE, _tH = V.getTrees([tE, tH], ['Eulerian', 'Hybrid'])
    if not _tE or not _tH: return 

    V.show(f"{'||':>57}\r" + '||'+'{:-^53}'.format(' Generate BEM Panels '))
    Interface = V.getHybridDomainBEMInterface(_tH)
    PP = Parameters['PrivateParameters']
    Sigma0 = PP['Sigma0'][0]
    unique = setDonorsIndex(Interface, tE, DonorsName = 'BEMDonorsIndices')
    Fields = getDonorsFields(tE, DonorsName = 'BEMDonorsIndices')
    Zone = C.convertArray2Node(J.createZone('Zone', [Fields['CenterX'], Fields['CenterY'], \
                                                                         Fields['CenterZ']], 'xyz'))
    nx0, ny0, nz0 = J.getVars(Interface, V.vectorise('n', False))
    nx, ny, nz = J.invokeFields(Zone, V.vectorise('Normal'))
    nx[:] = nx0[unique]; ny[:] = ny0[unique]; nz[:] = nz0[unique]
    AllDonors = V.find_panel_clusters(Zone, Sigma0, 45) != 0
    AllIndex = np.array([-1]*len(AllDonors), dtype = np.int32)
    AllIndex[AllDonors] = np.arange(np.sum(AllDonors))
    for zone in I.getZones(V.getEulerianBase(tE)):
        Index = np.ravel(I.getNodeFromName(zone, 'BEMDonorsIndices')[1], order = 'F')
        Donors = -1 < Index
        Index[Donors] = AllIndex[Index[Donors].astype(np.int32)]

    Fields = getDonorsFields(tE, DonorsName = 'BEMDonorsIndices')
    PP['NumberOfBEMSources'] = np.array([len(Fields['CenterX'])], dtype = np.int32,
                                                                                        order = 'F')
    BEMParticles = C.convertArray2Node(J.createZone('BEMParticles', [Fields['CenterX'], \
                                                      Fields['CenterY'], Fields['CenterZ']], 'xyz'))
    nx0, ny0, nz0, t1x0, t1y0, t1z0, t2x0, t2y0, t2z0, ax, ay, az, an, sigma = \
                   J.invokeFields(BEMParticles, V.vectorise(['Normal', 'Tangential1', 'Tangential2', \
                                                                    'Alpha']) + ['AlphaN', 'Sigma'])
    nx0[:] = nx[AllDonors]
    ny0[:] = ny[AllDonors]
    nz0[:] = nz[AllDonors]
    #t1 = ez vec n 
    t1x0[:] = -1.*ny0
    t1y0[:] = 1.*nx0
    t1z0[:] = 0.*nz0
    t1 = np.linalg.norm(np.vstack([t1x0, t1y0, t1z0]), axis = 0)
    t1[t1 < 1e-12] = np.inf
    t1x0[:] /= t1
    t1y0[:] /= t1
    t1z0[:] /= t1
    #t2 = n vec t1
    t2x0[:] = ny0*t1z0 - nz0*t1y0
    t2y0[:] = nz0*t1x0 - nx0*t1z0
    t2z0[:] = nx0*t1y0 - ny0*t1x0
    t2 = np.linalg.norm(np.vstack([t2x0, t2y0, t2z0]), axis = 0)
    t2[t2 < 1e-12] = np.inf
    t2x0[:] /= t2
    t2y0[:] /= t2
    t2z0[:] /= t2
    sigma[:] = findMinimumDistanceBetweenParticles(Fields['CenterX'], Fields['CenterY'],
                                                                                  Fields['CenterZ'])
    sigma[sigma < Sigma0/2.] = Sigma0/2.
    I._sortByName(I.getNodeFromName(BEMParticles, 'FlowSolution'))

    msg =  f"{'||':>57}\r" + '|| ' + '{:32}'.format('Number of BEM panels') + ': ' + \
                                  '{:d}'.format(PP['NumberOfBEMSources'][0]) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:32}'.format('Targeted Particle spacing') + ': ' + \
                                                                    '{:.4f}'.format(Sigma0) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:32}'.format('Mean Particle spacing') + ': ' + \
                                                            '{:.4f}'.format(np.mean(sigma)) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' +'{:32}'.format('Particle spacing deviation') + ': ' + \
                                                             '{:.4f}'.format(np.std(sigma)) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:32}'.format('Maximum Particle spacing')  + ': ' + \
                                                             '{:.4f}'.format(np.max(sigma)) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:32}'.format('Minimum Particle spacing')  + ': ' + \
                                                             '{:.4f}'.format(np.min(sigma)) + ' m\n'
    msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Done ')
    V.show(msg)
    return [BEMParticles]

def generateImmersedParticles(tE = [], tH = [], Parameters = {}):
    '''
    Initialises the Eulerian Immersed particles from the Eulerian field.

    Parameters
    ----------
        tE : Tree
            Eulerian field.

        tH : Tree
            Hybrid Domain.

        Parameters : :py:class:`dict` of :py:class:`dict`
            User-provided VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.User.compute`
    Returns
    -------
        Immersed particles : Zone
            Eulerian Immersed particles.
    '''
    _tE, _tH = V.getTrees([tE, tH], ['Eulerian', 'Hybrid'])
    if not _tE or not _tH: return 

    V.show(f"{'||':>57}\r" + '||'+'{:-^53}'.format(' Generate Eulerian Panels '))
    Interface = V.getHybridDomainInnerInterface(_tH)
    PP = Parameters['PrivateParameters']
    Sigma0 = PP['Sigma0'][0]
    unique = setDonorsIndex(Interface, tE, DonorsName = 'CFDDonorsIndices')
    Fields = getDonorsFields(tE, DonorsName = 'CFDDonorsIndices')
    Zone = C.convertArray2Node(J.createZone('Zone', [Fields['CenterX'], Fields['CenterY'], \
                                                                         Fields['CenterZ']], 'xyz'))
    nx0, ny0, nz0 = J.getVars(Interface, V.vectorise('n', False))
    nx, ny, nz = J.invokeFields(Zone, V.vectorise('Normal'))
    nx[:] = nx0[unique]; ny[:] = ny0[unique]; nz[:] = nz0[unique]
    AllDonors = V.find_panel_clusters(Zone, Sigma0, 45) != 0
    AllIndex = np.array([-1]*len(AllDonors), dtype = np.int32)
    AllIndex[AllDonors] = np.arange(np.sum(AllDonors))
    for zone in I.getZones(V.getEulerianBase(tE)):
        Index = np.ravel(I.getNodeFromName(zone, 'CFDDonorsIndices')[1], order = 'F')
        Donors = -1 < Index
        Index[Donors] = AllIndex[Index[Donors].astype(np.int32)]

    Fields = getDonorsFields(tE, DonorsName = 'CFDDonorsIndices')
    ImmersedParticles = C.convertArray2Node(J.createZone('ImmersedParticles', [Fields['CenterX'],
                                                      Fields['CenterY'], Fields['CenterZ']], 'xyz'))
    nx0, ny0, nz0, ax, ay, az, an, sigma = J.invokeFields(ImmersedParticles, V.vectorise(['Normal', \
                                                                    'Alpha']) + ['AlphaN', 'Sigma'])
    nx0[:] = nx[AllDonors]
    ny0[:] = ny[AllDonors]
    nz0[:] = nz[AllDonors]
    sigma[:] = findMinimumDistanceBetweenParticles(Fields['CenterX'], Fields['CenterY'],
                                                                                  Fields['CenterZ'])
    sigma[sigma < Sigma0/2.] = Sigma0/2.
    I._sortByName(I.getNodeFromName(ImmersedParticles, 'FlowSolution'))

    PP['NumberOfCFDSources'] = np.array([len(Fields['CenterX'])], dtype = np.int32,
                                                                                        order = 'F')

    msg  = f"{'||':>57}\r" + '|| ' + '{:32}'.format('Number of CFD panels') + ': ' +  \
                                                   '{:d}'.format(PP['NumberOfCFDSources'][0]) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:32}'.format('Targeted Particle spacing') + ': ' + \
                                                                    '{:.4f}'.format(Sigma0) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:32}'.format('Mean Particle spacing')     + ': ' + \
                                                            '{:.4f}'.format(np.mean(sigma)) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' +'{:32}'.format('Particle spacing deviation') + ': ' + \
                                                             '{:.4f}'.format(np.std(sigma)) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:32}'.format('Maximum Particle spacing')  + ': ' + \
                                                             '{:.4f}'.format(np.max(sigma)) + ' m\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:32}'.format('Minimum Particle spacing')  + ': ' + \
                                                             '{:.4f}'.format(np.min(sigma)) + ' m\n'
    msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Done ')
    V.show(msg)
    return [ImmersedParticles]

def generateEulerianBCZone(tE = []):
    '''
    Initialises the far field BC nodes.

    Parameters
    ----------
        tE : Tree
            Eulerian field.
    Returns
    -------
        Eulerian BC : Zone
            Far field BC nodes.
    '''
    if not tE: return []
    t, tc = V.getEulerianBases(tE)
    zones = I.getZones(t)
    zonesc = I.getZones(tc)
    Nbc = np.sum([len(I.getNodeByName(zone, 'BCFarFieldIndices')[1][0]) for zone in zones])
    x, y, z = [], [], []# = np.zeros(Nbc)
    for zone, zonec in zip(zones, zonesc):
        flag = tuple(I.getNodeFromName(zone, 'BCFarFieldIndices')[1])
        x += I.getNodeFromPath(zonec, './GridCoordinates/CoordinateX')[1][flag].tolist()
        y += I.getNodeFromPath(zonec, './GridCoordinates/CoordinateY')[1][flag].tolist()
        z += I.getNodeFromPath(zonec, './GridCoordinates/CoordinateZ')[1][flag].tolist()

    BC = C.convertArray2Node(J.createZone('Zone', [np.array(x), np.array(y), np.array(z)], 'xyz'))
    J.invokeFields(BC, V.vectorise('Velocity'))
    BC[0] = 'EulerianBC'
    I._sortByName(I.getNodeFromName(BC, 'FlowSolution'))
    # I._rmNodesByName(BC, 'GridCoordinates')
    return BC

def generateHybridDomain(tE = [], Parameters = {}):
    '''
    Sets the vorticity donors from the Eulerian Mesh, generates the BEM, Inner and Outer
    Interfaces of the Hybrid Domain and retrieves the far field Eulerian BC.

    Parameters
    ----------
        tE : Tree
            Eulerian field.

        Parameters : :py:class:`dict` of :py:class:`dict`
            User-provided VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.User.compute`
    Returns
    -------
        HybridDomain : Tree
            BEM, Inner and Outer Interfaces of the Hybrid Domain, BC nodes and vorticity sources.
    '''
    _tE = V.getTrees([tE], ['Eulerian'])
    if not _tE: return []

    Interfaces = generateHybridDomainInterfaces(_tE)
    HybridSources = generateHybridSources(_tE, Parameters)
    BC = generateEulerianBCZone(_tE)
    return C.newPyTree(['HybridDomain', I.getZones(HybridSources) + I.getZones(Interfaces) + \
                                                                                    I.getZones(BC)])

def initialiseHybridParticles(tL = [], tE = [], tH = [], Parameters = {}):
    '''
    Initialises the hybrid particles.

    Parameters
    ----------
        tL : Tree
            Lagrangian field.

        tE : Tree
            Eulerian field.

        tH : Tree
            Hybrid Domain.

        Parameters : :py:class:`dict` of :py:class:`dict`
            User-provided VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.User.compute`
    '''
    _tL, _tE, _tH = V.getTrees([tL, tE, tH], ['Particles', 'Eulerian', 'Hybrid'])
    if not _tE or not _tL or not _tH: return {}

    Nl = Parameters['HybridParameters']['NumberOfHybridLayers']
    Nll = Parameters['PrivateParameters']['NumberOfLiftingLineSources']
    Nh = Parameters['PrivateParameters']['NumberOfHybridSources']
    Nhl = Parameters['HybridParameters']['MaximumSourcesPerLayer']
    Sigma0 = Parameters['PrivateParameters']['Sigma0'][0]
    Fields = getDonorsFields(_tE, 'HybridDonorsIndices', V.vectorise('Vorticity'))
    Layers = J.getVars(V.getHybridSources(_tH), ['Layers'])[0]
    w = np.linalg.norm(np.vstack([Fields['VorticityX'], Fields['VorticityY'],
                                                                   Fields['VorticityZ']]), axis = 0)
    flag = 0.01*np.max(w) < w
    Cells = C.convertArray2Node(J.createZone('Zone', [Fields['CenterX'][flag],
                                          Fields['CenterY'][flag], Fields['CenterZ'][flag]], 'xyz'))
    J.invokeFields(Cells, ['VorticityX'])[0][:] = Fields['VorticityX'][flag]
    J.invokeFields(Cells, ['VorticityY'])[0][:] = Fields['VorticityY'][flag]
    J.invokeFields(Cells, ['VorticityZ'])[0][:] = Fields['VorticityZ'][flag]
    J.invokeFields(Cells, ['Layers'])[0][:] = Layers[flag]
    Hybrids = V.find_cell_clusters(Cells, Sigma0, 45., Nl[0], Nhl[0])
    Nh[0] = len(Hybrids[0])
    Zeros = np.zeros(Nh[0])
    V.addParticlesToTree(_tL, Hybrids[0], Hybrids[1], Hybrids[2], Zeros, Zeros, Zeros, Hybrids[9],
                                                                                             Nll[0])
    Cs, Nu = J.getVars(V.getFreeParticles(_tL), ['Cvisq', 'Nu'])
    Cs[Nll[0]: Nll[0] + Nh[0]] = Parameters['ModelingParameters']['EddyViscosityConstant'][0]
    Nu[Nll[0]: Nll[0] + Nh[0]] = Parameters['FluidParameters']['KinematicViscosity'][0]

def flagNodesInsideSurface(X = [], Y = [], Z = [], Surface = []):
    '''
    Selects the particles inside the user-given Surface.

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
    -------
        inside : numpy.ndarray of :py:class:`bool`
            Flag of the particles inside the Surface.
    '''
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    if Surface and X.any() and Y.any() and Z.any():
        Surface = T.join(C.convertArray2Hexa(G.close(CX.connectMatch(Surface))))
        box = [np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf]
        for BC in I.getZones(Surface):
            x, y, z = J.getxyz(BC)
            box = [min(box[0], np.min(x)), min(box[1], np.min(y)), min(box[2], np.min(z)),
                         max(box[3], np.max(x)), max(box[4], np.max(y)), max(box[5], np.max(z))]
        inside = (box[0] < X)*(box[1] < Y)*(box[2] < Z)*(X < box[3])*(Y < box[4])*(Z < box[5])#does a first cleansing to avoid checking far away particles
        x, y, z = X[inside], Y[inside], Z[inside]
        if len(x) and len(y) and len(z):
            mask = C.convertArray2Node(J.createZone('Zone', [x, y, z], 'xyz'))
            mask = CX.blankCells(C.newPyTree(['Base', mask]), [[Surface]], np.array([[1]]),
                                             blankingType = 'node_in', delta = 0, dim = 3, tol = 0.)
            cellN = J.getVars(I.getZones(mask)[0], ['cellN'], 'FlowSolution')[0]
            inside[inside] = (cellN == 0)

    else:
        inside = [False]*len(X)

    return np.array(inside, order = 'F', dtype = np.bool_)

def eraseParticlesInHybridDomain(tL = [], tH = []):
    '''
    Erases the particles inside the Inner Interface of the Hybrid Domain.

    Parameters
    ----------
        tL : Tree
            Lagrangian field.

        tH : Tree
            Hybrid Domain.
    Returns
    -------
        Ndelete : :py:class:`int`
            Number of deleted particles.
    '''
    _tL, _tH = V.getTrees([tL, tH], ['Particles', 'Hybrid'])
    if not _tH or not _tL: return
    x, y, z = J.getxyz(V.getFreeParticles(_tL))
    flag = flagNodesInsideSurface(x, y, z, V.getHybridDomainOuterInterface(_tH))
    flag[:V.getParameter(_tL, 'NumberOfLiftingLineSources')[0]] = False
    V.delete(_tL, flag)
    return np.sum(flag)

def redistributeVorticitySources(Sources = [], tL = []):
    '''
    Redistributes the vorticity sources inside the Hybrid Domain onto a finer grid and incorporates
    them amongst the free particles.

    Parameters
    ----------
        Sources : Zone
            Vorticity Sources.

        tL : Tree
            Lagrangian field.
    '''
    if not Sources or not tL: return

    Nll, Nh, Nu0, Cs0, Nl, amin, h, KOrder = V.getParameters(tL, ['NumberOfLiftingLineSources',
                             'NumberOfHybridSources', 'KinematicViscosity', 'EddyViscosityConstant',
                                               'NumberOfHybridLayers', 'MinimumSplitStrengthFactor',
                                                         'Resolution', 'HybridRedistributionOrder'])
    Hybrids = V.split_hybrid_particles(Sources, amin[0], Nl[0], h[0], KOrder[0])
    Nh[0] = len(Hybrids[0])
    V.addParticlesToTree(tL, Hybrids[0], Hybrids[1], Hybrids[2], Hybrids[3], Hybrids[4], Hybrids[5],
                                                                                 Hybrids[6], Nll[0])
    Cs, Nu = J.getVars(V.getFreeParticles(tL), ['Cvisq', 'Nu'])
    Cs[Nll[0]: Nll[0] + Nh[0]] = Cs0
    Nu[Nll[0]: Nll[0] + Nh[0]] = Nu0

def solveHybridParticlesStrength(Sources = [], tL = []):
    '''
    Initialise the strength of Particles so that they induce the vorticity given be the user.

    Parameters
    ----------
        Sources : Zone
            Vorticity Sources.

        tL : Tree
            Lagrangian field.
    '''
    Method = V.Method_str2int[V.getParameter(tL, 'ParticleGenerationMethod')]
    eps, maxIte, eps_ratio = V.getParameters(tL, ['RelaxationThreshold',
                                                 'MaxHybridGenerationIteration', 'RelaxationRatio'])
    
    output =  V.solve_hybrid_particle_strength(Sources, tL, Method, eps[0], maxIte[0], eps_ratio[0])
    eps_ratio[0] = output[3]
    return {"Number of iterations": int(output[0]), "Rel. err. of Vorticity Eulerian": output[1],
                                                                "Iterative methode time": output[2]}

def findMinimumDistanceBetweenParticles(X = [], Y = [], Z = []):
    '''
    Gives the distance between a the of nodes and their closest neighbour.

    Parameters
    ----------
        X : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Positions along the x axis.

        Y : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Positions along the y axis.

        Z : :py:class:`list` or numpy.ndarray of :py:class:`float`
            Positions along the z axis.

    Returns
    -------
        MinimumDistance : numpy.ndarray of :py:class:`float`
            Closest neighbours distances.
    '''
    return V.find_minimum_distance_between_particles(X, Y, Z)

def updateBEMMatrix(tL = []):
    '''
    Computes and inverse the BEM matrix used to impose the boundary condition on the solid.

    Parameters
    ----------
        tL : Tree
            Lagrangian field.
    '''
    HybridParameters = V.getHybridParameters(tL)
    HybridParameters['BEMMatrix'][:] = V.inverse_bem_matrix(tL,
                                                          HybridParameters['NumberOfBEMUnknown'][0])

def updateBEMSources(tL = [], tE = []):
    '''
    Impose the boundary condition on the solid by solving the BEM equation and updating the 
    strength of the solid bound particles.

    Parameters
    ----------
        tL : Tree
            Lagrangian field.

        tE : Tree
            Eulerian field.
    '''
    _tL, _tE = V.getTrees([tL, tE], ['Particles', 'Eulerian'])
    if not _tE or not _tL: return

    it, Ramp, Nbem, U0 = V.getParameters(_tL, ['CurrentIteration', 'StrengthRampAtbeginning',
                                                        'NumberOfBEMSources', 'VelocityFreestream'])
    ImmersedParticles = V.getImmersedParticles(_tL)
    Np = V.getBEMParticlesNumber(_tL, pointer = True)
    Ramp = np.sin(min((it[0] + 1)/Ramp[0], 1.)*np.pi/2.)
    Fields = getDonorsFields(_tE, 'BEMDonorsIndices')
    BEM_BC = C.convertArray2Node(J.createZone('Zone', [Fields['CenterX'], Fields['CenterY'], \
                                                                         Fields['CenterZ']], 'xyz'))
    ax, ay, az, an = J.getVars(ImmersedParticles, V.vectorise('Alpha') + ['AlphaN'])
    ax[:] *= -1.
    ay[:] *= -1.
    az[:] *= -1.
    an[:] *= -1.
    J.invokeFields(BEM_BC, V.vectorise('Velocity'))
    Targets = C.newPyTree(['BC', BEM_BC])
    V.extract_velocity_BC(Targets, _tL, 0, 1)
    ax[:] *= -1.
    ay[:] *= -1.
    az[:] *= -1.
    an[:] *= -1.
    Particles = V.getBEMParticles(_tL)
    x, y, z = J.getxyz(Particles)
    ax, ay, az, an, s, nx, ny, nz = J.getVars(Particles, V.vectorise('Alpha') + ['AlphaN', 'Sigma'] +\
                                                                                V.vectorise('Normal'))
    ux, uy, uz = J.getVars(I.getZones(Targets)[0], V.vectorise('Velocity'))
    surf = np.square(s)#for the negative velocity
    ux *= surf
    uy *= surf
    uz *= surf
    x[:] = Fields['CenterX']
    y[:] = Fields['CenterY']
    z[:] = Fields['CenterZ']
    ax[:] = ny*uz - nz*uy
    ay[:] = nz*ux - nx*uz
    az[:] = nx*uy - ny*ux
    an[:] = nx*ux + ny*uy + nz*uz

def updateCFDSources(tL = [], tE = []):
    '''
    Updates the Eulerian Immersed particles embedded on the Inner Interface of the Hybrid Domain
    from the solution of the Eulerian Field.

    Parameters
    ----------
        tL : Tree
            Lagrangian field.

        tE : Tree
            Eulerian field.
    '''
    _tL, _tE = V.getTrees([tL, tE], ['Particles', 'Eulerian'])
    if not _tE or not _tL: return
    Particles = V.getImmersedParticles(_tL)
    it, Ramp, U0 = V.getParameters(_tL, ['CurrentIteration', 'StrengthRampAtbeginning', \
                                                                              'VelocityFreestream'])
    Ramp = np.sin(min((it[0] + 1)/Ramp[0], 1.)*np.pi/2.)
    Fields = getDonorsFields(_tE, 'CFDDonorsIndices', V.vectorise('Velocity'))
    x, y, z = J.getxyz(Particles)
    ax, ay, az, an, s, nx, ny, nz = J.getVars(Particles, V.vectorise('Alpha') + ['AlphaN', 'Sigma'] +\
                                                                                V.vectorise('Normal'))
    surf = np.square(s)*Ramp
    Fields['VelocityX'] = (Fields['VelocityX'] - U0[0])*surf
    Fields['VelocityY'] = (Fields['VelocityY'] - U0[1])*surf
    Fields['VelocityZ'] = (Fields['VelocityZ'] - U0[2])*surf
    x[:] = Fields['CenterX']
    y[:] = Fields['CenterY']
    z[:] = Fields['CenterZ']
    ax[:] = ny*Fields['VelocityZ'] - nz*Fields['VelocityY']
    ay[:] = nz*Fields['VelocityX'] - nx*Fields['VelocityZ']
    az[:] = nx*Fields['VelocityY'] - ny*Fields['VelocityX']
    an[:] = -(nx*Fields['VelocityX'] + ny*Fields['VelocityY'] + nz*Fields['VelocityZ'])

def computeEulerianNextTimeStep(tL = [], tE = [], tH = []):
    '''
    Advances the Eulerian field one lagrangian time step forward and updates both the BEM and
    Eulerian Immersed particles.

    Parameters
    ----------
        tL : Tree
            Lagrangian field.

        tE : Tree
            Eulerian field.

        tH : Tree
            Hybrid Domain.
    '''
    _tL, _tE, _tH = V.getTrees([tL, tE, tH], ['Particles', 'Eulerian', 'Hybrid'])
    if not _tE or not _tH or not _tL: return {}
    IterationInfo = {'Eulerian time': J.tic()}
    dtL, dtE = V.getParameters(_tL, ['TimeStep', 'EulerianTimeStep'])
    ndt = int(round(dtL[0]/dtE[0]))
    if ndt < 1: raise ValueError(J.FAIL + 'The Eulerian timestep (%g s) can not be bigger '%dtE + \
                                                 'than the Lagrangian timestep (%g s)'%dtL + J.ENDC)
    # BCM1 = V.getEulerianBC(_tH)
    # BC = induceEulerianBC(_tL, _tE)
    # for step in range(ndt):
    #     updateEulerianBC(_tE, interpolateEulerianBC(BC, BCM1, (step + 1)/ndt))
    computeFast(_tE)

    # computeEulerianVorticity(_tE, removeVelocityGradients = True)
    # updateCFDSources(_tL, _tE)#in that order
    # updateBEMSources(_tL, _tE)
    # storeEulerianBC(BC, BCM1)
    IterationInfo['Eulerian time'] = J.tic() - IterationInfo['Eulerian time']
    return IterationInfo

def shedVorticitySourcesFromHybridDomain(tL = [], tE = [], tH = []):
    '''
    Generates hybrid particles inside the Hybrid Domain.

    Parameters
    ----------
        tL : Tree
            Lagrangian field.

        tE : Tree
            Eulerian field.

        tH : Tree
            Hybrid Domain.
    '''
    _tL, _tE, _tH = V.getTrees([tL, tE, tH], ['Particles', 'Eulerian', 'Hybrid'])
    if not _tE or not _tL or not _tH: return {}

    IterationInfo = {'Eulerian generation time': J.tic()}
    Sources = filterHybridSources(_tL, _tE, _tH)
    IterationInfo['Number of shed particles Eulerian'] = -eraseParticlesInHybridDomain(_tL, _tH)
    IterationInfo.update(solveHybridParticlesStrength(Sources, _tL))
    redistributeVorticitySources(Sources, _tL)
    IterationInfo['Number of shed particles Eulerian'] += \
                                                     V.getParameter(_tL, 'NumberOfHybridSources')[0]
    IterationInfo['Eulerian generation time'] = J.tic() - IterationInfo['Eulerian generation time']
    return IterationInfo

def updateEulerianBC(tE = [], BC = []):
    '''
    Imposes the velocity, density and temperature on the far field BC of the Eulerian Domain..

    Parameters
    ----------
        tE : Tree
            Eulerian field.

        BC : Zone
            Containes the velocity to impose on the Eulerian BC.
    '''
    if not tE or not BC: return

    t = V.getEulerianBase(tE)
    RefState = I.getNodeFromName1(t, 'ReferenceState')
    rhoL = I.getValue(I.getNodeFromName1(RefState, 'Density'))
    TL = I.getValue(I.getNodeFromName1(RefState, 'Temperature'))
    ULx, ULy, ULz = J.getVars(I.getZones(BC)[0], V.vectorise('Velocity'))
    Nb = 0
    for i, zone in enumerate(I.getZones(t)):
        indices = tuple(I.getNodeByName(zone, 'BCFarFieldIndices')[1])
        # weightE = I.getNodeByName(zone, 'BCInterpolationWeight')[1]
        # weightL = 1. - weightE
        Nbnext = len(indices[0]) + Nb
        UEx = I.getNodeFromPath(zone, './FlowSolution#Centers/VelocityX')[1]
        UEy = I.getNodeFromPath(zone, './FlowSolution#Centers/VelocityY')[1]
        UEz = I.getNodeFromPath(zone, './FlowSolution#Centers/VelocityZ')[1]
        rhoE = I.getNodeFromPath(zone, './FlowSolution#Centers/Density')[1]
        TE = I.getNodeFromPath(zone, './FlowSolution#Centers/Temperature')[1]
        UEx[indices] = ULx[Nb: Nbnext]#*weightL + weightE*UEx[indices]
        UEy[indices] = ULy[Nb: Nbnext]#*weightL + weightE*UEy[indices]
        UEz[indices] = ULz[Nb: Nbnext]#*weightL + weightE*UEz[indices]
        rhoE[indices] = rhoL#*weightL + weightE*rhoE[indices]
        TE[indices] = TL#*weightL + weightE*TE[indices]
        Nb = Nbnext

def storeEulerianBC(newBC = [], oldBC = []):
    '''
    Copy the FlowSolution from oldBC to newBC.

    Parameters
    ----------
        newBC : Zone
            Containes the velocity to impose on the Eulerian BC at the current time step.

        oldBC : Tree
            Containes the velocity imposed on the Eulerian BC at the previous time step.
    '''
    for target, source in zip(I.getZones(oldBC), I.getZones(newBC)):
        FlowSolution = I.getNodeFromName1(target, 'FlowSolution')
        FlowSolutionM1 = I.getNodeFromName1(source, 'FlowSolution')
        for Field, FieldM1 in zip(FlowSolution[2], FlowSolutionM1[2]):
            Field[1][:] = FieldM1[1][:]

def interpolateEulerianBC(BC = [], BCM1 = [], ratio = 1.):
    '''
    Linearly interpolate the BC FlowSolution between BC and BCM1.

    Parameters
    ----------
        BC : Zone
            Containes the velocity to imposed on the Eulerian BC at the next time step.

        BCM1 : Zone
            Containes the velocity imposed on the Eulerian BC at the current time step.

        ratio : :py:class:`float`
            Interpolation weight. If ratio == 0, the current time step BC is returned. If ratio == 0
            , the previous time step BC is returned.
    Returns
    -------
        newBC : Zone
            Containes the interpolated BC.
    '''
    newBC = I.copyTree(BC)
    ratio2 = 1. - ratio
    for zone, zoneM1, newzone in zip(I.getZones(BC), I.getZones(BCM1), I.getZones(newBC)):
        FlowSolution = I.getNodeFromName1(zone, 'FlowSolution')
        FlowSolutionM1 = I.getNodeFromName1(zoneM1, 'FlowSolution')
        newFlowSolution = I.getNodeFromName1(newzone, 'FlowSolution')
        for Field, FieldM1, newField in zip(FlowSolution[2], FlowSolutionM1[2], newFlowSolution[2]):
            newField[1][:] = ratio*Field[1][:] + ratio2*FieldM1[1][:]

    return newBC

def getEulerianBC(tH = []):
    '''
    Gets the far field BC of the Eulerian field.

    Parameters
    ----------
        tH : Tree
            Hybrid Domain.
    Returns
    -------
        BC : Zone
            Containes the stored BC.
    '''
    _tH = V.getTrees([tH], ['Hybrid'])
    if not _tH: return []
    return I.getNodeFromName2(_tH, 'EulerianBC')

def induceEulerianBC(tL = [], tE = []):
    '''
    Induced the far field BC of the Eulerian field by all the particles, free, BEM and Immersed.

    Parameters
    ----------
        tL : Tree
            Lagrangian field.

        tE : Tree
            Eulerian field.
    Returns
    -------
        BC : Zone
            Induced BC.
    '''
    _tL, _tE = V.getTrees([tL, tE], ['Particles', 'Eulerian'])
    if not _tE or not _tL: return {}

    xt, yt, zt = [], [], []
    t, tc = V.getEulerianBases(_tE)
    zones = I.getZones(t)
    zonesc = I.getZones(tc)
    for zone, zonec in zip(zones, zonesc):
        flag = tuple(I.getNodeFromName(zone, 'BCFarFieldIndices')[1])
        xt += I.getNodeFromPath(zonec, './GridCoordinates/CoordinateX')[1][flag].tolist()
        yt += I.getNodeFromPath(zonec, './GridCoordinates/CoordinateY')[1][flag].tolist()
        zt += I.getNodeFromPath(zonec, './GridCoordinates/CoordinateZ')[1][flag].tolist()

    Zone = C.convertArray2Node(J.createZone('Zone', [np.array(xt), np.array(yt), np.array(zt)],
                                                                                             'xyz'))
    J.invokeFields(Zone, V.vectorise('Velocity'))
    BC = C.newPyTree(['BC', Zone])
    V.extract_velocity_BC(BC, _tL, 1, 1)
    return BC

def computeFastMetrics(tE = []):
    '''
    Warms up the Eulerian field for the FAST solver.

    Parameters
    ----------
        tE : Tree
            Eulerian field.
    '''
    _tE = V.getTrees([tE], ['Eulerian'])
    if not _tE: return

    t, tc = V.getEulerianBases(_tE)
    FastC.HOOK = None
    (t, tc, V.FastMetrics[0]) = FastS.warmup(t, tc)
    V.deletePrintedLines(len(I.getZones(t)))
    Bases = [I.getNodeFromName1(tE, 'CGNSLibraryVersion')]
    for base in I.getBases(tE):
        if base[0] == 'EulerianBase': Bases += [t]
        elif base[0] == 'EulerianBaseCenter': Bases += [tc]
        else: Bases += [base]

    tE[2] = Bases

def computeFast(tE = []):#, SubIterations = 1):
    '''
    Computes one Eulerian timestep.

    Parameters
    ----------
        tE : Tree
            Eulerian field.
    '''
    _tE = V.getTrees([tE], ['Eulerian'])
    if not _tE: return

    t, tc = V.getEulerianBases(_tE)
    CurrentIteration = I.getNodeFromName1(t, 'Iteration')
    time = I.getNodeFromName1(t, 'Time')
    dt = I.getNodeFromName(t, 'time_step')
    if CurrentIteration: CurrentIteration = I.getValue(CurrentIteration)
    else: CurrentIteration = 0
    if time: time = I.getValue(time)
    else: time = 0
    if dt: dt = I.getValue(dt)
    else: dt = 0
    nitrun, NIT = max(0, CurrentIteration), 1#, SubIterations
    FastS._compute(t = t, tc = tc, metrics = V.FastMetrics[0], nitrun = nitrun, NIT = NIT)
    if CurrentIteration == 1: V.deletePrintedLines()
    I.createUniqueChild(t, 'Iteration', 'DataArray_t', value = nitrun + NIT)
    I.createUniqueChild(t, 'Time', 'DataArray_t', value = time + NIT*dt) 

def computeEulerianVorticity(tE = [], removeVelocityGradients = False):
    '''
    Computes the Eulerian vorticity.

    Parameters
    ----------
        tE : Tree
            Eulerian field.

        removeVelocityGradients : :py:class:`bool`
            States whether the velocity gradients are conserved or deleted.
    '''
    _tE = V.getTrees([tE], ['Eulerian'])
    if not _tE: return
    t = V.getEulerianBase(_tE)
    FastS._computeGrad(t, V.FastMetrics[0], V.vectorise('Velocity'), 2)
    C._initVars(t, '{centers:VorticityX}={centers:gradyVelocityZ} - {centers:gradzVelocityY}')
    C._initVars(t, '{centers:VorticityY}={centers:gradzVelocityX} - {centers:gradxVelocityZ}')
    C._initVars(t, '{centers:VorticityZ}={centers:gradxVelocityY} - {centers:gradyVelocityX}')
    for dir in 'xyz':
        for var in V.vectorise('grad' + dir + 'Velocity'): I._rmNodesByName(t, var)
    for zone in I.getZones(t):
        SolverParam = I.getNodeFromName1(zone, '.Solver#VPM')
        index = tuple(I.getNodeFromName1(SolverParam, 'GhostCellsIndices')[1])
        FlowSolution = I.getNodeFromName1(zone, 'FlowSolution#Centers')
        for var in V.vectorise('Vorticity'):
            I.getNodeFromName1(FlowSolution, var)[1][index] = 0.

    fillGhostCells(tE, V.vectorise('Vorticity'))

def computeEulerianVelocityGradients(tE = []):
    '''
    Computes the Eulerian velocity gradients.

    Parameters
    ----------
        tE : Tree
            Eulerian field.
    '''
    _tE = V.getTrees([tE], ['Eulerian'])
    if not _tE: return

    t = V.getEulerianBase(_tE)
    FastS._computeGrad(t, V.FastMetrics[0], V.vectorise('Velocity'), 2)
    for zone in I.getZones(t):
        SolverParam = I.getNodeFromName1(zone, '.Solver#VPM')
        index = tuple(I.getNodeFromName1(SolverParam, 'GhostCellsIndices')[1])
        FlowSolution = I.getNodeFromName1(zone, 'FlowSolution#Centers')
        for dir in 'xyz':
            for var in V.vectorise('grad' + dir + 'Velocity'):
                I.getNodeFromName1(FlowSolution, var)[1][index] = 0.

    fillGhostCells(tE, V.vectorise('gradxVelocity') + V.vectorise('gradyVelocity') +
                                                                         V.vectorise('gradzVelocity'))

def rmGhostCells(tE = []):
    '''
    Deletes the ghost cells of the Eulerian mesh.

    Parameters
    ----------
        tE : Tree
            Eulerian field.
    '''
    return C.rmGhostCells(tE, tE, 2, adaptBCs=1)

def fillGhostCells(tE = [], vars = None):#does not work a 100% ... the very last row of ghostcells is not given in the connect match in tc or t ... the GhostCellsIndices are used to get rid of the f uped vorticity where the transfer is not done. Might use C._cpVars(t, 'centers:Vorticity', tc, 'Vorticity')
    '''
    Sets the solution on the ghost cells from the Eulerian mesh.

    Parameters
    ----------
        tE : Tree
            Eulerian field.

        vars : :py:class:`str`, list of :py:class:`str` or numpy.ndarray of :py:class:`str`
            Flow solutions to fill.
    '''
    if vars == None: vars = ['Density', 'Temperature'] + V.vectorise('Velocity') + \
                                                                            V.vectorise('Vorticity')
    t, tc = V.getEulerianBases(tE)
    if not t or not tc: return
    if isinstance(vars, str): vars = [vars]
    for zc in I.getZones(tc):
        zDonor = I.getNodeFromName1(t, zc[0])
        DonorFields = I.getNodeFromName1(zDonor, 'FlowSolution#Centers')
        if vars: DonorFields = [I.getNodeFromName1(DonorFields, var) for var in vars]
        else: DonorFields = DonorFields[2]

        DonorFields = [Field for Field in DonorFields if I.isType(Field, 'DataArray_t')]
        DonorFields = [[Field[0], np.ravel(Field[1], order = 'F')] for Field in DonorFields]
        for Match in I.getNodesFromType1(zc, 'ZoneSubRegion_t'):
            Donors = I.getNodeFromName1(Match, 'PointList')[1]
            Ghosts = I.getNodeFromName1(Match, 'PointListDonor')[1]
            zGhost = I.getNodeFromName1(t, I.getValue(Match))
            GhostFields = I.getNodeFromName1(zGhost, 'FlowSolution#Centers')
            for DonorField in DonorFields:
                GhostField = I.getNodeFromName1(GhostFields, DonorField[0])[1].ravel(order = 'F')
                GhostField[Ghosts] = DonorField[1][Donors]

