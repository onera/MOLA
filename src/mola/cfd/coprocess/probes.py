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

import Converter.PyTree as C

def hasProbes():
    for Extraction in setup.Extractions:
        if Extraction['type'] == 'Probe':
            return True
    return False

def searchZoneAndIndexForProbes(t, method='getNearestPointIndex', tol=1e-2):
    
    '''
    Search for the nearest vertex from each probe in **setup.Extractions** in a PyTree.

    Parameters
    ----------
    t : PyTree
        Input PyTree.

    method : str
        One of 'getNearestPointIndex' (from Cassiopee Geom module) or 'nearestNodes' (from Converter module).

    tol : float, optional
        The tolerance for minimum distance. Default is 1e-2.

    Notes
    -----
        - The function modifies the probe dictionaries by adding information about the zone, element, distance to the nearest vertex, and processor rank.
        - Probes that are too far from the nearest vertex are removed from the list.
    '''
    # Put data at cell center, including coordinates
    # IMPORTANT: In this function, the mesh will be now the dual mesh, with nodes corresponding cell centers of the input mesh
    t = C.node2Center(t)

    probesToKeep = []

    for Probe in setup.Extractions:
        if Probe['type'] != 'Probe':
            continue

        # Search the nearest points in all zones
        nearestElement = None
        minDistance = 1e20
        for zone in I.getZones(t):
            x = J.getx(zone)
            if x is None:
                # This zone is a skeleton zone, so the current processor is not in charge of this zone
                continue

            if method == 'getNearestPointIndex':
                element, squaredDistance = D.getNearestPointIndex(zone, Probe['location'])
                distance = np.sqrt(squaredDistance)

            elif method == 'nearestNodes':
                # Get the nearest node of the dual mesh 
                # Prefer this function C.nearestNodes to D.getNearestPointIndex for performance
                # (see https://elsa.onera.fr/issues/8236)
                hook = C.createGlobalHook(zone, function='nodes')
                nodes, distances = C.nearestNodes(hook, D.point(Probe['location']))
                element, distance = nodes[0], distances[0]
            
            else:
                raise Exception('method must be getNearestPointIndex or nearestNodes')

            if distance < minDistance:
                minDistance = distance
                nearestElement = element
                probeZone = zone
        
        Probe['rank'] = -1
        Cmpi.barrier()
        minDistanceForAllProcessors = comm.allreduce(minDistance, op=MPI.MIN)
        if minDistance == minDistanceForAllProcessors:
            # Probe on this proc
            Probe['rank'] = rank
            Probe['zone'] = I.getName(probeZone)
            Probe['element'] = nearestElement
            Probe['distanceToNearestCellCenter'] = minDistance     
            x, y, z = J.getxyz(probeZone)
            Probe['location'] = x.ravel(order='F')[nearestElement], y.ravel(order='F')[nearestElement], z.ravel(order='F')[nearestElement]
            if 'name' not in Probe:
                Probe['name'] = 'Probe_{:.3g}_{:.3g}_{:.3g}'.format(Probe['location'][0], Probe['location'][1], Probe['location'][2])
        Cmpi.barrier()
        rankForComm = comm.allreduce(Probe['rank'], op=MPI.MAX)
        Cmpi.barrier()
        UpdatedProbe = comm.bcast(Probe, root=rankForComm)
        Cmpi.barrier()
        Probe.update(UpdatedProbe)

        if minDistanceForAllProcessors > tol:
            printCo(f'The probe {Probe["name"]} is too far from the nearest vertex ({minDistanceForAllProcessors} m). It is removed.', 0, J.WARN)
        else:
            probesToKeep.append(Probe)

    # Overwrite extractions to keep only applicable probes
    setup.Extractions = [extraction for extraction in setup.Extractions if extraction['type'] != 'Probe']  # all extractions except probes
    setup.Extractions.extend(probesToKeep)  # add applicable probes
  

def appendProbes2Arrays(t, arrays):
    '''
    Append probes with picked data in **arrays**.

    Parameters
    ----------

        t : PyTree

        arrays : dict

    '''
    for Probe in setup.Extractions:
        if Probe['type'] != 'Probe':
            continue
        if Probe['rank'] != rank:
            continue
        ProbesDict = dict( IterationNumber = CurrentIteration-1 )
        if setup.elsAkeysNumerics['time_algo'] != 'steady': 
            ProbesDict['Time'] = ProbesDict['IterationNumber'] * setup.elsAkeysNumerics['timestep']

        variables = Probe['variables']
        if isinstance(variables, str):
            variables = [variables]
        zone = I.getNodeFromName2(t, Probe['zone'])
        variablesDict = J.getVars2Dict(zone, VariablesName=variables, Container='FlowSolution#Init')
        for var, value in variablesDict.items():
            ProbesDict[var] = value.ravel('F')[Probe['element']]

        appendDict2Arrays(arrays, ProbesDict, Probe['name'])

  
