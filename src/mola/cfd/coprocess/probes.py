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

import numpy as np
from treelab import cgns

from .tools import (
    mpi_allgather_and_merge_trees, 
    update_signals_using, 
)

def has_probes(workflow):
    return any([ext['Type']=='Probe' for ext in workflow.Extractions])

def search_zone_and_index_for_probes(coprocess_manager, comm):
    '''
    Search for the nearest vertex from each probe in **Extractions** in a PyTree.

    Notes
    -----
        - The function modifies the probe dictionaries by adding information about the zone, element, distance to the nearest vertex, and processor rank.
        - Probes that are too far from the nearest vertex are removed from the list.
    '''
    from mpi4py import MPI
    import Converter.PyTree as C
    import Converter.Internal as I
    import Geom.PyTree as D

    rank = comm.Get_rank()

    coprocess_manager.mola_logger.info('initialize probes', rank=0)

    # Put data at cell center, including coordinates
    # IMPORTANT: In this function, the mesh will be now the dual mesh, with nodes corresponding cell centers of the input mesh
    t = C.node2Center(coprocess_manager.workflow.tree)

    probesToKeep = []

    for Probe in coprocess_manager.Extractions:
        if Probe['Type'] != 'Probe':
            continue

        # Search the nearest points in all zones
        nearestElement = None
        minDistance = 1e20
        for zone in I.getZones(t):
            xnode = I.getNodeFromName(zone, 'CoordinateX')
            if xnode is None or I.getValue(xnode) is None:
                # This zone is a skeleton zone, so the current processor is not in charge of this zone
                continue

            if Probe['Method'] == 'getNearestPointIndex':
                element, squaredDistance = D.getNearestPointIndex(zone, tuple(Probe['Position']))
                distance = np.sqrt(squaredDistance)

            elif Probe['Method'] == 'nearestNodes':
                # Get the nearest node of the dual mesh 
                # Prefer this function C.nearestNodes to D.getNearestPointIndex for performance
                # (see https://elsa.onera.fr/issues/8236)
                hook = C.createGlobalHook(zone, function='nodes')
                nodes, distances = C.nearestNodes(hook, D.point(Probe['Position']))
                element, distance = nodes[0], distances[0]
            
            else:
                raise Exception('Method must be getNearestPointIndex or nearestNodes')

            if distance < minDistance:
                minDistance = distance
                nearestElement = element
                probeZone = zone
        
        Probe['rank'] = -1
        comm.barrier()
        minDistanceForAllProcessors = comm.allreduce(minDistance, op=MPI.MIN)
        if minDistance == minDistanceForAllProcessors:
            # Probe on this proc
            Probe['rank'] = rank
            Probe['zone'] = I.getName(probeZone)
            Probe['element'] = nearestElement
            Probe['distanceToNearestCellCenter'] = minDistance     
            x = I.getValue(I.getNodeFromName(probeZone, 'CoordinateX'))
            y = I.getValue(I.getNodeFromName(probeZone, 'CoordinateY'))
            z = I.getValue(I.getNodeFromName(probeZone, 'CoordinateZ'))
            Probe['Position'] = x.ravel(order='F')[nearestElement], y.ravel(order='F')[nearestElement], z.ravel(order='F')[nearestElement]
            if 'Name' not in Probe:
                xp, yp, zp = Probe['Position'][0], Probe['Position'][1], Probe['Position'][2]
                Probe['Name'] = f'Probe_{xp:.3g}_{yp:.3g}_{zp:.3g}'

        comm.barrier()
        rankForComm = comm.allreduce(Probe['rank'], op=MPI.MAX)
        comm.barrier()
        UpdatedProbe = comm.bcast(Probe, root=rankForComm)
        comm.barrier()
        Probe.update(UpdatedProbe)

        if minDistanceForAllProcessors > Probe['Tolerance']:
            coprocess_manager.mola_logger.warning(f'The probe {Probe["Name"]} is too far from the nearest vertex ({minDistanceForAllProcessors} m). It is removed.', rank=0)
        else:
            probesToKeep.append(Probe)

    # Overwrite extractions to keep only applicable probes
    coprocess_manager.Extractions = [extraction for extraction in coprocess_manager.Extractions if extraction['Type'] != 'Probe']  # all extractions except probes
    coprocess_manager.Extractions.extend(probesToKeep)  # add applicable probes

def extract_probe(output_tree: cgns.Tree, extraction: dict, coprocess_manager):

    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()

    t = cgns.Tree()
    base = cgns.Base(Name='Probes', Parent=t)

    if extraction['rank'] == rank:
    
        zone = cgns.Zone(Name=extraction['Name'], Parent=base)
        fs = cgns.Node(Name='FlowSolution', Type='FlowSolution', Parent=zone)

        cgns.Node(Name='Iteration', Type='DataArray', Parent=fs, Value=np.array([coprocess_manager.iteration]))
        if coprocess_manager.workflow.Numerics['TimeMarching'] == 'Unsteady': 
            time = coprocess_manager.iteration * coprocess_manager.workflow.Numerics['TimeStep']
            cgns.Node(Name='Time', Type='DataArray', Parent=fs, Value=np.array([time]))

        zone = output_tree.get(Name=extraction['zone'], Type='Zone')
        variablesDict = zone.allFields(ravel=True)

        if isinstance(extraction['Fields'], str):
            extraction['Fields'] = [extraction['Fields']]
            
        for var in extraction['Fields']:
            vp = variablesDict[var][extraction['element']]
            cgns.Node(Name=var, Type='DataArray', Parent=fs, Value=np.array([vp]))

    current_iteration_signals = mpi_allgather_and_merge_trees(t)

    if 'Data' in extraction and extraction['Data'] is not None:
        previous_signals_to_be_updated = extraction['Data']
        update_signals_using(current_iteration_signals, previous_signals_to_be_updated)
    else: 
        extraction['Data'] = current_iteration_signals

  
