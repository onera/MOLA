#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute self.iteration and/or modify
#    self.iteration under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that self.iteration will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

import os
from pathlib import Path
from fnmatch import fnmatch
import numpy as np

from treelab import cgns
from mola.logging import MolaException
import mola.naming_conventions as names

from . import rank, comm

def write_tagfile(tag : str, coprocess_manager):

    if rank == 0:
        if coprocess_manager is not None:
            run_dir = coprocess_manager.workflow.RunManagement.get('RunDirectory','.')
            run_dir_absolute = Path(run_dir).resolve() # added twice sometimes
            if not run_dir_absolute.is_dir(): # HACK
                run_dir_absolute = run_dir_absolute.parent

            filename_as_tag = run_dir_absolute / Path(tag)
        else:
            filename_as_tag =  tag

        with open(filename_as_tag, 'w') as f: 
            f.write(tag)


def write_extraction_log(extraction):
    def _check_data(extraction):
        try: 
            assert isinstance(extraction['Data'], cgns.Tree)
        except KeyError:
            raise MolaException(f"No 'Data' in extraction {extraction}")
        except AssertionError:
            raise MolaException(f"extraction['Data'] must be a cgns.Tree (now its type is {type(extraction['Data'])})")

    _check_data(extraction)
    extraction_log = dict((k,v) for k,v in extraction.items() if k not in ['Data', 'IsToExtract', 'IsToSave'])

    if extraction['Type'] in ['BC', 'IsoSurface']:
        for base in extraction['Data'].bases():
            base.setParameters(names.CGNS_NODE_EXTRACTION_LOG, **extraction_log)

    elif extraction['Type'] in ['Residuals', 'Integral', 'Probe']:
        for zone in extraction['Data'].zones():
            zone.setParameters(names.CGNS_NODE_EXTRACTION_LOG, **extraction_log)

def move_log_files():
    if rank == 0:
        dir_log = Path(names.DIRECTORY_LOG)
        dir_log.mkdir(exist_ok=True)

        for fn in Path('.').glob('*.log'):
            FilenameBase = str(fn)[:-4]
            i = 1
            NewFilename = Path(f'{FilenameBase}-{i}.log')
            while Path(dir_log / NewFilename).is_file():
                i += 1
                NewFilename = Path(f'{FilenameBase}-{i}.log')
            fn.rename(dir_log/NewFilename)

    comm.barrier()
    
def check_stderr():
    # TODO Simple check for now, but it should be different if this function is called in the coprocess script
    if rank == 0:
        try:
            with open(names.FILE_STDERR,'r') as f:
                Error = f.read()
            raise Exception(Error)
        except FileNotFoundError:
            pass

def mpi_allgather_and_merge_trees(local_tree : cgns.Tree, comm=comm ) -> cgns.Tree:

    comm.barrier()
    trees = comm.allgather(local_tree)
    merged_tree = cgns.merge(trees)
    comm.barrier() 

    return merged_tree

def update_signals_using( current_iteration_signals : cgns.Tree,
                          previous_signals_to_be_updated : cgns.Tree ) -> None:
    
    previous_tree = previous_signals_to_be_updated
    current_tree = current_iteration_signals

    for current_base in current_tree.bases():
        previous_base = previous_tree.get(Name=current_base.name(), Type='CGNSBase_t', Depth=1)
        if not previous_base:
            current_base.attachTo(previous_tree)
            continue

        for current_zone in current_base.zones():
            previous_zone = previous_base.get(Name=current_zone.name(), Type='Zone_t', Depth=1)
            if not previous_zone:
                current_zone.attachTo(previous_base)
                continue
        
            _update_signals_zones(current_zone, previous_zone)

def _update_signals_zones(current_zone : cgns.Zone, previous_zone : cgns.Zone) -> None:

    previous_flow_sol = previous_zone.get(Name='FlowSolution',Depth=1) # this is modified in-place
    current_flow_sol  =  current_zone.get(Name='FlowSolution',Depth=1)


    PreviousIterationsNode = previous_flow_sol.get(Name='Iteration',Type='DataArray_t',Depth=1)
    if not PreviousIterationsNode: return
    PreviousIterations = PreviousIterationsNode.value()
    CurrentIterationsNode = current_flow_sol.get(Name='Iteration',Type='DataArray_t',Depth=1)
    if not CurrentIterationsNode: return
    CurrentIterations = CurrentIterationsNode.value()


    override_all = True if CurrentIterations[0] <= PreviousIterations[0] else False
    stack_all = True if CurrentIterations[0] > PreviousIterations[-1] else False
    
    if override_all:
        _update_signals_container_overriding_all(previous_flow_sol, current_flow_sol)

    elif stack_all:
        _update_signals_container_stacking_all(previous_flow_sol, current_flow_sol)

    elif not override_all and not stack_all:
        _update_signals_container_stacking_partially(previous_flow_sol, current_flow_sol)
    
    else:
        raise MolaException(f"unexpected case override_all={override_all} stack_all={stack_all}")
    
    current_zone.updateShape()

def _update_signals_container_overriding_all(previous_flow_sol, current_flow_sol):

    for current_data in current_flow_sol.children():
        if current_data.type() != 'DataArray_t': continue

        previous_data = previous_flow_sol.get(current_data.name(),Type='DataArray_t',Depth=1)
        if not previous_data:
            current_data.attachTo(previous_flow_sol)
            continue

        previous_data.setValue(current_data.value())

def _update_signals_container_stacking_all(previous_flow_sol, current_flow_sol):

    for current_data in current_flow_sol.children():
        if current_data.type() != 'DataArray_t': continue
        
        previous_data = previous_flow_sol.get(current_data.name(),Type='DataArray_t',Depth=1)
        if not previous_data:
            previous_it = previous_flow_sol.get('Iteration',Type='DataArray_t',Depth=1).value()
            previous_value = np.empty_like(previous_it)
            previous_value[:] = np.nan
        else:
            previous_value = previous_data.value()
        
        current_value = current_data.value()
        updated_value = np.hstack((previous_value, current_value))

        previous_data.setValue(updated_value)

def _update_signals_container_stacking_partially(previous_flow_sol, current_flow_sol):

    PreviousIterations = previous_flow_sol.get(Name='Iteration',Type='DataArray_t',Depth=1).value()
    CurrentIterations = current_flow_sol.get(Name='Iteration',Type='DataArray_t',Depth=1).value()

    ε = 1e-12
    UpdatePortion = PreviousIterations > (CurrentIterations[0] - ε)
    if len(UpdatePortion) == 1 and not UpdatePortion[0]:
        FirstPreviousIndex2Update = len(PreviousIterations) - 1 
    else:
        try:
            FirstPreviousIndex2Update = np.where(UpdatePortion)[0][0]
        except IndexError:
            msg = "FATAL: add case to test_update_signals:\n"
            msg+= f'PreviousIterations:\n{PreviousIterations}\n'
            msg+= f'CurrentIterations:\n{CurrentIterations}\n'
            msg+= f'UpdatePortion={UpdatePortion}\n'
            msg+= f'np.where(UpdatePortion)={np.where(UpdatePortion)}'
            raise IndexError(msg)

    for current_data in current_flow_sol.children():
        if current_data.type() != 'DataArray_t': continue

        previous_data = previous_flow_sol.get(current_data.name(),Type='DataArray_t',Depth=1)
        if not previous_data:
            previous_it = previous_flow_sol.get('Iteration',Type='DataArray_t',Depth=1).value()
            previous_value = np.empty_like(previous_it)
            previous_value[:] = np.nan
        else:
            previous_value = previous_data.value()
        
        current_value = current_data.value()
        updated_value = np.hstack((previous_value[:FirstPreviousIndex2Update], current_value))
        previous_data.setValue(updated_value)

def get_bc_families_in_extraction(extraction, DictBCNames2Type):
    families = []
    for BCFamilyName in DictBCNames2Type:
        BCType = DictBCNames2Type[BCFamilyName]
        if fnmatch(BCType, extraction['Source']):
            # Case of source matching one or several names of BC: 'BCWall', 'BCInflow*', '*', etc.
            source = BCType
            family = BCFamilyName
        elif fnmatch(BCFamilyName, extraction['Source']):
            # Case of source matching a family name
            source = BCFamilyName
            family = BCFamilyName
        else:
            continue
        families.append(family)

    return families
    
def extract_memory_usage(extraction, iteration):
    import psutil

    t = cgns.Tree()

    SLURM_CPUS_ON_NODE = os.getenv('SLURM_CPUS_ON_NODE')
    if SLURM_CPUS_ON_NODE is None:
        extraction['Data'] = t
        return 
    else:
        CoreNumberPerNode = int(SLURM_CPUS_ON_NODE)
    
    if rank % CoreNumberPerNode == 0:
        base = cgns.Base(Name='MemoryUsage', Parent=t)
        zone = cgns.Zone(Name=f'MemoryUsageOfProc{rank}', Parent=base)
        fs = cgns.Node(Name='FlowSolution', Type='FlowSolution', Parent=zone)
        cgns.Node(Name='Iteration', Type='DataArray', Parent=fs, Value=np.array([iteration]))
        cgns.Node(Name='UsedMemoryInPercent', Type='DataArray', Parent=fs, Value=np.array([psutil.virtual_memory().percent]))
        cgns.Node(Name='UsedMemory', Type='DataArray', Parent=fs, Value=np.array([psutil.virtual_memory().used]))

    current_iteration_signals = mpi_allgather_and_merge_trees(t)

    if 'Data' in extraction and extraction['Data'] is not None:
        previous_signals_to_be_updated = extraction['Data']
        update_signals_using(current_iteration_signals, previous_signals_to_be_updated)
    else: 
        extraction['Data'] = current_iteration_signals

def remove_not_needed_fields(extraction):
    '''Remove nodes that are not required in Fields'''
    if 'Data' not in extraction or extraction['Data'] is None:
        return
    
    COORDINATES = ['ChannelHeight', 'Radius', 'Theta']

    for FS in extraction['Data'].group(Type='FlowSolution'):
        for node in FS.group(Type='DataArray', Depth=1):
            if node.name() in COORDINATES:
                continue
            if 'Fields' not in extraction or node.name() not in extraction['Fields']:
                node.remove()
        if len(FS.group(Type='DataArray', Depth=1)) == 0:
            FS.remove()
