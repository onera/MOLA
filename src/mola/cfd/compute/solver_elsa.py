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

# ----------------------- IMPORT SYSTEM MODULES ----------------------- #
import os
from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
NumberOfProcessors = comm.Get_size()
import numpy as np

from treelab import cgns
import mola.naming_conventions as names
from mola.cfd.compute.read_cfd_files import read_cfd_files
import mola.pytree.InternalShortcuts as J

import Converter.Internal as I

def apply_to_solver(workflow):

    import elsAxdt
    elsAxdt.trace(0)

    Cfdpb, elsA_user = set_parameters_in_elsa_objects(workflow.SolverParameters, workflow.Numerics)

    e = read_cfd_files.apply(workflow)

    from mola.cfd.coprocess.manager import CoprocessManager
    coprocess_manager = CoprocessManager(workflow)
    workflow._coprocess_manager = coprocess_manager

    if workflow.has_moving_overset_component():
        loadMotionForElsA(elsA_user, workflow._Skeleton, coprocess_manager.mola_logger)

    e.mode  = elsAxdt.READ_MESH
    e.mode |= elsAxdt.READ_CONNECT
    e.mode |= elsAxdt.READ_BC
    e.mode |= elsAxdt.READ_BC_INIT
    e.mode |= elsAxdt.READ_INIT
    e.mode |= elsAxdt.READ_FLOW
    e.mode |= elsAxdt.READ_COMPUTATION
    e.mode |= elsAxdt.READ_OUTPUT
    e.mode |= elsAxdt.READ_TRACE
    e.mode |= elsAxdt.SKIP_GHOSTMASK # NOTE https://elsa.onera.fr/issues/3480
    e.action=elsAxdt.TRANSLATE

    e.compute()

    if rank==0:
        table = e.symboltable()
        with open(os.path.join(names.DIRECTORY_LOG, f'symbol_table.log'), 'w') as f:
            import pprint
            f.write(pprint.pformat(table))

    if workflow.has_overset_component():
        readStaticMasksForElsA(e, elsA_user, workflow._Skeleton, coprocess_manager.mola_logger)

    if workflow.has_moving_overset_component():
        loadUnsteadyMasksForElsA(e, elsA_user, workflow._Skeleton, coprocess_manager.mola_logger)

    Cfdpb.compute()
    Cfdpb.extract()

    coprocess_manager.finalize()
    del workflow._coprocess_manager
    

def set_parameters_in_elsa_objects(SolverParameters, Numerics):
    import elsA_user

    Cfdpb = elsA_user.cfdpb(name='cfd')
    Mod   = elsA_user.model(name='Mod')
    Num   = elsA_user.numerics(name='Num')

    CfdDict  = SolverParameters['cfdpb']
    ModDict  = SolverParameters['model']
    NumDict  = SolverParameters['numerics']

    elsAobjs = [Cfdpb,   Mod,     Num]
    elsAdics = [CfdDict, ModDict, NumDict]

    for obj, dic in zip(elsAobjs, elsAdics):
        [obj.set(v,dic[v]) for v in dic if not isinstance(dic[v], dict)]

    Num.set('niter', Numerics['NumberOfIterations'])
    Num.set('inititer', Numerics['IterationAtInitialState'])
    Num.set('itime', Numerics['TimeAtInitialState'])


    funDict = get_cfl_function(NumDict)
    if funDict:
        set_cfl_function(elsA_user, Num, funDict)
    
    if SolverParameters['cfdpb']['config'] == "2d":
        Cfdpb.set_ghostcell(2,2,2,2,0,1)

    return Cfdpb, elsA_user

def get_cfl_function(NumDict):
    for k in NumDict:
        if '.Solver#Function' in k:
            funDict = NumDict[k]
            if funDict['name'] == NumDict['cfl_fct']:
                return funDict
    
    return None

def set_cfl_function(elsA_user, Num, funDict):
    f_cfl = elsA_user.function(funDict['function_type'], name=funDict['name'])
    for v in ('iterf','iteri','valf','vali'):
        f_cfl.set(v,  funDict[v])
    Num.attach('cfl', function=f_cfl)


def loadUnsteadyMasksForElsA(e, elsA_user, Skeleton, logger):
    
    comm.barrier()
    AllMaskedZones = dict()
    for base in I.getBases(Skeleton):
        elsA_masks = []
        masks = I.getNodeFromName1(base, '.MOLA#Masks')
        if not masks: continue

        for mask in masks[2]:
            mask_name = I.getValue(mask).replace('.','_').replace('-','_')
            WndNames, ZonePaths, PtWnds = [], [], []
            for i, patch in enumerate(I.getNodesFromName(mask,'patch*')):
                zone_name = I.getValue(I.getNodeFromName1(patch,'Zone'))
                base_name = J._getBaseWithZoneName(Skeleton, zone_name)[0]
                wnd_node = I.getNodesFromName(patch,'Window*')[0]
                wnd_node_path = "/".join([base[0], masks[0], mask[0], patch[0], wnd_node[0]])
                wnd_node_from_file = cgns.load_from_path(names.FILE_INPUT_SOLVER, wnd_node_path)
                w = wnd_node_from_file.value()
                wnd_node[1] = w
                wnd_name = 'wnd_'+mask_name+'%d'%i
                wnd_name = wnd_name.replace('-','_').replace('.','_')
                WndNames += [ wnd_name ]
                ZonePaths += [ base_name + '/' +  zone_name ]
                PtWnds += [ [ int(w[0,0]), int(w[0,1]),
                              int(w[1,0]), int(w[1,1]),
                              int(w[2,0]), int(w[2,1])] ]

            elsA_windows = []
            for name, path, wnd in zip(WndNames, ZonePaths, PtWnds):
                elsA_windows += [elsA_user.window(
                                   e.e_getBlockInternalName(path), name=name)]
                elsA_windows[-1].set('wnd',wnd)
                elsA_windows[-1].show()

            logger.info('setting unsteady mask '+mask_name,rank=0)
            elsA_masks += [ elsA_user.mask( ' '.join(WndNames), name=mask_name ) ] 
            
            Parameters = J.get(mask,'Parameters')
            for p in Parameters:
                value = Parameters[p]
                if isinstance(value, np.ndarray):
                    dtype = str(value.dtype)
                    if dtype.startswith('int'):
                        value = int(value)
                    elif dtype.startswith('float'):
                        value = float(value)
                    else: 
                        raise TypeError('FATAL: numpy dtype %s not supported'%dtype)
                    Parameters[p] = value
                

            elsA_masks[-1].setDict(Parameters)

            Neighbours = I.getValue(I.getNodeFromName(mask,'MaskedZones')).split(' ')
            for n in Neighbours:
                elsA_masks[-1].attach(e.e_getBlockInternalName(n))
                if n not in AllMaskedZones:
                    AllMaskedZones[n] = ZonePaths
                else:
                    for zp in ZonePaths:
                        if zp not in AllMaskedZones[n]:
                            AllMaskedZones[n] += [ zp ]
            elsA_masks[-1].show()
    comm.barrier()

    # create ghost masks
    for ghost_zonename in AllMaskedZones:
        ghost_name = 'maskG_'+ghost_zonename
        ghost_name = ghost_name.replace('.','_').replace('/','_').replace('-','_')
        ghost_mask = elsA_user.mask( e.blockwindow(ghost_zonename), name=ghost_name)
        ghost_mask.set('type','ghost')
        for interpolant_name in AllMaskedZones[ghost_zonename]:
            ghost_mask.attach( e.blockname(interpolant_name) )

    comm.barrier()

def readStaticMasksForElsA(e, elsA_user, Skeleton, logger):
    
    comm.barrier()
    bases = I.getBases(Skeleton)
    for base in bases:

        meshInfo = J.get(base,'.MOLA#InputMesh')
        if 'Motion' in meshInfo: continue
        
        for zone in I.getZones(base):
            mask_file = os.path.join(names.DIRECTORY_OVERSET,
                                     'hole_%s_%s.v3d'%(base[0],zone[0]))
            
            if os.path.isfile(mask_file):
                baseAndZoneName = base[0]+'/'+zone[0]
                blockname = e.blockname( baseAndZoneName )
                mask_name = 'staticMask_%s_%s'%(base[0],zone[0])
                mask_name = mask_name.replace('.','_')
                logger.info('setting static mask %s at base %s'%(mask_file,base[0]), rank=0)
                mask = elsA_user.mask(e.blockwindow(baseAndZoneName), name=mask_name)
                mask.set('type', 'file')
                mask.set('file', mask_file)
                mask.set('format', 'bin_v3d')
                mask.attach(blockname)
    comm.barrier()

def loadMotionForElsA(elsA_user, Skeleton, logger):
    comm.barrier()
    
    AllMotions = []
    bases = I.getBases(Skeleton)
    for base in bases:

        motion = I.getNodeFromName2(base, '.Solver#Motion')
        if not motion:
            raise ValueError(f".Solver#Motion not found in {base[0]}")


        function_name = I.getNodeFromName1(motion, 'function_name')
        if not function_name:
            raise ValueError(f"function_name not found in {motion[0]}")

        function_name = I.getValue(function_name)

        MOLA_motion = I.getNodeFromName2(base, '.MOLA#Motion')
        Parameters = dict()
        for p in MOLA_motion[2]:
            parameter_name = p[0]
            value = I.getValue(p)
            if isinstance(value, np.ndarray): value = value.tolist()
            Parameters[parameter_name] = value
        
        logger.info('setting elsA motion function %s at base %s'%(function_name,base[0]), rank=0)
        AllMotions.append(elsA_user.function(Parameters['type'],name=function_name))
        AllMotions[-1].setDict(Parameters)
        AllMotions[-1].show()
    comm.barrier()
