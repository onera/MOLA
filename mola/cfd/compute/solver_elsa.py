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

def adapt_to_solver(workflow):
    
    # ----------------------- IMPORT SYSTEM MODULES ----------------------- #
    import os
    from mpi4py import MPI
    comm   = MPI.COMM_WORLD
    rank   = comm.Get_rank()
    NumberOfProcessors = comm.Get_size()

    # ------------------------- IMPORT  CASSIOPEE ------------------------- #
    import Converter.PyTree as C
    import Converter.Internal as I
    import Converter.Filter as Filter
    import Converter.Mpi as Cmpi

    # ------------------------------ SETTINGS ------------------------------ #
    # TODO: List all MOLA keywords in mola.__init__.py ? 
    FULL_CGNS_MODE   = False
    FILE_CGNS        = 'main.cgns'
    FILE_SURFACES    = 'surfaces.cgns'
    FILE_ARRAYS      = 'arrays.cgns'
    FILE_FIELDS      = 'tmp-fields.cgns' # BEWARE of tmp- suffix
    FILE_COLOG       = 'coprocess.log'
    DIRECTORY_OUTPUT = 'OUTPUT'
    DIRECTORY_LOGS   = 'LOGS'

    if rank==0:
        os.makedirs(DIRECTORY_OUTPUT, exist_ok=True)
        os.makedirs(DIRECTORY_LOGS, exist_ok=True)

    # --------------------------- END OF IMPORTS --------------------------- #

    # ----------------- DECLARE ADDITIONAL GLOBAL VARIABLES ----------------- #
    # CO.invokeCoprocessLogFile()
    # arrays = CO.invokeArrays()

    # if workflow.Numerics['NumberOfIterations'] == 0:
    #     CO.printCo('WARNING: niter = 0 -> will only make extractions', proc=0, color=J.YELLOW)
    # inititer = setup.elsAkeysNumerics['inititer']
    # itmax    = inititer+niter-2 # BEWARE last iteration accessible trigger-state-16

    # Skeleton = CO.loadSkeleton()

    # ========================== LAUNCH ELSA ========================== #

    launch_elsa_computation(workflow, FILE_CGNS)


def launch_elsa_computation(workflow, FILE_CGNS):
    import elsA_user
    if not hasattr(workflow, '_FULL_CGNS_MODE'):

        Cfdpb = elsA_user.cfdpb(name='cfd')
        Mod   = elsA_user.model(name='Mod')
        Num   = elsA_user.numerics(name='Num')

        CfdDict  = workflow.SolverParameters['cfdpb']
        ModDict  = workflow.SolverParameters['model']
        NumDict  = workflow.SolverParameters['numerics']

        elsAobjs = [Cfdpb,   Mod,     Num]
        elsAdics = [CfdDict, ModDict, NumDict]

        for obj, dic in zip(elsAobjs, elsAdics):
            [obj.set(v,dic[v]) for v in dic if not isinstance(dic[v], dict)]

        for k in NumDict:
            if '.Solver#Function' in k:
                funDict = NumDict[k]
                funName = funDict['name']
                if funName == 'f_cfl':
                    f_cfl=elsA_user.function(funDict['function_type'],name=funName)
                    for v in funDict:
                        if v in ('iterf','iteri','valf','vali'):
                            f_cfl.set(v,  funDict[v])
                    Num.attach('cfl', function=f_cfl)

    import elsAxdt
    elsAxdt.trace(0)

    if workflow.SplittingAndDistribution['Strategy'].lower() == 'atcomputation':
        if workflow.SplittingAndDistribution['Splitter'].lower() == 'pypart':
            from ..preprocess.mesh.split import splitWithPyPart
            t, Skeleton, PyPartBase, Distribution = splitWithPyPart()
        elif workflow.SplittingAndDistribution['Splitter'].lower() == 'maia':
            from ..preprocess.mesh.split import splitWithMaia
            t, Distribution = splitWithMaia()
        else:
            raise Exception(f"Unkwown Splitter: {workflow.SplittingAndDistribution['Splitter']}")
        e = elsAxdt.XdtCGNS(tree=t, links=[], paths=[])
        e.distribution = Distribution
    else:
        e = elsAxdt.XdtCGNS(FILE_CGNS)

    e.action=elsAxdt.COMPUTE
    e.mode=elsAxdt.READ_ALL
    e.compute()
    e.save('solution.cgns')


def compute_test(workflow):
    
    # ----------------------- IMPORT SYSTEM MODULES ----------------------- #
    import sys
    import os
    import numpy as np
    import shutil
    import timeit
    LaunchTime = timeit.default_timer()
    from mpi4py import MPI
    comm   = MPI.COMM_WORLD
    rank   = comm.Get_rank()
    NumberOfProcessors = comm.Get_size()

    # ------------------------- IMPORT  CASSIOPEE ------------------------- #
    import Converter.PyTree as C
    import Converter.Internal as I
    import Converter.Filter as Filter
    import Converter.Mpi as Cmpi

    # ------------------------------ SETTINGS ------------------------------ #
    FULL_CGNS_MODE   = False
    FILE_CGNS        = 'main.cgns'
    FILE_SURFACES    = 'surfaces.cgns'
    FILE_ARRAYS      = 'arrays.cgns'
    FILE_FIELDS      = 'tmp-fields.cgns' # BEWARE of tmp- suffix
    FILE_COLOG       = 'coprocess.log'
    DIRECTORY_OUTPUT = 'OUTPUT'
    DIRECTORY_LOGS   = 'LOGS'

    if rank==0:
        try: os.makedirs(DIRECTORY_OUTPUT)
        except: pass
        try: os.makedirs(DIRECTORY_LOGS)
        except: pass

    # --------------------------- END OF IMPORTS --------------------------- #

    # ========================== LAUNCH ELSA ========================== #

    import elsA_user
    if not FULL_CGNS_MODE:

        Cfdpb = elsA_user.cfdpb(name='cfd')
        Mod   = elsA_user.model(name='Mod')
        Num   = elsA_user.numerics(name='Num')

        CfdDict  = workflow.SolverParameters['cfdpb']
        ModDict  = workflow.SolverParameters['model']
        NumDict  = workflow.SolverParameters['numerics']

        elsAobjs = [Cfdpb,   Mod,     Num]
        elsAdics = [CfdDict, ModDict, NumDict]

        for obj, dic in zip(elsAobjs, elsAdics):
            [obj.set(v,dic[v]) for v in dic if not isinstance(dic[v], dict)]

        for k in NumDict:
            if '.Solver#Function' in k:
                funDict = NumDict[k]
                funName = funDict['name']
                if funName == 'f_cfl':
                    f_cfl=elsA_user.function(funDict['function_type'],name=funName)
                    for v in funDict:
                        if v in ('iterf','iteri','valf','vali'):
                            f_cfl.set(v,  funDict[v])
                    Num.attach('cfl', function=f_cfl)

    import elsAxdt
    elsAxdt.trace(0)
    # CO.elsAxdt = elsAxdt

    if workflow.SplittingAndDistribution['Strategy'] == 'AtComputation':
        if workflow.SplittingAndDistribution['Splitter'] == 'PyPart':
            e = elsAxdt.XdtCGNS(tree=t, links=[], paths=[])
            e.distribution = Distribution
        elif workflow.SplittingAndDistribution['Splitter'] == 'maia':
            import maia
            dist_tree  = maia.io.file_to_dist_tree(FILE_CGNS, comm)
            zone_to_parts = maia.factory.partitioning.compute_balanced_weights(dist_tree, comm)
            part_tree = maia.factory.partition_dist_tree(dist_tree, comm, zone_to_parts=zone_to_parts)
            C.convertPyTree2File(part_tree, f'part_tree_{rank}.cgns')
            comm.barrier()
            e = elsAxdt.XdtCGNS(tree=part_tree, links=[], paths=[])
            # e.distribution = Distribution
        else:
            raise Exception(f"Unknown splitter: {workflow.SplittingAndDistribution['Splitter']}")
    else:
        e=elsAxdt.XdtCGNS(FILE_CGNS)

    e.action=elsAxdt.COMPUTE
    e.mode=elsAxdt.READ_ALL
    e.compute()
    e.save('solution.cgns')
