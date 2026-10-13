import os
from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
NumberOfProcessors = comm.Get_size()

import elsAxdt
elsAxdt.trace(0)

import Converter.PyTree as C
import Converter.Internal as I
import Converter.Mpi as Cmpi
import Converter.Filter as Filter
import Distributor2.PyTree as D2 
import etc.pypart.PyPart as PPA

if rank == 0:
    os.makedirs('OUTPUT', exist_ok=True)
    os.makedirs('LOGS', exist_ok=True)

    # Create an empty file coprocess.py
    # If it already exists, rename this file before
    if os.path.exists('coprocess.py'):
        os.rename('coprocess.py', 'coprocess_backup.py')
        open('coprocess.py', 'w').close()
comm.barrier()

def set_parameters_in_elsa_objects(filename):
    import elsA_user

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


    SolverParameters = dict()
    param_tree = Filter.readNodesFromPaths(filename, ['CGNSTree/WorkflowParameters/SolverParameters'])[0]
    for parent_name in ['cfdpb', 'model', 'numerics']:
        SolverParameters[parent_name] = dict()
        parent_node = I.getNodeFromName(param_tree, parent_name)
        for child in I.getChildren(parent_node):
            key, value = I.getName(child), I.getValue(child)
            if key == '.Solver#Function': 
                SolverParameters[parent_name]['.Solver#Function'] = dict()
                for node in I.getChildren(child):
                    key, value = I.getName(node), I.getValue(node)
                    SolverParameters[parent_name]['.Solver#Function'][key] = value
            else:
                SolverParameters[parent_name][key] = value

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

    funDict = get_cfl_function(NumDict)
    if funDict:
        set_cfl_function(elsA_user, Num, funDict)
    
    if SolverParameters['cfdpb']['config'] == "2d":
        Cfdpb.set_ghostcell(2,2,2,2,0,1)

    return Cfdpb, elsA_user

def remove_workflow_attributes_from_bases(tree):
    # Pypart put WorkflowParameters node, and all its children, bellow the Base
    # This function removes them
    attributes_names = ['Fluid', 'Flow', 'Turbulence', 'SplittingAndDistribution', 
                        'Overset', 'Numerics', 'Motion', 'Initialization', 'RunManagement', 
                        'ApplicationContext', 'SolverParameters']
    I._rmNodesFromName1(tree, 'WorkflowParameters')
    for base in I.getBases(tree):
        I._rmNodesFromName1(base, 'WorkflowParameters')
        for name in attributes_names:
            I._rmNodesFromName1(base, name)

Cfdpb, elsA_user = set_parameters_in_elsa_objects('main.cgns')

# Read tree with PyPart
PyPartBase = PPA.PyPart('main.cgns',
                        lksearch=['OUTPUT', '.'],
                        loadoption='partial',
                        mpicomm=MPI.COMM_WORLD,
                        LoggingInFile=False,
                        LoggingFile='partTree',
                        LoggingVerbose=40  # Filter: None=0, DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
                        )
# reorder=[6, 2] is recommended, mostly for unstructured mesh.
# It is also mandatory to use lussorscawf on unstructured mesh.
PartTree = PyPartBase.runPyPart(method=2, partN=1, reorder=[6, 2])
PyPartBase.finalise(PartTree, savePpart=True, method=1)
Skeleton = PyPartBase.getPyPartSkeletonTree()

# Put Distribution into the Skeleton
Distribution = PyPartBase.getDistribution()
for zone in I.getZones(Skeleton):
    zonePath = I.getPath(Skeleton, zone, pyCGNSLike=True)[1:]
    Cmpi._setProc(zone, Distribution[zonePath])

# Remove MOLA parameters from tree
remove_workflow_attributes_from_bases(PartTree)
remove_workflow_attributes_from_bases(Skeleton)

PartTree = I.merge([Skeleton, PartTree])

e = elsAxdt.XdtCGNS(tree=PartTree, links=[], paths=[])
e.distribution = D2.getProcDict(Skeleton, prefixByBase=True)


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
Cfdpb.compute()
Cfdpb.extract()

tsol = I.merge([PartTree, elsAxdt.get(elsAxdt.OUTPUT_TREE)])
PyPartBase.mergeAndSave(tsol, os.path.join('OUTPUT', 'PyPart_fields'))

if rank == 0:
    open('COMPLETED', 'a').close()
