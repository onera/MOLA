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

from treelab import cgns
from mola.cfd import apply_to_solver
from mola.logging import mola_logger, MolaException, MolaUserError
from mola.cfd.preprocess.mesh.tools import to_partitioned
from .initialization_with_turbo import initialize_flow_with_turbo

INIT_ANALYTICAL_METHODS = ['uniform', 'turbo']

def apply(workflow):
    '''
    Initialize the flow solution.

    #. Compute FlowSolution#Init in all zones

    #. Adapt this node to the solver
    '''
    FlowSolution_name = 'FlowSolution#Init'

    add_reference_state(workflow) 

    initialization_functions = dict(
        uniform = initialize_flow_with_reference_state,
        copy = initialize_flow_from_file_by_copy,
        interpolate = initialize_flow_from_file_by_interpolation,
        from_previous = initialize_flow_from_previous,
        turbo = initialize_flow_with_turbo,
    )
    
    is_dist = bool(workflow.tree.get(':CGNS#Distribution'))
    is_part = bool(workflow.tree.get(':CGNS#GlobalNumbering'))
    is_maia_tree = is_dist or is_part
    if is_maia_tree and workflow.Initialization['Method'] == 'copy':
        # 'copy' method is not available because splitting will be different
        mola_logger.warning("Method='copy' for initialization is not compatible with maia -> Method='interpolate' will be used instead.")
        workflow.Initialization['Method'] = 'interpolate'

    initialize_flow_with_given_method = initialization_functions[workflow.Initialization['Method']]

    initialize_flow_with_given_method(workflow, FlowSolution_name)
    check_initial_flow_is_in_all_zones(workflow, FlowSolution_name)
    compute_wall_distance_if_needed(workflow)
    
    apply_to_solver(workflow)



def add_reference_state(workflow):
    '''
    Add ``ReferenceState`` node to CGNS using user-provided conditions
    '''

    ReferenceState = dict(**workflow.Flow['ReferenceState'])

    for var in ['Mach','Pressure','Temperature']:
        ReferenceState[var] = workflow.Flow[var]


    namesForCassiopee = dict(
        cv                    = 'Cv',
        Gamma                 = 'Gamma',
        SutherlandViscosity   = 'Mus',
        SutherlandConstant    = 'Cs',
        SutherlandTemperature = 'Ts',
        Prandtl               = 'Pr',
    )
    # Needed by Fast, but not other solvers 
    for key, value in namesForCassiopee.items():
        ReferenceState[value] = workflow.Fluid[key]

    for base in workflow.tree.bases():
        base.setParameters('ReferenceState', ContainerType='ReferenceState', **ReferenceState)

def initialize_flow_with_reference_state(workflow, FlowSolution_name):
    mola_logger.info(' - initialize flow with uniform reference values',rank=0)
    workflow.tree.findAndRemoveNodes(Name=FlowSolution_name)
    workflow.tree.newFields(workflow.Flow['ReferenceState'], Container=FlowSolution_name, GridLocation='CellCenter')

def initialize_flow_from_file_by_interpolation(workflow, FlowSolution_name):
    '''
    Initialize the flow solution of **t** from the flow solution in the file **sourceFilename**.

    Parameters
    ----------

        workflow : :py:obj:`mola.workflow.worflow.Workflow`
    '''
    from mpi4py import MPI
    import maia

    if isinstance(workflow.Initialization['Source'], str):
        mola_logger.info(f" - initialize flow by interpolation from {workflow.Initialization['Source']}", rank=0)
        tree_source = maia.io.file_to_dist_tree(workflow.Initialization['Source'], MPI.COMM_WORLD)
        tree_source = cgns.castNode(tree_source)
    else:
        mola_logger.info(f" - initialize flow by interpolation from the given tree", rank=0)
        tree_source = workflow.Initialization['Source']

    workflow.Initialization.setdefault('SourceContainer', FlowSolution_name)
    # Rename FlowSolution nodes in the source tree if needed
    if workflow.Initialization['SourceContainer'] != FlowSolution_name:
        for FS in tree_source.group(Name=workflow.Initialization['SourceContainer'], Type='FlowSolution'):
            FS.setName(FlowSolution_name)

    # Check that all needed quantities are indeed in the container
    varNames = list(workflow.Flow['ReferenceState'])
    for FS in tree_source.group(Name=FlowSolution_name, Type='FlowSolution'):
        for var in varNames:
            if FS.get(Name=var, Depth=1) is None:
                raise MolaException(f'{var} cannot be found in {FS.path()}')
    
    tree_source = to_partitioned(tree_source)
    workflow.tree = to_partitioned(workflow.tree)

    maia.algo.part.interpolate(
        tree_source, 
        workflow.tree, 
        MPI.COMM_WORLD, 
        containers_name=[FlowSolution_name], 
        location='CellCenter',
        strategy='Closest',
        n_closest_pt=4,
        )
    
    if maia.pytree.get_node_from_name(tree_source, 'FlowSolution#Height'):
        maia.algo.part.interpolate(
            tree_source, 
            workflow.tree, 
            MPI.COMM_WORLD, 
            containers_name=['FlowSolution#Height'], 
            location='Vertex',
            strategy='Closest',
            n_closest_pt=4,
            )
    
    workflow.tree = cgns.castNode(workflow.tree)

def initialize_flow_from_file_by_copy(workflow, FlowSolution_name):
    '''
    Initialize the flow solution of **workflow.tree** by copying the flow solution in the file or tree
    **workflow.Initialization['Source']**.
    Modify the tree in-place.

    Parameters
    ----------

        workflow : :py:obj:`mola.workflow.worflow.Workflow`
    '''
    # FIXME Won't work if workflow.tree is a dist_tree or a part_tree (because zone names are modified)
    if isinstance(workflow.Initialization['Source'], str):
        mola_logger.info(f" - initialize flow by copy of {workflow.Initialization['Source']}", rank=0)
        errtag = workflow.Initialization['Source']
        tree_source = cgns.load(workflow.Initialization['Source'])
    else:
        mola_logger.info(f" - initialize flow by copy of the given tree", rank=0)
        tree_source = workflow.Initialization['Source']
        errtag = 'source tree'

    workflow.Initialization.setdefault('SourceContainer', FlowSolution_name)

    for zone in workflow.tree.zones():
        FSpath = zone.path() + '/' + workflow.Initialization['SourceContainer']
        try:
            FlowSolutionInSourceTree = tree_source.getAtPath(FSpath)
            assert FlowSolutionInSourceTree is not None
        except:
            raise MolaException(f"The node {FSpath} is not found in {errtag}")

        #Rename the container if needed
        if workflow.Initialization['SourceContainer'] != FlowSolution_name:
            FlowSolutionInSourceTree.setName(FlowSolution_name)

        # Check that all needed quantities are indeed in the container
        for var in list(workflow.Flow['ReferenceState']):
            if FlowSolutionInSourceTree.get(Name=var, Depth=1) is None:
                raise MolaException(f'{var} cannot be found in {FSpath}')

        zone.addChild(FlowSolutionInSourceTree, override_sibling_by_name=True)

        # Copy ChannelHeight if possible
        try:
            FSpath = zone.path() + '/FlowSolution#Height'
            FlowSolutionHeightInSourceTree = tree_source.getAtPath(FSpath)
            assert FlowSolutionInSourceTree is not None
            zone.addChild(FlowSolutionHeightInSourceTree, override_sibling_by_name=True)
        except:
            pass
            

def initialize_flow_from_previous(**kwargs):
    raise MolaException(
        'Initialization Method="from_previous" is a special method that should '
        'have been replaced during the "prepare" of a WorkflowManager. '
        'If you was using a Workflow directly (without WorkflowManager), '
        'then this Method should not be used.'
        )

def check_initial_flow_is_in_all_zones(workflow, FlowSolution_name):
    for zone in workflow.tree.zones():
        if not zone.get(Name=FlowSolution_name, Type='FlowSolution', Depth=1):
            raise MolaException(f'{FlowSolution_name} is missing in zone {zone.name()}')

def compute_wall_distance_if_needed(workflow):
    init_opts = workflow.Initialization
    if workflow.Turbulence['Model'] == 'Euler':
        init_opts['ComputeWallDistanceAtPreprocess'] = False
    elif workflow.Solver.lower() == 'fast':
        init_opts['ComputeWallDistanceAtPreprocess'] = True
        init_opts['WallDistanceComputingTool'] = 'cassiopee'

    if init_opts['ComputeWallDistanceAtPreprocess']:
        mola_logger.info(" - computing wall distance", rank=0)
        tool = init_opts['WallDistanceComputingTool']
        if tool == 'maia':
            workflow.tree = compute_wall_distance_with_maia(workflow.tree)
        elif tool == 'cassiopee':
            workflow.tree = compute_wall_distance_with_cassiopee(workflow.tree)
        else:
            raise MolaUserError(f"unsupported Initialization/WallDistanceComputingTool={tool}")
        
        force_grid_location_as_first_sibling(workflow.tree) # HACK

def compute_wall_distance_with_maia(tree: cgns.Tree):
    import maia
    import maia.pytree as PT
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # TODO compute row by row for turbomachinery application
    # TODO get out this function if no BC Wall*
    # This function needs to be after the definition of boundary conditions

    tree = to_partitioned(tree)
    maia.algo.part.compute_wall_distance(tree, comm)  # create a FlowSolution container named WallDistance
    tree = cgns.castNode(tree)
    for zone in tree.zones():
        FlowSolution = zone.get(Name='FlowSolution#Init')
        WallDistance = zone.get(Name='WallDistance')
        if not WallDistance: 
            continue
        TurbulentDistance = WallDistance.get(Name='TurbulentDistance')
        TurbulentDistance.dettach()
        TurbulentDistance.attachTo(FlowSolution)
        WallDistance.remove()

    return tree

def compute_wall_distance_with_cassiopee(tree: cgns.Tree):

    import Converter.PyTree as C
    import Converter.Mpi as Cmpi
    import Converter.Internal as I
    import Dist2Walls.PyTree as DTW

    old_container = I.__FlowSolutionCenters__
    I.__FlowSolutionCenters__ = 'FlowSolution#Init'
    walls = C.extractBCOfType(tree, 'BCWall')
    walls = C.newPyTree(['Base', walls])
    walls = Cmpi.allgatherTree(walls)
    walls = I.getZones(walls)
    if walls != []:
        DTW._distance2Walls(tree, walls, loc='centers', type='ortho')
    else: C._initVars(tree, 'centers:TurbulentDistance', 1000000.0)
    I.__FlowSolutionCenters__ = old_container
    tree = cgns.castNode(tree)

    return tree


def force_grid_location_as_first_sibling( tree : cgns.Tree ):
    
    for fs in tree.group(Type='FlowSolution_t', Depth=4):

        gl = fs.get(Type='GridLocation_t', Depth=1)

        if not gl: continue

        gl.dettach()
        gl.attachTo(fs, position=0)