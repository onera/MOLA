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
from mola.logging import mola_logger, MolaException
from mola.server import MaiaParallel

def apply(workflow):
    '''
    Initialize the flow solution.

    #. Compute FlowSolution#Init in all zones
    
    #. Adapt this node to the solver
    '''
    initialization_functions = dict(
        uniform = initialize_flow_with_reference_state,
        copy = initialize_flow_from_file_by_copy,
        interpolate = initialize_flow_from_file_by_interpolation,
    )

    initialize_flow_with_given_method = initialization_functions[workflow.Initialization['Method']]
    initialize_flow_with_given_method(workflow)

    initialize_flow_with_given_method(workflow)
    check_initial_flow_is_in_all_zones(workflow)
    # workflow.tree = compute_turbulent_distance_with_maia(workflow.tree)
    
    apply_to_solver(workflow)

def initialize_flow_with_reference_state(workflow):
    mola_logger.info('Initialize FlowSolution with uniform reference values')
    workflow.tree.newFields(workflow.Flow['ReferenceState'], Container='FlowSolution#Init')

def initialize_flow_from_file_by_interpolation(workflow):
    '''
    Initialize the flow solution of **t** from the flow solution in the file
    **sourceFilename**.
    Modify the tree **t** in-place.

    Parameters
    ----------

        workflow : :py:obj:`mola.workflow.worflow.Workflow`
    '''
    if isinstance(workflow.Initialization['Source'], str):
        mola_logger.info(f"Initialize FlowSolution by interpolation from {workflow.Initialization['Source']}")
    else:
        mola_logger.info(f"Initialize FlowSolution by interpolation from the given tree")
    
    raise Exception('Not yet implemented')

def initialize_flow_from_file_by_copy(workflow):
    '''
    Initialize the flow solution of **workflow.tree** by copying the flow solution in the file or tree
    **workflow.Initialization['Source']**.
    Modify the tree in-place.

    Parameters
    ----------

        workflow : :py:obj:`mola.workflow.worflow.Workflow`
    '''
    if isinstance(workflow.Initialization['Source'], str):
        mola_logger.info(f"Initialize FlowSolution by copy of {workflow.Initialization['Source']}")
    else:
        mola_logger.info(f"Initialize FlowSolution by copy of the given tree")

    keepTurbulentDistance = workflow.Initialization.get('KeepTurbulentDistance', False)

    sourceTree = cgns.load(workflow.Initialization['Source'])

    varNames = list(workflow.Flow['ReferenceState'])
    if keepTurbulentDistance:
        varNames += ['TurbulentDistance', 'TurbulentDistanceIndex']

    for zone in workflow.tree.zones():
        FSpath = zone.path() + '/FlowSolution#Init'
        try:
            FlowSolutionInSourceTree = sourceTree.getAtPath(FSpath)
            zone.addChild(FlowSolutionInSourceTree, override_sibling_by_name=True)
        except AttributeError:
            raise MolaException(f"The node {FSpath} is not found in {workflow.Initialization['Source']}")

def check_initial_flow_is_in_all_zones(workflow):
    for zone in workflow.tree.zones():
        if not zone.get(Name='FlowSolution#Init', Type='FlowSolution', Depth=1):
            raise MolaException(f'FlowSolution#Init is missing in zone {zone.name()}')

@MaiaParallel
def compute_turbulent_distance_with_maia(dist_tree):
    '''
    The input tree has to be distributed, read by maia.
    '''
    import maia
    import maia.pytree as PT
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # TODO Add test to check that the tree was read with maia
    # This function needs to be after the definition of boundary conditions

    part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
    maia.algo.part.compute_wall_distance(part_tree, comm) #, out_fs_name='FlowSolution#Init')  # create a FlowSolution container named WallDistance
    maia.transfer.part_tree_to_dist_tree_all(dist_tree, part_tree, comm)

    # If out_fs_name='FlowSolution#Init' is not used, we need to move the TurbulentDistance node
    for zone in PT.iter_all_Zone_t(dist_tree):
        FlowSolution = PT.get_child_from_name(zone, 'FlowSolution#Init')
        WallDistance =  PT.get_child_from_name(zone, 'WallDistance') 
        TurbulentDistance = PT.get_child_from_name(WallDistance, 'TurbulentDistance')
        PT.add_child(FlowSolution, TurbulentDistance)
        PT.rm_child(zone, WallDistance)

