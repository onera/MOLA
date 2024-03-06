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
from mola import misc

def apply(workflow):
    '''
    Initialize the flow solution.

    #. Compute FlowSolution#Init in all zones
    
    #. Adapt this node to the solver
    '''

    if workflow.Initialization['method'] is None:
        pass
    elif workflow.Initialization['method'] == 'uniform':
        print(misc.CYAN + 'Initialize FlowSolution with uniform reference values' + misc.ENDC)
        workflow.tree.newFields(workflow.Flow['ReferenceState'], Container='FlowSolution#Init')

    elif workflow.Initialization['method'] == 'interpolate':
        if isinstance(workflow.Initialization['source'], str):
            print(misc.CYAN + f"Initialize FlowSolution by interpolation from {workflow.Initialization['source']}" + misc.ENDC)
        else:
            print(misc.CYAN + f"Initialize FlowSolution by interpolation from the given tree" + misc.ENDC)
        initialize_flow_from_file_by_interpolation(workflow)
        
    elif workflow.Initialization['method'] == 'copy':
        if isinstance(workflow.Initialization['source'], str):
            print(misc.CYAN + f"Initialize FlowSolution by copy of {workflow.Initialization['source']}" + misc.ENDC)
        else:
            print(misc.CYAN + f"Initialize FlowSolution by copy of the given tree" + misc.ENDC)
        initialize_flow_from_file_by_copy(workflow)
    else:
        raise Exception(misc.RED+'The key "method" of the dictionary workflow.Initialization is mandatory'+misc.ENDC)

    for zone in workflow.tree.zones():
        if not zone.get(Name='FlowSolution#Init', Type='FlowSolution', Depth=1):
            MSG = 'FlowSolution#Init is missing in zone {}'.format(zone.name)
            raise ValueError(misc.RED + MSG + misc.ENDC)
    
    misc.apply_to_solver(workflow)



def initialize_flow_from_file_by_interpolation(workflow):
    '''
    Initialize the flow solution of **t** from the flow solution in the file
    **sourceFilename**.
    Modify the tree **t** in-place.

    Parameters
    ----------

        workflow : :py:obj:`mola.workflow.worflow.Workflow`
    '''
    raise Exception('Not yet implemented')

def initialize_flow_from_file_by_copy(workflow):
    '''
    Initialize the flow solution of **workflow.tree** by copying the flow solution in the file or tree
    **workflow.Initialization['source']**.
    Modify the tree in-place.

    Parameters
    ----------

        workflow : :py:obj:`mola.workflow.worflow.Workflow`
    '''
    keepTurbulentDistance = workflow.Initialization.get('keepTurbulentDistance', False)

    sourceTree = cgns.load(workflow.Initialization['source'])

    varNames = list(workflow.Flow['ReferenceState'])
    if keepTurbulentDistance:
        varNames += ['TurbulentDistance', 'TurbulentDistanceIndex']

    for zone in workflow.tree.zones():
        FSpath = zone.path() + '/FlowSolution#Init'
        try:
            FlowSolutionInSourceTree = sourceTree.getAtPath(FSpath)
            zone.addChild(FlowSolutionInSourceTree, override_brother_by_name=True)
        except AttributeError:
            ERROR_MSG = f"The node {FSpath} is not found in {workflow.Initialization['source']}"
            raise Exception(misc.RED+ERROR_MSG+misc.ENDC)

def compute_turbulent_distance_with_maia(workflow):
    '''
    The input tree has to be distributed, read by maia.
    '''
    import maia
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # TODO Add test to check that the tree was read with maia
    # This function needs to be after the definition of boundary conditions

    part_tree = maia.factory.partition_dist_tree(workflow.tree, comm)
    maia.algo.part.compute_wall_distance(part_tree, comm, out_fs_name='FlowSolution#Init')  # create a FlowSolution container named WallDistance
    maia.transfer.part_tree_to_dist_tree_all(workflow.tree, part_tree, comm)
    cgns.castNode(workflow.tree)

    # FIXME there is a issue with the PointRange: data are stored in 1D either if the mesh is structured

    ## If out_fs_name='FlowSolution#Init' is not used, we need to move the TurbulentDistance node
    # for zone in workflow.tree.zones():
    #     FlowSolution = zone.get(Name='FlowSolution#Init', Depth=1)
    #     WallDistance = zone.get(Name='WallDistance', Depth=1)
    #     TurbulentDistance = WallDistance.get(Name='TurbulentDistance')
    #     TurbulentDistance.moveTo(FlowSolution)
    #     WallDistance.remove()

