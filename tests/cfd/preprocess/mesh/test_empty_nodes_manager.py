import pytest
import numpy as np
from treelab import cgns
from mola.cfd.preprocess.mesh.tools import EmptyNodesManager


@pytest.mark.unit
@pytest.mark.cost_level_0
class TestEmptyNodesManager:
    """Test suite for EmptyNodesManager class."""

    def test_init(self):
        """Test that EmptyNodesManager initializes with empty dictionary."""
        manager = EmptyNodesManager()
        assert manager.empty_nodes_by_path == {}

    def test_is_empty_flow_solution_no_dataarray(self):
        """Test detection of FlowSolution with no DataArray children."""
        manager = EmptyNodesManager()
        fs_node = cgns.Node(Name='FlowSolution', Type='FlowSolution_t', Parent=None)
        assert manager._is_empty_flow_solution(fs_node) is True

    def test_is_empty_flow_solution_with_none_dataarray(self):
        """Test detection of FlowSolution with DataArray storing None values."""
        manager = EmptyNodesManager()
        fs_node = cgns.Node(Name='FlowSolution', Type='FlowSolution_t', Parent=None)
        data_array = cgns.Node(Name='DataArray', Type='DataArray_t', Parent=fs_node)
        data_array.setValue(None)
        assert manager._is_empty_flow_solution(fs_node) is True

    def test_is_empty_flow_solution_with_valid_dataarray(self):
        """Test detection of FlowSolution with valid DataArray children."""
        manager = EmptyNodesManager()
        fs_node = cgns.Node(Name='FlowSolution', Type='FlowSolution_t', Parent=None)
        data_array = cgns.Node(Name='DataArray', Type='DataArray_t', Parent=fs_node)
        data_array.setValue(np.array([1.0, 2.0, 3.0]))
        assert manager._is_empty_flow_solution(fs_node) is False

    def test_is_empty_zone_with_empty_flow_solutions(self):
        """Test detection of zone with only empty FlowSolution nodes."""
        manager = EmptyNodesManager()
        zone = cgns.Zone(Name='Zone0')
        fs1 = cgns.Node(Name='FlowSolution1', Type='FlowSolution_t', Parent=zone)
        fs2 = cgns.Node(Name='FlowSolution2', Type='FlowSolution_t', Parent=zone)
        assert manager._is_empty_zone(zone) is True

    def test_is_empty_zone_with_non_empty_flow_solutions(self):
        """Test detection of zone with at least one non-empty FlowSolution node."""
        manager = EmptyNodesManager()
        zone = cgns.Zone(Name='Zone0')
        fs1 = cgns.Node(Name='FlowSolution1', Type='FlowSolution_t', Parent=zone)
        fs2 = cgns.Node(Name='FlowSolution2', Type='FlowSolution_t', Parent=zone)
        data_array = cgns.Node(Name='DataArray', Type='DataArray_t', Parent=fs2)
        data_array.setValue(np.array([1.0, 2.0, 3.0]))
        assert manager._is_empty_zone(zone) is False

    def test_add_grid_location_missing_vertex(self):
        """Test adding GridLocation when missing with loc='node'."""
        manager = EmptyNodesManager()
        fs_node = cgns.Node(Name='FlowSolution', Type='FlowSolution_t', Parent=None)
        loc_data = cgns.Node(Name='loc', Type='DataArray_t', Parent=fs_node)
        loc_data.setValue('node')
        manager._add_grid_location(fs_node)
        grid_location = fs_node.get(Type='GridLocation')
        assert grid_location is not None
        assert grid_location.value() == 'Vertex'

    def test_add_grid_location_missing_cell(self):
        """Test adding GridLocation when missing with loc='cell'."""
        manager = EmptyNodesManager()
        fs_node = cgns.Node(Name='FlowSolution', Type='FlowSolution_t', Parent=None)
        loc_data = cgns.Node(Name='loc', Type='DataArray_t', Parent=fs_node)
        loc_data.setValue('cell')
        manager._add_grid_location(fs_node)
        grid_location = fs_node.get(Type='GridLocation')
        assert grid_location is not None
        assert grid_location.value() == 'CellCenter'

    def test_add_grid_location_already_present(self):
        """Test that GridLocation is not duplicated when already present."""
        manager = EmptyNodesManager()
        fs_node = cgns.Node(Name='FlowSolution', Type='FlowSolution_t', Parent=None)
        existing_grid_location = cgns.Node(
            Name='GridLocation', Type='GridLocation_t', Value='Vertex', Parent=fs_node
        )
        manager._add_grid_location(fs_node)
        grid_locations = fs_node.group(Type='GridLocation', Depth=1)
        assert len(grid_locations) == 1
        assert grid_locations[0] is existing_grid_location

    def test_remove_pypart_suffix_with_suffix(self):
        """Test removing PyPart suffix from paths."""
        manager = EmptyNodesManager()
        path = '/Base/Zone0.P12.N34/FlowSolution'
        result = manager._remove_PyPart_suffix(path)
        assert result == '/Base/Zone0/FlowSolution'

    def test_remove_pypart_suffix_without_suffix(self):
        """Test that paths without PyPart suffix remain unchanged."""
        manager = EmptyNodesManager()
        path = '/Base/Zone0/FlowSolution'
        result = manager._remove_PyPart_suffix(path)
        assert result == '/Base/Zone0/FlowSolution'

    def test_remove_pypart_suffix_with_multiple_suffixes(self):
        """Test removing PyPart suffix with various number formats."""
        manager = EmptyNodesManager()
        test_cases = [
            ('/Base/Zone0.P1.N1/FlowSolution', '/Base/Zone0/FlowSolution'),
            ('/Base/Zone0.P123.N456/FlowSolution', '/Base/Zone0/FlowSolution'),
            ('/Base/Zone0.P99999.N99999/FlowSolution', '/Base/Zone0/FlowSolution'),
        ]
        for input_path, expected in test_cases:
            result = manager._remove_PyPart_suffix(input_path)
            assert result == expected

    def test_get_empty_flow_solution_nodes_detects_nodes(self):
        """Test detection of empty FlowSolution nodes in a tree."""
        manager = EmptyNodesManager()
        tree = cgns.Tree()
        base = cgns.Node(Name='Base', Type='CGNSBase_t', Parent=tree)
        zone = cgns.Zone(Name='Zone0', Parent=base)
        # Add empty FlowSolution
        fs_empty = cgns.Node(Name='EmptySolution', Type='FlowSolution_t', Parent=zone)
        data_array_empty = cgns.Node(Name='Density', Type='DataArray_t', Parent=fs_empty)
        data_array_empty.setValue(None)
        # Add non-empty FlowSolution
        fs_non_empty = cgns.Node(Name='NonEmptySolution', Type='FlowSolution_t', Parent=zone)
        data_array = cgns.Node(Name='Density', Type='DataArray_t', Parent=fs_non_empty)
        data_array.setValue(np.array([1.0, 2.0, 3.0]))

        manager.get_empty_FlowSolution_nodes(tree, remove=False)

        # Should detect only the empty node
        assert len(manager.empty_nodes_by_path) == 1
        path = list(manager.empty_nodes_by_path)[0]
        assert path == 'CGNSTree/Base/Zone0/EmptySolution'

    def test_get_empty_flow_solution_nodes_with_remove(self):
        """Test that remove=True removes empty nodes from tree."""
        manager = EmptyNodesManager()
        tree = cgns.Tree()
        base = cgns.Node(Name='Base', Type='CGNSBase_t', Parent=tree)
        zone = cgns.Zone(Name='Zone0', Parent=base)
        fs_empty = cgns.Node(Name='EmptySolution', Type='FlowSolution_t', Parent=zone)

        manager.get_empty_FlowSolution_nodes(tree, remove=True)

        # Node should be removed from tree
        remaining_fs = zone.group(Type='FlowSolution', Depth=1)
        assert len(remaining_fs) == 0
        # Node should be stored in manager
        assert len(manager.empty_nodes_by_path) == 1

    def test_get_empty_flow_solution_nodes_with_pypart_suffix_removal(self):
        """Test that remove_pypart_suffix=True strips PyPart suffixes from paths."""
        manager = EmptyNodesManager()
        tree = cgns.Tree()
        base = cgns.Node(Name='Base', Type='CGNSBase_t', Parent=tree)
        zone = cgns.Zone(Name='Zone0.P12.N34', Parent=base)
        fs_empty = cgns.Node(Name='EmptySolution', Type='FlowSolution_t', Parent=zone)

        manager.get_empty_FlowSolution_nodes(tree, remove=False, remove_pypart_suffix=True)

        # Path should not contain PyPart suffix
        assert len(manager.empty_nodes_by_path) == 1
        path = list(manager.empty_nodes_by_path)[0]
        assert path == 'CGNSTree/Base/Zone0/EmptySolution'

    def test_ensure_unique_rank_for_empty_nodes(self):
        """Test that lowest rank is selected as owner for duplicate paths."""
        manager = EmptyNodesManager()
        # Simulate paths found on multiple ranks
        manager.empty_nodes_by_path = {
            '/Base/Zone0/FS1': (2, None),  # Found on rank 2
            '/Base/Zone0/FS2': (0, None),  # Found on rank 0
            '/Base/Zone0/FS3': (1, None),  # Found on rank 1
        }
        # Add duplicate entries as if found on multiple ranks
        manager.empty_nodes_by_path['/Base/Zone0/FS1'] = (0, None)  # Also on rank 0

        manager._ensure_unique_rank_for_empty_nodes()

        # Lowest rank should be owner for all paths
        assert manager.empty_nodes_by_path == {
            '/Base/Zone0/FS1': (0, None),  # Found on rank 2
            '/Base/Zone0/FS2': (0, None),  # Found on rank 0
            '/Base/Zone0/FS3': (1, None),  # Found on rank 1
        }

    def test_restore_empty_flow_solution_nodes(self):
        """Test restoration of empty FlowSolution nodes to tree."""
        manager = EmptyNodesManager()
        tree = cgns.Tree()
        base = cgns.Node(Name='Base', Type='CGNSBase_t', Parent=tree)
        zone = cgns.Zone(Name='Zone0', Parent=base)
        # Set zone processor for single-rank scenario
        solver_param = cgns.Node(Name='.Solver#Param', Type='FamilyName_t', Parent=zone)
        proc = cgns.Node(Name='proc', Type='DataArray_t', Parent=solver_param)
        proc.setValue(0)

        # Create empty FlowSolution and store it
        fs_empty = cgns.Node(Name='EmptySolution', Type='FlowSolution_t', Parent=zone)
        data_array = cgns.Node(Name='DataArray', Type='DataArray_t', Parent=fs_empty)
        data_array.setValue(None)

        manager.get_empty_FlowSolution_nodes(tree, remove=True)

        # Restore the empty node
        manager.restore_empty_FlowSolution_nodes(tree)

        # Node should be restored to tree
        remaining_fs = zone.group(Type='FlowSolution', Depth=1)
        assert len(remaining_fs) == 1

    def test_restore_empty_flow_solution_nodes_in_file(self, tmp_path):
        """Test restoration of empty FlowSolution nodes to file."""
        manager = EmptyNodesManager()
        tree = cgns.Tree()
        base = cgns.Node(Name='Base', Type='CGNSBase_t', Parent=tree)
        zone = cgns.Zone(Name='Zone0', Parent=base)
        solver_param = cgns.Node(Name='.Solver#Param', Type='FamilyName_t', Parent=zone)
        proc = cgns.Node(Name='proc', Type='DataArray_t',Value=0, Parent=solver_param)

        # Create empty FlowSolution and store it
        fs_empty = cgns.Node(Name='EmptySolution', Type='FlowSolution_t', Parent=zone)
        # data_array = cgns.Node(Name='DataArray', Type='DataArray_t', Parent=fs_empty)
        fs_empty.setParameters(ContainerName='.Solver#Output', var=['ro rou'])

        manager.get_empty_FlowSolution_nodes(tree)

        # Remove .Solver#Output before saving tree
        tree.findAndRemoveNode(Name='.Solver#Output')

        # Save tree to file
        test_file = tmp_path / 'test.cgns'
        tree.save(str(test_file))

        # Restore empty nodes to file
        manager.restore_empty_FlowSolution_nodes_in_file(str(test_file))

        # Verify file was modified (at least no exception raised)
        assert test_file.exists()
        tree = cgns.load(str(test_file))
        solver_output = tree.get(Name='.Solver#Output')
        assert solver_output
        var_node = solver_output.get(Name='var')
        assert var_node and var_node.value() == ['ro', 'rou']
