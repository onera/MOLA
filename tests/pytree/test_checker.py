import pytest

from mola.pytree.user import checker
from .tree_generator import dummy_tree

from mola.logging.exceptions import MolaException

backends = ["maia","cassiopee","treelab"]

@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("backend", backends)
def test_full_tree_is_not_partitioned_for_use_in_maia(backend, dummy_tree):

    full_tree = dummy_tree
    assert not checker.is_partitioned_for_use_in_maia(full_tree, backend)  


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("backend", backends)
def test_full_tree_is_not_distributed_for_use_in_maia(backend, dummy_tree):

    full_tree = dummy_tree
    assert not checker.is_distributed_for_use_in_maia(full_tree, backend)  


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("backend", backends)
def test_dist_tree_is_distributed_and_not_partitioned_for_use_in_maia(backend, dummy_tree):
    from mpi4py import MPI
    import maia
    full_tree = dummy_tree
    dist_tree = maia.factory.full_to_dist_tree(full_tree, MPI.COMM_WORLD)
    assert checker.is_distributed_for_use_in_maia(dist_tree, backend)
    assert not checker.is_partitioned_for_use_in_maia(dist_tree, backend)


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("backend", backends)
def test_part_tree_is_partitioned_and_not_distributed_for_use_in_maia(backend, dummy_tree):
    from mpi4py import MPI
    import maia
    full_tree = dummy_tree
    dist_tree = maia.factory.full_to_dist_tree(full_tree, MPI.COMM_WORLD)
    part_tree = maia.factory.partition_dist_tree(dist_tree, MPI.COMM_WORLD, data_transfer='ALL')
    assert checker.is_partitioned_for_use_in_maia(part_tree, backend)
    assert not checker.is_distributed_for_use_in_maia(part_tree, backend)


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("backend", backends)
def test_assert_zones_have_zone_type_node(backend, dummy_tree):
    t = dummy_tree
    checker.assert_zones_have_zone_type_node(t, backend)

    from treelab import cgns
    t_no_zone_type = cgns.castNode(t)
    t_no_zone_type.findAndRemoveNode(Name='ZoneType')

    with pytest.raises(MolaException) as e_info:
        checker.assert_zones_have_zone_type_node(t_no_zone_type, backend)



if __name__ == '__main__':
    test_full_tree_is_not_partitioned_for_use_in_maia("treelab")