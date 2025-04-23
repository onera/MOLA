import pytest

from mola.pytree.user import checker
from .tree_generator import build_dummy_tree

backends = ["maia","cassiopee","treelab"]

@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("backend", backends)
def test_full_tree_is_not_partitioned_for_use_in_maia(backend):

    full_tree = build_dummy_tree()
    assert not checker.is_partitioned_for_use_in_maia(full_tree, backend)  


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("backend", backends)
def test_full_tree_is_not_distributed_for_use_in_maia(backend):

    full_tree = build_dummy_tree()
    assert not checker.is_distributed_for_use_in_maia(full_tree, backend)  


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("backend", backends)
def test_dist_tree_is_distributed_and_not_partitioned_for_use_in_maia(backend):
    from mpi4py import MPI
    import maia
    full_tree = build_dummy_tree()
    dist_tree = maia.factory.full_to_dist_tree(full_tree, MPI.COMM_WORLD)
    assert checker.is_distributed_for_use_in_maia(dist_tree)
    assert not checker.is_partitioned_for_use_in_maia(dist_tree)


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("backend", backends)
def test_part_tree_is_partitioned_and_not_distributed_for_use_in_maia(backend):
    from mpi4py import MPI
    import maia
    full_tree = build_dummy_tree()
    dist_tree = maia.factory.full_to_dist_tree(full_tree, MPI.COMM_WORLD)
    part_tree = maia.factory.partition_dist_tree(dist_tree, MPI.COMM_WORLD, data_transfer='ALL')
    assert checker.is_partitioned_for_use_in_maia(part_tree)
    assert not checker.is_distributed_for_use_in_maia(part_tree)


if __name__ == '__main__':
    test_full_tree_is_not_partitioned_for_use_in_maia("treelab")