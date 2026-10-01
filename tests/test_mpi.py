import pytest
from mola.mpi import run_as_mpi_subprocess, get_mpi_elements

comm, rank, size, _ = get_mpi_elements()

_extra_env = {"I_MPI_DEBUG": "5"} # for better MPI debugging diagnostics

@pytest.mark.unit
@pytest.mark.mpi
@pytest.mark.parametrize("size", [1, 2])
def test_mpi(size):

    def actual_test():
        from mola import solver
        from mola.mpi import get_mpi_elements
        _, rank, size, _ = get_mpi_elements()

        print(f"Rank={rank} Size={size} solver={solver}")

    run_as_mpi_subprocess(actual_test, size, extra_env=_extra_env)
                                                        

@pytest.mark.unit
@pytest.mark.mpi
@pytest.mark.parametrize("size", [1, 2])
def test_mpi_with_error(size):

    def actual_test():
        raise ValueError("this is designed to raise an error")

    try:
        run_as_mpi_subprocess(actual_test, size, extra_env=_extra_env)
    except AssertionError:
        pass
