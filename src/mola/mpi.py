import os
import sys
import multiprocessing
import textwrap
import tempfile
import subprocess
import inspect
import numpy as np
from . import __MOLA_PATH__
from .dependency_injector.retriever import load_source

def run_as_mpi_subprocess(func, size=None, extra_env=None, *func_args, **func_kwargs):
    
    # Extract and clean function source
    src_lines = inspect.getsourcelines(func)[0]
    src_lines = [line for line in src_lines if not line.strip().startswith("@")]
    src = textwrap.dedent("".join(src_lines))
    func_name = func.__name__

    # Build argument string
    args_repr = ", ".join([
        *(repr(arg) for arg in func_args),
        *(f"{k}={repr(v)}" for k, v in func_kwargs.items())
    ])

    # Script to be run
    script = f"""
import sys
from mpi4py import MPI

{src}

if __name__ == "__main__":
    try:
        {func_name}({args_repr})
    except Exception as e:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
"""

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(script)
        tmp_file_path = tmp_file.name

    # # for debug:
    # tmp_file_path = os.path.join(os.getcwd(),"mola_subprocess.py")
    # with open(tmp_file_path,'w') as tmp_file:
    #     tmp_file.write(script)


    try:
        run_env = os.environ.copy()
        if extra_env:
            run_env.update(extra_env)


        launcher = get_mpi_launcher(imposed_size=size)

        cmd = launcher.split() + [sys.executable, tmp_file_path]
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=run_env
        )

        print(result.stdout)
        print(result.stderr)

        assert result.returncode == 0, f"MPI test failed with code {result.returncode} and error msg:\n{result.stderr}\ncmd={' '.join(cmd)}"

    finally:
        os.remove(tmp_file_path)

def get_mpi_launcher(run_management : dict = {}, imposed_size : int = None) -> str:
    from .server import guess_machine

    if imposed_size:
        size = imposed_size
    else:
        if 'NumberOfProcessors' in  run_management:
            size = run_management['NumberOfProcessors']
        else:
            size = _get_mpi_task_count()

    scheduler = run_management.get('Scheduler','local')
    machine = run_management.get('Machine', 'localhost')
    if machine == 'localhost':
        machine = guess_machine(machine)
    mola_target_path = run_management.get('mola_target_path', __MOLA_PATH__)
    scheduler_defaults = _get_scheduler_defaults(machine, mola_target_path)

    try:    pmi2_lib = scheduler_defaults.MOLA_PMI2_LIB
    except: pmi2_lib = None

    if pmi2_lib:
        launcher = (
            "srun --mpi=pmi2 "
            "--overlap "
            "--exact "
            "--cpu-bind=cores "
           f"--export=ALL,I_MPI_PMI_LIBRARY={pmi2_lib}")
        if size is not None:
            launcher += f' -n {size}'

    else:

        launcher = 'mpirun'
        if size is not None:
            if _is_openmpi() and scheduler == 'local' and not _is_slurm_job_running():
                launcher += " --use-hwthread-cpus"
            launcher += f" -np {size}"


    return launcher

def _is_slurm_job_running() -> bool:
    
    # Fast path: check common SLURM environment variables
    if any(key in os.environ for key in ["SLURM_NTASKS", "SLURM_CPUS_ON_NODE", "SLURM_JOB_ID"]):
        return True
    
    # Fallback: query SLURM scheduler directly (slower)
    # try:
    #     result = subprocess.run(
    #         ["squeue", "--me"],
    #         capture_output=True,
    #         text=True,
    #         timeout=5
    #     )
    #     return result.returncode == 0 and bool(result.stdout.strip())
    # except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
    #     return False

    return False

def _is_openmpi() -> bool:
    try:
        result = subprocess.run(
            ["mpirun", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0 and "Open MPI" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False

def _get_mpi_task_count(size=None) -> int:
    
    # Priority 1: Explicit size parameter
    if size is not None:
        return size
    
    # Priority 2: SLURM environment
    if "SLURM_JOB_ID" in os.environ:
        ntasks = os.environ.get("SLURM_NTASKS")
        if ntasks:
            return int(ntasks)
        # Fallback: SLURM_CPUS_ON_NODE if SLURM_NTASKS not set
        cpus = os.environ.get("SLURM_CPUS_ON_NODE")
        if cpus:
            return int(cpus)
    
    # Priority 3: Local execution - detect available CPUs
    # Use hardware threads (hyperthreading) for better utilization
    try:
        # Try to get physical CPU count first
        import psutil
        cpu_count = psutil.cpu_count(logical=True)  # Includes hyperthreads
        if cpu_count:
            return cpu_count
    except ImportError:
        pass
    
    # Fallback: Python's multiprocessing
    return multiprocessing.cpu_count()

def _int_env(name):
    try:
        return int(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return None


def _is_current_slurm_task():
    task_pid = _int_env("SLURM_TASK_PID")
    return task_pid is not None and task_pid == os.getpid()


def get_mpi_identity_from_environment():
    env = os.environ

    # Variables inherited from an interactive shell must not be trusted.
    # Trust Slurm/PMI metadata only when this process is the task launched
    # by Slurm itself.
    if _is_current_slurm_task():
        rank = _int_env("SLURM_PROCID")
        if rank is None:
            rank = _int_env("PMI_RANK")

        size = _int_env("SLURM_STEP_NUM_TASKS")
        if size is None:
            size = _int_env("PMI_SIZE")

        if rank is not None and size is not None and size >= 1:
            return rank, size

    # Open MPI normally provides launcher-specific variables. This branch
    # is useful outside Slurm.
    if "SLURM_JOB_ID" not in env:
        rank = _int_env("OMPI_COMM_WORLD_RANK")
        size = _int_env("OMPI_COMM_WORLD_SIZE")
        if rank is not None and size is not None:
            return rank, size

    # Safe fallback: this process is treated as sequential.
    return 0, 1

def get_mpi_elements():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    nb_digs = int(np.ceil(np.log10(size+1)))

    return comm, rank, size, nb_digs

# TODO this function is duplicated with the one in remote.py, but since we
# cannot import it due to circular dependencies, we shall redefine it here.
# This demonstrates that intricated dependencies between logging/server/mpi shall
# be redesigned
def _get_scheduler_defaults(machine, mola_target_path=__MOLA_PATH__):
    network = os.getenv('MOLA_NETWORK')

    for path in [
        os.path.join(mola_target_path, 'mola', 'env', network, machine, 'scheduler_defaults.py'),
        os.path.join(__MOLA_PATH__, 'mola', 'env', network, machine, 'scheduler_defaults.py')
        ]:
        try:
            return load_source('scheduler_defaults', path)
        except FileNotFoundError:
            pass

    return None


if __name__ == "__main__":
    print(get_mpi_identity_from_environment())
