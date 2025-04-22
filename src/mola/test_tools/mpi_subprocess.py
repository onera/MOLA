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

import os
import textwrap
import tempfile
import subprocess
import inspect

def run_as_mpi_subprocess(func, size, extra_env=None, *func_args, **func_kwargs):
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
        sys.exit(1)
"""

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(script)
        tmp_file_path = tmp_file.name

    try:
        run_env = os.environ.copy()
        if extra_env:
            run_env.update(extra_env)

        cmd = ["mpirun", "-np", str(size), "python3", tmp_file_path]
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=run_env
        )

        print(result.stdout)
        print(result.stderr)

        assert result.returncode == 0, f"MPI test failed with code {result.returncode}"

    finally:
        os.remove(tmp_file_path)
