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

import numpy as np
import os
import pprint
import textwrap
import tempfile
import subprocess
import inspect

AutoGridLocation = {'FlowSolution':'Vertex',
                    'FlowSolution#Centers':'CellCenter',
                    'FlowSolution#Height':'Vertex',
                    'FlowSolution#EndOfRun':'CellCenter',
                    'FlowSolution#Init':'CellCenter',
                    'FlowSolution#SourceTerm':'CellCenter',
                    'FlowSolution#EndOfRun#Coords':'Vertex'}

CoordinatesShortcuts = dict(CoordinateX='CoordinateX',
                            CoordinateY='CoordinateY',
                            CoordinateZ='CoordinateZ',
                            x='CoordinateX',
                            y='CoordinateY',
                            z='CoordinateZ',
                            X='CoordinateX',
                            Y='CoordinateY',
                            Z='CoordinateZ')


def sortListsUsingSortOrderOfFirstList(*arraysOrLists):
    '''
    This function accepts an arbitrary number of lists (or arrays) as input.
    It sorts all input lists (or arrays) following the ordering of the first
    list after sorting.

    Returns all lists with new ordering.

    Parameters
    ----------

        arraysOrLists : comma-separated arrays or lists
            Arbitrary number of arrays or lists

    Returns
    -------

        NewArrays : list
            list containing the new sorted arrays or lists following the order
            of first the list or array (after sorting).

    Examples
    --------

    ::

        import numpy as np
        import MOLA.Data.Core as C

        First = [5,1,6,4]
        Second = ['a','c','f','h']
        Third = np.array([10,20,30,40])

        NewFirst, NewSecond, NewThird = C.sortListsUsingSortOrderOfFirstList(First,Second,Third)
        print(NewFirst)
        print(NewSecond)
        print(NewThird)

    will produce

    ::

        [1, 4, 5, 6]
        ['c', 'h', 'a', 'f']
        [20, 40, 10, 30]

    '''
    SortInd = np.argsort(arraysOrLists[0])
    NewArrays = []
    for a in arraysOrLists:
        if type(a) == 'ndarray':
            NewArray = np.copy(a,order='K')
            for i in SortInd:
                NewArray[i] = a[i]

        else:
            NewArray = [a[i] for i in SortInd]

        NewArrays.append( NewArray )

    return NewArrays


def writeFileFromModuleObject(settings, filename='.MOLA.py'):
    Lines = '#!/usr/bin/python\n'

    for Item in dir(settings):
        if not Item.startswith('_'):
            Lines+=Item+"="+pprint.pformat(getattr(settings, Item))+"\n\n"

    with open(filename,'w') as f: f.write(Lines)

    try: os.remove(filename+'c')
    except: pass

def allclose_dict(d1, d2, tol_abs=None, tol_rel=1e-6, empty_eq_None=True):
    '''
    taken from https://gist.github.com/durden/4236551

    Compare two dicts recursively (just as standard '==' except floating point
    values are compared within given precision.
    A kind of `numpy.allclose()` function applied to dictionaries.

    Parameters
    ----------
    d1 : dict
        first dictionary
    d2 : dict
        second dictionary to compre to **d1**
    tol_abs : float or None, optional
        If not None, the absolute tolerance to use. By default None
    tol_rel : _type_, optional
        If **tol_abs** if None, the relative tolerance to use for the comparison. 
        Thus applicated absolute tolerance will be :py:math:`tol_{abs} = e \times tol_{rel}`, 
        where :py:math:`e` is the compared element in **d1**.
        By default 1e-6

    Returns
    -------
    bool
        result of the comparison
    '''
    from .logging import mola_logger # here in order to avoid circular import in exceptions.py
    if len(d1) != len(d2):
        mola_logger.debug(f'Both dictionary have not the same length ({len(d1)} and {len(d2)} respectively)')
        return False

    for k, v in d1.items():
        # Make sure all the keys are equal
        if k not in d2:
            mola_logger.debug(f'{k}: {k} not in {d2}')
            return False

        # Fuzzy float comparison
        if isinstance(v, float) and isinstance(d2[k], float):
            if tol_abs is not None:
                precision = tol_abs
            elif abs(v) < tol_rel:
                precision = tol_rel
            else:
                precision = abs(v) * tol_rel
            if not abs(v - d2[k]) < precision:
                mola_logger.debug(f'{k}: {v} != {d2[k]}')
                return False
        # Recursive compare if there are nested dicts
        elif isinstance(v, dict):
            if not allclose_dict(v, d2[k], tol_abs, tol_rel, empty_eq_None=empty_eq_None):
                mola_logger.debug(f'{k}: {v} != {d2[k]}')
                return False
        elif isinstance(v, list):
            if not allclose_lists(v, d2[k], tol_abs, tol_rel, empty_eq_None=empty_eq_None):
                mola_logger.debug(f'{k}: {v} != {d2[k]}')
                return False
        elif isinstance(v, np.ndarray):
            if np.all(v != d2[k]):
                mola_logger.debug(f'{k}: {v} != {d2[k]}')
                return False
        elif empty_eq_None and v is None:
            if d2[k] not in [None, [], dict()]:
                return False
        # Fall back to default
        elif v != d2[k]:
            mola_logger.debug(f'{k}: {v} != {d2[k]}')
            return False

    return True

def allclose_lists(l1, l2, tol_abs=None, tol_rel=1e-6, empty_eq_None=True):
    from .logging import mola_logger # here in order to avoid circular import in exceptions.py

    if not isinstance(l1, list):
        raise TypeError(f'The first argument is not a list: {l1}')
    
    if empty_eq_None and l1 == [] and l2 is None:
        return True
    elif not isinstance(l2, (list, np.ndarray)):
        return False  #raise TypeError(f'The second argument is not a list or a ndarray: {l2}')
    
    if len(l1) != len(l2):
        return False
    
    for item1, item2 in zip(l1, l2):
        if isinstance(item1, dict) and isinstance(item2, dict):
            if not allclose_dict(item1, item2, tol_abs=tol_abs, tol_rel=tol_rel, empty_eq_None=empty_eq_None):
                return False
        elif isinstance(item1, list) and isinstance(item2, list):
            if not allclose_lists(item1, item2, tol_abs=tol_abs, tol_rel=tol_rel, empty_eq_None=empty_eq_None):
                return False
        elif isinstance(item1, np.ndarray) and isinstance(item2, np.ndarray):
            if np.all(item1 != item2):
                mola_logger.debug(f'{item1} != {item2}')
                return False
        elif item1 != item2:
            return False
    return True


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

    try:
        run_env = os.environ.copy()
        if extra_env:
            run_env.update(extra_env)

        if "SLURM_JOB_ID" in run_env:
            launcher = ["srun"]
            if size is not None:
                launcher.extend(["-n", str(size)])
        else:
            launcher = ["mpirun"]
            if size is not None:
                launcher.extend(["-np", str(size)])

        cmd = launcher + ["python3", tmp_file_path]
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=run_env
        )

        print(result.stdout)
        print(result.stderr)

        assert result.returncode == 0, f"MPI test failed with code {result.returncode}"

    finally:
        os.remove(tmp_file_path)
