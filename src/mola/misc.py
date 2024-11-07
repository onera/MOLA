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
import inspect
import sys
import os
import pprint
import shutil
from typing import Union
from .logging.formatters import BOLD, CYAN, PINK, ENDC

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

def load_source(ModuleName, filename, safe=True):
    '''
    Load a python file as a module guaranteeing intercompatibility between
    different Python versions

    Parameters
    ----------

        ModuleName : str
            name to be provided to the new module

        filename : str
            full or relative path of the file containing the source (module)
            to be loaded

        safe : bool
            if :py:obj:`True`, then cached files of previously loaded versions
            are explicitely removed

    Returns
    -------

        module : module
            the loaded module
    '''
    if safe:
        current_path_file = filename.split(os.path.sep)[-1]
        for fn in [filename, current_path_file]:
            try: os.remove(fn+'c')
            except: pass
        try: shutil.rmtree('__pycache__')
        except: pass

    if sys.version_info[0] == 3 and sys.version_info[1] >= 5:
        import importlib.util
        spec = importlib.util.spec_from_file_location(ModuleName, filename)
        LoadedModule = importlib.util.module_from_spec(spec)
        sys.modules[ModuleName] = LoadedModule
        spec.loader.exec_module(LoadedModule)
    elif sys.version_info[0] == 3 and sys.version_info[1] < 5:
        from importlib.machinery import SourceFileLoader
        LoadedModule = SourceFileLoader(ModuleName, filename).load_module()
    else:
        raise ValueError("Not supporting Python version "+sys.version)
    return LoadedModule


def reload_source(module):
    '''
    Reload a python module guaranteeing intercompatibility between
    different Python versions

    Parameters
    ----------

        module : module
            pointer towards the previously loaded module
    '''

    import importlib
    importlib.reload(module)


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
