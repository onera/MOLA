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
import sys
import shutil
import inspect
import pathlib
import importlib.util

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
        spec = importlib.util.spec_from_file_location(ModuleName, filename)
        LoadedModule = importlib.util.module_from_spec(spec)
        sys.modules[ModuleName] = LoadedModule
        try:
            spec.loader.exec_module(LoadedModule)
        except ImportError as e:
            raise ImportError(f"failed sourcing {filename}") from e
    else:
        raise ValueError("Not supporting Python version "+sys.version)
    return LoadedModule

def get_wrap_module_path(backend : str,
        interface_switch=["user", "wrap/<backend>"]):

    previous_module_path = get_caller_path()
    try:
        # previous_module_path is like '/<root_path>/src/mola/folder/user/module.py'
        # In the floowing lines, it is split to prevent unwanted modification of '/<root_path>/' if it contains 'user'
        root_path, previous_module_path = previous_module_path.split('src/mola/')
        wrap_path = previous_module_path.replace(interface_switch[0],
                               interface_switch[1].replace('<backend>',f'{backend}'))
        wrap_path = os.path.join(root_path,'src/mola', wrap_path)
    except ValueError:
        # the pattern "src/mola/" is not in previous_module_path
        # It appends to import modules from "test" for instance
        wrap_path = previous_module_path.replace(interface_switch[0],
                               interface_switch[1].replace('<backend>',f'{backend}'))

    return wrap_path

def get_caller_path(
    # CAUTION put in exclude_modules all the module directly importing retriever:
    exclude_modules={__name__,"backend_function_caller","retriever"}):
    
    stack = inspect.stack()
    for frame_info in stack[1:]:
        module_name = frame_info.frame.f_globals.get('__name__', '').split('.')[-1]
        if module_name not in exclude_modules and not module_name.startswith('_pytest'):
            caller_path = str(pathlib.Path(frame_info.filename).resolve())
            return caller_path

    raise RuntimeError("Caller path could not be determined.")

def get_backend_module_from(expected_module_path : str):
    try:
        backend_module = load_source('backend_module', expected_module_path)
    except FileNotFoundError as e:
        msg = (f'Missing backend module "{expected_module_path}"')
        raise FileNotFoundError(msg) from e

    return backend_module



