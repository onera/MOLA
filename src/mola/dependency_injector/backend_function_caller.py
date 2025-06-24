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
import inspect
import pathlib
from mola.dependency_injector.retriever import (get_wrap_module_path,
                                                get_backend_module_from)

def call_backend_function(function_name : str, backend : str, *args, **kwargs):
    '''
    This is a generic function that is used for calling a backend-specific 
    implementation of a function contained in a module named exactly like this one
    but located in a wrapper folder.
    '''

    module_path = get_wrap_module_path(backend)
    backend_module = get_backend_module_from(module_path)
    fun = get_function_at_backend_from(backend_module, function_name)

    return fun(*args, **kwargs)

def get_function_at_backend_from(backend_module, function_name : str):
    try:
        fun = getattr(backend_module, function_name)
    except AttributeError as e:
        msg = f'Function {function_name} not implemented'
        raise AttributeError(msg) from e

    return fun