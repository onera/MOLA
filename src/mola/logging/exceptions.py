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

''' 
This module defines internal Exceptions. 
It should only contain class named <something>Exception or <something>Error.
Each class should inherit from the parent class MolaException.
'''

from typing import Union
import inspect
from .formatters import format_message_according_level, CYAN, BOLD, PINK, ENDC
from .excepthook import enable_mpi_excepthook, disable_mpi_excepthook
enable_mpi_excepthook()

class MolaException(Exception):

    def __init__(self, message=''):
        message = format_message_according_level(message, level='ERROR')
        super().__init__(message)

class MolaAssertionError(MolaException):
    pass

class MolaUserError(MolaException):
    pass

class MolaNotImplementedError(MolaException):
    pass

class MolaUserAttributeError(MolaException):

    def __init__(self, fun, error):
        error_str = str(error)
        signature = get_signature(fun)
        msg = f'{error_str}.\nThe allowed arguments are:{ENDC}\n{signature}'

        super().__init__(msg)


class MolaMissingFieldsError(MolaException):
    pass

def get_signature(fun):
    signature = inspect.signature(fun)

    args = ''
    optional_args = ''
    kw_only_args = ''
    
    for param in signature.parameters.values():
        if param.name == 'self': continue
        if param.annotation == inspect.Parameter.empty:
            annotation_name = ''
        elif hasattr(param.annotation, "__origin__") and param.annotation.__origin__ == Union:
            annotation_name = '(' + ', '.join([CYAN+arg.__name__+ENDC for arg in param.annotation.__args__]) + ')'
        else:
            annotation_name = '(' + CYAN + param.annotation.__name__ + ENDC + ')'
        line = f'  {BOLD}{param.name}{ENDC} {annotation_name}'
        if param.default != param.empty and param.kind != param.KEYWORD_ONLY:
            line += f' : {PINK}{param.default}{ENDC}'
        line += '\n'
        
        if param.default == param.empty and param.kind == param.POSITIONAL_OR_KEYWORD:
            args += line
        elif param.default != param.empty and param.kind == param.POSITIONAL_OR_KEYWORD:
            optional_args += line
        elif param.kind == param.KEYWORD_ONLY:
            kw_only_args += line


    # msg = f'{BOLD}name{ENDC} ({CYAN}allowed types{ENDC}) : {PINK}default value{ENDC}\n'
    msg = ''
    if args:
        msg += "Mandatory:\n"+str(args)
    if kw_only_args:
        msg += "Mandatory keyword-only:\n"+str(kw_only_args)
    if optional_args:
        msg += f"Optional:\n"+str(optional_args)

    return msg
