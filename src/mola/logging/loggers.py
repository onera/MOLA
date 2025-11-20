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
This module defines Loggers only. 
It should only contains class named <something>Logger  
'''

import sys
import os
import logging
import warnings
import numpy as np

from .formatters import CustomFormatter, YELLOW, ENDC
import mola.naming_conventions as names
    
class MaxLevelFilter(logging.Filter):

    def __init__(self, max_level):
        self.max_level = max_level
        super().__init__()

    def filter(self, record):
        return record.levelno <= self.max_level

class MolaLogger(logging.Logger):
    
    def __init__(self, name='mola_logger', level='INFO', stream=True, filename=None):
        self._init_MPI()
        super().__init__(name, level)
        formatter = CustomFormatter()
        if stream:
            self.add_stream_handler(formatter)
        if filename:
            if self.rank == 0:
                if os.path.exists(filename):
                    os.remove(filename)
            if self.NumberOfProcessors > 1:
                self.comm.barrier()
            self.add_file_handler(formatter, filename)
        
    def _init_MPI(self):
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.NumberOfProcessors = self.comm.Get_size()
            self.comm.barrier()
        except:
            self.comm = None
            self.rank = 0
            self.NumberOfProcessors = 1
            
        if self.NumberOfProcessors > 1:
            nbOfDigitsOfNProcs = int(np.ceil(np.log10(self.NumberOfProcessors+1)))
            self.preffix = ('[{:0%d}]: '%nbOfDigitsOfNProcs).format(self.rank)
        else:
            self.preffix = ''

    def set_level(self, level):
        self.setLevel(level)

    def set_format(self, format):
        for handler in self.handlers:
            handler.setFormatter(logging.Formatter(format))
    
    def add_stream_handler(self, formatter):
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.addFilter(MaxLevelFilter(logging.WARNING))
        self.addHandler(stdout_handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(formatter)
        stderr_handler.setLevel(logging.ERROR) 
        self.addHandler(stderr_handler) 
    
    def add_file_handler(self, formatter, filename):
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)

    def _has_something_to_write(self, rank):
        return rank is None or rank == self.rank

    def debug(self, msg, rank=None, *args, **kwargs):
        if self._has_something_to_write(rank): 
            super().debug(self.preffix+msg, *args, **kwargs)
    
    def info(self, msg, rank=None, *args, **kwargs):
        if self._has_something_to_write(rank): 
            super().info(self.preffix+msg, *args, **kwargs)

    def user_warning(self, msg, rank=None, *args, **kwargs):
        # Not a real Python warning, just a message with INFO level, 
        # but tagged as a "User warning" and colored in yellow to catch user attention
        if self._has_something_to_write(rank): 
            msg = f'{YELLOW}User warning: {msg}{ENDC}'
            super().info(self.preffix+msg, *args, **kwargs)
    
    def warning(self, msg, rank=None, *args, **kwargs):
        if self._has_something_to_write(rank): 
            super().warning(self.preffix+msg, *args, **kwargs)
            warnings.warn(self.preffix+msg)

    
    def error(self, msg, rank=None, *args, **kwargs):
        if self._has_something_to_write(rank): 
            super().error(self.preffix+msg, *args, **kwargs)
    
    def critical(self, msg, rank=None, *args, **kwargs):
        if self._has_something_to_write(rank): 
            super().critical(self.preffix+msg, *args, **kwargs)
    
    fatal = critical
    
