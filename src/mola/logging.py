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

import sys
import logging

LOG_LEVEL = 'INFO'  # TODO This value should be modificable with python parser for example
log_level = getattr(logging, LOG_LEVEL)

class MolaLogger(logging.Logger):
    
    def __init__(self, name='mola_logger', level=LOG_LEVEL, stream=True, filename=None):
        super().__init__(name, level)
        formatter = CustomFormatter()
        if stream:
            self.add_stream_handler(formatter)
        if filename:
            self.add_file_handler(formatter, filename)

    def set_level(self, level):
        self.setLevel(level)

    def set_format(self, format):
        for handler in self.handlers:
            handler.setFormatter(logging.Formatter(format))
    
    def add_stream_handler(self, formatter):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)
    
    def add_file_handler(self, formatter, filename):
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)
    
    def error(self, msg, exit=True, *args, **kwargs):
        super().error(msg, *args, **kwargs)
        if exit:
            sys.exit(1)
    
    def critical(self, msg, *args, **kwargs):
        super().critical(msg, *args, **kwargs)
        sys.exit(1)
    
    fatal = critical

class CustomFormatter(logging.Formatter):
    '''
    This class defines the format for all loggers in MOLA.
    '''

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format='%(levelname)s: %(message)s'
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    # format = "%(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + "%(message)s" + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + 'MOLA %(levelname)s: %(message)s' + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    

class ParallelLogger(MolaLogger):
    '''
    Replace function Coprocess.printCo() in MOLA v1.

    Example
    -------

        >>> logger = ParallelLogger()
        >>> logger.info('info', rank=0)

    '''

    def __init__(self, name='mola_logger.parallel', level=LOG_LEVEL, stream=False, filename='coprocess.log'):
        super().__init__(name, level=LOG_LEVEL, stream=stream, filename=filename)
        try:
            import numpy as np
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            self.rank = comm.Get_rank()
            NumberOfProcessors = comm.Get_size()
            nbOfDigitsOfNProcs = int(np.ceil(np.log10(NumberOfProcessors+1)))
            self.preffix = ('[{:0%d}]: '%nbOfDigitsOfNProcs).format(self.rank)
        except:
            raise Exception(f'Cannot initialize ParallelLogger {name}')
    
    def has_something_to_write(self, rank):
        return rank is None or rank == self.rank
        
    def debug(self, msg, rank=None, *args, **kwargs):
        if self.has_something_to_write(rank): 
            super().debug(self.preffix+msg, *args, **kwargs)
    
    def info(self, msg, rank=None, *args, **kwargs):
        if self.has_something_to_write(rank): 
            super().info(self.preffix+msg, *args, **kwargs)
    
    def warning(self, msg, rank=None, *args, **kwargs):
        if self.has_something_to_write(rank): 
            super().warning(self.preffix+msg, *args, **kwargs)
    
    def error(self, msg, rank=None, exit=False, *args, **kwargs):
        if self.has_something_to_write(rank): 
            super().error(self.preffix+msg, exit=exit, *args, **kwargs)
    
    def exception(self, msg, rank=None, *args, **kwargs):
        if self.has_something_to_write(rank): 
            super().exception(self.preffix+msg, *args, **kwargs)
    
    def critical(self, msg, rank=None, *args, **kwargs):
        if self.has_something_to_write(rank): 
            super().critical(self.preffix+msg, *args, **kwargs)
    
    fatal = critical
    
mola_logger = MolaLogger(filename='mola.log')
