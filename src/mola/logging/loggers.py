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
import logging
import io
import contextlib
from .formatters import CustomFormatter

class MolaLogger(logging.Logger):
    
    def __init__(self, name='mola_logger', level='INFO', stream=True, filename=None):
        super().__init__(name, level)
        formatter = CustomFormatter()
        if stream:
            self.add_stream_handler(formatter)
        if filename:
            if os.path.exists(filename):
                os.remove(filename)
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
    
    # def error(self, msg, exit=True, *args, **kwargs):
    #     super().error(msg, *args, **kwargs)
    #     if exit:
    #         raise MolaException
    
    # def critical(self, msg, *args, **kwargs):
    #     super().critical(msg, *args, **kwargs)
    #     raise MolaException
    
    # fatal = critical


class ParallelLogger(MolaLogger):
    '''
    Replace function Coprocess.printCo() in MOLA v1.

    Example
    -------

        >>> logger = ParallelLogger()
        >>> logger.info('info', rank=0)

    '''

    def __init__(self, name='mola_logger.parallel', level='INFO', stream=False, filename='coprocess.log'):
        super().__init__(name, level=level, stream=stream, filename=filename)
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


@contextlib.contextmanager
def redirect_streams_to_logger(logger, stdout_level=logging.INFO, stderr_level=logging.ERROR):
    tmp_stdout = io.StringIO()
    tmp_stderr = io.StringIO()
    
    # Redirect stdout to the temporary temporary object
    with contextlib.redirect_stdout(tmp_stdout), contextlib.redirect_stderr(tmp_stderr):
        yield tmp_stdout, tmp_stderr

    # Next lines are executed when leaving context ('with')
    
    def write_with_logger_if_needed(tmp_StringIO, level):
        std_str = tmp_StringIO.getvalue().rstrip('\n')
        if len(std_str) > 0 and not std_str.isspace(): 
            logger.log(level, std_str)

    write_with_logger_if_needed(tmp_stdout, stdout_level)
    write_with_logger_if_needed(tmp_stderr, stderr_level)

