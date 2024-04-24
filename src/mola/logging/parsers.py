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
import os
import argparse
from .catchers import mute_stderr

def parse_args(args):
    parser = argparse.ArgumentParser('Logger parser')
    parser.add_argument('-v', '--verbosity', help='Level of verbosity', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    parser.add_argument('-l', '--logfile', help='Name of log file', type=str)
    return parser.parse_args(args)

@mute_stderr
def get_log_level_and_log_file():
    try:
        # Need that try/except because pytest does not work with the following lines
        parser = parse_args(sys.argv[1:])
        log_level = parser.verbosity
        log_file  = parser.logfile
    except:
        script_name = os.path.basename(sys.argv[0])
        if script_name in ['pytest', 'py.test']:
            log_level = 'DEBUG'
            log_file  = None
        else:
            log_level = 'INFO'
            log_file  = None

    return log_level, log_file
