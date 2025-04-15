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
This module defines Formatters (used by Loggers) and useful related functions.
'''
import logging

BOLD = '\033[1m'
UNDERLINE = '\033[4m'
GREY = "\x1b[38;20m"
RED  = '\033[91m'
BOLD_RED = "\x1b[31;1m"
GREEN = '\033[92m'
BOLD_GREEN = '\033[92m;1m'
YELLOW = '\033[93m'
PINK  = '\033[95m'
CYAN  = '\033[96m'
BOLD_CYAN = '\033[96m;1m'
ENDC  = '\033[0m'

class CustomFormatter(logging.Formatter):
    '''
    This class defines the format for all loggers in MOLA.
    '''
    format='%(levelname)s: %(message)s'
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: GREY + format + ENDC,
        logging.INFO: GREY + "%(message)s" + ENDC,
        logging.WARNING: YELLOW + format + ENDC,
        logging.ERROR: RED + format + ENDC,
        logging.CRITICAL: BOLD_RED + format + ENDC
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    

def format_message_according_level(msg, level):
    log_level = logging._checkLevel(level)
    log_format = CustomFormatter.FORMATS[log_level]
    msg_with_format = log_format.replace('%(message)s', msg).replace('%(levelname)s', level)
    return "\n"+msg_with_format

def compare_with_expected_message_at_level(msg, expected_msg, level):
    return msg == format_message_according_level(expected_msg, level)
