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

class CustomFormatter(logging.Formatter):
    '''
    This class defines the format for all loggers in MOLA.
    '''

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = '\033[92m'
    pink  = '\033[95m'
    cyan  = '\033[96m'
    underline = '\033[4m'
    reset = "\x1b[0m"
    format='%(levelname)s: %(message)s'
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    # format = "%(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + "%(message)s" + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    

def format_message_according_level(msg, level):
    log_level = logging._checkLevel(level)
    log_format = CustomFormatter.FORMATS[log_level]
    msg_with_format = log_format.replace('%(message)s', msg).replace('%(levelname)s', level)
    return msg_with_format

def compare_with_expected_message_at_level(msg, expected_msg, level):
    return msg == format_message_according_level(expected_msg, level)
