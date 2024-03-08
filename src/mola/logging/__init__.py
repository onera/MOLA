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

from .loggers import *
from .exceptions import *
from .catchers import *
from .formatters import *
from .parsers import get_log_level_and_log_file

LOG_LEVEL, LOG_FILE = get_log_level_and_log_file()

mola_logger = MolaLogger(level=LOG_LEVEL, filename=LOG_FILE)

def print(msg, level='INFO', logger=mola_logger):
    logging_functions_by_level = dict(
        DEBUG = logger.debug,
        INFO = logger.info,
        WARNING = logger.warning,
        ERROR = logger.error,
        CRITICAL = logger.critical
    )
    logging_functions_by_level[level](msg)

def check_error_message(expected_msg, fun, *args, **kwargs):
    try:
        fun(*args, **kwargs)
    except MolaException as e:
        exception_name = e.__class__.__name__
        mola_logger.debug(f'check_error_message() catched a {exception_name} for {fun.__name__}')
        error_msg = e.args[0]
        assert compare_with_expected_message_at_level(error_msg, expected_msg, level='ERROR')
    else:
        raise AssertionError(f'{fun.__name__} should raise a MolaError and it does not!')
    
