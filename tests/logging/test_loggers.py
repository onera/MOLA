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


import pytest

from mola.logging import mola_logger, MolaUserError

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_log_info():
    mola_logger.info('toto')

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_log_warning():
    mola_logger.warning('this is a warning test')

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_mola_user_error():
    try:
        raise MolaUserError('error')
    except MolaUserError:
        pass