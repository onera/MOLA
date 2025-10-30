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
from mola.logging import MolaException
from mola import __MOLA_PATH__

def run_script(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            contenu = f.read()
        exec(contenu, globals())

    except FileNotFoundError:
        raise MolaException(f"The file {filename} does not exist.")



@pytest.mark.user_case
@pytest.mark.cost_level_4
def test_open_workflow_rotating_component_turbomachinery_rotor37():
    filename = f'{__MOLA_PATH__}/../examples/open/workflow/rotating_component/turbomachinery/rotor37/run_sator.py'
    run_script(filename)

@pytest.mark.user_case
@pytest.mark.cost_level_4
def test_open_workflow_rotating_component_turbomachinery_SRV2():
    filename = f'{__MOLA_PATH__}/../examples/open/workflow/rotating_component/turbomachinery/SRV2/run_sator.py'
    run_script(filename)

# To be completed
