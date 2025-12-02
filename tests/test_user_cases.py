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
from pathlib import Path
from mola.logging import MolaException
from mola import __MOLA_PATH__, solver

EXAMPLE_CASES = [
    # Airplane
    'open/workflow/fixed_component/airplane/isolated_wing/run_elsa.py',
    'open/workflow/fixed_component/airplane/isolated_wing/run_sonics.py',
    'open/workflow/fixed_component/airplane/isolated_wing/run_fast.py',
    #
    'open/workflow/rotating_component/turbomachinery/rotor37/run_sator.py',
    # 'open/workflow/rotating_component/turbomachinery/SRV2/run_sator.py'
]

def run_script(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        exec(content, globals())

    except FileNotFoundError:
        raise MolaException(f"The file {filename} does not exist.")



@pytest.mark.user_case
@pytest.mark.cost_level_4
@pytest.mark.parametrize(
    'run_script',
    EXAMPLE_CASES
    )
def test_examples(run_script):
    run_script = Path(f'{__MOLA_PATH__}/../examples/{run_script}')
    
    if not run_script.exists():
        pytest.skip(f"cannot find {run_script}")

    # Also skip test if already launched
    # if already_launch:
    #     pytest.skip(f"{run_script} already launched")

    run_script(run_script)

# def check_cases():
#     for run_script in EXAMPLE_CASES:
#         run_script = Path(f'{__MOLA_PATH__}/../examples/{run_script}')
#         path = run_script.parent()
#         script_name = run_script.name()
        


#         # check remotely if COMPLETED
#         # if so, use python3 visualize.py and repatriate light files with mola_repatriate -l

# def _skip_if_not_with_current_solver(script_name): 
#     if solver == 'elsa' and 'elsa' not in script_name:
#         pytest.skip(f"cannot find {run_script}")
