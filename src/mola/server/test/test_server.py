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
import sys
import os
import time
from mola.server import server as SV
from mola.server import files_operations as FOP
from mola.logging import MolaException

onera_only = pytest.mark.skipif(SV.get_network() != 'onera', reason="test on ONERA machines")

@onera_only
@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize('path_machine', [
    ('/tmp_user/sator/test', 'sator'),
    ('/scratchm/toto/', 'spiro'),
])
def test_guess_machine_from_path(path_machine):
    path, machine = path_machine
    assert SV.guess_machine_from_path(path) == machine

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_guess_machine_from_path_error():
    try:
        SV.guess_machine_from_path('.') 
        assert False
    except MolaException:
        return
    except: 
        assert False

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_submit_command():
    try:
        localhost = SV.guess_localhost()
    except:
        # submit_command command cannot be tested
        return
    
    # create an empty file with a python command send with submit_command
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_submit_command_file')
    SV.submit_command(f'touch {filename}', localhost)
    os.remove(filename)

@onera_only
@pytest.mark.unit
@pytest.mark.cost_level_1
def test_submit_command_sator():
    machine = 'sator'
    # create an empty file with a python command send with submit_command
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_submit_command_file')
    SV.submit_command(f'touch {filename}', machine)
    FOP.remove_path(filename, machine=machine)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_submit_python_command():
    try:
        localhost = SV.guess_localhost()
    except:
        # submit_command command cannot be tested
        return
    
    # create an empty file with a python command send with submit_command
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_submit_command_file')
    # python_command = f'''{sys.executable} -c "open('{filename}', 'w').close()"'''
    # SV.submit_command(python_command, localhost)
    code = f'''
with open('{filename}', 'w') as f:
    f.write('test')
'''
    SV.submit_command(sys.executable, localhost, input=code)
    
    os.remove(filename)

# @onera_only
# @pytest.mark.unit
# @pytest.mark.cost_level_1
# def test_submit_command_python_sator():
#     machine = 'sator'

#     # create an empty file with a python command send with submit_command
#     filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_submit_command_file')
#     code = f'''
# with open('{filename}', 'w') as f:
#     f.write('test')
# '''
#     SV.submit_command(sys.executable, machine, input=code)
    
#     FOP.remove_path(filename, machine=machine)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_wait_until():
    def sleep(duration):
        elapsed_time = time.time() - tic 
        if elapsed_time < duration:
            return False
        else:
            return True
    tic = time.time()
    SV.wait_until(sleep, duration=0.03, period=0.01)
    # Call the function and overshoot timeout
    try:
        SV.wait_until(sleep, duration=10, timeout=0.001, period=0.01)
        assert False
    except MolaException:
        return
    except:
        assert False
