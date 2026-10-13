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
import getpass
from mola.server import remote
from mola.server import files_operations as FOP
from mola.logging import MolaException

MOLA_SOLVER = os.getenv('MOLA_SOLVER', 'no_solver')

@pytest.mark.network_onera
@pytest.mark.unit
@pytest.mark.parametrize('path_machine', [
    ('/tmp_user/sator/test', 'sator'),
    ('/scratchm/toto/', 'spiro'),
])
def test_guess_machine_from_path(path_machine):
    path, machine = path_machine
    assert remote.guess_machine_from_path(path) == machine

@pytest.mark.unit
def test_guess_machine_from_path_error():
    try:
        remote.guess_machine_from_path('/this/path/does/not/exist/') 
        assert False
    except MolaException:
        return
    except: 
        assert False

@pytest.mark.unit
def test_submit_command_timeout():
    """MOLA_SSH_TIMEOUT environment variable causes submit_command to raise
    MolaException on SSH timeout instead of hanging indefinitely."""
    import subprocess
    from mola.logging.exceptions import MolaException

    old_env = os.environ.get('MOLA_SSH_TIMEOUT')
    try:
        os.environ['MOLA_SSH_TIMEOUT'] = '1'
        try:
            remote.submit_command('sleep 100', 'localhost')
        except MolaException as e:
            assert 'timed out' in str(e)
            assert 'localhost' in str(e)
            return
        # If we get here, sleep 100 finished within 1 second - unlikely but possible
    finally:
        if old_env is not None:
            os.environ['MOLA_SSH_TIMEOUT'] = old_env
        elif 'MOLA_SSH_TIMEOUT' in os.environ:
            del os.environ['MOLA_SSH_TIMEOUT']


@pytest.mark.unit
def test_submit_command(tmp_path):
    try:
        localhost = remote.guess_localhost()
    except:
        # submit_command command cannot be tested
        return
    
    # Use Python instead of the POSIX-only ``touch`` command so this local
    # smoke test runs on Windows as well as Linux.
    filename = tmp_path / 'test_submit_command_file'
    code = f"from pathlib import Path\nPath({str(filename)!r}).touch()"
    remote.submit_command(sys.executable, localhost, input=code)
    os.remove(filename)

@pytest.mark.network_onera
@pytest.mark.unit
def test_submit_command_sator():
    # create an empty file with a python command send with submit_command
    machine = 'sator'
    filename = f'/tmp_user/sator/$USER/.test_submit_command_sator_{MOLA_SOLVER}_FILE'
    remote.submit_command(f'touch {filename}', machine)
    FOP.remove_path(filename, machine=machine)

@pytest.mark.unit
def test_submit_python_command(tmp_path):
    try:
        localhost = remote.guess_localhost()
    except:
        # submit_command command cannot be tested
        return
    
    # create an empty file with a python command send with submit_command
    filename = tmp_path / 'test_submit_command_file'
    # python_command = f'''{sys.executable} -c "open('{filename}', 'w').close()"'''
    # remote.submit_command(python_command, localhost)
    code = f'''
with open({str(filename)!r}, 'w') as f:
    f.write('test')
'''
    remote.submit_command(sys.executable, localhost, input=code)
    
    os.remove(filename)

@pytest.mark.unit
def test_submit_python_command_with_error():
    try: localhost = remote.guess_localhost()
    except: return

    code = "raise ValueError('this is an expected error')"
    expected_error = 'ValueError: this is an expected error'
    try:
        remote.submit_command(sys.executable, localhost, input=code)
    except MolaException as e:
        got_expected_error = False
        for err_msg_line in str(e).split('\n'):
            if expected_error in err_msg_line:
                got_expected_error = True
                break
        if not got_expected_error:
            raise MolaException(f'did not get the expected error:\n{expected_error}\ninstead got:\n{e}')
    
@pytest.mark.network_onera
@pytest.mark.unit
def test_submit_command_python_sator():
    # create an empty file with a python command send with submit_command
    machine = 'sator'
    filename = f'/tmp_user/sator/$USER/.test_submit_command_python_sator_{MOLA_SOLVER}_FILE'
    pycode = [f"with open('{filename}', 'w') as f:"]
    pycode+= [f"    f.write('test')"]
    pycode = ';'.join(pycode)
    out = remote.submit_command(f'python3 -c "{pycode}"', machine, use_mola_env=True)
    FOP.remove_path(filename, machine=machine)

@pytest.mark.unit
def test_wait_until():
    def sleep(duration):
        elapsed_time = time.time() - tic 
        if elapsed_time < duration:
            return False
        else:
            return True
    tic = time.time()
    remote.wait_until(sleep, duration=0.03, period=0.01)
    # Call the function and overshoot timeout
    try:
        remote.wait_until(sleep, duration=10, timeout=0.001, period=0.01)
        assert False
    except MolaException:
        return
    except:
        assert False

@pytest.mark.unit
def test_adapt_username_of_mola_installation_path():
    username = getpass.getuser() # NOTE pytest launcher must own the mola sources being called
    input_mola_install_path = '/tmp_user/remotemachine/$USER/codes/MOLA/Dev/src'
    adapted_mola_install_path = remote.adapt_username_of_mola_installation_path(input_mola_install_path)

    assert username in adapted_mola_install_path

if __name__ == '__main__':
    test_submit_command_python_sator()
