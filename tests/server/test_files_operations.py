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
import os
import shutil
from mola.server import files_operations as FOP
from mola.logging import check_error_message
import mola.naming_conventions as names

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_is_existing_path_local_file(tmp_path):
    filepath = tmp_path / 'test_is_existing_path_FILE'
    with open(filepath, 'w') as fi:
        fi.write('test')
    assert FOP.is_existing_path(filepath)
    assert FOP.is_file(filepath)
    assert not FOP.is_directory(filepath)
    os.unlink(filepath)
    assert not FOP.is_existing_path(filepath)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_is_existing_path_local_directory(tmp_path):
    dirpath = tmp_path / 'test_is_existing_path_DIR'
    try:
        shutil.rmtree(dirpath)
    except: 
        pass
    os.makedirs(dirpath)
    assert FOP.is_existing_path(dirpath)
    assert not FOP.is_file(dirpath)
    assert FOP.is_directory(dirpath)
    shutil.rmtree(dirpath)
    assert not FOP.is_existing_path(dirpath)



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_scp_local_destination_is_a_directory(tmp_path):
    filename = '.dummy_test_file'
    source = tmp_path / filename
    with open(source, 'w') as fi:
        fi.write('test')

    # destination is a directory: the file must be copied inside
    destination_dir = tmp_path / '.new_dummy_dir'
    destination = destination_dir / filename

    FOP.scp(source, str(destination_dir) + '/')

    assert FOP.is_file(destination)
    shutil.rmtree(destination_dir)
    assert not FOP.is_existing_path(destination_dir)
    os.unlink(source)
    assert not FOP.is_existing_path(source)
    


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_scp_local_destination_is_a_file(tmp_path):
    filename = '.dummy_test_file'
    source = tmp_path / filename
    with open(source, 'w') as fi:
        fi.write('test')

    # destination is a file: the file must be copied by changing its name
    destination_dir = tmp_path / '.new_dummy_dir/'
    destination = destination_dir / filename

    FOP.scp(source, destination)

    assert FOP.is_file(destination)
    shutil.rmtree(destination_dir)
    assert not FOP.is_existing_path(destination_dir)
    os.unlink(source)
    assert not FOP.is_existing_path(source)



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_scp_local_destination_is_an_existing_file(tmp_path):
    source = tmp_path / '.dummy_test_file'
    with open(source, 'w') as fi:
        fi.write('test')
    
    destination = tmp_path / '.dummy_test_file_2'
    with open(destination, 'w') as fi:
        fi.write('test')
    
    # destination is an existing file: an error should be raised
    expected_msg = f'The destination path {destination} already exists. To force copy and erase previous path, use force_copy=True.'
    check_error_message(expected_msg, FOP.scp, source, destination)

    # destination is an existing file but force copy
    FOP.scp(source, destination, force_copy=True)
    
    os.unlink(source)
    os.unlink(destination)
    assert not FOP.is_existing_path(source)
    assert not FOP.is_existing_path(destination)



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_scp_local_destination_and_source_are_the_same(tmp_path):
    source = tmp_path / '.dummy_test_file'
    with open(source, 'w') as fi:
        fi.write('test')

    expected_msg = f'The source path and the destination path are the same ({source}).'
    check_error_message(expected_msg, FOP.scp, source, source, force_copy=True)

    os.unlink(source)
    assert not FOP.is_existing_path(source)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_copy_remote_local_destination_is_a_file(tmp_path):
    filename = '.dummy_test_file'
    source = tmp_path / filename
    with open(source, 'w') as fi:
        fi.write('test')

    # destination is a file: the file must be copied by changing its name
    destination_dir = tmp_path / '.new_dummy_dir/'
    destination = destination_dir / filename

    FOP.copy_remote(source, destination)

    assert FOP.is_file(destination)
    shutil.rmtree(destination_dir)
    assert not FOP.is_existing_path(destination_dir)
    os.unlink(source)
    assert not FOP.is_existing_path(source)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_move_remote_local(tmp_path):
    filename = 'dummy_test_file'
    source = tmp_path / 'dummy_test_file'
    with open(source, 'w') as fi:
        fi.write('test')
    destination = tmp_path / 'test_dir' / filename

    FOP.move_remote(source, destination)
    assert not FOP.is_existing_path(source)
    assert FOP.is_existing_path(destination)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_read_text_file_from_errors_local(tmp_path):
    source = tmp_path / '.dummy_test_err_file.log'
    expected_err_msg = (
        'From this line the file is registered, since there is the word ERROR\n'
        'so this line is registered as well\n'
        'and this one.\n')
    with open(source, 'w') as fi:
        fi.write('This line will not be catched by scanner\n')
        fi.write('this one neither, even if it contains word UserWarning.\n')
        fi.write('this one neither, even if it contains both ERROR and UserWarning.\n')
        fi.write(expected_err_msg)

    err_msg = FOP.read_text_file_from_errors(source)
    os.unlink(source)
    assert 'SCANNED_ERRORS\n'+expected_err_msg == err_msg


@pytest.mark.network_onera
@pytest.mark.unit
@pytest.mark.cost_level_2
def test_read_text_file_from_errors_sator():
    MOLA_SOLVER = os.getenv('MOLA_SOLVER', 'no_solver')
    directory = f'/tmp_user/sator/$USER/.test_read_text_file_from_errors_sator_{MOLA_SOLVER}/'
    filename = 'dummy_test_err_file.log'
    expected_err_msg = (
        'From this line the file is registered, since there is the word ERROR\n'
        'so this line is registered as well\n'
        'and this one.\n')
    full_txt = 'This line will not be catched by scanner\nThis one neither\n'+expected_err_msg
    FOP.save_file_maybe_remote(filename, full_txt, directory, machine='sator', force_copy=True)
    err_msg = FOP.read_text_file_from_errors(directory+filename, 'sator')
    FOP.remove_path(directory+filename,'sator',file_only=True)
    assert 'SCANNED_ERRORS\n'+expected_err_msg+'\n' == err_msg




@pytest.mark.unit
@pytest.mark.cost_level_0
def test_read_last_run_error_file_in_log_directory_local(tmp_path):

    log_path = os.path.join(tmp_path,names.DIRECTORY_LOG)
    os.makedirs(log_path)
    nfiles = 5
    expected_msg = 'SCANNED_ERRORS\n'
    for i in range(1,nfiles):
        filename = names.FILE_STDERR.replace('.log','-%d.log'%i)
        source = tmp_path / os.path.join(log_path,filename)

        with open(source, 'w') as f:
            if i == nfiles-1:
                expected_msg += '%d this was an error\nwill show this'%i
                f.write('%d this was an error\nwill show this'%i)
            else:
                f.write('%d maybe error, but ignored'%i)

    out = FOP.read_last_run_error_file_in_log_directory(tmp_path)

    assert out == expected_msg



if __name__ == '__main__':
    import pathlib
    module_directory = pathlib.Path(__file__).parent.resolve()
    # test_read_text_file_from_errors_local(module_directory)
    test_read_last_run_error_file_in_log_directory_local(module_directory)