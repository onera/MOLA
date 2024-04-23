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

LOCAL_TEST_DIR = os.path.dirname(os.path.realpath(__file__))

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_is_existing_path_local_file():
    filepath = os.path.join(LOCAL_TEST_DIR, 'test_is_existing_path_FILE')
    with open(filepath, 'w') as fi:
        fi.write('test')
    assert FOP.is_existing_path(filepath)
    assert FOP.is_file(filepath)
    assert not FOP.is_directory(filepath)
    os.unlink(filepath)
    assert not FOP.is_existing_path(filepath)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_is_existing_path_local_directory():
    dirpath = os.path.join(LOCAL_TEST_DIR, 'test_is_existing_path_DIR')
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
def test_scp_local_destination_is_a_directory():
    source = os.path.join(LOCAL_TEST_DIR, '.dummy_test_file')
    with open(source, 'w') as fi:
        fi.write('test')

    # destination is a directory: the file must be copied inside
    destination_dir = os.path.join(LOCAL_TEST_DIR, '.new_dummy_dir/')
    destination = os.path.join(destination_dir, '.dummy_test_file')

    FOP.scp(source, destination_dir)

    assert FOP.is_file(destination)
    shutil.rmtree(destination_dir)
    assert not FOP.is_existing_path(destination_dir)
    os.unlink(source)
    assert not FOP.is_existing_path(source)
    


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_scp_local_destination_is_a_file():
    source = os.path.join(LOCAL_TEST_DIR, '.dummy_test_file')
    with open(source, 'w') as fi:
        fi.write('test')

    # destination is a file: the file must be copied by changing its name
    destination_dir = os.path.join(LOCAL_TEST_DIR, '.new_dummy_dir')
    destination = os.path.join(destination_dir, '.new_dummy_file')

    FOP.scp(source, destination)

    assert FOP.is_file(destination)
    shutil.rmtree(destination_dir)
    assert not FOP.is_existing_path(destination_dir)
    os.unlink(source)
    assert not FOP.is_existing_path(source)



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_scp_local_destination_is_an_existing_file():
    source = os.path.join(LOCAL_TEST_DIR, '.dummy_test_file')
    with open(source, 'w') as fi:
        fi.write('test')
    
    destination = os.path.join(LOCAL_TEST_DIR, '.dummy_test_file_2')
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
def test_scp_local_destination_and_source_are_the_same():
    source = os.path.join(LOCAL_TEST_DIR, '.dummy_test_file')
    with open(source, 'w') as fi:
        fi.write('test')

    expected_msg = f'The source path and the destination path are the same ({source}).'
    check_error_message(expected_msg, FOP.scp, source, source, force_copy=True)

    os.unlink(source)
    assert not FOP.is_existing_path(source)

