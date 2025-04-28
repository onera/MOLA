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
from mola.workflow.workflow import Workflow
from mola.cfd.postprocess.remove_cfd_files import remove_cfd_files

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_write_and_remove_files(tmp_path):
    w = Workflow(RunManagement=dict(RunDirectory=tmp_path))
    remove_cfd_files.write_dummy_files_for_testing(w)
    remove_cfd_files.apply(w)
