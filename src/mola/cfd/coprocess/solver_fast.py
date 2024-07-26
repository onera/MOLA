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

import os
import glob
import shutil
from fnmatch import fnmatch

from treelab import cgns

from mola.logging import MolaException
import mola.naming_conventions as names
# no relative imports possible for the following line because the current file is called by
# call_solver_specific_function in manager.py
from mola.cfd.coprocess import mola_logger, rank, comm


def perform_extractions(workflow, coprocess_manager):
    mola_logger.warning('coprocess extractions to be implemented for FAST ')
