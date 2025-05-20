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

from pathlib import Path
from mola.cfd import apply_to_solver
from mola.cfd.coprocess.tools import write_tagfile
import mola.naming_conventions as names

def apply(workflow):
    try:
        remove_status_files()
        apply_to_solver(workflow)
    except BaseException as e:
        try: 
            comanager = workflow._coprocess_manager
        except AttributeError:
            # becase it may fail also before instantiating coprocess manager
            comanager = None 
        write_tagfile(names.FILE_JOB_FAILED, comanager)
        raise BaseException(e)

def remove_status_files():
    for filename in names.STATUS_FILES:
        Path(filename).unlink(missing_ok=True)
