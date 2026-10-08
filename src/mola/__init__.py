#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations
import os
import getpass
from pathlib import Path

try:
    import pwd
except ImportError:  # pragma: no cover - exercised on Windows
    pwd = None

__version__ = "v2.0.1-rcX"

__MOLA_PATH__ = os.path.sep.join(__file__.split(os.path.sep)[:-2])
__ONLY_DOC__ = 0 # set to True for sphinx compilation

solver = os.getenv("MOLA_SOLVER", "none")

def get_file_owner(path) -> str:
    path = Path(path)
    if pwd is not None:
        try:
            uid = path.stat().st_uid
            return pwd.getpwuid(uid).pw_name
        except (KeyError, OSError):
            pass
    return getpass.getuser()

__owner__ = get_file_owner(__file__)

try:
    def getSHA():
        with open('{}/.git/HEAD'.format(__MOLA_PATH__), 'r') as HEAD:
            line = HEAD.readlines()[0]
            ref = line.split('ref: ')[-1].rstrip('\n')
        with open('{}/.git/{}'.format(__MOLA_PATH__, ref), 'r') as REF:
            SHA = REF.readlines()[0].rstrip('\n')
        return SHA
    __SHA__ = getSHA()
except:
    __SHA__ = 'unknown'
