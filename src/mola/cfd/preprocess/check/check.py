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

import numpy as np
from mola.cfd import apply_to_solver
from mola.logging import mola_logger, MolaException, GREEN, ENDC
from mola.cfd.preprocess.mesh.tools import to_full_tree_at_rank_0

def apply(workflow):
    check_empty_bc(workflow)
    apply_to_solver(workflow)

def check_empty_bc(workflow):

    def isEmpty(emptyBC):
        if isinstance(emptyBC, list) or isinstance(emptyBC, np.ndarray):
            for i in emptyBC:
                return isEmpty(i)
            return False
        elif np.isfinite(emptyBC):
            return True
        else:
            raise ValueError(f'unexpected type {type(emptyBC)}')

    try:
        import Converter.PyTree as C
        import Converter.Internal as I
    except ModuleNotFoundError:
        mola_logger.warning('could not import Cassiopee Converter. Cannot check if there is any empty BC')
        return

    t = None

    from mpi4py import MPI
    mpi_size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    if mpi_size > 1:
        # TODO check /stck/jcoulet/dev/dev-Tools/maia/Support/lbernard/find_empty_bc.py
        is_dist = bool(workflow.tree.get(':CGNS#Distribution'))
        if is_dist:
            if not workflow.tree.isStructured:
                t = to_full_tree_at_rank_0(workflow.tree)
    if t is None: 
        t = workflow.tree.copy()

    I._adaptPE2NFace(t)
    emptyBC = C.getEmptyBC(t, dim=3)
    hasEmpty = MPI.COMM_WORLD.reduce(isEmpty(emptyBC))
    if rank ==0:
        if hasEmpty:
            mola_logger.error('UNDEFINED BC IN TREE')
        else:
            mola_logger.info(f'{GREEN}No undefined BC found in tree{ENDC}')
