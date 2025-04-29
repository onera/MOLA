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
import pprint
from mola.cfd import apply_to_solver
from mola.logging import mola_logger, MolaException, GREEN, ENDC
from mola.cfd.preprocess.mesh.tools import to_full_tree_at_rank_0

def apply(workflow):
    check_empty_bc(workflow)
    check_no_overlap_between_bcs(workflow.tree)
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

    assert_bc_and_connectivity_coherency(t)

    I._adaptPE2NFace(t)

    emptyBC = C.getEmptyBC(t, dim=3)
    hasEmpty = MPI.COMM_WORLD.reduce(isEmpty(emptyBC))
    if rank ==0:
        if hasEmpty:
            mola_logger.error('UNDEFINED BC IN TREE')
        else:
            check_no_empty_Family_of_BC(workflow.tree)
            mola_logger.info(f'{GREEN}No undefined BC found in tree{ENDC}')


def assert_bc_and_connectivity_coherency(tree):
    import Converter.Internal as I
    import Converter.PyTree as C
    errors = []
    for check_code in (5,6,9):
        errors += I.checkPyTree(tree, level=check_code)
    if errors:
        C.convertPyTree2File(tree, 'debug.cgns')
        raise MolaException(pprint.pformat(errors))


def check_no_empty_Family_of_BC(tree):
    for bc in tree.group(Type='BC', Value='FamilySpecified'):
        try:
            FamilyName = bc.get(Type='FamilyName').value()
        except:
            raise MolaException(f'No FamilyName in FamilyDefined BC {bc.path()}')
        
        Family = tree.get(Type='Family', Name=FamilyName, Depth=2)
        if Family.get(Type='FamilyBC', Depth=1) is None:
            raise MolaException(f'Undefined BC Family {Family.name()}: a FamilyBC node is missing.')

def check_no_overlap_between_bcs(tree):
    for zone in tree.zones():
        if zone.isUnstructured():
            # TODO develop the function for unstructured zones
            continue

        PointRanges = []
        names = []
        for bc in zone.group(Type='BC_t') + zone.group(Type='GridConnectivity1to1') + zone.group(Type='GridConnectivity'):
            PointRange = bc.get(Name='PointRange', Depth=1).value()

            for pt, name in zip(PointRanges, names):
                if is_included_in_range(PointRange, pt):
                    raise Exception(f"In {zone.name()}, {bc.name()} is included in {name}")
                elif is_included_in_range(pt, PointRange):
                    raise Exception(f"In {zone.name()}, {name} is included in {bc.name()}")

            PointRanges.append(PointRange)
            names.append(bc.name())

def is_included_in_range(PointRange1, PointRange2):

    def _build_indices_from_PointRange(PointRange):
        # return [np.arange(*range_i) for range_i in PointRange]
        indices = []
        for range_i in PointRange:
            if range_i[0] == range_i[1]:
                indices.append(np.array([range_i[0]]))
            elif range_i[0] < range_i[1]:
                indices.append(np.arange(range_i[0], range_i[1]))
            else:
                indices.append(np.arange(range_i[1], range_i[0]))
        return indices

    assert PointRange1.shape == PointRange2.shape
    indices1 = _build_indices_from_PointRange(PointRange1)
    indices2 = _build_indices_from_PointRange(PointRange2)
    for range1, range2 in zip(indices1, indices2):
        if not np.all(np.isin(range1, range2)):
            return False
    return True
