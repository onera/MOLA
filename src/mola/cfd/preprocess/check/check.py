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
from mola.logging import mola_logger, MolaException, GREEN, YELLOW, ENDC
from mola.cfd.preprocess.mesh.tools import to_full_tree_at_rank_0
from mola.cfd.preprocess.mesh.families import shall_define_overlap_type_directly

def apply(workflow):
    check_empty_bc(workflow)
    check_no_overlap_between_bcs(workflow.tree)
    mola_logger.info(" - making solver-specific checkings", rank=0)
    apply_to_solver(workflow)

def check_empty_bc(workflow):
    mola_logger.info(" - checking if there is any undefined BC", rank=0)
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
        mola_logger.warning('could not import Cassiopee Converter. Cannot check if there is any empty BC', rank=0)
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

    no_rotor_stator_interface_in_tree = all([
        bc['Type'] not in ['MixingPlane', 'UnsteadyRotorStatorInterface', 'ChorochronicInterface'] 
            for bc in workflow.BoundaryConditions
            ])
    if no_rotor_stator_interface_in_tree:
        # Not compatible with maia renaming of connectivities at a rotor/stator interface with *.N?.P?
        assert_bc_and_connectivity_coherency(t)
    
    _ignore_undefined_periodic_boundaries_in_2D_structured_grids(t)

    # CAUTION BUG https://elsa.onera.fr/issues/12076#note-5
    if workflow.Solver == 'sonics': 
        mola_logger.user_warning(f'UNABLE TO DETERMINE IF UNDEFINED BC EXIST https://elsa.onera.fr/issues/12076#note-5')
        return
    
    I._adaptPE2NFace(t)
    emptyBC = C.getEmptyBC(t, dim=3)
    empty_bcs = MPI.COMM_WORLD.reduce(isEmpty(emptyBC))
    
    raise_error = False
    if rank ==0:
        if empty_bcs:
            raise_error = True
        else:
            check_no_empty_Family_of_BC(workflow.tree)
            mola_logger.info(f'{GREEN}No undefined BC found in tree{ENDC}', rank=0)

    if raise_error:
        _raise_undefined_bc_error_saving_undefined_bc_surfaces(t, rank)


def _ignore_undefined_periodic_boundaries_in_2D_structured_grids(t):
    import Converter.PyTree as C
    
    for zone in t.zones():
        if not zone.isStructured(): return
        
        shape = zone.shape()
        dims = len(shape)
        if dims != 3:
            raise MolaException(f"must be 3D, but got dims={dims} for zone {zone.path()}")

        if shape[2] != 2: return

        C._addBC2Zone(zone, "IGNORED", "IGNORED_t",'kmin')
        C._addBC2Zone(zone, "IGNORED", "IGNORED_t",'kmax')


def assert_bc_and_connectivity_coherency(tree):
    mola_logger.info(' - checking BC and connectivity coherency',rank=0)
    import Converter.Internal as I
    import Converter.PyTree as C
    
    checks = {
        5:'valid BC range', 
        6:'valid opposite BC range for match and nearmatch',
        # 9:'valid connectivity', # BUG https://github.com/onera/Cassiopee/issues/324
    }
    
    errors = []
    for check_code in list(checks):
        errors += I.checkPyTree(tree, level=check_code)
    if errors:
        C.convertPyTree2File(tree, 'debug.cgns')
        with open('debug.log', 'w') as fi:
            fi.write(pprint.pformat(errors))
        raise MolaException('Error in BC or connectivity coherency. See debug.cgns and debug.log (contains the error returned by Cassiopee)') 


def _raise_undefined_bc_error_saving_undefined_bc_surfaces(t, rank):
    import Converter.PyTree as C
    import Converter.Internal as I

    C._fillEmptyBCWith(t,'UNDEFINED','UNDEFINED_t')
    surfs = C.extractBCOfType(t,'UNDEFINED_t')
    if not surfs:
        raise MolaException("expected undefined BC but finnally did not found them")
    I._rmNodesByType(surfs,'FlowSolution_t')
    C.convertPyTree2File(surfs,f'debug_undefined_bc_{rank}.cgns')
    raise MolaException(f'UNDEFINED BC IN TREE, CHECK debug_undefined_bc_{rank}.cgns')




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
    mola_logger.info(" - checking if there is no overlapping BC", rank=0)
    for zone in tree.zones():
        if zone.isUnstructured():
            # TODO develop the function for unstructured zones
            continue

        PointRanges = []
        names = []
        for bc in zone.group(Type='BC_t') + zone.group(Type='GridConnectivity1to1') + zone.group(Type='GridConnectivity'):
            PointRange = bc.get(Name='PointRange', Depth=1).value()


            for pt, name in zip(PointRanges, names):
                accepted_overlapping = shall_define_overlap_type_directly(name) and shall_define_overlap_type_directly(bc.name())

                if accepted_overlapping: 
                    continue
                
                if are_point_ranges_overlapping(PointRange, pt):
                    raise Exception(f"In {zone.name()}, {bc.name()} is included in {name}, because {PointRange} lies in {pt}")
                elif are_point_ranges_overlapping(pt, PointRange):
                    raise Exception(f"In {zone.name()}, {name} is included in {bc.name()}")

            PointRanges.append(PointRange)
            names.append(bc.name())

def are_point_ranges_overlapping(PointRange1, PointRange2):

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


def are_point_ranges_overlapping(pr1, pr2):

    if len(pr1) != len(pr2):
        return False

    overlapping_mask = len(pr1) * [False]

    for i in range(len(pr1)):

        row1 = pr1[i,:]
        row2 = pr2[i,:]
        if (min(row1) < max(row2) and max(row1) > min(row2)) or \
            (row1[0] == row1[1] == row2[0] == row2[1]):
            overlapping_mask[i] = True
    
    return all(overlapping_mask)

    
