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

from fnmatch import fnmatch
from mola.logging import mola_logger, MolaException

def get_bc_from_bc_type(workflow, bctypes):
    if isinstance(bctypes, str):
        bctypes = [bctypes]

    # Get BCs
    BCs = []
    for bctype in bctypes:
        BCs += [bc for bc in workflow.BoundaryConditions if fnmatch(bc['Type'], bctype)]

    # Check unicity
    if len(BCs) == 0:
        raise MolaException(f'There is no Family in BoundaryConditions matching the type {bctypes}')
    elif len(BCs) > 1:
        raise MolaException(f'There is more than one Family in BoundaryConditions matching the type {bctypes}: {[BC for BC in BCs]}')
    else:
        BC = BCs[0]

    return BC

def get_surface_of_family(tree, Family):
    import Converter.PyTree as C
    import Post.PyTree as P

    zones = C.extractBCOfName(tree, f'FamilySpecified:{Family}')
    SurfaceTree = C.convertArray2Tetra(zones)
    SurfaceTree = C.initVars(SurfaceTree, 'ones=1')
    Surface = P.integ(SurfaceTree, var='ones')[0]        # Compute normalization coefficient
    mola_logger.debug(f'Surface of family {Family} = {Surface} m^2')

    return Surface
