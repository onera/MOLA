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

import Converter.PyTree as C
import Converter.Internal as I
import Post.PyTree as P

from mola import misc

def get_surface_of_inflow(workflow):
    '''
    Compute the inflow surface from the inflow families.

    Returns
    -------

        Surface : float
            surface
    '''
    # Get inflow BCs
    InflowBCs = [bc for bc in workflow.BoundaryConditions \
        if bc['type'].startswith('Inflow') or bc['type'].startswith('inj')]

    # Check unicity
    if len(InflowBCs) != 1:
        MSG = 'Please provide a reference surface as "Surface" in '
        MSG += 'ReferenceValues or provide a unique inflow BC in BoundaryConditions'
        raise Exception(misc.RED + MSG + misc.ENDC)
    
    InflowFamily = InflowBCs[0]['Family']
    
    return get_surface_of_family(workflow, InflowFamily)

def get_surface_of_family(workflow, Family):

    zones = C.extractBCOfName(workflow.tree, f'FamilySpecified:{Family}')
    SurfaceTree = C.convertArray2Tetra(zones)
    SurfaceTree = C.initVars(SurfaceTree, 'ones=1')
    Surface = P.integ(SurfaceTree, var='ones')[0]        # Compute normalization coefficient
    print(f'Reference surface = {Surface} m^2 (computed from family {Family})')

    return Surface

