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

from mola.logging import mola_logger, MolaException

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
        if bc['Type'].startswith('Inflow') or bc['Type'].startswith('inj')]

    # Check unicity
    if len(InflowBCs) != 1:
        raise MolaException( 'Please provide a reference surface as "Surface" in ReferenceValues or provide a unique inflow BC in BoundaryConditions')
    
    InflowFamily = InflowBCs[0]['Family']
    Surface = get_surface_of_family(workflow.tree, InflowFamily)
    try:
        Surface *= workflow.ApplicationContext['NormalizationCoefficient'][InflowFamily]['FluxCoef']
    except:
        pass

    mola_logger.info(f'Reference surface = {Surface} m^2 (computed from inflow family {InflowFamily})')
    
    return Surface

def get_surface_of_family(tree, Family):
    import Converter.PyTree as C
    import Post.PyTree as P

    zones = C.extractBCOfName(tree, f'FamilySpecified:{Family}')
    SurfaceTree = C.convertArray2Tetra(zones)
    SurfaceTree = C.initVars(SurfaceTree, 'ones=1')
    Surface = P.integ(SurfaceTree, var='ones')[0]        # Compute normalization coefficient
    mola_logger.debug(f'Surface of family {Family} = {Surface} m^2')

    return Surface
