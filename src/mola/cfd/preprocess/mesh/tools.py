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
    
    return get_surface_of_family(workflow.tree, InflowFamily)

def get_surface_of_family(tree, Family):
    import Converter.PyTree as C
    import Post.PyTree as P

    zones = C.extractBCOfName(tree, f'FamilySpecified:{Family}')
    SurfaceTree = C.convertArray2Tetra(zones)
    SurfaceTree = C.initVars(SurfaceTree, 'ones=1')
    Surface = P.integ(SurfaceTree, var='ones')[0]        # Compute normalization coefficient
    print(f'Reference surface = {Surface} m^2 (computed from family {Family})')

    return Surface

def compute_azimuthal_extension_from_family(t, FamilyName):
    '''
    Compute the azimuthal extension in radians of the mesh **t** for the row **FamilyName**.

    .. warning:: This function needs to calculate the surface of the slice in X
                 at Xmin + 5% (Xmax - Xmin). If this surface is crossed by a
                 solid (e.g. a blade) or by the inlet boundary, the function
                 will compute a wrong value of the number of blades inside the
                 mesh.

    Parameters
    ----------

        t : PyTree
            mesh tree

        FamilyName : str
            Name of the row, identified by a ``FamilyName``.

    Returns
    -------

        deltaTheta : float
            Azimuthal extension in radians

    '''
    import Converter.PyTree as C
    import Post.PyTree as P

    # Extract zones in family
    zonesInFamily = C.getFamilyZones(t, FamilyName)
    # Slice in x direction at middle range
    xmin = C.getMinValue(zonesInFamily, 'CoordinateX')
    xmax = C.getMaxValue(zonesInFamily, 'CoordinateX')
    sliceX = P.isoSurfMC(zonesInFamily, 'CoordinateX', value=xmin+0.05*(xmax-xmin))
    # Compute Radius
    C._initVars(sliceX, '{Radius}=({CoordinateY}**2+{CoordinateZ}**2)**0.5')
    Rmin = C.getMinValue(sliceX, 'Radius')
    Rmax = C.getMaxValue(sliceX, 'Radius')
    # Compute surface
    SurfaceTree = C.convertArray2Tetra(sliceX)
    SurfaceTree = C.initVars(SurfaceTree, 'ones=1')
    Surface = P.integ(SurfaceTree, var='ones')[0]
    # Compute deltaTheta
    deltaTheta = 2* Surface / (Rmax**2 - Rmin**2)
    return deltaTheta
