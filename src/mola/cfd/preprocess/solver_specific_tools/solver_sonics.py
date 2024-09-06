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

from treelab import cgns
from mola.logging import mola_logger

SonicsCGNS2MOLA = {
    'conv_flux(Momentum)#0': 'ForceX',
    'conv_flux(Momentum)#1': 'ForceY',
    'conv_flux(Momentum)#2': 'ForceZ',
    'conv_flux(Density)#0': 'MassFlow',
}

def translate_extraction_variables_to_sonics(Variables, solver):
    if isinstance(Variables, str):
        Variables = [Variables]
        
    treg = solver.terms

    translator = dict(
        Conservatives = treg.conservatives(treg.full),
        Primitives = treg.primitives(treg.full),

        Density = treg.Density,
        Momentum = treg.Momentum,
        MomentumX = treg.Momentum,
        MomentumY = treg.Momentum,
        MomentumZ = treg.Momentum,
        EnergyStagnationDensity = treg.EnergyStagnationDensity,

        Velocity = treg.Velocity,
        VelocityX = treg.Velocity,
        VelocityY = treg.Velocity,
        VelocityZ = treg.Velocity,
        Mach = treg.Mach,
        Temperature = treg.Temperature,
        # Pressure not available in Sonics !

        ViscosityMolecular = treg.LaminarViscosity,
        ViscosityEddy = treg.TurbulentViscosity,
        TurbulentDistance = treg.TurbulentDistance,

        # NormalVector = treg.SurfaceNormal,
        yPlus = treg.XYZPlusMeshSize,
        Friction = treg.SkinFriction,
        Force = treg.conv_flux(treg.Momentum), 
        MassFlow = treg.conv_flux(treg.Density),
    )
    
    sonics_var = []
    for var in Variables:
        if var in translator:
            var_tr = translator[var]
            if var_tr not in sonics_var:
                sonics_var.append(var_tr)
        else:
            mola_logger.warning(f'Unkwnown variable for SoNICS: {var}. It is ignored.')
    return sonics_var

def translate_sonics_CGNS_field_names_to_MOLA(container_node : cgns.Node):

    for node in container_node.children():
        node_name = node.name()
        if node_name in SonicsCGNS2MOLA:
            node.setName( SonicsCGNS2MOLA[node_name] )
