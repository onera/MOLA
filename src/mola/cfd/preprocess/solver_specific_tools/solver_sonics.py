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
    'IterationValues': 'Iteration',  # HACK ValveLawRadialEquilibrium writes a node IterationValues instead Iteration like other Triggers
    'conv_flux(Momentum)X': 'ForceX',
    'conv_flux(Momentum)Y': 'ForceY',
    'conv_flux(Momentum)Z': 'ForceZ',
    'conv_torqueX': 'TorqueX',
    'conv_torqueY': 'TorqueY',
    'conv_torqueZ': 'TorqueZ',
    'conv_flux(Density)': 'MassFlow',
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
        Pressure = treg.Pressure,

        ViscosityMolecular = treg.LaminarViscosity,
        ViscosityEddy = treg.TurbulentViscosity,
        TurbulentDistance = treg.TurbulentDistance,

        # NormalVector = treg.SurfaceNormal,
        yPlus = treg.XYZPlusMeshSize,
        SkinFriction = treg.SkinFriction,

        # FIXME for Force and Torque, we should extract the sum ofocnv and diff fluxes
        Force = treg.conv_flux(treg.Momentum), 
        Torque = treg.conv_torque, 
        MassFlow = treg.conv_flux(treg.Density),
    )
    
    sonics_var = []
    for var in Variables:
        if var in translator:
            var_tr = translator[var]
            if var_tr not in sonics_var:
                sonics_var.append(var_tr)
        else:
            mola_logger.user_warning(f'Unkwnown variable for SoNICS: {var}. It is ignored.', rank=0)
    return sonics_var

def translate_extraction_variables_to_sonics_function(Variables):
    if isinstance(Variables, str):
        Variables = [Variables]

    translator = dict(
        Conservatives = lambda treg: treg.conservatives(treg.full),
        Primitives = lambda treg: treg.primitives(treg.full),

        Density = lambda treg: treg.Density,
        Momentum = lambda treg: treg.Momentum,
        MomentumX = lambda treg: treg.Momentum,
        MomentumY = lambda treg: treg.Momentum,
        MomentumZ = lambda treg: treg.Momentum,
        EnergyStagnationDensity = lambda treg: treg.EnergyStagnationDensity,

        Velocity = lambda treg: treg.Velocity,
        VelocityX = lambda treg: treg.Velocity,
        VelocityY = lambda treg: treg.Velocity,
        VelocityZ = lambda treg: treg.Velocity,
        Mach = lambda treg: treg.Mach,
        Temperature = lambda treg: treg.Temperature,
        Pressure = lambda treg: treg.Pressure,

        ViscosityMolecular = lambda treg: treg.LaminarViscosity,
        ViscosityEddy = lambda treg: treg.TurbulentViscosity,
        TurbulentDistance = lambda treg: treg.TurbulentDistance,

        # NormalVector = treg.SurfaceNormal,
        yPlus = lambda treg: treg.XYZPlusMeshSize,
        SkinFriction = lambda treg: treg.SkinFriction,

        # HACK for integral outputs, need treg.dummy
        # see https://numerics.gitlab-pages.onera.net/coupling/miles/v0.0.4dev/known_issues/index.html#extracting-both-convective-diffusive-fluxes-in-the-same-trigger-deadlocks
        Force = lambda treg: treg.conv_flux(treg.Momentum), 
        Torque = lambda treg: treg.conv_torque, 
        MassFlow = lambda treg: treg.conv_flux(treg.Density),
    )
    
    sonics_var = []
    for var in Variables:
        if var in translator:
            var_tr = translator[var]
            if var_tr not in sonics_var:
                sonics_var.append(var_tr)
        else:
            mola_logger.user_warning(f'Unkwnown variable for SoNICS: {var}. It is ignored.', rank=0)

    sonics_var_fun = lambda treg: [var(treg) for var in sonics_var]
    return sonics_var_fun

def translate_sonics_CGNS_field_names_to_MOLA(container_node : cgns.Node):

    for node in container_node.children():
        node_name = node.name()
        if node_name.startswith('dummy('):
            node_name = node_name.replace('dummy(', '')
            if node_name.endswith(')'):
                # remove final ")" that is expected
                # NOTE that if test is performed because if the name is too long, the final parenthesis may be not present
                node_name = node_name[:-1]  

        if node_name in SonicsCGNS2MOLA:
            node.setName( SonicsCGNS2MOLA[node_name] )
