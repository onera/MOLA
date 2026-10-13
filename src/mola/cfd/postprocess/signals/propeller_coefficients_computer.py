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
import numpy as np 

from mola.logging.exceptions import MolaMissingFieldsError, MolaUserError
from . import fields_manipulator as fields


def add_aerodynamic_coefficients_to( integral_extraction : dict, ApplicationContext : dict,
        diameter : float, density : float, axial_velocity : float):
    
    if integral_extraction["Type"] != "Integral":
        return

    t : cgns.Tree = integral_extraction["Data"] 
    zone : cgns.Zone
            

    for zone in t.zones():

        for container_name in [n.name() for n in zone.group(Type='FlowSolution_t')]:
            coefs = fields.new_coefficients_from(zone, container_name,
                field_names = ['Thrust','Torque','Power','CT','CP','FigureOfMeritHover','PropulsiveEfficiency'])

            fx, fy, fz = fields.get_forces_from(zone, container_name)

            _update_force_coefficients(coefs, fx, fy, fz,
                ApplicationContext, diameter, density)
            
            try:
                tx, ty, tz = fields.get_moments_from(zone, container_name)
                _update_torque_coefficients(coefs, tx, ty, tz, 
                    ApplicationContext, diameter, density, axial_velocity)
            except MolaMissingFieldsError:
                pass # HACK relaxing until TODO https://gitlab.onera.net/numerics/solver/sonics/-/issues/83

def _update_force_coefficients(coefs : dict, fx, fy, fz, ApplicationContext : dict,
        diameter : float, density : float):
    axis = ApplicationContext["ShaftAxis"]
    axis /= np.linalg.norm(axis)

    RPS = getRPS(ApplicationContext)

    Thrust = fields.project_load(fx,fy,fz, np.sign(RPS)*axis )

    coefs["Thrust"][:] = Thrust
    coefs["CT"][:] = Thrust / (density * RPS**2 * diameter**4)


def _update_torque_coefficients(coefs : dict, tx, ty, tz, ApplicationContext : dict,
        diameter : float, density : float, axial_velocity : float):
    axis = ApplicationContext["ShaftAxis"]
    axis /= np.linalg.norm(axis)

    rotation_center = ApplicationContext.get('RotationCenter',np.array([0.0,0.0,0.0]))
    
    if np.linalg.norm(rotation_center) != 0:
        raise NotImplementedError("rotatation center must be (0,0,0)")

    RPS = getRPS(ApplicationContext)
    RPM = 60 * RPS

    Torque = fields.project_load(tx,ty,tz,-np.sign(RPS)*axis)
    Power = Torque * np.abs(RPM) * np.pi/30

    Thrust = coefs['Thrust']
    CT = coefs['CT']
    CP = Power / (density * RPS**3 * diameter**5)
    FM = np.sqrt(2./np.pi)* np.sign(CT)*np.abs(CT)**1.5 / CP
    eta = axial_velocity*Thrust/Power

    coefs['Torque'][:] = Torque
    coefs['Power'][:] = Power
    coefs['CT'][:] = CT
    coefs['CP'][:] = CP
    coefs['FigureOfMeritHover'][:] = FM
    coefs['PropulsiveEfficiency'][:] = eta


def getRPS( ApplicationContext : dict):
    omega_units = ApplicationContext["ShaftRotationSpeedUnit"]
    if omega_units == 'rpm':
        rev_per_second = ApplicationContext["ShaftRotationSpeed"] / 60.0
    elif omega_units == 'rad/s':
        rev_per_second = ApplicationContext["ShaftRotationSpeed"] / (2*np.pi)
    else:
        raise MolaUserError(f'got wrong ShaftRotationSpeedUnit "{omega_units}", shall be "rpm" or "rad/s"')

    return rev_per_second