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

def add_aerodynamic_coefficients_to( integral_extraction : dict, ApplicationContext : dict):
    
    t : cgns.Tree = integral_extraction["Data"] 
    zone : cgns.Zone
            
    for zone in t.zones():
        for container_name in [n.name() for n in zone.group(Type='FlowSolution_t')]:
            
            fx, fy, fz, tx, ty, tz = _get_forces_and_moments_from(zone, container_name)
            coefs = _get_or_create_coefficients_from(zone, container_name)
            
            _update_force_coefficients(coefs, fx, fy, fz, ApplicationContext)
            _update_torque_coefficients(coefs, tx, ty, tz, ApplicationContext)


def _get_or_create_coefficients_from(zone : cgns.Zone, container_name : str,
    field_names = ['CL','CD','CS','CX','CY','CZ','CmX','CmY','CmZ']):

    coefs = zone.fields(field_names, Container=container_name,
                        BehaviorIfNotFound='create', return_type='dict')

    return coefs


def _get_forces_and_moments_from(zone : cgns.Zone, container_name : str,
    field_names = [ 'ForceX', 'ForceY', 'ForceZ', 'TorqueX','TorqueY','TorqueZ']):

    existing = zone.fields(field_names, Container=container_name,
                    BehaviorIfNotFound='raise', return_type='dict')
    fx, fy, fz = [existing[n] for n in ['ForceX','ForceY','ForceZ']]
    tx, ty, tz = [existing[n] for n in ['TorqueX','TorqueY','TorqueZ']]
    
    return fx, fy, fz, tx, ty, tz

def _update_force_coefficients(coefs : dict, fx, fy, fz, ApplicationContext : dict):
    flux_coef      = ApplicationContext["FluxCoef"]
    lift_direction = ApplicationContext["LiftDirection"]
    drag_direction = ApplicationContext["DragDirection"]
    side_direction = ApplicationContext["SideDirection"]

    coefs["CL"][:] = _project_load(fx,fy,fz,lift_direction) * flux_coef
    coefs["CD"][:] = _project_load(fx,fy,fz,drag_direction) * flux_coef
    coefs["CS"][:] = _project_load(fx,fy,fz,side_direction) * flux_coef

    coefs["CX"][:] = fx * flux_coef
    coefs["CY"][:] = fy * flux_coef
    coefs["CZ"][:] = fz * flux_coef


def _project_load(fx, fy, fz, vector):
    return fx*vector[0]+fy*vector[1]+fz*vector[2]

def _update_torque_coefficients(coefs : dict, tx, ty, tz, ApplicationContext : dict):
    torque_coef    = ApplicationContext["TorqueCoef"]
    lift_direction = ApplicationContext["LiftDirection"]
    drag_direction = ApplicationContext["DragDirection"]
    side_direction = ApplicationContext["SideDirection"]

    coefs["CmL"][:] = _project_load(tx,ty,tz,lift_direction) * torque_coef
    coefs["CmD"][:] = _project_load(tx,ty,tz,drag_direction) * torque_coef
    coefs["CmS"][:] = _project_load(tx,ty,tz,side_direction) * torque_coef

    coefs["CmX"][:] = tx * torque_coef
    coefs["CmY"][:] = ty * torque_coef
    coefs["CmZ"][:] = tz * torque_coef

