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

from mola.logging.exceptions import MolaMissingFieldsError


def add_aerodynamic_coefficients_to( integral_extraction : dict, ApplicationContext : dict):
    
    if integral_extraction["Type"] != "Integral": return

    t : cgns.Tree = integral_extraction["Data"] 
    zone : cgns.Zone
            
    for zone in t.zones():
        for container_name in [n.name() for n in zone.group(Type='FlowSolution_t')]:
            coefs = _new_coefficients_from(zone, container_name)

            fx, fy, fz = _get_forces_from(zone, container_name)
            _update_force_coefficients(coefs, fx, fy, fz, ApplicationContext)
            
            try:
                tx, ty, tz = _get_moments_from(zone, container_name)
                _update_torque_coefficients(coefs, tx, ty, tz, ApplicationContext)
            except MolaMissingFieldsError:
                pass # HACK relaxing until TODO https://gitlab.onera.net/numerics/solver/sonics/-/issues/83


def _new_coefficients_from(zone : cgns.Zone, container_name : str,
    field_names = ['CL','CD','CS','CX','CY','CZ','CmL','CmD','CmS','CmX','CmY','CmZ']):

    zone.removeFields(field_names, Container=container_name)

    try:
        coefs = zone.newFields(field_names, Container=container_name,
            GridLocation='Vertex', return_type='dict')

    except Exception as e:
        _handle_exception_when_unable_to_create_fields(zone,e)

    return coefs

def _handle_exception_when_unable_to_create_fields(zone,e):
    msg = f'could not add fields to zone "{zone.path()}"'
    try:
        zone.save('debug.cgns')
        msg += ', check debug.cgns.'
    except:
        msg += ' and could NOT write it into debug.cgns.'

    msg += ' Check full Traceback.'
    
    raise Exception(msg) from e

def _get_forces_from(zone : cgns.Zone, container_name : str,
                     field_names = [ 'ForceX', 'ForceY', 'ForceZ']):

    try:
        existing = zone.fields(field_names, Container=container_name,
                        BehaviorIfNotFound='raise', return_type='dict')
    except ValueError as e:
        raise MolaMissingFieldsError(f"missing required forces at {zone.path()}") from e

    fx, fy, fz = [existing[n] for n in ['ForceX','ForceY','ForceZ']]
    
    return fx, fy, fz


def _get_moments_from(zone : cgns.Zone, container_name : str,
                      field_names = ['TorqueX','TorqueY','TorqueZ']):

    try:
        existing = zone.fields(field_names, Container=container_name,
                        BehaviorIfNotFound='raise', return_type='dict')
    except ValueError as e:
        raise MolaMissingFieldsError(f"missing required moments at {zone.path()}") from e

    tx, ty, tz = [existing[n] for n in ['TorqueX','TorqueY','TorqueZ']]
    
    return tx, ty, tz

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

