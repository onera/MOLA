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

def new_coefficients_from(zone : cgns.Zone, container_name : str,
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

def get_forces_from(zone : cgns.Zone, container_name : str,
                     field_names = [ 'ForceX', 'ForceY', 'ForceZ']):

    try:
        existing = zone.fields(field_names, Container=container_name,
                        BehaviorIfNotFound='raise', return_type='dict')
    except ValueError as e:
        raise MolaMissingFieldsError(f"missing required forces at {zone.path()}") from e

    fx, fy, fz = [existing[n] for n in ['ForceX','ForceY','ForceZ']]
    
    return fx, fy, fz


def get_moments_from(zone : cgns.Zone, container_name : str,
                      field_names = ['TorqueX','TorqueY','TorqueZ']):

    try:
        existing = zone.fields(field_names, Container=container_name,
                        BehaviorIfNotFound='raise', return_type='dict')
    except ValueError as e:
        raise MolaMissingFieldsError(f"missing required moments at {zone.path()}") from e

    tx, ty, tz = [existing[n] for n in ['TorqueX','TorqueY','TorqueZ']]
    
    return tx, ty, tz


def project_load(fx, fy, fz, vector):
    return fx*vector[0]+fy*vector[1]+fz*vector[2]