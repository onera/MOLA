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

import numpy as np
from typing import Union
from fnmatch import fnmatch
from treelab import cgns
from mola.logging import mola_logger, MolaException
from ..families import join_families
from .reader import read

AUTOGRID_SPECIAL_BASES = ['Numeca*', 'meridional_base', 'tools_base']

# def set_reader_defaults(
#         Name             : str = 'auto',
#         InitialFrame     : dict = dict(Point=[0,0,0], Axis1=[0,0,1], Axis2=[1,0,0], Axis3=[0,1,0]),
#         DefaultToleranceForConnection : float = 1e-8,
#         Unit             : str  = 'm',
#         CleaningMacro    : str  = None,
#         Families         : list = None,
#         Positioning      : list = [],
#         Connection       : list = [],
#         OversetOptions   : dict = None,
#         *,
#         Source           : Union[str, cgns.tree.Tree, cgns.base.Base, cgns.zone.Zone],
#         ):
#     # This function is mandatory to be called by WorkflowInterface.
#     # Its signature will be checked.
#     # from mola.workflow.workflow_interface import WorkflowInterface
#     # default_values = WorkflowInterface.get_default_values_from_local_signature()

#     DefaultRotation =  dict(
#         Type='TranslationAndRotation',
#         InitialFrame=InitialFrame,
#         RequestedFrame=dict(
#             Point=[0,0,0],
#             Axis1=[1,0,0],
#             Axis2=[0,1,0],
#             Axis3=[0,0,1]),
#     )
#     if not any([item['Type'] == 'TranslationAndRotation' for item in Positioning]):
#         Positioning.append(DefaultRotation)


def reader(w, component):
    
    # TODO These parameters should be managed by an interface
    #################################################################################
    if w.Name == 'WorkflowPropeller':
        component.setdefault('CleaningMacro', 'Autogrid_Propeller') 
    else:
        component.setdefault('CleaningMacro', 'Autogrid_joinBC') 

    # Defaults for Connection
    component.setdefault('Connection', [])

    # Defaults for Positioning
    InitialFrame = component.get('InitialFrame', dict(Point=[0,0,0], Axis1=[0,0,1], Axis2=[1,0,0], Axis3=[0,1,0]))
    DefaultRotation =  dict(
        Type='TranslationAndRotation',
        InitialFrame=InitialFrame,
        RequestedFrame=dict(
            Point=[0,0,0],
            Axis1=[1,0,0],
            Axis2=[0,1,0],
            Axis3=[0,0,1]),
    )
    component.setdefault('Positioning', [])
    if not any([item['Type'] == 'TranslationAndRotation' for item in component['Positioning']]):
        component['Positioning'].append(DefaultRotation)
    #################################################################################
    
    mesh = read(w, component['Source'])

    blade_numbers = get_blade_number_from_mesh(mesh)

    # Apply a cleaning macro
    if component['CleaningMacro'] == 'Autogrid':
        # TODO handle families inlet_bulb* and outlet_bulb*, and merge them with other families
        apply_cleaning_macro_autogrid(mesh)
    elif component['CleaningMacro'] == 'Autogrid_joinBC':
        apply_cleaning_macro_autogrid(mesh)
        join_families(mesh, 'HUB')
        join_families(mesh, 'SHROUD')
    elif component['CleaningMacro'] == 'Autogrid_Propeller':
        apply_cleaning_macro_autogrid_propeller(mesh)


    # Check GridConnectivity nodes
    if need_to_add_gc(mesh):
        # There is no GC in the mesh --> add them automatically
        component['Connection'].append(dict(Type='Match', Tolerance=component['DefaultToleranceForConnection']))
    
    if not mesh.get(Type='Periodic'):
        # There is no periodic GC in the mesh --> add them automatically
        # update_Connection_from_mesh(mesh, w.Solver, component, w.ApplicationContext.get('ShaftAxis'))
        if w.Solver == 'sonics':
            raise MolaException('Periodic BCs must be already defined in the input mesh for sonics.')
        periodic_connections = get_periodic_match_from_Autogrid_BladeNumber(
            mesh, 
            Tolerance=component['DefaultToleranceForConnection'], 
            blade_numbers=blade_numbers,
            axis=w.ApplicationContext.get('ShaftAxis'),
            )
        component['Connection'] += periodic_connections
    else:
        remove_periodic_families_and_bc_but_keep_gc(mesh)    


    nb_of_bases = len(mesh.bases())
    if nb_of_bases != 1:
        raise MolaException(f"component {component['Name']} must have exactly 1 base (got {nb_of_bases})")

    base = mesh.bases()[0]
    try:
        base.setName(component['Name'])
    except KeyError:
        component['Name'] = base.name()

    return base
            
def apply_cleaning_macro_autogrid(mesh: cgns.Tree):
    clean_autogrid_log_bases(mesh)
    clean_family_properties(mesh)
    remove_gc_abutting(mesh)

    shorten_zones_names(mesh)

def apply_cleaning_macro_autogrid_propeller(mesh: cgns.Tree):

    apply_cleaning_macro_autogrid(mesh)

    def get_unique_zone_family_name(mesh: cgns.Tree):
        zone_families = []
        for family_node in mesh.group(Type='Family', Depth=2):
            family = family_node.name()
            for zone in mesh.zones():
                if zone.get(Type='*FamilyName', Value=family, Depth=1):
                    zone_families.append(family)
                    break

        if len(zone_families) > 1:
            raise MolaException(
                f'More than one Family of Zones found in mesh: {zone_families}. '
                'It is uncompatible with WorkflowPropeller.'
                )
        
        elif len(zone_families) == 0:
            raise MolaException('No Family of Zones found in mesh.')

        return zone_families[0]

    # Family name "Propeller" is mandatory for WorkflowPropeller

    # For butterfly mesh, Autogrid uses default family names
    mesh.renameFamily('inlet_bulb', 'Propeller')
    mesh.renameFamily('outlet_bulb', 'Propeller')

    mesh.findAndRemoveNode(Name='Propeller', Type='Family', Depth=2)  # for next line. This family will be recreated after
    zone_family_name = get_unique_zone_family_name(mesh)  # at this stage, there must be only one family of zones remaining
    mesh.renameFamily(f'*{zone_family_name}*', 'Propeller')

    # rename blade family
    guess_names_for_blade_family = [
        'Propeller_Propeller', 'Propeller_Blade', 'Propeller_BLADE', 'Propeller_Main_Blade', 'Propeller_MAIN_BLADE',
        'Propeller_far_field_SOLID_1'  # blade tip
        ]
    for fam in guess_names_for_blade_family:
        if mesh.get(Type='Family', Name=fam, Depth=2) is not None:
            mesh.renameFamily(fam, 'BLADE')
    # Put blade tip in a different family, to let the possibility to user to use WallInscid BC if wanted
    # FIXME Incompatible with WorkflowPropeller _compute_maximum_blade_radius
    # mesh.renameFamily('Propeller_far_field_SOLID_1', 'BLADE_TIP') 

    # For convenience
    mesh.renameFamily('FAR_FIELD', 'FARFIELD')
    join_families(mesh, 'HUB')
    mesh.renameFamily('*HUB*', 'SPINNER')

    # Remove BC Families *__CON_* at the interface of Propeller and Farfield zones
    # If the mesh is well defined, these BC are redundant with GC already well defined
    for fakeBC in ['*_CON_*']:
        mesh.findAndRemoveNodes(Name=fakeBC, Type='Family', Depth=2)
        for bc in mesh.group(Type='BC'):
            for node in bc.group(Type='*FamilyName'):
                family = node.value()
                if fnmatch(family, fakeBC):
                    bc.remove()
                    break

    # Remove GC, they will be recreated by MOLA
    remove_periodic_families_and_bc_but_keep_gc(mesh)
    mesh.findAndRemoveNodes(Type='ZoneGridConnectivity')


def clean_autogrid_log_bases(t):
    for name in AUTOGRID_SPECIAL_BASES:
        t.findAndRemoveNodes(Name=name, Type='CGNSBase', Depth=1)

    t.findAndRemoveNodes(Name='blockName', Type='UserDefinedData', Depth=3)
    t.findAndRemoveNodes(Name='NumecaBlockName', Type='Descriptor', Depth=3)

def clean_family_properties(t):
    # Clean Names
    # - Recover BladeNumber and Clean Families
    for fam in t.group(Type='Family'): 
        fam.findAndRemoveNodes(Name='RotatingCoordinates')
        fam.findAndRemoveNodes(Name='Periodicity')
        fam.findAndRemoveNodes(Name='DynamicData')
    t.findAndRemoveNodes(Name='FamilyProperty')

def shorten_zones_names(t):
    # Delete some usual patterns in AG5
    patterns = ['_flux_1', '_flux_2', '_flux_3', '_Main_Blade']
    for zone in t.zones():
        for pattern in patterns:
            name = zone.name()
            if pattern in name:
                new_name = name.replace(pattern, '')
                zone.setName(new_name)
                for node in t.group(Value=name):
                    node.setValue(new_name)
    t.setUniqueZoneNames()

def get_blade_number_from_mesh(mesh):
    blade_numbers = dict()
    for family in mesh.group(Type='Family', Depth=2):
        try:
            blade_number = family.get(Name='BladeNumber').value()
        except:
            continue

        blade_numbers[family.name()] = blade_number

    return blade_numbers

def get_periodic_match_from_Autogrid_BladeNumber(mesh, Tolerance, blade_numbers=None, axis=np.array([1,0,0])):
    if blade_numbers is None:
        blade_numbers = get_blade_number_from_mesh(mesh)

    Connections = []
    for row, blade_number in blade_numbers.items():
        angle = 360./float(blade_number)
        mola_logger.info('  angle = {:g} deg ({} blades)'.format(angle, int(360./angle)), rank=0)
        Connections.append(
            dict(
                Type='PeriodicMatch', 
                Tolerance=Tolerance, 
                RotationAngle=angle*axis,
                Families=(f'{row}_PER1', f'{row}_PER2'),
                )
            )
        
    return Connections

def remove_gc_abutting(t):
    for gc in t.group(Type='GridConnectivity'):
        if gc.get(Type='GridConnectivityType', Value='Abutting'):
            gc.remove()

def remove_periodic_families_and_bc_but_keep_gc(mesh):
    # keep GC but remove BC and family
    for BC in mesh.group(Type='BC'):
        try:
            fam = BC.get(Type='FamilyName').value()
        except: 
            continue
        if "_PER" in fam:
            BC.remove()    
            mesh.findAndRemoveNode(Type='Family', Name=fam, Depth=2)

def need_to_add_gc(mesh):
    excluded_bases = []
    for name in AUTOGRID_SPECIAL_BASES:
        excluded_bases += [base.name() for base in mesh.group(Type='CGNSBase', Depth=1, Name=name)]

    for base in mesh.group(Type='CGNSBase', Depth=1):
        if base.name() in excluded_bases: 
            continue
        if base.numberOfZones() < 2:
            # normal that there is no GC because there is only on zone
            continue

        join_gc_list = [gc for gc in base.group(Type='GridConnectivity1to1') if not gc.get(Type='Periodic')]
        if len(join_gc_list) == 0:
            mola_logger.debug(f'no GridConnectivity node detected in base {base.name()}')
            return True
    return False
