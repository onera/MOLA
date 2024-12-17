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

    mola_logger.info(f'Read component {component["Name"]} with Autogrid reader')
    
    # TODO These parameters should be managed by an interface
    #################################################################################
    component.setdefault('CleaningMacro', 'Autogrid_joinBC') 

    # Defaults for Connection
    component.setdefault('DefaultToleranceForConnection', 1e-8)
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

    if need_to_add_gc(mesh):
        # There is no GC in the mesh --> add them automatically
        component['Connection'].append(dict(Type='Match', Tolerance=component['DefaultToleranceForConnection']))
    
    if not mesh.get(Type='Periodic'):
        # There is no periodic GC in the mesh --> add them automatically
        # update_Connection_from_mesh(mesh, w.Solver, component, w.ApplicationContext.get('ShaftAxis'))
        if w.Solver == 'sonics':
            raise MolaException('Periodic BCs must be already defined in the input mesh for sonics.')
        periodic_connections = get_periodic_match_from_Autogrid_BladeNumber(mesh, component['DefaultToleranceForConnection'], w.ApplicationContext.get('ShaftAxis'))
        component['Connection'] += periodic_connections
    else:
        remove_periodic_families_and_bc_but_keep_gc(mesh)    

    if component['CleaningMacro'] == 'Autogrid':
        # TODO handle families inlet_bulb* and outlet_bulb*, and merge them with other families
        apply_cleaning_macro_autogrid(mesh)
    elif component['CleaningMacro'] == 'Autogrid_joinBC':
        apply_cleaning_macro_autogrid(mesh)
        join_families(mesh, 'HUB')
        join_families(mesh, 'SHROUD')

    nb_of_bases = len(mesh.bases())
    if nb_of_bases != 1:
        raise MolaException(f"component {component['Name']} must have exactly 1 base (got {nb_of_bases})")

    base = mesh.bases()[0]
    try:
        base.setName(component['Name'])
    except KeyError:
        component['Name'] = base.name()

    return base
            
def apply_cleaning_macro_autogrid(mesh):
    clean_autogrid_log_bases(mesh)
    clean_family_properties(mesh)
    remove_gc_abutting(mesh)

    shorten_zones_names(mesh)

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

def get_periodic_match_from_Autogrid_BladeNumber(mesh, Tolerance, axis=np.array([1,0,0])):
    base = mesh.bases()[0]
    Connections = []
    for family in base.group(Type='Family', Depth=1):
        node = family.get(Name='BladeNumber')
        if node is None:
            continue
        angle = 360./float(node.value())

        mola_logger.info('  angle = {:g} deg ({} blades)'.format(angle, int(360./angle)))
        row = family.name()
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
