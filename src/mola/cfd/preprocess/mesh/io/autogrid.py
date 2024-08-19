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
    component.setdefault('CleaningMacro', 'Autogrid') 
    JoinHubAndShroudFamilies = True

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
    update_Connection_from_mesh(mesh, component, w.ApplicationContext['ShaftAxis'])

    if component['CleaningMacro'] == 'Autogrid':
        apply_cleaning_macro_autogrid(mesh, w.Solver, JoinHubAndShroudFamilies)

    nb_of_bases = len(mesh.bases())
    if nb_of_bases != 1:
        raise MolaException(f"component {component['Name']} must have exactly 1 base (got {nb_of_bases})")

    base = mesh.bases()[0]
    try:
        base.setName(component['Name'])
    except KeyError:
        component['Name'] = base.name()

    return base

def update_Connection_from_mesh(mesh, component, axis):
    # Only if grid connectivities are not already in the mesh
    # TODO: Test on the presence of GC
    # component['Connection'].append(dict(Type='Match', Tolerance=component['DefaultToleranceForConnection']))

    periodic_connections = get_periodic_match_from_Autogrid_BladeNumber(mesh, component['DefaultToleranceForConnection'], axis)
    component['Connection'] += periodic_connections

def apply_cleaning_macro_autogrid(mesh, solver, JoinHubAndShroudFamilies=True):
    clean_autogrid_log_bases(mesh)
    shorten_zones_names(mesh)
    clean_family_properties(mesh)
    mesh.findAndRemoveNodes(Type='ZoneGridConnectivity_t') # TODO: The objective should be to keep GC if there are already in the tree
    # remove_gc_abutting(mesh)
    if solver != 'sonics':
        remove_periodic_bc_and_families(mesh)

    if JoinHubAndShroudFamilies:
        join_families(mesh, 'HUB')
        join_families(mesh, 'SHROUD')

    # # Clean RS interfaces
    # t.findAndRemoveNodes(Type='InterfaceType')
    # t.findAndRemoveNodes(Type='DonorFamily')

def clean_autogrid_log_bases(t):
    t.findAndRemoveNodes(Name='Numeca*', Type='CGNSBase', Depth=1)
    t.findAndRemoveNodes(Name='meridional_base', Type='CGNSBase', Depth=1)
    t.findAndRemoveNodes(Name='tools_base', Type='CGNSBase', Depth=1)

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
        name = zone.name()
        for pattern in patterns:
            if pattern in name:
                new_name = name.replace(pattern, '')
                zone.setName(new_name)
                for node in t.group(Value=name):
                    node.setValue(new_name)

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

def remove_periodic_bc_and_families(t):
    # In a mesh from Autogrid, Periodic connectivities are stored as BC
    periodicFamilies = t.group(Name='*PER*', Type='Family', Depth=2)
    for familyNode in periodicFamilies:
        for BC in t.group(Type='BC'):
            for FamilyName in BC.group(Type='*FamilyName'): # FamilyName or AdditionalFamilyName
                if FamilyName.value() == familyNode.name():
                    BC.remove()
                    break
        familyNode.remove()
