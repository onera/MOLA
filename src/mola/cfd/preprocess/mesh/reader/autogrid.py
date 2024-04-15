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
from mola.logging import mola_logger, MolaException
from ..families import join_families


SCALE_DICT = dict(
    mm = 0.001,
    cm = 0.01,
    dm = 0.1,
    m  = 1.
)

def reader(component):

    name = component['Name'] if 'Name' in component else ''
    mola_logger.info(f'Read component {name} with Autogrid reader')
        
    component.setdefault('Tolerance', 1e-8)
    unit = component.get('unit', 'm')
    InitialFrame = component.get('InitialFrame', dict(Point=[0,0,0], Axis1=[0,0,1], Axis2=[1,0,0], Axis3=[0,1,0]))

    DefaultPositioning = [
        dict(
            Type='TranslationAndRotation',
            InitialFrame=InitialFrame,
            RequestedFrame=dict(
                Point=[0,0,0],
                Axis1=[1,0,0],
                Axis2=[0,1,0],
                Axis3=[0,0,1]),
            ),
        dict(
            Type  = 'scale',
            Scale = SCALE_DICT[unit],
            ),
    ]

    component.setdefault('Positioning', DefaultPositioning)
    component.setdefault('Connection', [])

    mesh = cgns.load(component['Source'])
    clean_autogrid_log_bases(mesh)

    # Only if grid connectivities are not already in the mesh
    # TODO: Test on the presence of GC
    component['Connection'].append(dict(Type='Match', Tolerance=component['Tolerance']))

    periodic_connections = get_periodic_match_from_Autogrid_BladeNumber(mesh, component['Tolerance'])
    component['Connection'] += periodic_connections

    clean_mesh_from_autogrid(mesh)

    nb_of_bases = len(mesh.bases())
    if nb_of_bases != 1:
        raise MolaException(f"component {component['Name']} must have exactly 1 base (got {nb_of_bases})")

    base = mesh.bases()[0]
    try:
        base.setName(component['Name'])
    except KeyError:
        component['Name'] = base.name()

    return base

def get_periodic_match_from_Autogrid_BladeNumber(mesh, Tolerance):
    base = mesh.bases()[0]
    angles = set()
    for node in base.group(Name='BladeNumber'):
        angles.add(360./float(node.value()))
    
    Connections = []
    for angle in angles:
        mola_logger.info('  angle = {:g} deg ({} blades)'.format(angle, int(360./angle)))
        Connections.append(
            dict(Type='PeriodicMatch', Tolerance=Tolerance, RotationAngle=[angle,0.,0.])
            )
    return Connections

def clean_mesh_from_autogrid(t): #, basename='Base#1', zonesToRename={}):
    '''
    Clean a CGNS mesh from Autogrid 5.
    The sequence of operations performed are the following:

    #. remove useless nodes specific to AG5
    #. rename base
    #. rename zones
    #. clean Joins & Periodic Joins
    #. clean Rotor/Stator interfaces
    #. join HUB and SHROUD families

    Parameters
    ----------

        t : PyTree
            CGNS mesh from Autogrid 5

        basename: str
            Name of the base. Will replace the default AG5 name.

        zonesToRename : dict
            Each key corresponds to the name of a zone to modify, and the associated
            value is the new name to give.

    Returns
    -------

        t : PyTree
            modified mesh tree

    '''
    clean_family_properties(t)
    # rename_zones(t, zonesToRename=dict())
    clean_grid_connectivities(t)

    # # Clean RS interfaces
    # t.findAndRemoveNodes(Type='InterfaceType')
    # t.findAndRemoveNodes(Type='DonorFamily')

    # Join HUB and SHROUD families
    join_families(t, 'HUB')
    join_families(t, 'SHROUD')
    return t

def clean_autogrid_log_bases(t):
    t.findAndRemoveNodes(Name='Numeca*', Type='CGNSBase', Depth=1)
    t.findAndRemoveNodes(Name='meridional_base', Type='CGNSBase', Depth=1)
    t.findAndRemoveNodes(Name='tools_base', Type='CGNSBase', Depth=1)

    t.findAndRemoveNodes(Name='blockName', Type='UserDefinedData', Depth=3)

def clean_family_properties(t):
    # Clean Names
    # - Recover BladeNumber and Clean Families
    for fam in t.group(Type='Family'): 
        fam.findAndRemoveNodes(Name='RotatingCoordinates')
        fam.findAndRemoveNodes(Name='Periodicity')
        fam.findAndRemoveNodes(Name='DynamicData')
    t.findAndRemoveNodes(Name='FamilyProperty')

def rename_zones(t, zonesToRename=dict()):
    for zone in t.zones():
        name = zone.name()
        if name in zonesToRename:
            newName = zonesToRename[name]
            mola_logger.info("Zone {} is renamed: {}".format(name, newName))
            I._renameNode(t, name, newName)
            continue
        # Delete some usual patterns in AG5
        new_name = name
        for pattern in ['_flux_1', '_flux_2', '_flux_3', '_Main_Blade']:
            new_name = new_name.replace(pattern, '')
        I._renameNode(t, name, new_name)

def clean_grid_connectivities(t):
    # Clean Joins & Periodic Joins
    # TODO: The objective should be to keep GC if there are already in the tree
    t.findAndRemoveNodes(Type='ZoneGridConnectivity_t')

    periodicFamilies = t.group(Name='*PER*', Type='Family', Depth=2)
    for familyNode in periodicFamilies:
        for BC in t.group(Type='BC'):
            for FamilyName in BC.group(Type='*FamilyName'): # FamilyName or AdditionalFamilyName
                if FamilyName.value() == familyNode.name():
                    BC.remove()
                    break
        familyNode.remove()
