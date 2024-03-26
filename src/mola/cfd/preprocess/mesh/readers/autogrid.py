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

def prepare_mesh(self):
    self.tree = PRE.mesh.read_mesh(mesher='Autogrid')

    self.InputMeshes = generate_input_meshes_from_autogrid(t,
        scale=scale, rotation=rotation, tol=tol, PeriodicTranslation=PeriodicTranslation)

    t = clean_mesh_from_autogrid(t, basename=InputMeshes[0]['baseName'], zonesToRename=zonesToRename)

def reader_autogrid(base, 
                    unit='m',
                    Tolerance=1e-8, 
                    InitialFrame=dict(Point=[0,0,0], Axis1=[0,0,1], Axis2=[1,0,0], Axis3=[0,1,0])
                    ):
    
    # CAREFUL : Update Positioning, Connection, etc. rather than write over them

    ScaleDict = dict(
        mm = 0.001,
        cm = 0.01,
        dm = 0.1,
        m  = 1.
    )

    Positioning=[
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
            Scale = ScaleDict[unit],
            ),
    ]

    # Only if grid connectivities are not already in the mesh
    # TODO: Test on the presence of GC
    Connection = [
        dict(Type='Match', Tolerance=Tolerance),
    ]

    # Set automatic periodic connections
    angles = set()
    for node in base.group(Name='BladeNumber'):
        angles.add(360./float(node.value()))
    for angle in angles:
        print('  angle = {:g} deg ({} blades)'.format(angle, int(360./angle)))
        Connection.append(
            dict(type='PeriodicMatch', Tolerance=Tolerance, rotationAngle=[angle,0.,0.])
            )

    return component

def clean_mesh_from_autogrid(t, basename='Base#1', zonesToRename={}):
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

    t.findAndRemoveNodes(Name='Numeca*')
    t.findAndRemoveNodes(Name='blockName')
    t.findAndRemoveNodes(Name='meridional_base')
    t.findAndRemoveNodes(Name='tools_base')

    # Clean Names
    # - Recover BladeNumber and Clean Families
    for fam in t.group(Type='Family'): 
        t.findAndRemoveNodes(Name='RotatingCoordinates')
        t.findAndRemoveNodes(Name='Periodicity')
        t.findAndRemoveNodes(Name='DynamicData')
    t.findAndRemoveNodes(Name='FamilyProperty')

    # - Rename base
    base = t.get(Type='CGNSBase')
    base.setName(basename)

    # - Rename Zones
    for zone in t.zones():
        name = zone.name()
        if name in zonesToRename:
            newName = zonesToRename[name]
            print("Zone {} is renamed: {}".format(name, newName))
            I._renameNode(t, name, newName)
            continue
        # Delete some usual patterns in AG5
        new_name = name
        for pattern in ['_flux_1', '_flux_2', '_flux_3', '_Main_Blade']:
            new_name = new_name.replace(pattern, '')
        I._renameNode(t, name, new_name)

    # Clean Joins & Periodic Joins
    # TODO: The objective should be to keep GC if there are already in the tree
    t.findAndRemoveNodes(Type='ZoneGridConnectivity_t')

    periodicFamilies = t.group(Name='*PER*', Type='Family', Depth=2)
    for familyNode in periodicFamilies:
        for BC in t.group(Type='BC'):
            for FamilyName in BC.group(Type='*FamilyName'): # FamilyName or AdditionalFamilyName
                if FamilyName.name() == familyNode.name():
                    BC.remove()
                    break
        familyNode.remove()

    # Clean RS interfaces
    t.findAndRemoveNodes(Type='InterfaceType')
    t.findAndRemoveNodes(Type='DonorFamily')

    # Join HUB and SHROUD families
    J.joinFamilies(t, 'HUB')
    J.joinFamilies(t, 'SHROUD')
    return t

