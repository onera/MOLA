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

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()

from treelab import cgns
from mola.logging import MolaException
from mola.pytree.user import checker

def iso_surface(t, IsoSurfaceField, IsoSurfaceValue, IsoSurfaceContainer, Name, tool='cassiopee'):
    CellDimension = t.base().dim()

    if tool.lower() == 'cassiopee':
        from .extractions_with_cassiopee import iso_surface
        zones = iso_surface(t, IsoSurfaceField, IsoSurfaceValue, IsoSurfaceContainer)
        extraction = get_renamed_tree(zones, Name, CellDimension=CellDimension)
        extraction = cgns.castNode(extraction)
        restore_families(extraction, t)

    elif tool.lower() == 'maia':
        from .extractions_with_maia import iso_surface
        extraction = iso_surface(t, IsoSurfaceField, IsoSurfaceValue, IsoSurfaceContainer, comm=MPI.COMM_WORLD)
        extraction = cgns.castNode(extraction)
        extraction.bases()[0].setName(Name)

    else:
        raise MolaException(f'iso_surface is available only with cassiopee (now tool={tool})')

    return extraction
    
    
def extract_bc(t, Family, BaseName=None, tool='cassiopee'):
    CellDimension = t.base().dim()
    if not BaseName:
        BaseName = Family

    if tool.lower() == 'cassiopee':
        from .extractions_with_cassiopee import extract_bc
        zones = extract_bc(t, Family=Family, Name=None, Type=None)
        extraction = get_renamed_tree(zones, BaseName, CellDimension=CellDimension)
        extraction = cgns.castNode(extraction)
        restore_families(extraction, t)
    
    elif tool == 'maia':
        from .extractions_with_maia import extract_bc_from_family
        tree = extract_bc_from_family(t, Family=Family, comm=MPI.COMM_WORLD)
        extraction = cgns.castNode(tree)
    
    elif tool == 'maia_zsr':
        from .extractions_with_maia import extract_bc_from_zsr
        zones = extract_bc_from_zsr(t, Family=Family, comm=MPI.COMM_WORLD)
        extraction = get_renamed_tree_maia(zones, BaseName, CellDimension=CellDimension)
        extraction = cgns.castNode(extraction)
        for zsr in extraction.group(Type='ZoneSubRegion'):
            zsr.setType('FlowSolution')
            zsr.findAndRemoveNode(Name='PointList')
        restore_families(extraction, t)

    else:
        raise MolaException(f'extract_bc is available only with cassiopee (now tool={tool})')

    return extraction
    
def get_renamed_tree(zones, basename, CellDimension=3, PhysicalDimension=3):
    tree = cgns.Tree()
    base = cgns.Base(Parent=tree, Name=basename)
    base.setCellDimension(CellDimension-1)
    base.setPhysicalDimension(PhysicalDimension)

    if zones is None: 
        return tree
        
    for i, zone in enumerate(zones):
        zone = cgns.castNode(zone)
        # The name of the parent zone is kept in a temporary node .parentZone, 
        # that will be removed before saving
        # There might be a \ in zone name if it is a result of C.ExtractBCOfType
        zoneName = zone.name().split('\\')[0]
        cgns.Node(Name='.parentZone', Type='UserDefinedData_t', Value=zoneName, Parent=zone)
        # Rename zones like the base
        zone.setName(f'{basename}_R{rank}N{i}')
        base.addChild(zone)

    return tree

def get_renamed_tree_maia(zones, basename, CellDimension=3, PhysicalDimension=3):
    tree = cgns.Tree(basename=[])
    base = tree.bases()[0]
    base.setName(basename)
    base.setCellDimension(CellDimension-1)
    base.setPhysicalDimension(PhysicalDimension)

    if zones is None: 
        return tree
        
    for i, zone in enumerate(zones):
        if len(zone) != 4:
            raise TypeError(f"wrong zone: {str(zone)}")

        zone = cgns.castNode(zone)
        # The name of the parent zone is kept in a temporary node .parentZone, 
        # that will be removed before saving
        # There might be a \ in zone name if it is a result of C.ExtractBCOfType
        zoneName = zone.name()
        suffix = get_maia_suffix(zoneName)
        cgns.Node(Name='.parentZone', Type='UserDefinedData_t', Value=zoneName, Parent=zone)
        # Rename zones like the base
        zone.setName(f'{basename}{suffix}')
        base.addChild(zone)

    return tree

def get_maia_suffix(name):
    import re
    # regular expression to find a pattern ".P*.N*", with * a number with 1 to 5 figures
    maia_pattern = r'\.P(\d{1,5})\.N(\d{1,5})'
    match = re.search(maia_pattern, name)
    if match:
        pattern_found = match.group(0) 
    else:
        pattern_found = ''
    return pattern_found

def restore_families(surfaces, skeleton):
    '''
    Restore families in the PyTree **surfaces** (e.g read from
    ``'surfaces.cgns'``) based on information in **skeleton** (e.g read from
    ``'main.cgns'``). Also add the ReferenceState to be able to use function
    computeVariables from Cassiopee Post module.

    .. tip:: **skeleton** may be a skeleton tree.

    Parameters
    ----------

        surfaces : PyTree
            tree where zone names are the same as in **skeleton** (or with a
            suffix in '\\<bcname>'), but without information on families and
            ReferenceState.

        skeleton : PyTree
            tree of the full 3D domain with zones, families and ReferenceState.
            No data is needed so **skeleton** may be a skeleton tree.
    '''
    ReferenceState = skeleton.get(Type='ReferenceState', Depth=2) 
    family_nodes = skeleton.group(Type='Family', Depth=2) 

    for base in surfaces.bases():
        if ReferenceState:
            base.addChild(ReferenceState)

        families_in_base = []
        for zone in base.zones():
            parentZone_node = zone.get(Name='.parentZone')
            zone_name = parentZone_node.value()
            zone_in_full_tree = skeleton.get(Name=zone_name, Type='Zone')
            if zone_in_full_tree:  
                fam = zone_in_full_tree.get(Type='FamilyName', Depth=1)
                if fam:
                    zone.addChild(fam)
                    families_in_base.append(fam.value())
            else:
                # This is an extracted BC
                fam = zone.get(Type='FamilyName', Depth=1)
                if fam: 
                    families_in_base.append(fam.value())

            parentZone_node.remove()
            
        for family in family_nodes:
            if family.name() in families_in_base:
                base.addChild(family)

def merge_bases_and_rename_unique_base(t, basename):
    # Add suffix .P<rank>.N<index> to mimic a maia part_tree
    # (previously, suffix was _R<rank>N<index> like a cassiopee part_tree)

    base0 =  t.bases()[0]
    base0.setName(basename)
    i = 0
    for zone in base0.zones():
        zone.setName(f"{basename}.P{rank}.N{i}")
        i += 1
    for base in t.bases()[1:]:
        for zone in base.zones():
            zone.setName(f"{basename}.P{rank}.N{i}")
            i += 1
            zone.moveTo(base0)
        base.remove()
