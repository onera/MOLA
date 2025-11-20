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

import copy
import treelab.cgns as cgns
import Converter.Internal as I
from mola.logging import mola_logger, MolaException, MolaUserError

structured_locations = ('imin','imax','jmin','jmax','kmin','kmax')

def apply(workflow):

    if any(['Families' in component for component in workflow.RawMeshComponents]):

        mola_logger.info("  🏷 defining tags", rank=0)

        t = workflow.tree
        from mpi4py import MPI
        mpi_size = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
        if mpi_size > 1:
            import maia
            is_dist = bool(workflow.tree.get(':CGNS#Distribution'))
            if is_dist:
                from mola.cfd.preprocess.mesh.tools import to_full_tree_at_rank_0
                t = to_full_tree_at_rank_0(workflow.tree)  
                if rank==0:
                    t = cgns.castNode(t)          

        if rank == 0:
            for base in t.bases():
                # Add new BC families
                component = workflow.get_component(base.name())
                try:
                    families = component['Families']
                except: 
                    families = []

                for operation in families:
                    FamilyName = operation['Name']
                    location   = operation.get('Location')
                    set_bc_family_from_location(base, FamilyName, location)  # use cassiopee, and need a full_tree as input

        if mpi_size > 1:
            MPI.COMM_WORLD.barrier()
            workflow.tree = maia.factory.full_to_dist_tree(t, MPI.COMM_WORLD, owner=0)
            workflow.tree = cgns.castNode(workflow.tree)
            MPI.COMM_WORLD.barrier()
        else:
            workflow.tree = cgns.castNode(workflow.tree)

    for base in workflow.tree.bases():
        # Add families that exist in zones or BC in base if needed
        append_bc_families_to_base(base)
        append_zone_families_to_base(base)

        # Add a default family to untagged zones if needed
        append_default_zone_family_to_zones(base)

def set_bc_family_from_location(base, FamilyName, location):
    import Converter.PyTree as C

    mola_logger.info(f'setting Family {FamilyName} in base {base.name()}', rank=0)
                    
    if shall_define_overlap_type_directly(FamilyName):
        specification = 'BCOverlap'
    else:
        specification = 'FamilySpecified:'+FamilyName

    if location in structured_locations:
        for zone in base.zones():
            C._addBC2Zone(zone, FamilyName, specification, location)

    elif location == 'remaining':
        C._fillEmptyBCWith(base, FamilyName, specification, dim=base.dim())

    elif location.startswith('plane'):
        if not base.isStructured():
            msg = f'component "{base.name()}" is not composed exclusively of '
            msg+= f'structured zones: hence, BC family "{FamilyName}" cannot '
            msg+= f'be applied at requested location "{location}"'
            raise ValueError(msg)

        for zone in base.zones():
            WindowTags = get_window_tags_at_plane(cgns.castNode(zone), planeTag=location)
            for winTag in WindowTags:
                C._addBC2Zone(zone, FamilyName, specification, winTag)

def shall_define_overlap_type_directly( family_name : str):
    # HACK https://elsa.onera.fr/issues/7869
    # HACK https://elsa.onera.fr/issues/7868
    lower_case_family = family_name.lower()
    if family_name.startswith('F_OV_') or 'overset' in lower_case_family or 'overlap' in lower_case_family:
        return True
    return False

def get_window_tags_at_plane(zone : cgns.Zone, planeTag='planeXZ', tolerance=1e-8):
    '''
    Returns the windows keywords of a structured zone that entirely lies (within
    a geometrical tolerance) on a plane provided by user.

    Parameters
    ----------

        zone : zone
            a structured zone

        planeTag : str
            a keyword used to specify the requested plane.
            Possible tags: ``'planeXZ'``, ``'planeXY'`` or ``'planeYZ'``

        tolerance : float
            maximum geometrical distance allowed to all window
            coordinates to be satisfied if the window is a valid candidate

    Returns
    -------

        WindowTagsAtPlane : :py:class:`list` of :py:class:`str`
            A list containing any of the
            following window tags: ``'imin', 'imax', 'jmin', 'jmax', 'kmin', 'kmax'``.

            .. important:: If no window lies on the plane, the function returns an empty list.
                If more than one window entirely lies on the plane, then the returned
                list will have several items.
    '''
    WindowTags = ('imin','imax','jmin','jmax','kmin','kmax')
    Windows = zone.boundaries()

    if planeTag.endswith('XZ') or planeTag.endswith('ZX'):
        DistanceVariable = 'y'
    elif planeTag.endswith('XY') or planeTag.endswith('YX'):
        DistanceVariable = 'z'
    elif planeTag.endswith('YZ') or planeTag.endswith('ZY'):
        DistanceVariable = 'x'
    else:
        raise AttributeError('planeTag %s not implemented'%planeTag)

    WindowTagsAtPlane = []
    for window, tag in zip(Windows, WindowTags):
        if DistanceVariable == 'x':
            coordinate = window.x()
        elif DistanceVariable == 'y':
            coordinate = window.y()
        else:
            coordinate = window.z()

        PositiveDistance = coordinate.max()
        NegativeDistance = coordinate.min()
        if abs(PositiveDistance) > tolerance: continue
        if abs(NegativeDistance) > tolerance: continue
        WindowTagsAtPlane += [tag]

    return WindowTagsAtPlane

def append_bc_families_to_base(base):
    cgns.castNode(base)
    AllFamilyNames = set()
    for zone in base.zones():
        for zbc in zone.group(Type='ZoneBC', Depth=1):
            for bc in zbc.group(Type='BC', Depth=1):
                if bc.value() != 'FamilySpecified': continue
                FamilyNameNode = bc.get(Type='FamilyName', Depth=1)
                if not FamilyNameNode: continue
                AllFamilyNames.add( FamilyNameNode.value() )
    for FamilyName in AllFamilyNames:
        cgns.Node(Name=FamilyName, Type='Family', Parent=base)

def append_zone_families_to_base(base):
    families_to_add = []
    for zone in base.zones():
        FamilyName = zone.get(Type='FamilyName', Depth=1)
        if FamilyName is not None and not base.get(Type='Family', Name=FamilyName.value(), Depth=1):
            families_to_add.append(FamilyName.value())

    for family in families_to_add:
        cgns.Node(Name=family, Type='Family', Parent=base)

def append_default_zone_family_to_zones(base, default_family_name='DefaultFamily'):
    need_to_add_default_family = False
    for zone in base.zones():
        FamilyName = zone.get(Type='FamilyName', Depth=1)
        if not FamilyName:
            cgns.Node(Name='FamilyName', Type='FamilyName', Value=default_family_name, Parent=zone)

    if need_to_add_default_family and not base.get(Name=default_family_name, Type='Family', Depth=1):
        cgns.Node(Name=default_family_name, Type='Family', Parent=base)

def join_families(t, pattern, mode=2):
    '''
    In the CGNS tree t, gather all the Families <ROW_I>_<PATTERN>_<SUFFIXE> into
    Families <ROW_I>_<PATTERN>, so as many as rows.
    Useful to join all the row_i_HUB* or (row_i_SHROUD*) together

    Parameters
    ----------

        t : PyTree
            A PyTree read by Cassiopee

        pattern : str
            The pattern used to gather CGNS families. Should be for example 'HUB' or 'SHROUD'
    '''
    fam2remove = []
    fam2keep = []
    # Loop on the BCs in the tree
    for bc in t.group(Type='BC'):
        # Get BC family name
        famBC_node = bc.get(Type='FamilyName')
        if not famBC_node: 
            continue
        famBC = famBC_node.value()
        # Check if the pattern is present in FamilyBC name
        if pattern not in famBC:
            continue
        # Split to get the short name based on pattern
        split_fanBC = famBC.split(pattern)
        assert len(split_fanBC) == 2, (
            f'The pattern {pattern} is present more than once in the FamilyBC f{famBC}. ' 
            'It must be more selective.'
        )
        preffix, suffix = split_fanBC
        if mode == 1:
            # Add the short name to the set fam2keep
            short_name = f'{preffix}{pattern}'
            if short_name not in fam2keep: 
                fam2keep.append(short_name)
            if suffix != '':
                # Change the family name
                famBC_node.setValue(short_name)
                if famBC not in fam2remove: 
                    fam2remove.append(famBC)
        else:
            # Add the short name to the set fam2keep
            short_name = f'{pattern}'
            if short_name not in fam2keep: 
                fam2keep.append(short_name)
            if preffix != '' or suffix != '':
                # Change the family name
                famBC_node.setValue(short_name)
                if famBC not in fam2remove: 
                    fam2remove.append(famBC)

    # Remove families
    for fam in fam2remove:
        mola_logger.debug(f'Remove family {fam}')
        t.findAndRemoveNodes(Name=fam, Type='Family', Depth=2)

    # Check that families to keep still exist
    base = t.get(Type='CGNSBase')
    added_families = []
    for fam in fam2keep:
        fam_node = t.get(Name=fam, Type='Family', Depth=2)
        if fam_node is None:
            added_families.append(fam)
            cgns.Node(Name=fam, Type='Family', Parent=base)

    # Print information on which families were added or removed
    logger_text = ''  

    if len(fam2remove) == 1:
        logger_text += f'Remove family {fam2remove[0]}. '
    elif len(fam2remove) > 1:
        logger_text += f'Remove the following families: {", ".join(fam2remove)}. '

    if len(added_families) == 1:
        logger_text += f'Add family {added_families[0]}.'
    elif len(added_families) > 1:
        logger_text += f'Add the following families: {", ".join(added_families)}.'
    
    if len(logger_text) > 0:
        mola_logger.user_warning(logger_text)

def get_family_to_BCType( t : cgns.Tree ) -> dict:
    families_to_bctype = dict()
    for famnode in t.group(Type='Family', Depth=2):
        bctype = famnode.get(Type='FamilyBC')
        if bctype is not None:
            families_to_bctype[famnode.name()] = bctype.value()
    return families_to_bctype

def get_zone_family_from_bc_or_gc_family(tree: cgns.Tree, bc_family: str) -> str:
    for zone in tree.zones():
        for bc in zone.group(Type='BC') + zone.group(Type='GridConnectivity*'):
            if bc.get(Type='*FamilyName', Value=bc_family):
                FamilyName = zone.get(Type='FamilyName', Depth=1)
                if FamilyName:
                    return FamilyName.value()
    
    raise MolaUserError(f'Cannot find a zone Family from the BC or GC Family {bc_family}. Check the input tree and family names.')

def get_bc_family_nodes_from_patterns(tree, patterns):
    # TODO Put this function (and others that are associated) in treelab ? 
    bc_family_names = get_bc_family_names_from_patterns(tree, patterns)
    nodes = []
    for name in bc_family_names:
        family_node = tree.get(Type='Family_t', Name=name, Depth=2)
        if not family_node:
            raise MolaException(f'could not find CGNSBase_t/Family_t named "{name}"')
        nodes += [ family_node ]
    return nodes

def get_bc_family_names_from_patterns(tree, patterns) -> list:
    nodes = get_bc_family_name_nodes_from_patterns(tree, patterns)
    family_names = set()
    for n in nodes:
        family_names.add(n.value())
    return list(family_names)

def get_bc_family_name_nodes_from_patterns(tree, patterns):
    family_name_nodes = []
    for pattern in generate_case_variations(patterns):
        for zone in tree.zones():
            zone_bc = zone.get(Type='ZoneBC_t', Depth=1)
            if not zone_bc:
                raise MolaException(f'zone {zone.path()} did not have a node ZoneBC_t')
            nodes = zone_bc.group(Name='FamilyName', Value=f'*{pattern}*')
            for family_name_node in nodes:
                family_name_nodes += [ family_name_node ]

    return family_name_nodes

def generate_case_variations(patterns):
    '''
    For each <NAME> in the list **patterns**, add Name, name and NAME.
    '''
    extended_patterns = copy.deepcopy(patterns)
    for pattern in patterns:
        newNames = [pattern.lower(), pattern.upper(), pattern.capitalize()]
        for name in newNames:
            if name not in extended_patterns:
                extended_patterns.append(name)
    return extended_patterns

def _ungroupBCsByBCType(t, forced_starting=''):
    for BC in I.getNodesFromType(t,'BC_t'):
        BCvalue = I.getValue(BC)
        if BCvalue == 'FamilySpecified':
            FamilyBC = I.getValue(I.getNodeFromName1(BC,'FamilyName'))
            BCType = getFamilyBCTypeFromFamilyBCName(t, FamilyBC)
            if forced_starting:
                if BCType.startswith(forced_starting):
                    BCType = forced_starting
            I.setValue(BC,BCType)

def getFamilyBCTypeFromFamilyBCName(t, FamilyBCName):
    '''
    Get the *BCType* of BCs defined by a given family BC name.

    Parameters
    ----------

        t : PyTree
            main CGNS tree

        FamilyBCName : str
            requested name of the *FamilyBC*

    Returns
    -------

        BCType : str
            the resulting *BCType*. Returns:py:obj:`None` if **FamilyBCName** is not
            found
    '''
    FamilyNode = I.getNodeFromNameAndType(t, FamilyBCName, 'Family_t')
    if not FamilyNode: return

    FamilyBCNode = I.getNodeFromType1(FamilyNode, 'FamilyBC_t')
    if not FamilyBCNode: return

    FamilyBCNodeType = I.getValue(FamilyBCNode)
    if FamilyBCNodeType != 'UserDefined': return FamilyBCNodeType

    SolverBC = I.getNodeFromName1(FamilyNode,'.Solver#BC')
    if SolverBC:
        SolverBCType = I.getNodeFromName1(SolverBC,'type')
        if SolverBCType:
            BCType = I.getValue(SolverBCType)
            return BCType

    SolverOverlap = I.getNodeFromName1(FamilyNode,'.Solver#Overlap')
    if SolverOverlap: return 'BCOverlap'

    BCnodes = I.getNodesFromType(t, 'BC_t')
    for BCnode in BCnodes:
        FamilyNameNode = I.getNodeFromName1(BCnode, 'FamilyName')
        if not FamilyNameNode: continue

        FamilyNameValue = I.getValue( FamilyNameNode )
        if FamilyNameValue == FamilyBCName:
            BCType = I.getValue( BCnode )
            if BCType != 'FamilySpecified': return BCType
            break
