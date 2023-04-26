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

import Converter.PyTree as C # TODO _addBC2Zone, _fillEmptyBCWith
import mola.cgns as cgns
# from mola.cfd.preprocess.boundary_conditions import BoundaryConditionsNames

structured_locations = ('imin','imax','jmin','jmax','kmin','kmax')

def apply(workflow):
    t = workflow.tree
    for base in t.bases():
        component = workflow.get_component(base.name())

        if 'Families' not in component: continue

        for operation in component['Families']:
            FamilyName = operation['Name']
            location   = operation['Location']
            print(f'setting Family {FamilyName} at base {base.name()}')
            
            if location in structured_locations:
                for zone in base.zones():
                    C._addBC2Zone(zone, FamilyName,
                                  'FamilySpecified:'+FamilyName,
                                  location)

            elif location == 'remaining':
                C._fillEmptyBCWith(base, FamilyName,
                    'FamilySpecified:'+FamilyName,dim=base.dim())

            elif location.startswith('plane'):
                if not base.isStructured():
                    msg = f'component "{base.name()}" is not composed exclusively of '
                    msg+= f'structured zones: hence, BC family "{FamilyName}" cannot '
                    msg+= f'be applied at requested location "{location}"'
                    raise ValueError(msg)

                WindowTags = getWindowTagsAtPlane(zone, planeTag=location)

        appendFamiliesToBase(base)


    workflow.tree = cgns.castNode(t)

def getWindowTagsAtPlane(zone, planeTag='planeXZ', tolerance=1e-8):
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
    Windows = zone.exteriorFaces()

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

def appendFamiliesToBase(base):
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
