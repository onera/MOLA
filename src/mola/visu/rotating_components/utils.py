
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

def unwrap(t, meanRadius=None):
    import Converter.PyTree as C
    import Converter.Internal as I
    import Transform.PyTree as T

    name = I.getName(t)
    unwrappedTree = T.join(I.getZones(t))
    # unwrappedTree = I.copyTree(t)

    C._initVars(unwrappedTree, '{Radius} = ({CoordinateY}**2+{CoordinateZ}**2)**0.5')
    C._initVars(unwrappedTree, '{Theta} = arctan2({CoordinateZ},{CoordinateY})')
    if not meanRadius:
        meanRadius = C.getMeanValue(unwrappedTree, 'Radius')
    C._initVars(unwrappedTree, '{{CoordinateY}} = {meanR}*{{Theta}}'.format(meanR=meanRadius))
    C._initVars(unwrappedTree, '{{CoordinateZ}} = 0')
    I.setName(unwrappedTree, name)

    # T._cart2Cyl(unwrappedTree, (0,0,0), (1,0,0))
    return unwrappedTree, meanRadius

def make_rotate(tree, iteration: int, Numerics, Motion):
    '''
    For a turbomachinery case, make rotate all the domains depending on their motion. 
    '''
    import maia

    currentTime = Numerics['TimeAtInitialState'] + Numerics['TimeStep'] * (iteration - Numerics['IterationAtInitialState'])

    for row, motion in Motion.items(): 

        for zone in tree.zones():
            if zone.get(Type='*FamilyName', Value=row, Depth=1):
                maia.algo.transform_affine(
                    zone, 
                    translation=motion['TranslationSpeed'], 
                    rotation_center=motion['RotationAxisOrigin'], 
                    rotation_angle=motion['RotationSpeed']*currentTime
                )

    tree = cgns.castNode(tree)
    return tree


def duplicate_rows(tree, **rows_to_duplicate):
    '''
    Duplicate tree for visualization, for turbomachinery applications.

    Parameters
    ----------
    tree : PyTree
        Input tree. It must be either a top tree or a base.

    rows_to_duplicate : kwargs
        Row families to duplicate. For each key, the value corresponds to the wished number of duplications.

    Raises
    ------
    TypeError
        The input tree must be either a top tree or a base.
    '''
    from mola.cfd.preprocess.mesh.duplicate import _duplicate_with_maia, _duplicate_with_cassiopee
    from mola.cfd.preprocess.mesh.tools import to_distributed

    if isinstance(tree, (cgns.Tree, cgns.Base)):
        tree = to_distributed(tree)
        # duplication_parameters = dict((row, dict(number_of_duplications=n_dupli, is_360=False)) for row, n_dupli in rows_to_duplicate.items())
        duplication_operations = [
            dict(Family=row, NumberOfDuplications=n_dupli)
            for row, n_dupli in rows_to_duplicate.items()
        ]
        # tree = _duplicate_with_maia(tree, duplication_operations, merge_zones=True)
        tree = _duplicate_with_cassiopee(tree, duplication_operations)
    else:
        raise TypeError('The input tree must be either a top tree or a base')

    return tree
