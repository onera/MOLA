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

def BCWall(workflow, Family):
    '''
    Set a wall boundary condition.

    Parameters
    ----------

        workflow : Workflow object

        Family : str
            Name of the family on which the boundary condition will be imposed

    '''
    wall_family = workflow.tree.get(Name=Family, Type='Family', Depth=2)
    wall_family.findAndRemoveNodes(Type='FamilyBC', Depth=1)
    cgns.Node( Name='FamilyBC', Value='BCWall', Type='FamilyBC', Parent=wall_family)


def BCFarfield(workflow, Family):
    '''
    Set a farfield boundary condition.

    Parameters
    ----------

        workflow : Workflow object

        Family : str
            Name of the family on which the boundary condition will be imposed

    '''
    farfield_family = workflow.tree.get(Name=Family, Type='Family', Depth=2)
    farfield_family.findAndRemoveNodes(Type='FamilyBC', Depth=1)
    cgns.Node( Name='FamilyBC', Value='BCFarfield', Type='FamilyBC', Parent=farfield_family )


def BCSymmetryPlane(workflow, Family):
    '''
    Set a SymmetryPlane boundary condition.

    Parameters
    ----------

        workflow : Workflow object

        Family : str
            Name of the family on which the boundary condition will be imposed

    '''
    farfield_family = workflow.tree.get(Name=Family, Type='Family', Depth=2)
    farfield_family.findAndRemoveNodes(Type='FamilyBC', Depth=1)
    cgns.Node( Name='FamilyBC', Value='BCSymmetryPlane', Type='FamilyBC', Parent=farfield_family )

