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
from treelab import cgns
from mola.cfd.preprocess.motion.motion import is_mobile


def BCWall(workflow, Family, Motion=None):
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
    cgns.Node( Name='FamilyBC', Value='BCWallViscous', Type='FamilyBC', Parent=wall_family)
    # add motion
    if Motion is not None:
        mobile_coef = 1. if is_mobile(Motion) else 0.
        wall_family.setParameters('.Solver#Property', mobile_coef=mobile_coef)  

def BCWallInviscid(workflow, Family, Motion=None):
    '''
    Set an inviscid wall boundary condition.

    Parameters
    ----------

        workflow : Workflow object

        Family : str
            Name of the family on which the boundary condition will be imposed

    '''
    wall_family = workflow.tree.get(Name=Family, Type='Family', Depth=2)
    wall_family.findAndRemoveNodes(Type='FamilyBC', Depth=1)
    cgns.Node( Name='FamilyBC', Value='BCWallInviscid', Type='FamilyBC', Parent=wall_family)
    # add motion
    if Motion is not None:
        mobile_coef = 1. if is_mobile(Motion) else 0.
        wall_family.setParameters('.Solver#Property', mobile_coef=mobile_coef)  


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

def BCInj1(workflow, Family, **kwargs):
    '''
    Set a Inj1 boundary condition.

    Parameters
    ----------

        workflow : Workflow object

        Family : str
            Name of the family on which the boundary condition will be imposed

    '''
    ImposedVariables = Inj1_interface(workflow, **kwargs)

    family = workflow.tree.get(Name=Family, Type='Family', Depth=2)
    family.findAndRemoveNodes(Type='FamilyBC', Depth=1)
    cgns.Node(Name='FamilyBC', Value='BCInj1', Type='FamilyBC', Parent=family)
    # BC data cannot be set in Family, they must be in BC nodes, 
    # even if data are scalar
    set_bc_with_imposed_variables(workflow.tree, Family, ImposedVariables)
    
def Inj1_interface(workflow, **kwargs):
    from mola.cfd.preprocess.boundary_conditions.solver_elsa import inj1_interface
    ImposedVariables, _ = inj1_interface(workflow, **kwargs)
    # names of nodes has no importance, but the order is crucial ['dOx', 'dOy', 'dOz', 'pa', 'ha']
    order_of_variables = [
        'VelocityUnitVectorX',
        'VelocityUnitVectorY',
        'VelocityUnitVectorZ',
        'PressureStagnation',
        'EnthalpyStagnation',
        'TurbulentSANuTilde',
    ]

    try:
        ImposedVariables = dict(
            sorted(ImposedVariables.items(), 
                key= lambda item: order_of_variables.index(item[0])
                )
            )
    except ValueError:
        from pprint import pformat as pretty
        raise ValueError(pretty(ImposedVariables))

    return ImposedVariables

def BCOutpres(workflow, Family, Pressure=None):
    '''
    Set an Outpres boundary condition.

    Parameters
    ----------

        workflow : Workflow object

        Family : str
            Name of the family on which the boundary condition will be imposed

    '''
    if Pressure is None:
        Pressure =workflow.Flow['Pressure']
        
    family = workflow.tree.get(Name=Family, Type='Family', Depth=2)
    family.findAndRemoveNodes(Type='FamilyBC', Depth=1)
    cgns.Node(Name='FamilyBC', Value='BCOutpres', Type='FamilyBC', Parent=family)
    # BC data cannot be set in Family, they must be in BC nodes, 
    # even if data are scalar
    set_bc_with_imposed_variables(workflow.tree, Family, dict(Pressure=Pressure))

def set_bc_with_imposed_variables(tree, Family, ImposedVariables):
    for value in ImposedVariables.values():
        assert isinstance(value, (float, int))
        # TODO impose a 2D map

    def _get_bc_size(bc):
        PointRange = bc.get(Type='IndexRange').value()
        bc_shape = PointRange[:, 1] - PointRange[:, 0]
        if bc_shape[0] == 0:
            bc_shape = (bc_shape[1], bc_shape[2])
        elif bc_shape[1] == 0:
            bc_shape = (bc_shape[0], bc_shape[2])
        elif bc_shape[2] == 0:
            bc_shape = (bc_shape[0], bc_shape[1])
        bc_size = bc_shape[0] * bc_shape[1]
        return bc_size

    for bc in tree.group(Type='BC'):
        if not bc.get(Type='FamilyName').value() == Family:
            continue
        ones = np.ones(_get_bc_size(bc))
        local_values = dict((key, value*ones) for key, value in ImposedVariables.items())
        bc.setParameters('.Solver#Property', **local_values)

