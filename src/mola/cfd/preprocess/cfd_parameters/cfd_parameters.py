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
from mola.logging import mola_logger, MolaException, MolaAssertionError
from mola.cfd import apply_to_solver

def apply(workflow):
    add_governing_equations(workflow)
    apply_to_solver(workflow)

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        elif isinstance(v, list):
            d[k].extend(v)
        else:
            d[k] = v
    return d

def add_governing_equations(workflow):
    '''
    Add the nodes corresponding to `FlowEquationSet_t`
    '''

    FlowEquationSet = cgns.Node(Name='FlowEquationSet', Type='FlowEquationSet_t')

    if workflow.Turbulence['Model'] in ['LES', 'ILES', 'DNS','Laminar']:
        Value = 'NSLaminar'

    elif workflow.Turbulence['Model'] == 'Euler':
        Value = 'Euler'

    else:
        Value = 'NSTurbulent'

    cgns.Node(Parent=FlowEquationSet,
              Name='GoverningEquations',
              Type='GoverningEquations_t',
              Value=Value)

    cgns.Node(Parent=FlowEquationSet,
              Name='EquationDimension',
              Type='EquationDimension_t',
              Value=workflow.ProblemDimension)

    workflow.tree.findAndRemoveNodes(Type='FlowEquationSet_t', Depth=3)
    for base in workflow.tree.bases():
        base.addChild(FlowEquationSet)
