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
from .operations import slidding_average, slidding_std, slidding_rsd

AVAILABLE_OPERATIONS_ON_SIGNALS = ['avg', 'std', 'rsd']

def apply_operations_on_signal(node: cgns.Node, var_name: str, window_size_interations: int, operations: list = None) -> None:

    try:
        Iteration = node.get(Name='Iteration').value()
    except:
        return 
    window_size = len(Iteration[Iteration>(Iteration[-1]-window_size_interations)])
    if window_size < 2: return

    operation_tree = _build_operation_tree(var_name, operations)
    _walk_on_operation_tree(operation_tree, node, window_size)

def _walk_on_operation_tree(operation_tree: cgns.Node, node: cgns.Node, window_size:int):
    unit_operations = operation_tree.getChildrenNames()
    var_name = _get_complete_variable_name(operation_tree)

    try:
        data_node = node.get(Name=var_name) 
        parent = data_node.parent()  # Because data_node may be not a direct child of the input node
        InstantaneousArray = data_node.value()
        InvalidValues = np.logical_not(np.isfinite(InstantaneousArray))
        InstantaneousArray[InvalidValues] = 0.
    except:
        return

    apply_unit_operations(parent, var_name, InstantaneousArray, unit_operations, window_size)

    # next step
    for child in operation_tree.children():
        _walk_on_operation_tree(child, node, window_size)
    
def apply_unit_operations(parent, var_name, var_value, unit_operations, window_size):
    # Sort operations to do them only once
    sorting_dict = dict(avg=0, std=1, rsd=2)
    unit_operations.sort(key=lambda n: sorting_dict[n])
    avg = None
    std = None
    rsd = None

    def save_node(name, value, parent):
        result_node = parent.get(Name=name)
        if result_node:
            result_node.setValue(value)
        else:
            cgns.Node(Type='DataArray', Name=name, Value=value, Parent=parent)

    for operation in unit_operations:

        if operation.lower() == 'avg':
            if avg is None: 
                avg = slidding_average(var_value, window_size)
            save_node(f'avg-{var_name}', avg, parent)

        elif operation.lower() == 'std':
            if std is None: 
                std = slidding_std(var_value, window_size, avg=avg)
            save_node(f'std-{var_name}', std, parent)
            
        elif operation.lower() == 'rsd':
            rsd = slidding_rsd(var_value, window_size, avg=avg, std=std)
            save_node(f'rsd-{var_name}', rsd, parent)
    
def _build_operation_tree(var: str, operations: list) -> list:
    '''
    from possible compound variable name and compound operations, return the 
    graph of unit operations to perform from the root variable.

    Parameters
    ----------
    var : str
    operations : list

    Example
    -------
    .. graphviz::
        :align: center
        :caption: operation_tree given by `_build_operation_tree('std-MassFlow', ['rsd-avg', 'std'])`

        digraph Sphinx {
            MassFlow -> std;
            std -> avg;
            std -> std2;
            std2 [label="std"];
            avg -> rsd;
        }
    '''
    if operations is None:
        operations = []
    elif isinstance(operations, str): 
        operations = [operations]

    # if any([var.startswith(op+'-') for op in AVAILABLE_OPERATIONS_ON_SIGNALS]):
    unit_vars = var.split('-')
    root_var = unit_vars[-1]
    operation_tree = cgns.Node(Name=root_var)
    current_node = operation_tree
    for prefix in reversed(unit_vars[:-1]):
        current_node = cgns.Node(Name=prefix, Parent=current_node)

    current_node.setValue(operations)
    _unfold_operation_tree(current_node)

    return operation_tree

def _unfold_operation_tree(operation_tree: cgns.Node):
    operations = operation_tree.value()
    if operations is None or len(operations) == 0:
        return
    else:
        if isinstance(operations, str):
            operations = [operations]
        for operation in operations:
            op_split = operation.split('-')
            next_operation = op_split[-1]
            remaining_operations = '-'.join(op_split[:-1])
            cgns.Node(Name=next_operation, Value=remaining_operations, Parent=operation_tree)
        # remove value of root node, to mark it is already unfold 
        operation_tree.setValue(None)
        for child in operation_tree.children():
            _unfold_operation_tree(child)

def _get_complete_variable_name(node: cgns.Node):
    '''
    if the path of the node in the operation_tree is var/avg/std, 
    then the complete name of the correspondant variable is std-avg-var
    '''
    path = node.path()
    tmp = path.split('/')
    tmp.reverse()
    return '-'.join(tmp)   


def update_zones_shape_using_iteration_number(tree : cgns.Tree, Container="FlowSolution"):

    zone : cgns.Zone
    for zone in tree.zones():
        zone_shape = zone.value()
        flow_sol = zone.get(Name=Container, Depth=1)
        if not flow_sol: continue
        it_nb_node = flow_sol.get(Name='Iteration')
        if not it_nb_node: continue
        zone_shape[0] = it_nb_node.value().size
