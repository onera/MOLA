import numpy as np

from noder import Node, Array
from noder.cgns import Zone
from noder.core import nodeToPyCGNS, pyCGNSToNode

# -------------------------------- decorators -------------------------------- #

def using_node_as_first_arg_and_returns_a_node(fun):
    def wrapper(*args, **kwargs):
        args = list(args[:])
        args[0] = pyCGNSToNode(args[0])
        result_with_node = fun(*args, **kwargs)
        if result_with_node:
            return nodeToPyCGNS(result_with_node)
        else:
            return None

    return wrapper

def using_node_as_first_arg_and_returns_a_list_of_nodes(fun):
    def wrapper(*args, **kwargs):
        args = list(args[:])
        args[0] = pyCGNSToNode(args[0])
        result_with_node = fun(*args, **kwargs)
        return [nodeToPyCGNS(n) for n in result_with_node]

    return wrapper

def using_node_as_single_arg_and_natural_return(fun):
    def wrapper(*args, **kwargs):
        args = list(args[:])
        args[0] = pyCGNSToNode(args[0])
        return fun(*args, **kwargs)

    return wrapper

def using_node_as_first_and_second_arg_and_returns_a_node(fun):
    def wrapper(*args, **kwargs):
        args = list(args[:])
        args[0] = pyCGNSToNode(args[0])
        args[1] = pyCGNSToNode(args[1])
        result_with_node = fun(*args, **kwargs)
        if result_with_node:
            return nodeToPyCGNS(result_with_node)
        else:
            return None

    return wrapper

# ----------------------------- node attributes ----------------------------- #

@using_node_as_single_arg_and_natural_return
def getName(node : Node):
    return node.name()

@using_node_as_single_arg_and_natural_return
def getType(node : Node):
    return node.type()

@using_node_as_single_arg_and_natural_return
def getValue(node : Node):
    data = node.data()

    if isinstance(data, Array):
        if data.hasString():
            return data.extractString()
        else:
            return data.getPyArray()
    return None

# ----------------------------- research by name ----------------------------- # 


@using_node_as_first_arg_and_returns_a_node
def getNodeFromName(node : Node, name : str):
    return node.pick().by_name_glob(name)

@using_node_as_first_arg_and_returns_a_node
def getNodeFromName1(node : Node, name : str):
    return node.pick().child_by_name(name)

@using_node_as_first_arg_and_returns_a_node
def getNodeFromName2(node : Node, name : str):
    return node.pick().by_name(name, depth=2)

@using_node_as_first_arg_and_returns_a_node
def getNodeFromName3(node : Node, name : str):
    return node.pick().by_name(name, depth=3)

@using_node_as_first_arg_and_returns_a_list_of_nodes
def getNodesFromName(node : Node, name : str):
    return node.pick().all_by_name_glob(name)

# ----------------------------- research by type ----------------------------- # 

@using_node_as_first_arg_and_returns_a_node
def getNodeFromType(node : Node, type : str):
    return node.pick().by_type_glob(type)

@using_node_as_first_arg_and_returns_a_node
def getNodeFromType1(node : Node, type : str):
    return node.pick().child_by_type(type)

@using_node_as_first_arg_and_returns_a_node
def getNodeFromType2(node : Node, type : str):
    return node.pick().by_type(type, depth=2)

@using_node_as_first_arg_and_returns_a_node
def getNodeFromType3(node : Node, type : str):
    return node.pick().by_type(type, depth=3)

@using_node_as_first_arg_and_returns_a_list_of_nodes
def getNodesFromType(node : Node, type : str):
    return node.pick().all_by_type_glob(type)

# ------------------------------- manipulation ------------------------------- #

@using_node_as_first_and_second_arg_and_returns_a_node
def addChild(node : Node, child_to_add : Node, pos=-1, override_sibling_by_name=True):
    node.add_child(child_to_add, position=pos, override_sibling_by_name=override_sibling_by_name)
    return node

def createNode(name, ntype, value=None, children=None, parent=None):
    node = Node(name=name,type=ntype)

    if value:
        node.set_data(value)

    if children:
        list_of_children = [pyCGNSToNode(n) for n in children]
        node.add_children(list_of_children)
    
    if parent:
        parent_node = pyCGNSToNode(parent)
        node.attach_to(parent_node)
    
    return nodeToPyCGNS(node)

# I.getZones
# I.copyTree
# I.newCGNSBase
# I._rmNodesByName
