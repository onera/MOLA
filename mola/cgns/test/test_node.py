import os
import pprint
import numpy as np
import mola.cgns as cgns

def test_init1():
    node = cgns.Node()
    assert node == ['Node', None, [], 'DataArray_t']

def test_init2():
    node = cgns.Node(Name='MyName', Value=[1,2,3,4], Type='DataArray')
    assert str(node) == "['MyName', array([1, 2, 3, 4], dtype=int32), [], 'DataArray_t']"

def test_init3():
    node = cgns.Node( ['name', np.array([0,1,2]), [], 'DataArray_t'] )
    assert str(node) == "['name', array([0, 1, 2]), [], 'DataArray_t']"

def test_cast():
    node = cgns.castNode( ['name', np.array([0,1,2]), [], 'DataArray_t'] )
    assert str(node) == "['name', array([0, 1, 2]), [], 'DataArray_t']"

def test_parent():
    # create a node and attach it to another node 
    a = cgns.Node( Name='TheParent')
    b = cgns.Node( Name='TheChild', Parent=a )

    p = b.parent()
    assert p is a
    
    p = a.parent()
    assert p is None

def test_path():
    a = cgns.Node( Name='TheParent')
    b = cgns.Node( Name='TheChild', Parent=a )

    path = b.path() 
    assert path == 'TheParent/TheChild'

def test_save():
    n = cgns.Node( Name='jamon' )
    n.save('test_node_save.cgns', verbose=False)
    os.unlink('test_node_save.cgns')

def test_name():
    n = cgns.Node( Name='tortilla' )
    assert n.name() == 'tortilla'

def test_value1():
    node = cgns.Node( Value='jamon' )
    value_str = node.value() # will return a readable str
    assert value_str == 'jamon'

def test_value2():
    node = cgns.Node( Value='jamon tortilla croquetas' )

    value_str = node.value() # will return a readable list of str
    assert value_str ==  ['jamon', 'tortilla', 'croquetas']

def test_value3():
    # create a multi-dimensional array to be attributed to several nodes
    # it is very important to set order='F' !
    array = np.array( [[0,1,2],
                        [3,4,5],
                        [6,7,8]], order='F' )

    # create our two nodes, and attribute their values to our array
    jamon    = cgns.Node( Name='jamon', Value=array )
    tortilla = cgns.Node( Name='tortilla', Value=array )

    # get the value of jamon
    jamon_value = jamon.value() # in this case, the same as jamon[1]

    # get a flattened view of the array of node tortilla
    tortilla_value = tortilla.value(ravel=True)


    # our arrays share memory
    assert np.shares_memory(tortilla_value, jamon_value)
    assert np.shares_memory(tortilla_value, array)

    # hence we can modify it in different fashions, all changes will be propagated
    tortilla_value[0] = 12
    array[1,:] = -2

    assert str(tortilla_value) == "[12 -2  6  1 -2  7  2 -2  8]"

def test_setParameters(filename=''):
    t = cgns.Node(Name='main')
    
    def f(x,y): return x+y
    
    t.setParameters('Parameters',
        none=None,
        Fun=f,
        EmptyList=[],
        EmptyDict={},
        EmptyTuple=(),
        NumpyArray=np.array([0,1,2]),
        BooleanList=[True,False,False],
        Boolean=True,
        Int=12,
        IntList=[1,2,3,4],
        Float=13.0,
        FloatList=[1.0,2.0,3.0],
        Str='jamon',
        StrList=['croquetas', 'tortilla'],
        Dict={'Str':'paella','Int':12},
        DictOfDict=dict(SecondDict={'Str':'gazpacho','Int':12}),
        ListOfDict=[{'Str':'pescaito','Int':12},
                    {'Str':'calamares','Int':12},
                    {'Str':'morcilla','Int':12}],
        Node=cgns.Node(Name='n1'),
        DictOfNode={'a':cgns.Node(Name='n5'), 'b':cgns.Node(Name='n6')},
        ListOfNode=[cgns.Node(Name='n2'),cgns.Node(Name='n3'),cgns.Node(Name='n4')],
        )
    
    if filename: t.save(filename)

def test_getParameters(filename=''):
    t = cgns.Node(Name='main')

    def f(x,y): return x+y

    set_params = dict(
        none=None,
        Fun=f,
        EmptyList=[],
        EmptyDict={},
        EmptyTuple=(),
        NumpyArray=np.array([0,1,2]),
        BooleanList=[True,False,False],
        Boolean=True,
        Int=12,
        IntList=[1,2,3,4],
        Float=13.0,
        FloatList=[1.0,2.0,3.0],
        Str='jamon',
        StrList=['croquetas', 'tortilla'],
        Dict={'Str':'paella','Int':12},
        DictOfDict=dict(SecondDict={'Str':'gazpacho','Int':12}),
        ListOfDict=[{'Str':'pescaito','Int':12},
                    {'Str':'calamares','Int':12},
                    {'Str':'morcilla','Int':12}],
        Node=cgns.Node(Name='n1'),
        DictOfNode={'a':cgns.Node(Name='n5'), 'b':cgns.Node(Name='n6')},
        ListOfNode=[cgns.Node(Name='n2'),cgns.Node(Name='n3'),cgns.Node(Name='n4')],
        )

    expected="{'Boolean': array([1], dtype=int32),\n"
    expected+=" 'BooleanList': array([1, 0, 0], dtype=int32),\n"
    expected+=" 'Dict': {'Int': array([12], dtype=int32), 'Str': 'paella'},\n"
    expected+=" 'DictOfDict': {'SecondDict': {'Int': array([12], dtype=int32),\n"
    expected+="                               'Str': 'gazpacho'}},\n"
    expected+=" 'DictOfNode': {'a': None, 'b': None},\n"
    expected+=" 'EmptyDict': None,\n"
    expected+=" 'EmptyList': None,\n"
    expected+=" 'EmptyTuple': None,\n"
    expected+=" 'Float': array([13.]),\n"
    expected+=" 'FloatList': array([1., 2., 3.]),\n"
    expected+=" 'Fun': None,\n"
    expected+=" 'Int': array([12], dtype=int32),\n"
    expected+=" 'IntList': array([1, 2, 3, 4], dtype=int32),\n"
    expected+=" 'ListOfDict': [{'Int': array([12], dtype=int32), 'Str': 'pescaito'},\n"
    expected+="                {'Int': array([12], dtype=int32), 'Str': 'calamares'},\n"
    expected+="                {'Int': array([12], dtype=int32), 'Str': 'morcilla'}],\n"
    expected+=" 'ListOfNode': None,\n"
    expected+=" 'Node': None,\n"
    expected+=" 'NumpyArray': array([0, 1, 2]),\n"
    expected+=" 'Str': 'jamon',\n"
    expected+=" 'StrList': ['croquetas', 'tortilla'],\n"
    expected+=" 'none': None}"

    t.setParameters('Parameters', **set_params)
    get_params = t.getParameters('Parameters')
    try:
        assert pprint.pformat(get_params) == expected
    except AssertionError:
        msg = 'parameters are not equivalent:\n'
        msg+= 'expected:\n'
        msg+= expected+'\n'
        msg+= 'got:\n'
        msg+= pprint.pformat(get_params)
        raise ValueError(msg)

def test_remove():
    # create a node and attach it to another node 
    a = cgns.Node( Name='TheParent')
    b = cgns.Node( Name='TheChild', Parent=a )
    b.remove()
    assert a == ['TheParent', None, [], 'DataArray_t']

def test_findAndRemoveNode():
    # create a node and attach it to another node 
    a = cgns.Node( Name='TheParent')
    b = cgns.Node( Name='TheChild', Value=1, Type='DataArray', Parent=a )

    a.findAndRemoveNode(Name='TheChild', Value=1, Type='DataArray', Depth=1)
    assert a == ['TheParent', None, [], 'DataArray_t']
    
def test_findAndRemoveNodes():
    # create a node and attach it to another node 
    a = cgns.Node( Name='TheParent')
    b = cgns.Node( Name='TheChild', Value=1, Type='DataArray', Parent=a )

    a.findAndRemoveNodes(Name='TheChild', Value=1, Type='DataArray', Depth=1)
    assert a == ['TheParent', None, [], 'DataArray_t']

def test_getPaths():
    parent = cgns.Node( Name='Parent')
    for n in range(2):
        child = cgns.Node( Name=f'Child{n}', Parent=parent)
    for n in range(2):
        cgns.Node( Name=f'GrandChild{n}', Parent=child)
    paths = parent.getPaths()
    assert paths == ['Parent', 'Parent/Child0', 'Parent/Child1', 'Parent/Child1/GrandChild0', 'Parent/Child1/GrandChild1']
