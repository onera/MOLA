import mola.cgns as c
import numpy as np
import os

def test_init1():
    node = c.Node()
    assert node == ['Node', None, [], 'DataArray_t']

def test_init2():
    node = c.Node(Name='MyName', Value=[1,2,3,4], Type='DataArray')
    assert str(node) == "['MyName', array([1, 2, 3, 4], dtype=int32), [], 'DataArray_t']"

def test_init3():
    node = c.Node( ['name', np.array([0,1,2]), [], 'DataArray_t'] )
    assert str(node) == "['name', array([0, 1, 2]), [], 'DataArray_t']"

def test_cast():
    node = c.castNode( ['name', np.array([0,1,2]), [], 'DataArray_t'] )
    assert str(node) == "['name', array([0, 1, 2]), [], 'DataArray_t']"

def test_parent():
    # create a node and attach it to another node 
    a = c.Node( Name='TheParent')
    b = c.Node( Name='TheChild', Parent=a )

    p = b.parent()
    assert p is a
    
    p = a.parent()
    assert p is None

def test_path():
    a = c.Node( Name='TheParent')
    b = c.Node( Name='TheChild', Parent=a )

    path = b.path() 
    assert path == 'TheParent/TheChild'

def test_save():
    n = c.Node( Name='jamon' )
    n.save('test_node_save.cgns', verbose=False)
    os.unlink('test_node_save.cgns')

def test_name():
    n = c.Node( Name='tortilla' )
    assert n.name() == 'tortilla'

def test_value1():
    node = c.Node( Value='jamon' )
    value_str = node.value() # will return a readable str
    assert value_str == 'jamon'

def test_value2():
    node = c.Node( Value='jamon tortilla croquetas' )

    value_str = node.value() # will return a readable list of str
    assert value_str ==  ['jamon', 'tortilla', 'croquetas']

def test_value3():
    # create a multi-dimensional array to be attributed to several nodes
    # it is very important to set order='F' !
    array = np.array( [[0,1,2],
                        [3,4,5],
                        [6,7,8]], order='F' )

    # create our two nodes, and attribute their values to our array
    jamon    = c.Node( Name='jamon', Value=array )
    tortilla = c.Node( Name='tortilla', Value=array )

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

def test_setParameters(save_filename=''):
    t = c.Node(Name='main')
    t.setParameters('Parameters',
        none=None,
        EmptyList=[],
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
                    {'Str':'morcilla','Int':12}]
        )
    if save_filename: t.save(save_filename)

