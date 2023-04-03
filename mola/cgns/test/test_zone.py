import os
import pprint
import numpy as np
import mola.cgns as cgns

def get_cart():
    x, y, z = np.meshgrid( np.linspace(0,1,3),
                           np.linspace(0,1,3),
                           np.linspace(0,1,3), indexing='ij')
    cart = cgns.newZoneFromArrays( 'block', ['x','y','z'],
                                            [ x,  y,  z ])
    return cart

def test_newFields1():
    zone = get_cart()
    f = zone.newFields('field')
    expected_f = np.zeros((3,3,3), dtype=np.float64)
    assert str(f) == str(expected_f)

def test_newFields2():
    zone = get_cart()
    f = zone.newFields('field', Container='FlowSolution#Centers')
    expected_f = np.zeros((2,2,2), dtype=np.float64)
    assert str(f) == str(expected_f)

def test_newFields3():
    zone = get_cart()
    f1, f2 = zone.newFields({'f1':1,'f2':2})
    expected_f1 = np.full((3,3,3), 1, dtype=np.float64)
    assert str(f1) == str(expected_f1)

    expected_f2 = np.full((3,3,3), 2, dtype=np.float64)
    assert str(f2) == str(expected_f2)

def test_boundaries():
    zone = get_cart()
    zone.newFields(['f1','f2'])
    zone.newFields('f3', Container='FlowSolution#Centers')
    boundaries = zone.boundaries()
    
