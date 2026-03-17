import pytest
import copy
import numpy as np
from treelab import cgns

try:
    from mola.migration import internals as I
except ModuleNotFoundError:
    pytest.skip(reason="noder not found. Install it using 'python3 -m pip install noder'")

@pytest.fixture
def zone():
    x, y, z = np.meshgrid( np.linspace(0,1,5),
                           np.linspace(0,1,5),
                           np.linspace(0,1,5), 
                           indexing='ij')
    zone = cgns.newZoneFromArrays( 'block', ['x','y','z'],
                                            [ x,  y,  z ])
    field_names = ['ForceX', 'ForceY', 'ForceZ', 'TorqueX', 'TorqueY', 'TorqueZ']
    fx, fy, fz, tx, ty, tz = zone.fields(field_names, BehaviorIfNotFound='create')
    fx[:] = 3
    fy[:] = 2
    fz[:] = 1
    tx[:] = 1
    ty[:] = 2
    tz[:] = 3

    return zone


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_getName(zone):
    name = I.getName(zone)
    assert name == 'block'


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_getType(zone):
    nodetype = I.getType(zone)
    assert nodetype == 'Zone_t'



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_getValue(zone):
    
    zone_value = I.getValue(zone)
    assert zone_value is zone[1]
    assert isinstance(zone_value,np.ndarray)

    zone[2] += [ ["toto", np.array("hello"), [], "toto_t"] ]
    t = I.getNodeFromName(zone,"toto")
    t_value = I.getValue(t)
    assert t_value == 'hello'

    n = I.getNodeFromName(zone,"ZoneType")
    zone_struct_value = I.getValue(n)
    assert zone_struct_value == 'Structured' # BUG

# ----------------------------- research by name ----------------------------- # 
@pytest.mark.unit
@pytest.mark.cost_level_0
def test_getNodeFromName(zone):
    x = I.getNodeFromName(zone,"CoordinateX")
    assert x
    assert x[0] == 'CoordinateX'
    assert x[3] == 'DataArray_t'

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_getNodeFromName1(zone):
    x = I.getNodeFromName1(zone,"GridCoordinates")
    assert x
    assert x[0] == 'GridCoordinates'
    assert x[3] == 'GridCoordinates_t'

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_getNodeFromName2(zone):
    x = I.getNodeFromName2(zone,"CoordinateX")
    assert x
    assert x[0] == 'CoordinateX'
    assert x[3] == 'DataArray_t'

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_getNodeFromName3(zone):
    this_zone = copy.deepcopy(zone)

    # HACK noder does not return in-place view after nodeToPyCGNS, pyCGNSToNode switch
    this_zone[2][1][2][0][2] += [['toto',None,[], "toto_t"]]

    n = I.getNodeFromName3(this_zone,"toto")
    assert n
    assert n[0] == 'toto'
    assert n[3] == 'toto_t'

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_getNodesFromName(zone):

    nodes = I.getNodesFromName(zone,"Coordinate*")
    assert len(nodes) == 3
    assert nodes[0][0] == 'CoordinateX'
    assert nodes[1][0] == 'CoordinateY'
    assert nodes[2][0] == 'CoordinateZ'
    assert nodes[0][3] == 'DataArray_t'
    assert nodes[1][3] == 'DataArray_t'
    assert nodes[2][3] == 'DataArray_t'


# ----------------------------- research by type ----------------------------- # 

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_getNodeFromType(zone):
    x = I.getNodeFromType(zone,"DataArray_t")
    assert x
    assert x[0] == 'CoordinateX'
    assert x[3] == 'DataArray_t'

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_getNodeFromType1(zone):
    n = I.getNodeFromType1(zone,"ZoneType_t")
    assert n
    assert n[0] == 'ZoneType'
    assert n[3] == 'ZoneType_t'

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_getNodeFromType2(zone):
    x = I.getNodeFromType2(zone,"DataArray_t")
    assert x
    assert x[0] == 'CoordinateX'
    assert x[3] == 'DataArray_t'

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_getNodeFromType3(zone):

    this_zone = copy.deepcopy(zone)
    # # HACK noder does not return in-place view after nodeToPyCGNS, pyCGNSToNode switch
    # zone[2][1][2][0][2] += [['toto',None,[], "toto_t"]]
    I.createNode("toto","toto_t",parent=this_zone)
    n = I.getNodeFromType3(this_zone,"toto_t")
    assert n
    assert n[0] == 'toto'
    assert n[3] == 'toto_t'

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_getNodesFromType(zone):

    nodes = I.getNodesFromType(zone,"DataArray_t")
    assert len(nodes) == 9
    for n in nodes:
        n[3] == 'DataArray_t'

# ------------------------------- manipulation ------------------------------- #

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_addChild(zone):
    toto = ['toto',None,[],'toto_t']
    I.addChild(zone, toto)
    zone[2][-1][0] == "toto"
    zone[2][-1][3] == "toto_t"


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_createNode(zone):
    I.createNode("toto","toto_t",parent=zone)

    zone[2][-1][0] == "toto"
    zone[2][-1][3] == "toto_t"
