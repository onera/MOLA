import pytest

@pytest.fixture
def dummy_tree():
    import Generator.PyTree as G
    import Converter.PyTree as C

    zone = G.cart((0,0,0),(1,1,1),(11,11,11))
    tree = C.newPyTree(['Base',zone])
    return tree
