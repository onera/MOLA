import pytest
from treelab import cgns
from mola.cfd.preprocess.mesh import families


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_bc_family_name_nodes_from_patterns_blade(tree_r37_like):
    tree = tree_r37_like
    patterns = ['blade', 'aube', 'propeller', 'rotor', 'stator']
    family_nodes_found_in_tree = families.get_bc_family_name_nodes_from_patterns(tree, patterns)
    assert family_nodes_found_in_tree

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_bc_family_name_nodes_from_patterns_hub(tree_r37_like):
    tree = tree_r37_like
    patterns = ['hub', 'moyeu', 'spinner']
    family_nodes_found_in_tree = families.get_bc_family_name_nodes_from_patterns(tree, patterns)
    assert family_nodes_found_in_tree

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_bc_family_name_nodes_from_patterns_shroud(tree_r37_like):
    tree = tree_r37_like
    patterns = ['shroud', 'carter']
    family_nodes_found_in_tree = families.get_bc_family_name_nodes_from_patterns(tree, patterns)
    assert family_nodes_found_in_tree


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_bc_family_names_from_patterns_blade(tree_r37_like):
    tree = tree_r37_like
    patterns = ['blade', 'aube', 'propeller', 'rotor', 'stator']
    family_names_found_in_tree = families.get_bc_family_names_from_patterns(tree, patterns)
    expected_family_names = ['Rotor_Blade', 'Rotor_OUTFLOW', 'Rotor_INFLOW']
    assert len(family_names_found_in_tree) == len(expected_family_names)
    for name in family_names_found_in_tree:
        assert name in expected_family_names

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_bc_family_names_from_patterns_hub(tree_r37_like):
    tree = tree_r37_like
    patterns = ['hub', 'moyeu', 'spinner']
    family_names_found_in_tree = families.get_bc_family_names_from_patterns(tree, patterns)
    expected_family_names = ['HUB']
    assert len(family_names_found_in_tree) == len(expected_family_names)
    for name in family_names_found_in_tree:
        assert name in expected_family_names


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_bc_family_names_from_patterns_shroud(tree_r37_like):
    tree = tree_r37_like
    patterns = ['shroud', 'carter']
    family_names_found_in_tree = families.get_bc_family_names_from_patterns(tree, patterns)
    expected_family_names = ['SHROUD']
    assert len(family_names_found_in_tree) == len(expected_family_names)
    for name in family_names_found_in_tree:
        assert name in expected_family_names


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_bc_family_nodes_from_patterns_blade(tree_r37_like):
    tree = tree_r37_like
    patterns = ['blade', 'aube', 'propeller', 'rotor', 'stator']
    family_nodes_found_in_tree = families.get_bc_family_nodes_from_patterns(tree, patterns)
    expected_family_names = families.get_bc_family_names_from_patterns(tree, patterns)
    assert len(family_nodes_found_in_tree) == len(expected_family_names)
    for node in family_nodes_found_in_tree:
        assert node.name() in expected_family_names


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_bc_family_nodes_from_patterns_hub(tree_r37_like):
    tree = tree_r37_like
    patterns = ['hub', 'moyeu', 'spinner']
    family_nodes_found_in_tree = families.get_bc_family_nodes_from_patterns(tree, patterns)
    expected_family_names = families.get_bc_family_names_from_patterns(tree, patterns)
    assert len(family_nodes_found_in_tree) == len(expected_family_names)
    for node in family_nodes_found_in_tree:
        assert node.name() in expected_family_names


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_bc_family_nodes_from_patterns_shroud(tree_r37_like):
    tree = tree_r37_like
    patterns = ['shroud', 'carter']
    family_nodes_found_in_tree = families.get_bc_family_nodes_from_patterns(tree, patterns)
    expected_family_names = families.get_bc_family_names_from_patterns(tree, patterns)
    assert len(family_nodes_found_in_tree) == len(expected_family_names)
    for node in family_nodes_found_in_tree:
        assert node.name() in expected_family_names


# --------------------------------- fixtures --------------------------------- #
@pytest.fixture
def tree_r37_like():
    tree = cgns.Tree(Base=[])
    base = tree.bases()[0]

    family_t_names = ['Rotor', 'Rotor_OUTFLOW', 'Rotor_INFLOW', 'Rotor_Blade', 'HUB', 'SHROUD']
    for name in family_t_names:
        cgns.Node(Name=name,Type='Family_t', Parent=base)

    zone_and_names = dict(
        Rotor_Blade_downStream={'bc_9':'HUB',
                                'bc_10':'SHROUD',
                                'bc_13':'Rotor_OUTFLOW'},

        Rotor_Blade_outlet={'bc_9':'HUB',
                            'bc_10':'SHROUD'},

        Rotor_Blade_upStream={'bc_9':'Rotor_INFLOW',
                              'bc_10':'HUB',
                              'bc_11':'SHROUD'},

        Rotor_Blade_inlet={'bc_7':'HUB',
                           'bc_8':'SHROUD'},

        Rotor_Blade_skin={'bc_10':'HUB',
                          'bc_11':'SHROUD',
                          'bc_12':'Rotor_Blade',
                          'bc_13':'Rotor_Blade',},

        Rotor_Blade_up={'bc_9':'HUB',
                        'bc_10':'SHROUD'},

        Rotor_Blade_down={'bc_9':'HUB',
                          'bc_10':'SHROUD'},

    )

    for zone_name, bcname_to_famname in zone_and_names.items():
        zone = cgns.Node(Name=zone_name, Type='Zone_t', Parent=base)
        cgns.Node(Name='FamilyName',Type='FamilyName_t',Value='Rotor', Parent=zone)
        zone_bc = cgns.Node(Name='ZoneBC', Type='ZoneBC_t', Parent=zone)
        for bcname, famname in bcname_to_famname.items():
            bc = cgns.Node(Name=bcname, Type='BC_t', Value='FamilySpecified', Parent=zone_bc)
            cgns.Node(Name='FamilyName', Type='FamilyName_t', Value=famname, Parent=bc)

    return cgns.castNode(tree)
