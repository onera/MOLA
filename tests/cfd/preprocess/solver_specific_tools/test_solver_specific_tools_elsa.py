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

import pytest
from mola.cfd.preprocess.solver_specific_tools import solver_elsa

pytestmark = pytest.mark.elsa

RSM_CGNS2ElsaDict = dict(
        TurbulentDissipationRate = 'inj_tur7',
        VelocityCorrelationXX    = 'inj_tur1',
        VelocityCorrelationXY    = 'inj_tur2', 
        VelocityCorrelationXZ    = 'inj_tur3',
        VelocityCorrelationYY    = 'inj_tur4', 
        VelocityCorrelationYZ    = 'inj_tur5', 
        VelocityCorrelationZZ    = 'inj_tur6',
    )

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_translate_to_elsa_dict():
    d = dict((key, 0) for key in solver_elsa.cgns_to_elsa_bc_field_name)
    res = solver_elsa.translate_to_elsa(d)
    assert set(res) == set(list(solver_elsa.cgns_to_elsa_bc_field_name.values())+['inj_tur7'])

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_translate_to_elsa_dict_rsm():
    d = dict((key, 0) for key in RSM_CGNS2ElsaDict)
    res = solver_elsa.translate_to_elsa(d)

    assert res == dict((value, 0) for value in RSM_CGNS2ElsaDict.values())

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_translate_to_elsa_list():
    input_list = list(solver_elsa.cgns_to_elsa_bc_field_name)
    res = solver_elsa.translate_to_elsa(input_list)
    expected_list = list(solver_elsa.cgns_to_elsa_bc_field_name.values())
    # replace the first inj_tur2 by inj_tur7
    i = input_list.index('TurbulentDissipationRate')
    expected_list[i] = 'inj_tur7'
    assert res == expected_list

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_translate_to_elsa_str():
    for cgns_name, elsa_name in solver_elsa.cgns_to_elsa_bc_field_name.items():
        res = solver_elsa.translate_to_elsa(cgns_name)
        assert res == elsa_name

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_translate_to_elsa_error():
    for var in [1, 1., None, True, False]:
        try:
            res = solver_elsa.translate_to_elsa(var)
        except TypeError as e:
            assert e.args[0] == 'Variables must be of type dict, list or string'
        else:
            raise AssertionError(f'translate_to_elsa should raise an TypeError if the argument is of type {type(var)}')
