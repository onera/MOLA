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

import numpy as np
from treelab import cgns

from mola.cfd.postprocess.signals import airplane_coefficients_computer as acc


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_update_torque_coefficients(application_context):
    shape = 10
    coefs = {
        "CmL":np.zeros(shape, dtype=np.float64),
        "CmD":np.zeros(shape, dtype=np.float64),
        "CmS":np.zeros(shape, dtype=np.float64),
        "CmX":np.zeros(shape, dtype=np.float64),
        "CmY":np.zeros(shape, dtype=np.float64),
        "CmZ":np.zeros(shape, dtype=np.float64),
    }

    tx = np.full(shape, 1, dtype=np.float64)
    ty = np.full(shape, 2, dtype=np.float64)
    tz = np.full(shape, 3, dtype=np.float64)

    acc._update_torque_coefficients(coefs, tx, ty, tz, application_context)

    expected_values = {
        "CmL":  ty[0] * application_context["TorqueCoef"],
        "CmD": -tx[0] * application_context["TorqueCoef"],
        "CmS":  tz[0] * application_context["TorqueCoef"],
        "CmX":  tx[0] * application_context["TorqueCoef"],
        "CmY":  ty[0] * application_context["TorqueCoef"],
        "CmZ":  tz[0] * application_context["TorqueCoef"],
    }

    for key, expected_value in expected_values.items():
        assert np.allclose(coefs[key], np.full(shape, expected_value, dtype=np.float64))

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_update_force_coefficients(application_context):
    shape = 10
    coefs = {
        "CL":np.zeros(shape, dtype=np.float64),
        "CD":np.zeros(shape, dtype=np.float64),
        "CS":np.zeros(shape, dtype=np.float64),
        "CX":np.zeros(shape, dtype=np.float64),
        "CY":np.zeros(shape, dtype=np.float64),
        "CZ":np.zeros(shape, dtype=np.float64),
    }

    fx = np.full(shape, 3, dtype=np.float64)
    fy = np.full(shape, 2, dtype=np.float64)
    fz = np.full(shape, 1, dtype=np.float64)

    acc._update_force_coefficients(coefs, fx, fy, fz, application_context)

    expected_values = {
        "CL":  fy[0] * application_context["FluxCoef"],
        "CD": -fx[0] * application_context["FluxCoef"],
        "CS":  fz[0] * application_context["FluxCoef"],
        "CX":  fx[0] * application_context["FluxCoef"],
        "CY":  fy[0] * application_context["FluxCoef"],
        "CZ":  fz[0] * application_context["FluxCoef"],
    }

    for key, expected_value in expected_values.items():
        assert np.allclose(coefs[key], np.full(shape, expected_value, dtype=np.float64))


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_add_aerodynamic_coefficients_to(zone_with_loads, application_context):
    tree = cgns.Tree(Base=zone_with_loads)
    extraction = dict(Data=tree, Type='Integral')
    acc.add_aerodynamic_coefficients_to(extraction, application_context)

    assert_coefficients_correctly_added_to_extraction_data(extraction, application_context)



# --------------------------------- fixtures --------------------------------- #

@pytest.fixture
def application_context():
    app_ctxt = dict(
        FluxCoef=10.0,
        TorqueCoef=20.0, 
        LiftDirection=np.array([0,1,0]),
        DragDirection=np.array([-1,0,0]),
        SideDirection=np.array([0,0,1]))
    return app_ctxt



@pytest.fixture
def zone_with_loads():
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


# -------------------------------- utilities -------------------------------- #
def assert_coefficients_correctly_added_to_extraction_data(extraction : dict,
        application_context : dict):
    updated_zone = extraction["Data"].zones()[0]

    load_names = ['ForceX', 'ForceY', 'ForceZ', 'TorqueX', 'TorqueY', 'TorqueZ']
    coef_names = ['CL','CD','CS','CX','CY','CZ','CmL','CmD','CmS','CmX','CmY','CmZ']
    v = updated_zone.fields(load_names+coef_names, BehaviorIfNotFound='raise', return_type='dict')

    expected_values = {
        "CL":  v["ForceY"][0] * application_context["FluxCoef"],
        "CD": -v["ForceX"][0] * application_context["FluxCoef"],
        "CS":  v["ForceZ"][0] * application_context["FluxCoef"],
        "CX":  v["ForceX"][0] * application_context["FluxCoef"],
        "CY":  v["ForceY"][0] * application_context["FluxCoef"],
        "CZ":  v["ForceZ"][0] * application_context["FluxCoef"],
       "CmL":  v["TorqueY"][0] * application_context["TorqueCoef"],
       "CmD": -v["TorqueX"][0] * application_context["TorqueCoef"],
       "CmS":  v["TorqueZ"][0] * application_context["TorqueCoef"],
       "CmX":  v["TorqueX"][0] * application_context["TorqueCoef"],
       "CmY":  v["TorqueY"][0] * application_context["TorqueCoef"],
       "CmZ":  v["TorqueZ"][0] * application_context["TorqueCoef"],
    }

    expected_shape = updated_zone.shape()
    for key, expected_value in expected_values.items():
        assert np.allclose(v[key], np.full(expected_shape, expected_value, dtype=np.float64))