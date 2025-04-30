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

from mola.cfd.postprocess.signals import airplane_coefficients_computer as acc

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_project_load():
    shape = (10,)
    fx = np.full( shape, 1, dtype=np.float64 )
    fy = np.full( shape, 2, dtype=np.float64 )
    fz = np.full( shape, 3, dtype=np.float64 )

    vector = np.array([3, 2, 1],dtype=np.float64)
    vector /= np.linalg.norm(vector)

    result = acc._project_load(fx,fy,fz,vector)
    expected = np.full(shape, vector.dot(np.array([fx[0], fy[0], fz[0]])))

    assert np.allclose(result,expected)

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


@pytest.fixture
def application_context():
    app_ctxt = dict(
        FluxCoef=10.0,
        TorqueCoef=20.0, 
        LiftDirection=np.array([0,1,0]),
        DragDirection=np.array([-1,0,0]),
        SideDirection=np.array([0,0,1]))
    return app_ctxt

