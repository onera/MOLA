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

from mola.cfd.postprocess.signals import propeller_coefficients_computer as pcc


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_update_force_coefficients(application_context):
    shape = 10

    coefs = {
        'Thrust':np.zeros(shape, dtype=np.float64),
        'CT':np.zeros(shape, dtype=np.float64),
    }

    fx = np.full(shape, 3, dtype=np.float64)
    fy = np.full(shape, 2, dtype=np.float64)
    fz = np.full(shape, 1, dtype=np.float64)

    diameter = 1
    density = 1

    pcc._update_force_coefficients(coefs, fx, fy, fz, application_context,
        diameter, density)

    expected_values = {
        "Thrust": fx[0],
        "CT":     fx[0],
    }

    for key, expected_value in expected_values.items():
        assert np.allclose(coefs[key], np.full(shape, expected_value, dtype=np.float64))


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_update_torque_coefficients(application_context):
    shape = 10
    coefs = {
        'Thrust':np.zeros(shape, dtype=np.float64),
        'Torque':np.zeros(shape, dtype=np.float64),
        'Power':np.zeros(shape, dtype=np.float64),
        'CT':np.zeros(shape, dtype=np.float64),
        'CP':np.zeros(shape, dtype=np.float64),
        'FigureOfMeritHover':np.zeros(shape, dtype=np.float64),
        'PropulsiveEfficiency':np.zeros(shape, dtype=np.float64),
    }


    # required to update forces, since they are used in torque context
    fx = np.full(shape, 3, dtype=np.float64)
    fy = np.full(shape, 2, dtype=np.float64)
    fz = np.full(shape, 1, dtype=np.float64)
    diameter = 1.0
    density = 1.0
    axial_velocity = 1.0
    pcc._update_force_coefficients(coefs, fx, fy, fz, application_context,
        diameter, density)

    tx = np.full(shape, 1, dtype=np.float64)
    ty = np.full(shape, 2, dtype=np.float64)
    tz = np.full(shape, 3, dtype=np.float64)

    pcc._update_torque_coefficients(coefs, tx, ty, tz, application_context,
        diameter, density, axial_velocity)

    rpm = application_context['ShaftRotationSpeed']

    thrust = coefs['Thrust']
    ct = coefs['CT']
    torque = -tx[0]
    power = torque * rpm * (np.pi/30)
    cp = power / (density * (rpm/60)**3 * diameter**5)
    fm = np.sqrt(2.0/np.pi) * np.sign(ct) * np.abs(ct)**1.5 / cp
    eta = axial_velocity * thrust / power

    expected_values = {
        "Torque":  torque,
        "Power": power,
        "CT":  ct,
        "CP":  cp,
        "FigureOfMeritHover":  fm,
        "PropulsiveEfficiency":  eta,
    }

    for key, expected_value in expected_values.items():
        assert np.allclose(coefs[key], np.full(shape, expected_value, dtype=np.float64)), key

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_add_aerodynamic_coefficients_to(zone_with_loads, application_context):
    tree = cgns.Tree(Base=zone_with_loads)
    extraction = dict(Data=tree, Type='Integral')
    diameter = 1.0
    density = 1.0
    axial_velocity = 1.0
    pcc.add_aerodynamic_coefficients_to(extraction, application_context, diameter, density, axial_velocity)

    assert_coefficients_correctly_added_to_extraction_data(extraction, application_context)



# --------------------------------- fixtures --------------------------------- #

@pytest.fixture
def application_context():
    app_ctxt = dict(
        ShaftAxis=[1.0,0.0,0.0],
        ShaftRotationSpeedUnit='rpm',
        ShaftRotationSpeed=60.0,
        NormalizationCoefficient=dict(
            BLADE=dict(FluxCoef=1.0)
        )
    )

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
    coef_names = ['Thrust', 'Torque', 'Power', 'CT', 'CP', 'FigureOfMeritHover', 'PropulsiveEfficiency']
    v = updated_zone.fields(load_names+coef_names, BehaviorIfNotFound='raise', return_type='dict')

    fx, fy, fz = 3, 2, 1
    tx, ty, tz = 1, 2, 3

    diameter = 1.0
    density = 1.0
    axial_velocity = 1.0
    rpm = application_context['ShaftRotationSpeed']

    thrust = fx
    ct = fx
    torque = -tx
    power = torque * rpm * (np.pi/30)
    cp = power / (density * (rpm/60)**3 * diameter**5)
    fm = np.sqrt(2.0/np.pi) * np.sign(ct) * np.abs(ct)**1.5 / cp
    eta = axial_velocity * thrust / power

    expected_values = {
        "Thrust":  thrust,
        "CT":  ct,
        "Torque": torque,
        "Power":  power,
        "CP":  cp,
        "FigureOfMeritHover":  fm,
        "PropulsiveEfficiency":  eta,
    }

    expected_shape = updated_zone.shape()
    for key, expected_value in expected_values.items():
        assert np.allclose(v[key], np.full(expected_shape, expected_value, dtype=np.float64))