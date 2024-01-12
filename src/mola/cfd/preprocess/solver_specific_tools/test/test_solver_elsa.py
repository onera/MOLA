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

from mola.cfd.preprocess.solver_specific_tools import solver_elsa

CGNS2ElsaDict = dict(
        PressureStagnation       = 'stagnation_pressure',
        EnthalpyStagnation       = 'stagnation_enthalpy',
        TemperatureStagnation    = 'stagnation_temperature',
        Pressure                 = 'pressure',
        MassFlow                 = 'globalmassflow',
        SurfacicMassFlow         = 'surf_massflow',
        VelocityUnitVectorX      = 'txv',
        VelocityUnitVectorY      = 'tyv',
        VelocityUnitVectorZ      = 'tzv',
        TurbulentSANuTilde       = 'inj_tur1',
        TurbulentEnergyKinetic   = 'inj_tur1',
        TurbulentDissipationRate = 'inj_tur2',
        TurbulentDissipation     = 'inj_tur2',
        TurbulentLengthScale     = 'inj_tur2',
        
        BoundaryLayer            = 'bl_quantities_2d bl_quantities_3d bl_ue',
        NormalVector             = 'normalvector',
        Friction                 = 'frictionvector', 
        yPlus                    = 'yplusmeshsize',
        MomentumFlux             = 'flux_rou flux_rov flux_row',
        TorqueFlux               = 'torque_rou torque_rov torque_row',
    )

RSM_CGNS2ElsaDict = dict(
        TurbulentDissipationRate = 'inj_tur7',
        VelocityCorrelationXX    = 'inj_tur1',
        VelocityCorrelationXY    = 'inj_tur2', 
        VelocityCorrelationXZ    = 'inj_tur3',
        VelocityCorrelationYY    = 'inj_tur4', 
        VelocityCorrelationYZ    = 'inj_tur5', 
        VelocityCorrelationZZ    = 'inj_tur6',
    )

def test_translate_to_elsa_dict():
    d = dict((key, 0) for key in CGNS2ElsaDict)
    res = solver_elsa.translate_to_elsa(d)
    assert res == dict((value, 0) for value in CGNS2ElsaDict.values())

def test_translate_to_elsa_dict_rsm():
    d = dict((key, 0) for key in RSM_CGNS2ElsaDict)
    res = solver_elsa.translate_to_elsa(d)

    assert res == dict((value, 0) for value in RSM_CGNS2ElsaDict.values())

def test_translate_to_elsa_list():
    res = solver_elsa.translate_to_elsa(list(CGNS2ElsaDict))
    assert res == list(CGNS2ElsaDict.values())

def test_translate_to_elsa_str():
    for cgns_name, elsa_name in CGNS2ElsaDict.items():
        res = solver_elsa.translate_to_elsa(cgns_name)
        assert res == elsa_name

def test_translate_to_elsa_error():
    for var in [1, 1., None, True, False]:
        try:
            res = solver_elsa.translate_to_elsa(var)
        except TypeError as e:
            assert e.args[0] == 'Variables must be of type dict, list or string'
        else:
            raise AssertionError(f'translate_to_elsa should raise an TypeError if the argument is of type {type(var)}')
