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
from treelab import cgns

@pytest.mark.unit
@pytest.mark.elsa
@pytest.mark.cost_level_0
def test_rename_variables_from_turbo_to_mola():
    from mola.cfd.postprocess.tool_interface import turbo

    input_variables = [
                'StagnationPressureRelDim', 'StagnationTemperatureRelDim',
                'EntropyDim',
                'Viscosity_EddyMolecularRatio',
                'VelocitySoundDim', 'StagnationEnthalpyAbsDim',
                'MachNumberAbs', 'MachNumberRel',
                'VelocityXAbsDim', 'VelocityRadiusAbsDim', 'VelocityThetaAbsDim',
                'VelocityMeridianDim', 'VelocityRadiusRelDim', 'VelocityThetaRelDim',
            ]
    expected_output_variables = [
                'PressureStagnationRel', 'TemperatureStagnationRel',
                'Entropy',
                'Viscosity_EddyMolecularRatio',
                'VelocitySound', 'EnthalpyStagnationAbs',
                'MachNumberAbs', 'MachNumberRel',
                'VelocityAbsX', 'VelocityAbsRadius', 'VelocityAbsTheta',
                'VelocityMeridian', 'VelocityRelRadius', 'VelocityRelTheta',
            ]

    tree = cgns.Tree()
    base = cgns.Base(Name=turbo.AVERAGES_0D_BASE, Parent=tree)
    zone = cgns.Zone(Name='Zone', Parent=base)
    zone.setParameters('FlowSolution', ContainerType='FlowSolution_t', **dict((name, 0) for name in input_variables))    
    assert set(tree.get(Type='Zone').getParameters('FlowSolution')) == set(input_variables)

    turbo.rename_variables_from_turbo_to_mola(tree)
    assert set(tree.get(Type='Zone').getParameters('FlowSolution')) == set(expected_output_variables)

