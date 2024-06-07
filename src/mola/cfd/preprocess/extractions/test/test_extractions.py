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
from mola.cfd.preprocess.extractions import extractions

class FakeWorkflow():

    def __init__(self):
        self.Extractions = [
            dict(Type='BC', Source='BCWallViscous', Fields=['Pressure']),
            dict(Type='BC', Source='BCWallInviscid'),
            dict(Type='3D', Fields=['Density', 'Momentum', 'Energy'])
        ]
        self.Flow = dict(
            ReferenceState = dict(
                Density = 1.2,
                Momentum = 10.,
                Energy = 5.,
            )
        )


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_process_extractions_2d():
    workflow = FakeWorkflow()
    extractions.process_extractions_2d(workflow)

    assert workflow.Extractions == [
            {'Type': 'BC', 'Source': 'BCWallViscous', 'Fields': ['Pressure']},
            {'Type': 'BC', 'Source': 'BCWallInviscid', 'Fields': []},
            {'Type': '3D', 'Fields': ['Density', 'Momentum', 'Energy']}]
    
