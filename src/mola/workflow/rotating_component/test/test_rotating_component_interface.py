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

from mola.workflow.rotating_component.workflow import WorkflowRotatingComponentInterface

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_HubRotationIntervals():

    class FakeInterface():

        def __init__(self, HubRotationIntervals=None):
            self.ApplicationContext = dict()
            if HubRotationIntervals is not None:
                self.ApplicationContext['HubRotationIntervals'] = HubRotationIntervals

    fake = FakeInterface()
    with pytest.raises(KeyError):
        WorkflowRotatingComponentInterface.set_HubRotationIntervals(fake)

    fake = FakeInterface([])
    WorkflowRotatingComponentInterface.set_HubRotationIntervals(fake)
    assert fake.ApplicationContext['HubRotationIntervals'] == []

    fake = FakeInterface([dict(xmin=1., xmax=2.)])
    WorkflowRotatingComponentInterface.set_HubRotationIntervals(fake)
    assert fake.ApplicationContext['HubRotationIntervals'] == [dict(xmin=1., xmax=2.)]

    fake = FakeInterface([(1., 2.)])
    WorkflowRotatingComponentInterface.set_HubRotationIntervals(fake)
    assert fake.ApplicationContext['HubRotationIntervals'] == [dict(xmin=1., xmax=2.)]

    fake = FakeInterface([dict(xmin=1.)])
    WorkflowRotatingComponentInterface.set_HubRotationIntervals(fake)
    assert fake.ApplicationContext['HubRotationIntervals'] == [dict(xmin=1., xmax=1e20)]

    fake = FakeInterface([dict(xmax=2.)])
    WorkflowRotatingComponentInterface.set_HubRotationIntervals(fake)
    assert fake.ApplicationContext['HubRotationIntervals'] == [dict(xmin=-1e20, xmax=2.)]

  