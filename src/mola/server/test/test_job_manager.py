import pytest
import copy

from mola.workflow.workflow import Workflow
import mola.server.job_manager as JM
from dataclasses import dataclass

def get_fake():
    @dataclass
    class Fake(Workflow):

        BoundaryConditions = [
            dict(Family='INFLOW'),
            dict(Family='OUTFLOW', type='OutflowPressure', Pressure=10),
        ]

        RunManagement = dict(
            RunDirectory = 'test',
            other = 3,
        )
    return Fake()

def test_set_value_on_leaf_1():
    d = dict(x=1, y=dict(z=3))
    JM.set_value_on_leaf(d, ['y', 'z'], 2)
    assert d['y']['z'] == 2

def test_set_value_on_leaf_2():
    l = [dict(x=1), dict(y=2, z=3)]
    JM.set_value_on_leaf(l, ['y=2', 'z'], 5)
    assert l[1]['z'] == 5

def test_set_value_on_leaf_3():
    fake = get_fake()
    JM.set_value_on_leaf(fake, ['RunManagement', 'RunDirectory'], 'other_test')
    assert fake.RunManagement['RunDirectory'] == 'other_test'

def test_set_value_on_leaf_4():
    fake = get_fake()
    JM.set_value_on_leaf(fake, ['BoundaryConditions', 'Family=OUTFLOW', 'Pressure'], 20)
    assert fake.BoundaryConditions[1]['Pressure'] == 20

def test_get_value_on_leaf():
    fake = get_fake()
    value = JM.get_value_on_leaf(fake, ['BoundaryConditions', 'Family=OUTFLOW', 'Pressure'])
    assert value == 10
