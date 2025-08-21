import pytest
import numpy as np
from treelab import cgns
from mola.cfd.preprocess.mesh import connect


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_clip_small_rotation_angles():
    t = cgns.Tree()
    perio = cgns.Node(Name='Periodic', Type='Periodic_t', Parent=t)
    rot = cgns.Node(Name='RotationAngle', Value=[1., 1e-13, 1e-11], Parent=perio)
    connect._clip_small_rotation_angles(t)
    assert np.all(t.get('RotationAngle').value() == np.array([1., 0., 1e-11]))
