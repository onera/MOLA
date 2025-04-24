import os
import pytest

from . import dummy_module
import mola.dependency_injector.retriever as test

dummy_backends = ["backend1","backend2"]



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_caller_path():
    this_path = test.get_caller_path()
    expected_path = os.environ["MOLA"].replace("/src","")+"/tests/dependency_injector/user/test_retriever.py"
    assert this_path == expected_path


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("backend", dummy_backends )
def test_get_wrap_module_path(backend):
    path = test.get_wrap_module_path(backend,
        interface_switch=["user", "wrap/<backend>"])
    expected_path = os.environ["MOLA"].replace("/src","")+f"/tests/dependency_injector/wrap/{backend}/test_retriever.py"
    assert path == expected_path

@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("backend", dummy_backends)
def test_get_backend_module_from(backend):
    expected_path = os.environ["MOLA"].replace("/src","")+f"/tests/dependency_injector/wrap/{backend}/dummy_module.py"
    module = test.get_backend_module_from(expected_path)
    assert module is not None

if __name__ == '__main__':
    # test_get_wrap_module_path('backend1')
    test_get_backend_module_from("backend1")