import pytest
from pathlib import Path

from . import dummy_module
import mola.dependency_injector.retriever as test

dummy_backends = ["backend1","backend2"]



@pytest.mark.unit
def test_get_caller_path():
    this_path = test.get_caller_path()
    assert Path(this_path) == Path(__file__).resolve()


@pytest.mark.unit
@pytest.mark.parametrize("backend", dummy_backends )
def test_get_wrap_module_path(backend):
    path = test.get_wrap_module_path(backend,
        interface_switch=["/user/", "/wrap/<backend>/"])
    expected_path = Path(__file__).resolve().parents[1] / "wrap" / backend / "test_retriever.py"
    assert Path(path) == expected_path

@pytest.mark.unit
@pytest.mark.parametrize("backend", dummy_backends)
def test_get_backend_module_from(backend):
    expected_path = Path(__file__).resolve().parents[1] / "wrap" / backend / "dummy_module.py"
    module = test.get_backend_module_from(expected_path)
    assert module is not None

if __name__ == '__main__':
    # test_get_wrap_module_path('backend1')
    test_get_backend_module_from("backend1")
