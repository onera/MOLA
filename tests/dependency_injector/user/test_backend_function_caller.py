import os
import pytest

import mola.dependency_injector.backend_function_caller as test
from . import dummy_module

dummy_backends = ["backend1","backend2"]

@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("backend", dummy_backends)
def test_get_function_at_backend_from(backend):
    function_name = 'dummy_function'
    module_path = os.environ["MOLA"].replace("/src","")+f"/tests/dependency_injector/wrap/{backend}/dummy_module.py"
    module = test.get_backend_module_from(module_path)
    fun = test.get_function_at_backend_from(module, function_name)
    assert fun() == backend


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("backend", dummy_backends)
def test_call_backend_function(backend):
    funct_return = dummy_module.dummy_function(backend)
    assert funct_return == backend



if __name__ == '__main__':
    test_call_backend_function('backend1')
    # test_get_caller_path()