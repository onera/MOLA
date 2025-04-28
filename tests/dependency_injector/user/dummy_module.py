from mola.dependency_injector.backend_function_caller import call_backend_function

def dummy_function(backend):
    return call_backend_function('dummy_function', backend)
