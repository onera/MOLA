import pytest
import os
import timeit
import warnings
from mola import server as SV

# useful example if we need to pass option to pytest:
#   https://stackoverflow.com/questions/47559524/pytest-how-to-skip-tests-unless-you-declare-an-option-flag


cost_levels = {
    'cost_level_0' : (    0,  0.5),
    'cost_level_1' : (  0.3,  5.0),
    'cost_level_2' : (  3.0, 10.0),
    'cost_level_3' : (  7.0, 30.0),
    'cost_level_4' : ( 20.0,  1e6),
}

def pytest_configure(config):

    config.addinivalue_line(
        "markers", "elsa: test is relevant only for the solver elsa")

    config.addinivalue_line(
        "markers", "fast: test is relevant only for the solver fast")

    config.addinivalue_line(
        "markers", "sonics: test is relevant only for the solver sonics")

    config.addinivalue_line(
        "markers", "unit: unit test of an isolated operation, usually fast")

    config.addinivalue_line(
        "markers", "integration: test of a sequence of operations, usually slow")

    config.addinivalue_line(
        "markers", "user_case: application representative test, user-oriented, usually very costly")

    config.addinivalue_line(
        "markers", "restricted_user_case: like user_case, but involving not open input data, cannot be shared publicly")

    for cost_level, boundaries in cost_levels.items():
        config.addinivalue_line(
            "markers", f"{cost_level}: tests with expected cost {boundaries} sec")
        
    config.addinivalue_line(
        "markers", "network_onera: test available on ONERA machines only")

    config.addinivalue_line(
        "markers", "mpi: test shall be run in mpi")



def get_cost_marker(marker_container):
    for marker in marker_container:
        if marker in cost_levels:
            return marker


def pytest_runtest_call(item):
    cost_marker = get_cost_marker(item.keywords)
    if cost_marker:
        item.obj = check_cost(item.obj, cost_marker)

def check_cost(func, marker):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        cpu_cost = end_time - start_time
        if cpu_cost < cost_levels[marker][0]:
            msg = (f'{func.__name__} took {cpu_cost} seconds, which is lower than '
                   f'the suggested minimum {cost_levels[marker][0]} for marker "{marker}".')
            warnings.warn(msg)
        
        if cpu_cost > cost_levels[marker][1]:
            msg = (f'{func.__name__} took {cpu_cost} seconds, which is outside the '
                   f'maximum boundary {cost_levels[marker][1]} for marker "{marker}".')
            warnings.warn(msg)

        return result
    return wrapper

def pytest_collection_modifyitems(config, items):

    valid_markers = {'unit', 'integration', 'user_case'}
    not_tagged_tests = []

    for item in items:
        skip_if_not_on_onera_network(item)
        skip_if_solver_not_compatible_with_env(item)

        markers = {marker.name for marker in item.iter_markers()}
        if not markers.intersection(valid_markers):
            not_tagged_tests.append(item.nodeid)

    if len(not_tagged_tests) > 0:
        new_line_double_space = os.linesep + '  '
        raise pytest.UsageError(
            f"Each test must be tagged with one of the following markers: {', '.join(valid_markers)}."
            f"The following tests are missing a required marker:{os.linesep}" 
            f"  {new_line_double_space.join(not_tagged_tests)}"
        )
    
    # Ignore tests with marker "user_case" except it is explicitely asked with pytest -m user_case
    if not config.option.markexpr:
        for item in items:
            if "user_case" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Use pytest -m user_case to force test"))

def skip_if_not_on_onera_network(item):
    skip_onera = pytest.mark.skip(reason="test available on ONERA machines only")
    if ("network_onera" in item.keywords) and (SV.get_network() != 'onera'):
         item.add_marker(skip_onera)

def skip_if_solver_not_compatible_with_env(item):
    solvers = ['elsa', 'sonics', 'fast', 'coda']
    current_solver = os.getenv('MOLA_SOLVER')
    skip_solver = pytest.mark.skip(reason=f"test not available in the current environment ({current_solver})")
    is_marked_with_current_solver = current_solver in item.keywords
    is_marked_with_another_solver = any([solver in item.keywords for solver in solvers if solver != current_solver])
    if not is_marked_with_current_solver and is_marked_with_another_solver:
         item.add_marker(skip_solver)
