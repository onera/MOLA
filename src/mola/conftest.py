import pytest
import timeit
import warnings

cost_levels = {
    'cost_level_0' : (    0,  0.5),
    'cost_level_1' : (  0.3,  5.0),
    'cost_level_2' : (  3.0, 10.0),
    'cost_level_3' : (  7.0, 30.0),
    'cost_level_4' : ( 20.0,  1e6),
}

def pytest_configure(config):

    config.addinivalue_line(
        "markers", "unit: unit test of an isolated operation, usually fast")

    config.addinivalue_line(
        "markers", "integration: test of a sequence of operations, usually slow")

    config.addinivalue_line(
        "markers", "user_case: application representative test, user-oriented, usually very costly")

    for cost_level, boundaries in cost_levels.items():
        config.addinivalue_line(
            "markers", f"{cost_level}: tests with expected cost {boundaries} sec")

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
            msg = (f'{func.__name__} took {cpu_cost} seconds, which is'
             f' lower than the suggested minimum {cost_levels[marker][0]} for marker "{marker}".')
            warnings.warn(msg)
        
        assert cpu_cost <= cost_levels[marker][1], \
            f'{func.__name__} took {cpu_cost} seconds, which is outside the maximum boundary {cost_levels[marker][1]} for marker "{marker}".'

        return result
    return wrapper
