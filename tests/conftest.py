import os
import sys
import importlib.abc
import importlib.util
from pathlib import Path
import timeit
import warnings

# Make the source checkout directly testable without installing MOLA or
# sourcing the Linux environment scripts.  These defaults are deliberately
# test-only and can be overridden by the caller's environment.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))
os.environ.setdefault("MOLA", str(SOURCE_ROOT))
os.environ.setdefault("MOLA_SOLVER", "none")
os.environ.setdefault("MOLA_NETWORK", "template")

import pytest


OPTIONAL_DEPENDENCY_ROOTS = {
    # MPI runtime
    'mpi4py',
    # Cassiopee and modules built on top of it
    'Converter',
    'Connector',
    'Transform',
    'Generator',
    'Intersector',
    'Geom',
    'Post',
    'Distributor2',
    # Maia and solver-specific Python packages
    'maia',
    'Fast',
    'FastC',
    'FastS',
    'elsA',
    'elsAxdt',
    'elsA_user',
    'maia4elsA',
    'miles',
    'sonics',
}


class _OptionalDependencyBlocker(importlib.abc.MetaPathFinder,
                                 importlib.abc.Loader):
    """Turn unavailable optional imports into pytest skips during collection.

    Several MOLA modules import MPI/Cassiopee at module import time.  A
    normal ``pytest_collection_modifyitems`` skip is too late because pytest
    must import those modules before it can inspect their markers.
    """

    def __init__(self, missing_roots):
        self.missing_roots = set(missing_roots)

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.', 1)[0]
        if root in self.missing_roots:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        pytest.skip(
            f"optional dependency '{module.__name__.split('.', 1)[0]}' is unavailable",
            allow_module_level=True,
        )


def unavailable_optional_dependency_roots():
    if sys.platform == 'win32':
        return set(OPTIONAL_DEPENDENCY_ROOTS)

    unavailable = set()
    for root in OPTIONAL_DEPENDENCY_ROOTS:
        try:
            if importlib.util.find_spec(root) is None:
                unavailable.add(root)
        except (ImportError, ModuleNotFoundError, ValueError):
            unavailable.add(root)
    return unavailable


UNAVAILABLE_OPTIONAL_DEPENDENCY_ROOTS = unavailable_optional_dependency_roots()


def install_optional_dependency_blocker():
    missing_roots = UNAVAILABLE_OPTIONAL_DEPENDENCY_ROOTS
    if missing_roots and not any(
            isinstance(finder, _OptionalDependencyBlocker)
            for finder in sys.meta_path):
        sys.meta_path.insert(0, _OptionalDependencyBlocker(missing_roots))


install_optional_dependency_blocker()

# useful example if we need to pass option to pytest:
#   https://stackoverflow.com/questions/47559524/pytest-how-to-skip-tests-unless-you-declare-an-option-flag

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
        
    config.addinivalue_line(
        "markers", "network_onera: test available on ONERA machines only")

    config.addinivalue_line(
        "markers", "mpi: test shall be run in mpi")


def pytest_collection_modifyitems(config, items):

    valid_markers = {'unit', 'integration', 'user_case'}
    not_tagged_tests = []

    for item in items:
        skip_if_not_on_onera_network(item)
        skip_if_solver_not_compatible_with_env(item)

        callspec = getattr(item, 'callspec', None)
        backend = callspec.params.get('backend') if callspec else None
        if backend in UNAVAILABLE_OPTIONAL_DEPENDENCY_ROOTS:
            item.add_marker(pytest.mark.skip(
                reason=f"optional backend '{backend}' is unavailable"))

        if sys.platform == 'win32':
            windows_incompatible_markers = {
                'mpi', 'elsa', 'fast', 'sonics', 'network_onera'
            }
            if any(marker in item.keywords for marker in windows_incompatible_markers):
                item.add_marker(pytest.mark.skip(
                    reason='MPI and proprietary solver tests are unavailable on Windows'))
        elif 'mpi' in item.keywords and 'mpi4py' in UNAVAILABLE_OPTIONAL_DEPENDENCY_ROOTS:
            item.add_marker(pytest.mark.skip(reason='mpi4py is unavailable'))

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
    if ("network_onera" in item.keywords) and (os.getenv("MOLA_NETWORK") != 'onera'):
         item.add_marker(skip_onera)

def skip_if_solver_not_compatible_with_env(item):
    solvers = ['elsa', 'sonics', 'fast', 'coda']
    current_solver = os.getenv('MOLA_SOLVER')
    skip_solver = pytest.mark.skip(reason=f"test not available in the current environment ({current_solver})")
    is_marked_with_current_solver = current_solver in item.keywords
    is_marked_with_another_solver = any([solver in item.keywords for solver in solvers if solver != current_solver])
    if not is_marked_with_current_solver and is_marked_with_another_solver:
         item.add_marker(skip_solver)
