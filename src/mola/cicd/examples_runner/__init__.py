"""Run, monitor, validate, and repatriate marked MOLA examples."""

from .cleanup import clear_examples
from .cli import main
from mola import server as SV
from .postprocess import postprocess_examples
from .repatriate import (
    publish_light_results,
    relocate_examples,
    repatriate_examples,
)
from .state import ExampleCIState
from .status import check_status, query_multi_status_remote, validate_examples
from .submit import (
    detect_execution_mode,
    execute_example,
    parse_job_id_from_output,
    submit_examples,
)

__all__ = [
    "ExampleCIState",
    "check_status",
    "clear_examples",
    "detect_execution_mode",
    "execute_example",
    "main",
    "parse_job_id_from_output",
    "postprocess_examples",
    "publish_light_results",
    "query_multi_status_remote",
    "relocate_examples",
    "repatriate_examples",
    "submit_examples",
    "validate_examples",
    "SV",
]
