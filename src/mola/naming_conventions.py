#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

# INPUTS
FILE_INPUT_WORKFLOW = 'workflow.cgns'
FILE_INPUT_SOLVER = 'main.cgns'
FILE_WORKFLOW_MANAGER = 'workflow_manager.yaml'
DIRECTORY_OVERSET = 'OVERSET'

# OUTPUTS
DIRECTORY_OUTPUT = 'OUTPUT'
FILE_OUTPUT_3D = 'fields.cgns' 
FILE_OUTPUT_2D = 'extractions.cgns'
FILE_OUTPUT_1D = 'signals.cgns'

# LOG
DIRECTORY_LOG = 'LOGS'
FILE_STDOUT = 'stdout.log'
FILE_STDERR = 'stderr.log'
FILE_STDOUT_PRE = 'stdout_pre.log'
FILE_STDERR_PRE = 'stderr_pre.log'
CGNS_NODE_EXTRACTION_LOG = 'MOLA:Extraction-Log'
FILE_LOG_REMOTE_DIRECTORY = '.mola_run_directory'

# CI/CD state files.
DIRECTORY_CI_STATE = '.release/example_states'
FILE_CI_STATE_SUFFIX = '.ci_state.yaml'

# JOB
FILE_JOB = 'job.sh'
FILE_JOB_SEQUENCE = 'job_sequence.sh'
FILE_JOB_PREPARE = 'job_prepare.sh'
FILE_COMPUTE = 'compute.py'
FILE_COPROCESS = 'coprocess.py'
FILE_COLOG = 'coprocess.log'
FILE_COLOG_PATTERN = 'coprocess-*.log'
FILE_GATHER_RESULTS = 'gather_results.py'

# Job end files
FILE_JOB_COMPLETED = 'COMPLETED'
FILE_JOB_FAILED = 'FAILED'
FILE_NEWJOB_REQUIRED = 'NEWJOB_REQUIRED'
FILE_RELAUNCH_JOB = 'RELAUNCH_JOB'
FILE_ERROR_PREPARING_WORKFLOW = 'ERROR_PREPARING_WORKFLOW'
# Transient phase marker written while workflow.cgns is being consumed.
# It is intentionally not part of STATUS_FILES: it is not a terminal job
# status and must not be removed by compute.remove_status_files().
FILE_PREPROCESSING = 'PREPROCESSING'
STATUS_FILES = [FILE_JOB_COMPLETED,
                FILE_JOB_FAILED,
                FILE_NEWJOB_REQUIRED,
                FILE_RELAUNCH_JOB,
                FILE_ERROR_PREPARING_WORKFLOW]

# Per-case statuses used by status/reporting APIs.
STATUS_TO_PREPARE = 'TO_PREPARE'
STATUS_TO_LAUNCH = 'TO_LAUNCH'
STATUS_PREPROCESS = 'PREPROCESS'
STATUS_READY = 'READY'
STATUS_RUNNING = 'RUNNING'
STATUS_PENDING = 'PENDING'

# Submission events emitted by workflow processes and consumed by CI tooling.
SUBMISSION_EVENT_PREFIX = 'MOLA_SUBMISSION: '

# Repatriation patterns shared by the command-line and Python APIs.
REPATRIATION_IMAGE_PATTERNS = ('**.png', '**.pdf', '**.svg')
REPATRIATION_LOG_PATTERNS = ('*.log', f'{DIRECTORY_LOG}**')
REPATRIATION_SIGNALS_PATTERNS = (
    f'{DIRECTORY_OUTPUT}/',
    f'{DIRECTORY_OUTPUT}/{FILE_OUTPUT_1D}',
)
REPATRIATION_EXTRACTIONS_PATTERNS = (
    f'{DIRECTORY_OUTPUT}/',
    f'{DIRECTORY_OUTPUT}/{FILE_OUTPUT_2D}',
)
REPATRIATION_OUTPUT_PATTERNS = (f'{DIRECTORY_OUTPUT}**',)
REPATRIATION_LIGHT_PATTERNS = (
    '**/',
    f'{DIRECTORY_OUTPUT}/',
    f'{DIRECTORY_OUTPUT}/{FILE_OUTPUT_1D}',
    f'**/{DIRECTORY_OUTPUT}/',
    f'**/{DIRECTORY_OUTPUT}/{FILE_OUTPUT_1D}',
    f'{DIRECTORY_LOG}/',
    f'{DIRECTORY_LOG}/{FILE_COLOG_PATTERN}',
    f'**/{DIRECTORY_LOG}/',
    f'**/{DIRECTORY_LOG}/{FILE_COLOG_PATTERN}',
    *REPATRIATION_IMAGE_PATTERNS,
)

# Aggregate status groups used by CI/CD status reporting.
FAILED_STATUSES = frozenset([
    FILE_JOB_FAILED,
    FILE_ERROR_PREPARING_WORKFLOW,
    'FAILED',
    'ERROR',
])
RUNNING_STATUSES = frozenset([
    STATUS_RUNNING,
    STATUS_PENDING,
    'QUEUED',
    FILE_RELAUNCH_JOB,
    FILE_NEWJOB_REQUIRED,
    STATUS_PREPROCESS,
    STATUS_READY,
    STATUS_TO_PREPARE,
    STATUS_TO_LAUNCH,
])

# CGNS containers
CONTAINER_WORKFLOW_PARAMETERS = 'WorkflowParameters'
CONTAINER_INITIAL_FIELDS = 'FlowSolution#Init'
CONTAINER_OUTPUT_FIELDS_AT_VERTEX = 'FlowSolution#Output@Vertex'
CONTAINER_OUTPUT_FIELDS_AT_CENTER = 'FlowSolution#Output@Center'
