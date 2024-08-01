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
FILE_INPUT_WORKLFOW = 'workflow.cgns'
FILE_INPUT_SOLVER = 'main.cgns'

# OUTPUTS
DIRECTORY_OUTPUT = 'OUTPUT'
FILE_OUTPUT_3D = 'fields.cgns' 
FILE_OUTPUT_2D = 'extractions.cgns'
FILE_OUTPUT_1D = 'signals.cgns'

# LOG
DIRECTORY_LOG = 'LOGS'
FILE_STDOUT = 'stdout.log'
FILE_STDERR = 'stderr.log'

# JOB
FILE_JOB = 'job.sh'
FILE_JOB_SEQUENCE = 'job_sequence.sh'
FILE_COMPUTE = 'compute.py'
FILE_COPROCESS = 'coprocess.py'
FILE_COLOG = 'coprocess.log'

# Job end files
FILE_JOB_COMPLETED = 'COMPLETED' 
FILE_JOB_FAILED = 'FAILED' 
FILE_NEWJOB_REQUIRED = 'NEWJOB_REQUIRED' 
FILE_ERROR_PREPARING_WORKFLOW = 'ERROR_PREPARING_WORKFLOW'

# CGNS containers
CONTAINER_WORKLFOW_PARAMETERS = 'WorkflowParameters'
CONTAINER_INITIAL_FIELDS = 'FlowSolution#Init'

DIRECTORY_OVERSET = 'OVERSET'

