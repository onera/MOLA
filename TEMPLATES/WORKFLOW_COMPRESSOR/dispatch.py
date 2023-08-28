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

import sys
import os

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.WorkflowCompressor as WF
import MOLA.JobManager as JM

import JobsConfiguration as config

DIRECTORY_DISPATCHER = os.path.join(config.DIRECTORY_WORK, 'DISPATCHER')

for case in config.JobsQueues:

    if case['NewJob']:
        JM.buildJob(case, config)

    WorkflowParams = dict()
    for key, value in case.items():
        if key not in ['ID', 'CASE_LABEL', 'NewJob', 'JobName']:
            WorkflowParams[key] = case[key]
    WorkflowParams['COPY_TEMPLATES'] = False

    caseDir = os.path.join(config.DIRECTORY_WORK, case['JobName'], case['CASE_LABEL'])
    os.makedirs(caseDir, exist_ok=True)
    os.chdir(caseDir)

    WF.prepareMainCGNS4ElsA(**WorkflowParams)
    JM.getTemplates('Compressor',
            otherWorkflowFiles=['monitor_perfos.py', 'preprocess.py', 'postprocess.py'],
            JobInformation=case['JobInformation'])

    os.chdir(DIRECTORY_DISPATCHER)

for case in config.JobsQueues:
    if case['NewJob']:
        JM.launchComputationJob(case, config, submitReserveJob=False)
