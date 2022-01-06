import sys
import os

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.WorkflowCompressor as WF
import MOLA.JobManager as JM

import JobsConfiguration as config

for case in config.JobsQueues:

    if case['NewJob']:
        JM.buildJob(case, config, config.NProc, 'job_dispatch.sh')

    WorkflowParams = dict()
    for key, value in case.items():
        if key not in ['ID', 'CASE_LABEL', 'NewJob', 'JobName']:
            WorkflowParams[key] = case[key]

    WF.prepareMainCGNS4ElsA(**WorkflowParams)
    JM.putFilesInComputationDirectory(case)

for case in config.JobsQueues:
    if case['NewJob']:
        JM.launchComputationJob(case, config, submitReserveJob=False)
