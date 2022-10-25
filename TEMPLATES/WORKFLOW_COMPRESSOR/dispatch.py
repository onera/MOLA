import sys
import os

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.WorkflowCompressor as WF
import MOLA.JobManager as JM

import JobsConfiguration as config

DIRECTORY_DISPATCHER = os.path.join('..', '..', 'DISPATCHER')

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
    os.symlink(os.path.join(DIRECTORY_DISPATCHER, case['mesh']), case['mesh'])

    WF.prepareMainCGNS4ElsA(**WorkflowParams)
    JM.getTemplates('Compressor',
            otherWorkflowFiles=['monitor_perfos.py', 'preprocess.py', 'postprocess.py'],
            JobInformation=case['JobInformation'])

    os.chdir(DIRECTORY_DISPATCHER)

for case in config.JobsQueues:
    if case['NewJob']:
        JM.launchComputationJob(case, config, submitReserveJob=False)
