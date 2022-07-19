import sys
import os

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.WorkflowAirfoil as WF
import MOLA.JobManager  as JM

import JobsConfiguration as config

Airfoil = config.FILE_GEOMETRY.split(os.path.sep)[-1]
DIRECTORY_DISPATCHER = os.path.join(config.DIRECTORY_WORK, 'DISPATCHER')

for case in config.JobsQueues:

    if case['NewJob']:
        t, meshParams = WF.buildMesh(Airfoil,
                                     meshParams=case['meshParams'],
                                     save_meshParams=True,
                                     save_mesh=False)
        JM.buildJob(case, config)

    caseDir = os.path.join(config.DIRECTORY_WORK, case['JobName'], case['CASE_LABEL'])
    os.makedirs(caseDir, exist_ok=True)
    os.chdir(caseDir)

    WF.prepareMainCGNS4ElsA(t, meshParams=meshParams,
                       CoprocessOptions=case['CoprocessOptions'],
                       ImposedWallFields=case['ImposedWallFields'],
                       TransitionZones=case['TransitionZones'],
                       JobInformation=case['JobInformation'],
                       COPY_TEMPLATES=False,
                       **case['elsAParams'])
    JM.getTemplates('Airfoil',
            otherWorkflowFiles=['monitor_loads.py', 'preprocess.py', 'postprocess.py'],
            JobInformation=case['JobInformation'])

    os.chdir(DIRECTORY_DISPATCHER)

for case in config.JobsQueues:
    if case['NewJob']:
        JM.launchComputationJob(case, config, submitReserveJob=False)
