import sys
import os

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.WorkflowAirfoil as WF
import MOLA.JobManager  as JM

import JobsConfiguration as config

Airfoil = config.FILE_GEOMETRY.split(os.path.sep)[-1]

for case in config.JobsQueues:

    if case['NewJob']:
        t, meshParams = WF.buildMesh(Airfoil,
                                     meshParams=case['meshParams'],
                                     save_meshParams=True,
                                     save_mesh=False)
        JM.buildJob(case, config, meshParams['options']['NProc'], 'job_template.sh')

    WF.prepareMainCGNS4ElsA(t, meshParams=meshParams,
                       CoprocessOptions=case['CoprocessOptions'],
                       ImposedWallFields=case['ImposedWallFields'],
                       TransitionZones=case['TransitionZones'],
                       **case['elsAParams'])
    JM.putFilesInComputationDirectory(case)

for case in config.JobsQueues:
    if case['NewJob']:
        JM.launchComputationJob(case, config, submitReserveJob=False)
