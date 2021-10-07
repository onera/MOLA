import sys
import os

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.WorkflowAirfoil as WF
import MOLA.JobManager  as JM

import PolarConfiguration as config

Airfoil = config.GeomPath.split(os.path.sep)[-1]
jobTemplate = '/tmp_user/sator/lbernard/MOLA/Dev/TEMPLATES/WORKFLOW_AIRFOIL/job_{}.sh'.format(config.machine)


for case in config.JobsQueues:

    if case['NewJob']:
        t, meshParams = WF.buildMesh(Airfoil,
                                     meshParams=case['meshParams'],
                                     save_meshParams=True,
                                     save_mesh=False)
        JM.buildJob(case, config, meshParams['options']['NProc'], jobTemplate)

    WF.prepareMainCGNS(t=t, meshParams=meshParams,
                       CoprocessOptions=case['CoprocessOptions'],
                       ImposedWallFields=case['ImposedWallFields'],
                       TransitionZones=case['TransitionZones'],
                       **case['elsAParams'])
    JM.putFilesInComputationDirectory(case)

for case in config.JobsQueues:
    if case['NewJob']:
        JM.launchComputationJob(case, config, submitReserveJob=False)
