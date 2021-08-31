import sys
import os

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.WorkflowAirfoil as WF

import PolarConfiguration as config

Airfoil = config.AirfoilPath.split(os.path.sep)[-1]


for case in config.JobsQueues:

    if case['NewJob']:
        t, meshParams = WF.buildMesh(Airfoil,
                                     meshParams=case['meshParams'],
                                     save_meshParams=True,
                                     save_mesh=False)
        WF.buildJob(case, config, NProc=meshParams['options']['NProc'])
    
    WF.prepareMainCGNS(t=t, meshParams=meshParams,
                       CoprocessOptions=case['CoprocessOptions'],
                       ImposedWallFields=case['ImposedWallFields'],
                       TransitionZones=case['TransitionZones'],
                       **case['elsAParams'])
    WF.putFilesInComputationDirectory(case)

for case in config.JobsQueues:
    if case['NewJob']:
        WF.launchComputationJob(case, config, submitReserveJob=False)


