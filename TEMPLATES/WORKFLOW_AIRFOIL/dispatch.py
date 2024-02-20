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

import MOLA.WorkflowAirfoil as WF
import MOLA.JobManager  as JM

import JobsConfiguration as config

Airfoil = config.FILE_GEOMETRY.split(os.path.sep)[-1]
DIRECTORY_DISPATCHER = os.path.join(config.DIRECTORY_WORK, 'DISPATCHER')

for case in config.JobsQueues:

    if case['NewJob']:
        if case['CASE_LABEL'].startswith('M'):
            t, meshParams = WF.buildMesh(Airfoil,
                                         meshParams=case['meshParams'],
                                         save_meshParams=True,
                                         save_mesh=False)
            JM.buildJob(case, config, jobTemplate='job_template.sh', JobFile = 'jobM.sh', routineTemplate = 'routineM.sh')
        else:
            t, meshParams = WF.buildMesh(Airfoil,
                                         meshParams=case['meshParams'],
                                         save_meshParams=True,
                                         save_mesh=False)
            JM.buildJob(case, config, jobTemplate='job_template.sh', JobFile = 'jobP.sh', routineTemplate = 'routineP.sh')

    caseDir = os.path.join(config.DIRECTORY_WORK, case['JobName'], case['CASE_LABEL'])
    os.makedirs(caseDir, exist_ok=True)
    os.chdir(caseDir)

    WF.prepareMainCGNS4ElsA(t, meshParams=meshParams,
                       CoprocessOptions=case['CoprocessOptions'],
                       ImposedWallFields=case['ImposedWallFields'],
                       TransitionZones=case['TransitionZones'],
                       JobInformation=case['JobInformation'],
                       **case['elsAParams'])

    os.chdir(DIRECTORY_DISPATCHER)

for case in config.JobsQueues:
    if case['NewJob']:
        if case['CASE_LABEL'].startswith('M'):
            JM.launchComputationJob(case, config, JobFilename='jobM.sh', submitReserveJob=False)
        else:
            JM.launchComputationJob(case, config, JobFilename='jobP.sh', submitReserveJob=False)
