'''
Preprocess routine used for:
    1 - Create appropriate symbolic links of restart fields
    2 - Updates the timeout of setup.py

19/05/2021 - L. Bernardos
'''

import sys
import os
import glob

import MOLA.Preprocess as PRE

import setup

def fileExist(*path): return os.path.isfile(os.path.join(*path))

def getDirectoriesNamesOfJob():
    Directories = glob.glob('..'+os.path.sep+'*'+os.path.sep)
    Directories = sorted([d.split(os.path.sep)[-2] for d in Directories])

    return Directories

HasOutput = os.path.isdir('OUTPUT')

if not HasOutput:
    CurrentCaseName = os.getcwd().split(os.path.sep)[-1]
    CasesNames = getDirectoriesNamesOfJob()
    CurrentCaseIndex = CasesNames.index(CurrentCaseName)

    if CurrentCaseIndex == 0:
        raise IOError('Initial case %s does not have OUTPUT restart fields'%CurrentCaseName)

    PreviousCaseName = CasesNames[CurrentCaseIndex - 1]
    InitialCaseName = CasesNames[0]

    if CurrentCaseName.startswith('M') and not PreviousCaseName.startswith('M'):
        RestartCaseName = InitialCaseName
    else:
        RestartCaseName = PreviousCaseName

    if fileExist('..',RestartCaseName,'FAILED'):
        with open('FAILED','w') as f: f.write('previous restart case failed')

    else:
        os.makedirs('OUTPUT')
        RestartLink=os.path.join('..','..',RestartCaseName,'OUTPUT','fields.cgns')
        os.symlink(RestartLink, os.path.join('OUTPUT','fields.cgns') )

if fileExist('NEWJOB_REQUIRED'):
    FailureFiles = glob.glob('core.*') + glob.glob('elsA.x.*') + glob.glob('bk*')
    FailureFiles.append('NEWJOB_REQUIRED')
    for fn in FailureFiles:
        try: os.remove(fn)
        except: pass


ElapsedTime = float(sys.argv[1])
try:
    InitialTO = setup.ReferenceValues['CoprocessOptions']['InitialTimeOutInSeconds']
except KeyError:
    InitialTO = 54000.0
setup.ReferenceValues['CoprocessOptions']['TimeOutInSeconds'] = InitialTO-ElapsedTime
PRE.writeSetupFromModuleObject(setup, setupFilename='setup.py')
