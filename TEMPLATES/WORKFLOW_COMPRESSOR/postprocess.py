'''
postprocess routine used for:
    1 - determine if computation was OK or not
    2 - override fields.cgns with tmp-fields.cgns
    3 - save relevant files into LOG directory
    4 - switch to robust or fail-safe mode if required

19/05/2021 - L. Bernardos
'''

import sys
import os
import glob
import shutil
import numpy as np

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.Preprocess as PRE
import MOLA.InternalShortcuts as J


RobustModes = [
dict(name='robust-0',
     NumericalKeys=dict(
        flux         = 'jameson',
        avcoef_k2    = 0.5,
        avcoef_k4    = 0.016,
        avcoef_sigma = 1.0,
        filter       = 'incr_new+prolong',
        cutoff_dens  = 0.005,
        cutoff_pres  = 0.005,
        cutoff_eint  = 0.005,
        artviscosity = 'dismrt',
        av_mrt       = 0.3,)),

dict(name='robust-1',
     NumericalKeys=dict(
        flux         = 'jameson',
        avcoef_k2    = 1.0,
        avcoef_k4    = 0.032,
        avcoef_sigma = 1.0,
        filter       = 'incr_new+prolong',
        cutoff_dens  = 0.005,
        cutoff_pres  = 0.005,
        cutoff_eint  = 0.005,
        artviscosity = 'dismrt',
        av_mrt       = 0.3,)),
]

FailSafeMode = dict(
    name='fail-safe',
    NumericalKeys=dict(
        flux         = 'roe',
        limiter      = 'valbada'))


def fileExists(*path): return os.path.isfile(os.path.join(*path))


def anyFile(*path): return any(glob.glob(os.path.join(*path)))


def removeNumericalSchemeKeys(setup):
    Keys2Remove = ('flux','ausm_wiggle','ausmp_diss_cst','ausmp_press_vel_cst',
                    'ausm_tref','ausm_pref','ausm_mref','limiter','avcoef_k2',
                    'avcoef_k4','avcoef_sigma','filter','cutoff_dens',
                    'cutoff_pres','cutoff_eint','artviscosity','av_mrt')
    for k in Keys2Remove: setup.elsAkeysNumerics.pop(k, None)


def wasPoorlyConverged():
    try: stdCLthreshold = setup.ReferenceValues['CoprocessOptions']['MaxConvergedCLStd']
    except: return False

    ArraysTree = C.convertFile2PyTree(os.path.join('OUTPUT','arrays.cgns'))
    AirfoilZone = [z for z in I.getZones(ArraysTree) if z[0] == 'AIRFOIL'][0]
    stdCL, = J.getVars(AirfoilZone,['std-CL'])

    if stdCL is None: return False

    PoorlyConverged = stdCL[-1] > 10**(int(np.log10(stdCLthreshold))+2)

    return PoorlyConverged


def getComputationMode():
    try: setup = J.load_source('setup', 'setup.py')
    except: return

    try:
        return setup.ReferenceValues['ComputationMode']
    except:
        return 'accurate'


def getNextRobustMode():
    ComputationMode = getComputationMode()

    if ComputationMode == 'fail-safe': return

    if ComputationMode == 'accurate':
        return RobustModes[0]

    for i, rm in enumerate(RobustModes):
        if rm['name'] == ComputationMode:
            break

    try:
        return RobustModes[i+1]
    except IndexError:
        return


def useNextRobustMode():

    try: setup = J.load_source('setup', 'setup.py')
    except: return

    RobustMode = getNextRobustMode()
    if not RobustMode: return

    removeNumericalSchemeKeys(setup)
    setup.elsAkeysNumerics.update(RobustMode['NumericalKeys'])
    setup.ReferenceValues['ComputationMode'] = RobustMode['name']
    PRE.writeSetupFromModuleObject(setup, setupFilename='setup.py')

    # allow for restart
    try: os.remove('COMPLETED')
    except: pass


def useFailSafeMode():

    try: setup = J.load_source('setup', 'setup.py')
    except: return

    if getComputationMode() == FailSafeMode['name']: return

    removeNumericalSchemeKeys(setup)
    setup.elsAkeysNumerics.update(FailSafeMode['NumericalKeys'])
    setup.ReferenceValues['ComputationMode'] = FailSafeMode['name']
    PRE.writeSetupFromModuleObject(setup, setupFilename='setup.py')

    # allow for restart
    files2Remove = ('COMPLETED','FAILED')
    for f in files2Remove:
        try: os.remove('COMPLETED')
        except: pass

    # move OUTPUT folder so that new run will restart from previous healthy one
    shutil.move('OUTPUT', 'OUTPUT_FAILED')



# ---------------------------- SCRIPT STARTS HERE ---------------------------- #


for failure in ('bk*','elsA.x.*','core.*'):
    if anyFile(failure):
        with open('FAILED','w') as f: f.write('dump %s files detected'%failure)
        break


if fileExists('OUTPUT','tmp-fields.cgns'):
    shutil.move(os.path.join('OUTPUT','tmp-fields.cgns'),
                os.path.join('OUTPUT','fields.cgns'))

# elif not anyFile('NEWJOB_REQUIRED','FAILED'):
#     with open('FAILED','w') as f:
#         f.write('tmp-fields.cgns missing')

for fn in glob.glob('*.log'):
    FilenameBase = fn[:-4]
    i = 1
    NewFilename = FilenameBase+'-%d'%i+'.log'
    while fileExists('LOGS', NewFilename):
        i += 1
        NewFilename = FilenameBase+'-%d'%i+'.log'

    shutil.move(fn, os.path.join('LOGS', NewFilename))
for fn in glob.glob('elsA_MPI*'):
    shutil.move(fn, os.path.join('LOGS', fn))

if fileExists('COMPLETED'):

    for fn in glob.glob(os.path.join('LOGS','stdout*')):
        try: os.remove(fn) # because standard output of elsA is excessively big
        except: pass

    if wasPoorlyConverged(): useNextRobustMode()

elif fileExists('FAILED'):
    useFailSafeMode()
