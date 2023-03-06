#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

'''
coprocess.py template
'''

# Control Flags for interactive control using command 'touch <flag>'
if CO.getSignal('QUIT'): os._exit(0)
CONVERGED         = CO.getSignal('CONVERGED')
SAVE_SURFACES     = CO.getSignal('SAVE_SURFACES')
SAVE_FIELDS       = CO.getSignal('SAVE_FIELDS')
SAVE_ARRAYS       = CO.getSignal('SAVE_ARRAYS')
SAVE_BODYFORCE    = CO.getSignal('SAVE_BODYFORCE')
SAVE_ALL          = CO.getSignal('SAVE_ALL')

if SAVE_ALL:
    SAVE_SURFACES  = True
    SAVE_FIELDS    = True
    SAVE_ARRAYS    = True
    SAVE_BODYFORCE = True


if CO.getSignal('RELOAD_SETUP'):
    if setup and setup.__name__ != "__main__": J.reload_source(setup)
    CO.setup = setup
    niter    = setup.elsAkeysNumerics['niter']
    inititer = setup.elsAkeysNumerics['inititer']
    itmax    = inititer+niter-2 # BEWARE last iteration accessible trigger-state-16


UpdateFieldsFrequency     = CO.getOption('UpdateFieldsFrequency', default=1e3)
UpdateArraysFrequency     = CO.getOption('UpdateArraysFrequency', default=20)
UpdateSurfacesFrequency   = CO.getOption('UpdateSurfacesFrequency', default=500)
UpdateCWIPICouplingFrequency = CO.getOption('UpdateCWIPICouplingFrequency', default=None)
BodyForceSaveFrequency    = CO.getOption('BodyForceSaveFrequency', default=500)
BodyForceComputeFrequency = CO.getOption('BodyForceComputeFrequency', default=500)
BodyForceInitialIteration = CO.getOption('BodyForceInitialIteration', default=1000)
MarginBeforeTimeOut       = CO.getOption('SecondsMargin4QuitBeforeTimeOut', default=120.)
TimeOut                   = CO.getOption('TimeOutInSeconds', default=53100.0)
ItersMinEvenIfConverged   = CO.getOption('ItersMinEvenIfConverged', default=1e3)
ConvergenceCriteria       = CO.getOption('ConvergenceCriteria', default=[])
RequestedStatistics       = CO.getOption('RequestedStatistics', default=[])


# BEWARE! state 16 => triggers *before* iteration, which means
# that variable "it" represents actually the *next* iteration
it = elsAxdt.iteration()
CO.CurrentIteration = it
CO.printCo('iteration %d'%it, proc=0)


# ENTER COUPLING CONDITIONS:

if not SAVE_FIELDS:
    SAVE_FIELDS = all([it%UpdateFieldsFrequency == 0, it>inititer])

if not SAVE_SURFACES:
    SAVE_SURFACES = all([it%UpdateSurfacesFrequency == 0, it>inititer])

if not SAVE_ARRAYS:
    SAVE_ARRAYS = all([it%UpdateArraysFrequency == 0, it>inititer])

CWIPY_COUPLING = all([UpdateCWIPICouplingFrequency,
                      it%UpdateCWIPICouplingFrequency == 0,
                      it>inititer])

ElapsedTime = timeit.default_timer() - LaunchTime
ReachedTimeOutMargin = CO.hasReachedTimeOutMargin(ElapsedTime, TimeOut,
                                                            MarginBeforeTimeOut)
anySignal = any([SAVE_ARRAYS, SAVE_SURFACES, SAVE_FIELDS, CWIPY_COUPLING,
                CONVERGED, it>=itmax])
ENTER_COUPLING = anySignal or ReachedTimeOutMargin

if ENTER_COUPLING:

    t = CO.extractFields(Skeleton)

    if SAVE_FIELDS:
        CO.save(t, os.path.join(DIRECTORY_OUTPUT,FILE_FIELDS))

    if SAVE_SURFACES:
        surfs = CO.extractSurfaces(t, setup.Extractions)
        CO.save(surfs,os.path.join(DIRECTORY_OUTPUT,FILE_SURFACES))
        arraysTree = CO.monitorTurboPerformance(surfs, arrays, RequestedStatistics)
        SAVE_ARRAYS = True

        if (it-inititer)>ItersMinEvenIfConverged and not CONVERGED:
            CONVERGED = CO.isConverged(ConvergenceCriteria)

    if CWIPY_COUPLING:
        toWithUpdatedBC, CWIPIdata = WAT.cwipiCoupling(t, pyC2Connections, setup, it)
        arraysTree = WAT.appendCWIPIDict2Arrays(arrays, CWIPIdata, it, RequestedStatistics)
        SAVE_ARRAYS = True
        CO.printCo('updating BCs in elsA...', proc=0)
        elsAxdt.xdt(elsAxdt.PYTHON,(elsAxdt.RUNTIME_TREE, toWithUpdatedBC, 1))

    if SAVE_ARRAYS:
        arraysTree = CO.extractArrays(t, arrays, RequestedStatistics=RequestedStatistics,
                  Extractions=setup.Extractions, addMemoryUsage=True)
        CO.save(arraysTree, os.path.join(DIRECTORY_OUTPUT,FILE_ARRAYS))

    if CONVERGED or it >= itmax or ReachedTimeOutMargin:
        if ReachedTimeOutMargin:
            CO.printCo('REACHED TIMEOUT', proc=0, color=J.WARN)
            if rank == 0:
                with open('NEWJOB_REQUIRED','w') as f: f.write('NEWJOB_REQUIRED')

        if it >= itmax or CONVERGED:
            if it >= itmax:
                CO.printCo('REACHED itmax = %d'%itmax, proc=0, color=J.GREEN)
            if rank==0:
                with open('COMPLETED','w') as f: f.write('COMPLETED')

        Cmpi.barrier()
        elsAxdt.safeInterrupt()
