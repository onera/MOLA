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
COMPUTE_BODYFORCE = CO.getSignal('COMPUTE_BODYFORCE')

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
    itmax    = inititer+niter-1 # BEWARE last iteration accessible trigger-state-16

    try: BodyForceInputData = setup.BodyForceInputData
    except: BodyForceInputData = None

UpdateFieldsFrequency     = CO.getOption('UpdateFieldsFrequency', default=1e3)
UpdateArraysFrequency     = CO.getOption('UpdateArraysFrequency', default=20)
UpdateSurfacesFrequency   = CO.getOption('UpdateSurfacesFrequency', default=500)
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

if not SAVE_BODYFORCE:
    SAVE_BODYFORCE = all([BodyForceInputData,
                          it % BodyForceSaveFrequency == 0,
                          it > inititer])

if BodyForceInputData and not COMPUTE_BODYFORCE:
    if it >= BodyForceInitialIteration:
        COMPUTE_BODYFORCE = any([it%BodyForceComputeFrequency == 0,
                                 not BODYFORCE_INITIATED])

ElapsedTime = timeit.default_timer() - LaunchTime
ReachedTimeOutMargin = CO.hasReachedTimeOutMargin(ElapsedTime, TimeOut,
                                                            MarginBeforeTimeOut)
anySignal = any([SAVE_ARRAYS, SAVE_SURFACES, SAVE_BODYFORCE, COMPUTE_BODYFORCE,
                 SAVE_FIELDS, CONVERGED, it>=itmax])
ENTER_COUPLING = anySignal or ReachedTimeOutMargin

if ENTER_COUPLING:

    t = CO.extractFields(Skeleton)

    if COMPUTE_BODYFORCE:
        BODYFORCE_INITIATED = True
        elsAxdt.free('xdt-runtime-tree')
        Cmpi.barrier()
        toWithSourceTerms = CO.updateBodyForce(t, toWithSourceTerms)

        if SAVE_BODYFORCE:
            BodyForceTree = C.newPyTree(['BODYFORCE'])
            BodyForceBase = I.getBases(BodyForceTree)[0]
            for zone in I.getZones(toWithSourceTerms):
                if I.getNodeFromName1(zone, 'FlowSolution#SourceTerm'):
                    I.addChild(BodyForceBase, zone)
            CO.save(BodyForceTree, os.path.join(DIRECTORY_OUTPUT, FILE_BODYFORCESRC))

    if SAVE_FIELDS:
        CO.save(t, os.path.join(DIRECTORY_OUTPUT,FILE_FIELDS))

    if SAVE_SURFACES:
        surfs = CO.extractSurfaces(t, setup.Extractions)
        CO.save(surfs,os.path.join(DIRECTORY_OUTPUT,FILE_SURFACES))
        arraysTree = CO.monitorTurboPerformance(surfs, arrays, RequestedStatistics)
        SAVE_ARRAYS = True

        if (it-inititer)>ItersMinEvenIfConverged and not CONVERGED:
            CONVERGED = CO.isConverged(ConvergenceCriteria)

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

        CO.printCo('TERMINATING COMPUTATION', proc=0, color=CO.GREEN)
        CO.updateAndWriteSetup(setup)
        elsAxdt.safeInterrupt()

if BODYFORCE_INITIATED:
    Cmpi.barrier()
    # CO.printCo('sending source terms to elsA...', proc=0)
    elsAxdt.xdt(elsAxdt.PYTHON, ('xdt-runtime-tree', toWithSourceTerms, 1))
