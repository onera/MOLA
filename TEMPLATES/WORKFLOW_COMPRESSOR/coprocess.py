'''
coprocess.py template
'''

# Control Flags for interactive control using command 'touch <flag>'
if CO.getSignal('QUIT'): os._exit(0)
CONVERGED         = CO.getSignal('CONVERGED')
SAVE_SURFACES     = CO.getSignal('SAVE_SURFACES')
SAVE_FIELDS       = CO.getSignal('SAVE_FIELDS')
SAVE_LOADS        = CO.getSignal('SAVE_LOADS')
SAVE_BODYFORCE    = CO.getSignal('SAVE_BODYFORCE')
COMPUTE_BODYFORCE = CO.getSignal('COMPUTE_BODYFORCE')

if CO.getSignal('RELOAD_SETUP'):
    # BEWARE: in Python v >= 3.4 rather use: importlib.reload(setup)
    if setup and setup.__name__ != "__main__": imp.reload(setup)
    CO.setup = setup
    niter    = setup.elsAkeysNumerics['niter']
    inititer = setup.elsAkeysNumerics['inititer']
    itmax    = inititer+niter-1 # BEWARE last iteration accessible trigger-state-16

    try: BodyForceInputData = setup.BodyForceInputData
    except: BodyForceInputData = None

    if BodyForceInputData:
        LocalBodyForceInputData = LL.getLocalBodyForceInputData(BodyForceInputData)
        LL.invokeAndAppendLocalObjectsForBodyForce(LocalBodyForceInputData)
        NumberOfSerialRuns = LL.getNumberOfSerialRuns(BodyForceInputData, NProcs)


UpdateFieldsFrequency     = CO.getOption('UpdateFieldsFrequency', default=1e3)
UpdateLoadsFrequency      = CO.getOption('UpdateLoadsFrequency', default=20)
UpdateSurfacesFrequency   = CO.getOption('UpdateSurfacesFrequency', default=500)
BodyForceSaveFrequency    = CO.getOption('BodyForceSaveFrequency', default=500)
BodyForceComputeFrequency = CO.getOption('BodyForceComputeFrequency', default=500)
BodyForceInitialIteration = CO.getOption('BodyForceInitialIteration', default=1000)
MarginBeforeTimeOut       = CO.getOption('SecondsMargin4QuitBeforeTimeOut', default=120.)
TimeOut                   = CO.getOption('TimeOutInSeconds', default=53100.0)
ItersMinEvenIfConverged   = CO.getOption('ItersMinEvenIfConverged', default=1e3)
MaxConvergedCriterionStd  = CO.getOption('MaxConvergedCriterionStd', default=1e-5)
ConvergenceFamilyName     = CO.getOption('ConvergenceCriterionFamilyName', default='NONE')
ConvergenceFluxName       = CO.getOption('ConvergenceFluxName', default='std-MassFlowIn')

DesiredStatistics = ['std-{}'.format(var) for var in ['MassFlowIn', 'MassFlowOut',
    'PressureStagnationRatio', 'TemperatureStagnationRatio', 'EfficiencyIsentropic',
    'PressureStagnationLossCoeff']]


# BEWARE! state 16 => triggers *before* iteration, which means
# that variable "it" represents actually the *next* iteration
it = elsAxdt.iteration()
CO.CurrentIteration = it
CO.printCo('iteration %d'%it, proc=0)


# ENTER COUPLING CONDITIONS:

if not SAVE_FIELDS:
    SAVE_FIELDS = all([(it-inititer)%UpdateFieldsFrequency == 0, it>inititer])

if not SAVE_SURFACES:
    SAVE_SURFACES = all([(it-inititer)%UpdateSurfacesFrequency == 0, it>inititer])

if not SAVE_LOADS:
    SAVE_LOADS = all([(it-inititer)%UpdateLoadsFrequency == 0, it>inititer])

if not SAVE_BODYFORCE:
    SAVE_BODYFORCE = all([ BodyForceInputData,
                          (it - inititer) % BodyForceSaveFrequency == 0,
                          it>inititer])

if BodyForceInputData and not COMPUTE_BODYFORCE:
    if it >= BodyForceInitialIteration:
        COMPUTE_BODYFORCE = any([(it-inititer)%BodyForceComputeFrequency == 0,
                                 not BODYFORCE_INITIATED])

ElapsedTime = timeit.default_timer() - LaunchTime
ReachedTimeOutMargin = CO.hasReachedTimeOutMargin(ElapsedTime, TimeOut,
                                                            MarginBeforeTimeOut)
anySignal = any([SAVE_LOADS, SAVE_SURFACES, SAVE_BODYFORCE, COMPUTE_BODYFORCE,
                 SAVE_FIELDS, CONVERGED, it>=itmax])
ENTER_COUPLING = anySignal or ReachedTimeOutMargin

if ENTER_COUPLING:

    to = elsAxdt.get(elsAxdt.OUTPUT_TREE)
    toWithSkeleton = I.merge([Skeleton, to])
    CO.adaptEndOfRun(toWithSkeleton)

    if SAVE_FIELDS:
        CO.save(toWithSkeleton, os.path.join(DIRECTORY_OUTPUT,FILE_FIELDS))

    if SAVE_LOADS:
        CO.extractIntegralData(to, loads)
        CO.addMemoryUsage2Loads(loads)
        loadsTree = CO.loadsDict2PyTree(loads)
        CO.save(loadsTree, os.path.join(DIRECTORY_OUTPUT,FILE_LOADS))

    if SAVE_SURFACES:
        surfs = CO.extractSurfaces(toWithSkeleton, setup.Extractions)
        CO.save(surfs,os.path.join(DIRECTORY_OUTPUT,FILE_SURFACES))
        CO.monitorTurboPerformance(surfs, loads, DesiredStatistics)

        if (it-inititer)>ItersMinEvenIfConverged and not CONVERGED:
            CONVERGED=CO.isConverged(ZoneName=ConvergenceFamilyName,
                                     FluxName=ConvergenceFluxName,
                                     FluxThreshold=MaxConvergedCriterionStd)

    if CONVERGED or it >= itmax or ReachedTimeOutMargin:
        if ReachedTimeOutMargin:
            if rank == 0:
                with open('NEWJOB_REQUIRED','w') as f: f.write('NEWJOB_REQUIRED')

        if it >= itmax or CONVERGED:
            if rank==0:
                with open('COMPLETED','w') as f: f.write('COMPLETED')

        CO.printCo('TERMINATING COMPUTATION', proc=0, color=CO.GREEN)
        CO.updateAndWriteSetup(setup)
        elsAxdt.safeInterrupt()
