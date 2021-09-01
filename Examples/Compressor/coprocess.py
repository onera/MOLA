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



UpdateFieldsFrequency   = CO.getOption('UpdateFieldsFrequency', default=1e3)
UpdateLoadsFrequency    = CO.getOption('UpdateLoadsFrequency', default=20)
UpdateSurfacesFrequency = CO.getOption('UpdateSurfacesFrequency', default=500)
MarginBeforeTimeOut     = CO.getOption('SecondsMargin4QuitBeforeTimeOut', default=120.)
TimeOut                 = CO.getOption('TimeOutInSeconds', default=53100.0)
MaxConvergedCLStd       = CO.getOption('MaxConvergedCLStd', default=1e-5)
ItersMinEvenIfConverged = CO.getOption('ItersMinEvenIfConverged', default=1e3)
ConvergenceFamilyName   = CO.getOption('ConvergenceCriterionFamilyName', default='NONE')
BodyForceSaveFrequency  = CO.getOption('BodyForceSaveFrequency', default=500)
BodyForceComputeFrequency = CO.getOption('BodyForceComputeFrequency', default=500)
BodyForceInitialIteration = CO.getOption('BodyForceInitialIteration', default=1000)


DesiredStatistics=['std-CL', 'std-CD']


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
reachedTimeOutMargin = CO.hasReachedTimeOutMargin(ElapsedTime, TimeOut,
                                                            MarginBeforeTimeOut)
anySignal = any([SAVE_LOADS, SAVE_SURFACES, SAVE_BODYFORCE, COMPUTE_BODYFORCE,
                 SAVE_FIELDS, CONVERGED, it>=itmax])
ENTER_COUPLING = anySignal or reachedTimeOutMargin

# Desactivate loads Saving
SAVE_LOADS = False

if ENTER_COUPLING:

    to = elsAxdt.get(elsAxdt.OUTPUT_TREE)
    toWithSkeleton = I.merge([Skeleton, to])
    CO.adaptEndOfRun(toWithSkeleton)


    if COMPUTE_BODYFORCE:
        BODYFORCE_INITIATED = True
        CO.printCo('COMPUTING BODYFORCE', proc=0, color=CO.MAGE)
        BodyForceDisks = LL.computePropellerBodyForce(toWithSkeleton,
                                                      NumberOfSerialRuns,
                                                      LocalBodyForceInputData)

        CO.addBodyForcePropeller2Loads(loads, BodyForceDisks)

        CO.distributeAndSavePyTree(BodyForceDisks, FILE_BODYFORCESRC,
                                   tagWithIteration=False)
        SAVE_BODYFORCE = False


        elsAxdt.free('xdt-runtime-tree')
        del toWithSourceTerms
        CO.printCo('migrating computed source terms...', proc=0, color=CO.MAGE)
        toWithSourceTerms = LL.migrateSourceTerms2MainPyTree(BodyForceDisks,
                                                             toWithSkeleton)


    if SAVE_FIELDS:
        CO.saveDistributedPyTree(toWithSkeleton, FILE_FIELDS)
        Cmpi.barrier()

    if SAVE_LOADS:
        CO.updateAndSaveLoads(to, loads, DesiredStatistics, monitorMemory=True)

        if (it-inititer)>ItersMinEvenIfConverged and not CONVERGED:
            CONVERGED=CO.isConverged(ZoneName=ConvergenceFamilyName,
                                     FluxName='std-CL',
                                     FluxThreshold=MaxConvergedCLStd)

    if SAVE_SURFACES:
        CO.saveSurfaces(toWithSkeleton, loads, tagWithIteration=False, onlyWalls=False)


    if SAVE_BODYFORCE:
        CO.distributeAndSavePyTree(BodyForceDisks, FILE_BODYFORCESRC,
                                   tagWithIteration=False)

    if CONVERGED or it >= itmax or reachedTimeOutMargin:
        if reachedTimeOutMargin:
            CO.printCo('REACHED MARGIN BEFORE TIMEOUT', proc=0, color=CO.WARN)

        CO.saveAll(toWithSkeleton, to, loads, DesiredStatistics,
                   BodyForceInputData, BodyForceDisks, quit=True)



if BODYFORCE_INITIATED:
    Cmpi.barrier()
    CO.printCo('sending source terms to elsA...', proc=0)
    elsAxdt.xdt(elsAxdt.PYTHON, ('xdt-runtime-tree', toWithSourceTerms, 1) )
