import MOLA.WorkflowAirfoil as WF
import MOLA.JobManager      as JM

PREFIX_JOB = 'j' # MUST BE UNIQUE for each airfoil
AER = '31447034F'
machine = 'sator'
DIRECTORY_WORK = '/tmp_user/sator/lbernard/MYPOLARS/'
GeomPath = '/home/ffalissa/H2T/ETUDES/MOTUS/FLUX_2/POLAIRES/PROFILS/Airfoil_20.tp'


AoARange    = [0,1,2,4,6,10,12,14,16,-1,-2,-4]
MachRange  =  [0.5, 0.6]
ReynoldsOverMach = 600000.0

if AoARange[0] != 0: raise ValueError('AoARange MUST start at 0')

AoAMatrix, MachMatrix  = np.meshgrid(AoARange, MachRange)
ReynoldsMatrix = MachMatrix * ReynoldsOverMach

AoA_  =      AoAMatrix.ravel(order='K')
M_    =     MachMatrix.ravel(order='K')
Re_   = ReynoldsMatrix.ravel(order='K')
NewJobs = AoA_ == 0

JobsQueues = []
for i, (AoA, Reynolds, Mach, NewJob) in enumerate(zip(AoA_, Re_, M_, NewJobs)):

    print('Assembling run {} AoA={} Re={} M={} | NewJob = {}'.format(
            i, AoA, Reynolds, Mach, NewJob))

    if NewJob:
        JobName = PREFIX_JOB+'%d'%i
        writeOutputFields = True
    else:
        writeOutputFields = False

    CASE_LABEL = '%06.2f'%abs(AoA)+'_'+JobName # CAVEAT tol AoA >= 0.01 deg
    if AoA < 0: CASE_LABEL = 'M'+CASE_LABEL

    meshParams = WF.getMeshingParameters()
    meshParams['References'].update({'Reynolds':Reynolds})


    EffectiveMach = np.maximum(Mach, 0.2)
    TransitionMode = 'NonLocalCriteria' if Reynolds < 3e5 else None

    CoprocessOptions = dict(
        UpdateFieldsFrequency   = 2000,
        UpdateLoadsFrequency    = 50,
        NewSurfacesFrequency    = 500,
        AveragingIterations     = 3000,
        MaxConvergedCLStd       = 1e-6,
        ItersMinEvenIfConverged = 3000,
        TimeOutInSeconds        = 54000.0, # 15 h * 3600 s/h = 53100 s
        SecondsMargin4QuitBeforeTimeOut = 900.,
                            )

    TransitionZones = dict(
                          TopOrigin = 0.003,
                       BottomOrigin = 0.05,
              TopLaminarImposedUpTo = 0.01,
            TopLaminarIfFailureUpTo = 0.07,
            TopTurbulentImposedFrom = 0.95,
           BottomLaminarImposedUpTo = 0.10,
         BottomLaminarIfFailureUpTo = 0.50,
         BottomTurbulentImposedFrom = 0.95,
                           )

    ImposedWallFields = [
        # dict(fieldname    = 'intermittency_imposed',
        #      side         = 'top',
        #      Xref         = TopXtr,
        #      LengthScale  = TopLengthScale,
        #      a_boost      = a_boost,
        #      sa           = sa,
        #      sb           = sb,
        #      ),

        # dict(fieldname    = 'clim_imposed',
        #      side         = 'top',
        #      Xref         = TopXtr+TopLengthScale*sc,
        #      LengthScale  = 1e-6,
        #      a_boost      = 1.0,
        #      sa           = 1e-3,
        #      sb           = 0.0,
        #      ),

        # dict(fieldname    = 'intermittency_imposed',
        #      side         = 'bottom',
        #      Xref         = BottomXtr,
        #      LengthScale  = BottomLengthScale,
        #      a_boost      = a_boost,
        #      sa           = sa,
        #      sb           = sb,
        #      ),

        # dict(fieldname    = 'clim_imposed',
        #      side         = 'bottom',
        #      Xref         = BottomXtr+BottomLengthScale*sc,
        #      LengthScale  = 1e-6,
        #      a_boost      = 1.0,
        #      sa           = 1e-3,
        #      sb           = 0.0,
        #      ),
    ]


    elsAParams = dict(Reynolds=Reynolds,
                      Mach=EffectiveMach,
                      AngleOfAttackDeg=AoA,
                      writeOutputFields=writeOutputFields,
                      TurbulenceLevel=0.1 * 1e-2,
                      TransitionMode=TransitionMode,
                      TurbulenceModel='Wilcox2006-klim',
                      InitialIteration=1, NumberOfIterations=50000,
                      NumericalScheme='ausm+',
                      TimeMarching='steady')

    JobsQueues.append( dict(ID=i, CASE_LABEL=CASE_LABEL, NewJob=NewJob,
        JobName=JobName, meshParams=meshParams, elsAParams=elsAParams,
        CoprocessOptions=CoprocessOptions, TransitionZones=TransitionZones,
        ImposedWallFields=ImposedWallFields,) )

JM.saveJobsConfiguration(JobsQueues, AER, machine, DIRECTORY_WORK,GeomPath)

JM.launchJobsConfiguration()
