    WF.prepareMainCGNS(t=t, meshParams=meshParams,
                        writeOutputFields=False,
                        Reynolds=1.6e5, Mach=0.3, AngleOfAttackDeg=6.0,
                        TurbulenceLevel=0.1 * 1e-2,
                        TransitionMode='NonLocalCriteria',
                        TurbulenceModel='Wilcox2006-klim',
                        InitialIteration=1, NumberOfIterations=100000,
                        NumericalScheme='ausm+',
                        TimeMarching='steady')
    WF.prepareJob(JobName, AER, DIRECTORY_RESTART,
                  NProcs=meshParams['options']['NProcs'])
