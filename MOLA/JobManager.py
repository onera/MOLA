'''
MOLA - JobManager.py

------------------------- IMPORTANT ----------------------------------
This module should be the ONLY ONE in MOLA to import the module _cpmv_
----------------------------------------------------------------------

First creation:
07/10/2021 - T. Bontemps - creation
'''

import os
import subprocess
import socket
import getpass
import pprint
import shutil

import Converter.PyTree         as C
import Converter.Internal       as I


from . import __version__
from . import InternalShortcuts as J
from . import _cpmv_            as ServerTools

MOLA_PATH = os.getenv('MOLA','/stck/lbernard/MOLA/%s'%__version__)

def checkDependencies():
    '''
    Make a series of functional tests in order to determine if the user
    environment is correctly set for using MOLA. Each Workflow may need to
    make additional tests.
    '''
    def checkModuleVersion(module, MinimumRequired):
        print('Checking %s...'%module.__name__)
        print('used version: '+module.__version__)
        print('minimum required: '+MinimumRequired)
        VerList = module.__version__.split('.')
        VerList = [int(v) for v in VerList]

        ReqVerList = MinimumRequired.split('.')
        ReqVerList = [int(v) for v in ReqVerList]

        for used, required in zip(VerList,ReqVerList):
            if used > required:
                print(J.GREEN+'%s version OK'%module.__name__+J.ENDC)
                return True

            elif used < required:
                print(J.WARN+'WARNING: using outdated version of %s'%module.__name__+J.ENDC)
                print(J.WARN+'Please upgrade, for example, try:')
                print(J.WARN+'pip install --user --upgrade %s'%module.__name__+J.ENDC)
                return False

    import numpy as np
    checkModuleVersion(np, '1.16.6')

    import scipy
    checkModuleVersion(scipy, '1.2.3')

    print('\nChecking interpolations...')
    AbscissaRequest = np.array([1.0,2.0,3.0])
    AbscissaData = np.array([1.0,2.0,3.0,4.0,5.0])
    ValuesData = AbscissaData**2
    for Law in ('interp1d_linear', 'interp1d_quadratic','cubic','pchip','akima'):
        J.interpolate__(AbscissaRequest, AbscissaData, ValuesData, Law=Law)
    print(J.GREEN+'interpolation OK'+J.ENDC)

    print('\nAttempting file/directories operations on SATOR...')
    TestFile = 'testfile.txt'
    with open(TestFile,'w') as f: f.write('test')
    UserName = getpass.getuser()
    DIRECTORY_TEST = '/tmp_user/sator/%s/MOLAtest/'%UserName
    Source = TestFile
    Destination = os.path.join(DIRECTORY_TEST,TestFile)
    ServerTools.cpmvWrap4MultiServer('mv', Source, Destination)
    repatriate(Destination, Source, removeExistingDestinationPath=False)
    print(DIRECTORY_TEST)
    ServerTools.cpmvWrap4MultiServer('rm', DIRECTORY_TEST)
    ServerTools.cpmvWrap4MultiServer('rm', Source)
    print('Attempting file/directories operations on SATOR... done')

    import matplotlib
    checkModuleVersion(matplotlib, '2.2.5')
    print('producing figure...')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(dpi=150)
    t = np.linspace(0, 1, 100)
    ax.plot(t, np.sin(2*np.pi*t), color='C0')
    ax.set_xlabel('t')
    ax.set_ylabel('sin($2\pi t$)')
    ax.set_title('Test figure')
    plt.tight_layout()
    print('saving figure...')
    plt.savefig('test-figure.pdf')
    print('showing figure... (close figure to continue)')
    plt.show()


def buildJob(case, config, NProc, jobTemplate):
    '''
    Produce a computation job file.

    Parameters
    ----------

        case : dict
            current item of **JobsQueues** list

        config : module
            import of ``JobsConfiguration.py``

        NProc : int
            number of procs of the job

            .. danger:: must be coherent with the ``main.cgns`` distribution

    Returns
    -------

        None : None
            Write file ``job.sh``
    '''
    JobFile = 'job.sh'

    with open('routine.sh','r') as f: RoutineText = f.read()
    with open(jobTemplate,'r') as f: JobText = f.read()

    JobText = JobText.replace('<JobName>', case['JobName'])
    JobText = JobText.replace('<AERnumber>', config.AER)
    JobText = JobText.replace('<NProcs>', str(NProc))
    JobText = JobText.split('mpirun ')[0]

    # Add source to /etc/bashrc
    if not 'source /etc/bashrc' in JobText:
        if 'module purge' in JobText:
            JobText = JobText.replace('module purge', \
                                      'module purge\nsource /etc/bashrc')
        else:
            # Insert line after the last #SBATCH line
            JobTextSplitBySBATCH = JobText.split('#SBATCH')
            lastSBATCHline = '#SBATCH' + JobTextSplitBySBATCH[-1].split('\n')[0] + '\n'
            headerSBATCH = '#SBATCH'.join(JobTextSplitBySBATCH[:-1]) + lastSBATCHline
            TextAfterheaderSBATCH = '\n'.join(JobTextSplitBySBATCH[-1].split('\n')[1:])
            JobText = headerSBATCH + '\nsource /etc/bashrc\n' + TextAfterheaderSBATCH

    with open(JobFile,'w+') as f:
        f.write(JobText)
        f.write(RoutineText)
        f.write('\n')

    os.chmod(JobFile, 0o777)

def saveJobsConfiguration(JobsQueues, AER, machine, DIRECTORY_WORK,
                           FILE_GEOMETRY=None, NProc=None):
    '''
    Generate the file ``JobsConfiguration.py`` from provided user-data.

    Parameters
    ----------

        JobsQueues : :py:class:`list` of :py:class:`dict`
            Each dictionary defines the configuration of a CFD run case.
            Each dictionary has the following keys:

            * ID : :py:class:`int`
                the unique identification number of the elsA run

            * CASE_LABEL : :py:class:`str`
                the unique identification label of the elsA run

            * NewJob : bool
                if :py:obj:`True`, then this case starts as a new job
                and employs a uniform initialization of flowfields using
                ``ReferenceState`` (this is, no restart is produced employing
                previous ID case)

            * JobName : :py:class:`str`
                the job name employed by the case


        AER : str
            full AER code for launching simulations on SATOR

        machine : str
            name of the machine ``'sator'``, ``'spiro'``, ``'eos'``...

            .. warning:: only ``'sator'`` has been fully implemented

        DIRECTORY_WORK : str
            the working directory at computation server.
            If it does not exist, then it will be automatically created.

        FILE_GEOMETRY : :py:class:`str` or :py:obj:`None`
            location where geometry is located. Not written if :py:obj:`None`

        NProc : int
            Number of processors

    Returns
    -------

        None : None
            writes file ``JobsConfiguration.py``
    '''
    JobsConfigurationFilename = 'JobsConfiguration.py'

    if not DIRECTORY_WORK.endswith('/'): raise ValueError('DIRECTORY_WORK must end with "/"')

    AllowedMachines = ('spiro', 'sator','sator-new','local','eos','ld')
    if machine not in AllowedMachines:
        raise ValueError('Machine %s not supported. Must be one of: %s'%(machine,str(AllowedMachines)))

    Lines = ["'''\n%s file automatically generated by MOLA\n'''\n"%JobsConfigurationFilename]
    Lines+=['DIRECTORY_WORK="'+DIRECTORY_WORK+'"\n']
    if FILE_GEOMETRY:
        Lines+=['FILE_GEOMETRY="'+FILE_GEOMETRY+'"\n']
    if NProc:
        Lines+=['NProc="'+str(NProc)+'"\n']
    Lines+=['machine="'+machine+'"\n']
    Lines+=['AER="'+AER+'"\n\n']
    Lines+=['JobsQueues='+pprint.pformat(JobsQueues)+'\n']

    try: os.remove(JobsConfigurationFilename)
    except: pass

    try: os.remove(JobsConfigurationFilename+'c')
    except: pass

    with open(JobsConfigurationFilename,'w') as f:
        for l in Lines:
            f.write(l)

    print('\nwritten file '+J.GREEN+JobsConfigurationFilename+J.ENDC+'\n')

def launchJobsConfiguration(
        templatesFolder=MOLA_PATH+'/TEMPLATES/WORKFLOW_AIRFOIL',
        jobTemplate=MOLA_PATH+'/TEMPLATES/job_template.sh',
        DispatchFile='dispatch.py',
        routineFile='routine.sh',

        # TODO group these attributes in a list (specific to the WF)
        preprocessFile='preprocess.py',
        computeFile='compute.py',
        coprocessFile='coprocess.py',
        postprocessFile='postprocess.py',

        otherFiles=[]):
    '''
    Migrates a set of required scripts and launch the multi-jobs script.

    Parameters
    ----------

        templatesFolder : str
            location of the templates files

        preprocessFile : str
            name of the ``preprocess.py`` file required just *before* elsA execution

        computeFile : str
            name of the ``compute.py`` file required to launch elsA computation

        coprocessFile : str
            ``coprocess.py`` file used *during* elsA computation

        postprocessFile : str
            ``postprocess.py`` file used *after* elsA computation

        DispatchFile : str
            ``dispatch.py`` file used to produce
            the required CGNS and setup files as well as the jobs of the polars, and
            launch them.

        routineFile : str
            bash file used as template for each computation job execution.

        otherFiles : :py:class:`list` of :py:class:`str`
            other files to copy for the simulation.

            .. attention:: There are not searched into **templatesFolder**,
                because they are specific to the simulation. Please provide the
                full relative or absolute paths.
    '''
    NAME_DISPATCHER = 'DISPATCHER'


    config = J.load_source('config', 'JobsConfiguration.py')


    print(J.WARN+config.DIRECTORY_WORK+J.ENDC)

    DIRECTORY_DISPATCHER = os.path.join(config.DIRECTORY_WORK, NAME_DISPATCHER)

    JOBS_CONFIG_REMOTE = os.path.join(DIRECTORY_DISPATCHER,'JobsConfiguration.py')
    if remoteFileExists(JOBS_CONFIG_REMOTE, remote_machine=config.machine):
        raise ValueError(J.FAIL+'DIRECTORY %s ALREADY EXISTS. CANCELLING.'%config.DIRECTORY_WORK+J.ENDC)

    Files2Copy = ['JobsConfiguration.py']
    if hasattr(config, 'FILE_GEOMETRY'):
        Files2Copy.append(config.FILE_GEOMETRY)

    for filename in [preprocessFile, computeFile, coprocessFile, \
        postprocessFile, DispatchFile, routineFile]:
        Files2Copy.append(os.path.join(templatesFolder, filename))
    Files2Copy.append(jobTemplate)

    for filename in otherFiles:
        Files2Copy.append(filename)

    for filepath in Files2Copy:
        filename = filepath.split(os.path.sep)[-1]
        Source = filepath
        Destination = os.path.join(DIRECTORY_DISPATCHER, filename)
        # repatriate(Source, Destination, removeExistingDestinationPath=True)
        ServerTools.cpmvWrap4MultiServer('cp', Source, Destination)

    ComputeServers = ('sator', 'sator-new', 'spiro')
    if config.machine not in ComputeServers:
        print('Machine "%s" not in %s: assuming that local subprocess launch of elsA is possible.'%(config.machine, str(ComputeServers)))
        # launch local mesher-dispatcher job by subprocess - no sbatch no wait ssh
        out = open((os.path.join(DIRECTORY_DISPATCHER,'Dispatch-out.log')),'w')
        err = open((os.path.join(DIRECTORY_DISPATCHER,'Dispatch-err.log')),'w')
        ssh = subprocess.Popen('python dispatch.py',
                                shell=True,
                                stdout=out,
                                stderr=err,
                                env=os.environ.copy(),
                                cwd=DIRECTORY_DISPATCHER,
                                )
        # ssh.wait()

    else:
        print('Assuming SLURM job manager')

        # 1 - Create slurm mesher-dispatcher job
        JobFile = 'dispatcherJob.sh'
        with open(os.path.join(templatesFolder, jobTemplate),'r') as f:
            JobText = f.read()

        JobText = JobText.replace('<JobName>', 'dispatcher')
        JobText = JobText.replace('<AERnumber>', config.AER)
        JobText = JobText.replace('#SBATCH -t 0-15:00', '#SBATCH -t 0-0:30')
        JobText = JobText.replace('<NProcs>', '1')
        JobText = JobText.split('mpirun ')[0]

        with open(JobFile,'w+') as f:
            f.write(JobText)
            f.write('cd '+DIRECTORY_DISPATCHER+'\n')
            f.write('python dispatch.py 1>Dispatch-out.log 2>Dispatch-err.log\n')

        os.chmod(JobFile, 0o777)
        ServerTools.cpmvWrap4MultiServer('cp',JobFile,
                                     os.path.join(DIRECTORY_DISPATCHER,JobFile))

        # 2 - launch job with sbatch
        launchDispatcherJob(DIRECTORY_DISPATCHER, JobFile, machine=config.machine)

    print('submitted DISPATCHER files and launched dispatcher job')

def remoteFileExists(absolute_file_path, remote_machine='sator'):

    if remote_machine not in ('sator','spiro'):
        return os.path.isfile(absolute_file_path)

    HostName = socket.gethostname()
    UserName = getpass.getuser()

    CMD='[ -f %s ] && echo 1 || echo 0'%absolute_file_path
    CMD = 'ssh '+UserName+"@"+remote_machine+' '+CMD
    ssh = subprocess.Popen(CMD, shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    ssh.wait()
    Output = ServerTools.readStdout(ssh)
    Error = ServerTools.readStderr(ssh)
    if len(Error) >0: return False
    return bool(int(Output[0]))


def launchComputationJob(case, config, JobFilename='job.sh',
                         submitReserveJob=False):
    '''
    This function is designed to be called from ``dispatch.py`` file
    It launches the computation job.

    Parameters
    ----------

        case : dict
            current item of **JobsQueues** list

        config : module
            import of ``JobsConfiguration.py``

        JobFilename : str
            name of the job file

        submitReserveJob : bool
            If :py:obj:`True`, sends the job twice (one awaits in reserve)

    '''
    HostName = socket.gethostname()
    UserName = getpass.getuser()
    DIRECTORY_JOB = os.path.join(config.DIRECTORY_WORK, case['JobName'])

    CMD = 'sbatch '+JobFilename
    cwd = DIRECTORY_JOB

    if config.machine not in HostName:
        InSator = True if os.getcwd().startswith('/tmp_user/sator') else False
        if not InSator:
            Host = UserName+"@"+machine
            CMD = '"cd %s; %s"'%(DIRECTORY_JOB, CMD)
            CMD = 'ssh %s %s'%(Host,CMD)
            cwd = None

    ssh = subprocess.Popen(CMD, shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, cwd=cwd)

    ssh.wait()
    Output = ServerTools.readStdout(ssh)
    Error = ServerTools.readStderr(ssh)

    for o in Output: print(o)
    for e in Error: print(e)

    if submitReserveJob:
        CMD = 'sbatch %s --dependency=singleton'%(JobFilename)

        if not InSator:
            Host = UserName+"@"+machine
            CMD = '"cd %s; %s"'%(DIRECTORY_JOB, CMD)
            CMD = 'ssh %s %s'%(Host,CMD)
            cwd = None

        ssh = subprocess.Popen(CMD, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, cwd=cwd)

        ssh.wait()
        Output = ServerTools.readStdout(ssh)
        Error = ServerTools.readStderr(ssh)

        for o in Output: print(o)
        for e in Error: print(e)

def launchDispatcherJob(DIRECTORY_DISPATCHER, JobFilename, machine='sator'):
    '''
    Launch the dispatcher job (containing call of`` dispatch.py`` file)

    Parameters
    ----------

        DIRECTORY_DISPATCHER : str
            path where the ``DISPATCHER`` directory is
            located (contains ``dispatch.py`` and it is the location where
            meshes are produced)

        JobFilename : str
            Name of the job that executes ``dispatch.py``.
            In this workflow, it is named ``dispatcherJob.sh``

        machine : str
            The name of the employed machine where computations are
            launched

    '''
    HostName = socket.gethostname()
    UserName = getpass.getuser()
    CMD = '"cd %s; sbatch %s"'%(DIRECTORY_DISPATCHER, JobFilename)

    if machine not in HostName:
        SatorProd = True if os.getcwd().startswith('/tmp_user/sator') else False
        if not SatorProd:
            Host = UserName+"@"+machine
            CMD = 'ssh %s %s'%(Host,CMD)

    print(CMD)

    ssh = subprocess.Popen(CMD, shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    ssh.wait()
    Output = ServerTools.readStdout(ssh)
    Error = ServerTools.readStderr(ssh)

    for o in Output: print(o)
    for e in Error: print(e)

def putFilesInComputationDirectory(case):
    '''
    This function is designed to be called from ``dispatch.py``.

    Send required files from ``DISPATCH`` directory towards relevant computation
    directory.

    Parameters
    ----------

        case : dict
            current item of **JobsQueues** list
    '''

    Items2Copy = ('preprocess.py','compute.py','coprocess.py',
                  'postprocess.py','meshParams.py')
    Items2Move = ('setup.py','main.cgns','OUTPUT')

    DIRECTORY_JOB = '../%s'%case['JobName']
    DIRECTORY_CASE = os.path.join(DIRECTORY_JOB,case['CASE_LABEL'])

    ServerTools.cpmv('cp', 'job.sh', os.path.join(DIRECTORY_JOB,'job.sh'))

    for item in Items2Copy:
        ServerTools.cpmv('cp', item, os.path.join(DIRECTORY_CASE,item) )

    for item in Items2Move:
        if item == 'OUTPUT' and not case['NewJob']: continue
        ServerTools.cpmv('mv', item, os.path.join(DIRECTORY_CASE,item) )

def getJobsConfiguration(DIRECTORY_WORK, useLocalConfig=False,
        filename='JobsConfiguration.py'):
    '''
    Get a possibly remotely located JobsConfiguration.py file and
    load it as a module object *config*.

    Parameters
    ----------

        DIRECTORY_WORK : str
            directory where ``JobsConfiguration.py`` file is located

        useLocalConfig : bool
            if :py:obj:`True`, use the local ``JobsConfiguration.py``
            file instead of retreiving it from **DIRECTORY_WORK**

        filename : str
            Name of the configuration file. By default, it is
            ``JobsConfiguration.py``

    Returns
    -------

        config : module
            ``JobsConfiguration.py`` data as a mobule object
    '''
    if not useLocalConfig:
        Source = os.path.join(DIRECTORY_WORK,'DISPATCHER',filename)
        repatriate(Source, filename, removeExistingDestinationPath=True)

    config = J.load_source('config', filename)

    return config

def loadJobsConfiguration():
    config = J.load_source('config', 'JobsConfiguration.py')
    return config


def statusOfCase(config, CASE_LABEL):
    '''
    Get the status of a possibly remote CFD case.

    Parameters
    ----------

        config : module
            ``JobsConfiguration.py`` data as a mobule object as
            returned by function :py:func:`getJobsConfiguration`

        CASE_LABEL : str
            unique identifying label of the requested case

    Returns
    -------

        status : str
            can be one of:
            ``'COMPLETED'``  ``'FAILED'``  ``'TIMEOUT'``  ``'RUNNING'``
            ``'PENDING'``
    '''
    JobTag = '_'.join(CASE_LABEL.split('_')[1:])
    DIRECTORY_CASE = os.path.join(config.DIRECTORY_WORK, JobTag, CASE_LABEL)

    HostName = socket.gethostname()
    UserName = getpass.getuser()
    CMD = 'ls '+DIRECTORY_CASE

    if config.machine not in HostName:
        SatorProd = True if os.getcwd().startswith('/tmp_user/sator') else False
        if not SatorProd:
            Host = UserName+"@"+config.machine
            CMD = 'ssh %s %s'%(Host,CMD)

    ssh = subprocess.Popen(CMD, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=os.environ.copy())
    ssh.wait()
    Output = ServerTools.readStdout(ssh)
    Output = [o.replace('\n','') for o in Output]
    Error = ServerTools.readStderr(ssh)

    if len(Error) > 0: raise ValueError('\n'.join(Error))

    if 'COMPLETED' in Output:
        return 'COMPLETED'

    if 'FAILED' in Output:
        return 'FAILED'

    for o in Output:
        if o.startswith('core') or o.startswith('elsA.x'):
            return 'TIMEOUT'

    if 'coprocess.log' in Output:
        return 'RUNNING'

    return 'PENDING'

def getCurrentJobsStatus(machine='sator'):
    '''
    This function is literally a wrap of command

    .. code-block:: console

        squeue -l -u <username>

    to be launched to a machine defined by the user.
    It shows the queue job status in terminal and returns it as a string.

    Parameters
    ----------

        machine : str
            the machine (possibly remote) where the status command is to be launched

    Returns
    -------

        Output : :py:class:`list` of :py:class:`str`
            standard output of provided command

        Error : :py:class:`list` of :py:class:`str`
            standard error of provided command
    '''
    HostName = socket.gethostname()
    UserName = getpass.getuser()
    CMD = 'squeue -l -u '+UserName

    if machine not in HostName:
        SatorProd = True if os.getcwd().startswith('/tmp_user/sator') else False
        if not SatorProd:
            Host = UserName+"@"+machine
            CMD = 'ssh %s %s'%(Host,CMD)

    ssh = subprocess.Popen(CMD, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=os.environ.copy())
    ssh.wait()
    Output = ServerTools.readStdout(ssh)
    Error = ServerTools.readStderr(ssh)
    for o in Output: print(o)

    return Output, Error

def getCaseArrays(config, CASE_LABEL, basename='AIRFOIL'):
    '''
    Repatriate the remote ``OUTPUT/arrays.cgns`` file and return its contents in a
    form of dictionary like :

    ::

            {'CL':float,
             'CD':float,
                ...    }

    .. note::  we only keep the last iteration of each integral value contained
        in ``arrays.cgns``

    Parameters
    ----------

        config : module
            ``JobsConfiguration.py`` data as a mobule object as
            obtained from :py:func:`getPolarConfiguration`

        CASE_LABEL : str
            unique identifying label of the requested case.
            Can be determined by :py:func:`getCaseLabelFromAngleOfAttackAndMach`

        basename : str
            Name of the base to read in ``arrays.cgns``

    Returns
    -------

        ArraysDict : dict
            containts the airfoil arrays
    '''
    FILE_ARRAYS = 'arrays.cgns'
    JobTag = '_'.join(CASE_LABEL.split('_')[1:])

    Source = os.path.join(config.DIRECTORY_WORK, JobTag, CASE_LABEL, 'OUTPUT',
                                                                    FILE_ARRAYS)

    try:
        repatriate(Source, FILE_ARRAYS, removeExistingDestinationPath=True)
    except:
        print(J.WARN+'could not retrieve arrays.cgns of case %s'%CASE_LABEL+J.ENDC)
        return

    ArraysTree = C.convertFile2PyTree(FILE_ARRAYS)
    ArraysZone = I.getNodeFromName2(ArraysTree, basename)

    ArraysDict = J.getVars2Dict(ArraysZone,
                               C.getVarNames(ArraysZone,excludeXYZ=True)[0])

    for v in ArraysDict: ArraysDict[v] = ArraysDict[v][-1]

    return ArraysDict

def repatriate(SourcePath, DestinationPath, removeExistingDestinationPath=True,
               moveInsteadOfCopy=False, wait4FileFromServerOptions={}):
    '''
    Repatriate a file or directory (SourcePath) towards a destination location
    (DestinationPath) raising an error if after some timeout the repatriated
    source is not detected in the destination path.


    INPUTS

    SourcePath - (string) - path string of the source to be repatriated.
        May correspond to a directory or a file.

    DestinationPath - (string) - path string of the destination where the source
        will be repatriated. If it makes reference to an inexistent directory,
        then all required paths are automatically created in order to
        satisfy the destination path (if permissions allow for it).

    removeExistingDestinationPath - (boolean) -  Use True for removing any
        pre-existing item at DestinationPath. This takes more time, but is safer
        for multiserver communications, as it avoids confusion between the newly
        copied item and the pre-existing one when evaluating if repatriation is
        correctly performed. This may be caused due to latency of directory
        updates between servers when making file operations.

    moveInsteadOfCopy - (boolean) - if True, the repatration operation will
        remove the original source (move operation, instead of copy).

    wait4FileFromServerOptions - (Python dictionary) - Options to be passed to
        wait4FileFromServer() function, which by default are:
                        dict(requestInterval=0.5, timeout=60.)
        This is used for determine if repatration succeeded
    '''
    Defaultwait4FileFromServerOptions = dict(requestInterval=0.5,
                                             timeout=60.,)
    Defaultwait4FileFromServerOptions.update(wait4FileFromServerOptions)


    if removeExistingDestinationPath:
        print( 'Removing pre-existing {} ...'.format(DestinationPath) )
        ServerTools.cpmvWrap4MultiServer('rm', DestinationPath, 'none')

    if moveInsteadOfCopy:
        print( 'Repatriating {} by MOVE...'.format(SourcePath) )
        ServerTools.cpmvWrap4MultiServer('mv', SourcePath, DestinationPath)
    else:
        print( 'Repatriating {} by COPY...'.format(SourcePath) )
        ServerTools.cpmvWrap4MultiServer('cp', SourcePath, DestinationPath)

    print( 'Waiting for {} ...'.format(DestinationPath) )
    gotFile = ServerTools.wait4FileFromServer(DestinationPath,
                               **Defaultwait4FileFromServerOptions)

    if not gotFile:
        MSG = J.FAIL+"Could not repatriate "+DestinationPath+J.ENDC
        raise IOError(MSG)
    print(J.GREEN+'ok'+J.ENDC)


def fileExists(*path): return os.path.isfile(os.path.join(*path))

def anyFile(*path): return any(glob.glob(os.path.join(*path)))
