'''
MOLA - XFoil.py

XFoil - Python wrapper

Collection of Python functions designed for batch execution of
XFoil of M. Drela (https://web.mit.edu/drela/Public/web/xfoil/)

A local installation of xfoil is required.

File history:
23/05/2020 - v1.8 - L. Bernardos - Shipped to MOLA v1.8
19/04/2020 - v1.7 - L. Bernardos - Shipped to MOLA v1.7
03/04/2020 - v1.6.01 - L. Bernardos - Coherent with MOLA v1.6,
    added PyZonePolarKind argument in computePolar() and
    adapted convertXFoilDict2PyZonePolar accordingly
26/03/2020 - v0.3 - L. Bernardos - added robust mode
24/03/2020 - v0.2 - L. Bernardos - 1st fully functional version
20/03/2020 - v0.1 - L. Bernardos - Creation
'''

import sys
import os
import re
import subprocess
import glob
import copy
import timeit
import numpy as np

XFOIL_EXEC = '/home/lbernard/TOOLS/XFOIL/visio/xfoil' # to be adapted for each machine

# Global variables
PolarVariables = ['AoA','Cl', 'Cd', 'Cdp', 'Cm', 'Top_Xtr', 'Bot_Xtr']
FoilVariables = ['Cp', 's', 'x', 'y', 'Ue_over_Vinf', 'delta1', 'theta', 'Cf', 'H', 'Hstar','P','m','K']
translateDistributionVariable = {
    'Ue/Vinf':'Ue_over_Vinf',
    'Dstar'  :'delta1',
    'Theta'  :'theta',
    'H*'  :'Hstar',
}

def scan__(line,OutputType=float, RegExpr=r'[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?'):
    '''
    This is an internal function employed for scanning numbers
    contained in plain-text files.

    It is used in XFoil.readBlFile(), for example.
    '''
    scanned = re.findall(RegExpr,line)
    return [OutputType(item) for item in scanned]



def computePolars(airfoilFilename, Reynolds, Mach, AoAs, Ncr=9,
    PyZonePolarKind='Struct_AoA_Mach', itermax=60, LogStdOutErr=False, lowMachForced=0.3, storeFoilVariables=True, rediscretizeFoil=False, removeTmpXFoilFiles=True):
    '''
    This function launches XFoil polars of a list of specified
    airfoils filenames for specified set of angles of attack for
    each pair of Reynolds and Mach numbers (optionally also Ncr).

    Hence, the number of Reynolds points must be identical to the
    number of Mach numbers and optionally Ncr.

    INPUTS:

    airfoilFilename (str) - Name of the file containing the
        airfoil's coordinates in two-column format (x,y).

        example: 'OA209.dat'

    Reynolds (list, 1D numpy array)  - Vector containing all the
        Reynolds number to be computed for the polars. This vector
        is used with Mach vector (each set of pair elements set a
        sweep run of angle of attack).

        example: [1e6,1.5e6]

    Mach (list, 1D numpy array) - Vector containing all the Mach
        numbers to be computed for the polars. This vector
        is used with Reynolds vector (each set of pair elements
        set a sweep run of angle of attack).

        example: [0.6,0.7]

    AoAs (list, 1D numpy array) - Vector containing all the
        angles of attack used for computing the polar.

        example: np.linspace(0,15,10)

    Ncr (float) - Value of the total amplification factor used for
        computing the transition onset of the boundary layer.

        example: 9

    itermax (integer) - Maximum number of iterations for the
        viscid/inviscid coupling used in XFoil.

        example: 200

    LogStdOutErr (boolean) - Flag to indicate whether to write
        log standard output and error files or not. Such files
        are labeled incrementally for each run of XFoil command.

        example: True

    lowMachForced (float) - This value is used establishes
        the threshold used for incompressible flow computation.
        This means, for any requested Mach number lower than
        lowMachForced, the value sent to XFoil is exactly 0,
        which means that fully incompressible computation is done.

        example: 0.3

    storeFoilVariables (boolean) - Flag to indicate to store the
        boundary-layer quantities and Cp distributions.

        example: True

    rediscretizeFoil (boolean) - if True, then the points of the
        input airfoil are relocated following a curvature
        criterion. This makes use of the function PANE of XFOIL.

    removeTmpXFoilFiles (boolean) - if True, then all files
        created by the XFoil runs labeled "tmpXFoil_*" in the
        working directory will be deleted if the Results output
        dictionary is correctly created.

        example: True


    OUTPUTS:

    Results (Python dictionary) - in this Python dictionary, all
        the results of the XFoil runs are stored. Each variable
        is accessible through the dictionary's keys and their
        labels are self-explanatory.

        For example, if user wants to access the results of the
        lift coefficient, then he/she should use the following
        command:

        Results['Cl'].

        Use the command Results.keys() to see a full list of
        available variables.

        ---------------HOW VARIABLES ARE STORED---------------

        Polar data variables (such as Cl, Cd, Cm...) are stored
        as 2D matrices, where i-index (rows) indicates the
        angle-of-attack and j-index (columns) indicates both
        Mach and Reynolds numbers.

        For example, let us consider a case where user requests
        the following input arguments:

        Reynolds = [ 1e5,  2e5, 3e5]
        Mach     = [ 0.1,  0.5, 0.6]
        AoAs     = [0, 2, 4, 6, 8, 10]

        then, when user requests Results['Cl'], the expected
        matrix is organized as a 6x3 2D matrix, as follows:

                    |               Cl values
        ROWS (AoA)  |              (2D matrix)
        ____________|________________________________________
             0      |    value        value        value
                    |
             2      |    value        value        value
                    |
             4      |    value        value        value
                    |
             6      |    value        value        value
                    |
             8      |    value        value        value
                    |
            10      |    value        value        value
        ---------------------------------------------------
        COLUMNS     |
          Reynolds  |     1e5          2e5          3e5
          and       |--------------------------------------
          Mach      |     0.1          0.5          0.6
        ---------------------------------------------------

        Hence, for accessing the lift-coefficient at AoA=6
        with Mach=0.6 (and Reynolds = 3e5), then user shall
        use the command:

        Results['Cl'][3,2] # fourth row (AoA= 6 deg) and
                           # third column (Re=3e5, Mach=.6)


        Airfoil-data variables (such as Cp, theta, Cf...) are
        stored in a similar fashion as polar-data variables.
        Instead of storing real numbers in the 2D matrix,
        vectors are used. This means that such variables are
        3D matrices of shape (number_AoAs x number_Re x
        number_Points_of_Airfoil). Using the previous example
        the structure of Results['Cp'] is the following:

                    |               Cp values
        ROWS (AoA)  |              (3D matrix)
        ____________|________________________________________
             0      |    vector       vector       vector
                    |
             2      |    vector       vector       vector
                    |
             4      |    vector       vector       vector
                    |
             6      |    vector       vector       vector
                    |
             8      |    vector       vector       vector
                    |
            10      |    vector       vector       vector
        ---------------------------------------------------
        COLUMNS     |
          Reynolds  |     1e5          2e5          3e5
          and       |--------------------------------------
          Mach      |     0.1          0.5          0.6
        ---------------------------------------------------

        Hence, all vectors have the same length. The number
        of elements of each vector is equal to the number of
        points of the input airfoil.

        Let us consider the user wants to plot the Cp(x)
        distribution of the airfoil for AoA=6 deg and
        Re=3e5 (and Mach=.6). The following commands could
        be employed:

        import matplotlib.pyplot as plt
        x  =  Results['x'][3,2,:]
        Cp = Results['Cp'][3,2,:]
        plt.plot(x,Cp); plt.show()

    '''


    # Make input coherency verifications
    nRe  = len(Reynolds)
    nM   = len(Mach)
    nAoA = len(AoAs)
    if nRe == nM == nAoA == 1: PyZonePolarKind='Unstr_AoA_Mach_Reynolds'
    if nRe != nM:
        raise AttributeError('The quantity of Reynolds (%d) differs from the quantity of Mach (%d). They must be equal.'%(nRe,nM))

    if PyZonePolarKind != 'Struct_AoA_Mach':
        if nAoA != nRe:
            raise AttributeError('For PyZonePolarKind=%s, number of elements of Reynolds (%d), Mach (%d) and angle-of-attack (%d) must ALL be equal.'%(PyZonePolarKind,nRe,nM,nAoA))
        PyZonePolarKind = 'Unstr_AoA_Mach_Reynolds'


    # Prepare containers where XFoil results will be stored in
    Results = dict(PyZonePolarKind=PyZonePolarKind)
    AllVariables = PolarVariables+FoilVariables if storeFoilVariables else PolarVariables
    # Stores Reynolds and Mach number ranges
    if type(Reynolds) != np.ndarray: Reynolds = np.array(Reynolds, dtype=np.float64, order='F')
    if type(Mach) != np.ndarray: Mach = np.array(Mach, dtype=np.float64, order='F')
    if type(AoAs) != np.ndarray: AoAs = np.array(AoAs, dtype=np.float64, order='F')
    if PyZonePolarKind == 'Struct_AoA_Mach':
        Results['Reynolds'] = Reynolds
        Results['Mach'] = Mach
        for var in PolarVariables:
            Results[var] = np.zeros((nAoA, nM),dtype=np.float64,order='F')
        if storeFoilVariables:
            for var in FoilVariables:
                MegaList = Results[var] = []
                for a in AoAs:
                    MegaList += [[]]
                    for m in Mach:
                        MegaList[-1] += [[]]

        # Build macro-vectors of parameters
        ReVec = np.broadcast_to(Reynolds,(nAoA,nRe)).T.flatten()
        MVec  = np.broadcast_to(Mach,    (nAoA, nM)).T.flatten()
        AoAVec= np.broadcast_to(AoAs,    (nM, nAoA)).flatten()
    else:
        Results['Reynolds'] = []
        Results['Mach']     = []
        for var in PolarVariables:
            Results[var] = []
        if storeFoilVariables:
            for fvar in FoilVariables:
                Results[fvar] = []
        # Build macro-vectors of parameters
        ReVec = Reynolds.flatten()
        MVec  = Mach.flatten()
        AoAVec= AoAs.flatten()

    # Make as many XFoil runs as sweeps of AoA
    NComps = len(ReVec)
    nrun  = -1 # current Xfoil run
    icomp = 0  # current computation iterator
    while icomp <= NComps-1:
        nrun  += 1

        # Build input XFoil commands and stores them in a file
        AirfoilLabel = airfoilFilename.replace(' ','')
        if NComps==1:
            stdin_fn = 'tmpXfoilSingle_%s_stdin_run%d'%(AirfoilLabel,nrun)
        else:
            stdin_fn = 'tmpXfoil_%s_stdin_run%d'%(AirfoilLabel,nrun)
        if os.path.isfile(stdin_fn): os.remove(stdin_fn)

        f = open(stdin_fn,'w')

        #  Read airfoil
        f.write(airfoilFilename+'\n')

        # Eventually rediscretize airfoil based on curvature
        # keeping the total number of total points
        if rediscretizeFoil: f.write('pane\n')

        # Set-up numerical and physical parameters
        f.write('oper\n')
        f.write('v\n')

        # Set Reynolds number
        f.write('%g\n'%ReVec[icomp])

        # Set Mach number
        VirtualMach = 0 if MVec[icomp] <= lowMachForced else MVec[icomp]
        f.write('M %g\n'%VirtualMach)
        f.write('iter %d\n'%itermax)

        # Compute transition, set critical total
        # amplification factor
        f.write('vpar\n')
        f.write('n %g\n\n'%Ncr)

        # Store polar and set angle of attack
        f.write('pacc\n')
        PolarFilename = stdin_fn.replace('stdin','polar_Re%g_M%g'%(ReVec[icomp],MVec[icomp]))
        if os.path.isfile(PolarFilename): os.remove(PolarFilename)
        f.write(PolarFilename+'\n\n')


        CpFilenames, BlFilenames, LastRunAoAs = [], [], []
        NoReChange = NoMachChange = True
        while NoReChange and NoMachChange:
            f.write('a %g\n'%AoAVec[icomp])
            LastRunAoAs += [AoAVec[icomp]]

            # Store additional quantities, if wished
            if storeFoilVariables:
                CpFilename = stdin_fn.replace('stdin','Cp_Re%g_M%g_AoA%g'%(ReVec[icomp],MVec[icomp],AoAVec[icomp]))
                if os.path.isfile(CpFilename): os.remove(CpFilename)
                f.write('cpwr %s\n'%CpFilename)
                CpFilenames += [CpFilename]

                BlFilename = stdin_fn.replace('stdin','Bl_Re%g_M%g_AoA%g'%(ReVec[icomp],MVec[icomp],AoAVec[icomp]))
                if os.path.isfile(BlFilename): os.remove(BlFilename)
                f.write('dump %s\n'%BlFilename)
                BlFilenames += [BlFilename]

            if icomp == NComps-1:
                icomp += 1
                break
            else:
                NoReChange = ReVec[icomp] == ReVec[icomp+1]
                NoMachChange = MVec[icomp] == MVec[icomp+1]

                icomp +=1

        # Quit XFoil - end of run
        f.write('\n\n\n\n\nquit\n')
        f.close()

        LastRunAoAs = np.array(LastRunAoAs,dtype=np.float64,order='F')
        # Log files of standard output and errors
        if LogStdOutErr:
            stdout = open(stdin_fn.replace('stdin','stdout'),'w')
            stderr_fn = stdin_fn.replace('stdin','stderr')
            stderr = open(stderr_fn,'w')
        else:
            stderr = stdout = open(os.devnull, 'wb')

        # Launch XFoil
        stdin  = open(stdin_fn,'r')
        proc = subprocess.call([''],
            executable=XFOIL_EXEC,
            stdin=stdin,stdout=stdout,stderr=stderr)

        # Read polar
        AllRunFailed = True
        if os.path.isfile(PolarFilename):
            data = np.loadtxt(PolarFilename, comments='#', unpack=False, skiprows=12)
            if len(data)>0:
                AllRunFailed = False
                if len(data.shape)==1: data = np.broadcast_to(data,(1,len(data)))

        tol = 1e-6 # used to determine if AoA had converged
        if PyZonePolarKind == 'Struct_AoA_Mach':
            if not AllRunFailed:
                for iAoA in range(len(LastRunAoAs)):
                    AoAConverged = False
                    for jAoA in range(len(data[:,0])):
                        if data[jAoA,0] < (LastRunAoAs[iAoA]) + tol and \
                           data[jAoA,0] > (LastRunAoAs[iAoA]) - tol:
                            AoAConverged = True
                            break

                    # Attempt single-shot for not converged
                    if not AoAConverged:
                        SingRun = computePolars(airfoilFilename, [ReVec[icomp-1]], [MVec[icomp-1]], [LastRunAoAs[iAoA]], Ncr=Ncr,  PyZonePolarKind='Unstr_AoA_Mach_Reynolds', itermax=itermax, LogStdOutErr=False, lowMachForced=lowMachForced, storeFoilVariables=storeFoilVariables, rediscretizeFoil=rediscretizeFoil, removeTmpXFoilFiles=False)
                        for var in SingRun:
                            if var in PolarVariables:
                                Results[var][iAoA,nrun] += [SingRun[var][0]]
                            elif var in FoilVariables:
                                Results[var][iAoA][nrun] += [SingRun[var][0].tolist()]

                    else:
                        for ivar in range(len(PolarVariables)):
                            Results[PolarVariables[ivar]][iAoA,nrun] = data[jAoA,ivar]

                        # Read airfoil variables
                        if storeFoilVariables:
                            for fvar in FoilVariables:
                                if AoAConverged:
                                    # Store Cp, Bl...
                                    CpFilename = stdin_fn.replace('stdin','Cp_Re%g_M%g_AoA%g'%(ReVec[icomp-1],MVec[icomp-1],data[jAoA,0]))
                                    dataCp = np.loadtxt(CpFilename, comments='#', unpack=True, skiprows=2)
                                    Cp = Results['Cp'][iAoA][nrun] = dataCp[1].tolist()
                                    foilNPts = len(Cp)

                                    BlFilename = stdin_fn.replace('stdin','Bl_Re%g_M%g_AoA%g'%(ReVec[icomp-1],MVec[icomp-1],data[jAoA,0]))
                                    dataBl = readBlFile(BlFilename)
                                    for varBl in dataBl:
                                        try: varRes = translateDistributionVariable[varBl]
                                        except KeyError: varRes = varBl
                                        Results[varRes][iAoA][nrun] = dataBl[varBl][:foilNPts].tolist()

                                else:
                                    Results[fvar][iAoA][nrun] = np.nan
            else:
                print('All RUN failed') # TODO: Boundary-layer with NaN ??
                for var in PolarVariables: Results[var][:,nrun] = np.nan
                if storeFoilVariables:
                    for fvar in FoilVariables:
                        for iAoA in range(len(AoAs)):
                            Results[fvar][iAoA][nrun] = np.nan
        else:
            if not AllRunFailed:
                for iAoA in range(len(LastRunAoAs)):
                    AoAConverged = False
                    for jAoA in range(len(data[:,0])):
                        if data[jAoA,0] < (LastRunAoAs[iAoA]) + tol and \
                           data[jAoA,0] > (LastRunAoAs[iAoA]) - tol:
                            AoAConverged = True
                            break

                    # Attempt single-shot for not converged
                    if not AoAConverged and len(LastRunAoAs)>1:
                        SingRun = computePolars(airfoilFilename, [ReVec[icomp-1]], [MVec[icomp-1]], [LastRunAoAs[iAoA]], Ncr=Ncr,    PyZonePolarKind='Unstr_AoA_Mach_Reynolds', itermax=itermax, LogStdOutErr=False, lowMachForced=lowMachForced, storeFoilVariables=storeFoilVariables, rediscretizeFoil=rediscretizeFoil, removeTmpXFoilFiles=False)
                        if not np.isnan(SingRun['Cl'][0]):
                            for var in AllVariables:
                                Results[var] += [SingRun[var][0]]
                            Results['Reynolds'] += [ReVec[icomp-1]]
                            Results['Mach']     += [MVec[icomp-1]]
                    else:
                        Results['Reynolds'] += [ReVec[icomp-1]]
                        Results['Mach']     += [MVec[icomp-1]]

                        for ivar in range(len(PolarVariables)):
                            Results[PolarVariables[ivar]] += [data[jAoA,ivar]]

                        if storeFoilVariables:
                            CpFilename = stdin_fn.replace('stdin','Cp_Re%g_M%g_AoA%g'%(ReVec[icomp-1],MVec[icomp-1],data[jAoA,0]))
                            dataCp = np.loadtxt(CpFilename, comments='#', unpack=True)
                            Results['Cp'] += [dataCp[1]]
                            foilNPts = len(dataCp[1])

                            BlFilename = stdin_fn.replace('stdin','Bl_Re%g_M%g_AoA%g'%(ReVec[icomp-1],MVec[icomp-1],data[jAoA,0]))
                            dataBl = readBlFile(BlFilename)
                            for varBl in dataBl:
                                try: varRes = translateDistributionVariable[varBl]
                                except KeyError: varRes = varBl
                                Array  = dataBl[varBl][:foilNPts]
                                Results[varRes] += [Array]
            elif NComps == 1:
                for var in AllVariables:
                    Results[var] = [np.nan]


    if PyZonePolarKind == 'Struct_AoA_Mach':
        # Cleaning-up not converged results
        # Detect non-converged simulations (containing NaN)
        NaNElts = np.isnan(Results['AoA'])

        # Broadcast foil variables of NaN
        if storeFoilVariables:
            for iAoA in range(nAoA):
                for irun in range(nM):
                    for fvar in FoilVariables:
                        if len(Results[fvar][iAoA][irun]) != foilNPts:
                            NaNvec = np.empty(foilNPts)
                            NaNvec[:] = np.nan
                            Results[fvar][iAoA][irun] = NaNvec.tolist()

            for var in FoilVariables:
                Results[var] = np.array(Results[var],dtype=np.float64,order='F')

        # Determine fully non-converged runs
        RemoveRuns = np.all(NaNElts,axis=0)
        KeepRuns = np.logical_not(RemoveRuns)

        # Remove integral data fully non-converged runs
        Runs2RemoveIndices = np.where(RemoveRuns)[0]
        for var in AllVariables:
            Results[var] = np.delete(Results[var],Runs2RemoveIndices,axis=1)

        # Stores final Reynolds and final Mach
        Results['Reynolds'] = Reynolds[KeepRuns]
        Results['Mach'] = Mach[KeepRuns]

        # Determine rows to delete (some non-converged AoA)
        NaNElts = np.isnan(Results['AoA']) # Update NaNElts
        RemoveAoA = np.any(NaNElts,axis=1)

        # Remove integral data where any non-converged AoA exist
        AoA2RemoveIndices = np.where(RemoveAoA)[0]
        for var in AllVariables:
            Results[var] = np.delete(Results[var],AoA2RemoveIndices,axis=0)

    else:
        for var in PolarVariables+['Mach','Reynolds']:
            Results[var] = np.array(Results[var],dtype=np.float64, order='F')
        if storeFoilVariables:
            for fvar in FoilVariables:
                Results[fvar] = np.vstack(Results[fvar])

    if removeTmpXFoilFiles:
        fileList = glob.glob('./tmpXfoil*')
        for filePath in fileList:
            try:
                os.remove(filePath)
            except OSError:
                print("Error while deleting file %s"%filePath)

    # Remove spurious XFoil file
    try: os.remove(":00.bl")
    except: pass

    return Results


def readBlFile(BoundaryLayerFile):
    '''
    Read the XFoil boundary-layer file produced with XFoil's
    internal command "DUMP".

    Each column of the file is read and transformed into a
    numpy 1D vector.

    INPUTS

    BoundaryLayerFile (string) - The name of the file to be
        read.

        example: 'MyBLdata.dat'

    OUTPUTS

    OutputDict (dictionary) - Python dictionary containing 1D numpy vectors
        of each variable contained in the input file (column).
    '''

    with open(BoundaryLayerFile,'r') as f:
        lines = f.readlines()

    variables = lines[0].split()[1:]

    rows = []
    for l in lines[1:]:
        row = scan__(l,float)
        rows += [row]

    nCols = len(rows[0])
    # if nCols != len(variables):
    #     raise ValueError('File %s incoherency'%BoundaryLayerFile)
    OutputDict = dict()
    for c in range(nCols):
        Col = []
        for r in rows:
            current_nCols = len(r)
            if c < current_nCols:
                Col += [r[c]]
        array = np.array(Col,order='F')
        if np.isnan(array).any():
            print('NaN detected in %s'%BoundaryLayerFile)
        OutputDict[variables[c]] = array

    return OutputDict


def convertXFoilDict2File(ResultsDictionary, outputfilename):
    '''
    This function converts the Python-dictionary [result
    of the call of function XFoil.computePolars()] into
    a file with a format allowed by Cassiopee's Converter
    module. HOWEVER, the recommended file extension is CGNS.

    Example:
    XFoil.convertXFoilDict2File(Results, 'MyPolarData.cgns')

    INPUTS

    ResultsDictionary (Python Dictionary) - Exactly the
        result of the function XFoil.computePolars().

    outputfilename (string) - Filename of the file where
        the polar data will be saved.
    '''


    # This function requires some additional dependencies:
    import Converter.PyTree as C
    from . import InternalShortcuts as J

    # Save Polar sweep Ranges
    RangesVars = ['Reynolds','Mach']
    Arrays = [ResultsDictionary[var] for var in RangesVars]
    RangesData = J.createZone('RangesData',Arrays, RangesVars)
    zones = [RangesData]

    # Save Polar data (Cl, Cd, Cm,...)
    Arrays = [ResultsDictionary[var] for var in PolarVariables]
    PolarData = J.createZone('PolarData',Arrays, PolarVariables)

    zones += [PolarData]

    # Save airfoil data, if any (Cp, Cf...)
    if 'Cp' in ResultsDictionary:
        Arrays = [ResultsDictionary[var] for var in FoilVariables]

        FoilData = J.createZone('FoilData',Arrays, FoilVariables)

        zones += [FoilData]

    # Save data
    C.convertPyTree2File(zones, outputfilename)



def convertFile2XFoilDict(filename):
    '''
    This function converts an existing CGNS file containing
    XFoil computation results, into a simple Python-dictionary
    as produced by XFoil.computePolars() function.

    INPUT

    filename (string) - CGNS file containing polar data.

    OUTPUT

    Results (Python dictionary) - The structure is identical
        to the output of XFoil.computePolars() function (see
        doc of computePolars() for detailed information)
    '''

    # This function requires some additional dependencies:
    import Converter.PyTree as C
    import Converter.Internal as I
    from . import InternalShortcuts as J

    # Reads data
    t = C.convertFile2PyTree(filename)

    Results = {} # Here results will be stored

    for z in I.getZones(t):
        FS_n = I.getNodeFromName1(z,'FlowSolution')
        for fs in FS_n[2]:
            Results[fs[0]] = I.getValue(fs)

        x_n = I.getNodeFromName2(z,'CoordinateX')
        if x_n is not None: Results['x'] = I.getValue(x_n)

        y_n = I.getNodeFromName2(z,'CoordinateY')
        if y_n is not None: Results['y'] = I.getValue(y_n)

    return Results


def convertXFoilDict2PyZonePolar(ResultsDictionary, Title, Variables2Store=[],
     BigAngleOfAttackAoA=[-180, -160, -140, -120, -100, -80, -60, -40,-30,
                           30, 40, 60, 80, 100, 120, 140, 160, 180],
     BigAngleOfAttackCl=[ 0, 0.73921, 1.13253, 0.99593, 0.39332, -0.39332,
                         -0.99593, -1.13253, -0.99593, 0.99593, 1.13253,
                         0.99593, 0.39332, -0.39332, -0.99593, -1.13253,
                         -0.73921, 0],
     BigAngleOfAttackCd=[4e-05, 0.89256, 1.54196, 1.94824,  2.1114,  2.03144,
                         1.70836,  1.14216,  0.76789, 0.76789, 1.14216, 1.70836,
                         2.03144,  2.1114,  1.94824,  1.54196,  0.89256, 4e-05],
     BigAngleOfAttackCm=[0, 0.24998, 0.46468,  0.5463, 0.53691,  0.51722,
                        0.49436,  0.40043,  0.31161,-0.31161,-0.40043,-0.49436,
                        -0.51722, -0.53691,  -0.5463, -0.46468, -0.24998,0],
        ):
    '''
    This function translates the XFoil Python dictionary result
    of XFoil.computePolars() into a PyZonePolar ready to be used
    in BEMT codes of PropellerAnalysis.py (or other related
    interpolating functions).

    INPUTS

    ResultsDictionary (Python dictionary) - The structure is
        identical         to the output of XFoil.computePolars()
        function (see doc of computePolars() for detailed
        information)

    Title (string) - Title to give to the PyZonePolar. This is
        usually the identifier of the airfoil.

    Variables2Store (list of strings) - Specify which variables
        shall be stored. If Variables2Store is an empty list,
        then all available data are stored. For minimal
        BEMT-like storage, one may state: ['Cl','Cd']

    BigAngleOfAttack(Cl-to-Cm) - (2 element tupple or None)
        If this variable is None, then default values are given
        for the requests of aero-coefficient corresponding to
        big angles of attack (outside polar, typically).
        Otherwise, this variable may be a 2 element tupple.
        Each element is a 1D numpy array, containing:
         -> For the first element (e.g. BigAngleOfAttackCl[0])
            the values of the angle-of-attack outside range.
         -> For the second element (e.g. BigAngleOfAttackCl[1])
            the corresponding aero-coefficient values.

    OUTPUTS

    PyZonePolar (CGNS zone) - see PropellerAnalysis.py for more
        information on the structure of PyZonePolar files.
    '''


    # Additional dependencies
    import Converter.PyTree as C
    import Converter.Internal as I
    from . import InternalShortcuts as J

    # Some default values:
    default_Big={}
    default_Big['AoA']= np.array(BigAngleOfAttackAoA,dtype=np.float64, order='F')
    default_Big['Cl'] = np.array(BigAngleOfAttackCl,dtype=np.float64, order='F')
    default_Big['Cd'] = np.array(BigAngleOfAttackCd,dtype=np.float64, order='F')
    default_Big['Cm'] = np.array(BigAngleOfAttackCm,dtype=np.float64, order='F')


    # Prepare values to be stored in PyZonePolar
    if len(Variables2Store)==0: Variables2Store = PolarVariables
    Arrays = [ResultsDictionary[var] for var in Variables2Store]

    # Build main PyZonePolar
    PyZonePolar = J.createZone(Title, Arrays, Variables2Store)

    # Add .Polar#Range node
    if ResultsDictionary['PyZonePolarKind'] == 'Struct_AoA_Mach':
        AoARange = ResultsDictionary['AoA'][:,0]
    else:
        AoARange = ResultsDictionary['AoA']
    children = [
    ['AngleOfAttack',                   AoARange],
    ['Mach',           ResultsDictionary['Mach']],
    ['Reynolds',       ResultsDictionary['Reynolds']],
    ]
    BigAoADict = {'Cl':BigAngleOfAttackCl, 'Cd':BigAngleOfAttackCd, 'Cm':BigAngleOfAttackCm}
    for v in BigAoADict:
        BigAoADict[v] = default_Big['AoA'], default_Big[v]
        children += [['BigAngleOfAttack%s'%v, BigAoADict[v][0]]]
    J._addSetOfNodes(PyZonePolar,'.Polar#Range',children)


    # Add out-of-range big Angle Of Attack values
    children = [['BigAngleOfAttack%s'%v, BigAoADict[v][1]] for v in BigAoADict]
    J._addSetOfNodes(PyZonePolar,'.Polar#OutOfRangeValues',children)

    # Add Foil-data information
    children = []
    for var in ResultsDictionary:
        if (var in FoilVariables)    and \
           (var not in ('s','x','y')):
            children += [[var, ResultsDictionary[var]]]
    if len(children) > 0:
        J._addSetOfNodes(PyZonePolar,'.Polar#FoilValues',children)

    # Add Foil Geometry
    if ('x' in ResultsDictionary) and ('y' in ResultsDictionary):
        RankCoords = len(ResultsDictionary['x'].shape)
        if RankCoords == 3:
            Xcoord = ResultsDictionary['x'][0,0,:]
            Ycoord = ResultsDictionary['y'][0,0,:]
        elif RankCoords == 2:
            Xcoord = ResultsDictionary['x'][0,:]
            Ycoord = ResultsDictionary['y'][0,:]
        else:
            Xcoord = ResultsDictionary['x']
            Ycoord = ResultsDictionary['y']
        children = [['CoordinateX', Xcoord],
                    ['CoordinateY', Ycoord]]
        J._addSetOfNodes(PyZonePolar,'.Polar#FoilGeometry',children)

    # Add .Polar#Interp node
    children=[
    ['PyZonePolarKind',ResultsDictionary['PyZonePolarKind']],
    ]
    if ResultsDictionary['PyZonePolarKind'] == 'Unstr_AoA_Mach_Reynolds':
        children += [['Algorithm','RbfInterpolator']]
    elif ResultsDictionary['PyZonePolarKind'] == 'Struct_AoA_Mach':
        children += [['Algorithm','RectBivariateSpline']]
    J._addSetOfNodes(PyZonePolar,'.Polar#Interp',children)

    return PyZonePolar
