import os
import re
import numpy as np

from mola.pytree import InternalShortcuts as J
import Converter.Internal as I
import Generator.PyTree as G
import Converter.PyTree as C

def buildPolarsInterpolatorDict(PyZonePolars, InterpFields=['Cl', 'Cd','Cm'],
        Nrequest=None):
    """
    Build a Python dictionary of interpolation functions of polars from a list
    of **PyZonePolars**. Each key is the name of the **PyZonePolar**
    (the airfoil's tag) and the value is the interpolation function.

    .. note:: typical usage of the returned dictionary goes like this:

        >>> Cl, Cd, Cm = InterpDict['MyPolar'](AoA, Mach, Reynolds)

        where ``AoA``, ``Mach`` and ``Reynolds`` are :py:class:`float` or
        numpy 1D arrays (all yielding the same length)


    Parameters
    ----------

        PyZonePolars : :py:class:`list` of zone
            list of special zones containing the 2D aerodynamic polars of the
            airfoils.

        InterpFields : :py:class:`list` of :py:class:`str`
            list of names of fields to be interpolated.
            Acceptable names are the field names contained in
            all **PyZonePolars** fields located in ``FlowSolution`` container.

        Nrequest : int
            if provided, set the number of points requested by the
            interpolation.

            .. note:: only relevant if technique is ``'PyZoneExtractMesh'``
                *(which is being deprecated!)*

    Returns
    -------

        InterpDict : dict
            resulting python dictionary containing the interpolation functions
            of the 2D polars.
    """
    InterpDict = {}
    for polar in I.getZones(PyZonePolars):
        PolarInterpNode = I.getNodeFromName1(polar,'.Polar#Interp')
        if PolarInterpNode is None: continue
        mode = I.getValue(I.getNodeFromName1(PolarInterpNode,'Algorithm'))

        if mode == 'RectBivariateSpline':
            InterpDict[polar[0]] = interpolatorFromPyZonePolar(polar, InterpFields=InterpFields)
        else:
            raise AttributeError('Mode %s not implemented.'%mode)

    return InterpDict

def buildPolarsAnalyticalDict(Name='MyPolar', CLmin=-1.0, CLmax=1.5, CL0=0.0, CLa=2*np.pi,
        CD0 = 0.011, CD2u = 0.004, CD2l = 0.013, CLCD0 = 0.013, REref = 1.e6,
        REexp = 0.):
    """
    Construct a python dictionary of analytical functions allowing for
    determination of aerodynamic coefficients :math:`(c_l,\, c_d,\, c_m)`.
    The call of the analytical functions is made as follows:

    >>> Cl, Cd, Cm = AnalyticalDict['MyPolar'](AoA, Mach, Reynolds)


    The paramaters of :py:func:`buildPolarsAnalyticalDict` are coefficients
    (:py:class:`float`) that define the mathematical analytical functions like
    follows:

    ::

            # Linear for CL(AoA)
            CL = np.minimum(np.maximum((CL0 + CLa*np.deg2rad(AoA))/np.sqrt(1-Mach**2),CLmin),CLmax)

            # Double parabola for CD(CL)
            CD2 = CL*0
            CD2[CL>CLCD0]  = CD2u
            CD2[CL<=CLCD0] = CD2l
            CD = (CD0+CD2*(CL-CLCD0)**2)*(Reynolds/REref)**REexp

            CM = 0.

    Returns
    -------

        AnalyticalDict : dict
            dictionary containing the analytical functions.
    """
    def analyticalPolar(AoA,Mach,Reynolds):
        # Linear for CL(AoA)
        CL = np.minimum(np.maximum((CL0 + CLa*np.deg2rad(AoA))/np.sqrt(1-Mach**2),CLmin),CLmax)

        # Double parabola for CD(CL)
        CD2 = CL*0
        CD2[CL>CLCD0]  = CD2u
        CD2[CL<=CLCD0] = CD2l
        CD = (CD0+CD2*(CL-CLCD0)**2)*(Reynolds/REref)**REexp

        CM = 0.

        DictOfVals = dict(Cl=CL, Cd=CD, Cm=CM)

        return DictOfVals['Cl'], DictOfVals['Cd'], DictOfVals['Cm']

    InterpDict = dict(Name=analyticalPolar)

    return InterpDict

def interpolatorFromPyZonePolar(PyZonePolar, interpOptions=None,
        InterpFields=['Cl', 'Cd', 'Cm']):
    '''
    This function create the interpolation function of Polar
    data of an airfoil stored as a PyTree Zone.

    It handles out-of-range polar-specified angles of attack.

    Parameters
    ----------

        PyZonePolar : zone
            PyTree Zone containing Polar information,
            as produced by e.g. :py:func:`convertHOSTPolarFile2PyZonePolar`

        interpOptions : options to pass to the interpolator function.

            .. warning:: this will be included as **PyZonePolar** node in future
                versions

        InterpFields : :py:class:`tuple` of :py:class:`str`
            contains the strings of the variables to be interpolated.

    Returns
    -------

        InterpolationFunctions : function
            a function to be employed like this:

            >>> InterpolationFunctions(AoA, Mach, Reynolds, ListOfEquations=[])

    '''
    import scipy.interpolate as si


    # Get the fields to interpolate
    Data = {}
    DataRank = {}
    FS_n = I.getNodeFromName1(PyZonePolar,'FlowSolution')
    FV_n = I.getNodeFromName1(PyZonePolar,'.Polar#FoilValues')
    for IntField in InterpFields:
        Field_n = I.getNodeFromName1(FS_n,IntField)
        if Field_n:
            Data[IntField] = I.getValue(Field_n)
            DataRank[IntField] = len(Data[IntField].shape)
        else:
            Field_n = I.getNodeFromName1(FV_n,IntField)
            if Field_n:
                Data[IntField] = I.getValue(Field_n)
                DataRank[IntField] = len(Data[IntField].shape)

    PR_n = I.getNodeFromName1(PyZonePolar,'.Polar#Range')
    AoARange=I.getValue(I.getNodeFromName1(PR_n,'AngleOfAttack'))
    NAoARange=len(AoARange)
    MachRange=I.getValue(I.getNodeFromName1(PR_n,'Mach'))
    MachRangeMax = MachRange.max()
    MachRangeMin = MachRange.min()
    NMachRange=len(MachRange)

    OutOfRangeValues_ParentNode = I.getNodeFromName1(PyZonePolar,'.Polar#OutOfRangeValues')

    BigAoARange      = {}
    OutOfRangeValues = {}
    for IntField in InterpFields:
        BigAoARangeVar_n = I.getNodeFromName1(PR_n,'BigAngleOfAttack%s'%IntField)
        if BigAoARangeVar_n is None:
            BigAoARangeVar_n = I.getNodeFromName1(PR_n,'BigAngleOfAttackCl')
        BigAoARange[IntField] = I.getValue(BigAoARangeVar_n)

        OutOfRangeValues_n = I.getNodeFromName1(OutOfRangeValues_ParentNode,'BigAngleOfAttack%s'%IntField)
        if OutOfRangeValues_n is not None:
            OutOfRangeValues[IntField] = I.getValue(OutOfRangeValues_n)


    # -------------- BUILD INTERPOLATOR -------------- #
    # Currently, only scipy.interpolator based objects
    # are supported. In future, this shall be extended
    # to e.g: dakota, rncarpio, scattered, sparse...

    # 2D interpolation
    try:
        ReynoldsOverMach = I.getNodeFromName(PyZonePolar,'ReynoldsOverMach')[1][0]
    except TypeError:
        ReynoldsOverMach = None
        ReynoldsRange= I.getNodeFromName(PyZonePolar,'Reynolds')[1]

    if interpOptions is None: interpOptions = dict(kx=1, ky=1)


    # (AoA, Mach) interpolation
    # using si.RectBivariateSpline()
    tableInterpFuns = {}
    for IntField in InterpFields:


        if DataRank[IntField] == 2:
            # Integral quantity: Cl, Cd, Cm, Xtr...

            # Create extended angle-of-attack and data range
            lowAoA  = BigAoARange[IntField] < 0
            highAoA = BigAoARange[IntField] > 0
            ExtAoARange = np.hstack((BigAoARange[IntField][lowAoA],AoARange,BigAoARange[IntField][highAoA]))
            DataLow = np.zeros((np.count_nonzero(lowAoA),NMachRange),dtype=np.float64,order='F')
            DataHigh = np.zeros((np.count_nonzero(highAoA),NMachRange),dtype=np.float64,order='F')
            for m in range(NMachRange):
                if IntField in OutOfRangeValues:
                    DataLow[:,m]  = OutOfRangeValues[IntField][lowAoA]
                    DataHigh[:,m] = OutOfRangeValues[IntField][highAoA]
                else:
                    DataLow[:,m]  = 0
                    DataHigh[:,m] = 0

            ExtData = np.vstack((DataLow,Data[IntField],DataHigh))
            # Create Extended data range
            tableInterpFuns[IntField] = si.RectBivariateSpline(ExtAoARange,MachRange, ExtData, **interpOptions)
            # tableInterpFuns[IntField] is a function

        elif DataRank[IntField] == 3:
            # Foil-distributed quantity: Cp, delta*, theta...
            tableInterpFuns[IntField] = []
            for k in range(Data[IntField].shape[2]):
                interpFun = si.RectBivariateSpline(AoARange,MachRange, Data[IntField][:,:,k], **interpOptions)
                tableInterpFuns[IntField] += [interpFun]
            # tableInterpFuns[IntField] is a list of functions

        else:
            raise ValueError('FATAL ERROR: Rank of data named "%s" to be interpolated is %d, and must be 2 (for integral quantities like Cl, Cd...) or 3 (for foil-distributed quantities like Cp, theta...).\nCheck your PyZonePolar data.'%(IntField,DataRank[IntField]))


    def interpolationFunction(AoA, Mach, Reynolds):
        # This function should be as efficient as possible

        # BEWARE : RectBiVariate ignores Reynolds at
        # interpolation step


        Mach = np.clip(Mach,MachRangeMin,MachRangeMax)

        # Apply RectBiVariate interpolator
        ListOfValues = []
        for IntField in InterpFields:


            if DataRank[IntField] == 2:
                ListOfValues += [tableInterpFuns[IntField](AoA, Mach,grid=False)]
            else:
                # DataRank[IntField] == 3
                FoilValues = []
                for tableIntFun in tableInterpFuns[IntField]:
                    FoilValues += [[tableIntFun(AoA[ir], Mach[ir], grid=False) for ir in range(len(AoA))]]

                ListOfValues += [np.array(FoilValues,dtype=np.float64,order='F')]


        return ListOfValues

    return interpolationFunction

def pyZonePolar2AirfoilZone(pyzonename, PyZonePolars):
    '''
    Conveniently use ``.Polar#FoilGeometry`` coordinates of a **PyZonePolar** in
    order to build a structured curve zone.

    Parameters
    ----------

        pyzonename : :py:class:`str` or :py:class:`list` of :py:class:`str`
            name(s) to be employed in new curve(s) zone(s) defining the airfoil
            geometry.

        PyZonePolars : :py:class:`list` of zone or :py:class:`str`
            list of special **PyZonePolar** zones
            containing 2D polar information as well as the geometry. Specifically,
            the polars must contain the node ``.Polar#FoilGeometry`` with the
            children nodes ``CoordinateX`` and ``CoordinateY``.

            .. hint:: 
                if **PyZonePolars** is a :py:class:`str`, then it will be
                interpreted as a file name, and it will attempt to open it and
                return the zones

    Returns
    -------

        AirfoilGeom : zone or :py:class:`list` of zone
            the 1D curve(s) of the airfoil.
    '''
    if isinstance(PyZonePolars,str):
        PyZonePolars = J.load(PyZonePolars, return_type='zones')

    if isinstance(pyzonename,str):
        pyzonenames = [pyzonename]
    else:
        pyzonenames = pyzonename
    
    PyZonePolars = I.getZones(PyZonePolars)
    if not PyZonePolars: raise ValueError('not PyZonePolars')


    AirfoilGeoms = []
    for pzn in pyzonenames:
        zone = J.getZoneFromListByName(PyZonePolars, pzn)
        if not zone:
            raise ValueError('not zone')
        FoilGeom_n = I.getNodeFromName1(zone,'.Polar#FoilGeometry')
        Xcoord = I.getNodeFromName1(FoilGeom_n,'CoordinateX')[1]
        Ycoord = I.getNodeFromName1(FoilGeom_n,'CoordinateY')[1]

        AirfoilGeoms.append( J.createZone(pzn,
                                [Xcoord,Ycoord,Ycoord*0],
                                ['CoordinateX','CoordinateY','CoordinateZ']) )

    if len(AirfoilGeoms) == 1:
        return AirfoilGeoms[0]
    else:
        return AirfoilGeoms

def plotStructPyZonePolars(PyZonePolars, addiationalQuantities=[],
        filesuffix='', fileformat='svg'):
    '''
    Convenient matplotlib function employed for plotting 2D airfoil's polar
    characteristics.

    It produces a set of figures in requested format.

    Parameters
    ----------

        PyZonePolars : :py:class:`list` of zone
            list of special PyZonePolar zones
            containing 2D polar information as well as the geometry.

        addiationalQuantities : :py:class:`list` of :py:class:`str`
            quantities to be plotted.
            Allowable names are the field names contained in **PyZonePolars**
            ``FlowSolution`` container.

        filesuffix : str
            suffix to append to new figure files produced by the function

        fileformat : str
            requested file format, that must be compatible with matplotlib. Some
            examples are: ``'svg'``, ``'pdf'``, ``'png'``

    Returns
    -------

        None : None
            A series of files (figures and legends) are written
    '''

    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator


    # if __PLOT_STIX_LATEX__:
    #     plt.rc('text', usetex=True)
    #     plt.rc('text.latex', preamble=r'\usepackage[notextcomp]{stix}')
    #     plt.rc('font',family='stix')


    for pzp in I.getZones(PyZonePolars):

        # Declare figures
        fig1, ax1 = plt.subplots(1,1,figsize=(4.75,4.25)) # CL
        fig2, ax2 = plt.subplots(1,1,figsize=(4.75,4.25)) # CL/CD
        fig3, ax3 = plt.subplots(1,1,figsize=(9.0,2.0)) # CL/CD

        FS_n = I.getNodeFromName1(pzp,'FlowSolution')
        Cl = I.getNodeFromName1(FS_n,'Cl')[1]
        Cd = I.getNodeFromName1(FS_n,'Cd')[1]
        CloCd = Cl/Cd
        PolRange_n = I.getNodeFromName1(pzp,'.Polar#Range')
        AoA = I.getNodeFromName1(PolRange_n,'AngleOfAttack')[1]
        Mach = I.getNodeFromName1(PolRange_n,'Mach')[1]
        Reynolds = I.getNodeFromName1(PolRange_n,'Reynolds')[1]
        FoilID = pzp[0]
        nMach = len(Mach)
        Colors = plt.cm.jet(np.linspace(0,1,nMach))
        for i in range(nMach):
            if nMach > 1:
                ax1.plot(AoA,Cl[:,i],color=Colors[i])
                ax2.plot(AoA,CloCd[:,i],color=Colors[i])
            else:
                ax1.plot(AoA,Cl,color=Colors[i])
                ax2.plot(AoA,CloCd,color=Colors[i])
            ax3.plot([],[],color=Colors[i],label='M=%g, Re=%g'%(Mach[i],Reynolds[i]))

        minLocX = AutoMinorLocator()
        ax1.xaxis.set_minor_locator(minLocX)
        minLocY1 = AutoMinorLocator()
        ax1.yaxis.set_minor_locator(minLocY1)

        minLocX2 = AutoMinorLocator()
        ax2.xaxis.set_minor_locator(minLocX2)
        minLocY2 = AutoMinorLocator()
        ax2.yaxis.set_minor_locator(minLocY2)

        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        ax1.set_xlabel(r'$\alpha$ (deg)')
        ax1.set_ylabel('$c_L$')
        ax1.set_title(FoilID)

        ax2.set_xlabel(r'$\alpha$ (deg)')
        ax2.set_ylabel('$c_L/c_D$')
        ax2.set_title(FoilID)

        plt.sca(ax1)
        plt.tight_layout()
        filename = 'PolarsCL_%s%s.%s'%(FoilID,filesuffix,fileformat)
        print('Saving %s ...'%filename)
        plt.savefig(filename)
        print('ok')

        plt.sca(ax2)
        plt.tight_layout()
        filename = 'PolarsEff_%s%s.%s'%(FoilID,filesuffix,fileformat)
        print('Saving %s ...'%filename)
        plt.savefig(filename)
        print('ok')

        plt.sca(ax3)
        ax3.legend(loc='upper left', ncol=4, bbox_to_anchor=(0.00, 1.00),bbox_transform=fig3.transFigure, title=FoilID, frameon=False)
        plt.axis('off')
        filename = 'PolarsLegend_%s%s.%s'%(FoilID,filesuffix,fileformat)
        print('Saving %s ...'%filename)
        plt.savefig(filename)
        print('ok')
        plt.close('all')

        for addQty in addiationalQuantities:
            fig1, ax1 = plt.subplots(1,1,figsize=(4.75,4.25))
            Field, = J.getVars(pzp,[addQty])
            for i in range(nMach):
                if nMach>1:
                    ax1.plot(AoA,Field[:,i],color=Colors[i])
                else: ax1.plot(AoA,Field,color=Colors[i])
            minLocX = AutoMinorLocator()
            ax1.xaxis.set_minor_locator(minLocX)
            minLocY1 = AutoMinorLocator()
            ax1.yaxis.set_minor_locator(minLocY1)

            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

            ax1.set_xlabel(r'$\alpha$ (deg)')
            ax1.set_ylabel(addQty)
            ax1.set_title(FoilID)

            plt.tight_layout()
            filename = 'Polars%s_%s%s.%s'%(addQty,FoilID,filesuffix,fileformat)
            print('Saving %s ...'%filename)
            plt.savefig(filename)
            print('ok')
            plt.close('all')

def convertHOSTPolarFile2Dict(filename):
    '''
    Extract airfoil polar information and convert it to Python
    Dictionnary, given a filename including path of a HOST
    formatted file.

    Parameters
    ----------

        filename : str
            full or relative path towards HOST formatted file

    Returns
    -------

        Result : dict
            Python Dictionnary containing the numerical values
    '''
    def scan(line,OutputType=float, RegExpr=r'[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?'):
        scanned = re.findall(RegExpr,line)
        return [OutputType(item) for item in scanned]

    with open(filename,'r') as f:
        lines = f.readlines()

    Data = {'Cl':{}, 'Cd':{},'Cm':{},}

    Aliases = {"cl": "Cl", "cd": "Cd", "cm": "Cm", "cz": "Cl", "cx": "Cd"}
    AllowedVars = [
        k.lower() for k in {*Data.keys(), *Aliases.keys(), *Aliases.values()}
    ]

    LinesQty = len(lines)

    Data['Title']='_'.join(lines[0].split()[1:])

    # Read Allowed Variables:

    for i in range(LinesQty):
        lS = lines[i].split()
        if (len(lS) >= 2) and (lS[1].lower() in AllowedVars):
            Var = Aliases[lS[1].lower()]

            AoAQty, MachQty = scan(lines[i+1],int)

            # Get Angles of Attack
            AoA = []
            j = i+1
            while len(AoA) < AoAQty:
                j += 1
                AoA += scan(lines[j],float)
            Data[Var]['AoA'] = np.array(AoA,order='F')

            # Get Mach numbers
            Mach = []
            while len(Mach) < MachQty:
                j += 1
                Mach += scan(lines[j],float)
            Data[Var]['Mach'] = np.array(Mach,order='F')

            # Get Variable
            VarNumpy = np.empty((AoAQty,MachQty),order='F')
            VarNumpy[:] = 1
            for a in range(AoAQty):
                VarLine = []
                while len(VarLine) < MachQty:
                    j += 1
                    VarLine += scan(lines[j],float)
                VarNumpy[a,:] = np.array(VarLine,order='F')
            Data[Var]['Array'] = VarNumpy

            # Read big angles
            j+=1
            NextTag = lines[j].split()
            SetOfBigAoA = []
            SetOfBigAoAValues = []
            while len(NextTag) == 1:
                BigAoA, BigAoAValues = [], []
                BigAoAQty = int(NextTag[0])
                while len(BigAoA) < BigAoAQty:
                    j += 1
                    BigAoA += scan(lines[j],float)
                while len(BigAoAValues) < BigAoAQty:
                    j += 1
                    BigAoAValues += scan(lines[j],float)
                SetOfBigAoA += BigAoA
                SetOfBigAoAValues += BigAoAValues
                j+=1
                try:
                    NextTag = lines[j].split()
                except IndexError:
                    break


            SortInd = np.argsort(SetOfBigAoA)
            SetOfBigAoA= np.array([SetOfBigAoA[i] for i in SortInd], order='F')
            SetOfBigAoAValues= np.array([SetOfBigAoAValues[i] for i in SortInd], order='F')

            Data[Var]['BigAoA'] = SetOfBigAoA
            Data[Var]['BigAoAValues'] = SetOfBigAoAValues
        elif '(C*L/NU)I0' in lines[i]:
            j=i
            ReynoldsOverMach = scan(lines[j],float)
            Data['ReynoldsOverMach'] = ReynoldsOverMach[-1]
            Data['Cl']['Reynolds'] = Data['ReynoldsOverMach']*Data['Cl']['Mach']
        elif (len(lS) == 2) and (lS[1] == 'Reynolds'):
            # Get Reynolds
            j = i+1
            ReynoldsQty = scan(lines[j],int)[0]
            if ReynoldsQty != MachQty:
                raise ValueError('ReynoldsQty (%g) is not equal to MachQty (%g). Check your HOST file.'%(ReynoldsQty,MachQty))
            Reynolds = []
            while len(Reynolds) < ReynoldsQty:
                j += 1
                Reynolds += scan(lines[j],float)
            for Var in AllowedVars:
                Data[Var]['Reynolds'] = np.array(Reynolds,order='F')
    Data['PyZonePolarKind'] = 'Struct_AoA_Mach'

    return Data

def convertDict2PyZonePolar(HostDictionnary):
    """
    Convert the dictionary obtained using :py:func:`convertHOSTPolarFile2Dict`
    to a CGNS format polar zone.

    Parameters
    ----------

        HostDictionnary : dict
            as provided by the function :py:func:`convertHOSTPolarFile2Dict`

    Returns
    -------

        PyZonePolar : zone
            CGNS structured data containing the 2D airfoil
            aerodynamic haracteristics and other relevant data for interpolation
            operations
    """

    # Get the size of the main data array
    Data = HostDictionnary
    DataDims = Data['Cl']['Array'].shape

    if len(DataDims)<3:
        Ni, Nj = DataDims
        Dims2Set = Ni, Nj, 1
    else:
        Dims2Set = Ni, Nj, Nk = DataDims

    # Produce an auxiliar zone where data will be stored
    PyZonePolar = G.cart((0,0,0),(1,1,1),Dims2Set)
    for var in ('Cl', 'Cd', 'Cm'):
        try:
            ArrayValues = Data[var]['Array']
        except KeyError:
            continue
        C._initVars(PyZonePolar,var,0.)
        Array = I.getNodeFromName(PyZonePolar,var)[1]
        Array[:] = ArrayValues

    try:
        Title = Data['Title']
    except KeyError:
        print("WARNING: convertDict2PyZonePolar() ->\n Provided data has no airfoil title.\nThis may produce future interpolation errors.")
        Title = 'Untitled'

    PyZonePolar[0]= Title
    I._rmNodesByName(PyZonePolar,'GridCoordinates')

    # Add information on data range
    children = [
    ['AngleOfAttack', Data['Cl']['AoA']],
    ['Mach', Data['Cl']['Mach']],
    ['BigAngleOfAttackCl', Data['Cl']['BigAoA']],
    ['BigAngleOfAttackCd', Data['Cd']['BigAoA']],
    ['BigAngleOfAttackCm', Data['Cm']['BigAoA']],
    ]

    if 'Reynolds' in Data['Cl']:
        children += [ ['Reynolds', Data['Cl']['Reynolds']], ]
    if 'ReynoldsOverMach' in Data:
        children += [ ['ReynoldsOverMach', Data['ReynoldsOverMach']], ]

    J._addSetOfNodes(PyZonePolar,'.Polar#Range',children)


    # Add out-of-range big Angle Of Attack values
    children = [
    ['BigAngleOfAttackCl', Data['Cl']['BigAoAValues']],
    ['BigAngleOfAttackCd', Data['Cd']['BigAoAValues']],
    ['BigAngleOfAttackCm', Data['Cm']['BigAoAValues']],
    ]
    J._addSetOfNodes(PyZonePolar,'.Polar#OutOfRangeValues',children)

    # Add .Polar#Interp node
    children=[
    ['PyZonePolarKind',Data['PyZonePolarKind']],
    ['Algorithm','RectBivariateSpline'],
    ]
    J._addSetOfNodes(PyZonePolar,'.Polar#Interp',children)

    return PyZonePolar

def convertHOSTPolarFile2PyZonePolar(filename):
    '''
    Convert a HOST-format 2D polar file into a CGNS-structured polar.

    Parameters
    ----------

        filename : str
            full or relative path of HOST polars

    Returns
    -------

        PyZonePolar : zone
            specific zone including 2D polar predictions
    '''
    Data        = convertHOSTPolarFile2Dict(filename)
    PyZonePolar = convertDict2PyZonePolar(Data)
    return PyZonePolar

def convertPolarsCGNS2HOSTformat(PyZonePolars,
                                 DIRECTORY_SAVE='POLARS',
                                 OutputFileNamePreffix='HOST_'):
    '''
    This function performs a conversion from CGNS *PyZonePolar* files towards
    HOST ascii format (neglecting special information that cannot be translated
    into HOST format)

    Parameters
    ----------

        PyZonePolars : :py:class:`list` of zone
            zones of 2D polars to be converted

        DIRECTORY_SAVE : str
            The directory where new HOST files are to be writen

        OutputFileNamePreffix : str
            a preffix to append to the name of the HOST files.

    Returns
    -------

        None : None
            HOST files
    '''

    AllowedQuantities = ('Cl','Cd','Cm')

    if not os.path.isdir(DIRECTORY_SAVE): os.makedirs(DIRECTORY_SAVE)

    for FoilZone in I.getZones(PyZonePolars):
        FoilName = I.getName( FoilZone )

        FileFullPath=os.path.join(DIRECTORY_SAVE,OutputFileNamePreffix+FoilName)
        print('writing %s'%FileFullPath)
        with open(FileFullPath,'w+') as f:

            FlowSol_n = I.getNodeFromName1(FoilZone,'FlowSolution')

            PolarRange = I.getNodeFromName1(FoilZone,'.Polar#Range')

            AngleOfAttackRange = I.getNodeFromName1(PolarRange,
                                                    'AngleOfAttack')[1]
            AoAQty = len(AngleOfAttackRange)

            MachRange = I.getNodeFromName1(PolarRange,'Mach')[1]
            MachQty = len(MachRange)

            ReynoldsRange = I.getNodeFromName1(PolarRange,'Reynolds')[1]

            AvrgReOverMach = np.mean(ReynoldsRange/MachRange)

            BigAoAsRange_nodes = I.getNodesFromName(PolarRange,
                                                    'BigAngleOfAttack*')
            BigAoAsRangesDict = {}
            for BigAoAsRange in BigAoAsRange_nodes:
                KeyName = BigAoAsRange[0].replace('BigAngleOfAttack','')
                BigAoAsRangesDict[KeyName] = BigAoAsRange[1]



            OutOfRange = I.getNodeFromName1(FoilZone,'.Polar#OutOfRangeValues')

            BigAoAsValue_nodes = I.getNodesFromName(OutOfRange,
                                                    'BigAngleOfAttack*')
            BigAoAsValuesDict = {}
            for BigAoAsValue in BigAoAsValue_nodes:
                KeyName = BigAoAsValue[0].replace('BigAngleOfAttack','')
                BigAoAsValuesDict[KeyName] = BigAoAsValue[1]

            f.write('      78      %s\n'%FoilName)

            for var in AllowedQuantities:
                var_n = I.getNodeFromName1(FlowSol_n,var)
                varName   = var
                varValues = I.getValue( var_n )

                f.write('    1 %s %s\n'%(varName,FoilName))
                f.write('%5i%5i\n'%(AoAQty, MachQty))
                inc=1
                for i in AngleOfAttackRange: 
                    f.write('%10.5f'%i)
                    if inc%8==0 or inc==len(AngleOfAttackRange) :
                        f.write('\n')
                    inc+=1
                inc=1
                for i in MachRange: 
                    f.write('%10.5f'%i)
                    if inc%8==0 or inc==len(MachRange):
                        f.write('\n')
                    inc+=1
                for row in varValues:
                    inc=1
                    for i in row: 
                        f.write('%10.5f'%i)
                        if inc==len(MachRange) :
                            f.write('\n')
                        inc+=1
                BigAoARange = BigAoAsRangesDict[varName]
                BigAoAValue = BigAoAsValuesDict[varName]
                LowAoABool  = BigAoARange < 0
                HighAoABool = BigAoARange > 0

                for BoolRange in (HighAoABool, LowAoABool):
                    # f.write('  ')
                    f.write('%5i\n'%len(BigAoARange[BoolRange]))
                    inc=1
                    for i in BigAoARange[BoolRange]: 
                        f.write('%10.5f'%i)
                        if inc%8==0 or inc==len(BigAoARange[BoolRange]) :
                            f.write('\n')
                        inc+=1
                    # f.write('  ')
                    inc=1
                    for i in BigAoAValue[BoolRange]: 
                        f.write('%10.5f'%i)
                        if inc%8==0 or inc==len(BigAoAValue[BoolRange]) :
                            f.write('\n')
                        inc+=1

            f.write('COEFFICIENT (C*L/NU)I0 (OU BIEN REYNOLDS/MACH) ............ %10.1f\n' %AvrgReOverMach)
            f.write('CORRECTION DE PRESSION GENERATRICE REYNOLDS/MACH=CSTE. .... SANS\n')
            f.write('EXPOSANT POUR CORRECTION DE REYNOLDS ( EXPREY) ............   -0.16667')
        os.chmod(FileFullPath, 0o755)

def loadPolarsInterpolatorDict( filenames, InterpFields=['Cl', 'Cd','Cm'] ):
    if isinstance(filenames, str):
        filenames = [ filenames ]
    elif not isinstance(filenames, list):
        raise TypeError('filenames %s not recognized'%str(type(filenames)))

    PyZonePolars = []
    for filename in filenames:
        PyZonePolars.extend( I.getZones( C.convertFile2PyTree(filename) ) )

    return buildPolarsInterpolatorDict(PyZonePolars, InterpFields=InterpFields)

