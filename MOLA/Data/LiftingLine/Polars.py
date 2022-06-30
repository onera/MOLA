'''
Main subpackage for airfoil Polars low-level operations

06/05/2022 - L. Bernardos - first creation
'''

from ..Core import (re,np,RED,GREEN,WARN,PINK,
                    CYAN,ENDC,interpolate)
import scipy.interpolate as si
from ..Node import Node
from ..Zone import Zone
from .. import load as load_tree
from ... import Data as M

DEFAULT_INTERPOLATORS_FIELDS = ['Cl', 'Cd', 'Cm']

def _HOSTtoDict(filename):
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

        AllowedVars = Data.keys()

        LinesQty = len(lines)

        TitleLineSplit = lines[0].split()
        if len(TitleLineSplit) == 1:
            Data['Title'] = TitleLineSplit[0]
        else:
            Data['Title'] = '_'.join(TitleLineSplit[1:])

        # Read Allowed Variables:
        for i in range(LinesQty):
            lS = lines[i].split()
            if (len(lS) >= 2) and (lS[1] in AllowedVars):
                Var = lS[1]

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
                    NextTag = lines[j].split()

                SortInd = np.argsort(SetOfBigAoA)
                SetOfBigAoA= np.array([SetOfBigAoA[i] for i in SortInd], order='F')
                SetOfBigAoAValues= np.array([SetOfBigAoAValues[i] for i in SortInd], order='F')

                Data[Var]['BigAoA'] = SetOfBigAoA
                Data[Var]['BigAoAValues'] = SetOfBigAoAValues
            elif len(re.findall(r'REYNOLDS/MACH',lines[i].upper()))==1:
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

def _dictToZone( PolarDict ):
    """
    Convert the dictionary obtained using :py:func:`_HOSTtoDict`
    to a CGNS format polar zone.

    Parameters
    ----------

         PolarDict : dict
            as provided by the function :py:func:`_HOSTtoDict`

    Returns
    -------

        PyZonePolar : zone
            CGNS structured data containing the 2D airfoil
            aerodynamic haracteristics and other relevant data for interpolation
            operations
    """

    # Get the size of the main data array
    Data =  PolarDict
    DataDims = Data['Cl']['Array'].shape

    if len(DataDims)<3:
        Ni, Nj = DataDims
        Dims2Set = Ni, Nj, 1
    else:
        Dims2Set = Ni, Nj, Nk = DataDims

    # Produce an auxiliar zone where data will be stored
    Arrays, ArraysNames = [], []
    for var in ('Cl', 'Cd', 'Cm'):
        Arrays += [ Data[var]['Array'] ]
        ArraysNames += [ var ]

    try:
        Title = Data['Title']
    except KeyError:
        print("WARNING: convertDict2PyZonePolar() ->\n Provided data has no airfoil title.\nThis may produce future interpolation errors.")
        Title = 'Untitled'

    PyZonePolar = M.newZoneFromArrays( Title, ArraysNames, Arrays )

    # Add information on data range
    PyZonePolar.setParameters('.Polar#Range',
                              AngleOfAttack=Data['Cl']['AoA'],
                              Mach=Data['Cl']['Mach'],
                              BigAngleOfAttackCl=Data['Cl']['BigAoA'],
                              BigAngleOfAttackCd=Data['Cd']['BigAoA'],
                              BigAngleOfAttackCm=Data['Cm']['BigAoA'])

    for key in ['Reynolds', 'Mach']:
        if key in Data['Cl']:
            PyZonePolar.setParameters('.Polar#Range', Reynolds=Data['Cl'][key])

    # Add out-of-range big Angle Of Attack values
    PyZonePolar.setParameters('.Polar#OutOfRangeValues',
                               BigAngleOfAttackCl=Data['Cl']['BigAoAValues'],
                               BigAngleOfAttackCd=Data['Cd']['BigAoAValues'],
                               BigAngleOfAttackCm=Data['Cm']['BigAoAValues'])

    # Add .Polar#Interp node
    PyZonePolar.setParameters('.Polar#Interp',
                              PyZonePolarKind=Data['PyZonePolarKind'],
                              Algorithm='RectBivariateSpline')

    return PyZonePolar

def _HOSTtoZone( filename ):
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
    Data        = _HOSTtoDict(filename)
    PyZonePolar = _dictToZone(Data)
    return PyZonePolar

def _buildInterpolators(PyZonePolars, InterpFields=DEFAULT_INTERPOLATORS_FIELDS):
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

    Returns
    -------

        InterpDict : dict
            resulting python dictionary containing the interpolation functions
            of the 2D polars.
    """
    InterpDict = {}
    for polar in M.getZones(PyZonePolars):
        PolarInterpNode = polar.childNamed('.Polar#Interp')
        if PolarInterpNode is None: continue
        mode = PolarInterpNode.childNamed('Algorithm').value()

        if mode == 'RbfInterpolator':
            InterpDict[polar[0]] = _buildUntructuredInterpolator(polar, InterpFields=InterpFields)
        elif mode == 'RectBivariateSpline':
            InterpDict[polar[0]] = _buildStructuredInterpolator(polar, InterpFields=InterpFields)
        else:
            raise AttributeError('Mode %s not implemented.'%mode)

    return InterpDict


def _buildStructuredInterpolator(PyZonePolar, interpOptions=None,
        InterpFields=DEFAULT_INTERPOLATORS_FIELDS):
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

    # Get the fields to interpolate
    Data = {}
    DataRank = {}
    FS_n = PyZonePolar.childNamed('FlowSolution')

    FV_n = PyZonePolar.childNamed('.Polar#FoilValues')
    for IntField in InterpFields:
        Field_n = FS_n.childNamed(IntField)
        if Field_n:
            Data[IntField] = Field_n.value()
            DataRank[IntField] = len(Data[IntField].shape)
        else:
            if FV_n:
                Field_n = FV_n.childNamed(IntField)
                if Field_n:
                    Data[IntField] = Field_n.value()
                    DataRank[IntField] = len(Data[IntField].shape)

    PR_n = PyZonePolar.childNamed('.Polar#Range')
    AoARange= PR_n.childNamed('AngleOfAttack').value()
    NAoARange=len(AoARange)
    MachRange=PR_n.childNamed('Mach').value()
    MachRangeMax = MachRange.max()
    MachRangeMin = MachRange.min()
    NMachRange=len(MachRange)

    OutOfRangeValues_ParentNode = PyZonePolar.childNamed('.Polar#OutOfRangeValues')

    BigAoARange      = {}
    OutOfRangeValues = {}
    for IntField in InterpFields:
        BigAoARangeVar_n = PR_n.childNamed('BigAngleOfAttack%s'%IntField)
        if BigAoARangeVar_n is None:
            BigAoARangeVar_n = PR_n.childNamed('BigAngleOfAttackCl')
        BigAoARange[IntField] = BigAoARangeVar_n.value()

        OutOfRangeValues_n = OutOfRangeValues_ParentNode.childNamed('BigAngleOfAttack%s'%IntField)
        if OutOfRangeValues_n is not None:
            OutOfRangeValues[IntField] = OutOfRangeValues_n.value()


    # -------------- BUILD INTERPOLATOR -------------- #
    # Currently, only scipy.interpolator based objects
    # are supported. In future, this shall be extended
    # to e.g: dakota, rncarpio, scattered, sparse...

    # 2D interpolation
    ReynoldsOverMach = PyZonePolar.get('ReynoldsOverMach')
    if ReynoldsOverMach:
        ReynoldsOverMach = ReynoldsOverMach.value()[0]
    else:
        ReynoldsRange= PyZonePolar.get('Reynolds').value()[1]

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


def _buildUntructuredInterpolator(PyZonePolar, InterpFields=DEFAULT_INTERPOLATORS_FIELDS):
    '''
    This function creates the interpolation function of Polar
    data of an airfoil stored as a PyTree Zone, using radial-basis-functions.

    It handles out-of-range polar-specified angles of attack.

    Parameters
    ----------

        PyZonePolar : PyTree Zone containing Polar information,
            as produced by e.g. :py:func:`convertHOSTPolarFile2PyZonePolar`

        interpOptions : dict
            options to pass to the interpolator function.

            .. warning:: this will be include in a node inside **PyZonePolar**

        InterpFields : :py:class:`tuple` of :py:class:`str`
            variables to be interpolated.

    Returns
    -------

        InterpolationFunction : function
            function of interpolation, with expected usage:

            >>> Cl, Cd, Cm = InterpolationFunction(AoA, Mach, Reynolds)
    '''



    # Check kind of PyZonePolar
    PolarInterpNode = PyZonePolar.childNamed('.Polar#Interp')
    PyZonePolarKind = PolarInterpNode.childNamed('PyZonePolarKind').value()
    Algorithm = PolarInterpNode.childNamed('Algorithm').value()
    if PyZonePolarKind != 'Unstr_AoA_Mach_Reynolds':
        raise AttributeError('RbfInterpolator object can only be associated with a PyZonePolar of type "Unstr_AoA_Mach_Reynolds". Check PyZonePolar "%s"'%PyZonePolar[0])
    if Algorithm != 'RbfInterpolator':
        raise ValueError("Attempted to use RbfInterpolator, but Algorithm node in PyZonePolar named '%s' was '%s'"%(PyZonePolar[0], Algorithm))

    # Get the fields to interpolate
    Data       = {}
    DataRank   = {}
    DataShape  = {}
    for IntField in InterpFields:
        Data[IntField] = PyZonePolar.childNamed(IntField).value()
        DataShape[IntField]  = Data[IntField].shape
        DataRank[IntField] = len(DataShape[IntField])

    # TODO continue here
    # Get polar independent variables (AoA, Mach, Reynolds)
    PolarRangeNode = PyZonePolar.childNamed('.Polar#Range')
    AoARange = PolarRangeNode.childNamed('AngleOfAttack').value()
    MachRange = PolarRangeNode.childNamed('Mach').value()
    ReRange = PolarRangeNode.childNamed('Reynolds').value()

    # Compute bounding box of independent variables
    AoAMin,  AoAMax =  AoARange.min(),  AoARange.max()
    ReMin,    ReMax =   ReRange.min(),   ReRange.max()
    MachMin,MachMax = MachRange.min(), MachRange.max()

    # Compute ranges of big angle-of-attack
    BigAoARange = {}
    OutOfRangeValues_ParentNode = PyZonePolar.get('.Polar#OutOfRangeValues')
    for IntField in InterpFields:
        BigAoARangeVar_n = PyZonePolar.get('BigAngleOfAttack%s'%IntField)
        if BigAoARangeVar_n is None:
            BigAoARangeVar_n = PyZonePolar.get('BigAngleOfAttackCl')
        BigAoARange[IntField] = BigAoARangeVar_n[1]

    # Compute Delaunay triangulation of independent variables
    # (AoA, Mach, Reynolds)
    points = np.vstack((AoARange,MachRange,ReRange)).T
    triDelaunay = Delaunay(points)

    # CONSTRUCT INTERPOLATORS
    # -> inside qhull : use Rbf interpolator
    # -> outside qhull but inside ranges BoundingBox : use
    #       NearestNDInterpolator
    # -> outside ranges BoundingBox : use interp1d_linear
    #       on Big angle-of-attack data, if available

    inQhullFun, outQhullFun, outMaxAoAFun, outMinAoAFun = {}, {}, {}, {}
    def makeNaNFun(dummyArray):
        newArray = dummyArray*0.
        newArray[:] = np.nan
        return newArray

    for IntField in InterpFields:
        if DataRank[IntField] == 1:
            # Integral quantity: Cl, Cd, Cm, Top_Xtr...
            '''
            Rbf functions:
            'multiquadric' # ok
            'inverse'      # bit expensive
            'gaussian'     # expensive (and innacurate?)
            'linear'       # ok
            'cubic'        # expensive
            'quintic'      # expensive
            'thin_plate'   # bit expensive
            '''
            inQhullFun[IntField] = si.Rbf(0.1*AoARange, MachRange,1e-6*ReRange, Data[IntField], function='multiquadric',
                smooth=1, # TODO: control through PyTree node
                )
            outQhullFun[IntField] = si.NearestNDInterpolator(points,Data[IntField])
            outBBRangeValues_n = OutOfRangeValues_ParentNode.get('BigAngleOfAttack%s'%IntField)
            if outBBRangeValues_n is not None:
                MaxAoAIndices = BigAoARange[IntField]>0
                outMaxAoAFun[IntField] = si.interp1d( BigAoARange[IntField][MaxAoAIndices], outBBRangeValues_n[1][MaxAoAIndices], assume_sorted=True, copy=False,fill_value='extrapolate')
                MinAoAIndices = BigAoARange[IntField]<0
                outMinAoAFun[IntField] = si.interp1d( BigAoARange[IntField][MinAoAIndices], outBBRangeValues_n[1][MinAoAIndices], assume_sorted=True, copy=False,fill_value='extrapolate')
            else:
                outMaxAoAFun[IntField] = makeNaNFun
                outMinAoAFun[IntField] = makeNaNFun

        elif DataRank[IntField] == 2:
            # Foil-distributed quantity: Cp, delta1, theta...
            inQhullFun[IntField]  = []
            outQhullFun[IntField] = []
            outBBFun[IntField]    = []

            outBBRangeValues_n = OutOfRangeValues_ParentNode.get('BigAngleOfAttack%s'%IntField)
            for k in range(DataShape[IntField][1]):
                inQhullFun[IntField] += [si.Rbf(0.1*AoARange, MachRange,1e-6*ReRange, Data[IntField][:,k], function='multiquadric',
                smooth=0, # TODO: control through PyTree node
                )]
                outQhullFun[IntField] += [si.NearestNDInterpolator(points,Data[IntField][:,k])]
                if outBBRangeValues_n is not None:
                    outBBFun[IntField] += [si.interp1d( BigAoARange[IntField][:,k], outBBRangeValues_n[1][:,k], assume_sorted=True, copy=False)]
                else:
                    outBBFun[IntField] += [makeNaNFun]

        else:
            raise ValueError('FATAL ERROR: Rank of data named "%s" to be interpolated is %d, and must be 1 (for integral quantities like Cl, Cd...) or 2 (for foil-distributed quantities like Cp, theta...).\nCheck your PyZonePolar data.'%(IntField,DataRank[IntField]))


    def interpolationFunction(AoA, Mach, Reynolds):

        # Check input data structure
        if isinstance(AoA,list): AoA = np.array(AoA,dtype=np.float64, order='F')
        if isinstance(Mach,list): Mach = np.array(Mach,dtype=np.float64, order='F')
        if isinstance(Reynolds,list): Reynolds = np.array(Reynolds,dtype=np.float64, order='F')

        # Replace some NaN in Mach or Reynolds number by 0
        if all(np.isnan(Mach)): raise ValueError('all-NaN Found in Mach')
        elif any(np.isnan(Mach)): Mach[np.isnan(Mach)] = 0

        if all(np.isnan(Reynolds)): raise ValueError('all-NaN Found in Reynolds')
        elif any(np.isnan(Reynolds)): Reynolds[np.isnan(Reynolds)] = 0

        # Find boolean ranges depending on requested data:
        OutAoAMax = AoA > AoAMax
        AnyOutAoAMax = np.any(OutAoAMax)
        OutAoAMin = AoA < AoAMin
        AnyOutAoAMin = np.any(OutAoAMin)
        outBB = OutAoAMax + OutAoAMin
        AllOutBB = np.all(outBB)
        AnyOutBB = np.any(outBB)
        inBB  = np.logical_not(outBB)

        # Interpolate for each requested field "IntField"
        Values = {}
        FirstField = True
        for IntField in InterpFields:

            if DataRank[IntField] == 1:
                Values[IntField] = AoA*0 # Declare array

                if not AllOutBB:
                    # Compute values inside Bounding-Box
                    Values[IntField][inBB] = inQhullFun[IntField](0.1*AoA[inBB], Mach[inBB], 1e-6*Reynolds[inBB])

                    # Determine compute points outside Qhull but
                    # still inside Bounding-Box
                    if FirstField:
                        inBBoutQhull = np.isnan(Values[IntField])
                        someInBBoutQhull = np.any(inBBoutQhull)

                    # Compute outside-Qhull points by nearest
                    # point algorithm
                    if someInBBoutQhull:
                        Values[IntField][inBBoutQhull] = outQhullFun[IntField](AoA[inBBoutQhull], Mach[inBBoutQhull], Reynolds[inBBoutQhull])

                # Compute outside big-angle of attack values
                fieldMaxAoAFun = outMaxAoAFun[IntField]
                fieldMinAoAFun = outMinAoAFun[IntField]
                if AnyOutAoAMax:
                    Values[IntField][OutAoAMax] = fieldMaxAoAFun(np.minimum(np.maximum(AoA[OutAoAMax],-180.),+180.))
                if AnyOutAoAMin:
                    Values[IntField][OutAoAMax] = fieldMinAoAFun(np.minimum(np.maximum(AoA[OutAoAMax],-180.),+180.))

            else:
                # DataRank[IntField] == 2
                FoilValues = []
                for k in range(DataShape[IntField][1]):
                    CurrentValues = AoA*0 # Declare array

                    if not AllOutBB:
                        # Compute values inside Bounding-Box
                        CurrentValues[inBB] = inQhullFun[IntField](AoA[inBB], Mach[inBB], Reynolds[inBB])

                        # Determine compute points outside Qhull but
                        # still inside Bounding-Box
                        if FirstField:
                            inBBoutQhull = np.isnan(Values[IntField])
                            someInBBoutQhull = np.any(inBBoutQhull)

                        # Compute outside-Qhull points by nearest
                        # point algorithm
                        if someInBBoutQhull:
                            CurrentValues[inBBoutQhull] = outQhullFun[IntField](AoA[inBBoutQhull], Mach[inBBoutQhull], Reynolds[inBBoutQhull])

                    # Compute outside big-angle of attack values
                    if AnyOutBB:
                        CurrentValues[outBB] = outBBFun[IntField](AoA[outBB])

                    FoilValues += [CurrentValues]

                Values[IntField] = np.vstack(FoilValues,dtype=np.float64,order='F')
            FirstField = False
        ListOfValues = [Values[IntField] for IntField in InterpFields]

        return ListOfValues

    return interpolationFunction

def loadInterpolators( filenames , InterpFields=DEFAULT_INTERPOLATORS_FIELDS ):
    if isinstance(filenames, str): filenames = [filenames]
    zones = []
    for filename in filenames:
        if filename.endswith('.cgns'):
            t = load_tree(filename)
            zones.extend( t.zones() )
        else:
            zones.append( _HOSTtoZone(filename) )

    return _buildInterpolators( zones, InterpFields=InterpFields )
