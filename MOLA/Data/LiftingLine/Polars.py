'''
Main subpackage for Polars related operations

06/05/2022 - L. Bernardos - first creation
'''

from ..Core import (re,np,RED,GREEN,WARN,PINK,
                    CYAN,ENDC,interpolate)
from ..Node import Node
from ..Zone import Zone


class PolarZone(Zone):
    """docstring for PolarZone"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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

        AllowedVars = Data.keys()

        LinesQty = len(lines)

        TitleLineSplit = lines[0].split()
        if len(TitleLineSplit) == 1:
            Data['Title'] = TitleLineSplit[0]
        else:
            Data['Title'] = TitleLineSplit[1]

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
