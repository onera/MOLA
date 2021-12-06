'''
MOLA - Wireframe.py

This module proposes wireframe geometry functionalities, both in
2D and 3D (see specifically each function's documentation).

Functions with the suffix 2D on the function's name are supposed
to be applied to 2D geometries in the XY plane.

This module makes use of Cassiopee modules.

File history:
27/02/2019 - v1.0 - L. Bernardos - Creation by recycling
'''

# System modules
import sys
import os
import numpy as np
from copy import deepcopy as cdeep
from timeit import default_timer as tic

# Scipy modules
try: import scipy.optimize
except:
    print ('%s: WARNING could not import scipy.optimize. This may cause error for some functions.'%__file__)
    pass

# Cassiopee
import Converter.PyTree as C
import Converter.Internal as I
import Geom.PyTree as D
import Post.PyTree as P
import Generator.PyTree as G
import Transform.PyTree as T
import Connector.PyTree as X
import Intersector.PyTree as XOR

# Generative modules
from . import InternalShortcuts as J

linelawVerbose = False

BADVALUE  = -999.

def distance(P1,P2):
    '''
    Compute the Euclidean distance between two points.

    P1 and P2 can be either a 3-float array, list or tuple; or a PyTree zone.

    Parameters
    ----------

        P1 : zone or :py:class:`list` or :py:class:`tuple` or array
            First point. It can be a CGNS zone of a point (as result of
            function :py:func:`D.point`) or a 3-float tuple, list or numpy array

        P2 : zone or :py:class:`list` or :py:class:`tuple` or array
            Second point. It can be a CGNS zone of a point (as result of
            function :py:func:`D.point`) or a 3-float tuple, list or numpy array

    Returns
    -------

        distance : float
            euclidean distance.

    Examples
    --------

    >>> TheDistance = W.distance((0,0,0),(1,0,0))
    '''
    if isPyTreePoint(P1):
        x1,y1,z1 = J.getxyz(P1)
        x1,y1,z1 = x1[0],y1[0],z1[0]
    else:
        x1,y1,z1 = P1

    if isPyTreePoint(P2):
        x2,y2,z2 = J.getxyz(P2)
        x2,y2,z2 = x2[0],y2[0],z2[0]
    else:
        x2,y2,z2 = P2

    Res = ((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1))**0.5

    return Res


def distanceOfPointToLine(Point, LineVector, LinePassingPoint):
    '''
    Compute the Euclidean minimum distance between a point in space and a line
    that passes through a point.

    Parameters
    ----------

        Point : zone or :py:class:`list` or :py:class:`tuple` or numpy array
            Includes first point coordinates.

        LineVector : :py:class:`list` or :py:class:`tuple` or numpy array
            Includes line direction vector.

        LinePassingPoint : :py:class:`list` or :py:class:`tuple` or numpy array
            Includes the line passing point coordinates.

    Returns
    -------

        distance : float
            minimum euclidean distance between the provided point and line
    '''

    if isPyTreePoint(Point):
        x,y,z = J.getxyz(Point)
        p = np.array([x[0],y[0],z[0]],dtype=np.float)
    else:
        p = np.array([Point[0],
                      Point[1],
                      Point[2]],dtype=np.float)

    if isPyTreePoint(LinePassingPoint):
        x,y,z = J.getxyz(LinePassingPoint)
        c = np.array([x[0],y[0],z[0]],dtype=np.float)
    else:
        c = np.array([LinePassingPoint[0],
                      LinePassingPoint[1],
                      LinePassingPoint[2]],dtype=np.float)

    l = np.array([LineVector[0],
                  LineVector[1],
                  LineVector[2]],dtype=np.float)
    l /= np.sqrt(l.dot(l))

    cp = p - c
    q = c + l*cp.dot(l)
    qp = p - q
    distance = np.sqrt(qp.dot(qp))
    
    return distance


def angle2D(P1,P2):
    '''
    Compute planar angle between the :math:`x`-axis and the direction
    given by vector :math:`\overrightarrow{P_1 P_2}` (vector defined
    ``P1->P2``).


    Parameters
    ----------

        P1 : zone or :py:class:`list` or :py:class:`tuple` or array
            First point. It can be a CGNS zone of a point (as result of
            function :py:func:`D.point`) or a 3-float tuple, list or numpy array

        P2 : zone or :py:class:`list` or :py:class:`tuple` or array
            Second point. It can be a CGNS zone of a point (as result of
            function :py:func:`D.point`) or a 3-float tuple, list or numpy array

    Returns
    -------

        angle : float
            2D planar angle in degrees


    Examples
    --------

    >>> MyAngle = W.angle2D((0.5,0,0),(1,1,0))
    '''

    if isPyTreePoint(P1):
        x1,y1,z1 = J.getxyz(P1)
        x1,y1,z1 = x1[0],y1[0],z1[0]
    else:
        x1,y1,z1 = P1

    if isPyTreePoint(P2):
        x2,y2,z2 = J.getxyz(P2)
        x2,y2,z2 = x2[0],y2[0],z2[0]
    else:
        x2,y2,z2 = P2

    Res = np.arctan2(y2-y1,x2-x1) * 180. / np.pi

    return Res


def isPyTreePoint(P):
    '''
    Return :py:obj:`True` if input argument **P** is a PyTree point (a zone).
    Otherwise, return :py:obj:`False`
    '''
    if (I.isStdNode(P) == -2) or (I.isStdNode(P) == 0): return False
    if C.getNPts(P) != 1: return False
    return True


def gets(curve):
    '''
    Get the numpy array of a 1D structured curve corresponding
    to its curvilinear abscissa.

    Parameters
    ----------
        curve : zone
            structured Zone PyTree curve

    Returns
    -------
        s : numpy.array
            Curvilinear abscissa :math:`s \in [0,\,1]`

    Examples
    --------

    >>> s = W.gets(curve)
    '''
    D._getCurvilinearAbscissa(curve)
    s = I.getNodeFromName(curve,'s')[1]
    return s


def getTanhDistTwo__(Nx, CellStart, CellEnd):
    '''
    .. note:: Private-level function, called by user-level
        :py:func:`linelaw`. Not intended to be called directly by the user.

    Build the 1D normalized distribution of **Nx** points, with
    provided **CellStart** and **CellEnd** sizes.

    .. note:: current version makes use of
        :py:func:`Generator.PyTree.enforcePlusX` and :py:func:`Generator.PyTree.enforceMoinsX`

    Parameters
    ----------

        Nx : int
            Number of points. (shall be :math:`\geq 6`).

        CellStart : float
            Normalized start cell size.

        CellEnd : float
            Normalized end cell size.

    Returns
    -------

        x : numpy.array
            1D vector in (0,1) with the normalized distribution

    Examples
    --------

    >>> x = W.getTanhDistTwo__(100,0.01,0.1)

    '''
    if Nx < 6: raise ValueError("getTanhDistTwo__: at least 6 pts are required")
    N = Nx -4
    l = D.line((0,0,0),(1,0,0),N)
    l = G.enforcePlusX(l,CellStart,(N-2,2),verbose=linelawVerbose)
    l = G.enforceMoinsX(l,CellEnd,(N-2,2),verbose=linelawVerbose)
    x = I.getNodeFromName(l,'CoordinateX')[1]
    x = J.getx(l)
    return x


def getTanhDist__(Nx, CellStart,isCellEnd=False):
    '''
    .. note:: Private-level function, called by user-level
        :py:func:`linelaw`. Not intended to be called directly by the user.

    Build the 1D normalized distribution of **Nx** points, with
    provided **CellStart** size.

    .. note:: current version makes use of
        :py:func:`Generator.PyTree.enforcePlusX` or:py:func:`Generator.PyTree.enforceMoinsX`



    Parameters
    ----------

        Nx : int
            Number of points. (shall be :math:`\geq 6`).

        CellStart : float
            Normalized start cell size.

        isCellEnd : bool
            `True` if reversed function (**CellStart** is actually **CellEnd**)

    Returns
    -------

        x : numpy.array
            1D vector in (0,1) with the normalized distribution

    Examples
    --------

    >>> x = W.getTanhDist__(100,0.001)
    '''

    if Nx < 4: raise ValueError("getTanhDist__: at least 4 pts are required")
    N = Nx - 2
    l = D.line((0,0,0),(1,0,0),N)
    l = G.enforcePlusX(l,CellStart,(N-2,2),verbose=linelawVerbose) if not isCellEnd else G.enforceMoinsX(l,CellStart,(N-2,2),verbose=linelawVerbose)
    x = J.getx(l)
    return x


def getTrigoLinDistribution__(Nx, p):
    '''
    .. note:: Private-level function, called by user-level
        :py:func:`linelaw`. Not intended to be called directly by the user.

    Build the 1D normalized distribution of **Nx** points, such that
    points are distributed following a trigonometric-linear
    composite law defined by the parameter **p** :math:`\in [-3,3]`.

    .. note:: Parameter **p** controls the discretization as the M. Drela XFoil
        or AVL tools.

    Parameters
    ----------

        Nx : int
            Number of points.

        p : float
            Discretization control parameter :math:`p \in [-3,3]`.

    Returns
    -------

        x : numpy.array
            1D vector in (0,1) with the normalized distribution

    Examples
    --------

    >>> x  = W.getTanhDist__(100, 2.2)
    '''

    x = np.linspace(0,1,Nx)
    if p <= 3 and p >= 2:
        L = x
        S1 = 1. + np.sin((np.pi/2.)*(x-1.))
        a = 3 - p
        b = p - 2
        return b*L + a*S1
    elif p <= 2 and p >= 1:
        S1 = 1. + np.sin((np.pi/2.)*(x-1.))
        C = 0.5*(1.+np.cos(np.pi*(x+1)))
        a = 2 - p
        b = p - 1
        return b*S1 + a*C
    elif p <= 1 and p >= 0:
        C = 0.5*(1.+np.cos(np.pi*(x+1)))
        L = x
        a = 1 - p
        b = p
        return b*C + a*L
    elif p <= 0 and p >= -1:
        L = x
        C = 0.5*(1.+np.cos(np.pi*(x+1)))
        a = - p
        b = p - (-1)
        return b*L + a*C
    elif p <= -1 and p >= -2:
        C = 0.5*(1.+np.cos(np.pi*(x+1)))
        S2 = np.sin((np.pi/2.)*x)
        a = (-1) - p
        b = p - (-2)
        return b*C + a*S2
    elif p <= -2 and p >= -3:
        S2 = np.sin((np.pi/2.)*x)
        L = x
        a = (-2) - p
        b = p - (-3)
        return b*S2 + a*L
    else:
        raise UserWarning('Parameter p=%g out of allowed bounds [3,-3]. Switched to p=0.'%p)
        return x


def linelaw(P1=(0,0,0), P2=(1,0,0), N=100, Distribution = None, verbose=linelawVerbose):
    '''
    Create a line of **N** points between **P1** and **P2** points, following
    a distribution constructed by the instructions contained
    in the dictionary **Distribution**.

    Parameters
    ----------

        P1 : :py:class:`list` of 3-:py:class:`float`
            Start point of the line

        P2 : :py:class:`list` of 3-:py:class:`float`
            End point of the line

        N : int
            Points quantity that the line will contain

        Distribution : dict

            Python dictionary specifying distribution instructions.
            Default value is :py:obj:`None`, which produces a uniform distribution.
            Accepted keys are:

            * kind : :py:class:`str`
                Can be one of:

                * ``'uniform'``
                    Makes an uniform spacing.

                * ``'tanhOneSide'``
                    Specifies the size of the first cell.

                * ``'tanhTwoSides'``
                    Specifies the size of the first and last cell.

                * ``'trigonometric'``
                    Employs a composite linear-trigonometric distribution.

                * ``'ratio'``
                    Employs a geometrical-growth type of law

            * FirstCellHeight : :py:class:`float`
                Specifies the size of the first cell

                .. note:: only relevant if **kind** is ``'tanhOneSide'`` ,
                    ``'tanhTwoSides'`` or ``'ratio'``

            * LastCellHeight : :py:class:`float`
                Specifies the size of the last cell

                .. note:: only relevant if **kind** is ``'tanhOneSide'`` or
                    ``'tanhTwoSides'``

            * parameter : :py:class:`float`
                Adjusts the composite linear-trigonometric distribution.

                .. note:: only relevant if **kind** is ``'trigonometric'``

                .. note:: **parameter** must be :math:`\in [-3,3]`

            * growth : :py:class:`float`
                geometrical growth rate

                .. note:: only relevant if **kind** is ``'ratio'``

    Returns
    -------

        Line : zone
            curve in form of a Structured Zone PyTree.

    Examples
    --------

    ::

        Line = W.linelaw( (0,0,1), (2,3,0), 200, dict(kind='tanhTwoSides',
                                                      FirstCellHeight=0.001,
                                                      LastCellHeight=0.02))


    '''

    if not Distribution:
        Line = D.line(P1, P2, N)
    elif 'kind' not in Distribution:
        Line = D.line(P1, P2, N)
    else:
        Line = D.line(P1,P2,N)
        Lx, Ly, Lz = J.getxyz(Line)
        if Distribution['kind'] == 'uniform':
            pass

        elif Distribution['kind'] == 'tanhOneSide':
            Length = distance(P1,P2)

            if 'FirstCellHeight' in Distribution:
                dy = Distribution['FirstCellHeight']/Length
                isCellEnd = False
            elif 'LastCellHeight' in Distribution:
                dy = Distribution['LastCellHeight']/Length
                isCellEnd = True
            else:
                raise ValueError('linelaw: kind %s requires "FirstCellHeight" or "LastCellHeight"')

            Dir = np.array([P2[0]-P1[0],P2[1]-P1[1],P2[2]-P1[2]])/Length
            S = getTanhDist__(N,dy,isCellEnd)*Length
            Height = S[1]-S[0] if not isCellEnd else S[-1] - S[-2]
            ErrorHeight = abs(100*(1-Height/(dy*Length)))
            if verbose and ErrorHeight > 1.:
                Msg="""
                --------
                Warning: Distribution of kind tanhOneSide resulted in an
                effective cell Height of: , %g , which differs from the
                desired one, %g, a relative amount of: %g prct.
                Try different discretization parameters for better result.
                --------\n"""%(Height,dy*Length,ErrorHeight)
                print (Msg)


            Lx[:] = S*Dir[0]+P1[0]
            Ly[:] = S*Dir[1]+P1[1]
            Lz[:] = S*Dir[2]+P1[2]

            return Line
        elif Distribution['kind'] == 'tanhTwoSides':
            Length = distance(P1,P2)
            dy = [0.,0.]
            dy[0] = Distribution['FirstCellHeight']/Length
            dy[1] = Distribution['LastCellHeight']/Length
            Dir = np.array([P2[0]-P1[0],P2[1]-P1[1],P2[2]-P1[2]])/Length
            S = getTanhDistTwo__(N,dy[0],dy[1])*Length
            Height1 = S[1]-S[0]; Height2 = S[-1]-S[-2]
            ErrorHeight1 = abs(100*(1-Height1/(dy[0]*Length)))
            ErrorHeight2 = abs(100*(1-Height2/(dy[1]*Length)))
            if verbose and ErrorHeight1 > 1.:
                Msg="""
--------
Warning: Distribution of kind tanhTwoSides resulted in an
effective first cell Height of: , %g , which differs from the
desired one, %g, a relative amount of: %g pctg.
Try different discretization parameters for better result.
--------\n"""%(Height1,(dy[0]*Length),ErrorHeight1)
                print (Msg)

            elif verbose and ErrorHeight2 > 1.:
                Msg="--------\n"
                Msg+='Warning: Distribution %s resulted in an\n'%Distribution['kind']

                Msg+='effective last cell Height of: %g\n'%Height2
                Msg+='which differs from the desired one, %g,\n'%(dy[1]*Length)
                Msg+='a relative amount of %g prct.\n'%ErrorHeight2
                Msg+='Try different discretization parameters for better result\n'
                Msg+="--------"
                print (Msg)


            Lx[:] = S*Dir[0]+P1[0]
            Ly[:] = S*Dir[1]+P1[1]
            Lz[:] = S*Dir[2]+P1[2]

        elif Distribution['kind'] == 'trigonometric':
            Length = distance(P1,P2)
            p = Distribution['parameter']
            Dir = np.array([P2[0]-P1[0],P2[1]-P1[1],P2[2]-P1[2]])/Length
            S = getTrigoLinDistribution__(N, p)*Length

            Lx[:] = S*Dir[0]+P1[0]
            Ly[:] = S*Dir[1]+P1[1]
            Lz[:] = S*Dir[2]+P1[2]
        elif Distribution['kind'] == 'ratio':
            growth = Distribution['growth']
            Length = distance(P1,P2)
            Dir = np.array([P2[0]-P1[0],P2[1]-P1[1],P2[2]-P1[2]])/Length
            dH = Distribution['FirstCellHeight']
            for i in range(1,N):
                Lx[i] = dH*Dir[0]+Lx[i-1]
                Ly[i] = dH*Dir[1]+Ly[i-1]
                Lz[i] = dH*Dir[2]+Lz[i-1]
                dH   *= growth
                CurrentLength = np.sqrt((Lx[i]-Lx[0])**2+(Ly[i]-Ly[0])**2+(Lz[i]-Lz[0])**2)
                if CurrentLength >= Length:
                    Line = T.subzone(Line,(1,1,1),(i,1,1))
                    break

            return Line


        else:
            raise AttributeError('Kind of distribution %s unknown.'%Distribution['kind'])

        # Strictly represents boundaries (no epsilon deviation):
        Lx[0] = P1[0]
        Ly[0] = P1[1]
        Lz[0] = P1[2]
        Lx[-1] = P2[0]
        Ly[-1] = P2[1]
        Lz[-1] = P2[2]

    return Line


def airfoil(designation='NACA0012',Ntop=None, Nbot=None, ChordLength=1.,
        TopDistribution=None, BottomDistribution=None,
        ClosedTolerance=True,LeadingEdgePos=None):
    """

    .. warning:: this function must be updated

    Creates a 4-digit or 5-digit series NACA airfoil including discretization
    parameters.

    Alternatively, reads a selig or lidnicer airfoil coordinate format (``.dat``)

    Parameters
    ----------

        designation : str
            NACA airfoil identifier of 4 or 5 digits

        Ntop : int
            Number of points of the Top side of the airfoil.

        Nbot : int
            Number of points of the Bottom side of the airfoil. If **Nbot** is not
            provided, then **Ntop** is the total number of points of the whole foil.

        ChordLength : float
            The chord length of the airfoil.

        TopDistribution : dict
            A distribution dictionary establishing the discretization
            law of the top side of the airfoil.

        BottomDistribution : dict
            A distribution dictionary establishing the discretization law of the
            bottom side of the airfoil.

        ClosedTolerance : float
            Geometrical criterion to determine if forcing closed
            Trailing Edge is desired or not (relative to **ChordLength**).

            .. hint:: Use **ClosedTolerance** :math:`\gg 1` for always forcing the closing of the airfoil.
                This will also trigger a slightly different 5-digit NACA formulation.

        LeadingEdgePos : float
            A float between :math:`\in (0,1)` and typically :math:`\\approx 0.5`
            establishing the parametric relative position of the
            Leading Edge position. It is used to accurately control
            the location of the leading edge refinement point. A value
            of 0 corresponds to the bottom side trailing edge, and a
            value of 1 corresponds to the top side trailing edge.
            If :py:obj:`None`, then makes no refinement based on curve length,
            but based only on the X coordinate.

    Returns
    -------

        Airfoil : zone
            the structured curve of the airfoil
    """

    # Prepares the X-distributions
    if not not Ntop and not Nbot:
        Nbot = Ntop/2 + Ntop%2
        Ntop /= 2



    NACAstringLoc = designation.find('NACA')
    # Determines the kind of airfoil to generate
    if designation.find('.') != -1: # Then user wants to import an airfoil from file
        Imported = np.genfromtxt(designation, dtype=np.float, skip_header=1, usecols=(0,1))
        # Deletes useless lines
        RowsToDelete = []
        for i in range(len(Imported[:,0])):
            if any(np.isnan(Imported[i])) or any(Imported[i]>1.5):
                RowsToDelete.append(i)
        Imported = np.delete(Imported, RowsToDelete, axis=0)
        # Checks for the format of the coordinate points
        Monot = np.diff(Imported[:,0])
        MonotIND = np.where(Monot < 0)[0]
        if len(MonotIND) == 1:
            # Lednicer format: Both sides start from Leading Edge
            MonotIND = MonotIND[0]
            if Imported[MonotIND,1] < Imported[-1,1]:
                xL = np.flipud(Imported[:MonotIND+1,0])
                yL = np.flipud(Imported[:MonotIND+1,1])
                xU = Imported[MonotIND+1:,0]
                yU = Imported[MonotIND+1:,1]

            else:
                xU = Imported[:MonotIND+1,0]
                yU = Imported[:MonotIND+1,1]
                xL = np.flipud(Imported[MonotIND+1:,0])
                yL = np.flipud(Imported[MonotIND+1:,1])
        else:
            # Selig format: Starts and ends from trailing edge
            if Imported[1,1] > Imported[-2,1]: Imported = np.flipud(Imported)
            xMin = np.argmin(Imported[:,0])
            xL= Imported[:xMin+1,0]
            yL = Imported[:xMin+1,1]
            xU= Imported[xMin:,0]
            yU = Imported[xMin:,1]
        Airfoil = D.line((0,0,0), (1,0,0), len(xL)+len(xU)-1 )
        Airfoil[0] = designation.split('.')[0]
        Airfoil_x = J.getx(Airfoil)
        Airfoil_y = J.gety(Airfoil)
        Airfoil_x[:] = np.hstack((xL,xU[1:]))
        Airfoil_y[:] = np.hstack((yL,yU[1:]))
    elif NACAstringLoc != -1: # Then user wants to generate a naca-series airfoil
        if not Ntop: Ntop = Nbot = 200
        xU = J.getx(linelaw((0,0,0), (1,0,0), Ntop, {'kind':'trigonometric','parameter':2}))
        xL = J.getx(linelaw((0,0,0), (1,0,0), Nbot, {'kind':'trigonometric','parameter':2}))
        xL[:] = np.flipud(xL)
        NACAidentifier = designation[NACAstringLoc + 4:len(designation)]
        # NACA constants
        a0= 0.2969; a1=-0.1260; a2=-0.3516; a3= 0.2843; a4=-0.1036 if ClosedTolerance>=1.0 else -0.1015

        if len(NACAidentifier) == 4: # 4-digit NACA
            m = float(NACAidentifier[0])*0.01  # Maximum camber
            p = float(NACAidentifier[1])*0.1   # Maximum camber location
            t = float(NACAidentifier[2:])*0.01 # Maximum thickness
            ytU = 5.*t*(a0*np.sqrt(xU)+a1*(xU)+a2*(xU)**2+a3*(xU)**3+a4*(xU)**4)
            ytL = 5.*t*(a0*np.sqrt(xL)+a1*(xL)+a2*(xL)**2+a3*(xL)**3+a4*(xL)**4)
            Airfoil = D.line((0,0,0), (1,0,0), Ntop+Nbot-1)
            Airfoil_x, Airfoil_y = J.getxy(Airfoil)
            if m == 0: # no cambered airfoil, it is symmetric
                Airfoil_x[:] = np.hstack((xL,xU[1:]))
                Airfoil_y[:] = np.hstack((-ytL,ytU[1:]))
            else:      # cambered airfoil, non-symmetric
                ycU = np.zeros(Ntop)
                ycL = np.zeros(Nbot)
                ycU[xU<=p]= m*( xU[xU<=p]/(p**2) )*(2.*p-(xU[xU<=p]))
                ycU[xU>p]= m*( (1.-xU[xU>p])/((1.-p)**2) )*(1.-2.*p+(xU[xU>p]))
                ycL[xL<=p]= m*( xL[xL<=p]/(p**2) )*(2.*p-(xL[xL<=p]))
                ycL[xL>p]= m*( (1.-xL[xL>p])/((1.-p)**2) )*(1.-2.*p+(xL[xL>p]))
                thU = np.zeros(Ntop)
                thL = np.zeros(Nbot)
                thU[xU<=p]= (2.*m/(p**2))*(p-(xU[xU<=p]))
                thU[xU>p]= (2.*m/((1.-p)**2))*(p-(xU[xU>p]))
                thL[xL<=p]= (2.*m/(p**2))*(p-(xL[xL<=p]))
                thL[xL>p]= (2.*m/((1.-p)**2))*(p-(xL[xL>p]))
                thU = np.arctan(thU); thL = np.arctan(thL)
                xU = xU - ytU*np.sin(thU); yU = ycU + ytU*np.cos(thU)
                xL = xL + ytL*np.sin(thL); yL = ycL - ytL*np.cos(thL)
                Airfoil_x[:] = np.hstack((xL,xU[1:]))
                Airfoil_y[:] = np.hstack((yL,yU[1:]))

        elif len(NACAidentifier) == 5: # 5-digit NACA
            cld = float(NACAidentifier[0]) *(3./2.)*0.1
            p = float(NACAidentifier[1])
            if p > 5:
                print ('Warning: second digit of 5-digit NACA identifier > 5, switched to 5')
                p = 5
            p   /= 20.
            q   = int(NACAidentifier[2])
            t   = float(NACAidentifier[3:])*0.01
            if q == 0: # standard
                P   = np.array([  0.05,     0.1,     0.15,    0.2,     0.25  ])
                R   = np.array([  0.0580,   0.1260,  0.2025,  0.2900,  0.3910])
                K   = np.array([361.4,     51.64,   15.957,   6.643,   3.230 ])
            else: # reflex
                P   = np.array([  0.1,      0.15,    0.2,     0.25  ])
                R   = np.array([  0.13,     0.217,   0.318,   0.441 ])
                K   = np.array([ 51.99,    15.793,   6.520,   3.191 ])
                K2K1= np.array([  0.000764, 0.00677, 0.0303,  0.1355])

            ytU = 5.*t*(a0*np.sqrt(xU)+a1*(xU)+a2*(xU)**2+a3*(xU)**3+a4*(xU)**4)
            ytL = 5.*t*(a0*np.sqrt(xL)+a1*(xL)+a2*(xL)**2+a3*(xL)**3+a4*(xL)**4)
            Airfoil = D.line((0,0,0), (1,0,0), Ntop+Nbot-1)
            Airfoil_x, Airfoil_y = J.getxy(Airfoil)

            if p == 0: # no cambered airfoil, it is symmetric
                Airfoil_x[:] = np.hstack((xL,xU[1:]))
                Airfoil_y[:] = np.hstack((-ytL,ytU[1:]))
            else:      # cambered airfoil, non-symmetric

                try: import scipy.interpolate
                except: raise ImportError("%s: This usage of airfoil requires scipy interpolate module."%__file__)

                inter_pr = scipy.interpolate.UnivariateSpline(P,R)
                inter_pk = scipy.interpolate.UnivariateSpline(P,K)
                r = inter_pr(p)
                k1= inter_pk(p)
                ycU = np.zeros(Ntop)
                ycL = np.zeros(Nbot)
                Scale = cld/0.3
                thU = np.zeros(Ntop)
                thL = np.zeros(Nbot)
                if q==0: # standard equations
                    ycU[xU<=r]= Scale*(k1/6.)*((xU[xU<=r])**3 - 3*r*(xU[xU<=r])**2 + (r**2)*(3-r)*(xU[xU<=r]) )
                    ycU[xU>r]= Scale*(k1/6.)*(r**3)*(1-(xU[xU>r]))
                    ycL[xL<=r]= Scale*(k1/6.)*((xL[xL<=r])**3 - 3*r*(xL[xL<=r])**2 + (r**2)*(3-r)*(xL[xL<=r]) )
                    ycL[xL>r]= Scale*(k1/6.)*(r**3)*(1-(xL[xL>r]))
                    thU[xU<=r]= Scale*(k1/6.)* ( 3.*(xU[xU<=r])**2 - 6.*r*(xU[xU<=r]) + (r**2)*(3.-r) )
                    thU[xU>r]= -Scale*(k1/6.)*(r**3)*(xU[xU>r]*0)
                    thL[xL<=r]= Scale*(k1/6.)* ( 3.*(xL[xL<=r])**2 - 6.*r*(xL[xL<=r]) + (r**2)*(3.-r) )
                    thL[xL>r]= -Scale*(k1/6.)*(r**3)*(xL[xL>r]*0)
                else:   # reflex equations
                    inter_pk2k1 = scipy.interpolate.UnivariateSpline(P,K2K1)
                    k2k1 = inter_pk2k1(p)
                    ycU[xU<=r]= Scale*(k1/6.)*((xU[xU<=r] - r)**3 -k2k1*(xU[xU<=r])*((1-r)**3 - r**3 ) + r**3 )
                    ycU[xU>r]= Scale*(k1/6.)*(k2k1*(xU[xU>r] - r)**3 -k2k1*(xU[xU>r])*((1-r)**3 - r**3 ) + r**3 )
                    ycL[xL<=r]= Scale*(k1/6.)*((xL[xL<=r] - r)**3 -k2k1*(xL[xL<=r])*((1-r)**3 - r**3 ) + r**3 )
                    ycL[xL>r]= Scale*(k1/6.)*(k2k1*(xL[xL>r] - r)**3 -k2k1*(xL[xL>r])*((1-r)**3 - r**3 ) + r**3 )
                    thU[xU<=r]= Scale*(k1/6.)*(3.*(xU[xU<=r] - r)**2 -k2k1*((1-r)**3) - r**3 )
                    thU[xU>r]= Scale*(k1/6.)*(k2k1*3.*(xU[xU>r] - r)**2 -k2k1*((1-r)**3) - r**3 )
                    thL[xL<=r]= Scale*(k1/6.)*(3.*(xL[xL<=r] - r)**2 -k2k1*((1-r)**3) - r**3 )
                    thL[xL>r]= Scale*(k1/6.)*(k2k1*3.*(xL[xL>r] - r)**2 -k2k1*((1-r)**3) - r**3 )
                thU = np.arctan(thU); thL = np.arctan(thL)
                xU = xU - ytU*np.sin(thU); yU = ycU + ytU*np.cos(thU)
                xL = xL + ytL*np.sin(thL); yL = ycL - ytL*np.cos(thL)
                Airfoil = D.line((0,0,0), (1,0,0), Ntop+Nbot-1)
                Airfoil_x, Airfoil_y = J.getxy(Airfoil)
                Airfoil_x[:] = np.hstack((xL,xU[1:]))
                Airfoil_y[:] = np.hstack((yL,yU[1:]))

    else:
        print ('airfoil: designation "',designation,'" not recognized.')
        return -1
    # Scaling
    Airfoil_x[:] *= ChordLength
    Airfoil_y[:] *= ChordLength

    TrailingEdgeDistance = np.sqrt(
        (Airfoil_x[-1]-Airfoil_x[0])**2+
        (Airfoil_y[-1]-Airfoil_y[0])**2)

    if TrailingEdgeDistance <= ClosedTolerance:
        Airfoil_x[-1] = Airfoil_x[0] = 0.5*(Airfoil_x[-1]+Airfoil_x[0])
        Airfoil_y[-1] = Airfoil_y[0] = 0.5*(Airfoil_y[-1]+Airfoil_y[0])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    #                           REFINEMENT                      #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Splits Top and Bottom parts
    Split_Index = np.argmin(Airfoil_x)
    Top = T.subzone(Airfoil,(Split_Index+1,1,1),(-1,1,1))
    Top_x, Top_y = J.getxy(Top)
    Bottom = T.subzone(Airfoil,(1,1,1),(Split_Index+1,1,1))
    Bottom_x, Bottom_y = J.getxy(Bottom)

    # Top side Refinement
    if not not TopDistribution:
        if not Ntop:
            Ntop = len(Top_x)
        print ('Ntop=',Ntop)
        if 'InterpolationAxe' in TopDistribution:
            if TopDistribution['InterpolationAxe'] == 'X':
                # Interpolates using the X axe as a reference
                Top_Line = linelaw(P1=(np.min(Top_x),0,0), P2=(np.max(Top_x),0,0), N=Ntop, Distribution=TopDistribution)
            else:
                # By default, interpolates following the curvilinear abscissa
                Top_Line = linelaw(P2=(D.getLength(Top),0,0), N=Ntop, Distribution=TopDistribution)
        else:
            # By default, interpolates following the curvilinear abscissa
            Top_Line = linelaw(P2=(D.getLength(Top),0,0), N=Ntop, Distribution=TopDistribution)
        Top_Dist = D.getDistribution(Top_Line)
        Top = G.map(Top, Top_Dist)

    # Bottom side Refinement
    if not not BottomDistribution:
        if not Nbot: Nbot = len(Bottom_x)
        if 'InterpolationAxe' in BottomDistribution:
            if BottomDistribution['InterpolationAxe'] == 'X':
                # Interpolates using the X axe as a reference
                Bottom_Line = linelaw(P1=(np.min(Bottom_x),0,0), P2=(np.max(Bottom_x),0,0), N=Nbot, Distribution=BottomDistribution)
            else:
                # By default, interpolates following the curvilinear abscissa
                Bottom_Line = linelaw(P2=(D.getLength(Bottom),0,0), N=Nbot, Distribution=BottomDistribution)
        else:
            # By default, interpolates following the curvilinear abscissa
            Bottom_Line = linelaw(P2=(D.getLength(Bottom),0,0), N=Nbot, Distribution=BottomDistribution)
        Bottom_Line_x = J.getx(Bottom_Line)
        Bottom_Line_x[:] = np.flipud(Bottom_Line_x)
        Bottom_Dist = D.getDistribution(Bottom_Line)
        Bottom = G.map(Bottom, Bottom_Dist)

    # Constructs the final Airfoil from top and bottom sides
    Top_x, Top_y = J.getxy(Top)
    Bottom_x, Bottom_y = J.getxy(Bottom)
    Airfoil = linelaw(N=len(Top_x)+len(Bottom_x)-1)
    Airfoil[0] = designation.split('.')[0]
    Airfoil_x, Airfoil_y = J.getxy(Airfoil)
    Airfoil_x[:len(Bottom_x)] = Bottom_x
    Airfoil_x[len(Bottom_x):] = Top_x[1:]
    Airfoil_y[:len(Bottom_y)] = Bottom_y
    Airfoil_y[len(Bottom_y):] = Top_y[1:]

    return Airfoil



def discretize(curve, N=None, Distribution=None, MappingLaw='Generator.map'):
    '''
    *(Re)*-discretize a *(ideally dense)* curve, using **N** points and
    following a distribution governed by python dictionary
    **Distribution** and using the technique specified in **MappingLaw**.

    .. important :: this function does not migrate the **curve**
        fields contained in nodes of type ``FlowSolution_t``. If desired, user
        may want to perform a :py:func:`Post.extractMesh` operation *after*
        calling this function

    Parameters
    ----------

        curve : zone
            Structured Zone PyTree corresponding to the curve to be
            (re)discretized

        N : int
            Number of desired points of final discretization

        Distribution : :py:class:`dict` or zone
            if the provided type is a :py:class:`dict`, then it is supposed to
            be a distribution dictionary compatible with :py:func:`linelaw`. If
            the provided input is a zone, then it is supposed to be a curve
            whose distribution is to be copied.

        MappingLaw : str
            Choose the discretization algorithm:

            * ``'Generator.map'``
                employs :py:func:`Generator.map` function

            * Law
                Any **Law** attribute (:py:class:`str`) supported by function
                :py:func:`MOLA.InternalShortcuts.interpolate__`

    Returns
    -------

        MyCurve : zone
            new Zone PyTree corresponding to the new curve's
            discretization

    '''

    if not N: N = C.getNPts(curve)

    if I.isStdNode(Distribution) == -1:
        curve_Distri = Distribution
    else:
        curve_Length = D.getLength(curve)
        curve_Distri = linelaw(P2=(curve_Length,0,0), N=N,
                        Distribution=Distribution)


    if MappingLaw == 'Generator.map':
        return G.map(curve,D.getDistribution(curve_Distri))
    else:
        s        = gets(curve)
        sMap     = gets(curve_Distri)
        x, y, z  = J.getxyz(curve)
        curveMap = D.line((0,0,0),(1,0,0),C.getNPts(curve_Distri))
        curveMap[0] = curve[0]+'.mapped'
        xMap, yMap, zMap = J.getxyz(curveMap)
        xMap[:] = J.interpolate__(sMap, s, x, Law=MappingLaw, axis=-1)
        yMap[:] = J.interpolate__(sMap, s, y, Law=MappingLaw, axis=-1)
        zMap[:] = J.interpolate__(sMap, s, z, Law=MappingLaw, axis=-1)

        return curveMap


def copyDistribution(curve):
    '''
    Copy the distribution of a curve.

    .. note:: this is a dimensional version of :py:func:`Geom.PyTree.getDistribution`

    Parameters
    ----------

        curve : zone
            structured curve

    Returns
    -------

        distribution : zone
            structured curve with :math:`x \in [0, \mathrm{L}]` where :math:`L`
            is the length of the provided **curve**
    '''
    Length = D.getLength(curve)
    dist = D.getDistribution(curve)
    x = J.getx(dist)
    x *= Length

    return dist


def concatenate(curves):
    '''
    Given a list of curves, this function joins all curves
    following the input order, concatenating the ending point
    of one curve with the starting point of the next curve,
    *regardless* of its actual distance.

    .. note:: this function also copies the fields contained at ``FlowSolution``
        container defined at ``Internal.__FlowSolutionNodes__``

    Parameters
    ----------
        curves : list
            List of PyTree Structured Zones of curves to be concatenated

    Returns
    -------
        mergedCurves : zone
            concatenated curve (structured zone)
    '''


    # Compute the amount of points of the final concatenated curve
    Ntot = 0
    for c in curves: Ntot += C.getNPts(c)
    SegmentsQty = len(curves)

    # Store the amount of FlowSolutions names to be invoked
    FlSolNames = []
    for c in curves:
        VarNames = C.getVarNames(c, excludeXYZ=True, loc='nodes')[0]
        for vn in VarNames:
            if vn not in FlSolNames: FlSolNames += [vn]
    FieldsQty = len(FlSolNames)


    concatenated        = linelaw(N=Ntot)
    ConcX, ConcY, ConcZ = J.getxyz(concatenated)
    ConcVars            = J.invokeFields(concatenated,FlSolNames)

    ListX, ListY, ListZ = [[] for i in range(3)] #map(lambda i: [] ,range(3))
    ConcVarsList        = [[] for i in range(FieldsQty)] #map(lambda i: [] ,range(FieldsQty))

    for i in range(SegmentsQty):

        # Store the GridCoordinates in list form
        cx, cy, cz  = J.getxyz(curves[i])
        ListX      += [cx]
        ListY      += [cy]
        ListZ      += [cz]


        # Store the FlowSolutions in list form
        cPts        = len(cx)
        cVars       = J.getVars(curves[i],FlSolNames)
        for j in range(FieldsQty):
            cVar = cVars[j]
            if cVar is not None:
                ConcVarsList[j] += [cVar]
            else:
                # Only suitable to FlowSolutions located at nodes
                VarLength = cPts
                ConcVarsList[j] += [np.full(VarLength,BADVALUE)]

    # Migrate data by numpy stacking
    ConcX[:], ConcY[:], ConcZ[:] = np.hstack(ListX), np.hstack(ListY), np.hstack(ListZ)
    for j in range(FieldsQty): ConcVars[j][:] = np.hstack(ConcVarsList[j])

    return concatenated


def polyDiscretize(curve, Distributions, MappingLaw='Generator.map'):
    '''
    *(Re)*-discretize a *(ideally dense)* structured curve using the
    instructions provided by the :py:class:`list` of **Distributions**, whose
    items are dictionaries used to discretize a portion of the curve,
    which is spatially determined using the key ``'BreakPoint'``.

    .. important :: this function does not migrate the **curve**
        fields contained in nodes of type ``FlowSolution_t``. If desired, user
        may want to perform a :py:func:`Post.extractMesh` operation *after*
        calling this function


    Parameters
    ----------

        curve : zone
            Structured Zone PyTree. Curve to be (re)discretized.

        Distributions : list
            List of Python dictionaries. Each :py:class:`dict` defines a
            distribution concerning a rediscretization interval.
            Acceptable values are the same as **Distribution** attribute of
            function  :py:func:`linelaw`.

            Each dictionary **must** contain, in **addition**, the following keys:

            * N : :py:class:`int`
                Number of points of the interval

            * BreakPoint : :py:class:`float`
                must be :math:`\in (0,1]`. Determines the
                breakpoint up to where the discretization is
                applied based on the curve's curvilinear
                abscissa, starting from the previous Break-
                point.

    Returns
    -------

        curveDisc : zone
            Structured Zone PyTree. Newly discretized curve.



    Examples
    --------

    ::

        curveDisc = W.polyDiscretize(curve,[{'N':10,'BreakPoint':0.2},
                                            {'N':10,'BreakPoint':1.0}])

    '''


    L = D.getLength(curve)
    s = gets(curve)
    s0 = 0.
    Segments = []
    Ntot = 0

    prevLength_segment = 0.
    for d in Distributions:
        N = d['N']
        s1 = d['BreakPoint']
        Lenght_segment = L*(s1-s0)
        Segment_Distri = linelaw(P1=(prevLength_segment,0,0),
            P2=(Lenght_segment+prevLength_segment,0,0), N=N, Distribution=d)
        Segments += [Segment_Distri]
        s0 = s1
        Ntot += N
        prevLength_segment += Lenght_segment

    SegmentsQty = len(Segments)
    if SegmentsQty>1: Ntot -= SegmentsQty-1

    joined = Segments[0]
    for i in range(1,len(Segments)):
        joined = T.join(joined,Segments[i])


    if MappingLaw == 'Generator.map':
        return G.map(curve,D.getDistribution(joined))
    else:
        s        = gets(curve)
        sMap     = gets(joined)
        x, y, z  = J.getxyz(curve)
        curveMap = D.line((0,0,0),(1,0,0),len(sMap))
        curveMap[0] = curve[0]+'.mapped'
        xMap, yMap, zMap = J.getxyz(curveMap)
        xMap[:] = J.interpolate__(sMap, s, x, Law=MappingLaw, axis=-1)
        yMap[:] = J.interpolate__(sMap, s, y, Law=MappingLaw, axis=-1)
        zMap[:] = J.interpolate__(sMap, s, z, Law=MappingLaw, axis=-1)

        return curveMap




def getAbscissaFromCoordinate(curve, station, coordinate='x'):
    '''
    .. danger:: **getAbscissaFromCoordinate** is being deprecated.
        Use :py:func:`getAbscissaAtStation` instead.
    '''
    print ('Warning: getAbscissaFromCoordinate() is being deprecated. Use getAbscissaAtStation() instead.')
    s = gets(curve)
    if   coordinate.lower() == 'y': x=J.gety(curve)
    elif coordinate.lower() == 'z': x=J.getz(curve)
    else: x = J.getx(curve)

    Npts = len(x)
    # Split as many pieces as changes in monotonicity
    diffx = np.diff(x)
    diffx = np.hstack((diffx,diffx[-1]))
    x_array, s_array = [], []
    x_piece, s_piece = [], []
    Pieces = 1
    for i in range(Npts):
        if i == Npts-1:
            x_piece += [x[i]]
            s_piece += [s[i]]
        else:
            if np.sign(diffx[i+1]) == np.sign(diffx[i]):
                x_piece += [x[i]]
                s_piece += [s[i]]
            else:
                x_piece += [x[i]]
                s_piece += [s[i]]
                x_array += [np.array(x_piece)]
                s_array += [np.array(s_piece)]
                x_piece, s_piece = [], []
                Pieces += 1
    x_array += [np.array(x_piece)]
    s_array += [np.array(s_piece)]

    # Produces multiple solutions
    Sol = np.zeros(Pieces, dtype=np.float64, order='F')
    for i in range(Pieces):
        if len(x_array[i])==1:
            Sol[i]=s_array[i][0]
        else:
            Reverse = int(np.sign(np.diff(x_array[i])[0]))
            # interpFunc = scipy.interpolate.interp1d( x_array[i][::Reverse], s_array[i][::Reverse], kind='linear', bounds_error=False)
            # Sol[i] = interpFunc(station)
            Sol[i] = np.interp(station, s_array[i][::Reverse], x_array[i][::Reverse])

    Sol = Sol[np.logical_not(np.isnan(Sol))]

    return Sol

def getAbscissaAtStation(curve, station, coordinate='x'):
    '''
    From a provided **curve**, compute the *(possibly multiple)* abscissa
    points where the curve intersects the plane of constant coordinate
    :math:`(x, y, z)` at provided **station** value.

    Parameters
    ----------

        curve : zone
            PyTree Zone curve, BAR or STRUCT.

        station : float
            Will define the intersecting plane

        coordinate : str
            string in [``'x'``, ``'y'``, ``'z'``]. Define the constant
            coordinate plane.

    Returns
    -------

        result : list
            List of floats :math:`\in [0,1]`. Each item defines the
            corresponding curvilinear abscissa :math:`s` value

    Examples
    --------

    >>> Abscissas = W.getAbscissaAtStation(MyAirfoil, station=0.1)
    '''


    curve = D.getCurvilinearAbscissa(curve)
    if   coordinate.lower() == 'x':
        n  = np.array([1.0,0.0,0.0])
        Pt = np.array([station,0.0,0.0])
    elif coordinate.lower() == 'y':
        n  = np.array([0.0,1.0,0.0])
        Pt = np.array([0.0,station,0.0])
    elif coordinate.lower() == 'z':
        n  = np.array([0.0,0.0,1.0])
        Pt = np.array([0.0,0.0,station])
    else:
        raise AttributeError('%s: getAbscissaAtStation coordinate %s not in ("x","y","z").'%(__file__,coordinate))


    # plane equation used for slice (c1,c2,c3,c4)
    PlaneCoefs = n[0],n[1],n[2],-n.dot(Pt)
    C._initVars(curve,'SliceVar=%0.12g*{CoordinateX}+%0.12g*{CoordinateY}+%0.12g*{CoordinateZ}+%0.12g'%PlaneCoefs)
    Slice = P.isoSurfMC(curve,'SliceVar',value=0.0)[0]
    s, = J.getVars(Slice,['s'])

    return s



def makeFillet2D(c1,c2,R,position=1,reverseFillet=False,intersectionElement=0):
    '''
    Perform a fillet of radius **R** between two curves (**c1**, **c2**)
    that lie on the :math:`OXY` plane.

    This function is expected to be used just before :py:func:`joinFillet`.

    Parameters
    ----------

        c1 : zone
            PyTree Zone. First curve.

        c2 : zone
            PyTree Zone. Second curve.

        R : float
            Fillet radius.

        position : int
            Must be in (1,2,3,4). Used to choose among the 4
            different geometrical possible results.

        reverseFillet : bool
            Used to eventually change the fillet
            complementary angle of the circle.

        intersectionElement : int
            Used to choose among the different points where **c1** and **c2** may
            intersect.

    Returns
    -------

        Fillet : zone
            PyTree Structured Zone. Arc of circle of the Fillet.
    '''

    import Intersector.PyTree as XOR

    if   position == 1: c1dir, c2dir =  1, 1
    elif position == 2: c1dir, c2dir = -1, 1
    elif position == 3: c1dir, c2dir =  1,-1
    elif position == 4: c1dir, c2dir = -1,-1
    else: raise ValueError('makeFillet: position shall be in (1,2,3,4)')
    c1n, c2n = cdeep(c1), cdeep(c2)
    c1x, c1y = A.getxy(c1n)
    c2x, c2y = A.getxy(c2n)
    c1Tx, c1Ty = A.getxy(D.getTangent(c1n))
    c2Tx, c2Ty = A.getxy(D.getTangent(c2n))
    for i in range(len(c1x)):
        c1x[i] +=  c1dir*c1Ty[i] * R
        c1y[i] += -c1dir*c1Tx[i] * R
    for i in range(len(c2x)):
        c2x[i] +=  c2dir*c2Ty[i] * R
        c2y[i] += -c2dir*c2Tx[i] * R

    try:
        Center = XOR.intersection(c1n,c2n)
    except:
        c1n[0] += '_2beIntersected'
        c2n[0] += '_2beIntersected'
        t = C.newPyTree(['FailedIntersection',[c1,c2,c1n,c2n]])
        C.convertPyTree2File(t,'FailedIntersection.cgns')
        raise AttributeError("makeFillet: Could not perform intersection. Try different position value.")

    Cx,Cy,Cz=A.getxyz(Center)
    if len(Cx)>1:
        try:
            Cx,Cy,Cz = Cx[intersectionElement],Cy[intersectionElement],Cz[intersectionElement]
        except IndexError:
            raise ValueError('intersectionElement=%d out of range (size=%d). Reduce this value.'%(intersectionElement,len(Cx)))
    Center = D.point((Cx,Cy,Cz))
    p1 = T.projectOrtho(Center,c1); p1[0]='p1'
    p2 = T.projectOrtho(Center,c2); p2[0]='p2'

    a1 = angle(Center,p1)
    a2 = angle(Center,p2)
    if reverseFillet:
        if a1<0: a1 +=360
        if a2<0: a2 +=360
    tetas, tetae = a1, a2
    # if tetae-tetas<0: tetas, tetae = 360+tetae, tetas # Should try this instead
    Fillet = D.circle((Cx,Cy,Cz),R,tetas,tetae,100); Fillet[0]='Fillet'

    return Fillet



def joinFillet(c1, Fillet, c2, c1dir=True, c2dir=True):
    '''
    Join the three elements of a Fillet. Resulting geometry
    is the joined smooth curve **c1** + **Fillet** + **c2**, where **c1** and
    **c2** have been conveniently split.

    Parameters
    ----------

        c1 : zone
            PyTree Structured Zone. First curve.

        Fillet : zone
            PyTree Structured Zone. Arc of circle (fillet)

        c2 : zone
            PyTree Structured Zone. Second curve.

        c1dir : bool
            Use to choose the preserved split portion of the **c1** curve.

        c2dir : bool
            Use to choose the preserved split portion of the **c2** curve.

    Returns
    -------

        joined : zone
            PyTree Structured Zone. Joined curve **c1** + **Fillet** + **c2**
    '''
    Fx, Fy, Fz = A.getxyz(Fillet)
    Fstart = (Fx[0], Fy[0],Fz[0])
    Fend   = (Fx[-1],Fy[-1],Fz[-1])

    # Connection 1
    if not c1dir: c1 = T.reorder(c1,(-1,2,3))
    if not c2dir: c2 = T.reorder(c2,(-1,2,3))
    c1_Fstart = J.getNearestPointIndex(c1,Fstart)[0]
    c1s = T.subzone(c1,(1,1,1),(c1_Fstart+1,1,1))

    c1sx,c1sy,c1sz = A.getxyz(c1s)
    c1sx[-1] = Fx[0]
    c1sy[-1] = Fy[0]
    c1sz[-1] = Fz[0]

    # Connection 2
    c2_Fend = J.getNearestPointIndex(c2,Fend)[0]
    c2s = T.subzone(c2,(c2_Fend+1,1,1),(-1,-1,-1))
    c2sx,c2sy,c2sy = A.getxyz(c2s)
    c2sx[0] = Fx[-1]
    c2sy[0] = Fy[-1]
    c2sz[0] = Fz[-1]

    # Join results
    joined = T.join(c1s,Fillet)
    joined = T.join(joined,c2s)

    return joined


def splitCurves(c1,c2,select=0,tol=1e-6):
    '''
    Split intersecting curves (**c1**, **c2**) into different portions.

    Parameters
    ----------

        c1 : zone
            Zone PyTree. First curve.

        c2 : zone
            Zone PyTree. Second curve.

        select : int
            May be one of:

            * ``0``
                no selection. All subparts are returned by function.

            * ``1``
                Only subparts belonging to **c1** are returned

            * ``2``
                Only subparts belonging to **c2** are returned

        tol : float
            tolerance. Squared distance used for determining
            which subparts correspond to each curve. Shall be small.

    Returns
    -------

        SplitCurves : list
            List of PyTree Zones. Corresponds to the
            different portions of the split.
    '''

    import Intersector.PyTree as XOR

    cu = C.convertArray2Tetra([c1,c2]);
    Int = XOR.conformUnstr(cu[0],cu[1],left_or_right=2)
    Res = T.splitTBranches(Int, tol=1.e-13)
    Str = [C.convertBAR2Struct(r) for r in Res]

    if   select == 0:
        Accepted = Str

    elif select == 1:
        c1WithIntersection = XOR.conformUnstr(cu[0],cu[1],left_or_right=0)
        Accepted = [c for c in Str if isSubzone(c,c1WithIntersection,tol)]

    elif select == 2:
        c2WithIntersection = XOR.conformUnstr(cu[0],cu[1],left_or_right=1)
        Accepted = [c for c in Str if isSubzone(c,c2WithIntersection,tol)]

    else:
        raise AttributeError("select = '%s' not recognized. It must be an integer: 0, 1 or 2."%str(select))

    return Accepted

def isSubzone(subzone,zone,tol=1.e-10):
    '''
    Check if a block is *totally* contained in another block.
    Hence, all points of **subzone** matches (between tolerance) some
    points of **zone**.

    Parameters
    ----------

        subzone : zone
            zone that function will determine if is subzone or not

        zone : zone
            zone where **subzone** may be contained

        tol : float
            tolerance squared distance used for determining result of function

    Returns
    -------

        ItIs : bool
            :py:obj:`True` if **subzone** totally lies in **zone**, :py:obj:`False` otherwise
    '''
    xs,ys,zs = J.getxyz(subzone)
    xs_ = xs.ravel(order='K')
    ys_ = ys.ravel(order='K')
    zs_ = zs.ravel(order='K')
    NPtsSub = len(xs_)

    NPts = C.getNPts(zone)

    if NPtsSub > NPts: return False

    SubzonePoints = [(xs_[i],ys_[i],zs_[i]) for i in range(NPtsSub)]

    # TODO optimize this
    Res = D.getNearestPointIndex(zone,SubzonePoints)
    sqdist = np.array([i[1] for i in Res])
    if np.any(sqdist>tol): return False

    return True


def addWakeLines2D(foil, WakePts, WakeLength, Name='Wake'):
    '''
    Construct a simple Wake line from the extrema of an airfoil
    (from its Trailing Edge). Useful for C-type grids extrusion.



    Parameters
    ----------

        foil : zone
            PyTree Zone. Corresponds to the airfoil contour.

            .. attention:: **foil** must be placed in :math:`OXY` plane

        WakePts : int
            Number of points to add in the wake

        WakeLength : float
            Length of the wake

        Name : str
            name of the newly created zone

    Returns
    -------

        Merged : zone
            PyTree Zone of the joined curve **foil** + **wakes**
    '''

    # TODO Add optional arguments defaulted to None:
    #     WakeTension = To control tension of wake
    #     WakeAngle   = To control angle of wake
    #     WakeDivergenceDistance = To control divergence
    #     WakeLastCellLength = (explicit)


    foil_x, foil_y = A.getxy(foil)
    foilNpts       = len(foil_x)
    foilTang = D.getTangent(foil)
    Tx, Ty = A.getxy(foilTang)
    WakeDir = 0.5*(-Tx[0]+Tx[-1]),0.5*(-Ty[0]+Ty[-1])
    panelLength = np.sqrt((foil_x[1]-foil_x[0])**2+(foil_y[1]-foil_y[0])**2)
    wa     = A.linelaw(P1=(foil_x[0],foil_y[0],0),P2=(foil_x[0]+WakeDir[0]*WakeLength,foil_y[0]+WakeDir[1]*WakeLength,0),
                         N=WakePts,Distribution={'kind':'tanhOneSide','FirstCellHeight':panelLength})
    wax,way = A.getxy(wa)


    MergedPts      = 2*WakePts+foilNpts-2
    Merged         = D.line((0,0,0),(1,0,0),MergedPts);
    Merged[0]      = Name
    Mx, My         = A.getxy(Merged)
    Mx[:WakePts]  = wax[::-1]
    My[:WakePts]  = way[::-1]
    Mx[WakePts-1:WakePts+foilNpts-1]= foil_x[:]
    My[WakePts-1:WakePts+foilNpts-1]= foil_y[:]
    Mx[-WakePts:] = wax[:]
    My[-WakePts:] = way[:]

    return Merged

def isCurveClosed(AirfoilCurve, tol=1.e-10):
    '''
    Check if provided curve is closed by its extrema points,
    given a tolerance tol. May be Unstructured (curve is first
    converted to structured).

    Parameters
    ----------

        curve : zone
            structured zone corresponding to the curve

        tol : float
            geometrical tolerance of the criterion.

    Returns
    -------

        Result : bool
            :py:obj:`True` if it is closed, :py:obj:`False` if it is open.
    '''

    isStructured = checkAirfoilAndGetTopo(AirfoilCurve)

    if not isStructured: AirfoilCurve = C.convertBAR2Struct(AirfoilCurve)

    x, y, z = J.getxyz(AirfoilCurve)

    ExtremaDistance = ( ( x[-1] - x[0] )**2 +
                        ( y[-1] - y[0] )**2 +
                        ( z[-1] - z[0] )**2   ) ** 0.5

    return True if ExtremaDistance < tol else False


def closeCurve(curve,NPts4closingGap=3, tol=1e-10):
    '''
    Close a curve by its extrema points, by making a line with
    user-provided number of points.

    Parameters
    ----------

        curve : zone
            structured Zone PyTree

        NPts4closingGap : int
            number of points discretizing the gap after closing the curve

        tol : float
            geometrical tolerance

    Returns
    -------

        ClosedCurve : zone
            Structured ZonePyTree.
    '''

    isStructured = checkAirfoilAndGetTopo(curve)

    if not isStructured: curve = C.convertBAR2Struct(curve)

    if not isCurveClosed(curve, tol=tol):
        x, y, z = J.getxyz(curve)

        JoinLine = D.line( (x[-1], y[-1], z[-1]),
                           (x[ 0], y[ 0], z[ 0]),
                           NPts4closingGap       )

        ClosedCurve = T.join(curve, JoinLine)

    else:
        ClosedCurve = curve


    return ClosedCurve

def _rootScalar__(func, x0, xmin=None, xmax=None, args=None, maxstep=1.e-3,
                maxiter=100,tol=1.e-8):
    '''
    Private root-finding function
    '''
    f0 = x0*0
    for i in range(len(x0)): f0[i] = func(x0[i],*args)

    iMin = np.nanargmin(f0)
    x  = x0[iMin]
    f  = f0[iMin]
    try:
        x1 = x0[iMin+1]
        f1 = f0[iMin+1]
    except IndexError:
        x1 = x0[iMin-1]
        f1 = f0[iMin-1]
        pass
    it=0
    while (it < maxiter) and (abs(f)>tol):
        it += 1
        xOld, x1Old, fOld, f1Old = x, x1, f, f1
        if f1==f:
            # print ('_rootScalar__() FAILED FOR x=%g, f=%g DUE TO STAGNATION'%(x,f))
            break
        xS = np.minimum(np.maximum(x-f*((x1-x)/(f1-f)),xmin),xmax)
        xS = x+np.minimum((xS-x),maxstep)
        fS = func(xS,*args)
        if np.isnan(fS):
            print ('_rootScalar__() FAILED FOR xS=%g, fS=%g'%(xS,fS))
            break
        x1, f1 = xOld, fOld
        x, f   = xS, fS
        # print ('it=%g, R=%g, Distance=%g'%(it,x,f))
    Converged = abs(f)<=tol
    return x, f, Converged

def extrapolate(curve, ExtrapDistance, mode='tangent', opposedExtremum=False):
    '''
    Extrapolate a curve from one of its boundaries.

    Parameters
    ----------

        curve : zone
            structured curve

        ExtrapDistance : float
            distance to extrude [m]

        mode : str
            choice of the mode of extrusion: currently, only ``'tangent'``
            is available

        opposedExtremum : bool
            if :py:obj:`True`, extrapolates starting from last index of **curve**

    Returns
    -------

        ExtrapolatedCurve : zone
            extrapolated curve (structured)
    '''

    if opposedExtremum: T._reorder(curve,(-1,2,3))

    cX, cY, cZ = J.getxyz(curve)
    Tangent    = D.getTangent(curve)
    tX, tY, tZ = J.getxyz(Tangent)

    if mode=='tangent':

        Points = [(cX[-1],cY[-1],cZ[-1])]
        Pt = Points[-1]
        Points += [(Pt[0]+tX[-1]*ExtrapDistance,
                    Pt[1]+tY[-1]*ExtrapDistance,
                    Pt[2]+tZ[-1]*ExtrapDistance)]
        Appendix = D.polyline(Points)
        ExtrapolatedCurve = T.join(curve,Appendix)


    if opposedExtremum: T._reorder(ExtrapolatedCurve,(-1,2,3))

    return ExtrapolatedCurve

def findLeadingEdgeAndSplit(Airfoil, LeadingEdgeCmin=0.5):
    '''
    .. danger:: **THIS FUNCTION IS BEING DEPRECATED. MUST BE REPLACED BY**
     :py:func:`findLeadingOrTrailingEdge`

    find the leading edge point and split the airfoil.

    This function supposes that airfoil is normalized (chordwise direction
    is on X-axis).

    Parameters
    ----------

        Airfoil : zone
            airfoil positioned through x-axis chordwise

        LeadingEdgeCmin : float
            relative value of X-coordinate (over chord) where to search leading edge

    Returns
    -------

        iLE : int
            nearest point index corresponding to leading edge

        Top : zone
            top side curve of the airfoil

        Bottom : zone
            bottom side curve of the airfoil
    '''


    FoilX, FoilY = J.getxy(Airfoil)
    D._getCurvatureRadius(Airfoil)
    Chord = FoilX.max()-FoilX.min()
    rad, = J.getVars(Airfoil,['radius'])

    # Find Leading Edge Method 1
    # LEignoreInterval = (FoilX - FoilX.min())/Chord > LeadingEdgeCmin
    # rad[LEignoreInterval] = 1e6 # Very Big value (will be bypassed)
    # iLE = np.argmin(rad)

    # Find Leading Edge Method 2
    LEinterval = (FoilX - FoilX.min())/Chord < LeadingEdgeCmin
    MinRad = np.min(rad[LEinterval]) # Minimum Radius (many points may be close to this)
    MaxRad = np.max(rad[LEinterval])

    # Look for LE candidates
    tol = 1e-12 # as percentage of (MaxRad-MinRad) distance
    LECandidates = rad[LEinterval] <= MinRad + tol*(MaxRad-MinRad)

    # AvrgPoint is the LECandidates average
    AvrgPoint = (np.mean(FoilX[LEinterval][LECandidates]), np.mean(FoilY[LEinterval][LECandidates]), 0.)
    # AvrgPointPyTree = D.point(AvrgPoint)
    # T._projectOrtho(AvrgPointPyTree,Airfoil)
    iLE, _ = J.getNearestPointIndex(Airfoil, AvrgPoint)

    # Split Top and Bottom sides
    Bottom = T.subzone(Airfoil,(iLE+1,1,1),(-1,-1,-1))
    Top    = T.subzone(Airfoil,(1,1,1),(iLE+1,1,1))

    return iLE, Top, Bottom

def findTrailingEdgeAndSplit(Airfoil):
    '''
    .. danger:: **THIS FUNCTION IS BEING DEPRECATED. MUST BE REPLACED BY**
     :py:func:`findLeadingOrTrailingEdge`

    find the trailing edge point and split the airfoil.

    This function supposes that airfoil is normalized (chordwise direction
    is on X-axis).

    Parameters
    ----------

        Airfoil : zone
            airfoil positioned through x-axis chordwise

    Returns
    -------

        iTE : int
            nearest point index corresponding to trailing edge

        Top : zone
            top side curve of the airfoil

        Bottom :
            bottom side curve of the airfoil
    '''
    isClockwise = is2DCurveClockwiseOriented(Airfoil)

    if not isClockwise: T._reorder(Airfoil,(-1,2,3))

    # Find Trailing Edge
    AirfoilX, AirfoilY = J.getxy(Airfoil)
    iTE = np.argmax(AirfoilX)

    # Split Top and Bottom sides
    if iTE > 0:
        Bottom = T.subzone(Airfoil,(iTE+1,1,1),(-1,-1,-1))
        Top    = T.subzone(Airfoil,(2,1,1),(iTE+1,1,1))
    else:
        iLE = np.argmin(AirfoilX)
        Bottom = T.subzone(Airfoil,(1,1,1),(iLE,1,1))
        Top    = T.subzone(Airfoil,(iLE,1,1),(len(AirfoilX),1,1))

    return iTE, Top, Bottom


def getChord(Airfoil):
    '''
    .. danger:: THIS FUNCTION IS BEING DEPRECATED AND MUST BE REPLACED

    Literally, compute the airfoils chord as max(x) - min(x)
    '''

    FoilX, FoilY = J.getxy(Airfoil)
    Chord = FoilX.max()-FoilX.min()
    return Chord

def getCamber(Airfoil, CamberDistribution=None, LeadingEdgeCmin=0.5,
                       camberMinSearchPoint=0.01):
    '''
    .. danger:: THIS FUNCTION IS BEING DEPRECATED AND MUST BE REPLACED
        BY :py:func:`buildCamber`
    '''

    WRNMSG = ('WARNING: getCamber() will be deprecated.\n'
              'Please use buildCamber() instead.')
    print(WRNMSG)

    iLE, Top, Bottom = findLeadingEdgeAndSplit(Airfoil,LeadingEdgeCmin)
    # Top    = discretize(Top,N=100)    # Densify Top Side
    # Bottom = discretize(Bottom,N=100) # Densify Bottom Side
    T._reorder(Bottom,(-1,2,3))
    # Compute accurate chord
    Chord = np.maximum(getChord(Top),getChord(Bottom))
    TopX, TopY       = J.getxy(Top)
    BottomX, BottomY = J.getxy(Bottom)

    TopTan = D.getTangent(Top)
    TopTanX, TopTanY = J.getxy(TopTan)

    Zones = [Top, Bottom]



    def gapFromTangency__(R,i,TopX,TopY,TopTanX,TopTanY,Bottom):
        Center   = (TopX[i]+TopTanY[i]*R,TopY[i]-TopTanX[i]*R,0)
        CenterP  = D.point((TopX[i]+TopTanY[i]*R,TopY[i]-TopTanX[i]*R,0))

        T._projectOrtho(CenterP,Bottom)
        projX = I.getNodeFromName(CenterP, 'CoordinateX')[1][0]
        projY = I.getNodeFromName(CenterP, 'CoordinateY')[1][0]
        Distance = np.sqrt((Center[0]-projX)**2+(Center[1]-projY)**2) - R

        return Distance


    # ------------> THIS CODE MAY BE OPTIMIZED <-------------
    Centers = []
    Radius  = []
    # Circles = []
    iMinSearchIndex = np.where(((TopX - TopX.min())/Chord >= camberMinSearchPoint))[0][0]
    for i in range(iMinSearchIndex,len(TopX)-1):
        Ropt, Distance, Converged = _rootScalar__(gapFromTangency__,
            np.linspace(0.001,0.1,3),
            args=(i,TopX,TopY,TopTanX,TopTanY,Bottom),
            xmin=0.001,xmax=0.5,
            maxstep=0.5,maxiter=100,tol=1.e-8)

        if Converged:
            Center   = (TopX[i]+TopTanY[i]*Ropt,TopY[i]-TopTanX[i]*Ropt,0)
            # Circle   = D.circle(Center, Ropt,N=360*2)
            Centers += [Center]
            Radius  += [Ropt]

            # Circles += [Circle]
    # <------------------------------------------------------>




    # Join the Leading and Trailing Edge
    LeadingEdge = ((TopX[-1]+BottomX[-1])*0.5,(TopY[-1]+BottomY[-1])*0.5,0)
    TrailingEdge = ((TopX[0]+BottomX[0])*0.5,(TopY[0]+BottomY[0])*0.5,0)
    CamberLine = D.polyline([LeadingEdge]+Centers[::-1]+[TrailingEdge])

    # Re-discretize CamberLine
    if CamberDistribution is None:
        CamberDistribution = dict(N=51,
                            kind='tanhOneSide',
                            FirstCellHeight=0.0001*Chord,
                            BreakPoint=1.)
        CamberDistribution = dict(N=200,
                            kind='trigonometric',
                            parameter=2,
                            BreakPoint=1.)
    elif 'BreakPoint' not in CamberDistribution:
        CamberDistribution['BreakPoint'] = 1.


    CamberLine = polyDiscretize(CamberLine,[CamberDistribution])

    # Store distance
    TopTri = C.convertArray2Tetra(G.stack([Top,T.translate(Top,(0,0,1))]))
    BottomTri = C.convertArray2Tetra(G.stack([Bottom,T.translate(Bottom,(0,0,1))]))

    TangentCamber = D.getTangent(CamberLine)
    tX, tY, tZ = J.getxyz(TangentCamber)
    e, = J.invokeFields(CamberLine,['RelativeThickness'])
    Distances2Top    = distancesCurve2SurfDirectional(CamberLine,TopTri,tY,-tX,tX*0)
    Distances2Bottom = distancesCurve2SurfDirectional(CamberLine,BottomTri,tY,-tX,tX*0)
    e[:] = Distances2Top+Distances2Bottom # Does not look like; but this is an average ;-)
    e[-1] = np.sqrt((TopX[-1]-BottomX[-1])**2 + (TopY[-1]-BottomY[-1])**2) / Chord


    return CamberLine

def distancesCurve2SurfDirectional(Curve,Surface,DirX,DirY,DirZ):
    '''
    .. warning:: THIS FUNCTION IS COSTLY AND WILL BE REMOVED IN FUTURE
    '''
    # CurveX, CurveY, CurveZ = J.getxyz(Curve)
    CurveX = I.getNodeFromName2(Curve,'CoordinateX')[1]
    CurveY = I.getNodeFromName2(Curve,'CoordinateY')[1]
    CurveZ = I.getNodeFromName2(Curve,'CoordinateZ')[1]
    NPts = len(CurveX)
    Distances = np.zeros(NPts,order='F')
    for i in range(NPts):
        Point = D.point((CurveX[i], CurveY[i], CurveZ[i]))
        Inter = T.projectDir(Point,Surface,(DirX[i],DirY[i],DirZ[i]),oriented=0)
        pX = I.getNodeFromName2(Point,'CoordinateX')[1]
        pY = I.getNodeFromName2(Point,'CoordinateY')[1]
        pZ = I.getNodeFromName2(Point,'CoordinateZ')[1]
        iX = I.getNodeFromName2(Inter,'CoordinateX')[1]
        iY = I.getNodeFromName2(Inter,'CoordinateY')[1]
        iZ = I.getNodeFromName2(Inter,'CoordinateZ')[1]
        Distances[i] = np.sqrt( (iX-pX)**2 + (iY-pY)**2 + (iZ-pZ)**2  )
    return Distances

def projectCurve2SurfDirectional(Curve,Surface,DirX,DirY,DirZ,oriented=False):
    '''
    .. warning:: THIS FUNCTION IS COSTLY AND WILL BE REMOVED IN FUTURE
    '''
    ProjectedCurve = I.copyTree(Curve)
    CurveX, CurveY, CurveZ = J.getxyz(ProjectedCurve)
    NPts = len(CurveX)
    if not isinstance(DirX,np.ndarray):
        DirX = np.zeros(NPts,order='F')+DirX
    if not isinstance(DirY,np.ndarray):
        DirY = np.zeros(NPts,order='F')+DirY
    if not isinstance(DirZ,np.ndarray):
        DirZ = np.zeros(NPts,order='F')+DirZ

    for i in range(NPts):
        Point = D.point((CurveX[i], CurveY[i], CurveZ[i]))
        T._projectDir(Point,Surface,(DirX[i],DirY[i],DirZ[i]),oriented=oriented)
        PointX, PointY, PointZ = J.getxyz(Point)
        CurveX[i] = PointX[0]
        CurveY[i] = PointY[0]
        CurveZ[i] = PointZ[0]

    return ProjectedCurve


def getCamberOptim(Airfoil,CamberDistribution=None,LeadingEdgeCmin=0.5,method='hybr',options=None):
    '''
    .. danger:: THIS FUNCTION IS BEING DEPRECATED AND MUST BE REPLACED
        BY :py:func:`buildCamber`
    '''

    WRNMSG = ('WARNING: getCamberOptim() will be deprecated.\n'
              'Please use buildCamber() instead.')
    print(WRNMSG)

    # Find Leading Edge based on curvature
    iLE, Top, Bottom = findLeadingEdgeAndSplit(Airfoil,LeadingEdgeCmin)
    # Top    = discretize(Top,N=100)    # Densify Top Side
    # Bottom = discretize(Bottom,N=100) # Densify Bottom Side
    T._reorder(Top,(-1,2,3))
    Chord = np.maximum(getChord(Top),getChord(Bottom))
    TopX, TopY       = J.getxy(Top)
    BottomX, BottomY = J.getxy(Bottom)

    TopTri = C.convertArray2Tetra(G.stack([Top,T.translate(Top,(0,0,1))]))
    BottomTri = C.convertArray2Tetra(G.stack([Bottom,T.translate(Bottom,(0,0,1))]))

    # TODO: Externalize these parameters
    CamberLineLengthLE = 0.02
    CamberLineInitNPts = 20 # A high value of this may slow-down computation


    TopApprox    = discretize(Top,CamberLineInitNPts)
    BottomApprox = discretize(Bottom,CamberLineInitNPts)
    TopApproxX, TopApproxY = J.getxy(TopApprox)
    BotApproxX, BotApproxY = J.getxy(BottomApprox)
    PointsApprox = []
    for i in range(0,len(TopApproxX)):
        Pt1 = (0.5*(TopApproxX[i]+BotApproxX[i]),0.5*(TopApproxY[i]+BotApproxY[i]),0)
        PointsApprox += [D.point(Pt1)]
    # CamberLineInit = D.polyline(map(lambda p: J.getxyz(p) ,PointsApprox))
    CamberLineInit = D.polyline([J.getxyz(p) for p in PointsApprox])

    # Re-discretize CamberLineInit
    CamberInitDistribution = dict(N=CamberLineInitNPts,
                            # kind='tanhOneSide',
                            # FirstCellHeight=CamberLineLengthLE*Chord,
                            BreakPoint=1.)
    CamberLineInit = polyDiscretize(CamberLineInit,[CamberInitDistribution])


    CamberInitX, CamberInitY = J.getxy(CamberLineInit)
    CamberLine = I.copyTree(CamberLineInit)
    CamberX, CamberY = J.getxy(CamberLine)

    def _updateDistancesDeltas__():
        TangentCamber = D.getTangent(CamberLine)
        # tX, tY, tZ = J.getxyz(TangentCamber)
        tX = I.getNodeFromName2(TangentCamber,'CoordinateX')[1]
        tY = I.getNodeFromName2(TangentCamber,'CoordinateY')[1]
        tZ = I.getNodeFromName2(TangentCamber,'CoordinateZ')[1]

        Distances2Top    = distancesCurve2SurfDirectional(CamberLine,TopTri,tY,-tX,tX*0)
        Distances2Bottom = distancesCurve2SurfDirectional(CamberLine,BottomTri,tY,-tX,tX*0)

        DistancesDeltas  = Distances2Top-Distances2Bottom
        return DistancesDeltas[1:-1]

    def _modifyCamberLine__(DeltaYvector):
        CamberY[1:-1] = CamberInitY[1:-1] + DeltaYvector
        DistancesDeltas = _updateDistancesDeltas__()
        return DistancesDeltas

    TangentCamber = D.getTangent(CamberLine)
    tX, tY, tZ = J.getxyz(TangentCamber)
    Distances2Top    = distancesCurve2SurfDirectional(CamberLine,TopTri,tY,-tX,tX*0)

    DeltaYvector0 = Distances2Top[1:-1]*0.1

    sol = scipy.optimize.root(_modifyCamberLine__,DeltaYvector0,method=method,options=options)
    # print (sol)
    if not sol.success:
        print ("Warning: CamberLine search did not converge. Check if output is satisfactory.")


    # Re-discretize CamberLine
    if CamberDistribution is None:
        CamberDistribution = dict(N=51,
                            kind='tanhOneSide',
                            FirstCellHeight=0.0001*Chord,
                            BreakPoint=1.)
        CamberDistribution = dict(N=200,
                            kind='trigonometric',
                            parameter=2,
                            BreakPoint=1.)
    elif 'BreakPoint' not in CamberDistribution:
        CamberDistribution['BreakPoint'] = 1.


    CamberLine = polyDiscretize(CamberLine,[CamberDistribution])

    # Store distance
    TangentCamber = D.getTangent(CamberLine)
    tX, tY, tZ = J.getxyz(TangentCamber)
    e, = J.invokeFields(CamberLine,['RelativeThickness'])
    Distances2Top    = distancesCurve2SurfDirectional(CamberLine,TopTri,tY,-tX,tX*0)
    Distances2Bottom = distancesCurve2SurfDirectional(CamberLine,BottomTri,tY,-tX,tX*0)
    e[:] = Distances2Top+Distances2Bottom # Does not look like; but this is an average ;-)
    e[-1] = np.sqrt((TopX[-1]-BottomX[-1])**2 + (TopY[-1]-BottomY[-1])**2) / Chord


    return CamberLine

def buildAirfoilFromCamberLine(CamberLine, NormalDirection=None,
                               TopDistribution=None, BottomDistribution=None):
    '''
    This function constructs an airfoil from a **CamberLine** (this means, a
    structured curve containing field ``{RelativeThickness}``).

    Parameters
    ----------

        CamberLine : zone
            structured curve indexed from leading towards
            trailing edge and containing ``FlowSolution`` field named
            ``RelativeThickness``

        NormalDirection : array of 3 :py:class:`float`
            chord-normal direction, which points
            towards the "top" of the airfoil. If not provided, the **CamberLine** is
            supposed to be placed at canonical :math:`OXY` plane.

        TopDistribution : zone
            a :py:func:`Generator.map` compatible distribution

            .. hint:: as obtained from :py:func:`copyDistribution`

        BottomDistribution : zone
            a :py:func:`Generator.map` compatible distribution

            .. hint:: as obtained from :py:func:`copyDistribution`

    Returns
    -------

        Airfoil : zone
            the new airfoil curve
    '''

    CamberLine = I.copyTree(CamberLine)

    x,y,z = J.getxyz(CamberLine)
    LeadingEdge = np.array([x[0], y[0], z[0]])
    Chord = distance(LeadingEdge, (x[-1],y[-1],z[-1]) )
    ChordDirection = np.array([x[-1]-x[0], y[-1]-y[0], z[-1]-z[0]])
    ChordDirection /= np.sqrt(ChordDirection.dot(ChordDirection))

    if NormalDirection is not None:
        NormalDirection = np.array(NormalDirection)
        NormalDirection/= np.sqrt(NormalDirection.dot(NormalDirection))

        BinormalDirection  = np.cross(ChordDirection,NormalDirection)
        NormalDirection    = np.cross(BinormalDirection, ChordDirection)
        NormalDirection   /= np.sqrt(NormalDirection.dot(NormalDirection))
        BinormalDirection /= np.sqrt(BinormalDirection.dot(BinormalDirection))

        FrenetOriginal = (tuple(ChordDirection),
                          tuple(NormalDirection),
                          tuple(BinormalDirection))
        FrenetAuxiliary = ((1,0,0),
                           (0,1,0),
                           (0,0,1))

        T._translate(CamberLine,-LeadingEdge)
        T._rotate(CamberLine, (0,0,0),FrenetOriginal, FrenetAuxiliary)


    CamberX, CamberY = J.getxy(CamberLine)

    NPts = len(CamberX)

    TangentCamber =  D.getTangent(CamberLine)
    tX, tY = J.getxy(TangentCamber)
    gamma = np.arctan2(tY,tX)
    Top    = D.line((0,0,0),(0,0,0),NPts);    Top[0]='Top'
    Bottom = D.line((0,0,0),(0,0,0),NPts); Bottom[0]='Bottom'

    TopX, TopY       = J.getxy(Top)
    BottomX, BottomY = J.getxy(Bottom)

    e, = J.getVars(CamberLine,['RelativeThickness'])
    if e is None:
        raise AttributeError('buildAirfoilFromCamberLine(): Input shall be a CamberLine with "RelativeThickness" FlowSolution.')

    for i in range(NPts):
        TopX[i] = CamberX[i]-0.5*e[i]*np.sin(gamma[i])*Chord
        TopY[i] = CamberY[i]+0.5*e[i]*np.cos(gamma[i])*Chord

        BottomX[i] = CamberX[i]+0.5*e[i]*np.sin(gamma[i])*Chord
        BottomY[i] = CamberY[i]-0.5*e[i]*np.cos(gamma[i])*Chord

    T._reorder(Bottom,(-1,2,3))


    if TopDistribution is not None:
        Top = G.map(Top,TopDistribution)
    if BottomDistribution is not None:
        Bottom = G.map(Bottom,BottomDistribution)

    try:
        Airfoil = T.join(Bottom,Top)
    except:
        # Airfoil = concatenate([Bottom,Top])
        print ('buildAirfoilFromCamberLine: Could not Join Top and Bottom parts of airfoil.')
        return None

    if NormalDirection is not None:
        T._rotate(Airfoil, (0,0,0), FrenetAuxiliary, FrenetOriginal)
        T._translate(Airfoil, LeadingEdge)

    return Airfoil


def setAirfoilCamberAndThickness(Airfoil, Camber=None, MaxCamberLocation=None,
        Thickness=None, MaxThicknessLocation=None):
    '''
    .. danger:: THIS IS FUNCTION IS BEING DEPRECATED. IT MUST BE REPLACED
        BY :py:func:`modifyAirfoil`
    '''
    iLE, Top, Bottom = findLeadingEdgeAndSplit(Airfoil,0.5)
    T._reorder(Top,(-1,2,3))
    T._reorder(Bottom,(-1,2,3))
    TopOriginalDistribution    = D.getDistribution(Top)
    BottomOriginalDistribution = D.getDistribution(Bottom)

    CamberLine = getCamberOptim(Airfoil)
    CamberX, CamberY = J.getxy(CamberLine)
    CamberChord = CamberX.max()-CamberX.min()

    Reference = cdeep((CamberX[-1],CamberY[-1],0))

    if Camber is not None:
        # Put CamberLine origin at (0,0,0)
        T._translate(CamberLine,(-CamberX[0],-CamberY[0],0))
        iMaxCamb = np.argmax(CamberY)
        OldCamber= CamberY[iMaxCamb]-CamberY[0]
        if OldCamber > 0.001*CamberChord:
            CamberY *= Camber/OldCamber
        else:
            # Non-cambered airfoil -> Set custom camber !
            if not MaxCamberLocation: MaxCamberLocation = 0.5
            NewCamberLine = D.bezier(D.polyline([
                (0,0,0),
                (MaxCamberLocation,Camber,0), # Not exact...
                (CamberX[-1],CamberY[-1],0)]),N=300)
            NewCamberLine = discretize(NewCamberLine,len(CamberX),
                {'kind':'trigonometric','parameter':2})

            # NewCamberLine = G.map(D.polyline([
            #     (0,0,0),
            #     (MaxCamberLocation,Camber,0), # Not exact...
            #     (CamberX[-1],CamberY[-1],0)]),D.getDistribution(linelaw(N=len(CamberX),
            #         Distribution={'kind':'trigonometric','parameter':2})))
            nCamberX, nCamberY = J.getxy(NewCamberLine)
            iMaxCamb  = np.argmax(nCamberY)
            newOldCamber = nCamberY[iMaxCamb]-nCamberY[0]
            nCamberY *= Camber/newOldCamber



            e, = J.getVars(CamberLine,['RelativeThickness'])
            ne = np.interp(nCamberX,CamberX,e)

            CamberX[:] = nCamberX
            CamberY[:] = nCamberY
            e[:] = ne
        CamberX, CamberY = J.getxy(CamberLine)

        # Put back CamberLine at its original position
        T._translate(CamberLine,(Reference[0]-CamberX[-1],Reference[1]-CamberY[-1],0))



    if MaxCamberLocation is not None and Camber != 0:
        # Put CamberLine origin at (0,0,0)
        T._translate(CamberLine,(-CamberX[0],-CamberY[0],0))


        iMaxCamb = np.argmax(CamberY)

        FrontCamber = T.subzone(CamberLine,(1,1,1),(iMaxCamb+1,1,1))
        RearCamber = T.subzone(CamberLine,(iMaxCamb+1,1,1),(-1,-1,-1))
        FrontCamberX,FrontCamberY = J.getxy(FrontCamber)
        RearCamberX,RearCamberY = J.getxy(RearCamber)

        InitialLength = RearCamberX[-1]-RearCamberX[0]
        FrontCamberX *= MaxCamberLocation*CamberChord/FrontCamberX[-1]
        FinalLength = CamberX[-1]-FrontCamberX[-1]

        RearCamberX -= RearCamberX[0] # Position at origin
        ScaleFactor = FinalLength/InitialLength
        RearCamberX *= ScaleFactor
        RearCamberX += FrontCamberX[-1]

        nCamberLine  = T.join(FrontCamber,RearCamber)
        # Check if reordering is necessary:
        nCamberX, nCamberY = J.getxy(nCamberLine)
        if nCamberX[1]<nCamberX[0]: T._reorder(nCamberLine,(-1,2,3))

        # Copy the interpolated thickness distribution
        e, = J.getVars(CamberLine,['RelativeThickness'])
        ne = np.interp(nCamberX,CamberX,e)

        CamberX[:] = nCamberX
        CamberY[:] = nCamberY
        e[:] = ne

        # Put back CamberLine at its original position
        T._translate(CamberLine,(Reference[0]-CamberX[-1],Reference[1]-CamberY[-1],0))



    if Thickness is not None:
        e, = J.getVars(CamberLine,['RelativeThickness'])
        e *= Thickness / e.max()

    if MaxThicknessLocation is not None:
        # Put CamberLine origin at (0,0,0)
        T._translate(CamberLine,(-CamberX[0],-CamberY[0],0))

        e, = J.getVars(CamberLine,['RelativeThickness'])

        # Important Note: Do exactly as MaxCamber, but change
        # the index (now based upon RelativeThickness).
        # I keep same variables names, I'm a bit lazy 8-)
        iMaxCamb = np.argmax(e)

        FrontCamber = T.subzone(CamberLine,(1,1,1),(iMaxCamb+1,1,1))
        RearCamber = T.subzone(CamberLine,(iMaxCamb+1,1,1),(-1,-1,-1))
        FrontCamberX,FrontCamberY = J.getxy(FrontCamber)
        FrontCamberY[:]= e[:iMaxCamb+1]
        RearCamberX,RearCamberY = J.getxy(RearCamber)
        RearCamberY[:]= e[iMaxCamb:]

        InitialLength = RearCamberX[-1]-RearCamberX[0]
        FrontCamberX *= MaxThicknessLocation*CamberChord/FrontCamberX[-1]
        FinalLength = CamberX[-1]-FrontCamberX[-1]

        RearCamberX -= RearCamberX[0] # Position at origin
        ScaleFactor = FinalLength/InitialLength
        RearCamberX *= ScaleFactor
        RearCamberX += FrontCamberX[-1]

        nCamberLine  = T.join(FrontCamber,RearCamber)
        # Check if reordering is necessary:
        nCamberX, nCamberY = J.getxy(nCamberLine)
        if nCamberX[1]<nCamberX[0]: T._reorder(nCamberLine,(-1,2,3))

        # Copy the interpolated thickness distribution
        ne = np.interp(CamberX,nCamberX,e)
        e[:] = ne


        # Put back CamberLine at its original position
        T._translate(CamberLine,(Reference[0]-CamberX[-1],Reference[1]-CamberY[-1],0))



    CamberX, CamberY = J.getxy(CamberLine)

    ModifiedAirfoil = buildAirfoilFromCamberLine(CamberLine,
        TopOriginalDistribution, BottomOriginalDistribution)

    return ModifiedAirfoil

def is2DCurveClockwiseOriented(curve):
    '''
    .. warning:: this function requires further validation

    returns :py:obj:`True` if provided **curve** supported on :math:`OXY` plane
    is oriented clockwise
    '''
    # TODO try to uniformly discretize curve before evaluating orientation
    cx, cy = J.getxy(curve)
    thesum = 0.
    for i in range(len(cx)-1):
        thesum += (cx[i+1]-cx[i])*(cy[i+1]+cy[i])

    isClockwise = True if thesum > 0 else False

    return isClockwise


def isAirfoilClockwiseOriented(curve):
    '''
    .. warning:: this function requires further validation

    returns :py:obj:`True` if provided **curve** supported on :math:`OXY` plane
    is oriented clockwise
    '''
    # TODO try to uniformly discretize curve before evaluating orientation
    raise ValueError('isAirfoilClockwiseOriented must be replaced with is2DCurveClockwiseOriented')
    cx, cy = J.getxy(curve)
    p2 = np.array([cx[2],cy[2],0])
    p1 = np.array([cx[1],cy[1],0])
    p0 = np.array([cx[0],cy[0],0])
    v12 = p2-p1
    v12 /= np.sqrt(v12.dot(v12))
    v01 = p1-p0
    v01 /= np.sqrt(v01.dot(v01))
    v = np.cross(v12,v01)

    isClockwise = True if v[2] > 0 else False

    return isClockwise


def putAirfoilClockwiseOrientedAndStartingFromTrailingEdge(foil):
    '''
    This function transforms the input airfoil into clockwise-oriented and
    starting from trailing edge.

    Parameters
    ----------

        foil : zone
            structured curve of the airfoil

            .. note:: **foil** is modified

    '''
    OriginalAirfoil = I.copyTree(foil)

    if not is2DCurveClockwiseOriented(foil): T._reorder(foil,(-1,2,3))

    x,y,z = J.getxyz(foil)

    XChord = x.max()-x.min()
    foilAux = C.initVars(foil,'ChordwiseIndicator=({CoordinateX}-%g)/%g'%(x.min(),XChord))
    SelectedRegion = P.selectCells(foilAux,'{ChordwiseIndicator}>0.9')
    SelectedRegion = C.convertBAR2Struct( SelectedRegion )
    SmoothParts = T.splitCurvatureAngle(SelectedRegion, 45.0 )
    YMax = [C.getMaxValue(sp,'CoordinateY') for sp in SmoothParts]
    YMax, SmoothParts = J.sortListsUsingSortOrderOfFirstList(YMax, SmoothParts)
    LowerPart = SmoothParts[0]
    xLP,yLP,zLP = J.getxyz(LowerPart)
    iMaxYLP = np.argmax(yLP)
    Pt = (xLP[iMaxYLP],yLP[iMaxYLP],zLP[iMaxYLP])
    iTE,_ = D.getNearestPointIndex(foil,Pt)
    NPts = len(x)
    RollPts = NPts - iTE
    if RollPts > 0 and RollPts != NPts:
        fields2roll =  J.getxyz(foil)
        for field in fields2roll:
            field[:] = np.hstack((field[iTE:],field[1:iTE],field[iTE]))

    gets(foil)

    J.migrateFields(OriginalAirfoil, foil, keepMigrationDataForReuse=False)


def setTrailingEdge(Airfoil):
    '''
    .. warning:: THIS FUNCTION IS BEING DEPRECATED
    '''
    _, Top, Bottom = findTrailingEdgeAndSplit(Airfoil)

    foil = concatenate([Bottom, Top])

    return foil


def getCurveNormalMap(curve):
    '''
    Equivalent of :py:func:`Generator.PyTree._getNormalMap`, but for use with
    curves.
    The normal vector (``{sx}``, ``{sy}``, ``{sz}``) is located on both
    nodes and centers. The computed normal vector is constructed
    from the unitary tangent vector of the curve and the mean
    binormal vector (mean oscullatory plane).

    Parameters
    ----------

        curve : zone
            PyTree curve. Possibly a BAR.

            .. note:: **curve** is modified


    Returns
    -------

        MeanNormal : float

        MeanTangent : float

        MeanBinormal : float
    '''

    # Step 1: Compute tangent vector (possibly BAR)
    cx, cy, cz = J.getxyz(curve)
    NPts = len(cx)
    cxyz = np.vstack((cx, cy, cz)).T

    fT = np.zeros((NPts,3),order='F')

    # Central difference tangent computation
    fT[1:-1,:] = 0.5*(np.diff(cxyz[:-1,:],axis=0)+np.diff(cxyz[1:,:],axis=0))

    TypeZone,_,_,_,_ = I.getZoneDim(curve)

    if TypeZone == 'Unstructured':
        # The curve is BAR type
        GridElts = I.getNodeFromName1(curve,'GridElements')
        EltConn  = I.getNodeFromName1(GridElts,'ElementConnectivity')[1]
        if EltConn[0] == EltConn[-1]:
            # BAR is closed
            fT[0,:] = 0.5*((cxyz[1,:]-cxyz[0,:])+(cxyz[-1,:]-cxyz[-2,:]))
            fT[-1,:] = (cxyz[-1,:]-cxyz[-2,:])
        else:
            # BAR is open. That's always good news ;-)
            fT[0,:] = (cxyz[1,:]-cxyz[0,:])
            fT[-1,:] = (cxyz[-1,:]-cxyz[-2,:])
    else:
        # Curve is Structured, and necessarily open
        fT[0,:] = (cxyz[1,:]-cxyz[0,:])
        fT[-1,:] = (cxyz[-1,:]-cxyz[-2,:])

    Norm = np.sqrt(np.sum(fT*fT, axis=1)).reshape((NPts,1),order='F')
    fT /= Norm

    # Step 2: Compute mean binormal vector
    binormal = np.mean(np.cross(fT[1:],fT[:-1]),axis=0)
    binormal /= (binormal[0]**2+binormal[1]**2+binormal[2]**2)**0.5

    # Step 3: Compute normal
    normal = np.cross(binormal,fT)

    # include as new fields, both in nodes and centers
    sx, sy, sz = J.invokeFields(curve,['sx','sy','sz'])
    sx[:], sy[:], sz[:] = normal[:,0], normal[:,1], normal[:,2]

    C._normalize(curve,['sx','sy','sz'])
    C.node2Center__(curve,'nodes:sx')
    C.node2Center__(curve,'nodes:sy')
    C.node2Center__(curve,'nodes:sz')
    C._normalize(curve,['centers:sx','centers:sy','centers:sz'])

    # Returns the average tangent, normal and binormal vectors
    MeanNormal = np.mean(normal,axis=0)
    MeanTangent = np.mean(fT,axis=0)
    MeanBinormal = binormal
    return MeanNormal, MeanTangent, MeanBinormal

def distances(zone1, zone2):
    '''
    Compute the average, minimum, and maximum distances between two zones.

    .. warning:: this function shall be optimized

    Parameters
    ----------

        zone1 : zone
            first zone to evaluate

        zone2 : zone
            second zone to evaluate

    Returns
    -------

        AverageDistance : float
            average distance between zones [m]

        MinimumDistance : float
            minimum distance between zones [m]

        MaximumDistance : float
            maximum distance between zones [m]
    '''
    x,y,z = J.getxyz(zone1)
    x = x.ravel(order='K')
    y = y.ravel(order='K')
    z = z.ravel(order='K')


    Nzone1 = len(x)
    PtsZone1 = [(x[i],y[i],z[i]) for i in range(Nzone1)]

    # TODO replace getNearestPointIndex
    try:
        Res = D.getNearestPointIndex(zone2,PtsZone1)
    except SystemError as e:
        print(PtsZone1)
        raise SystemError("FOUND SYSTEM ERROR")
    else:
        Res = D.getNearestPointIndex(zone2,PtsZone1)

    Distances = np.array([r[1] for r in Res])
    MinimumDistance = np.sqrt(np.min(Distances))
    AverageDistance = np.sqrt(np.mean(Distances))
    MaximumDistance = np.sqrt(np.max(Distances))

    return AverageDistance, MinimumDistance, MaximumDistance


def reOrientateAndOpenAirfoil(zoneFoil,maxTrailingEdgeThickness=0.01):
    """
    This function was created as an auxiliar operation after
    using :py:func:`MOLA.GenerativeShapeDesign.scanWing`.
    This function properly orientates
    airfoil (upper side: suction side; lower side: pressure
    side) and oriented such that freestream flows from the
    left to the right. It also opens the airfoil at a given
    coordinate. This function does NOT re-scale or re-twist
    the airfoil

    Parameters
    ----------

        zoneFoil : zone
            defininng 1D-structured curve situated at :math:`OXY` plane

        maxTrailingEdgeThickness : float
            distance *(absolute!)* used as reference for opening the trailing edge

    Returns
    -------

        NewFoil : zone
            modified airfoil
    """

    FoilName = zoneFoil[0]
    I._rmNodesByName(zoneFoil,'FlowSolution')

    # Temporarily split upper and lower sides:
    # Find Trailing Edge
    isClockwise = is2DCurveClockwiseOriented(zoneFoil)
    if not isClockwise: T._reorder(zoneFoil,(-1,2,3))

    AirfoilX, AirfoilY = J.getxy(zoneFoil)
    iLE = np.argmin(AirfoilX)

    # Split sides
    Side1 = T.subzone(zoneFoil,(iLE+1,1,1),(-1,-1,-1))
    Side2 = T.subzone(zoneFoil,(1,1,1),(iLE+1,1,1))
    Side1[0] = 'Side1'
    Side2[0] = 'Side2'
    S1x, S1y = J.getxy(Side1)
    S2x, S2y = J.getxy(Side2)
    if S1x[2]-S1x[0]<0: T._reorder(Side1,(-1,2,3))
    if S2x[2]-S2x[0]<0: T._reorder(Side2,(-1,2,3))

    # Determine if rotation around +X is required
    val1 = np.trapz(S1x,S1y)
    val2 = np.trapz(S2x,S2y)
    if val1>val2:
        T._rotate(zoneFoil,(0,0,0),(1,0,0),180.)

    # Determine if rotation around +Y is required
    CamberLine = getCamberOptim(zoneFoil)
    CLx, CLy = J.getxy(CamberLine)
    RelThicknes, = J.getVars(CamberLine,['RelativeThickness'])
    ThicknessIndex = np.argmax(RelThicknes)
    Thickness = RelThicknes[ThicknessIndex]
    MaxThicknessLocation = CLx[ThicknessIndex]
    if MaxThicknessLocation>=0.5:
        T._rotate(zoneFoil,(0,0,0),(0,1,0),180.)

    # Open the airfoil
    # split upper and lower sides:
    # Find Trailing Edge
    isClockwise = is2DCurveClockwiseOriented(zoneFoil)
    if not isClockwise: T._reorder(zoneFoil,(-1,2,3))

    AirfoilX, AirfoilY, AirfoilZ = J.getxyz(zoneFoil)
    iLE = np.argmin(AirfoilX)
    iTE = np.argmax(AirfoilX)

    # Put airfoil at (0,0)
    T._translate(zoneFoil,(-AirfoilX[iLE],-AirfoilY[iLE],-AirfoilZ[iLE]))


    Xmax = AirfoilX[iTE]
    Xmin = AirfoilX[iLE]
    Step = 1e-4
    CurrentPos = Xmax
    while CurrentPos > Xmin:
        CurrentPos -= Step
        Slice = P.isoLine(zoneFoil,'CoordinateX',CurrentPos)
        if C.getNPts(Slice) != 2:
            pass
        else:
            SliceX, SliceY = J.getxy(Slice)
            distance = ( (SliceX[1]-SliceX[0])**2 +
                         (SliceY[1]-SliceY[0])**2 ) **0.5
            if distance >= maxTrailingEdgeThickness:
                break



    # Split Top and Bottom sides
    Side1 = T.subzone(zoneFoil,(iTE+1,1,1),(-1,-1,-1))
    Side2 = T.subzone(zoneFoil,(1,1,1),(iTE+1,1,1))
    Side1[0] = 'Side1'
    Side2[0] = 'Side2'
    S1x, S1y = J.getxy(Side1)
    S2x, S2y = J.getxy(Side2)
    if S1x[2]-S1x[0]<0: T._reorder(Side1,(-1,2,3))
    if S2x[2]-S2x[0]<0: T._reorder(Side2,(-1,2,3))
    S1x, S1y = J.getxy(Side1)
    S2x, S2y = J.getxy(Side2)

    iCutSide1 = np.where(S1x>SliceX[1])[0][0]
    iCutSide2 = np.where(S2x>SliceX[0])[0][0]

    Side1 = T.subzone(Side1,(1,1,1),(iCutSide1+1,1,1))
    Side2 = T.subzone(Side2,(1,1,1),(iCutSide2+1,1,1))

    S1x, S1y = J.getxy(Side1)
    S2x, S2y = J.getxy(Side2)

    S1x[-1] = SliceX[1]
    S1y[-1] = SliceY[1]

    S2x[-1] = SliceX[0]
    S2y[-1] = SliceY[0]

    NewFoil = T.join(Side2,Side1)

    NFx, NFy = J.getxy(NewFoil)
    if NFy[0] > NFy[-1]: T._reorder(NewFoil,(-1,2,3))


    NewFoil[0] = 'Or_%s'%FoilName

    return NewFoil


def getFirstAndLastCellLengths(curve):
    '''
    From a 1D structured curve, get the First and Last cell (segment) lengths

    Parameters
    ----------

        curve : zone
            zone 1D Structured, input curve

    Returns
    -------

        FirstCellLength : float
            distance between first and second point of curve

        LastCellLength : float
            distance between last and before-last points
    '''
    x,y,z = J.getxyz(curve)
    FirstCellLength = ((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2)**0.5
    LastCellLength  = ((x[-1]-x[-2])**2+(y[-1]-y[-2])**2+(z[-1]-z[-2])**2)**0.5
    return FirstCellLength, LastCellLength


def cutTrailingEdgeOfAirfoil(foil, TEthickness, DesiredXChord=None,Xmin=0.8):
    '''
    Cut the trailing edge of an airfoil in order to verify a given
    trailing edge thickness.
    If **DesiredXChord** is not :py:obj:`None`, then appropriate scaling of airfoil
    is performed so that final cut airfoil yields the desired chord;
    otherwise, the new cut airfoil will systematically have a smaller chord.

    Parameters
    ----------

        foil : zone
            1D Structured zone, Airfoil placed at :math:`OXY` plane, with starting
            index at Trailing edge and oriented clockwise.

        TEThickness : float
            Absolute thickness of the new cut of trailing edge.

        DesiredXChord : float
            Resulting new xchord (must be lower than
            original provided xchord of **foil**)
            If not provided (:py:obj:`None`), new foil will be a simple cut of foil.

        Xmin : float
            Minimum allowable cut station. Cut searches are performed
            for stations bigger than Xmin.

    Returns
    -------

        FoilCut : zone

    '''

    xf = J.getx(foil)
    Xmax  = xf.max()
    Chord = Xmax - xf.min()
    Guess1factor = 0.05
    Guess2factor = 0.10

    def getThicknessAndOrXChordDeltaAtStation__(Xstation,curve,TypeOfReturn=0):
        n  = np.array([1.0,0.0,0.0])
        Pt = np.array([Xstation,0.0,0.0])

        # plane equation used for slice (c1,c2,c3,c4)
        PlaneCoefs = n[0],n[1],n[2],-n.dot(Pt)
        C._initVars(curve,'SliceVar=%0.12g*{CoordinateX}+%0.12g*{CoordinateY}+%0.12g*{CoordinateZ}+%0.12g'%PlaneCoefs)
        Slice = P.isoSurfMC(curve,'SliceVar',value=0.0)
        xc = J.getx(curve)

        if len(Slice) == 0:
            Thickness = 0.
            XChord = xc.max() - xc.min()
        else:
            Slice = Slice[0]
            x,y,z = J.getxyz(Slice)

            if len(x) != 2:
                Thickness = 0.
                XChord = x[0] - xc.min()
            else:
                Thickness = ((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2)**0.5
                XChord = x[0] - xc.min()

        if   TypeOfReturn == 0: return Thickness - TEthickness
        elif TypeOfReturn == 1: return XChord - DesiredXChord
        elif TypeOfReturn == 2: return Thickness-TEthickness, XChord-DesiredXChord
        else: raise AttributeError('TypeOfReturn %s not recognized'%TypeOfReturn)

    def getDeltaXChordForGivenCut__(Scale):
        curve = T.homothety(foil,(xf.min(),0,0),Scale)
        xc = J.getx(curve)
        NewChord = xc.max() - xc.min()
        NewXmin = xc.min() + Xmin*NewChord
        Xmax = C.getMaxValue(curve,'CoordinateX')
        Xguess1 = Guess1factor*NewXmin + (1-Guess1factor)*Xmax
        Xguess2 = Guess2factor*NewXmin + (1-Guess2factor)*Xmax
        sol = J.secant(getThicknessAndOrXChordDeltaAtStation__, x0=Xguess1, x1=Xguess2, ftol=1e-12, bounds=[NewXmin, Xmax-0.01*(Xmax-NewXmin)], maxiter=20, args=(curve,0))
        if not sol['converged']:
            print('NOT CONVERGED AT getDeltaXChordForGivenCut__')
            print(sol);sys.exit()
        StationCut = sol['root'][0]
        return getThicknessAndOrXChordDeltaAtStation__(StationCut, curve, TypeOfReturn=1)


    if DesiredXChord is None:
        NewXmin = xf.min() + Xmin*Chord
        Xmax = xf.max()
        Xguess1 = Guess1factor*NewXmin + (1-Guess1factor)*Xmax
        Xguess2 = Guess2factor*NewXmin + (1-Guess2factor)*Xmax

        sol = J.secant(getThicknessAndOrXChordDeltaAtStation__, x0=Xguess1, x1=Xguess2, ftol=1e-8, bounds=[NewXmin, Xmax-0.01*(Xmax-NewXmin)], maxiter=20, args=(foil,0))
        StationCut = sol['root'][0]

    else:
        GuessScale = DesiredXChord/Chord
        sol = J.secant(getDeltaXChordForGivenCut__, x0=GuessScale, x1=GuessScale+0.1, ftol=1e-8, bounds=[0.001, 99.], maxiter=20, args=())
        Scale = sol['root'][0]

        curve = T.homothety(foil,(xf.min(),0,0),Scale)
        xc = J.getx(curve)
        NewChord = xc.max() - xc.min()
        NewXmin = xc.min() + Xmin*NewChord
        Xmax = C.getMaxValue(curve,'CoordinateX')
        Xguess1 = Guess1factor*NewXmin + (1-Guess1factor)*Xmax
        Xguess2 = Guess2factor*NewXmin + (1-Guess2factor)*Xmax
        sol = J.secant(getThicknessAndOrXChordDeltaAtStation__, x0=Xguess1, x1=Xguess2, ftol=1e-12, bounds=[NewXmin, Xmax-0.01*(Xmax-NewXmin)], maxiter=20, args=(curve,0))
        if not sol['converged']:
            print('NOT CONVERGED AT getDeltaXChordForGivenCut__')
            print(sol);sys.exit()
        StationCut = sol['root'][0]
        foil = curve

    # Cut airfoil at StationCut
    n  = np.array([1.0,0.0,0.0])
    Pt = np.array([StationCut,0.0,0.0])

    # Slice
    PlaneCoefs = n[0],n[1],n[2],-n.dot(Pt)
    C._initVars(foil,'SliceVar=%0.12g*{CoordinateX}+%0.12g*{CoordinateY}+%0.12g*{CoordinateZ}+%0.12g'%PlaneCoefs)
    Slice = P.isoSurfMC(foil,'SliceVar',value=0.0)[0]
    x,y,z = J.getxyz(Slice)

    # Build line used to cut airfoil further with splitCurve
    CutLine = D.line((x[0],y[0],z[0]),(x[1],y[1],z[1]),3)
    Thickness = D.getLength(CutLine)
    CutLine = extrapolate(CutLine, 10*Thickness)
    CutLine = extrapolate(CutLine, 10*Thickness,opposedExtremum=True)

    Cuts = splitCurves(foil, CutLine, select=1)

    Xmins = np.array([C.getMinValue(c,'CoordinateX') for c in Cuts])
    FoilCut = Cuts[np.argmin(Xmins)]

    return FoilCut


def setImposedFieldLSTT(sidecurve, fieldname='intermittency_imposed',
                        Xref=0.2, LengthScale=0.01, a_boost=2.15,
                        sa=0.182, sb=0.414):
    '''
    Create new field in provided zone **sidecurve** following `LSTT19 <https://arc.aiaa.org/doi/10.2514/1.J057734>`_
    convention consisting in a cubic tangent polynomial.

    Parameters
    ----------

        sidecurve : zone
            Must correspond to the top or the bottom side of the airfoil and
            must be oriented (i-increasing) in the main streamwise direction.

            .. note:: **sidecurve** is modified

        fieldname : str
            The new field created in **sidecurve**. Can be, for
            example: ``'intermittency_imposed'`` or ``'clim_imposed'``

        Xref : float
            X-coordinate (absolute) corresponding to the start of
            the LSTT region. It is usually the transition onset location.

        LengthScale : float
            Absolute length scale that determines the LSTT
            polynomial scaling. It is usually the laminar bubble length or a
            factor of the reference integral boundary-layer thickness.

        a_boost : float
            Maximum value of the boosting of field **fieldname**.

        sa : float
            scale that determines the boost region (increase), such that
            ``sa * LengthScale`` correspond to the abscissa portion of the
            increase of the variable.

        sb : float
            scale that determines the relax region (decrease), such that
            ``sb * LengthScale`` correspond to the abscissa portion of the
            decrease of the variable.

    '''

    field, = J.invokeFields(sidecurve, [fieldname])
    s    = gets(sidecurve)
    x    = J.getx(sidecurve)
    s   *= D.getLength(sidecurve)

    sXref = np.interp(Xref, x, s)
    shat = (s - sXref) / LengthScale
    ZerosRegion   = shat < 0.
    BoostRegion   = ( shat >= 0  ) * ( shat < sa )
    RelaxRegion   = ( shat >= sa ) * ( shat <= sb )
    OnesRegion    = shat > sb

    field[ZerosRegion] = 0.

    field[BoostRegion] = a_boost*(-2.0*(shat[BoostRegion]/sa)**3
                                  +3.0*(shat[BoostRegion]/sa)**2)

    field[RelaxRegion] = (2.0*(a_boost-1.0)*((shat[RelaxRegion]-sa)/(sb-sa))**3
                        - 3.0*(a_boost-1.0)*((shat[RelaxRegion]-sa)/(sb-sa))**2
                        + a_boost)

    field[OnesRegion]  = 1.


def checkAirfoilAndGetTopo(AirfoilCurve):
    '''
    Make verifications on **AirfoilCurve**, which must be a PyTree 1D node
    manifold and single branch. Otherwise, raises an exception.

    Parameters
    ----------

        AirfoilCurve : zone
            zone to be checked

    Returns
    -------

        isStructured : bool
            :py:obj:`True` if the zone is structured
    '''
    if I.isStdNode(AirfoilCurve) != -1:
        raise AttributeError('Standard PyTree node is required')

    if I.getType(AirfoilCurve) != 'Zone_t':
        raise AttributeError('Argument must be a PyTree zone')

    ZoneTopo, ni_np, nj_ne, nk_eltsName, celldim = I.getZoneDim(AirfoilCurve)

    isStructured = ZoneTopo == 'Structured'

    if isStructured:
        nj, nk = nj_ne, nk_eltsName
        if nj > 1 or nk > 1:
            raise AttributeError('Argument must be a 1D curve')

    else:
        if nk_eltsName != 'BAR':
            raise AttributeError('Argument must be 1D curve')

        Manifold = T.splitManifold(AirfoilCurve)
        if len(Manifold) > 1:
            raise AttributeError('Argument must be manifold')

        Branches = T.splitTBranches(AirfoilCurve)
        if len(Branches) > 1:
            raise AttributeError('Argument must have a single branch')

    return isStructured


def getDecreasingDirections(OBB):
    '''
    Given a cartesian cell, as got from :py:func:`Generator.PyTree.OBB`, compute
    the main directions (absolute) in decrasing order, such that the first
    result is the longest direction of a OBB (*oriented bounding-box*).

    Parameters
    ----------

        OBB : zone
            oriented bounding-box as obtained from :py:func:`Generator.PyTree.OBB`

    Returns
    -------

        Directions : list
            3 lists of 1D numpy.array vectors, indicating the three main
            directions of the OBB, in decrasing order (longest axe first)

    '''
    x,y,z = J.getxyz(OBB)

    len1 = (  (x[1,0,0]-x[0,0,0])**2
            + (y[1,0,0]-y[0,0,0])**2
            + (z[1,0,0]-z[0,0,0])**2 ) ** 0.5

    len2 = (  (x[0,1,0]-x[0,0,0])**2
            + (y[0,1,0]-y[0,0,0])**2
            + (z[0,1,0]-z[0,0,0])**2 ) ** 0.5

    len3 = (  (x[0,0,1]-x[0,0,0])**2
            + (y[0,0,1]-y[0,0,0])**2
            + (z[0,0,1]-z[0,0,0])**2 ) ** 0.5

    Lengths = np.array([len1,len2,len3])
    ArgSort = np.argsort(Lengths)

    StartPoint = np.array([x[0,0,0],
                           y[0,0,0],
                           z[0,0,0],])

    Directions = []
    for i in ArgSort:
        if i == 0:
            EndPoint = np.array([x[1,0,0],
                                 y[1,0,0],
                                 z[1,0,0],])

        elif i == 1:
            EndPoint = np.array([x[0,1,0],
                                 y[0,1,0],
                                 z[0,1,0],])

        else:
            EndPoint = np.array([x[0,0,1],
                                 y[0,0,1],
                                 z[0,0,1],])

        Directions += [EndPoint-StartPoint]

    Directions = Directions[::-1]

    return Directions


def computeChordwiseAndThickwiseIndicators(AirfoilCurve):
    '''
    Given an airfoil **AirfoilCurve**, create new fields named:
    ``"ChordwiseIndicator"`` and ``"ThickwiseIndicator"``, both comprised
    between ``-1`` and ``+1``, where ``=-1`` and ``=+1`` indicates the extrema
    following the chordwise or thickwise directions, respectively.

    Parameters
    ----------

        AirfoilCurve : zone
            PyTree 1D curve, the input airfoil

            .. note:: **AirfoilCurve** is modified

    Returns
    -------

        ApproximateChord : float
            The approximated chord length [m]

        ApproximateThickness : float
            The approximated maximum airfoil thickness [m]
    '''

    AirfoilCurveForOBB = C.convertBAR2Struct(AirfoilCurve)
    AirfoilCurveForOBB = G.map(AirfoilCurveForOBB,
                         D.line((0,0,0),(1,0,0),C.getNPts(AirfoilCurveForOBB)))
    OBB = G.BB(AirfoilCurveForOBB, method='OBB')

    Barycenter = G.barycenter(OBB)

    DecreasingDirections = getDecreasingDirections(OBB)

    def _invokeIndicator(Direction, Name):

        ApproximateLength = ( Direction.dot(Direction) ) ** 0.5

        Direction /= ApproximateLength

        #       ( x - B ) dot u
        #         -   -       -
        Eqn = ('({x}-{Bx})*{ux} +'
               '({y}-{By})*{uy} +'
               '({z}-{Bz})*{uz}'
               ).format(
               x='{CoordinateX}',
               y='{CoordinateY}',
               z='{CoordinateZ}',
               Bx=Barycenter[0],
               By=Barycenter[1],
               Bz=Barycenter[2],
               ux=Direction[0],
               uy=Direction[1],
               uz=Direction[2],
               )
        C._initVars(AirfoilCurve,Name+'='+Eqn)
        Indicator, = J.getVars(AirfoilCurve, [Name])
        Indicator /= Indicator.max()

        return ApproximateLength

    ApproximateChord     = _invokeIndicator( DecreasingDirections[0],
                                             'ChordwiseIndicator'     )
    ApproximateThickness = _invokeIndicator( DecreasingDirections[1],
                                             'ThickwiseIndicator'     )

    return ApproximateChord, ApproximateThickness


def getApproximateChordAndThickness(AirfoilCurve):
    '''
    Given an airfoil **AirfoilCurve**, returns the approximative Chord and
    Thickness. It also creates Chordwise and Thickwise indicators if
    not already exist on curve, using
    :py:func:`computeChordwiseAndThickwiseIndicators` .

    Parameters
    ----------

        AirfoilCurve : zone
            PyTree 1D curve, the input airfoil

            .. note:: **AirfoilCurve** is modified

    Returns
    -------

        ApproximateChord : :py:class:`float`
            The approximated chord length [m]

        ApproximateThickness : :py:class:`float`
            The approximated maximum airfoil thickness [m]

    '''
    computedChordwiseIndicator = C.isNamePresent(AirfoilCurve,
                                'ChordwiseIndicator') >= 0

    computedThickwiseIndicator = C.isNamePresent(AirfoilCurve,
                                'ThickwiseIndicator') >= 0

    if not computedChordwiseIndicator or not computedThickwiseIndicator:
        Chord, Thickness = computeChordwiseAndThickwiseIndicators(AirfoilCurve)

    else:
        x, y, z = J.getxyz(AirfoilCurve)

        VarsDict = J.getVars2Dict( AirfoilCurve,  ['ChordwiseIndicator',
                                                   'ThickwiseIndicator'] )
        def _getLength(IndicatorName):

            Indicator = VarsDict[IndicatorName]

            MinArg = np.argmin( Indicator )
            MaxArg = np.argmax( Indicator )

            MinPoint = np.array( [ x[MinArg],
                                   y[MinArg],
                                   z[MinArg] ])

            MaxPoint = np.array( [ x[MaxArg],
                                   y[MaxArg],
                                   z[MaxArg] ])

            Min2MaxPointsVector = MaxPoint - MinPoint

            Length = ( Min2MaxPointsVector.dot(Min2MaxPointsVector) ) ** 0.5

            return Length

        Chord     = _getLength('ChordwiseIndicator')
        Thickness = _getLength('ThickwiseIndicator')

    return Chord, Thickness

def findLeadingOrTrailingEdge(AirfoilCurve, ChordwiseRegion='> +0.5',
        ToleranceRelativeRadius=1e-02):
    '''
    Given a curve of an airfoil **AirfoilCurve**, compute the characteristic
    edge (Leading or Trailing) based on curvature radius. The search
    region is defined by **ChordwiseRegion** argument.

    Parameters
    ----------

        AirfoilCurve : zone
            PyTree 1D, curve of the airfoil

        ChordwiseRegion : str
            comparison criterion for establishing the
            filtered region where search is performed

        ToleranceRelativeRadius : float
            small number determining the search
            region as

            ::

                [ rmin + ToleranceRelativeRadius * ( Chord - rmin ) ]


    Returns
    -------

        Point : zone
            PyTree point, characteristic point found

        rmin : :py:class:`float`
            minimum radius of candidates search procedure
    '''

    Chord, Thickness = getApproximateChordAndThickness(AirfoilCurve)

    SelectedRegion = P.selectCells(AirfoilCurve,'{ChordwiseIndicator}'+
                                                  ChordwiseRegion)

    SelectedRegion = C.convertBAR2Struct( SelectedRegion )

    # TODO: Investigate this in detail:
    # SelectedRegion = G.map(SelectedRegion,
    #                        D.line((0,0,0),(1,0,0),C.getNPts(SelectedRegion)))

    D._getCurvatureRadius( SelectedRegion )

    x, y, z =  J.getxyz( SelectedRegion )
    gets( SelectedRegion )
    radius, = J.getVars( SelectedRegion, ['radius'] )

    rmin = radius.min()

    CandidateMaxRadius = rmin + ToleranceRelativeRadius * ( Chord - rmin )

    CandidateRegion = P.selectCells( SelectedRegion,
                                    '{radius} < %g'%CandidateMaxRadius )


    CandidateCurves = T.splitManifold(CandidateRegion)

    AbscissaCandidates = []
    for cc in CandidateCurves:

        s,     = J.getVars( cc, ['s'] )

        if len(s) == 3:
            AbscissaCandidates.append( s[1] )

        else:
            AbscissaCandidates.append( 0.5 * ( s[0] + s[-1] ) )


    AbscissaEdge = np.mean( AbscissaCandidates )

    LeadingOrTrailingEdge, = P.isoSurfMC( SelectedRegion, 's',
                                         value=AbscissaEdge )

    NPtsResult = C.getNPts( LeadingOrTrailingEdge )

    if NPtsResult  == 1:
        return LeadingOrTrailingEdge, rmin

    elif NPtsResult == 2:
        Pt1 = T.subzone( LeadingOrTrailingEdge, [0], type='elements' )
        Pt2 = T.subzone( LeadingOrTrailingEdge, [1], type='elements' )

        Pt1coord = np.array(J.getxyz(Pt1)).flatten()

        Pt2coord = np.array(J.getxyz(Pt2)).flatten()

        Pt1toPt2 = Pt2coord - Pt1coord

        Distance = ( Pt1toPt2.dot(Pt1toPt2) ) ** 0.5

        if Distance < 1e-10:
            LeadingOrTrailingEdge = Pt1

        else:
            C.convertPyTree2File(LeadingOrTrailingEdge,'debug.cgns')
            ERRMSG = 'Unexpected double point. Dumped debug.cgns'
            raise ValueError(ERRMSG)

    else:
        C.convertPyTree2File(LeadingOrTrailingEdge,'debug.cgns')
        ERRMSG = 'Found non-unique characteristic point. Dumped debug.cgns'
        raise ValueError(ERRMSG)

    return LeadingOrTrailingEdge, rmin



def closeStructCurve(AirfoilCurve, tol=1e-10):
    '''
    Given a 1D curve defined by **AirfoilCurve**, check if it is closed
    within geomatrical tolerance **tol**. Otherwise, add new point and force
    closing.

    Parameters
    ----------

        AirfoilCurve : zone
            1D PyTree curve, input airfoil

    Returns
    -------

        AirfoilCurveOut : zone
            1D PyTree curve, output airfoil (closed)
    '''

    isStructured = checkAirfoilAndGetTopo(AirfoilCurve)

    if not isStructured: AirfoilCurve = C.convertBAR2Struct(AirfoilCurve)

    x, y, z = J.getxyz(AirfoilCurve)

    ExtremumDistance = ( ( x[-1] - x[0] )**2 +
                         ( y[-1] - y[0] )**2 +
                         ( z[-1] - z[0] )**2   ) ** 0.5

    if ExtremumDistance > tol:
        AirfoilCurve = G.addPointInDistribution(AirfoilCurve, len(x))

        x, y, z = J.getxyz(AirfoilCurve)

        x[-1] = x[0]
        y[-1] = y[0]
        z[-1] = z[0]

    return AirfoilCurve




def splitAirfoil(AirfoilCurve, FirstEdgeSearchPortion = 0.50,
        SecondEdgeSearchPortion = -0.50, RelativeRadiusTolerance = 1e-2,
        MergePointsTolerance = 1e-10,  DistanceCriterionTolerance = 1e-5,
        FieldCriterion='CoordinateY',
        SideChoiceCriteriaPriorities=['field','distance']):
    '''
    Split an airfoil shape into *top* (suction) and *bottom* (pressure) sides.

    Parameters
    ----------

        AirfoilCurve : zone
            1D PyTree zone, The airfoil shape.

        FirstEdgeSearchPortion : float
            Used for determining the search region
            of the *first* characteristic point of the airfoil (Leading Edge or
            Trailing Edge), so that search region is given by:

            ::

                {ChordwiseIndicator} > FirstEdgeSearchPortion

            Hence, **FirstEdgeSearchPortion** MUST be :math:`\in (0,1)`

        SecondEdgeSearchPortion : float
            Used for determining the search region
            of the *second* characteristic point of the airfoil (Leading Edge or
            Trailing Edge), so that search region is given by:

            ::

                {ChordwiseIndicator} < SecondEdgeSearchPortion

            Hence, **SecondEdgeSearchPortion** MUST be :math:`\in (-1, 0)`

        RelativeRadiusTolerance : float
            relatively small value used to
            determine the characteristic points region as specified in the
            documentation of function :py:func:`findLeadingOrTrailingEdge`

        MergePointsTolerance : float
            Small value used to infer if  characteristic points are to be merged
            on final sides result or not. It is also used for determining if
            input airfoil shall be closed or not.

    Returns
    -------

        TopSide : zone
            Structured curve, with increasing index from
            Leading Edge towards Trailing Edge, corresponding to Top Side.

        BottomSide : zone
            Structured curve, with increasing index from
            Leading Edge towards Trailing Edge, corresponding to Bottom Side.

    '''


    def splitSide(StartPoint, EndPoint):
        StartIndex =     getNextAbscissaIndex( StartPoint )
        EndIndex   = getPreviousAbscissaIndex( EndPoint )

        if EndIndex > StartIndex:
            side = T.subzone(AirfoilCurve, (StartIndex+1,1,1), (EndIndex+1,1,1))
        else:
            FirstPortion = T.subzone(AirfoilCurve, (StartIndex+1, 1, 1),
                                                    (NPtsAirfoil, 1, 1) )
            SecondPortion = T.subzone(AirfoilCurve, (         1, 1, 1),
                                                    (EndIndex+1, 1, 1) )
            side = T.join( FirstPortion, SecondPortion )

        PointsToEvaluateDistance = [J.getxyz(StartPoint), J.getxyz(EndPoint)]
        IndicesAndSquaredDistances = D.getNearestPointIndex(side,
                                                      PointsToEvaluateDistance)

        addStartPoint = IndicesAndSquaredDistances[0][1] ** 0.5 > MergePointsTolerance
        addEndPoint   = IndicesAndSquaredDistances[1][1] ** 0.5 > MergePointsTolerance

        I._rmNodesByName([StartPoint, side, EndPoint],
                         'ChordwiseIndicator')
        I._rmNodesByName([StartPoint, side, EndPoint],
                         'ThickwiseIndicator')
        if addStartPoint:
            side = concatenate([StartPoint, side])

        if addEndPoint:
            side = concatenate([side, EndPoint])

        return side

    def getNextAbscissaIndex(Point):
        PointAbscissa, = J.getVars(Point, ['s'])

        if PointAbscissa < 1:
            NextAbscissa = np.where(CurvilinearAbscissa > PointAbscissa)[0]
        else:
            NextAbscissa = np.where(CurvilinearAbscissa < PointAbscissa)[0]

        try:
            NextAbscissaIndex = NextAbscissa[0]
        except IndexError:
            ERRMSG = 'no abscissa found next to point {} with s={}'.format(
                Point[0], PointAbscissa)
            raise ValueError(ERRMSG)

        return NextAbscissaIndex

    def getPreviousAbscissaIndex(Point):
        PointAbscissa, = J.getVars(Point, ['s'])
        if PointAbscissa > 0:
            PreviousAbscissa = np.where(CurvilinearAbscissa < PointAbscissa)[0]
        else:
            PreviousAbscissa = np.where(CurvilinearAbscissa > PointAbscissa)[0]

        try:
            PreviousAbscissaIndex = PreviousAbscissa[-1]
        except IndexError:
            ERRMSG = 'no abscissa found previous to point {} with s={}'.format(
                Point[0], PointAbscissa)
            raise ValueError(ERRMSG)

        return PreviousAbscissaIndex

    def getMaximumDistanceToChordLine(side, LeadingEdge, ChordDirection):
        DistanceFieldName = 'Distance2ChordLine'
        addDistanceRespectToLine(side, LeadingEdge, ChordDirection,
                                 FieldNameToAdd=DistanceFieldName)
        Distance2ChordLine = J.getVars(side, [DistanceFieldName])
        MaxDistance2ChordLine = np.max(Distance2ChordLine)
        return MaxDistance2ChordLine

    def reorderSideFromLeadingToTrailingEdge(side):
        NearestLeadingEdgeIndex, _ = D.getNearestPointIndex( side,
                                                        tuple(LeadingEdgeCoords) )

        if NearestLeadingEdgeIndex > C.getNPts(side)/2:
            T._reorder(side, (-1,2,3))

    AirfoilCurve = I.copyRef(AirfoilCurve)
    I._rmNodesByType(AirfoilCurve, 'FlowSolution_t')
    AirfoilCurve = closeCurve( AirfoilCurve, NPts4closingGap=4,
                               tol=MergePointsTolerance )
    CurvilinearAbscissa = gets( AirfoilCurve )
    NPtsAirfoil = len( CurvilinearAbscissa )

    LE, LErmin = findLeadingOrTrailingEdge( AirfoilCurve,
                               ChordwiseRegion='> %g'%FirstEdgeSearchPortion,
                               ToleranceRelativeRadius=RelativeRadiusTolerance)

    TE, TErmin = findLeadingOrTrailingEdge( AirfoilCurve,
                               ChordwiseRegion='< %g'%SecondEdgeSearchPortion,
                               ToleranceRelativeRadius=RelativeRadiusTolerance)




    if TErmin > LErmin:
        TErmin , LErmin = LErmin , TErmin
        TE , LE = LE , TE
    C._rmVars([LE, TE], 'radius')
    LE[0] = 'LeadingEdge'
    TE[0] = 'TrailingEdge'
    LeadingEdgeCoords  = np.array(J.getxyz( LE )).flatten()
    TrailingEdgeCoords = np.array(J.getxyz( TE )).flatten()
    Chord = distance(LeadingEdgeCoords, TrailingEdgeCoords)
    ChordDirection = TrailingEdgeCoords - LeadingEdgeCoords
    for receiver in [LE, TE]:
        T._projectOrtho( receiver, AirfoilCurve )
        P._extractMesh( AirfoilCurve,
                        receiver,
                        mode='accurate',
                        extrapOrder=0,
                        constraint=MergePointsTolerance,
                        tol=MergePointsTolerance )


    BottomSide = splitSide( TE, LE )
    I.setName(BottomSide, 'BottomSideCandidate')
    TopSide    = splitSide( LE, TE )
    I.setName(TopSide, 'TopSideCandidate')

    for criterion in SideChoiceCriteriaPriorities:
        if criterion == 'distance':
            TopSideMaxDistanceToChordLine = getMaximumDistanceToChordLine(
                                                           TopSide,
                                                           LeadingEdgeCoords,
                                                           ChordDirection)/Chord
            BottomSideMaxDistanceToChordLine = getMaximumDistanceToChordLine(
                                                           BottomSide,
                                                           LeadingEdgeCoords,
                                                           ChordDirection)/Chord

            TooClose = abs(TopSideMaxDistanceToChordLine - \
                           BottomSideMaxDistanceToChordLine) \
                           < DistanceCriterionTolerance

            if TooClose:
                print('sides are too close to camber for using "distance"\n'
                      'criterion for naming top/bottom sides. Skipping.')
                continue

            if BottomSideMaxDistanceToChordLine > TopSideMaxDistanceToChordLine:
                TopSide, BottomSide = BottomSide, TopSide

        if criterion == 'field':
            fieldMaxTop = C.getMaxValue(TopSide, FieldCriterion)
            fieldMaxBottom = C.getMaxValue(BottomSide, FieldCriterion)

            if fieldMaxBottom > fieldMaxTop:
                TopSide, BottomSide = BottomSide, TopSide

    TopSide[0]    = 'TopSide'
    BottomSide[0] = 'BottomSide'

    reorderSideFromLeadingToTrailingEdge(    TopSide )
    reorderSideFromLeadingToTrailingEdge( BottomSide )

    return  TopSide, BottomSide

def computePlaneEquation(point, normal):
    '''
    Compute the plane equation using a passing point and a normal direction.

    Parameters
    ----------

        point : :py:class:`list` or :py:class:`tuple` or numpy of 3 :py:class:`float`
            :math:`(x,y,z)` coordinates of point

        normal : :py:class:`list` or :py:class:`tuple` or numpy of 3 :py:class:`float`
            3 components of the unitary normal vector of the plane,
            :math:`\\vec{n} = (n_x, n_y, n_z)`

    Returns
    -------

        Equation : str
            :py:func:`Converter.PyTree.initVars`-compatible string used as equation
    '''
    p = np.array(point)
    n = np.array(normal)
    A, B, C = n
    D = -n.dot(p)

    Equation = '{A}*{x}+{B}*{y}+{C}*{z}+{D}'.format(
        A=A, B=B, C=C, D=D,
        x="{CoordinateX}",
        y="{CoordinateY}",
        z="{CoordinateZ}")

    return Equation


def buildCamber(AirfoilCurve, MaximumControlPoints=100, StepControlPoints=10,
        StartRelaxationFactor=0.5, ConvergenceTolerance=1e-6, MaxIters=500,
        FinalDistribution=None, splitAirfoilOptions={}):
    '''
    Given an **AirfoilCurve**, build the Camber line oriented from Leading Edge
    towards Trailing Edge and containing ``{RelativeThickness}`` field.

    Parameters
    ----------

        AirfoilCurve : zone
            Airfoil structured curve from which to build the camber line

        MaximumControlPoints : int
            Maximum number of control points for determining the camber line
            during its iterative process.

        StepControlPoints : int
            Number of points to increase after each
            iterative step of searching of camber line.

        StartRelaxationFactor : float
            starting relaxation factor

        ConvergenceTolerance : float
            convergence tolerance threshold criterion
            used to determine if camber has been found. Residual must be lower
            to convergence tolerance in order to satisfy convergence condition.
            Residual is the L2 norm of the distance to top side minus the distance
            to bottom side.

        MaxIters : int
            maximum number of iterations for computing the camber line

        FinalDistribution : dict
            a :py:func:`linelaw`-compatible distribution
            dictionary for the discretization of the camber line

        splitAirfoilOptions : dict
            literally, all parameters passed to :py:func:`splitAirfoil` function

    Returns
    -------

        Camber : zone
            camber line including (among others) the field ``{RelativeThickness}``
    '''


    def prepareCamberLine(curve):
        Fields = J.invokeFieldsDict( curve,
                                   ['SquaredDistanceTop',
                                    'SquaredDistanceBottom',
                                    'TangentX',
                                    'TangentY',
                                    'TangentZ',
                                    'TopVectorX',
                                    'TopVectorY',
                                    'TopVectorZ',
                                    'BottomVectorX',
                                    'BottomVectorY',
                                    'BottomVectorZ',
                                    'residual'])
        Tangents = [Fields['TangentX'],
                    Fields['TangentY'],
                    Fields['TangentZ']]

        Coords = J.getxyz(curve)

        return Fields, Tangents, Coords


    def updatePerpendicularSquaredDistance(side, sideName):
        SquaredDistance = CamberPolylineFields['SquaredDistance'+sideName]
        VectorX = CamberPolylineFields[sideName+'VectorX']
        VectorY = CamberPolylineFields[sideName+'VectorY']
        VectorZ = CamberPolylineFields[sideName+'VectorZ']
        x,   y,  z = CamberPolylineCoords
        tx, ty, tz = CamberPolylineTangents
        SideX, SideY, SideZ = J.getxyz(side)
        SliceVar, = J.getVars(side, ['SliceVar'])
        for i in range(1, NCtrlPts-1):
            xi  =  x[i]
            yi  =  y[i]
            zi  =  z[i]
            txi = tx[i]
            tyi = ty[i]
            tzi = tz[i]

            SliceVar[:] = txi*SideX + \
                          tyi*SideY + \
                          tzi*SideZ - \
                          (txi*xi + tyi*yi + tzi*zi)

            SliceX = np.interp(0.,SliceVar, SideX)
            SliceY = np.interp(0.,SliceVar, SideY)
            SliceZ = np.interp(0.,SliceVar, SideZ)

            VectorX[i] = SliceX - x[i]
            VectorY[i] = SliceY - y[i]
            VectorZ[i] = SliceZ - z[i]

        SquaredDistance[:] = VectorX**2 + VectorY**2 + VectorZ**2


    def updateTangents():
        for coord, tang in zip(CamberPolylineCoords, CamberPolylineTangents):
            tang[1:-1] = 0.5 * ( np.diff(coord[:-1]) + np.diff(coord[1:]) )
            tang[0]  = coord[ 1] - coord[ 0]
            tang[-1] = coord[-1] - coord[-2]
        C._normalize(CamberPolyline, ['TangentX', 'TangentY', 'TangentZ'])


    def displaceCamber(relaxFactor):
        for coord, coordTag in zip(CamberPolylineCoords, ['X','Y','Z']):
            coord[:] += 0.5*(CamberPolylineFields['TopVector'+coordTag] +
                             CamberPolylineFields['BottomVector'+coordTag])*\
                             relaxFactor

    def getResidual():
        residualField = CamberPolylineFields['residual']
        SqrdDistTop   = CamberPolylineFields['SquaredDistanceTop']
        SqrdDistBotom = CamberPolylineFields['SquaredDistanceBottom']
        residualField[:] = abs( SqrdDistTop - SqrdDistBotom )
        normL2 = C.normL2(CamberPolyline,'residual')
        return normL2

    InitialAirfoilNPts = C.getNPts(AirfoilCurve)

    TopSide, BottomSide = splitAirfoil( AirfoilCurve, **splitAirfoilOptions)

    C._initVars(    TopSide, 'SliceVar', 0.)
    C._initVars( BottomSide, 'SliceVar', 0.)

    TopSideNPts  = C.getNPts(TopSide)
    LeadingEdge  = T.subzone(TopSide,(1,1,1),(1,1,1))
    TrailingEdge = T.subzone(TopSide,(TopSideNPts,1,1),(TopSideNPts,1,1))
    Chord = distance(LeadingEdge, TrailingEdge)


    NCtrlPts = 5
    CamberPolyline = D.line( J.getxyz( LeadingEdge ),
                             J.getxyz( TrailingEdge ),
                             NCtrlPts )

    ptrs = prepareCamberLine(CamberPolyline)
    CamberPolylineFields, CamberPolylineTangents, CamberPolylineCoords = ptrs

    AllZones = [TopSide, BottomSide]

    updateTangents()
    updatePerpendicularSquaredDistance(    TopSide, 'Top' )
    updatePerpendicularSquaredDistance( BottomSide, 'Bottom' )
    displaceCamber(1.0)
    PreviousResidual = getResidual()

    relax = StartRelaxationFactor
    for NCtrlPts in range(10, MaximumControlPoints+1, StepControlPoints):
        CamberPolyline = discretize(CamberPolyline, N=NCtrlPts)
        ptrs = prepareCamberLine(CamberPolyline)
        CamberPolylineFields, CamberPolylineTangents, CamberPolylineCoords = ptrs

        try:
            for j in range( MaxIters ):
                updateTangents()
                updatePerpendicularSquaredDistance(    TopSide, 'Top' )
                updatePerpendicularSquaredDistance( BottomSide, 'Bottom' )
                displaceCamber(relax)
                residual = getResidual()

                CONVERGED = residual < ConvergenceTolerance

                if CONVERGED:
                    Camber = I.copyTree(CamberPolyline)
                    break

                if residual > PreviousResidual:
                    relax = np.maximum(relax-0.02, 0.05)
                PreviousResidual = residual
        except:
            break

    if not FinalDistribution:
        FinalDistribution=dict(N=int((InitialAirfoilNPts+2)/2),
                               kind='trigonometric', parameter=1)


    NCtrlPts = FinalDistribution['N']
    Camber = discretize(Camber, N=NCtrlPts, Distribution=FinalDistribution)
    CamberPolyline = Camber
    ptrs = prepareCamberLine(CamberPolyline)
    e, = J.invokeFields(CamberPolyline,['RelativeThickness'])
    CamberPolylineFields, CamberPolylineTangents, CamberPolylineCoords = ptrs
    updateTangents()
    updatePerpendicularSquaredDistance(    TopSide, 'Top' )
    updatePerpendicularSquaredDistance( BottomSide, 'Bottom' )
    e[:] = (np.sqrt(CamberPolylineFields['SquaredDistanceTop']) + \
            np.sqrt(CamberPolylineFields['SquaredDistanceBottom'])) / Chord

    if not CONVERGED:
        print('Camber line not converged with residual %g'%residual)

    CamberPolyline[0] += '.camber'

    return CamberPolyline


def getAirfoilPropertiesAndCamber(AirfoilCurve, buildCamberOptions={},
                                  splitAirfoilOptions={}):
    '''
    This function computes the geometrical properties of an airfoil, and
    returns them as a python dictionary and stores them as ``UserDefinedData_t``
    node into the airfoil zone.

    The computed geometrical characterstics are :
        ``'LeadingEdge'``,
        ``'TrailingEdge'``,
        ``'Chord'``,
        ``'ChordDirection'``, ``'BinormalDirection'``, ``'NormalDirection'``
        ``'MaxRelativeThickness'``
        ``'MaxThickness'``
        ``'MaxThicknessRelativeLocation'``
        ``'MaxCamber'``
        ``'MaxRelativeCamber'``
        ``'MaxCamberRelativeLocation'``
        ``'MinCamber'``
        ``'MinRelativeCamber'``
        ``'MinCamberRelativeLocation'``

    Parameters
    ----------

        AirfoilCurve : zone
            structured curve of the airfoil.

            .. note:: **AirfoilCurve** is modified

        buildCamberOptions : dict
            literally, options to pass to :py:func:`buildCamber` function

        splitAirfoilOptions : dict
            literally, options to pass to :py:func:`splitAirfoil` function

    Returns
    -------

        AirfoilProperties : :py:class:`dict`
            contains the aforementioned airfoil geomatrical characteristics

        CamberLine : zone
            the camber line of the airfoil
    '''

    AirfoilProperties = dict()
    AirfoilCurve = I.copyRef(AirfoilCurve)
    TopSide, BottomSide = splitAirfoil(AirfoilCurve, **splitAirfoilOptions)
    TopX, TopY, TopZ = J.getxyz(TopSide)
    BottomX, BottomY, BottomZ = J.getxyz(BottomSide)

    buildCamberOptions['splitAirfoilOptions'] = splitAirfoilOptions
    CamberLine = buildCamber(AirfoilCurve, **buildCamberOptions)
    CamberLineX, CamberLineY, CamberLineZ = J.getxyz(CamberLine)
    RelativeThickness, = J.getVars(CamberLine, ['RelativeThickness'])

    LeadingEdge  = np.array([CamberLineX[0],
                             CamberLineY[0],
                             CamberLineZ[0]], dtype=np.float)
    AirfoilProperties['LeadingEdge'] = LeadingEdge

    TrailingEdge = np.array([CamberLineX[-1],
                             CamberLineY[-1],
                             CamberLineZ[-1]], dtype=np.float)
    AirfoilProperties['TrailingEdge'] = TrailingEdge

    ChordDirection = TrailingEdge - LeadingEdge
    Chord = np.sqrt(ChordDirection.dot(ChordDirection))
    ChordDirection /= Chord
    AirfoilProperties['Chord'] = Chord
    AirfoilProperties['ChordDirection'] = ChordDirection


    TopSideX, TopSideY, TopSideZ = J.getxyz(TopSide)
    ChordCoplanarDirection = np.array([TopSideX[1]-TopSideX[0],
                                       TopSideY[1]-TopSideY[0],
                                       TopSideZ[1]-TopSideZ[0]])
    ChordCoplanarDirection/=np.sqrt(ChordCoplanarDirection.dot(ChordCoplanarDirection))

    BinormalDirection = np.cross(ChordDirection, ChordCoplanarDirection)
    BinormalDirection /= np.sqrt(BinormalDirection.dot(BinormalDirection))
    AirfoilProperties['BinormalDirection'] = BinormalDirection
    NormalDirection = np.cross(BinormalDirection, ChordDirection)
    AirfoilProperties['NormalDirection'] = NormalDirection

    CamberValues = np.empty_like(CamberLineX)
    for i in range(len(CamberValues)):
        CamberValues[i] =((CamberLineX[i]-LeadingEdge[0])*NormalDirection[0] +
                          (CamberLineY[i]-LeadingEdge[1])*NormalDirection[1] +
                          (CamberLineZ[i]-LeadingEdge[2])*NormalDirection[2])



    iMaxThickness = np.argmax(RelativeThickness)
    AirfoilProperties['MaxRelativeThickness'] = RelativeThickness[iMaxThickness]
    AirfoilProperties['MaxThickness'] = RelativeThickness[iMaxThickness]*Chord
    MaxThicknessCoords = np.array([CamberLineX[iMaxThickness],
                                   CamberLineY[iMaxThickness],
                                   CamberLineZ[iMaxThickness]],dtype=np.float)
    MaxThicknessLocation = (MaxThicknessCoords-LeadingEdge).dot(ChordDirection)
    MaxThicknessLocation /= Chord
    AirfoilProperties['MaxThicknessRelativeLocation'] = MaxThicknessLocation

    iMaxCamber = np.argmax(CamberValues)
    AirfoilProperties['MaxCamber'] = CamberValues[iMaxCamber]
    AirfoilProperties['MaxRelativeCamber'] = CamberValues[iMaxCamber] / Chord
    MaxCamberCoords = np.array([CamberLineX[iMaxCamber],
                                CamberLineY[iMaxCamber],
                                CamberLineZ[iMaxCamber]],dtype=np.float)
    MaxCamberLocation = (MaxCamberCoords-LeadingEdge).dot(ChordDirection)
    MaxCamberLocation /= Chord
    AirfoilProperties['MaxCamberRelativeLocation'] = MaxCamberLocation

    iMinCamber = np.argmin(CamberValues)
    AirfoilProperties['MinCamber'] = CamberValues[iMinCamber]
    AirfoilProperties['MinRelativeCamber'] = CamberValues[iMinCamber] / Chord
    MinCamberCoords = np.array([CamberLineX[iMinCamber],
                                CamberLineY[iMinCamber],
                                CamberLineZ[iMinCamber]],dtype=np.float)
    MinCamberLocation = (MinCamberCoords-LeadingEdge).dot(ChordDirection)
    MinCamberLocation /= Chord
    AirfoilProperties['MinCamberRelativeLocation'] = MinCamberLocation


    return AirfoilProperties, CamberLine


def normalizeFromAirfoilProperties(t, AirfoilProperties, Fields2Rotate=[]):
    '''
    Performs in-place normalization of input PyTree **t** following the data
    contained in the dictionary **AirfoilProperties**, which can be obtained
    using :py:func:`getAirfoilPropertiesAndCamber` function.

    **Fields2Rotate** are passed to :py:func:`Transform.PyTree.rotate` function,
    and allows for rotation of vector fields.

    Parameters
    ----------

        t : PyTree, base, zone, list of zones
            object containing the item
            to be normalized following the provided airfoil properties.

            .. note:: Geometry (and eventually fields) of **t** are modified

        AirfoilProperties : dict
            as obtained from the function :py:func:`getAirfoilPropertiesAndCamber`

        Fields2Rotate : :py:func:`list` of :py:func:`str`
            list containing the field names to rotate during the normalization
    '''
    LeadingEdge = AirfoilProperties['LeadingEdge']
    Chord = AirfoilProperties['Chord']
    Frenet = (tuple(AirfoilProperties['ChordDirection']),
              tuple(AirfoilProperties['NormalDirection']),
              tuple(AirfoilProperties['BinormalDirection']))

    FrenetDestination = ((1.0,0.0,0.0),
                         (0.0,1.0,0.0),
                         (0.0,0.0,1.0))

    T._translate(t,-LeadingEdge)
    T._rotate(t,(0,0,0),Frenet,arg2=FrenetDestination, vectors=Fields2Rotate)
    T._homothety(t,(0,0,0),1./Chord)


def addDistanceRespectToLine(t, LinePassingPoint, LineDirection,
                                FieldNameToAdd='Distance2Line'):
    '''
    Add the distance to line of the points of a zone in form of a new field.

    Parameters
    ----------

        t : PyTree, base, zone or list of zones
            grid where distance to
            line is to be computed.

            .. important:: a field named after parameter **FieldNameToAdd** will
                be added to **t** at container ``FlowSolution``

        LinePassingPoint : array of 3 :py:class:`float`
            coordinates of the passing point :math:`(x,y,z)` of the line

        LineDirection : array of 3 :py:class:`float`
            unitary vector of the line direction :math:`\\vec{l} = (l_x,l_y,l_z)`

        FieldNameToAdd : str
            name to give to the new field to be added in **t**, representing the
            distance to the line
    '''
    a = np.array(LinePassingPoint,dtype=np.float)
    n = np.array(LineDirection,dtype=np.float)
    n /= np.sqrt(n.dot(n))
    for zone in I.getZones(t):
        x,y,z = J.getxyz(zone)
        x = x.ravel(order='K')
        y = y.ravel(order='K')
        z = z.ravel(order='K')
        Distance2Line, = J.invokeFields(zone, [FieldNameToAdd])
        Distance2Line = Distance2Line.ravel(order='K')
        for i in range(len(x)):
            p = MeshPoint = np.array([x[i], y[i], z[i]])
            PassingPoint2MeshPoint = a - MeshPoint
            v = (a-p)- ((a-p).dot(n))*n
            Distance2Line[i] = np.sqrt(v.dot(v))


def modifyAirfoil(AirfoilInput, Chord=None,
                  MaxThickness=None, MaxRelativeThickness=None,
                  MaxThicknessRelativeLocation=None,
                  MaxCamber=None, MaxRelativeCamber=None,
                  MaxCamberRelativeLocation=None,
                  MinCamber=None, MinRelativeCamber=None,
                  MinCamberRelativeLocation=None,
                  ScalingRelativeChord=0.25,
                  ScalingMode='auto',
                  buildCamberOptions={},
                  splitAirfoilOptions={},
                  InterpolationLaw='interp1d_cubic'):
    '''
    Create new airfoil by modifying geometrical properties of a provided
    airfoil curve.

    Parameters
    ----------

        AirfoilInput : zone
            structured zone representing the airfoil

        Chord : float
            Aimed chord [m] length of the new airfoil.
            Use :py:obj:`None` if this parameter is not aimed.

        MaxThickness : float
            Aimed thickness [m] of the new airfoil.
            Use :py:obj:`None` if this parameter is not aimed.

        MaxRelativeThickness : float
            Aimed relative thickness (with respect to chord length) of the new airfoil.
            Use :py:obj:`None` if this parameter is not aimed.

        MaxThicknessRelativeLocation : float
            Relative chordwise location of maximum thickness, with respect to the
            chord length. Must be :math:`\in (0,1)`
            Use :py:obj:`None` if this parameter is not aimed.

        MaxCamber : float
            Aimed maximum camber [m] (in top-side direction) of the new airfoil.
            Use :py:obj:`None` if this parameter is not aimed.

        MaxRelativeCamber : float
            Aimed relative (with respect to chord length) maximum camber
            (in top-side direction) of the new airfoil.
            Use :py:obj:`None` if this parameter is not aimed.

        MaxCamberRelativeLocation : float
            Relative chordwise location of maximum camber in top-side direction),
            with respect to the chord length. Must be :math:`\in (0,1)`
            Use :py:obj:`None` if this parameter is not aimed.

        MinCamber : float
            Aimed minimum camber [m] (in bottom-side direction) of the new airfoil.
            Use :py:obj:`None` if this parameter is not aimed.

        MinRelativeCamber : float
            Aimed relative (with respect to chord length) minimum camber
            (in bottom-side direction) of the new airfoil.
            Use :py:obj:`None` if this parameter is not aimed.

        MinCamberRelativeLocation : float
            Relative chordwise location (with respect to chord length) o
            minimum camber (in bottom-side direction), with respect to the
            chord length. Must be :math:`\in (0,1)`
            Use :py:obj:`None` if this parameter is not aimed.

        ScalingRelativeChord : float
            relative chordwise position at which scaling is applied.
            It must be :math:`\in (0,1)`

        ScalingMode : str
            How to scale the airfoil. Two possibilities:

            * ``'auto'``
                based on the chord length extracted after computing camber line

            * ``'airfoil'``
                based on the chord length approximated using the straight line
                from leading edge towards trailing edge

        buildCamberOptions : dict
            literally, options passed to :py:func:`buildCamber` function

        splitAirfoilOptions : dict
            literally, options passed to :py:func:`splitAirfoil` function

        InterpolationLaw : str
            interpolation law to be applied for the modification of camber and
            thickness

    Returns
    -------

        NewAirfoil : zone
            structured curve of the new airfoil
    '''


    Airfoil = I.copyTree(AirfoilInput)

    AirfoilProperties, Camber = getAirfoilPropertiesAndCamber(AirfoilInput,
                                        buildCamberOptions=buildCamberOptions,
                                        splitAirfoilOptions=splitAirfoilOptions)

    TE = AirfoilProperties['TrailingEdge']
    LE = AirfoilProperties['LeadingEdge']
    CamberCenter = ScalingRelativeChord*(TE-LE) + LE

    if ScalingMode == 'auto':
        Center = -CamberCenter
        Scale = Chord/AirfoilProperties['Chord'] if Chord else 1.0
    elif ScalingMode == 'airfoil':
        Center = (-LE[0]-ScalingRelativeChord,-LE[1],-LE[2])
        Scale = Chord if Chord else 1.0
    else:
        raise AttributeError('ScalingMode %s not recognized'%ScalingMode)


    T._translate(Camber,Center)
    T._homothety(Camber, (0,0,0), Scale)
    AirfoilProperties['ScalingCenter'] = (0,0,0)
    AirfoilProperties['Chord'] = Chord
    # TODO: update TrailingEdge and perhaps other characteristics


    NormalDirection = AirfoilProperties['NormalDirection']
    Camber = modifyThicknessOfCamberLine(Camber,
                    NormalDirection,
                    MaxThickness=MaxThickness,
                    MaxRelativeThickness=MaxRelativeThickness,
                    MaxThicknessRelativeLocation=MaxThicknessRelativeLocation,
                    InterpolationLaw=InterpolationLaw)

    Camber = modifyCamberOfCamberLine(Camber,
                    NormalDirection,
                    MaxCamber=MaxCamber,
                    MaxRelativeCamber=MaxRelativeCamber,
                    MaxCamberRelativeLocation=MaxCamberRelativeLocation,
                    MinCamber=MinCamber,
                    MinRelativeCamber=MinRelativeCamber,
                    MinCamberRelativeLocation=MinCamberRelativeLocation,
                    InterpolationLaw=InterpolationLaw)

    NewAirfoil = buildAirfoilFromCamberLine(Camber, NormalDirection)
    NewAirfoil[0] = AirfoilInput[0]+'.mod'
    J.set(NewAirfoil, '.AirfoilProperties', **AirfoilProperties)

    return NewAirfoil

def modifyThicknessOfCamberLine(CamberCurve, NormalDirection, MaxThickness=None,
                MaxRelativeThickness=None, MaxThicknessRelativeLocation=None,
                InterpolationLaw='interp1d_cubic'):
    '''
    Modify the ``RelativeThickness`` fields contained in a CamberLine zone.

    Parameters
    ----------

        CamberCurve : zone
            the camber curve to modify, as obtained from :py:func:`buildCamber`

            .. note::  **CamberCurve** is modified

        NormalDirection : array of 3 :py:class:`float`
            the normal direction, perpendicular to the chord direction, and
            pointing towards the top side of airfoil

        MaxThickness : float
            if provided, sets the maximum thickness in absolute value [m]

        MaxRelativeThickness : float
            if provided, sets the maximum thickness in relative value
            (normalized by the chord length)

        MaxThicknessRelativeLocation : float
            if provided, sets the chordwise relative location at which thickness
            is maximum.

        InterpolationLaw : str
            interpolation law to be applied for the modification of thickness

            .. note:: **InterpolationLaw** is the parameter **Law** of function
                :py:func:`MOLA.InternalShortcuts.interpolate__`
    '''

    needModification = MaxThickness or MaxRelativeThickness or MaxThicknessRelativeLocation
    if not needModification: return CamberCurve

    if MaxThickness and MaxRelativeThickness:
        raise AttributeError('Cannot specify both relative and absolute thickness')

    CamberLine = I.copyTree(CamberCurve)

    NormalDirection = np.array(NormalDirection)
    NormalDirection/= np.sqrt(NormalDirection.dot(NormalDirection))

    x,y,z = J.getxyz(CamberLine)
    LeadingEdge = np.array([x[0], y[0], z[0]])
    Chord = distance(LeadingEdge, (x[-1],y[-1],z[-1]) )
    ChordDirection = np.array([x[-1]-x[0], y[-1]-y[0], z[-1]-z[0]])
    ChordDirection /= np.sqrt(ChordDirection.dot(ChordDirection))
    BinormalDirection = np.cross(ChordDirection,NormalDirection)
    BinormalDirection /= np.sqrt(BinormalDirection.dot(BinormalDirection))
    FrenetOriginal = (tuple(ChordDirection),
                      tuple(NormalDirection),
                      tuple(BinormalDirection))
    FrenetAuxiliary = ((1,0,0),
                       (0,1,0),
                       (0,0,1))

    T._translate(CamberLine,-LeadingEdge)
    T._rotate(CamberLine, (0,0,0),FrenetOriginal, FrenetAuxiliary)
    x,y,z = J.getxyz(CamberLine)


    RelativeThickness, = J.getVars(CamberLine, ['RelativeThickness'])
    ArgMaxThickness = np.argmax(RelativeThickness)

    OriginalMaxThicknessRelativeLocation = x[ArgMaxThickness] / Chord

    if MaxThickness:
        MaxRelativeThickness = MaxThickness / Chord

    if MaxRelativeThickness:
        OriginalMaxRelativeThickness = RelativeThickness[ArgMaxThickness]
        RelativeThickness *= MaxRelativeThickness / OriginalMaxRelativeThickness

    if MaxThicknessRelativeLocation:

        FrontCamber = T.subzone(CamberLine,(1,1,1),(ArgMaxThickness+1,1,1))
        FrontCamberX, FrontCamberY, FrontCamberZ = J.getxyz(FrontCamber)
        OriginalJoinPoint = np.array([FrontCamberX[-1],FrontCamberY[-1],FrontCamberZ[-1]])
        RearCamber = T.subzone(CamberLine,(ArgMaxThickness+1,1,1),(-1,-1,-1))
        RearCamberX, RearCamberY, RearCamberZ = J.getxyz(RearCamber)
        TrailingEdge = np.array([RearCamberX[-1],RearCamberY[-1],RearCamberZ[-1]])

        OriginalRearChordwise = abs(TrailingEdge[0]-OriginalJoinPoint[0])

        ScaleFactor =         MaxThicknessRelativeLocation / \
                      OriginalMaxThicknessRelativeLocation

        T._scale(FrontCamber, (ScaleFactor, 1., 1.))
        T._translate(FrontCamber, (-FrontCamberX[0],
                                   -FrontCamberY[0],
                                   -FrontCamberZ[0],))
        NewJoinPoint = np.array([FrontCamberX[-1],FrontCamberY[-1],FrontCamberZ[-1]])


        NewRearChordwise = abs(TrailingEdge[0]-NewJoinPoint[0])
        ScaleFactor = NewRearChordwise / OriginalRearChordwise
        T._scale(RearCamber, (ScaleFactor, 1., 1.))
        T._translate(RearCamber, (x[-1]-RearCamberX[-1],
                                  y[-1]-RearCamberY[-1],
                                  z[-1]-RearCamberZ[-1],))

        AuxCamberLine = T.join(FrontCamber, RearCamber)
        AuxRelativeThickness, = J.getVars(AuxCamberLine, ['RelativeThickness'])
        AuxX = J.getx(AuxCamberLine)

        RelativeThickness[:] = J.interpolate__(x, AuxX, AuxRelativeThickness,
                                               Law=InterpolationLaw)


    T._rotate(CamberLine, (0,0,0), FrenetAuxiliary, FrenetOriginal)
    T._translate(CamberLine, LeadingEdge)

    return CamberLine


def modifyCamberOfCamberLine(CamberCurve, NormalDirection,
        MaxCamber=None, MaxRelativeCamber=None, MaxCamberRelativeLocation=None,
        MinCamber=None, MinRelativeCamber=None, MinCamberRelativeLocation=None,
        InterpolationLaw='interp1d_cubic'):
    '''
    Modify the camber geometry of a user-provided camber line.

    Parameters
    ----------

        CamberCurve : zone
            the camber curve to modify, as obtained from :py:func:`buildCamber`

            .. note::  **CamberCurve** is modified

        NormalDirection : array of 3 :py:class:`float`
            the normal direction, perpendicular to the chord direction, and
            pointing towards the top side of airfoil

        MaxCamber : float
            Aimed maximum camber [m] (in top-side direction) of the new airfoil.
            Use :py:obj:`None` if this parameter is not aimed.

        MaxRelativeCamber : float
            Aimed relative (with respect to chord length) maximum camber
            (in top-side direction) of the new airfoil.
            Use :py:obj:`None` if this parameter is not aimed.

        MaxCamberRelativeLocation : float
            Relative chordwise location of maximum camber in top-side direction),
            with respect to the chord length. Must be :math:`\in (0,1)`
            Use :py:obj:`None` if this parameter is not aimed.

        MinCamber : float
            Aimed minimum camber [m] (in bottom-side direction) of the new airfoil.
            Use :py:obj:`None` if this parameter is not aimed.

        MinRelativeCamber : float
            Aimed relative (with respect to chord length) minimum camber
            (in bottom-side direction) of the new airfoil.
            Use :py:obj:`None` if this parameter is not aimed.

        MinCamberRelativeLocation : float
            Relative chordwise location (with respect to chord length) o
            minimum camber (in bottom-side direction), with respect to the
            chord length. Must be :math:`\in (0,1)`
            Use :py:obj:`None` if this parameter is not aimed.

        InterpolationLaw : str
            interpolation law to be applied for the modification of thickness

            .. note:: **InterpolationLaw** is the parameter **Law** of function
                :py:func:`MOLA.InternalShortcuts.interpolate__`
    '''

    needModification = ((MaxCamber is not None) or
                        (MaxRelativeCamber is not None) or
                        MaxCamberRelativeLocation or
                        (MinCamber is not None) or
                        (MinRelativeCamber is not None) or
                        MinCamberRelativeLocation)
    if not needModification: return CamberCurve

    if (MaxCamber and MaxRelativeCamber) or \
       (MinCamber and MinRelativeCamber):
        raise AttributeError('Cannot specify both relative and absolute camber')

    CamberLine = I.copyTree(CamberCurve)

    NormalDirection = np.array(NormalDirection)
    NormalDirection/= np.sqrt(NormalDirection.dot(NormalDirection))

    # Temporarily put CamberLine in XY reference frame
    x,y,z = J.getxyz(CamberLine)
    NPts = len(x)
    LeadingEdge = np.array([x[0], y[0], z[0]])
    Chord = distance(LeadingEdge, (x[-1],y[-1],z[-1]) )
    ChordDirection = np.array([x[-1]-x[0], y[-1]-y[0], z[-1]-z[0]])
    ChordDirection /= np.sqrt(ChordDirection.dot(ChordDirection))
    BinormalDirection = np.cross(ChordDirection,NormalDirection)
    BinormalDirection /= np.sqrt(BinormalDirection.dot(BinormalDirection))
    FrenetOriginal = (tuple(ChordDirection),
                      tuple(NormalDirection),
                      tuple(BinormalDirection))
    FrenetAuxiliary = ((1,0,0),
                       (0,1,0),
                       (0,0,1))

    T._translate(CamberLine,-LeadingEdge)
    T._rotate(CamberLine, (0,0,0),FrenetOriginal, FrenetAuxiliary)

    # To replicate from here for Max/Min
    x,y,z = J.getxyz(CamberLine)
    RelativeCamber = y
    ArgMaxCamber = np.argmax(RelativeCamber)
    isMaxCamberModificationPossible = 0 < ArgMaxCamber < NPts-1
    if isMaxCamberModificationPossible:
        OriginalMaxCamberRelativeLocation = x[ArgMaxCamber] / Chord

        if MaxCamber is not None: MaxRelativeCamber = MaxCamber / Chord

        if MaxRelativeCamber is not None:
            OriginalMaxRelativeCamber = RelativeCamber[ArgMaxCamber]
            RelativeCamber *= MaxRelativeCamber / OriginalMaxRelativeCamber

        if MaxCamberRelativeLocation:
            FrontCamber = T.subzone(CamberLine,(1,1,1),(ArgMaxCamber+1,1,1))
            FrontCamberX, FrontCamberY, FrontCamberZ = J.getxyz(FrontCamber)
            OriginalJoinPoint = np.array([FrontCamberX[-1],FrontCamberY[-1],FrontCamberZ[-1]])
            RearCamber = T.subzone(CamberLine,(ArgMaxCamber+1,1,1),(-1,-1,-1))
            RearCamberX, RearCamberY, RearCamberZ = J.getxyz(RearCamber)
            TrailingEdge = np.array([RearCamberX[-1],RearCamberY[-1],RearCamberZ[-1]])

            OriginalRearChordwise = abs(TrailingEdge[0]-OriginalJoinPoint[0])

            ScaleFactor =         MaxCamberRelativeLocation / \
                          OriginalMaxCamberRelativeLocation

            T._scale(FrontCamber, (ScaleFactor, 1., 1.))
            T._translate(FrontCamber, (-FrontCamberX[0],
                                       -FrontCamberY[0],
                                       -FrontCamberZ[0],))
            NewJoinPoint = np.array([FrontCamberX[-1],FrontCamberY[-1],FrontCamberZ[-1]])


            NewRearChordwise = abs(TrailingEdge[0]-NewJoinPoint[0])
            ScaleFactor = NewRearChordwise / OriginalRearChordwise
            T._scale(RearCamber, (ScaleFactor, 1., 1.))
            T._translate(RearCamber, (x[-1]-RearCamberX[-1],
                                      y[-1]-RearCamberY[-1],
                                      z[-1]-RearCamberZ[-1],))

            AuxCamberLine = T.join(FrontCamber, RearCamber)
            AuxRelativeCamber = J.gety(AuxCamberLine)
            AuxX = J.getx(AuxCamberLine)

            RelativeCamber[:] = J.interpolate__(x, AuxX, AuxRelativeCamber,
                                                   Law=InterpolationLaw)

    # replicated from here for Min
    x,y,z = J.getxyz(CamberLine)
    RelativeCamber = y
    ArgMinCamber = np.argmin(RelativeCamber)
    isMinCamberModificationPossible = 0 < ArgMinCamber < NPts-1

    if isMinCamberModificationPossible:
        OriginalMinCamberRelativeLocation = x[ArgMinCamber] / Chord

        if MinCamber is not None: MinRelativeCamber = MinCamber / Chord

        if MinRelativeCamber is not None:
            OriginalMinRelativeCamber = RelativeCamber[ArgMinCamber]
            RelativeCamber *= MinRelativeCamber / OriginalMinRelativeCamber

        if MinCamberRelativeLocation:
            FrontCamber = T.subzone(CamberLine,(1,1,1),(ArgMinCamber+1,1,1))
            FrontCamberX, FrontCamberY, FrontCamberZ = J.getxyz(FrontCamber)
            OriginalJoinPoint = np.array([FrontCamberX[-1],FrontCamberY[-1],FrontCamberZ[-1]])
            RearCamber = T.subzone(CamberLine,(ArgMinCamber+1,1,1),(-1,-1,-1))
            RearCamberX, RearCamberY, RearCamberZ = J.getxyz(RearCamber)
            TrailingEdge = np.array([RearCamberX[-1],RearCamberY[-1],RearCamberZ[-1]])

            OriginalRearChordwise = abs(TrailingEdge[0]-OriginalJoinPoint[0])

            ScaleFactor =         MinCamberRelativeLocation / \
                          OriginalMinCamberRelativeLocation

            T._scale(FrontCamber, (ScaleFactor, 1., 1.))
            T._translate(FrontCamber, (-FrontCamberX[0],
                                       -FrontCamberY[0],
                                       -FrontCamberZ[0],))
            NewJoinPoint = np.array([FrontCamberX[-1],FrontCamberY[-1],FrontCamberZ[-1]])


            NewRearChordwise = abs(TrailingEdge[0]-NewJoinPoint[0])
            ScaleFactor = NewRearChordwise / OriginalRearChordwise
            T._scale(RearCamber, (ScaleFactor, 1., 1.))
            T._translate(RearCamber, (x[-1]-RearCamberX[-1],
                                      y[-1]-RearCamberY[-1],
                                      z[-1]-RearCamberZ[-1],))

            AuxCamberLine = T.join(FrontCamber, RearCamber)
            AuxRelativeCamber = J.gety(AuxCamberLine)
            AuxX = J.getx(AuxCamberLine)

            RelativeCamber[:] = J.interpolate__(x, AuxX, AuxRelativeCamber,
                                                   Law=InterpolationLaw)


    T._rotate(CamberLine, (0,0,0), FrenetAuxiliary, FrenetOriginal)
    T._translate(CamberLine, LeadingEdge)

    CamberLine[0] += '.mod'

    return CamberLine


def convertDatFile2PyTreeZone(filename, name='foil', skiprows=1):
    '''
    Convert a ``*.dat`` file containing a 3D structured curve (e.g. as obtained
    from Pointwise) to a suitable CGNS structured zone.
    Coordinates (x, y and z) should be organized vertically in input file.

    Parameters
    ----------

        filename : str
            file name (including relative or absolute path if
            necessary) of the input file to read where coordinates are found.

        name : str
            the name to give to the new zone to be created

        skiprows : int
            number of heading lines to ignore

    Returns
    -------

        curve : zone
            structured curve containing the coordinates of the curve
    '''
    x,y,z = np.loadtxt(filename, delimiter=None, skiprows=skiprows, unpack=True)
    curve = D.line((0,0,0),(1,0,0),len(x))
    xc, yc, zc = J.getxyz(curve)
    xc[:] = x
    yc[:] = y
    zc[:] = z
    I.setName(curve,name)

    return curve
