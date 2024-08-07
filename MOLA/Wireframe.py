#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

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

import MOLA
from . import InternalShortcuts as J
if not MOLA.__ONLY_DOC__:
    # System modules
    import sys
    import os
    import numpy as np
    from copy import deepcopy as cdeep
    from timeit import default_timer as tic

    # Cassiopee
    import Converter.PyTree as C
    import Converter.Internal as I
    import Geom.PyTree as D
    import Post.PyTree as P
    import Generator.PyTree as G
    import Transform.PyTree as T
    import Connector.PyTree as X
    import Intersector.PyTree as XOR


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
        p = np.array([x[0],y[0],z[0]],dtype=np.float64)
    else:
        p = np.array([Point[0],
                      Point[1],
                      Point[2]],dtype=np.float64)

    if isPyTreePoint(LinePassingPoint):
        x,y,z = J.getxyz(LinePassingPoint)
        c = np.array([x[0],y[0],z[0]],dtype=np.float64)
    else:
        c = np.array([LinePassingPoint[0],
                      LinePassingPoint[1],
                      LinePassingPoint[2]],dtype=np.float64)

    l = np.array([LineVector[0],
                  LineVector[1],
                  LineVector[2]],dtype=np.float64)
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
    if float(C.__version__.split('.')[-1]) > 3:
        l = G.enforcePlusX(l,CellStart,(N-2,2),verbose=linelawVerbose)
        l = G.enforceMoinsX(l,CellEnd,(N-2,2),verbose=linelawVerbose)
    else:
        silence = J.OutputGrabber()
        with silence:
            l = G.enforcePlusX(l,CellStart,(N-2,2))
            l = G.enforceMoinsX(l,CellEnd,(N-2,2))

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
    if float(C.__version__.split('.')[-1]) > 3:
        l = G.enforcePlusX(l,CellStart,(N-2,2),verbose=linelawVerbose) if not isCellEnd else G.enforceMoinsX(l,CellStart,(N-2,2),verbose=linelawVerbose)
    else:
        silence = J.OutputGrabber()
        with silence:
            l = G.enforcePlusX(l,CellStart,(N-2,2)) if not isCellEnd else G.enforceMoinsX(l,CellStart,(N-2,2))

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
            # Check there is only one constraint in Distribution 
            onlyFirstCell = 'FirstCellHeight' in Distribution and not 'LastCellHeight' in Distribution
            onlyLastCell = 'LastCellHeight' in Distribution and not 'FirstCellHeight' in Distribution
            MSG = 'tanhOneSide distribution requires "FirstCellHeight" or "LastCellHeight" (and only one of them).'
            assert onlyFirstCell or onlyLastCell, J.FAIL+MSG+J.ENDC

            Length = distance(P1,P2)

            if 'FirstCellHeight' in Distribution:
                dy = Distribution['FirstCellHeight']/Length
                isCellEnd = False
            else:
                dy = Distribution['LastCellHeight']/Length
                isCellEnd = True

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
            Dir = np.array([P2[0]-P1[0],P2[1]-P1[1],P2[2]-P1[2]],dtype=float)/Length
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
        Imported = np.genfromtxt(designation, dtype=np.float64, skip_header=1, usecols=(0,1))
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

        # List of variables to remap
        VarsNames = C.getVarNames(curve)[0]

        # Get the list of arrays, including coordinates
        OldVars = [I.getValue(I.getNodeFromName2(curve,vn)) for vn in VarsNames]
        if 's' in VarsNames:
            OldAbscissa = J.getVars(curve,['s'])[0]
        else:
            OldAbscissa = gets(curve)

        # Get the newly user-defined abscissa
        _,NewAbscissa,_ = J.getDistributionFromHeterogeneousInput__(curve_Distri)

        # Perform remapping (interpolation)
        VarsArrays = [J.interpolate__(NewAbscissa,OldAbscissa,OldVar, Law=MappingLaw) for OldVar in OldVars]

        # Invoke newly remapped curve
        curveMap = J.createZone(curve[0],VarsArrays,VarsNames)

        return curveMap

def discretizeAirfoil(airfoil, Ntop, Nbot=None, CellSizeAtLE=None, CellSizeAtTE=None, relativeToChord=False):
    '''
    *(Re)*-discretize the curve defining the given **airfoil**. The leading is automatically detected, and
    a 'tanhTwoSides' distribution (see doc of :py:func:`linelaw`) is used on suction side and pressure side.  

    Parameters
    ----------
    airfoil : PyTree
        Curve defining the airfoil, generated by :py:func:`airfoil` for instance.

    Ntop : int
        Number of points of the Top side of the airfoil.

    Nbot : int
        Number of points of the Bottom side of the airfoil. If **Nbot** is not
        provided, then **Ntop** is the total number of points of the whole foil.

    CellSizeAtLE : float
        Size of the first cells around the Leading edge.

    CellSizeAtTE : float, optional
        Size of the first cells around the Trailing edge. If not given, it is taken equal to **CellSizeAtLE**.

    relativeToChord : bool, optional
        If :py:obj:`True`, **CellSizeAtLE** and **CellSizeAtTE** are relative to chord length, else they are 
        absolute dimensions. By default :py:obj:`False`

    Returns
    -------
    Airfoil : zone
        The structured curve of the airfoil, oriented from the TE -> PS -> LE -> SS -> TE
    '''

    if not not Ntop and not Nbot:
        Nbot = Ntop/2 + Ntop%2
        Ntop /= 2

    if not CellSizeAtTE:
        CellSizeAtTE = CellSizeAtLE

    # Locate the leading edge and split the airfoil
    TopSide, BottomSide = splitAirfoil(airfoil)
    reverse(BottomSide, in_place=True)

    # Normalize cell sizes if necessary
    # getApproximateChordAndThickness is more efficient here (after splitAirfoil) because ChordwiseIndicator is already known
    if relativeToChord:
        Chord, _ = getApproximateChordAndThickness(airfoil)
        CellSizeAtLE *= Chord
        CellSizeAtTE *= Chord
    
    # Apply discretization
    BottomSide = discretize(BottomSide, N=Nbot, Distribution=dict(kind='tanhTwoSides', FirstCellHeight=CellSizeAtTE, LastCellHeight=CellSizeAtLE))
    TopSide = discretize(TopSide, N=Ntop, Distribution=dict(kind='tanhTwoSides', FirstCellHeight=CellSizeAtLE, LastCellHeight=CellSizeAtTE))
    
    # Join two sides to return a single curve
    new_airfoil = joinSequentially([BottomSide, TopSide])

    return new_airfoil

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
    all_s1 = []
    for d in Distributions:
        N = int( d['N'] )
        if 'BreakPoint' in d:
            s1 = d['BreakPoint']
        elif 'BreakPoint(x)' in d:
            s1 = getAbscissaAtStation(curve, d['BreakPoint(x)'], coordinate='x')[0]
        elif 'BreakPoint(y)' in d:
            s1 = getAbscissaAtStation(curve, d['BreakPoint(y)'], coordinate='y')[0]
        elif 'BreakPoint(z)' in d:
            s1 = getAbscissaAtStation(curve, d['BreakPoint(z)'], coordinate='z')[0]
        else:
            raise ValueError(J.FAIL+'you must define a BreakPoint'+J.ENDC)

        all_s1 += [ s1 ]
        Lenght_segment = L*(s1-s0)
        Segment_Distri = linelaw(P1=(prevLength_segment,0,0),
            P2=(Lenght_segment+prevLength_segment,0,0), N=N, Distribution=d)
        Segments += [Segment_Distri]
        s0 = s1
        Ntot += N
        prevLength_segment += Lenght_segment

    if not np.all( np.diff( all_s1 ) >= 0 ):
        ERMS = 'abcissas are not monotonically increasing: %s. Check your breakpoints.'%str(all_s1)
        raise ValueError(J.FAIL+ERMS+J.ENDC)

    SegmentsQty = len(Segments)
    if SegmentsQty>1: Ntot -= SegmentsQty-1

    joined = Segments[0]
    for i in range(1,len(Segments)):
        joined = T.join(joined,Segments[i])


    if MappingLaw == 'Generator.map':
        try:
            return G.map(curve,D.getDistribution(joined))
        except:
            C.convertPyTree2File(Segments,'debug.cgns');exit()
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
            tolerance distance used for determining result of function

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
    if np.any(sqdist>tol**2): return False

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
        for fs in I.getNodesFromType(curve, 'FlowSolution_t'):
            fsCopy = I.copyTree(fs)
            JoinLine[2] += [fsCopy]
            for field in fsCopy[2]:
                if field[3] != 'DataArray_t': continue
                field[1] = np.linspace(field[1][0],field[1][-1],NPts4closingGap)
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
    Extrapolate a curve from one of its boundaries. Fields are **not** 
    extrapolated, but rather initialized to zero.

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
    curve = I.copyRef(curve)

    if opposedExtremum: T._reorder(curve,(-1,2,3))

    cX, cY, cZ = J.getxyz(curve)
    Tangent    = D.getTangent(curve)
    tX, tY, tZ = J.getxyz(Tangent)

    fields_names = C.getVarNames(curve, excludeXYZ=True)[0]

    if mode=='tangent':

        Points = [(cX[-1],cY[-1],cZ[-1])]
        Pt = Points[-1]
        Points += [(Pt[0]+tX[-1]*ExtrapDistance,
                    Pt[1]+tY[-1]*ExtrapDistance,
                    Pt[2]+tZ[-1]*ExtrapDistance)]
        Appendix = D.polyline(Points)
        if fields_names: J.invokeFields(Appendix, fields_names)
        
        ExtrapolatedCurve = T.join(curve, Appendix)


    if opposedExtremum: T._reorder(ExtrapolatedCurve,(-1,2,3))

    return ExtrapolatedCurve


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


def putAirfoilClockwiseOrientedAndStartingFromTrailingEdge( airfoil, tol=1e-10,
                                                    trailing_edge_region=0.95):
    '''
    This function transforms the input airfoil into clockwise-oriented and
    starting from trailing edge.

    .. attention:: airfoil must be placed on :math:`XY` plane

    Parameters
    ----------

        airfoil : zone
            structured curve of the airfoil

            .. note:: **airfoil** is modified

        tol : float
            distance tolerance (absolute) to determine if two points are
            coincident. Coincident points are removed.

        trailing_edge_region : float
            relative distance to determine the trailing and leading edge point 
            research region, following the chordwise direction. 
            This value is passed to :py:func:`findLeadingOrTrailingEdge`

    '''
    tem = '%g'%trailing_edge_region
    LE,_ = findLeadingOrTrailingEdge( airfoil, ChordwiseRegion='> '+tem)
    TE,_ = findLeadingOrTrailingEdge( airfoil, ChordwiseRegion='< -'+tem)

    if C.getMaxValue(LE,"CoordinateX") > C.getMaxValue(TE,"CoordinateX"):
        LE, TE = TE, LE
    I._rmNodesByType(TE,'FlowSolution_t')
    x, y, z = J.getxyz(TE)
    TE_xyz = ( x[0], y[0], z[0] )

    x, y = J.getxy( airfoil )
    FieldNames = C.getVarNames(airfoil, excludeXYZ=True, loc='nodes')[0]
    fields = J.getVars( airfoil, FieldNames )

    if not is2DCurveClockwiseOriented( airfoil ):
        T._reorder( airfoil, (-1,2,3))

    roll_index, _ = D.getNearestPointIndex(airfoil, TE_xyz)
    x, y = J.getxy( airfoil )
    fields = J.getVars( airfoil, FieldNames )
    x[:] = np.roll(x, -roll_index)
    y[:] = np.roll(y, -roll_index)
    for field in fields:
        field[:] = np.roll(field, -roll_index)

    # roll multiple point
    s = gets(airfoil)
    delta_s = np.diff(s * getLength(airfoil))
    i = np.argmin(delta_s)
    if delta_s[i] < tol:
        x[i:-1] = x[i+1:]
        y[i:-1] = y[i+1:]
        x[-1] = x[0]
        y[-1] = y[0]
        for field in fields:
            field[i:-1] = field[i+1:]
            field[-1] = field[0]

    # add curvilinear abscissa starting & ending at VIRTUAL TrailingEdge
    airfoil_with_TE = T.subzone( airfoil, (2,1,1), (-2,-1,-1) )
    I._rmNodesByType( airfoil_with_TE, 'FlowSolution_t' )
    airfoil_with_TE = concatenate( [TE, airfoil_with_TE, TE] )
    gets( airfoil_with_TE )

    J.migrateFields( airfoil_with_TE, airfoil, keepMigrationDataForReuse=False )
    s, = J.getVars(airfoil, ['s'])
    s[0] = 0
    s[-1] = 1


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
        try:
            if EltConn[0] == EltConn[-1]:
                # BAR is closed
                fT[0,:] = 0.5*((cxyz[1,:]-cxyz[0,:])+(cxyz[-1,:]-cxyz[-2,:]))
                fT[-1,:] = (cxyz[-1,:]-cxyz[-2,:])
            else:
                # BAR is open. That's always good news ;-)
                fT[0,:] = (cxyz[1,:]-cxyz[0,:])
                fT[-1,:] = (cxyz[-1,:]-cxyz[-2,:])
        except:
            C.convertPyTree2File(curve,'debug.cgns')
            sys.exit()
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

def pointwiseDistances(first_curves, second_curves):
    '''
    compute the pointwise (using same point index) distances between two set of
    curves of exact same number of points

    Parameters
    ----------

        first_curves : tree, base, zone, list of zone
            set of zones with exact same total point number as **second_curves**

        second_curves : tree, base, zone, list of zone
            set of zones with exact same total point number as **first_curves**

    Returns
    -------

        MinimumDistance : float
            minimum distance between points of same indexing

        MaximumDistance : float
            maximum distance between points of same indexing

        AverageDistance : float
            average distance between points of same indexing
    '''
    curves1 = I.copyRef( I.getZones( first_curves ) )
    curves2 = I.copyRef( I.getZones( second_curves ) )
    I._rmNodesByType(curves1+curves2, 'FlowSolution_t')
    curve1 = concatenate( curves1 )
    curve2 = concatenate( curves2 )

    x1, y1, z1 = J.getxyz( curve1 )
    x2, y2, z2 = J.getxyz( curve2 )

    x1 = np.ravel(x1, order='K')
    y1 = np.ravel(y1, order='K')
    z1 = np.ravel(z1, order='K')
    x2 = np.ravel(x2, order='K')
    y2 = np.ravel(y2, order='K')
    z2 = np.ravel(z2, order='K')

    NPts = len(x1)
    if NPts != len(x2):
        raise ValueError(J.FAIL+'both arguments must have same number of points'+J.ENDC)

    d = np.sqrt([(x2[i]-x1[i])**2 + (y2[i]-y1[i])**2 + (z2[i]-z1[i])**2 for i in range(NPts)])

    return np.min(d), np.max(d), np.mean(d)

def pointwiseVectors(curve1, curve2, normalize=True, reverse=False):
    '''
    compute the pointwise (using same point index) vectors between two curves
    of exact same number of points

    Parameters
    ----------

        curve1 : zone
            zone with same points as **curve2** and equvalent indexing

            .. note::
                **curve1** is modified, with new ``{sx}``, ``{sy}``, ``{sz}``

        curve2 : zone
            zone with same points as **curve1** and equivalent indexing

    Returns
    -------

        MinimumDistance : float
            minimum distance between points of same indexing

        MaximumDistance : float
            maximum distance between points of same indexing

        AverageDistance : float
            average distance between points of same indexing
    '''
    sx, sy, sz = J.invokeFields( curve1 , ['sx', 'sy', 'sz'] )
    x1, y1, z1 = J.getxyz( curve1 )
    x2, y2, z2 = J.getxyz( curve2 )

    x1 = np.ravel(x1, order='K')
    y1 = np.ravel(y1, order='K')
    z1 = np.ravel(z1, order='K')
    x2 = np.ravel(x2, order='K')
    y2 = np.ravel(y2, order='K')
    z2 = np.ravel(z2, order='K')
    sx = np.ravel(sx, order='K')
    sy = np.ravel(sy, order='K')
    sz = np.ravel(sz, order='K')

    NPts = len(x1)
    if NPts != len(x2):
        raise ValueError(J.FAIL+'zones must have same number of points'+J.ENDC)

    sign = -1 if reverse else 1

    sx[:] = sign * ( x2-x1 )
    sy[:] = sign * ( y2-y1 )
    sz[:] = sign * ( z2-z1 )

    if normalize: C._normalize(curve1, ['sx', 'sy', 'sz'])


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
    AirfoilCurveForOBB = discretize(AirfoilCurve, MappingLaw='interp1d_linear')
    # AirfoilCurveForOBB = G.map(AirfoilCurveForOBB,
    #                      D.line((0,0,0),(1,0,0),C.getNPts(AirfoilCurveForOBB)))
    AirfoilCurveForOBB[0] = 'AirfoilCurveForOBB'
    OBB = G.BB(AirfoilCurveForOBB, method='OBB')
    OBB[0]='OBB'

    Barycenter = G.barycenter(OBB)
    Barycenter = np.array(Barycenter)

    DecreasingDirections = getDecreasingDirections(OBB)
    x,y,z = J.getxyz(AirfoilCurve)
    xyz = np.vstack((x,y,z)).T
    def _invokeIndicator(Direction, Name):

        ApproximateLength = ( Direction.dot(Direction) ) ** 0.5

        Direction /= ApproximateLength

        #       ( x - B ) dot u
        #         -   -       -
        Indicator, = J.invokeFields(AirfoilCurve, [Name])
        Indicator[:] = (xyz-Barycenter).dot(Direction)
        Indicator /= Indicator.max()
        Indicator -= Indicator.min()
        Indicator /= Indicator.max()
        Indicator *= 2
        Indicator -= 1


        return ApproximateLength

    ApproximateChord     = _invokeIndicator( DecreasingDirections[0],
                                             'ChordwiseIndicator'     )
    ApproximateThickness = _invokeIndicator( DecreasingDirections[1],
                                             'ThickwiseIndicator'     )
    if np.isnan(ApproximateChord ):
        J.save([AirfoilCurve,AirfoilCurveForOBB,OBB], 'debug.cgns' )

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
    ci, ti = J.getVars(AirfoilCurve,['ChordwiseIndicator','ThickwiseIndicator'])
    SelectedRegion = P.selectCells(AirfoilCurve,'{ChordwiseIndicator}'+
                                                  ChordwiseRegion)

    x = J.getx(SelectedRegion)
    if len(x) == 0:
        ci, = J.getVars(AirfoilCurve,['ChordwiseIndicator'])
        ERRMSG = ('requested chordwise region (%s) was outside the '
                  'available boundaries (%g,%g).'
                  'Please decrase the value of ChordwiseRegion.')%(ChordwiseRegion,ci.min(),ci.max())
        raise ValueError(ERRMSG)
    SelectedRegion = C.convertBAR2Struct( SelectedRegion )
    removeMultiplePoints(SelectedRegion)

    # rediscretize the selected region
    region_npts = 101
    delta_s = D.getLength( SelectedRegion ) / float( region_npts )
    SmoothParts = T.splitCurvatureAngle( SelectedRegion, 30.)
    NewSmoothParts = []
    for sp in SmoothParts:
        Length_subpart = D.getLength( sp )
        npts_subpart = int(Length_subpart / delta_s)
        new_subpart = discretize(sp, N=np.maximum(npts_subpart,3))
        NewSmoothParts.append( new_subpart )

    try:
        SelectedRegion = T.join( NewSmoothParts )
    except BaseException as e:
        J.save(NewSmoothParts,'debug.cgns')
        raise ValueError(J.FAIL+'could not join NewSmoothParts, check debug.cgns'+J.ENDC) from e

    D._getCurvatureRadius( SelectedRegion )
    aux_s = gets( SelectedRegion )
    radius, = J.getVars( SelectedRegion, ['radius'] )
    x, y, z =  J.getxyz( SelectedRegion )

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

        # TODO notify bug Cassiopee: addPointInDistribution modifies geometry !
        # AirfoilCurve = G.addPointInDistribution(AirfoilCurve, len(x))
        # WORKAROUND:
        new_pt = D.point(extremum(AirfoilCurve,True))
        AirfoilCurve = concatenate([AirfoilCurve,new_pt])

        x, y, z = J.getxyz(AirfoilCurve)

        x[-1] = x[0]
        y[-1] = y[0]
        z[-1] = z[0]

    return AirfoilCurve




def splitAirfoil(AirfoilCurve, FirstEdgeSearchPortion = 0.95,
        SecondEdgeSearchPortion = -0.95, RelativeRadiusTolerance = 1e-2,
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

        DistanceCriterionTolerance : float
            Small value used as threshold in order to determine if the distance
            criterion can be used in order to identify top and bottom sides of 
            the airfoil. If the absolute maximum distance between top and bottom
            sides with respect to the camber curve is lower than **DistanceCriterionTolerance**,
            then criterion ``'distance'`` is not suitable, a warning is raised, 
            and next geometrical criterion is employed.

        FieldCriterion : str
            Coordinate, or field name contained as FlowSolution, used to indicate
            the position of the top side of the airfoil, when the employed criterion 
            is ``'field'``

        SideChoiceCriteriaPriorities : :py:class:`list` of :py:class:`str`
            List of geometrical criteria used to identify the top and bottom
            sides of the airfoil. Criterion ``'distance'`` consists in identifying
            as top side the airfoil subpart that presents the maximum distance
            with respect to the camber line. This is not suitable for symmetrical
            or pseudo-symmetrical airfoils. Criterion ``'field'`` uses **FieldCriterion**
            as indicator of the top side (subpart having maximum value of **FieldCriterion**
            is identified as being the top side).

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
    I._rmNodesByType([LE, TE], 'FlowSolution_t')
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
                        tol=1e-7 )


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
                msg = ('sides are too close to camber for using "distance"'
                 f' criterion for identifying top/bottom sides using'
                 f'FieldCriterion {FieldCriterion}. Skipping to next criterion.\n'
                 f'or you may retry using different FieldCriterion')
                print(J.WARN+msg+J.ENDC)
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
        print(J.WARN+'Camber line not converged with residual %g'%residual+J.ENDC)

    CamberCoarse = I.copyTree(CamberPolyline)
    CamberCoarse[0] = 'coarse'
    CamberPolyline = discretize(CamberCoarse, N=FinalDistribution['N'],
                             Distribution=FinalDistribution, MappingLaw='akima')
    NCtrlPts = FinalDistribution['N']
    gets(TopSide)
    TopSide = discretize(TopSide, N=FinalDistribution['N'],
                             Distribution=FinalDistribution, MappingLaw='akima')
    gets(BottomSide)
    BottomSide = discretize(BottomSide, N=FinalDistribution['N'],
                             Distribution=FinalDistribution, MappingLaw='akima')

    ptrs = prepareCamberLine(CamberPolyline)
    e, = J.invokeFields(CamberPolyline,['RelativeThickness'])
    CamberPolylineFields, CamberPolylineTangents, CamberPolylineCoords = ptrs
    updateTangents()
    updatePerpendicularSquaredDistance(    TopSide, 'Top' )
    updatePerpendicularSquaredDistance( BottomSide, 'Bottom' )
    e[:] = (np.sqrt(CamberPolylineFields['SquaredDistanceTop']) + \
            np.sqrt(CamberPolylineFields['SquaredDistanceBottom'])) / Chord
    CamberFine = CamberPolyline
   


    # NOTE this could produce awful results dependending on the slope value...
    # CamberFine = discretize(CamberPolyline, N=FinalDistribution['N'],
    #                          Distribution=FinalDistribution, MappingLaw='akima')
    # fields_names = C.getVarNames(CamberFine, excludeXYZ=True)[0]
    # # interpolate coarse fields into fine fields
    # fields_coarse = J.getVars(CamberPolyline, fields_names)
    # fields_fine = J.getVars(CamberFine, fields_names)
    # s_coarse = gets(CamberPolyline)
    # s_fine = gets(CamberFine)
    # slope = 5
    # for field_coarse, field_fine in zip(fields_coarse, fields_fine):
    #     field_fine[:] = J.interpolate__(s_fine,s_coarse,field_coarse, Law='cubic',
    #                                     bc_type=((1,slope),'not-a-knot'))

    CamberFine[0] = CamberPolyline[0]+'.camber'
    gets(CamberFine)
    gets(CamberCoarse)

    # import matplotlib.pyplot as plt
    # s, e = J.getVars(CamberCoarse,['s','RelativeThickness'])
    # plt.plot(s,e,'o',mfc='None')
    # s, e = J.getVars(CamberFine,['s','RelativeThickness'])
    # plt.plot(s,e,'.-')
    # plt.show()

    
    
    return CamberFine


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
    TopSide, _ = splitAirfoil(AirfoilCurve, **splitAirfoilOptions)

    buildCamberOptions.setdefault('splitAirfoilOptions', splitAirfoilOptions)
    CamberLine = buildCamber(AirfoilCurve, **buildCamberOptions)
    CamberLineX, CamberLineY, CamberLineZ = J.getxyz(CamberLine)
    RelativeThickness, = J.getVars(CamberLine, ['RelativeThickness'])

    LeadingEdge  = np.array([CamberLineX[0],
                             CamberLineY[0],
                             CamberLineZ[0]], dtype=np.float64)
    AirfoilProperties['LeadingEdge'] = LeadingEdge

    TrailingEdge = np.array([CamberLineX[-1],
                             CamberLineY[-1],
                             CamberLineZ[-1]], dtype=np.float64)
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
                                   CamberLineZ[iMaxThickness]],dtype=np.float64)
    MaxThicknessLocation = (MaxThicknessCoords-LeadingEdge).dot(ChordDirection)
    MaxThicknessLocation /= Chord
    AirfoilProperties['MaxThicknessRelativeLocation'] = MaxThicknessLocation

    iMaxCamber = np.argmax(CamberValues)
    AirfoilProperties['MaxCamber'] = CamberValues[iMaxCamber]
    AirfoilProperties['MaxRelativeCamber'] = CamberValues[iMaxCamber] / Chord
    MaxCamberCoords = np.array([CamberLineX[iMaxCamber],
                                CamberLineY[iMaxCamber],
                                CamberLineZ[iMaxCamber]],dtype=np.float64)
    MaxCamberLocation = (MaxCamberCoords-LeadingEdge).dot(ChordDirection)
    MaxCamberLocation /= Chord
    AirfoilProperties['MaxCamberRelativeLocation'] = MaxCamberLocation

    iMinCamber = np.argmin(CamberValues)
    AirfoilProperties['MinCamber'] = CamberValues[iMinCamber]
    AirfoilProperties['MinRelativeCamber'] = CamberValues[iMinCamber] / Chord
    MinCamberCoords = np.array([CamberLineX[iMinCamber],
                                CamberLineY[iMinCamber],
                                CamberLineZ[iMinCamber]],dtype=np.float64)
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
    a = np.array(LinePassingPoint,dtype=np.float64)
    n = np.array(LineDirection,dtype=np.float64)
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
                  MinThickness=None, MinRelativeThickness=None, 
                  TrailingEdgeRadius=None, TrailingEdgeRelativeRadius=None,
                  SmoothingDistance=0.1,
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
        
        MinThickness : float
            if provided, sets the minimum thickness in absolute value [m]. 
            It is used only after the position of maximum thickness, to be applied near the trailing edge only.

        MinRelativeThickness : float
            if provided, sets the minimum thickness in relative value
            (normalized by the chord length)

        TrailingEdgeRadius : float 
            if provided, sets the radius to force a circular trailing edge shape in absolute value [m]. 

        TrailingEdgeRelativeRadius : float
            if provided, sets the radius to force a circular trailing edge shape in relative value
            (normalized by the chord length). 
            If MinRelativeThickness is not given, it will be taken equal to TrailingEdgeRelativeRadius.
        
        SmoothingDistance : float
            Smoothing distance (relative to the chord), used when minimum thickness is applied to smooth 
            the thickness law around the angular join. Default value is 0.1.

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

        NewAirfoilCamber : zone
            structured curve of the camber line of the new airfoil
    '''


    Airfoil = I.copyTree(AirfoilInput)

    AirfoilProperties, Camber = getAirfoilPropertiesAndCamber(AirfoilInput,
                                        buildCamberOptions=buildCamberOptions,
                                        splitAirfoilOptions=splitAirfoilOptions)
    InitialCamber = I.copyTree(Camber)

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
                    MinThickness=MinThickness, 
                    MinRelativeThickness=MinRelativeThickness, 
                    TrailingEdgeRadius=TrailingEdgeRadius, 
                    TrailingEdgeRelativeRadius=TrailingEdgeRelativeRadius,
                    SmoothingDistance=SmoothingDistance,
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

    x0,y0,z0 = J.getxyz(InitialCamber)
    xf,yf,zf = J.getxyz(Camber)

    T._translate(Camber,(x0[0]-xf[0], y0[0]-yf[0], z0[0]-zf[0]))

    NewAirfoil = buildAirfoilFromCamberLine(Camber, NormalDirection)
    NewAirfoil[0] = AirfoilInput[0]+'.mod'
    Camber[0] = NewAirfoil[0]+'.camber'
    J.set(NewAirfoil, '.AirfoilProperties', **AirfoilProperties)

    return NewAirfoil, Camber

def modifyThicknessOfCamberLine(CamberCurve, NormalDirection, MaxThickness=None,
                MaxRelativeThickness=None, MaxThicknessRelativeLocation=None,
                MinThickness=None, MinRelativeThickness=None, 
                TrailingEdgeRadius=None, TrailingEdgeRelativeRadius=None,
                SmoothingDistance=0.1,
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
        
        MinThickness : float
            if provided, sets the minimum thickness in absolute value [m]. 
            It is used only after the position of maximum thickness, to be applied near the trailing edge only.

        MinRelativeThickness : float
            if provided, sets the minimum thickness in relative value
            (normalized by the chord length)

        TrailingEdgeRadius : float 
            if provided, sets the radius to force a circular trailing edge shape in absolute value [m]. 

        TrailingEdgeRelativeRadius : float
            if provided, sets the radius to force a circular trailing edge shape in relative value
            (normalized by the chord length). 
            If MinRelativeThickness is not given, it will be taken equal to TrailingEdgeRelativeRadius.
        
        SmoothingDistance : float
            Smoothing distance (relative to the chord), used when minimum thickness is applied to smooth 
            the thickness law around the angular join. Default value is 0.1.

        InterpolationLaw : str
            interpolation law to be applied for the modification of thickness

            .. note:: **InterpolationLaw** is the parameter **Law** of function
                :py:func:`MOLA.InternalShortcuts.interpolate__`
    '''

    needModification = MaxThickness or MaxRelativeThickness or MaxThicknessRelativeLocation \
        or MinThickness or MinRelativeThickness or TrailingEdgeRadius or TrailingEdgeRelativeRadius
    if not needModification: return CamberCurve

    if MaxThickness and MaxRelativeThickness:
        raise AttributeError('Cannot specify both relative and absolute thickness')
    if MinThickness and MinRelativeThickness:
        raise AttributeError('Cannot specify both relative and absolute thickness')
    if TrailingEdgeRadius and TrailingEdgeRelativeRadius:
        raise AttributeError('Cannot specify both relative and absolute TE radii')

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
    s = gets(CamberLine)


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

        AuxS = gets(AuxCamberLine)
        RelativeThickness[:] = J.interpolate__(s, AuxS, AuxRelativeThickness,
                                               Law=InterpolationLaw)

    if MinThickness:
        MinRelativeThickness = MinThickness / Chord
    
    if TrailingEdgeRadius:
        TrailingEdgeRelativeRadius = TrailingEdgeRadius / Chord
    
    if not MinRelativeThickness and TrailingEdgeRelativeRadius:
        MinRelativeThickness = TrailingEdgeRelativeRadius

    if MinRelativeThickness:

        import scipy

        if not TrailingEdgeRelativeRadius:
            raise AttributeError('Must provide a TrailingEdgeRelativeRadius when giving a min thickness')
        
        # Above this index, thickness is modified (lower bound) 
        ModifIndex = ArgMaxThickness + np.argmin(np.abs(RelativeThickness[ArgMaxThickness:]-MinRelativeThickness))
        RelativeThickness[ModifIndex:] = MinRelativeThickness

        # Modify thickness near the TE to get a circular TE of radius TrailingEdgeRelativeRadius
        LocalAbscissa = s-(np.amax(s)-TrailingEdgeRelativeRadius) # zero at TE minus the chosen radius
        TEindices = np.where(LocalAbscissa>=0)[0]
        RelativeThickness[TEindices] = np.sqrt(np.maximum(0, TrailingEdgeRelativeRadius**2 - LocalAbscissa[TEindices]**2))

        # Apply a smoothing on thickness law on the distance sigma around the angular point
        SmoothingDistance = min(SmoothingDistance, 1-s[ModifIndex] - TrailingEdgeRelativeRadius)
        SmoothingIndices = np.where(np.abs(s-s[ModifIndex]) < SmoothingDistance)[0] # indices where smooting is performed
        # Compute the derivative at the distance sigma before the angular point
        # Remark : the derivative at the distance sigma after the angular point is set to zero
        deriv1 = (RelativeThickness[SmoothingIndices[0]+1]-RelativeThickness[SmoothingIndices[0]-1]) / (s[SmoothingIndices[0]+1] - s[SmoothingIndices[0]-1])
        # Smoothing : interpolation with a cubic spline, imposing the first derivative for both bounds
        smoothed_thickness = scipy.interpolate.CubicSpline([s[SmoothingIndices][0], s[SmoothingIndices][-1]], 
                                                           [RelativeThickness[SmoothingIndices][0], RelativeThickness[SmoothingIndices][-1]],
                                                           bc_type=((1, deriv1), (1, 0.))
                                                           )
        # Aplly the smoothing
        RelativeThickness[SmoothingIndices] = smoothed_thickness(s[SmoothingIndices])
        

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


def trimCurveAlongDirection(curve, direction, cut_point_1, cut_point_2):
    '''
    Trim a curve along a direction using two cut points (resulting curve is
    placed between the cut points, along the requested direction).

    .. note::
        this will be replaced with strategy shown in `10357 <https://elsa.onera.fr/issues/10357>`_

    Parameters
    ----------

        curve : zone
            the curve to be trimmed

        direction : :py:class:`list` or numpy array of 3-:py:class:`float`
            unitary vector of the direction used to trim

        cut_point_1 : :py:class:`list` or numpy array of 3-:py:class:`float`
            coordinates of the first point defining the trim location along
            **direction**

        cut_point_2 : :py:class:`list` or numpy array of 3-:py:class:`float`
            coordinates of the second point defining the trim location along
            **direction**

    Returns
    -------

        trimmed_curve : zone
            trimmed structured curve
    '''

    # TODO replace with strategy: https://elsa.onera.fr/issues/10357

    curve = I.copyRef(curve)
    x,y,z = J.getxyz(curve)
    FieldToSlice, = J.invokeFields(curve,['FieldToSlice'])
    x = x.ravel(order='F')
    y = y.ravel(order='F')
    z = z.ravel(order='F')
    xyz = np.vstack((x,y,z)).T
    distance1 = ( xyz - cut_point_1 ).dot( direction )
    distance2 = ( xyz - cut_point_2 ).dot( direction )
    PointsToKeep = distance1 * distance2 < 0
    FieldToSlice[PointsToKeep] = 1
    trimmed_element = P.selectCells(curve,'{FieldToSlice}>0.1',strict=1)
    number_of_subparts = len(I.getZones(T.splitConnexity(trimmed_element)))
    if number_of_subparts != 1:
        C.convertPyTree2File(curve,'debug.cgns')
        raise ValueError(f'could not trim along direction, since multiple subparts were obtained ({number_of_subparts}).')

    trimmed_curve = C.convertBAR2Struct(trimmed_element)
    I._rmNodesByType(trimmed_curve,'FlowSolution_t')

    # add first point
    n = direction
    Pt = cut_point_1
    PlaneCoefs = n[0],n[1],n[2],-n.dot(Pt)
    C._initVars(curve,'Slice=%0.12g*{CoordinateX}+%0.12g*{CoordinateY}+%0.12g*{CoordinateZ}+%0.12g'%PlaneCoefs)
    zones = P.isoSurfMC(curve, 'Slice', 0.0)
    if len(zones) > 0:
        pt1 = I.getZones(zones)[0]
        x,y,z = J.getxyz(pt1)
        pt1 = D.point((x[0],y[0],z[0]))
        trimmed_curve = concatenate([pt1,trimmed_curve])

    # add second point
    n = direction
    Pt = cut_point_2
    PlaneCoefs = n[0],n[1],n[2],-n.dot(Pt)
    C._initVars(curve,'Slice=%0.12g*{CoordinateX}+%0.12g*{CoordinateY}+%0.12g*{CoordinateZ}+%0.12g'%PlaneCoefs)
    zones = P.isoSurfMC(curve, 'Slice', 0.0)
    if len(zones) > 0:
        pt2 = I.getZones(zones)[0]
        x,y,z = J.getxyz(pt2)
        pt2 = D.point((x[0],y[0],z[0]))
        trimmed_curve = concatenate([trimmed_curve,pt2])

    return trimmed_curve


def getNearestIntersectingPoint(zone1, zone2):
    '''
    Return the coordinates of a point belonging to **zone2** such that
    its distance is minimum with respect to any point of **zone1**

    .. note::
        this is an 0th-order method. Will be replaced by `10356 <https://elsa.onera.fr/issues/10356>`_

    Parameters
    ----------

        zone1 : zone
            points where distance are computed

        zone2 : zone
            points from which minimum distance is selected

    Returns
    -------

        point : 3-:py:class:`float` numpy.array
            coordinates of the nearest intersecting point belonging to **zone2**
    '''
    xN,yN,zN = J.getxyz(zone1)
    x = np.ravel(xN,order='K')
    y = np.ravel(yN,order='K')
    z = np.ravel(zN,order='K')

    AllPoints = [(x[i], y[i], z[i]) for i in range(len(x))]
    res = D.getNearestPointIndex(zone2, AllPoints)
    xT, yT, zT = J.getxyz(zone2)
    x2 = np.ravel(xT,order='K')
    y2 = np.ravel(yT,order='K')
    z2 = np.ravel(zT,order='K')
    nearest_indices = [res[i][0] for i in range(len(res))]
    squared_distances = [res[i][1] for i in range(len(res))]
    nearest_index = nearest_indices[ np.argmin(squared_distances) ]

    return np.array([x2[nearest_index], y2[nearest_index], z2[nearest_index]])


def intersection(curves):
    '''
    Computes the intersecting points of a set of curves, including
    possibly self-intersecting points.

    .. note::
        Will be replaced by `10356 <https://elsa.onera.fr/issues/10356>`_

    Parameters
    ----------

        curves : py:class:`list` of zone
            list of structured curves from which intersections are being computed

    Returns
    ----------

        points : py:class:`list` of zone
            list of points (each point is a zone) of the intersections.
    '''
    # TODO to be replaced by `10356 <https://elsa.onera.fr/issues/10356>`_
    concatenated = concatenate(curves)
    InitialNPts = C.getNPts(concatenated)
    bar = C.convertArray2Tetra(concatenated)
    conformed = XOR.conformUnstr(bar, left_or_right=0, itermax=1, tol=1e-5)
    x,y,z = J.getxyz(conformed)
    FinalNPts = len(x)
    IntersectingPoints = []
    for i in range(InitialNPts,FinalNPts):
        IntersectingPoints.append( D.point((x[i],y[i],z[i])) )


    return IntersectingPoints

def writeAirfoilInSeligFormat(airfoil, filename='foil.dat'):
    '''
    write an airfoil curve into a  .dat *(or .txt)* SELIG file format

    Parameters
    ----------

        airfoil : zone
            a curve of the airfoil to be written

        filename : str
            the name of the file to produce
    '''
    foil = I.copyRef(airfoil)
    if is2DCurveClockwiseOriented(foil): T._reorder(foil,(-1,2,3))
    X, Y = J.getxy(foil)
    with open(filename,'w') as f:
        f.write(foil[0]+'\n')
        for x, y in zip(X,Y):
            f.write(' %0.6f   %0.6f\n'%(x,y))


def tangentExtremum(curve, opposite_extremum=False):
    '''
    get the unitary vector direction (tangent) at the extremum of a structured
    curve

    Parameters
    ----------

        curve : zone
            structured curve

        opposite_extremum : bool
            if :py:obj:`False`, compute the tangent at first index :math:`(i=0)`.
            If :py:obj:`True`, compute the tangent at last index :math:`(i=-1)`.

    Returns
    -------

        v : numpy.array of 3 float
            unitary vector of the direction of the extremum following index
            direction
    '''
    x,y,z = J.getxyz(curve)
    if opposite_extremum:
        v = np.array([x[-1]-x[-2], y[-1]-y[-2], z[-1]-z[-2]])
    else:
        v = np.array([x[1]-x[0], y[1]-y[0], z[1]-z[0]])
    v /= np.sqrt(v.dot(v))
    return v

def extremum(curve, opposite_extremum=False):
    '''
    get the coordinates of the extremum point of a curve

    Parameters
    ----------

        curve : zone
            structured curve

        opposite_extremum : bool
            if :py:obj:`False`, get the extremum at first index :math:`(i=0)`.
            If :py:obj:`True`, get the extremum at last index :math:`(i=-1)`.

    Returns
    -------

        pt : :py:class:`numpy.ndarray` of 3-:py:class:`float`
            coordinates :math:`(x,y,z)`
    '''
    if opposite_extremum:
        return point(curve,-1)
    else:
        return point(curve)

def extremumAsZone(curve, opposite_extremum=False):
    '''
    get the extremum point of a curve as a Zone_t

    Parameters
    ----------

        curve : zone
            structured curve

        opposite_extremum : bool
            if :py:obj:`False`, get the extremum at first index :math:`(i=0)`.
            If :py:obj:`True`, get the extremum at last index :math:`(i=-1)`.

    Returns
    -------

        pt : Zone
            extremum
    '''
    if opposite_extremum:
        pt = D.point(point(curve,-1))
        pt[0] = 'end'
    else:
        pt = D.point(point(curve,0))
        pt[0] = 'start'
    return pt


def extremumAsPointer(curve, opposite_extremum=False):
    '''
    get the extremum point of a curve as a list of 3 pointers (sharing data with
    curve)

    Parameters
    ----------

        curve : zone
            structured curve

        opposite_extremum : bool
            if :py:obj:`False`, get the extremum at first index :math:`(i=0)`.
            If :py:obj:`True`, get the extremum at last index :math:`(i=-1)`.

    Returns
    -------

        pt : :py:class:`list` of 3 :py:class:`numpy.ndarray` of size 1
            pointers of extremum of curve
    '''
    x, y, z = J.getxyz(curve)
    if opposite_extremum:
        return [x[-1:], y[-1:], z[-1:]]
    else:
        return [x[:1], y[:1], z[:1]]




def extrema(curve):
    '''
    get the coordinates of both the extremum points of a curve

    Parameters
    ----------

        curve : zone
            structured curve


    Returns
    -------

        pt : :py:class:`list` of 2 :py:class:`numpy.ndarray` of 3-:py:class:`float`
            the two coordinates of extrema :math:`(x,y,z)`
    '''
    return point(curve), point(curve,-1)

def extremaAsZones(curve):
    '''
    get the two extremum points of a curve as Zone_t

    Parameters
    ----------

        curve : zone
            structured curve

    Returns
    -------

        pt : :py:class:`list` of 2 Zone
            the two points of extrema
    '''
    return extremumAsZone(curve), extremumAsZone(curve,True)

def extremaAsPointers(curve):
    '''
    get the two extremum points of a curve as a list of pointers

    Parameters
    ----------

        curve : zone
            structured curve

    Returns
    -------

        pt : :py:class:`list` of result of :py:func:`extremumAsPointer`
            pointers of two points of extrema
    '''
    return extremumAsPointer(curve), extremumAsPointer(curve,True)

def gluePointers(pointer1, pointer2, mode='1-towards-2'):
    '''
    Given two points as pointers glue them together to make them spatially 
    coincident

    Parameters
    ----------

        pointer1 : :py:class:`list` of 3 :py:class:`numpy.ndarray` of size 1
            first point pointer as got using :py:func:`extremumAsPointer`

        pointer2 : :py:class:`list` of 3 :py:class:`numpy.ndarray` of size 1
            second point pointer as got using :py:func:`extremumAsPointer`

        mode : str
            one of:

            * ``'1-towards-2'``
                put coordinates of point1 into coordinates of point2

            * ``'2-towards-1'``
                put coordinates of point2 into coordinates of point1

            * ``'average'`` or ``'mean'``
                put both points into their equidistant point

    '''
    if mode == '1-towards-2':
        for i in range(3): pointer1[i][0] = pointer2[i][0]
    elif mode == '2-towards-1':
        for i in range(3): pointer2[i][0] = pointer1[i][0]
    elif mode in ['average','mean']:
        for i in range(3):
            avg = 0.5 * (pointer1[i][0] + pointer2[i][0])
            pointer1[i][0] = avg
            pointer2[i][0] = avg
    else:
        raise NotImplementedError(mode)


def glueCurvesAtExtrema(curve1, curve2, curve1_extremum='start',
        curve2_extremum='start', mode='1-towards-2'):
    '''
    Glue two curves at their extrema

    Parameters
    ----------

        curve1 : zone
            first structured curve

        curve2 : zone
            second structured curve

        curve1_extremum : str
            selects the extremum of **curve1** to glue. It may be ``'start'`` or ``'end'``.

        curve2_extremum : str
            selects the extremum of **curve2** to glue. It may be ``'start'`` or ``'end'``.

        mode : str
            mode of glue. See doc of :py:func:`gluePointers`

    '''
    if curve1_extremum == 'start':
        pointer1 = extremumAsPointer(curve1)
    elif curve1_extremum == 'end':
        pointer1 = extremumAsPointer(curve1, True)
    else:
        raise NotImplementedError(f'curve1_extremum="{curve1_extremum}" not implemented, must be "start" or "end"')

    if curve2_extremum == 'start':
        pointer2 = extremumAsPointer(curve2)
    elif curve2_extremum == 'end':
        pointer2 = extremumAsPointer(curve2, True)
    else:
        raise NotImplementedError(f'curve2_extremum="{curve2_extremum}" not implemented, must be "start" or "end"')

    gluePointers(pointer1, pointer2, mode=mode)


def reorderCurvesSequentially(curves):
    '''
    reorder the indexing of a set of structured curves so that they yield a
    coherent ordering for a sequential geometrical concatenation.

    Parameters
    ----------

        curves : :py:class:`list` of zones
            list of structured curves to be reordered

            .. warning::
                **curves** must contain at least 2 items

            .. important::
                the ordering of the curves is determined by the existing ordering
                of the first provided curve (first item of **curves**)

    Returns
    -------

        reordered_curves : :py:class:`list` of zones
            like the input, but the curves ordering is adapted if required
    '''
    ordered_curves = [ curves[0] ]
    not_ordered_curves = list(curves[1:])

    must_reverse = [True, False, False, True]

    while not_ordered_curves:

        last_ordered_curve = ordered_curves[-1]

        start0 = extremum(last_ordered_curve)
        end0 = extremum(last_ordered_curve,True)

        distances = []
        nearest_extrema = []

        for not_ordered_curve in not_ordered_curves:
            start = extremum(not_ordered_curve)
            end = extremum(not_ordered_curve, True)
            extrema_dist = np.array([distance(start0, start),
                                     distance(start0, end),
                                     distance(end0, start),
                                     distance(end0, end)])
            nearest_extrema += [ np.argmin( extrema_dist ) ]
            distances += [ extrema_dist[nearest_extrema[-1]] ]

        nearest_curve_index = np.argmin( distances )
        nearest_curve = not_ordered_curves[ nearest_curve_index ]

        if must_reverse[ nearest_extrema[nearest_curve_index] ]:
            T._reorder(nearest_curve,(-1,2,3))

        ordered_curves += [nearest_curve]

        del not_ordered_curves[ nearest_curve_index ]

    return ordered_curves

def sortCurvesSequentially(curves):
    '''
    sort a list of structured curves so that the order of the new list
    minimizes the distance between the ending and the starting points of each
    curve. This is useful for concatenation.

    Parameters
    ----------

        curves : :py:class:`list` of zones
            list of structured curves to be sorted

            .. warning::
                **curves** must contain at least 2 items

            .. important::
                the sorting of the curves starts from the first provided curve
                (first item of **curves**)

    Returns
    -------

        sorted_curves : :py:class:`list` of zones
            like the input, but the curves order in the list is adapted if required
    '''
    sorted_curves = [ curves[0] ]
    not_sorted_curves = list(curves[1:])

    while not_sorted_curves:

        last_sorted_curve = sorted_curves[-1]

        end0 = extremum(last_sorted_curve,True)

        distances = []
        for not_sorted_curve in not_sorted_curves:
            start = extremum(not_sorted_curve)
            distances += [distance(end0, start)]

        nearest_curve_index = np.argmin( distances )
        nearest_curve = not_sorted_curves[ nearest_curve_index ]

        sorted_curves += [nearest_curve]

        del not_sorted_curves[ nearest_curve_index ]

    return sorted_curves


def reorderAndSortCurvesSequentially(curves):
    '''
    Literally, combines :py:func:`reorderCurvesSequentially` and
    :py:func:`sortCurvesSequentially`
    '''
    return sortCurvesSequentially(reorderCurvesSequentially(curves))


def splitInnerContourFromOutterBoundariesTopology(boundaries, inner_contour,
        forced_split_index=None):
    '''
    From a closed boundary formed by a set of structured curves, conveniently
    splits another structured curve such that each boundary has a corresponding
    inner contour subpart with equal number of segments. This is useful as an
    intermediate step for H-shape meshing topologies.

    Parameters
    ----------

        boundaries : :py:class:`list` of zones
            List of structured curves defining the outter boundaries.

            .. warning::
                total number of segments of **boundaries** and **inner_contour**
                must be identical

        inner_contour : zone
            structured curve to be split

        forced_split_index : int
            forces split of **inner_contour** at given index

    Return
    ------

        adapted_boundaries : :py:class:`list` of 4 zones
            like **boundaries** but sequentially reordered and sorted

        inner_contour_split : :py:class:`list` of 4 zones
            **inner_contour** split in 4 parts with identical number of
            segments between each subpart and the boundaries, yielding same
            index ordering and position in list

        reference_split_index : int
            first index used for splitting
    '''
    N_bnds = len(boundaries)

    inner_contour_NPts = C.getNPts(inner_contour)
    N_segments_inner = inner_contour_NPts - 1
    N_segments_boundary = 0
    for b in boundaries:
        N_segments_boundary += C.getNPts(b)-1

    if N_segments_boundary != N_segments_inner:
        raise ValueError(J.FAIL+'total number of segments of boundaries (%d) must be the same as inner_contour (%d)'%(N_segments_boundary,N_segments_inner)+J.ENDC)

    def _getSplitIndicesAndDistances(i_corner):
        if (index > 0) and (index < inner_contour_NPts-2):
            partA = T.subzone(inner_contour,(1,1,1),(index+1,1,1))
            partB = T.subzone(inner_contour,(index+2,1,1),(-1,-1,-1))
            aux_inner_contour = joinSequentially([partB, partA],reorder=True,sort=True)
        else:
            aux_inner_contour = I.copyTree(inner_contour)

        aux_inner_contour = closeStructCurve(aux_inner_contour)
        x,y,z = J.getxyz(aux_inner_contour)
        distances = np.zeros(N_bnds,dtype=float)
        split_indices = np.zeros(N_bnds,dtype=int)
        split_index = 0

        for j in range(N_bnds):
            bnd_index = i_corner+j
            if bnd_index > N_bnds-1: bnd_index -= N_bnds
            if j>0: split_index += C.getNPts(boundaries[bnd_index-1]) - 1
            consequence_point = np.array([x[split_index],y[split_index],z[split_index]])
            distances[bnd_index] = distance(corners[bnd_index], consequence_point)
            split_indices[bnd_index] = split_index
            if bnd_index == 0: first_consequence_point = consequence_point

        return split_indices, distances, first_consequence_point, aux_inner_contour

    boundaries = I.copyRef(boundaries)
    inner_contour = I.copyRef(inner_contour)
    I._rmNodesByType(boundaries+[inner_contour],'FlowSolution_t')

    boundaries = reorderAndSortCurvesSequentially(boundaries)
    I._correctPyTree(boundaries, level=3)
    corners = [extremum(b) for b in boundaries]

    inner_contour = I.copyTree(inner_contour)
    index, sqrddist = D.getNearestPointIndex(inner_contour,tuple(corners[0]))
    xi, yi, zi = J.getxyz(inner_contour)
    if index == 0: u=np.array([xi[1]-xi[0],yi[1]-yi[0],zi[1]-zi[0]])
    else: u=np.array([xi[index]-xi[index-1],yi[index]-yi[index-1],zi[index]-zi[index-1]])
    u /= np.sqrt(u.dot(u))
    v = 0.5 * ( tangentExtremum(boundaries[0]) + tangentExtremum(boundaries[0],False))
    if u.dot(v) < 0: T._reorder(inner_contour,(-1,2,3))

    if forced_split_index is None:
        inner_contour_NPts = C.getNPts(inner_contour)
        all_distances = []
        first_points = []
        for i_corner, corner in enumerate(corners):
            index, sqrddist = D.getNearestPointIndex(inner_contour,tuple(corner))
            _, distances, first_point, _ = _getSplitIndicesAndDistances(i_corner)
            all_distances += [ distances ]
            first_points += [ first_point ]
        sum_all_distances = np.sum(all_distances, axis=0)
        mean_point = np.mean(np.vstack(tuple(first_points)), axis=0)

        index = D.getNearestPointIndex(inner_contour, tuple(mean_point))[0]
    else:
        index = forced_split_index
    reference_split_index = index

    split_indices, _,_, aux_inner_contour = _getSplitIndicesAndDistances(0)

    inner_contour_split = []
    for i in range(N_bnds-1):
        inner_contour_split += [ T.subzone(aux_inner_contour,
                        (split_indices[i]+1,1,1),(split_indices[i+1]+1,1,1)) ]
    inner_contour_split += [ T.subzone(aux_inner_contour,
                                       (split_indices[i+1]+1,1,1), (-1,-1,-1)) ]

    for inner, bnd in zip(inner_contour_split, boundaries):
        inner[0] = bnd[0] + '.inner'

    return boundaries, inner_contour_split, reference_split_index


def point(curve, index=0, as_pytree_point=False):
    '''
    extract a point from a curve at requested index

    Parameters
    ----------

        curve : zone
            structured curve

        index : int
            index at which the point is to be extracted from curve.

            .. hint::
                **index** can be negative just like Python indexing. Hence,
                ``index=-1`` can be used to extract the last point of **curve**

        as_pytree_point : bool
            if :py:obj:`True`, the returned type is a PyTree point zone.
            Otherwise, a 3-float numpy array is returned

    Returns
    -------

        point : numpy.array or zone
            extracted point of **curve** at requested **index** of type following
            the value of **as_pytree_point**
    '''
    x, y, z = J.getxyz( curve )
    x = np.ravel(x,order='K')
    y = np.ravel(y,order='K')
    z = np.ravel(z,order='K')
    try:
        pt = np.array([ x[index], y[index], z[index] ])
    except IndexError:
        ERRMSG='cannot extract point at index %d from curve %s with %d pts'%(index,curve[0],len(x))
        raise IndexError(J.FAIL+ERRMSG+J.ENDC)
    if as_pytree_point: return D.point(tuple(pt))
    return pt


def projectNormals(t, support, smoothing_iterations=0,
                   normal_projection_length=1e-3, projection_direction=None):
    '''
    project the normals vector ``{sx}``, ``{sy}`` and ``{sz}`` onto a support
    surface, optionally with smoothing iterations.

    Parameters
    ----------

        t : tree, base, zone list of zone
            structured zones with normals fields ``{sx}``, ``{sy}`` and ``{sz}``
            located at ``FlowSolution`` vertex.

            .. note::
                **curve** is modified

        support : tree, base, zone, list of zone
            surface of support for the normals

        smoothing_iterations : int
            number of iterations for smoothing the normals

        normal_projection_length : float
            Length used to compute the normals projection following the normal
            direction. Low values lead to more precise local match between the
            normal projection and the support surface, but extremely low values
            may lead to numerical innacuracies due to division by small values.

            .. hint::
                use a value slightly lower than the length of the smallest edge
                size of the cells of **support** surface


    Returns
    -------

        None : None

    '''

    for curve in I.getZones( t ):
        sx, sy, sz = J.getVars( curve, ['sx','sy','sz'] )

        curve_proj = I.copyTree( curve )
        I._rmNodesByType(curve_proj,'FlowSolution_t')
        if support:
            if projection_direction is not None:
                T._projectDir( curve_proj, support, projection_direction, oriented=1)
            else:
                T._projectOrtho( curve_proj, support )
        xp, yp, zp = J.getxyz( curve_proj )

        curve_offset = I.copyTree( curve_proj )
        xo, yo, zo = J.getxyz( curve_offset )

        xo += sx * normal_projection_length
        yo += sy * normal_projection_length
        zo += sz * normal_projection_length
        if support:
            if projection_direction is not None:
                T._projectDir( curve_offset, support, projection_direction, oriented=1)
            else:
                T._projectOrtho( curve_offset, support )


            xo, yo, zo = J.getxyz( curve_offset )

        sx[:] = xo - xp
        sy[:] = yo - yp
        sz[:] = zo - zp

        C._normalize( curve, ['sx', 'sy', 'sz'])

def reverseNormals(curves):
    '''
    reverse direction of normals

    Parameters
    ----------

        curves : list of zone
            list of curves containing fields ``{sx}``, ``{sy}``, ``{sz}``
            located at Vertex in container *FlowSolution*
    '''
    for c in I.getZones(curves):
        sx, sy, sz = J.getVars(c, ['sx', 'sy', 'sz'])
        sx *= -1
        sy *= -1
        sz *= -1


def addNormals(curves, support=None, smoothing_iterations=0,
               normal_projection_length=1e-3, reverse_normals=False):
    '''
    add normal vector to a set of structured curves using the mean oscullatory
    plane of them all and optionally projecting them onto a support surface.

    Parameters
    ----------

        curves : list of zone
            list of structured curves where normals fields is going to be
            added

            .. note::
                **curves** are modified (fields are added)

        support : tree, base, zone, list of zone or :py:obj:`None`
            see :py:func:`projectNormals`

        smoothing_iterations : int
            see :py:func:`projectNormals`

        normal_projection_length : float
            see :py:func:`projectNormals`

        reverse_normals : bool
            if :py:obj:`True`, then reverses the normals of **curves**
    '''
    curves = I.getZones(curves)
    closed_contour = concatenate( curves )
    I._rmNodesByType(closed_contour, 'FlowSolution_t')
    getCurveNormalMap(closed_contour)
    I._rmNodesByName(closed_contour, I.__FlowSolutionCenters__)
    J.migrateFields(closed_contour, curves, False)
    if reverse_normals: reverseNormals( curves )
    for c in curves:
        projectNormals(c, support, smoothing_iterations=smoothing_iterations,
                           normal_projection_length=normal_projection_length)

def getVisualizationNormals(t, length=1.):
    lines = []
    for zone in I.getZones(t):
        x,y,z = J.getxyz(zone)
        sx, sy, sz = J.getVars(zone,['sx','sy','sz'])
        x = x.ravel(order='K')
        y = y.ravel(order='K')
        z = z.ravel(order='K')
        sx = sx.ravel(order='K')
        sy = sy.ravel(order='K')
        sz = sz.ravel(order='K')

        for i in range(len(x)):
            lines += [D.line((x[i],y[i],z[i]),
                     (x[i]+length*sx[i],y[i]+length*sy[i],z[i]+length*sz[i]), 2)]
    I._correctPyTree(lines,level=3)
    return lines


def computeBarycenterDirectionalField(t, support=None, reverse=False,
                                      projectNormalsOptions={}):
    B = np.array( G.barycenter( t ) )
    if support:
        Bpt = D.point(tuple(B))
        T._projectOrtho(Bpt, support)
        B = point(Bpt)

    for curve in I.getZones(t):
        sx, sy, sz = J.invokeFields( curve, ['sx', 'sy', 'sz'] )
        x, y, z = J.getxyz( curve )
        sign = -1 if reverse else 1
        sx[:] = sign * (B[0] - x)
        sy[:] = sign * (B[1] - y)
        sz[:] = sign * (B[2] - z)
        C._normalize( curve, ['sx', 'sy', 'sz'] )
        if support: projectNormals(curve, support, **projectNormalsOptions)

def computeDirectionalField(t, Point, support=None, reverse=False,
                                      projectNormalsOptions={}):
    B = np.array( Point )
    if support:
        Bpt = D.point(tuple(B))
        T._projectOrtho(Bpt, support)
        B = point(Bpt)

    for curve in I.getZones(t):
        sx, sy, sz = J.invokeFields( curve, ['sx', 'sy', 'sz'] )
        x, y, z = J.getxyz( curve )
        sign = -1 if reverse else 1
        sx[:] = sign * (B[0] - x)
        sy[:] = sign * (B[1] - y)
        sz[:] = sign * (B[2] - z)
        C._normalize( curve, ['sx', 'sy', 'sz'] )
        if support: projectNormals(curve, support, **projectNormalsOptions)

def projectOnAxis(t, rotation_axis, rotation_center):
    a = np.array(rotation_axis, dtype=float)
    c = np.array(rotation_center, dtype=float)
    a /= a.dot(a)

    for zone in I.getZones(t):
        x, y, z = J.getxyz( zone )
        for i in range(len(x)):
            CX = np.array([x[i]-c[0], y[i]-c[1], z[i]-c[2]])
            P = c + a * CX.dot(a)
            x[i] = P[0]
            y[i] = P[1]
            z[i] = P[2]

def getCharacteristicLength(t):
    uns = C.convertArray2Tetra(t)
    uns = T.merge(uns)
    uns, = I.getZones(uns)
    BB = G.BB(uns,'OBB')
    L = distance(point(BB,0),point(BB,-1))
    return L



def buildBezierAtCurvesExtrema(curve1, curve2, number_of_points, tension1=0.5,
                              tension2=0.5, length1=None, length2=None,
                              support=None):

    x1, y1, z1 = J.getxyz( curve1 )
    sx1, sy1, sz1 = J.getVars( curve1, ['sx','sy','sz'])
    x2, y2, z2 = J.getxyz( curve2 )
    sx2, sy2, sz2 = J.getVars( curve2, ['sx','sy','sz'])

    # points
    start_1 = np.array([x1[0],y1[0],z1[0]],dtype=float)
    start_2 = np.array([x2[0],y2[0],z2[0]],dtype=float)
    end_1 = np.array([x1[-1],y1[-1],z1[-1]],dtype=float)
    end_2 = np.array([x2[-1],y2[-1],z2[-1]],dtype=float)


    # directions
    start_v1 = np.array([sx1[0],sy1[0],sz1[0]],dtype=float)
    start_v2 = np.array([sx2[0],sy2[0],sz2[0]],dtype=float)
    end_v1 = np.array([sx1[-1],sy1[-1],sz1[-1]],dtype=float)
    end_v2 = np.array([sx2[-1],sy2[-1],sz2[-1]],dtype=float)

    TotalLength_start = distance(start_1, start_2)
    TotalLength_end = distance(end_1, end_2)

    # first curve generation
    poly_start = D.polyline([tuple(start_1),
                        tuple(start_1 + tension1*TotalLength_start*start_v1),
                        tuple(start_2 - tension2*TotalLength_start*start_v2),
                        tuple(start_2)])
    if support: T._projectOrtho(poly_start, support)
    bezier_start = D.bezier(poly_start,N=500)
    if support: T._projectOrtho(bezier_start, support)
    first_curve = discretize(bezier_start,N=number_of_points,
        Distribution=dict(kind='tanhTwoSides',
            FirstCellHeight=length1,LastCellHeight=length2))
    if support: T._projectOrtho(first_curve, support)

    # second curve generation
    poly_end = D.polyline([tuple(end_1),
                        tuple(end_1 + tension1*TotalLength_end*end_v1),
                        tuple(end_2 - tension2*TotalLength_end*end_v2),
                        tuple(end_2)])
    if support: T._projectOrtho(poly_end, support)
    bezier_end = D.bezier(poly_end,N=500)
    if support: T._projectOrtho(bezier_end, support)
    second_curve = discretize(bezier_end,N=number_of_points,
        Distribution=dict(kind='tanhTwoSides',
            FirstCellHeight=length1,LastCellHeight=length2))
    if support: T._projectOrtho(second_curve, support)

    return first_curve, second_curve

def fillWithBezier(curve1, curve2, number_of_points, tension1=0.5, tension2=0.5,
          tension1_is_absolute=False, tension2_is_absolute=False,
          length1=None, length2=None, support=None, projection_direction=None,
          projection_point=None,
          only_at_indices=[]):

    x1, y1, z1 = J.getxyz( curve1 )
    sx1, sy1, sz1 = J.getVars( curve1, ['sx','sy','sz'])
    x2, y2, z2 = J.getxyz( curve2 )
    sx2, sy2, sz2 = J.getVars( curve2, ['sx','sy','sz'])

    if len(x1) != len(x2): raise ValueError('curves must have same nb of points')

    if not length1: length1 = meanSegmentLength(curve1)
    if not length2: length2 = meanSegmentLength(curve2)

    fill_curves = []
    indices = only_at_indices if len(only_at_indices)>0 else range(len(x1))

    for i in indices:
        pt1 = np.array([x1[i],y1[i],z1[i]],dtype=float)
        pt2 = np.array([x2[i],y2[i],z2[i]],dtype=float)
        if sx1 is None:
            v1 = 0.
        else:
            v1 = np.array([sx1[i],sy1[i],sz1[i]],dtype=float)
        if sx2 is None:
            v2 = 0.
        else:
            v2 = np.array([sx2[i],sy2[i],sz2[i]],dtype=float)

        TotalLength = distance(pt1, pt2)
        polypoints = [tuple(pt1)]

        if tension1 > 0:
            if tension1_is_absolute:
                TotalLength1 = 1.
            else:
                TotalLength1 = TotalLength
            polypoints += [tuple(pt1 + tension1*TotalLength1*v1)]

        if tension2 > 0:
            if tension2_is_absolute:
                TotalLength2 = 1.
            else:
                TotalLength2 = TotalLength
            polypoints += [tuple(pt2 + tension2*TotalLength2*v2)]
        polypoints += [tuple(pt2)]

        poly = D.polyline(polypoints)
        poly[0] = 'poly'
        if support:
            if projection_direction is not None:
                T._projectDir( poly, support, projection_direction, oriented=0)
            elif projection_point is not None:
                T._projectRay( poly, support, projection_point)
            else:
                T._projectOrtho( poly, support )
        tune(poly,pt1,0)
        tune(poly,pt2,-1)

        bezier = D.bezier(poly,N=100)
        bezier[0] = 'bezier'
        if support:
            if projection_direction is not None:
                T._projectDir( bezier, support, projection_direction, oriented=0)
            elif projection_point is not None:
                T._projectRay( bezier, support, projection_point)
            else:
                T._projectOrtho( bezier, support )
        tune(bezier,pt1,0)
        tune(bezier,pt2,-1)

        fill_curve = discretize(bezier,N=number_of_points,
                                    Distribution=dict(kind='tanhTwoSides',
                                                      FirstCellHeight=length1,
                                                      LastCellHeight=length2))
        fill_curve[0] = 'fill_curve'
        # if i == 2:
        #     for p in polypoints:
        #         print(p)
        #     C.convertPyTree2File([fill_curve, bezier, poly, curve1, curve2,
        #         support, D.point(pt1), D.point(pt2)],'test.cgns');exit()
        fill_curves += [ fill_curve ]

    if len(indices) > 1:
        fill_surface = G.stack(fill_curves)
        return fill_surface
    else:
        return fill_curves[0]


def meanSegmentLength(t):
    mean_segment_length = 0.
    for curve in I.getZones(t):
        x, y, z = J.getxyz(curve)
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        mean_segment_length += np.mean(np.sqrt(dx*dx+dy*dy+dz*dz))
    return mean_segment_length/float(len(curve))

def getConnectingCurves(curve,candidates):
    start = point(curve,0)
    end = point(curve,-1)
    all_distances_to_start = []
    all_distances_to_end = []
    for c in candidates:
        cand_start = point(c,0)
        cand_end = point(c,-1)
        all_distances_to_start += [ distance(cand_start, start) ]
        all_distances_to_start += [ distance(  cand_end, start) ]
        all_distances_to_end   += [ distance(cand_start,   end) ]
        all_distances_to_end   += [ distance(  cand_end,   end) ]
    arg_start = np.argmin(all_distances_to_start)
    arg_end   = np.argmin(all_distances_to_end)
    connected_at_start = candidates[int(arg_start/2)]
    connected_at_end = candidates[int(arg_end/2)]
    return connected_at_start, connected_at_end


def joinSequentially(curves, reorder=False, sort=False):
    '''
    Given a set of curves, joins them sequentially, hence solving the
    indexing ambiguity of closed-contours

    Parameters
    ----------

        curves : :py:class:`list` of zone
            list of structured curves to be joined

            .. note:: curves must be joinable

        reorder : bool
            if :py:obj:`True`, then applies :py:func:`reorderCurvesSequentially`
            before joining

        sort : bool
            if :py:obj:`True`, then applies :py:func:`sortCurvesSequentially`
            before joining

    Returns
    -------

        joined_curve : zone
            structured curve (join result of **curves**)
    '''
    if reorder: curves = reorderCurvesSequentially(curves)
    if sort: curves = sortCurvesSequentially(curves)

    joined_curve = curves[0]
    for curve in curves[1:]:
        x,y,z = J.getxyz(joined_curve)
        x[0] += 999
        y[0] += 999
        z[0] += 999

        joined_curve = T.join(joined_curve, curve)
        x,y,z = J.getxyz(joined_curve)
        x[0] -= 999
        y[0] -= 999
        z[0] -= 999

    return joined_curve

def segment(curve,index=0):
    x,y,z = J.getxyz(curve)
    if index>=0:
        dx = x[index+1]-x[index]
        dy = y[index+1]-y[index]
        dz = z[index+1]-z[index]
    else:
        dx = x[index-1]-x[index]
        dy = y[index-1]-y[index]
        dz = z[index-1]-z[index]
    return np.sqrt( dx*dx + dy*dy + dz*dz )

def splitAt(curve, values, field='s'):
    import scipy.interpolate
    curve, = I.getZones(curve)
    NPts = C.getNPts(curve)
    pts_arange = np.arange(NPts)
    if not isinstance(values, list):
        values = [ values ]
    parts = []
    last_i = 0
    for i in values:
        if isinstance(i, float):
            if field in ['s','length']:
                try:
                    s = I.getNodeFromName(curve, 's')[1]
                except:
                    curve = I.copyRef(curve)
                    s = gets(curve)
                if field == 'length':
                    L = D.getLength(curve)
                else:
                    L=1.
            else:
                try:
                    s = I.getNodeFromName(curve, field)[1]
                    L = 1.
                except:
                    raise ValueError(J.FAIL+'could not find field %s'%field+J.ENDC)
            interp= scipy.interpolate.interp1d(s*L,pts_arange,kind='nearest',
                                                assume_sorted=False, copy=False)
            i = int(interp(i))

        if i < 0: i += NPts - 1
        parts += [ T.subzone( curve, (last_i+1,1,1), (i+1,1,1) ) ]
        last_i = i
    parts += [ T.subzone( curve, (i+1,1,1), (-1,-1,-1) ) ]

    return parts

def reverse(curve, in_place=False):
    if in_place: return T._reorder(curve,(-1,2,3))
    else: return T.reorder(curve,(-1,2,3))

def tune(curve, point, index=0):
    x,y,z = J.getxyz(curve)
    x[index] = point[0]
    y[index] = point[1]
    z[index] = point[2]

def getLength(curve):
    xyz = np.vstack( J.getxyz(curve) )
    return np.sum(np.linalg.norm(np.diff(xyz,axis=1),axis=0))


def useEqualNumberOfPointsOrSameDiscretization(Airfoils, FoilDistribution=None):
    '''
    Given a list of curves (designed to be airfoils), force a rediscretization
    such that all airfoils will yield the same number of points, eventually using
    a user-defined specific distribution if provided.

    Parameters
    ----------

        Airfoils : :py:class:`list` of zone
            Airfoils (structured curves) to be rediscretized

        FoilDistribution : zone or :py:obj:`None`
            Indicates the dimensionless curvilinear abscissa to be employed for
            each section. Hence, each airfoil section is rediscretized. 
            If :py:obj:`None` is provided, then the distribution of the first
            airfoil in **Airfoils** is used for remapping the sections which
            yield different number of points.

            .. note:: as obtained from applying
                :py:func:`Geom.PyTree.getDistribution` to an airfoil curve with
                the desired distribution
    '''
    
    foilsNPtsArray = np.array([C.getNPts(a) for a in Airfoils])
    NAirfoils = len(Airfoils)
    AllSameNPts = np.unique(foilsNPtsArray).size == 1

    # if not all airfoils have the same nb. of points or new foilwise
    # distribution is required, re-map:

    if not AllSameNPts or FoilDistribution:
        RootFoil = Airfoils[0]
        SmoothParts = T.splitCurvatureAngle(RootFoil, 30.)
        indLongestEdge = np.argmax([D.getLength(c) for c in SmoothParts])
        RootFoil = SmoothParts[indLongestEdge]

        if FoilDistribution is None:
            Mapping = D.getDistribution(RootFoil)

        elif isinstance(FoilDistribution,dict):
            NewRootFoil = discretize(RootFoil, N=FoilDistribution['N'],
                                       Distribution=FoilDistribution)
            Mapping = D.getDistribution(NewRootFoil)

        elif isinstance(FoilDistribution,list) and isinstance(FoilDistribution[0],dict):
            NewRootFoil = polyDiscretize(RootFoil, FoilDistribution)
            Mapping = D.getDistribution(NewRootFoil)

        else:
            InputType = I.isStdNode(FoilDistribution)
            if InputType == -1:
                Mapping = D.getDistribution(FoilDistribution)
            elif InputType == 0:
                Mapping = D.getDistribution(FoilDistribution[0])
            else:
                raise ValueError('FoilDistribution not recognized')

        newAirfoils = []
        for ia in range(NAirfoils):
            SmoothParts = T.splitCurvatureAngle(Airfoils[ia], 30.)
            indLongestEdge = np.argmax([D.getLength(c) for c in SmoothParts])
            LongestEdge = SmoothParts[indLongestEdge]
            newFoil = G.map(LongestEdge,Mapping)
            newAirfoils += [newFoil]
        
        return newAirfoils
    
    else:
        return Airfoils


def interpolateAirfoils(Airfoils, Positions, RequestedPositions, order=1):
    '''
    Interpolate an airfoil geometry (or a series of airfoil geometries), from 
    a given list of Airfoil geometries.

    Parameters
    ----------

        Airfoils : :py:class:`list` of zone
            List of curves representing the different airfoils from which the 
            interpolation will be done. 

        Positions : :py:class:`list` of :py:class:`float`
            monotonically increasing vector of numbers representing the positions
            of each airfoil contained in **Airfoils**. For exemple, **Positions**
            may be the relative span of a wing.

            .. important:: the number of airfoils in **Airfoils** must be the
                same as the number of floats in **Positions**

        RequestedPositions : :py:class:`float` or :py:class:`list` of :py:class:`float`
            The requested position at which the interpolation will be computed.
            If the value or **RequestedPositions** lies outside the boundaries of
            **Positions**, then an extrapolation is performed. If multiple floats
            are given, then multiple interpolations are performed.

        order : int
            order of the interpolation

    Returns
    -------

        InterpolatedAirfoils : zone or :py:class:`list` of zone
            The interpolated geometry. If a list of **RequestedPositions** is 
            given, then **InterpolatedAirfoils** is a list of curves each one 
            corresponding to the requested position
    '''
    import scipy

    if len(Airfoils) != len(Positions):
        raise AttributeError('number of elements in Airfoils and Positions must be the same ')

    if not np.all(np.diff(Positions)) > 0:
        raise AttributeError('Positions must be monotonically increasing')

    try:
        if len(RequestedPositions) == 1:
            RequestedPositions = [RequestedPositions]
    except:
        RequestedPositions = [RequestedPositions]

    Ns = len(RequestedPositions)
    NinterFoils = len(Airfoils)
    ListOfNPts = np.array([C.getNPts(a) for a in Airfoils])
    if not all(ListOfNPts[0] == ListOfNPts):
        Airfoils = useEqualNumberOfPointsOrSameDiscretization(Airfoils)

    RediscretizedAirfoils = [Airfoils[0]]
    foil_Distri = D.getDistribution(RediscretizedAirfoils[0])
    for foil in Airfoils[1:]: RediscretizedAirfoils += [G.map(foil, foil_Distri)]
    NPts = C.getNPts(RediscretizedAirfoils[0])
    
    InterpolatedAirfoils = [D.line((0,0,0),(1,0,0),NPts) for _ in range(Ns)]

    InterpXmatrix = np.zeros((NinterFoils,NPts),dtype=np.float64,order='F')
    InterpYmatrix = np.zeros((NinterFoils,NPts),dtype=np.float64,order='F')
    for j in range(NinterFoils):
        InterpXmatrix[j,:] = J.getx(RediscretizedAirfoils[j])
        InterpYmatrix[j,:] = J.gety(RediscretizedAirfoils[j])

        u = gets(RediscretizedAirfoils[0])
        v = Positions
        interpX = scipy.interpolate.RectBivariateSpline(v,u,InterpXmatrix,
                                                        kx=order, ky=order)
        interpY = scipy.interpolate.RectBivariateSpline(v,u,InterpYmatrix,
                                                        kx=order, ky=order)

        InterpolatedX = interpX(RequestedPositions, u)
        InterpolatedY = interpY(RequestedPositions, u)

        for j in range(Ns):
            Section = InterpolatedAirfoils[j]
            Section[0] = 'foil_at_%g'%RequestedPositions[j]
            SecX,SecY = J.getxy(Section)
            SecX[:] = InterpolatedX[j,:]
            SecY[:] = InterpolatedY[j,:]

    if len(InterpolatedAirfoils)==1:
        return InterpolatedAirfoils[0]
    else:
        return InterpolatedAirfoils

def isStructuredCurve(arg):
    '''
    return :py:obj:`True` if **arg** is a structured curve.
    Otherwise return :py:obj:`False`
    '''
    if I.isStdNode(arg)==-1 and arg[3]=='Zone_t':
        Topo, Ni,Nj,Nk, dim = I.getZoneDim(arg)
        if dim == 1 and Topo[0] == 'S': return True
    return False


def removeMultiplePoints(curve, reltol=1e-5):
    if not isStructuredCurve(curve):
        raise AttributeError(J.FAIL+'curve must be a structured curve'+J.ENDC)
    
    xyz = np.vstack(J.getxyz(curve)).T
    NPts = xyz.shape[0]
    delta = np.diff(xyz,axis=0)
    relative_distances = np.linalg.norm(delta,axis=1)
    relative_distances/= np.max(relative_distances)
    boolean_mask_cells = relative_distances > reltol
    if all(boolean_mask_cells): return # no duplicated points
    boolean_mask_nodes = np.hstack((boolean_mask_cells,True))
    containers = I.getNodesFromType1(curve,'GridCoordinates_t') + \
                 I.getNodesFromType1(curve,'FlowSolution_t')
    for container in containers:
        for child in container[2]:
            if child[3] == 'DataArray_t':
                elts = len(child[1])
                if elts==NPts:
                    child[1] = child[1][boolean_mask_nodes]
                elif elts==(NPts-1):
                    child[1] = child[1][boolean_mask_cells]
                else:
                    path= '/'.join([curve[0],container[0],child[0]])
                    try: J.save(curve,'debug.cgns')
                    except: pass
                    raise ValueError(f'FATAL: unexpected dimensions of node {path}, check debug.cgns')
    newNCell = np.sum(boolean_mask_cells)
    curve[1][0][0] = newNCell+1
    curve[1][0][1] = newNCell


        

