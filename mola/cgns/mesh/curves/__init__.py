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
Main subpackage for curves operations

28/04/2022 - L. Bernardos - first creation
'''
import numpy as np
from mola.misc import RED,GREEN,YELLOW,PINK,CYAN,ENDC
from mola.math_tools import interpolate, tanhOneSideFromStep, tanhTwoSidesFromSteps
from ...zone import Zone

class Curve(Zone):
    """docstring for Curve"""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def point(self, index=0, return_type='numpy'):
        x, y, z = self.xyz()
        if return_type == 'numpy':
            return np.array([x[index], y[index], z[index]])
        elif return_type == 'list':
            return x[index], y[index], z[index]
        elif return_type == 'zone':
            from .point import Point
            return Point( Coordinates = [x[index], y[index], z[index]] )
        else:
            MSG = 'return_type=%s not supported'%return_type
            raise AttributeError(RED+MSG+ENDC)

    def start(self, return_type='numpy'):
        return self.point(0,return_type)

    def end(self, return_type='numpy'):
        return self.point(-1,return_type)

    def mid(self, return_type='numpy', exact=True):
        NPts = self.numberOfPoints()
        if exact and NPts%2 == 0:
            raise ValueError('exact mid requires odd number of points')
        return self.point(int(NPts/2),return_type)

    def segmentLength(self, index=0):
        x,y,z = self.xyz()
        if index>=0:
            dx = x[index+1]-x[index]
            dy = y[index+1]-y[index]
            dz = z[index+1]-z[index]
        else:
            dx = x[index-1]-x[index]
            dy = y[index-1]-y[index]
            dz = z[index-1]-z[index]
        return np.sqrt( dx*dx + dy*dy + dz*dz )

    def length(self):
        xyz = np.vstack( self.xyz() )
        return np.linalg.norm(np.sum(np.abs(np.diff(xyz,axis=1)),axis=1))

    def tangent(self, index=0, reverse=False):
        x,y,z = self.xyz()
        NPts = len(x)
        if index==0:
            dx = x[1]-x[0]
            dy = y[1]-y[0]
            dz = z[1]-z[0]
        elif index == -1 or index == NPts-1:
            dx = x[-1]-x[-2]
            dy = y[-1]-y[-2]
            dz = z[-1]-z[-2]

        else:
            dxM = x[index-1]-x[index]
            dyM = y[index-1]-y[index]
            dzM = z[index-1]-z[index]
            dxP = x[index+1]-x[index]
            dyP = y[index+1]-y[index]
            dzP = z[index+1]-z[index]
            dx = 0.5 * (dxM + dxP)
            dy = 0.5 * (dyM + dyP)
            dz = 0.5 * (dzM + dzP)

        t = np.array([dx,dy,dz])
        t /= np.sqrt(t.dot(t))

        if reverse: t *= -1

        return t

    def reverse(self):
        for field in self.allFields(return_type='list'):
            field[:] = field[::-1]

    def abscissa(self, dimensional=False):
        x, y, z = self.xyz()
        s = self.fields('s')
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        DimensionalAbscissa = np.cumsum(np.sqrt(dx**2+dy**2+dz**2))
        s[0] = 0.
        s[1:] = DimensionalAbscissa
        if not dimensional: s /= s[-1]
        return s

    def discretize(self, **kwargs):
        '''
        tanh and bitanh ref:

        NASA Contractor Report 3313
        On One-Dimensional Streatching Functions for Finite-Difference Calculations
        Marcel Vinokur, University of Santa Clara, California
        October 1980
        CR-3313
        '''
        try:
            kind = kwargs['kind']
        except KeyError:
            if 'ratio' in kwargs:
                kind = 'geometric'
            elif 'first' in kwargs and 'last' in kwargs:
                kind = 'bitanh'
            elif 'first' in kwargs or 'last' in kwargs:
                kind = 'tanh'
            elif 'parameter' in kwargs:
                kind = 'trigonometric'
            else:
                kind = 'uniform'

        try: N = kwargs['N']
        except KeyError: N = self.numberOfPoints()

        GC = self.childNamed( 'GridCoordinates' )
        x_node = GC.childNamed( 'CoordinateX' )
        y_node = GC.childNamed( 'CoordinateY' )
        z_node = GC.childNamed( 'CoordinateZ' )
        x = x_node.value()
        y = y_node.value()
        z = z_node.value()

        sv = np.linspace( 0, 1, len(x) )
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        DimensionalAbscissa = np.cumsum(np.sqrt(dx*dx+dy*dy+dz*dz))
        sv[1:] = DimensionalAbscissa
        sv /= sv[-1]

        L = self.length()
        snew = np.linspace( 0, 1, N )

        if kind == 'geometric':
            try:
                snew[1] = kwargs['first']/L
            except KeyError:
                raise AttributeError('discretize kind %s requires attribute "first"'%kind)

            if snew[1] > 1:
                raise ValueError('size of first segment cannot be greater than total curve length')

            try:
                r = kwargs['ratio']
            except KeyError:
                raise AttributeError('discretize kind %s requires attribute "ratio"'%kind)

            for i in range(2,N):
                snew[i] = snew[i-1] + r * (snew[i-1] - snew[i-2])
            snew = snew[ snew <= 1 ]

        elif kind == 'tanh':
            fromLast = False
            try:
                step = kwargs['first']/L
            except KeyError:
                fromLast = True

            if fromLast:
                try:
                    step = kwargs['last']/L
                except KeyError:
                    raise AttributeError('tanh requires first or last step data')
                snew = tanhOneSideFromStep(step, N)
                snew = np.abs(snew-1)[::-1]
            else:
                snew = tanhOneSideFromStep(step, N)
                snew = snew
        elif kind == 'bitanh':
            first_step = kwargs['first']/L
            last_step = kwargs['last']/L
            bitanh_options = dict()
            try: bitanh_options.update(step_tol=kwargs['step_tol'])
            except KeyError: pass
            try: bitanh_options.update(outitersmax=kwargs['outitersmax'])
            except KeyError: pass
            snew = tanhTwoSidesFromSteps(first_step, last_step, N, **bitanh_options)

        N = len(snew) # since kind='ratio' will not verify Npts in general
        x_node.setValue( interpolate(snew, sv, x, 'pchip') )
        y_node.setValue( interpolate(snew, sv, y, 'pchip') )
        z_node.setValue( interpolate(snew, sv, z, 'pchip') )
        self.setValue(np.array([[N,N-1,0]],dtype=np.int32,order='F'))

        if self.hasFields():
            selfCopy = self.copy()
            s = selfCopy.newFields('s')
            snew = selfCopy.abscissa()
            for fs in self.group( Type='FlowSolution_t' ):
                location = self.inferLocation( fs.name() )
                if location == 'CellCenter':
                    try:
                        s = sc
                    except NameError:
                        sc = 0.5 * ( sv[1:] + sv[:-1] )
                    s = sc
                elif location == 'Vertex':
                    s = sv
                else:
                    raise ValueError(RED+'unexpected location %s'%location+ENDC)

                for field in fs.children():
                    if field.type() != 'DataArray_t': continue
                    field.setValue( interpolate(snew, s, field.value(), 'cubic') )
