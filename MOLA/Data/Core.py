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
Multi-purpose module implementing low-level functions employed by
all other modules. This is not designed to be a user-level module.


29/04/2022 - L. Bernardos - first creation
'''


import sys
MIN_PYTHON = (3,6,1)
if sys.version_info < MIN_PYTHON:
    raise SystemError("Python %s or later is required.\n"%'.'.join([str(i) for i in MIN_PYTHON]))
import numpy as np
import pprint
import re
import os
from timeit import default_timer as toc
from pathlib import Path
home = str(Path.home())

RED  = '\033[91m'
GREEN = '\033[92m'
WARN  = '\033[93m'
PINK  = '\033[95m'
CYAN  = '\033[96m'
ENDC  = '\033[0m'

def load_source(ModuleName, filename, safe=True):
    '''
    Load a python file as a module guaranteeing intercompatibility between
    different Python versions

    Parameters
    ----------

        ModuleName : str
            name to be provided to the new module

        filename : str
            full or relative path of the file containing the source (moudule)
            to be loaded

        safe : bool
            if :py:obj:`True`, then cached files of previously loaded versions
            are explicitely removed

    Returns
    -------

        module : module
            the loaded module
    '''
    if safe:
        current_path_file = filename.split(os.path.sep)[-1]
        for fn in [filename, current_path_file]:
            try: os.remove(fn+'c')
            except: pass
        try: shutil.rmtree('__pycache__')
        except: pass

    if sys.version_info[0] == 3 and sys.version_info[1] >= 5:
        import importlib.util
        spec = importlib.util.spec_from_file_location(ModuleName, filename)
        LoadedModule = importlib.util.module_from_spec(spec)
        sys.modules[ModuleName] = LoadedModule
        spec.loader.exec_module(LoadedModule)
    elif sys.version_info[0] == 3 and sys.version_info[1] < 5:
        from importlib.machinery import SourceFileLoader
        LoadedModule = SourceFileLoader(ModuleName, filename).load_module()
    elif sys.version_info[0] == 2:
        import imp
        LoadedModule = imp.load_source(ModuleName, filename)
    else:
        raise ValueError("Not supporting Python version "+sys.version)
    
    return LoadedModule

def reload_source(module):
    '''
    Reload a python module guaranteeing intercompatibility between
    different Python versions

    Parameters
    ----------

        module : module
            pointer towards the previously loaded module
    '''
    if sys.version_info[0] == 3:
        import importlib
        importlib.reload(module)
    elif sys.version_info[0] == 2:
        import imp
        imp.reload(module)
    else:
        raise ValueError("Not supporting Python version "+sys.version)


class default_settings():
    def __init__(self):
        self.backend = 'h5py2cgns'

try:
    settings = load_source('settings', os.path.join(home,'.MOLA.py'))
except:
    settings = default_settings()



AutoGridLocation = {'FlowSolution':'Vertex',
                    'FlowSolution#Centers':'CellCenter',
                    'FlowSolution#EndOfRun':'CellCenter',
                    'FlowSolution#Init':'CellCenter',
                    'FlowSolution#SourceTerm':'CellCenter',
                    'FlowSolution#EndOfRun#Coords':'Vertex'}

def save_settings():
    writeFileFromModuleObject(settings, os.path.join(home,'.MOLA.py'))


def writeFileFromModuleObject(settings, filename='.MOLA.py'):
    Lines = '#!/usr/bin/python\n'

    for Item in dir(settings):
        if not Item.startswith('_'):
            Lines+=Item+"="+pprint.pformat(getattr(settings, Item))+"\n\n"

    with open(filename,'w') as f: f.write(Lines)

    try: os.remove(filename+'c')
    except: pass

def sortListsUsingSortOrderOfFirstList(*arraysOrLists):
    '''
    This function accepts an arbitrary number of lists (or arrays) as input.
    It sorts all input lists (or arrays) following the ordering of the first
    list after sorting.

    Returns all lists with new ordering.

    Parameters
    ----------

        arraysOrLists : comma-separated arrays or lists
            Arbitrary number of arrays or lists

    Returns
    -------

        NewArrays : list
            list containing the new sorted arrays or lists following the order
            of first the list or array (after sorting).

    Examples
    --------

    ::

        import numpy as np
        import MOLA.Data.Core as C

        First = [5,1,6,4]
        Second = ['a','c','f','h']
        Third = np.array([10,20,30,40])

        NewFirst, NewSecond, NewThird = C.sortListsUsingSortOrderOfFirstList(First,Second,Third)
        print(NewFirst)
        print(NewSecond)
        print(NewThird)

    will produce

    ::

        [1, 4, 5, 6]
        ['c', 'h', 'a', 'f']
        [20, 40, 10, 30]

    '''
    SortInd = np.argsort(arraysOrLists[0])
    NewArrays = []
    for a in arraysOrLists:
        if type(a) == 'ndarray':
            NewArray = np.copy(a,order='K')
            for i in SortInd:
                NewArray[i] = a[i]

        else:
            NewArray = [a[i] for i in SortInd]

        NewArrays.append( NewArray )

    return NewArrays



def secant(fun, x0=None, x1=None, ftol=1e-6, bounds=None, maxiter=20, args=()):
    '''
    Optimization function with similar interface as scipy's `root_scalar <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html>`_
    routine, but this version yields enhanced capabilities of error and bounds
    managment.

    Parameters
    ----------

        fun : callable function
            the scalar callable function where root has to be found.

            .. attention:: for convenience, ``fun()`` can
                return more than two objects, but **only the first one** is
                intended to be the float value where root has to be found.

        ftol : float
            absolute tolerance of function for termination.

        x0 : float
            first guess of the secant method

        x1 : float
            second guess of the secant method

        bounds : 2-float tuple
            minimum and maximum bounds of **x** for accepted search of the root.

        maxiter : int
            maximum number of search iterations. If
            algorithm reaches this number and **ftol** is not satisfied,
            then it returns the closest candidate to the root

        args : tuple
            Additional set of arguments to be passed to the function

    Returns
    -------

        sol : dict
            Contains the optimization problem solution and information
    '''

    if bounds is None: bounds = (-np.inf, +np.inf)

    # Allocate variables
    xguess=np.zeros(maxiter,dtype=np.float64)
    fval  =np.zeros(maxiter,dtype=np.float64)
    root  =np.array([0.0])
    froot =np.array([0.0])
    iters =np.array([0])

    sol = dict(
        xguess = xguess,
        fval   = fval,
        root   = root,
        froot  = froot,
        iters  = iters,
        converged = False,
        message = '',
        )


    def linearRootGuess(x,y,samples=2):
        xs = x[-samples:]
        if xs.max() - xs.min() < 1.e-6: return xs[-1], [0,0,0]
        p = np.polyfit(xs,y[-samples:],1)
        Roots = np.roots(p)
        if len(Roots) > 0: Xroot = Roots[0]
        else: Xroot = np.mean(x)

        return Xroot, p

    def parabolicRootGuess(x,y,samples=3):
        xs = x[-samples:]
        # Check if exist at least three different values in xs
        v0, v1, v2 = xs[-1], xs[-2], xs[-3]
        tol = 1.e-6
        if abs(v0-v1)<tol or abs(v0-v2)<tol or abs(v1-v2)<tol:
            return np.nan, [0,0,0]

        p = np.polyfit(xs,y[-samples:],2)
        roots = np.roots(p)
        dist = np.array([np.min(np.abs(xs-roots[0])),np.min(np.abs(xs-roots[1]))])
        closestRoot = np.argmin(dist)
        Xroot = roots[closestRoot]
        return Xroot, p


    # -------------- ROOT SEARCH ALGORITHM -------------- #
    GoodProgressSamplesCriterion = 5
    CheckIts = np.arange(GoodProgressSamplesCriterion)

    # Initialization
    xguess[0] = x0
    xguess[1] = x1
    fval[0]   = fun(x0,*args)
    fval[1]   = fun(x1,*args)
    bestInitialGuess = np.argmin(np.abs(fval[:2]))
    root[0] = bestInitialGuess
    iters[0] = 2

    for it in range(2,maxiter):

        iters[0] = it

        # Make new guess based on linear and parabolic fit
        rootL, pL = linearRootGuess(xguess[:it],fval[:it])
        rootP = rootL if it==2 else parabolicRootGuess(xguess[:it],fval[:it])[0]
        if np.iscomplex(rootP) or np.isnan(rootP): rootP=rootL
        newguess = 0.5*(rootL+rootP)

        # Handle bounds
        OutOfMaxBound =   newguess > bounds[1]
        if OutOfMaxBound: newguess = bounds[0]
        OutOfMinBound =   newguess < bounds[0]
        if OutOfMinBound: newguess = bounds[1]

        if OutOfMinBound or OutOfMaxBound:
            xguess[it] = newguess
            fval[it] = fun(newguess,*args)
            # Attempt largest set linear fit including new guess
            rootL, pL = linearRootGuess(xguess[:it+1],fval[:it+1],2)
            newguess = rootL

            inBounds = newguess >= bounds[0] and newguess <= bounds[1]

            if not inBounds:
                # Still not in bounds. Attempt to find a new
                # local gradient close to minimum bound
                xguessNew = np.array([bounds[0],bounds[0]+0.01*(bounds[1]-bounds[0])])
                fvalNew = xguessNew*0
                fvalNew[0] = fun(xguessNew[0],*args)
                fvalNew[1] = fun(xguessNew[1],*args)

                rootL, pL = linearRootGuess(xguessNew,fvalNew,2)
                newguess = rootL

                inBounds = newguess >= bounds[0] and newguess <= bounds[1]
                if not inBounds:
                    # Still not in bounds. Last attempt: try
                    # find root estimate in bounds by making
                    # a large linear fit on all iterations
                    rootL, pL = linearRootGuess(np.hstack((xguess[:it+1],xguessNew)),np.hstack((fval[:it+1],fvalNew)),it)
                    newguess = rootL
                    inBounds = newguess >= bounds[0] and newguess <= bounds[1]
                    if not inBounds:
                        # Ok, I give up now
                        # store current best guess
                        indBestGuess = np.argmin(np.abs(fval[:it+1]))
                        root[0]  = xguess[indBestGuess]
                        froot[0] = fval[indBestGuess]

                        sol['message'] = 'Out of bounds guess (%g). If your problem has a solution, try increasing the bounds and/or xtol.'%newguess
                        sol['converged'] = False
                        return sol

        # new guess may be acceptable
        if newguess == xguess[it-1]:
            newguess = np.mean(xguess[:it])
        if newguess == xguess[it-1]:
            newguess = 0.5*(bounds[0]+bounds[1])
        xguess[it] = newguess
        fval[it]   = fun(newguess,*args)

        # stores current best guess
        indBestGuess = np.argmin(np.abs(fval[:it+1]))
        root[0]  = xguess[indBestGuess]
        froot[0] = fval[indBestGuess]

        # Check if solution falls within tolerance
        converged = np.abs(fval[it]) < ftol
        sol['converged'] = converged
        if converged:
            sol['message'] = 'Solution converged within tolerance (ftol=%g)'%ftol
            sol['converged'] = converged
            break

        # Check if algorithm is making good progress
        GoodProgress = True
        if it >= GoodProgressSamplesCriterion:
            FinalIt, progress = linearRootGuess(it+CheckIts,fval[:it],GoodProgressSamplesCriterion)

            # if progress[1] <= 0:
            #     GoodProgress = False
            #     sol['message'] = 'Algorithm is making bad progress in the last %d iterations. Convergence would be obtained after %d iters, which is greater than user-provided maxiters (%d). Aborting.'%(GoodProgressSamplesCriterion, FinalIt,maxiter)
            #     return sol

            if FinalIt > maxiter:
                GoodProgress = False
                sol['message'] = 'Algorithm is not making good enough progress. Convergence would be obtained after %d iters, which is greater than user-provided maxiters (%d).'%(FinalIt,maxiter)

    if not converged:
        sol['message'] += '\nMaximum number of iterations reached.'

    return sol


def interpolate(AbscissaRequest, AbscissaData, ValuesData, Law='linear',
                **options):

    if 'linear' == Law:
        return np.interp(AbscissaRequest, AbscissaData, ValuesData, **options)

    elif Law.startswith('interp1d'):
        import scipy.interpolate as si
        ScipyLaw =  Law.split('_')[1]
        DefaultOptions=dict(axis=-1, kind=ScipyLaw, bounds_error=False,
            fill_value='extrapolate', assume_sorted=True, copy=False)
        DefaultOptions.update( options )
        interp = si.interp1d( AbscissaData, ValuesData, **DefaultOptions)
        return interp(AbscissaRequest)

    elif 'pchip' == Law:
        import scipy.interpolate as si
        DefaultOptions=dict(axis=-1, extrapolate=True)
        DefaultOptions.update( options )
        interp = si.PchipInterpolator(AbscissaData, ValuesData,**DefaultOptions)
        return interp(AbscissaRequest)

    elif 'akima' == Law:
        import scipy.interpolate as si
        DefaultOptions=dict(axis=-1, extrapolate=True)
        DefaultOptions.update( options )
        willExtrapolate = DefaultOptions['extrapolate']
        DefaultOptions.pop('extrapolate',None)
        interp = si.Akima1DInterpolator(AbscissaData,ValuesData,**DefaultOptions)
        return interp(AbscissaRequest, extrapolate=willExtrapolate)

    elif 'cubic' == Law:
        import scipy.interpolate as si
        DefaultOptions=dict(axis=-1, extrapolate=True)
        DefaultOptions.update( options )
        interp = si.CubicSpline(AbscissaData, ValuesData, **DefaultOptions)
        return interp(AbscissaRequest)

    else:
        raise AttributeError(RED+'Law %s not recognized.'%Law+ENDC)

def inverse_sinhXoverX(y):
    r'''
    Given :math:`y`, approximates :math:`x` by inversion of :math:`y = \frac{\sinh x}{x}`
    '''
    # apprximate inversion of y = sinh x / x (Appendix B)
    if y < 2.7829681:
        Y = y - 1 # (Eqn. 64)
        x = np.sqrt(6*Y)*(1 - .15*Y + .057321429*Y**2 - .024907295*Y**3 \
                               + .0077424461*Y**4 - .0010794123*Y**5) # (Eqn. 63)
    else:
        v = np.log(y) # (Eqn. 66)
        w = 1/y - .028527431 # (Eqn. 67)
        x = v + (1 + 1./v)*np.log(2*v) - .02041793 +.24902722*w \
                 + 1.9496443*w**2 - 2.6294547*w**3 + 8.56795911*w**4 # (Eqn. 65)
    return x

def inverse_sinXoverX(y):
    r'''
    Given :math:`y`, approximates :math:`x` by inversion of :math:`y = \frac{\sin x}{x}`
    '''
    # apprximate inversion of y = sin x / x (Appendix C)
    if y <= .26938972:
        π = np.pi
        x = π * (1 - y + y**2 - (1 + (π**2)/6.)*y**3 + 6.794732*y**4 \
                 - 13.205501*y**5 + 11.726095*y**6) # (Eqn. 69)
    elif .26938972 < y <= 1:
        Y = 1 - y # (Eqn. 71)
        x = np.sqrt(6*Y)*(1 + .15*Y + .057321429*Y**2 + .048774238*Y**3 \
                               - .053337753*Y**4 - .075845134*Y**5) # (Eqn. 70)
    else:
        raise ValueError('cannot use y>1')
    return x

def tanhOneSideFromSlope(slope, ji, tol=1e-3):
    ξ = ji # initial distribution (uniform form 0 to 1) or single point
    if slope < 1 - tol:
        Δx = inverse_sinXoverX( slope ) / 2. # (Eqn. 57)
        t = 1 + np.tan( Δx*( ξ - 1) ) / np.tan( Δx ) # (Eqn. 55)
    elif slope > 1 + tol:
        Δy = inverse_sinhXoverX( slope )/2. # (Eqn. 54)
        t = 1 + np.tanh( Δy*( ξ - 1) ) / np.tanh( Δy ) # (Eqn. 55)

    else:
        t = ξ * (1. - 0.5*(slope-1)*(1-ξ)*(2-ξ)) # (Eqn. 60)

    return t

def getFirstSlopeForTanhOneSide(guess_slope, uniform_step, requested_step):
    Δξ = uniform_step
    Δt = requested_step
    s0 = guess_slope
    def residual(x): return tanhOneSideFromSlope(x, Δξ) - Δt
    s0 = findRootWithScipy(residual, method='brenth', x0=0.7*s0, x1=0.9*s0,
            bracket=(1e-10, 10*s0), maxiter=100)

    return s0

def tanhOneSideFromStep(step, N):
    Δξ = 1. / (N - 1)
    Δt = step
    guess_slope = Δξ/Δt
    slope = getFirstSlopeForTanhOneSide(guess_slope, Δξ, Δt)
    ξ = np.linspace(0,1,N)
    t = tanhOneSideFromSlope(slope, ξ)

    return t

##################### tanh two sides #####################

def tanhTwoSidesFromSlopes(s0, s1, ji, tol=1e-3):
    ξ = ji # initial distribution (uniform form 0 to 1) or single point in [0,1]
    B = np.sqrt( s0 * s1 )       # (Eqn. 36)
    A = np.sqrt(s0/s1)           # (Eqn. 37)
    if B > 1 + tol:
        Δy = inverse_sinhXoverX( B ) # (Eqn. 46)
        u = 0.5 + np.tanh(Δy*(ξ-0.5))/ (2*np.tanh(0.5*Δy)) # (Eqn. 47)
        # t = np.tanh(ξ*Δy)/( np.sinh(Δy) + (1 - A*np.cosh(Δy)) ) # (Eqn. 72a)
    elif B < 1 - tol:
        Δx = inverse_sinXoverX( B ) # (Eqn. 49)
        # t = np.tan(ξ*Δx)/( A*np.sin(Δx) + (1 - A*np.cos(Δx))*np.tan(ξ*Δx) ) # (Eqn. 42)
        u = 0.5 + np.tan(Δx*(ξ-0.5))/ (2*np.tan(0.5*Δx)) # (Eqn. 50)
    else:
        u = ξ * ( 1 + 2*(B-1)*(ξ-0.5)*(1-ξ) )
    t = u / (A + (1-A)*u)
    return t

def getSlopesForTanhTwoSides(s0, s1, uniform_step, step_start, step_end,
                             outitersmax, step_tol):
    Δξ = uniform_step
    Δt0 = step_start
    Δt1 = step_end
    def residual0(x,s1): return tanhTwoSidesFromSlopes(x, s1, Δξ) - Δt0
    def residual1(x,s0): return tanhTwoSidesFromSlopes(s0, x, 1-Δξ) - (1-Δt1)
    for i in range( outitersmax ):
        s0 = findRootWithScipy(residual0, method='brenth', x0=0.7*s0, x1=0.9*s0,
                bracket=(1e-10, 10*s0), maxiter=100, args=s1)
        s1 = findRootWithScipy(residual1, method='brenth', x0=0.7*s1, x1=0.9*s1,
                bracket=(1e-10, 10*s1), maxiter=100, args=s0)
        res0 = residual0(s0,s1)
        res1 = residual1(s1,s0)
        if res0 < step_tol and res1 < step_tol: break

    return s0, s1

def tanhTwoSidesFromSteps(step_start, step_end, N, outitersmax=100, step_tol=1e-8):
    Δξ = 1. / (N - 1)
    Δt0 = step_start
    Δt1 = step_end
    s0 = Δξ/Δt0
    s1 = Δξ/Δt1
    s0, s1 = getSlopesForTanhTwoSides(s0, s1, Δξ, Δt0, Δt1, outitersmax, step_tol)
    ξ = np.linspace(0,1,N)
    t = tanhTwoSidesFromSlopes(s0, s1, ξ)

    return t

def findRootWithScipy(fun, **kwargs):
    from scipy.optimize import root_scalar
    sol = root_scalar(fun, **kwargs)
    return sol.root

