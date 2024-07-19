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

import numpy as np
from scipy.ndimage.filters import uniform_filter1d

def _extendArraysWithStatistics(node, VarName, Operations, AveragingIterations):

    try:
        IterationNumber = node.get(Name='IterationNumber')
    except BaseException as e:
        return # this is the case for GlobalConvergenceHistory at present
    IterationWindow = len(IterationNumber[IterationNumber>(IterationNumber[-1]-AveragingIterations)])
    if IterationWindow < 2: return

    if isinstance(Operations, str): 
        Operations = [Operations]

    for StatType in Operations:

        data_node = node.get(Name='VarName')
        if not data_node:
            continue

        try:
            InstantaneousArray = data_node.value()
            InvalidValues = np.logical_not(np.isfinite(InstantaneousArray))
            InstantaneousArray[InvalidValues] = 0.
        except:
            continue

        if StatType.lower() == 'avg':
            avg = slidding_average(InstantaneousArray, IterationWindow)
            cgns.Node(Type='DataArray', Name=f'avg-{VarName}', Value=avg, Parent=node)

        elif StatType.lower() == 'std':
            avg = slidding_average(InstantaneousArray, IterationWindow)
            StatisticArray = slidding_std(InstantaneousArray, IterationWindow, avg=avg)
            

        elif StatType.lower() == 'rsd':
            avg = slidding_average(InstantaneousArray, IterationWindow)
            arraysSubset['avg-'+VarName] = avg
            std = slidding_std(InstantaneousArray, IterationWindow, avg=avg)
            arraysSubset['std-'+VarName] = std
            StatisticArray = slidding_std(InstantaneousArray, IterationWindow, avg=avg, std=std)


def slidding_average(array, window):
    '''
    Compute the slidding average of the signal

    Parameters
    ----------
        array : numpy.ndarray
            input signal

        window : int
            length of the slidding window

    Returns
    -------

        average : numpy.ndarray
            sliding average
    '''
    average = uniform_filter1d(array, size=window)
    InvalidValues = np.logical_not(np.isfinite(average))
    average[InvalidValues] = 0.
    return average

def slidding_std(array, window, avg=None):
    '''
    Compute the slidding standard deviation of the signal

    Parameters
    ----------

        array : numpy.ndarray
            input signal

        window : int
            length of the slidding window

        avg : numpy.ndarray or :py:obj:`None`
            slidding average of **array** on the same **window**. If
            :py:obj:`None`, it is computed

    Returns
    -------

        std : numpy.ndarray
            sliding standard deviation
    '''
    if avg is None:
        avg = slidding_average(array, window)

    AvgSqrd = uniform_filter1d(array**2, size=window)

    InvalidValues = np.logical_not(np.isfinite(AvgSqrd))
    AvgSqrd[InvalidValues] = 0.
    AvgSqrd[AvgSqrd<0] = 0.

    std = np.sqrt(np.abs(AvgSqrd - avg**2))

    return std

def slidding_rsd(array, window, avg=None, std=None):
    '''
    Compute the relative slidding standard deviation of the signal

    .. math::

        rsd = std / avg

    Parameters
    ----------

        array : numpy.ndarray
            input signal

        window : int
            length of the slidding window

        average : numpy.ndarray or :py:obj:`None`
            slidding average of **array** on the same **window**. If
            :py:obj:`None`, it is computed

        std : numpy.ndarray or :py:obj:`None`
            slidding standard deviation of **array** on the same **window**. If
            :py:obj:`None`, it is computed

    Returns
    -------

        rsd : numpy.ndarray
            sliding relative standard deviation
    '''
    if avg is None:
        avg = slidding_average(array, window)
    if std is None:
        std = slidding_std(array, window, avg)

    rsd = std / np.abs(avg)

    InvalidValues = np.logical_not(np.isfinite(rsd))
    rsd[InvalidValues] = 0.

    return rsd

