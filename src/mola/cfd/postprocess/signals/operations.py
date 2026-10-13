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
from scipy.ndimage import uniform_filter1d

# TODO implement "maxdiff-" (for applying to residuals decrease for example)

def slidding_average(array: np.ndarray, window: int) -> np.ndarray:
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

def slidding_std(array: np.ndarray, window: int, avg=None) -> np.ndarray: 
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

def slidding_rsd(array: np.ndarray, window: int, avg=None, std=None) -> np.ndarray:
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

