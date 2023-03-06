#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

'''
Template for reading and showing the arrays.cgns file using Matplotlib

04/03/2021 - L. Bernardos
'''


import sys
import numpy as np

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.InternalShortcuts as J

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

FILE_ARRAYS = 'OUTPUT/arrays.cgns'
FluxName = 'MomentumXFlux'
figname = 'arrays.svg'

t = C.convertFile2PyTree(FILE_ARRAYS)

zones = I.getZones(t)

Zones2Remove = [z for z in zones if not I.getNodeFromName(z,FluxName)]

for z in Zones2Remove: zones.remove(z)

NbOfZones = len(zones)

fig, axes = plt.subplots(NbOfZones,2,figsize=(8.,8.),dpi=120, sharex=True)

if NbOfZones == 1: axes = [axes]

for zone, ax in zip(zones, axes):
    v = J.getVars2Dict(zone, ['IterationNumber',
                              FluxName,
                              'avg-'+FluxName,
                              'std-'+FluxName,])
    ax[0].set_title(zone[0])

    ax[0].plot(v['IterationNumber'], v[FluxName], label=FluxName, color='k')
    ax[0].plot(v['IterationNumber'], v['avg-'+FluxName],
               label='average '+FluxName, color='C0')


    ax[1].plot(v['IterationNumber'], v['std-'+FluxName],
            label='std(%s)'%FluxName, color='C1')
    ax[1].set_yscale('log')
    for a in ax:
        a.set_xlabel('iteration')
        a.legend(loc='best')

FlatListAxes = [i for j in axes for i in j]
drawMinorGrid = True
for ax in FlatListAxes:
    minLocX = AutoMinorLocator()
    ax.xaxis.set_minor_locator(minLocX)
    ScaleType = ax.get_yscale()
    if ScaleType == 'linear':
        minLocY = AutoMinorLocator()
        ax.yaxis.set_minor_locator(minLocY)
    elif ScaleType == 'log':
        locmaj = LogLocator(base=10.0,
                            subs=(1.0, ),
                            numticks=100)
        ax.yaxis.set_major_locator(locmaj)
        locmin = LogLocator(base=10.0,
                            subs=np.arange(2, 10) * .1,
                            numticks=100)
        ax.yaxis.set_minor_locator(locmin)
        # ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.grid(True, which='major')
    if drawMinorGrid:
        ax.xaxis.grid(True, which='minor',linestyle=':')
    else:
        ax.xaxis.grid(False, which='minor')

plt.tight_layout()
print('Saving %s%s%s ...'%(J.CYAN,figname,J.ENDC))
plt.savefig(figname)
print(J.GREEN+'ok'+J.ENDC)
plt.show()
