'''
Template for reading and showing the loads.cgns file using Matplotlib

MOLA v1.10 - 04/03/2021 - L. Bernardos
'''


import sys
import numpy as np

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.InternalShortcuts as J

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

FILE_LOADS = '/home/lbernard/PROJETS/OTTAWAN/CFD/loads.cgns'

FluxName = 'CL'

figname = 'loads.pdf' 

t = C.convertFile2PyTree(FILE_LOADS)

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