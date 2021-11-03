'''
Template for reading and showing arrays.cgns, which includes bodyforce
elements,  using Matplotlib

MOLA v1.10 - 04/03/2021 - L. Bernardos
'''


import sys
import numpy as np

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.InternalShortcuts as J

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

FILE_ARRAYS = '/home/lbernard/PROJETS/OTTAWAN/CFD/arrays.cgns'

FluxName = 'CL'
FluxNameZone = 'wallWING'

figname = 'arrays.pdf'

PropFluxName = 'Thrust'

t = C.convertFile2PyTree(FILE_ARRAYS)
zones = I.getZones(t)
ZonesDict = {}
for z in zones: ZonesDict[z[0]] = z


def _makeFormatDoubleAxes__(current, other):
    '''
    Makes a formatter adapted to matplotlib's GUI figure for showing
    double-axes data interactively.
    <current> and <other> are axes.
    '''
    def format_coord(x, y):
        # x, y are data coordinates
        # convert to display coords
        display_coord = current.transData.transform((x,y))
        inv = other.transData.inverted()
        # convert back to data coords with respect to ax
        ax_coord = inv.transform(display_coord)
        coords = [ax_coord, (x, y)]
        return ('Left: {:<40}    Right: {:<}'
                .format(*['(%d, %g)'%(x, y) for x,y in coords]))
    return format_coord

fig, ax1 = plt.subplots(1,1,figsize=(8.,8.),dpi=120, sharex=True)
ax2 = ax1.twinx()

# Plot wing
zone = ZonesDict[FluxNameZone]

v = J.getVars2Dict(zone, ['IterationNumber',
                          FluxName,
                          'avg-'+FluxName,
                          'std-'+FluxName,])

ax1.plot(v['IterationNumber'], v[FluxName], label=FluxName, color='k')
ax1.plot(v['IterationNumber'], v['avg-'+FluxName],
           label='average '+FluxName, color='r')

# Plot props
for zonename in ZonesDict:
    zone = ZonesDict[zonename]
    if not I.getNodeFromName(zone,PropFluxName): continue
    v = J.getVars2Dict(zone, ['IterationNumber', PropFluxName])
    ax2.plot(v['IterationNumber'], v[PropFluxName], label=zonename)

ax1.set_xlabel('iteration')
ax2.set_xlabel('iteration')
ax1.legend(loc='lower left')
ax2.legend(loc='lower right')

ax1.set_ylabel(FluxName)
ax2.set_ylabel(PropFluxName)

# FlatListAxes = [i for j in axes for i in j]
drawMinorGrid = True
for ax in [ax1, ax2]:
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

ax2.format_coord = _makeFormatDoubleAxes__(ax2, ax1)


plt.tight_layout()
print('Saving %s%s%s ...'%(J.CYAN,figname,J.ENDC))
plt.savefig(figname)
print(J.GREEN+'ok'+J.ENDC)
plt.show()
