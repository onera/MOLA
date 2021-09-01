'''
Template for reading and showing the loads.cgns file using Matplotlib

MOLA v1.10 - 04/03/2021 - L. Bernardos
'''


import sys
import numpy as np

import Converter.PyTree as C
import Converter.Internal as I
import Converter.Filter as Filter

import MOLA.InternalShortcuts as J

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

FILE_LOADS = 'OUTPUT/loads.cgns'

varnames = ['MassflowIn', 'PressureStagnationRatio', 'TemperatureStagnationRatio', 'EfficiencyIsentropic']
Nvars = len(varnames)

t = C.convertFile2PyTree(FILE_LOADS)
zones = I.getNodesFromNameAndType(t, 'PERFOS_*', 'Zone_t')

for zone in zones:

    row = I.getName(zone).lstrip('PERFOS_')
    figname = 'perfos_{}.pdf'.format(row)

    fig, axes = plt.subplots(Nvars,2,figsize=(8.,8.),dpi=120, sharex=True)

    if Nvars == 1: axes = [axes]

    for varname, ax in zip(varnames, axes):
        v = J.getVars2Dict(zone, ['IterationNumber',
                                  varname,
                                  'avg-'+varname,
                                  'std-'+varname,])
        ax[0].set_title('{} {}'.format(row, varname.rstrip('In')))

        ax[0].plot(v['IterationNumber'], v[varname], label=varname, color='k')
        ax[0].plot(v['IterationNumber'], v['avg-'+varname],
                   label='average '+varname, color='k', linestyle='--')

        ax[1].plot(v['IterationNumber'], v['std-'+varname],
                label='std(%s)'%varname, color='k')

        if varname == 'MassflowIn':
            varname = 'MassflowOut'

            v = J.getVars2Dict(zone, ['IterationNumber',
                                  varname,
                                  'avg-'+varname,
                                  'std-'+varname,])

            ax[0].plot(v['IterationNumber'], v[varname], label=varname, color='C0')
            ax[0].plot(v['IterationNumber'], v['avg-'+varname],
                       label='average '+varname, color='C0', linestyle='--')

            ax[1].plot(v['IterationNumber'], v['std-'+varname],
                    label='std(%s)'%varname, color='C0')

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
