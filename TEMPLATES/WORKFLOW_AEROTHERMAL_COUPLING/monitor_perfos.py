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
Template for reading and showing the arrays.cgns file using Matplotlib

MOLA v1.10 - 04/03/2021 - L. Bernardos
'''


import sys
import copy
import numpy as np

import Converter.PyTree as C
import Converter.Internal as I
import Converter.Filter as Filter

import MOLA.InternalShortcuts as J

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

FILE_ARRAYS = 'OUTPUT/arrays.cgns'

################################################################################
#########################        Performance       #############################
################################################################################

ordering = dict(MassFlowIn=0,
    PressureStagnationRatio=1,
    TemperatureStagnationRatio=2,
    EfficiencyIsentropic=3,
    PressureStagnationLossCoeff=4
    )

t = C.convertFile2PyTree(FILE_ARRAYS)
zones = I.getNodesFromNameAndType(t, 'PERFOS_*', 'Zone_t')

def shortvarname(varname):
    if varname == 'MassFlowIn':
        return 'mf in'
    elif varname == 'MassFlowOut':
        return 'mf out'
    elif varname == 'PressureStagnationRatio':
        return 'Pt ratio'
    elif varname == 'TemperatureStagnationRatio':
        return 'Tt ratio'
    elif varname == 'EfficiencyIsentropic':
        return 'eta'
    elif varname == 'PressureStagnationLossCoeff':
        return 'cPt'
    else:
        return varname

for zone in zones:

    row = I.getName(zone).replace('PERFOS_', '')
    figname = 'perfos_{}.pdf'.format(row)

    # Get variables in zone
    FS = I.getNodeFromType(zone, 'FlowSolution_t')
    varnames = [I.getName(n) for n in I.getNodesFromType(FS, 'DataArray_t')]
    varnames.remove('IterationNumber')
    varnames.remove('MassFlowOut')
    for var in copy.deepcopy(varnames):
        if any([pattern in var for pattern in ['avg-', 'std-', 'rsd-']]):
            varnames.remove(var)
    varnames = sorted(varnames, key=lambda k: ordering[k])
    Nvars = len(varnames)

    fig, axes = plt.subplots(Nvars,2,figsize=(8.,8.),dpi=120, sharex=True)

    if Nvars == 1: axes = [axes]

    for varname, ax in zip(varnames, axes):
        svar = shortvarname(varname)
        v = J.getVars2Dict(zone, ['IterationNumber',
                                  varname,
                                  'avg-'+varname,
                                  'rsd-'+varname,])
        ax[0].set_title('{} {}'.format(row, varname.rstrip('In')))

        ax[0].plot(v['IterationNumber'], v[varname], label=svar, color='k')
        if v['avg-'+varname][0] is not None:
            ax[0].plot(v['IterationNumber'], v['avg-'+varname], \
                label='avg %s'%svar, color='k', linestyle='--')
        if v['rsd-'+varname][0] is not None:
            ax[1].plot(v['IterationNumber'], v['rsd-'+varname], \
                label='rsd %s'%svar, color='k')

        if varname == 'MassFlowIn':
            varname = 'MassFlowOut'
            svar = shortvarname(varname)

            v = J.getVars2Dict(zone, ['IterationNumber',
                                  varname,
                                  'avg-'+varname,
                                  'rsd-'+varname,])

            ax[0].plot(v['IterationNumber'], v[varname], label=svar, color='C0')
            if v['avg-'+varname][0] is not None:
                ax[0].plot(v['IterationNumber'], v['avg-'+varname], \
                    label='avg %s'%svar, color='C0', linestyle='--')
            if v['rsd-'+varname][0] is not None:
                ax[1].plot(v['IterationNumber'], v['rsd-'+varname], \
                    label='rsd %s'%svar, color='C0')

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

################################################################################
#########################      CWIPI COUPLING      #############################
################################################################################

zones = I.getNodesFromNameAndType(t, 'SEND_*', 'Zone_t')

for zone in zones:

    row = I.getName(zone)
    figname = 'coupling_{}.pdf'.format(row.replace('SEND_', ''))

    # Get variables in zone
    FS = I.getNodeFromType(zone, 'FlowSolution_t')
    varnames = [I.getName(n) for n in I.getNodesFromType(FS, 'DataArray_t')]
    varnames.remove('IterationNumber')
    for var in copy.deepcopy(varnames):
        if any([pattern in var for pattern in ['avg-', 'std-', 'rsd-']]):
            varnames.remove(var)
    Nvars = len(varnames)

    fig, axes = plt.subplots(Nvars,2,figsize=(8.,8.),dpi=120, sharex=True)

    if Nvars == 1: axes = [axes]

    for varname, ax in zip(varnames, axes):
        svar = shortvarname(varname)
        v = J.getVars2Dict(zone, ['IterationNumber',
                                  varname,
                                  'avg-'+varname,
                                  'rsd-'+varname,])
        ax[0].set_title('{} {}'.format(row, varname.rstrip('In')))

        ax[0].plot(v['IterationNumber'], v[varname], label=svar, color='k')
        if v['avg-'+varname][0] is not None:
            ax[0].plot(v['IterationNumber'], v['avg-'+varname], \
                label='avg %s'%svar, color='k', linestyle='--')
        if v['rsd-'+varname][0] is not None:
            ax[1].plot(v['IterationNumber'], np.abs(v['rsd-'+varname]), \
                label='rsd %s'%svar, color='k')

        if varname == 'TemperatureMax':
            svar = 'TemperatureMax RECV'
            zoneRECV = I.getNodeFromNameAndType(t, I.getName(zone).replace('SEND_', 'RECV_'), 'Zone_t')
            if zoneRECV:
                v = J.getVars2Dict(zoneRECV, ['IterationNumber',
                                      varname,
                                      'avg-'+varname,
                                      'rsd-'+varname,])

                ax[0].plot(v['IterationNumber'], v[varname], label=svar, color='C0')
                if v['avg-'+varname][0] is not None:
                    ax[0].plot(v['IterationNumber'], v['avg-'+varname], \
                        label='avg %s'%svar, color='C0', linestyle='--')
                if v['rsd-'+varname][0] is not None:
                    ax[1].plot(v['IterationNumber'], v['rsd-'+varname], \
                        label='rsd %s'%svar, color='C0')

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


################################################################################
#########################         Massflow         #############################
################################################################################

massflows = dict()

for zone in I.getZones(t):
    massflowNode = I.getNodeFromName(zone, 'MassFlow')
    if massflowNode is not None:
        massflow = I.getValue(massflowNode)
        iterations = I.getValue(I.getNodeFromName(zone, 'IterationNumber'))
        massflows[I.getName(zone)] = dict(
            massflow = np.abs(massflow),
            iterations = iterations
        )

    else:
        massflowInNode = I.getNodeFromName(zone, 'MassFlowIn')
        massflowOutNode = I.getNodeFromName(zone, 'MassFlowOut')
        if massflowInNode is not None and massflowOutNode is not None:
            massflowIn = I.getValue(massflowInNode)
            massflowOut = I.getValue(massflowOutNode)
            iterations = I.getValue(I.getNodeFromName(zone, 'IterationNumber'))
            massflows[I.getName(zone)+'_In'] = dict(
                massflow = np.abs(massflowIn),
                iterations = iterations
            )
            massflows[I.getName(zone)+'_Out'] = dict(
                massflow = np.abs(massflowOut),
                iterations = iterations
            )

if massflows.keys() != []:
    plt.figure()
    zones = sorted(massflows.keys())
    for zone in zones:
        plt.plot(massflows[zone]['iterations'], massflows[zone]['massflow'], label=zone)
    plt.xlabel('Iterations')
    plt.ylabel('MassFlow (kg/s)')
    plt.legend(loc='best')
    plt.grid()
    figname = "massflow.pdf"
    print('Saving %s%s%s ...'%(J.CYAN,figname,J.ENDC))
    plt.savefig(figname, dpi=150, bbox_inches='tight')
    print(J.GREEN+'ok'+J.ENDC)


################################################################################
#########################         Residuals         ############################
################################################################################

ConvergenceHistory = I.getNodeFromName(t, 'GlobalConvergenceHistory')
if ConvergenceHistory:
    residuals = dict()
    FS = I.getNodeFromType(ConvergenceHistory, 'FlowSolution_t')
    for node in I.getNodesFromType(FS, 'DataArray_t'):
        residuals[I.getName(node)] = I.getValue(node)

    varList = list(residuals)
    varList.remove('IterationNumber')
    varList.sort()
    plt.figure()
    for varname in varList:
        plt.plot(residuals['IterationNumber'], residuals[varname], label=varname)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Residuals')
    plt.legend(loc='best')
    plt.grid()
    figname = "residuals.pdf"
    print('Saving %s%s%s ...'%(J.CYAN,figname,J.ENDC))
    plt.savefig(figname, dpi=150, bbox_inches='tight')
    print(J.GREEN+'ok'+J.ENDC)

plt.show()
