'''
Template for reading and showing probes signals in arrays.cgns file using Matplotlib

MOLA v1 23/06/2023 - T. Bontemps
'''


import numpy as np
import matplotlib.pyplot as plt

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.InternalShortcuts as J

FILE_ARRAYS = 'OUTPUT/arrays.cgns'

t = C.convertFile2PyTree(FILE_ARRAYS)
zones = I.getNodesFromNameAndType(t, 'probe*', 'Zone_t') 
zones+= I.getNodesFromNameAndType(t, 'Probe*', 'Zone_t')
zones+= I.getNodesFromNameAndType(t, 'PROBE*', 'Zone_t')

varnames = ['Pressure', 'Temperature']
xvar = 'IterationNumber'

for varname in varnames:
    figname = 'probes_{}.png'.format(varname)
    plt.figure(figsize=(8.,6.),dpi=120)

    for zone in zones:
        probe = I.getName(zone)
        # Get variables in zone
        FS = I.getNodeFromType(zone, 'FlowSolution_t')
        if not I.getNodeByName1(FS, varname):
            continue
        xarray = I.getValue(I.getNodeByName1(FS, xvar))
        yarray = I.getValue(I.getNodeByName1(FS, varname))
        plt.plot(xarray, yarray, label=probe)

    plt.xlabel('Iteration')
    plt.ylabel(varname)
    plt.grid()
    plt.legend(loc='best')

    plt.tight_layout()
    print('Saving %s%s%s ...'%(J.CYAN,figname,J.ENDC))
    plt.savefig(figname)
    print(J.GREEN+'ok'+J.ENDC)

plt.show()
