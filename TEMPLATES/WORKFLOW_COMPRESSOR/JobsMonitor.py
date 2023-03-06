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

#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import MOLA.WorkflowCompressor as WF

DIRECTORY_WORK = '/tmp_user/sator/tbontemp/rafale_rotor37/'

# Get performance in a dictionary
perfo = WF.printConfigurationStatusWithPerfo(
    DIRECTORY_WORK, useLocalConfig=True, monitoredRow='row_1')


linestyles = [dict(linestyle=ls, marker=m) for m in ['o', 's', 'd', 'h']
                                        for ls in ['-', ':', '--', '-.']]

fig, ax1 = plt.subplots()

# Total pressure ratio
color = 'teal'
ax1.set_xlabel('MassFlow (kg/s)')
ax1.set_ylabel('Total pressure ratio (-)', color=color)
for i, speed in enumerate(perfo['RotationSpeed']):
    if speed == []: continue
    else: speed = speed[0]
    ax1.plot(perfo['MassFlow'][i], perfo['PressureStagnationRatio'][i],
        color=color, label='{:.2f} rpm'.format(speed*60/2./WF.np.pi), **linestyles[i])
ax1.tick_params(axis='y', labelcolor=color)

# Isentropic efficiency
color = 'firebrick'
ax2 = ax1.twinx()
ax2.set_ylabel('Isentropic efficiency (-)', color=color)
for i, speed in enumerate(perfo['RotationSpeed']):
    if speed == []: continue
    else: speed = speed[0]
    ax2.plot(perfo['MassFlow'][i], perfo['EfficiencyIsentropic'][i],
        color=color, label=None, **linestyles[i])
    ax2.plot([], [], color='k', label='{:.0f}rpm'.format(speed/np.pi*30), **linestyles[i])
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(top=1)

Nspeeds = len(perfo['RotationSpeed'])
if Nspeeds > 1:
    if Nspeeds < 5:
        ax2.legend(loc='lower center', bbox_to_anchor= (0.5, 1.1), ncol=Nspeeds,
                borderaxespad=0, frameon=False)
    else:
        ax2.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

fig.tight_layout()
plt.savefig('isoSpeedLines.png', dpi=300)
plt.show()
