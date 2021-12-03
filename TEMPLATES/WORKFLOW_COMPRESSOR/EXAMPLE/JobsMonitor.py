#!/usr/bin/python
import MOLA.WorkflowCompressor as WF
import matplotlib.pyplot as plt

DIRECTORY_WORK = '/tmp_user/sator/tbontemp/rafale_rotor37/'

# Get performance in a dictionary
perfo = WF.printConfigurationStatusWithPerfo(DIRECTORY_WORK, useLocalConfig=True)


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
ax2.set_ylabel('Isentropic efficiency (%)', color=color)
for i, speed in enumerate(perfo['RotationSpeed']):
    if speed == []: continue
    else: speed = speed[0]
    ax2.plot(perfo['MassFlow'][i], perfo['EfficiencyIsentropic'][i],
        color=color, label=None, **linestyles[i])
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(top=1)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

fig.legend()

fig.tight_layout()
plt.savefig('isoSpeedLines.pdf', dpi=150)
plt.show()
