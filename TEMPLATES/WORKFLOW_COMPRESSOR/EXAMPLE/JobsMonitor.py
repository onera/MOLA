#!/usr/bin/python
import MOLA.WorkflowCompressor as WF
import matplotlib.pyplot as plt

DIRECTORY_WORK = '/tmp_user/sator/tbontemp/rafale_rotor37/'

MFR, RPI, ETA = WF.printConfigurationStatusWithPerfo(DIRECTORY_WORK, useLocalConfig=True)

fig, ax1 = plt.subplots()

color = 'teal'
ax1.set_xlabel('MassFlow (kg/s)')
ax1.set_ylabel('Total pressure ratio (-)', color=color)
ax1.plot(MFR, RPI, 'o', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'firebrick'
ax2.set_ylabel('Isentropic efficiency (%)', color=color)
ax2.plot(MFR, 100*ETA, 'o', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(top=100)

fig.tight_layout()
plt.savefig('isospeed.pdf', dpi=150)
plt.show()
