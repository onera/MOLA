#!/usr/bin/python
import MOLA.WorkflowCompressor as WF

perfo = WF.printConfigurationStatusWithPerfo(
    '/tmp_user/sator/tbontemp/rafale_rotor37/',
    monitoredRow='row_1'
    )
WF.plotIsoSpeedLine(perfo)