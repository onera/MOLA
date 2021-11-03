'''
prepareMesh.py template designed for COMPRESSOR Workflow.

Produces mesh.cgns from a CGNS from Autogrid 5

MOLA Dev
'''

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.WorkflowCompressor as WF

t = WF.prepareMesh4ElsA('r37.cgns', NProcs=8, ProcPointsLoad=50e3)
t = WF.parametrizeChannelHeight(t)
C.convertPyTree2File(t, 'mesh.cgns')
