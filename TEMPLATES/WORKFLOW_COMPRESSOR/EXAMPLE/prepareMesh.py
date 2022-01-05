'''
prepareMesh.py template designed for COMPRESSOR Workflow.

Produces mesh.cgns from a CGNS from Autogrid 5

MOLA Dev
'''

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.WorkflowCompressor as WF

t = WF.prepareMesh4ElsA('mesh_AG5_fan.cgns')
t = WF.parametrizeChannelHeight(t)
C.convertPyTree2File(t, 'mesh.cgns')