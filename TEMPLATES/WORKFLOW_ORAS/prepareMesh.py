'''
prepareMesh.py template designed for COMPRESSOR Workflow.

Produces mesh.cgns from a CGNS from Autogrid 5

MOLA Dev
'''
import sys
import Converter.PyTree as C
import Converter.Internal as I

import MOLA.WorkflowCompressor as WF


Filein = 'INPUT_MESHES/ORAS_ONERA_FP_67.0_RP_85.0_fromEZmesh.cgns'
t = C.convertFile2PyTree(Filein)

for fam in I.getNodesFromType(t,'FamilyName_t'):
    fam_value = I.getValue(fam)
    if 'Rotor_stator_30' in fam_value:
        new_value = fam_value.replace('30','10')
        if 'left' in new_value: new_value = new_value.replace('left','right')
        elif 'right' in new_value: new_value = new_value.replace('right','left')
        print('Remplacement de {} par {}'.format(fam_value,new_value))
        I.setValue(fam,new_value)

I._rmNodesByName(t,'Rotor_stator_30*')
# Blade All

I._rmNodesByName(t, 'FP_FP')
I._rmNodesByName(t, 'RP_RP')

I._renameNode(t, 'FP_far_field_SOLID_1_rot','FP_FP')
I._renameNode(t, 'RP_far_field_SOLID_1','RP_RP')

I._renameNode(t, 'FP_FP', 'FP_BladeWall')
I._renameNode(t, 'FP_HUB', 'FP_HubWall')
I._renameNode(t, 'RP_RP', 'RP_BladeWall')
I._renameNode(t, 'RP_HUB', 'RP_HubWall')

splitOptions= {} #dict(mode='imposed', NProcs=96)#, SplitDirections = [3])

t = WF.prepareMesh4ElsA(t, scale=0.001, splitOptions=splitOptions)



C.convertPyTree2File(t, 'mesh.cgns')
