'''
prepareMesh.py template designed for Workflow ORAS.

Produces mesh.cgns from a CGNS from Autogrid 5

MOLA 1.14
'''
import sys
import Converter.PyTree as C
import Converter.Internal as I

import MOLA.WorkflowORAS as WO


Filein = 'INPUT_MESHES/ORAS_ONERA_FP_67.0_RP_85.0_fromEZmesh.cgns'
t = C.convertFile2PyTree(Filein)


# Prepare the mesh of Autogrid to match the input form of the Workflow:

for fam in I.getNodesFromType(t,'FamilyName_t'):
    fam_value = I.getValue(fam)
    if 'Rotor_stator_30' in fam_value:
        new_value = fam_value.replace('30','10')
        if 'left' in new_value: new_value = new_value.replace('left','right')
        elif 'right' in new_value: new_value = new_value.replace('right','left')
        I.setValue(fam,new_value)

I._renameNode(t, 'Rotor_stator_10_left', 'MixingPlaneUpstream')
I._renameNode(t, 'Rotor_stator_10_right', 'MixingPlaneDownstream')

I._rmNodesByName(t,'Rotor_stator_30*')

I._renameNode(t, 'FP_far_field_SOLID_1_rot','Rotor_Blade')
I._renameNode(t, 'RP_far_field_SOLID_1','Stator_Blade')

I._renameNode(t, 'FP_FP', 'Rotor_Blade')
I._renameNode(t, 'FP_HUB', 'Rotor_Hub')
I._renameNode(t, 'RP_RP', 'Stator_Blade')
I._renameNode(t, 'RP_HUB', 'Stator_Hub')

I._renameNode(t, 'FP', 'Rotor')
I._renameNode(t, 'RP', 'Stator')

I._renameNode(t, 'inlet_bulb', 'Rotor')
I._renameNode(t, 'outlet_bulb', 'Stator')

I._renameNode(t, 'inlet_bulb_HUB', 'Rotor_Hub')
I._renameNode(t, 'outlet_bulb_HUB', 'Stator_Hub')

I._renameNode(t, 'FAR_FIELD', 'Farfield')

# Erase inutile families:
eraseFamilyNames = [I.getName(fam) for fam in I.getNodesFromType(t, "Family_t")
                       if 'CON' in I.getName(fam) or 'SOLID' in I.getName(fam)] 
    
for fname in eraseFamilyNames:
    C._rmBCOfType(t, 'FamilySpecified:%s'%fname)
    fbc = I.getNodeFromName2(t, fname)
    I.rmNode(t, fbc)


# The workflow use if the mesh is well adapted from Autogrid starts here:

splitOptions= dict(mode='imposed', NumberOfProcessors=96)

t = WO.prepareMesh4ElsA(t, scale=0.001, splitOptions=splitOptions)


C.convertPyTree2File(t, 'mesh.cgns')
