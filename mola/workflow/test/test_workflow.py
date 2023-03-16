import os
from mola.workflow.workflow import Workflow

def test_init():
    w = Workflow()

def test_write_tree():
    w = Workflow()
    w.write_tree('main.cgns')
    os.unlink('main.cgns')

def test_set_workflow_parameters_in_tree(filename=''):
    w = Workflow()
    w.set_workflow_parameters_in_tree()
    if filename: w.write_tree(filename)

def test_get_workflow_parameters_from_tree(filename=''):
    w = Workflow()
    w.set_workflow_parameters_in_tree()
    w.write_tree('test.cgns')
    w.tree = 'test.cgns'
    w.get_workflow_parameters_from_tree()
    os.unlink('test.cgns')
    if filename: w.write_tree(filename)

def test_assemble():
    import Generator.PyTree as G
    import Transform.PyTree as T

    mesh = G.cart((0,0,0),(1,1,1),(9,9,9))

    w = Workflow(
        RawMeshComponents=[
            dict(
                Name='cartesian',
                Source=mesh,
                Mesher=None,
                CleaningMacro=None,
                BoundariesLabels=[
                    dict(Name='Farfield',
                         Type='BCFarfield',
                         Location='remaining'),
                ],
                Positioning=[
                    dict(
                        Type='TranslationAndRotation',
                        InitialFrame=dict(
                            Point=[0,0,0],
                            Axis1=[1,0,0],
                            Axis2=[0,1,0],
                            Axis3=[0,0,1]),
                        InitialRequested=dict(
                            Point=[0,0,0],
                            Axis1=[1,0,0],
                            Axis2=[0,1,0],
                            Axis3=[0,0,1]),
                        ),
                    # dict(
                    #     Type='DuplicateByRotation',
                    #     RotationPoint=[0,0,0],
                    #     RotationAxis=[0,0,1],
                    #     RightHandRuleRotation=True,
                    #     NumberOfInstances=4,
                    #     AddInstancesAsNewComponents=True,
                    #     ),
                ],
                Connection = [
                    dict(Type='Match', tolerance=1e-8),
                ],
                OversetOptions=dict(),
                SplitMesh=True,
                )
        ]
        )
    w.assemble()
    w.write_tree()
