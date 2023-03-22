import os
import numpy as np
from mola.workflow.workflow import Workflow
import mola.cgns as cgns

def test_init():
    w = Workflow()

def get_workflow1():

    x, y, z = np.meshgrid( np.linspace(0,1,3),
                           np.linspace(0,1,3),
                           np.linspace(0,1,3), indexing='ij')
    mesh = cgns.newZoneFromArrays( 'block', ['x','y','z'],
                                            [ x,  y,  z ])

    w = Workflow(
        RawMeshComponents=[
            dict(
                Name='cartesian',
                Source=mesh,
                Mesher=None,
                CleaningMacro=None,
                Families=[
                    dict(Name='Farfield',
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
                        RequestedFrame=dict(
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
                    dict(Type='Match', Tolerance=1e-8),
                ],
                OversetOptions=dict(),
                )
        ]
        )
    return w

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

def test_prepare_workflow1():
    w = get_workflow1()
    w.assemble()
    w.positioning()
    w.connect()
    w.define_families()
    w.write_tree()
