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

