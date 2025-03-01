#######################
Workflow turbomachinery
#######################

.. py:currentmodule::  mola.workflow.rotating_component.turbomachinery

The Workflow turbomachinery can be imported with:

.. code-block:: python

    from mola.workflow.rotating component import turbomachinery
    workflow = turbomachinery.Workflow(...)

It is adapted for fan, compressor and turbine applications. Only tested for axial configurations.

************************************
Following objects are user interface
************************************

.. autoclass:: Workflow
.. autoclass:: WorkflowManager

****************************************
Following objects are not user interface
****************************************

.. autoclass:: mola.workflow.rotating_component.turbomachinery.workflow::WorkflowTurbomachinery

.. autoclass:: mola.workflow.rotating_component.turbomachinery.manager::WorkflowTurbomachineryManager
    :show-inheritance:
    :members:
