###########################
Workflow rotating component
###########################

.. py:currentmodule::  mola.workflow.rotating_component

The Workflow rotating component can be imported with:

.. code-block:: python

    from mola.workflow import rotating_component
    workflow = rotating_component.Workflow(...)

.. caution::

    It is **not** an applicative workflow ! It is designed to mutualized methods 
    between applications with rotating parts. Prefer to use workflows turbomachinery 
    or propeller depending on what you are doing.



.. autoclass:: Workflow

.. autoclass:: mola.workflow.rotating_component.workflow::WorkflowRotatingComponent

