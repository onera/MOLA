#######################
Workflow linear cascade
#######################

.. py:currentmodule::  mola.workflow.fixed.linear_cascade

The Workflow linear cascade can be imported with:

.. code-block:: python

    from mola.workflow.fixed import linear_cascade
    workflow = linear_cascade.Workflow(...)

It is adapted to configurations with a periodicity by translation.


Examples using this workflow can be found in:

.. code:: text
    
    $MOLA/examples/open/workflow/fixed/linear_cascade/turbomachinery


************************************
Default parameters for this workflow
************************************

Here are listed the default values for the attributes that were modified 
by this workflow:

* `RawMeshComponents`: the default value of the attribute `Mesher` is `'Autogrid'`.

* `SplittingAndDistribution`: The default `Splitter` is PyPart, used after job submission (`Strategy='AtComputation'`).

* `Flow`: the default `Generator` is `Internal` (➥ :ref:`documentation'<FlowGenerator-Internal>`).

* `Extractions`: **MassFlow** is automatically extracted on Inflow and Outflow 
  boundary conditions, including rotor/stator interfaces. More specifically, 
  every boundary with `Type` matching `Inflow*` or `Outflow*` are included.



*************************************
Specific parameters for this workflow
*************************************

ApplicationContext
==================

The workflow attribute `ApplicationContext` (of type dict) contains information to specify geometry and 
parameters of the turbomachine:

* `AngleOfAttackDeg` (float):
  Angle of attack in degrees of the inflow. It is a shortcut to set `Flow['Direction']`.

