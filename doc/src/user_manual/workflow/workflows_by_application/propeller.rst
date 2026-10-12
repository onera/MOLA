##################
Workflow propeller
##################

.. py:currentmodule::  mola.workflow.rotating_component.propeller

The Workflow propeller can be imported with:

.. code-block:: python

    from mola.workflow.rotating component import propeller
    workflow = propeller.Workflow(...)

It is adapted for a single propeller simulation, possibly with a hub and a wing.

Examples using this workflow can be found in:

.. code:: text
    
    $MOLA/examples/open/workflow/rotating_component/propeller

*******************************************
Constraints on input data for this workflow
*******************************************

There must be only on zone Family named "Propeller".

If the mesh was generated with Autogrid and that the parameter `Mesher` is set to `autogrid` accordingly, 
then all needed modifications are made automatically during preprocessing (change of name / merge of families,
remove BC Families `*__CON_*` at the interface of Propeller and Farfield zones).


************************************
Default parameters for this workflow
************************************

Here are listed the default values for the attributes that were modified 
by this workflow:

* `SplittingAndDistribution`: The default `Splitter` is PyPart, used after job submission (`Strategy='AtComputation'`).

* `Extractions`: The following extractions are automatically added:
  
  * `Pressure`, `BoundaryLayer` quantities and `yPlus` on viscous walls.

  * `Force` and `Torque` on all walls. Moreover, an automatic postprocess operation computes the 
    `Thrust` and `Power`, as well as the average value and the standard deviation
    of these quantities.


*************************************
Specific parameters for this workflow
*************************************

ApplicationContext
