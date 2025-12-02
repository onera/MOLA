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
==================


The workflow attribute `ApplicationContext` (of type dict) contains information to specify geometry and 
parameters of the propeller:

* `ShaftRotationSpeed` (float, int):
  The rotation speed of the shaft.

* `ShaftRotationSpeedUnit` (str):
  Could be either "rpm" or "rad/s". The default value is "rpm".

* `ShaftAxis` (list, tuple, numpy.ndarray):
  The default value is [1,0,0] (X-axis).

* `HubRotationIntervals` (list or tuple or function, optional):
  This parameter defines where the hub rotates and where it is fixed. 
  If not given, the hub is rotating in zones attached to a rotor Family. 
  Otherwise, it should be a list or a tuple of intervals (xmin, xmax) where the hub rotates.
  For instance:
  
  >>> HubRotationIntervals = [(-0.15,0),(0.2, 0.3)]

  or equivalently:

  >>> HubRotationIntervals = [dict(xmin=-0.15, xmax=0), dict(xmin=0.2, xmax=0.3)]

  means that hub rotates between -0.15m and 0m, and between 0.2m and 0.3m (in the direction defined by **ShaftAxis**), 
  and is fixed outside these intervals.

  `HubRotationIntervals` could also be a function returning rotating speed 
  from arguments among `CoordinateX`, `CoordinateY` and `CoordinateZ` (names *should be* these).
  It is particularly useful for radial compressors (see SRV2 example 
  :download:`run_sator.py <../../../../../examples/open/workflow/rotating_component/turbomachinery/SRV2/run_sator.py>`).

* `NumberOfBlades` (int): Number of blades in the real machine on 360°.

* `NumberOfBladesSimulated` (int): 
  Number of blades to simulate. 
  If needed (`NumberOfBladesSimulated<NumberOfBladesInInitialMesh`), mesh will be replicated to fit this value.
  Default Value is 1.

* `NumberOfBladesInInitialMesh` (int): 
  Number of blades present in the provided mesh. 
  It is normally automatically computed.


************************************
Specific methods for WorkflowManager
************************************

The workflow Propeller has its own WorkflowManager.

