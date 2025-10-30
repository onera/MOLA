#######################
Workflow turbomachinery
#######################

.. py:currentmodule::  mola.workflow.rotating_component.turbomachinery

The Workflow turbomachinery can be imported with:

.. code-block:: python

    from mola.workflow.rotating component import turbomachinery
    workflow = turbomachinery.Workflow(...)

It is adapted for fan, compressor and turbine applications. Only tested for axial configurations.

Examples using this workflow can be found in:

.. code:: text
    
    $MOLA/examples/open/workflow/rotating_component/turbomachinery


************************************
Default parameters for this workflow
************************************

Here are listed the default values for the attributes that were modified 
by this workflow:

* `RawMeshComponents`: the default value of the attribute `Mesher` is `'Autogrid'`.

* `SplittingAndDistribution`: The default `Splitter` is PyPart, used after job submission (`Strategy='AtComputation'`).

* `Flow`: the default `Generator` is `Internal` (➥ :ref:`documentation'<FlowGenerator-Internal>`).

* `Numerics`: the default spatial `Scheme` is `Roe`.

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


* `Rows` (dict):

  Each element of this dict is also a dict corresponding to one row (rotor or stator) in the machine
  with the following arguments:

  * `IsRotating` (bool): 
    True if the row is a rotor (its rotation speed is then `ShaftRotationSpeed`), 
    False if it is a stator. Default value is False.

  * `NumberOfBlades` (int): Number of blades in the real machine on 360°.

  * `NumberOfBladesSimulated` (int): 
    Number of blades to simulate. 
    If needed (`NumberOfBladesSimulated<NumberOfBladesInInitialMesh`), mesh will be replicated to fit this value.
    Default Value is 1.

  * `NumberOfBladesInInitialMesh` (int): 
    Number of blades present in the provided mesh. 
    It is normally automatically computed.

* `RowType` (str):

  Should be 'compressor' (by default) or 'turbine'. It is only used to define postprocessing quantities like efficiency.

************************************
Specific methods for WorkflowManager
************************************

.. autoclass:: mola.workflow.rotating_component.turbomachinery.manager::WorkflowTurbomachineryManager
    :show-inheritance:
    :members:
