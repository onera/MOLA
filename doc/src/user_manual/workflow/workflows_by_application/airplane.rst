#################
Workflow airplane
#################

.. py:currentmodule::  mola.workflow.fixed.airplane

The Workflow airplane can be imported with:

.. code-block:: python

    from mola.workflow.rotating component import airplane
    workflow = airplane.Workflow(...)


Examples using this workflow can be found in:

.. code:: text
    
    $MOLA/examples/open/workflow/fixed_component/airplane


************************************
Default parameters for this workflow
************************************

Here are listed the default values for the attributes that were modified 
by this workflow:

* `SplittingAndDistribution`: The default `Splitter` is PyPart, used after job submission (`Strategy='AtComputation'`).

* `Extractions`: The following extractions are automatically added:
  
  * `Pressure` on all walls and `BoundaryLayer` quantities and `yPlus` on viscous walls.

  * `Force` and `Torque` on all walls. Moreover, an automatic postprocess operation computes the 
    aerodynamic coefficients `CL` and `CD`, as well as the average value and the standard deviation
    of these quantities.

*************************************
Specific parameters for this workflow
*************************************

ApplicationContext
==================

The workflow attribute `ApplicationContext` (of type dict) contains information to specify geometry and 
parameters of the airplane/wing. The following parameters define the orientation of the body, and 
update the attribute **Flow['Direction']** (hence it does not have to be directly modify) :

* `AngleOfAttackDeg` (float):
   The default value is 0.

* `AngleOfSlipDeg` (float):
  The default value is 0.

* `YawAxis` (list, tuple, numpy.ndarray):
  The default value is [0,0,1] (Z-axis).

* `PitchAxis` (list, tuple, numpy.ndarray):
  The default value is [0,1,0] (Y-axis).

The following parameters are normalization coefficients 
and are used to compute the aerodynamic coefficients `CL` and `CD`:

* `Length` (float):
  The default value is 1.

* `Surface` (float):
  The default value is 1.

