###############
Workflow inputs
###############

.. py:currentmodule::  mola.workflow.interface

*********************************
Read and transform the input mesh 
*********************************
.. automethod:: WorkflowInterface.add_to_RawMeshComponents()


***********************************************
Reference values for fluid, flow and turbulence
***********************************************
.. automethod:: WorkflowInterface.set_Fluid()
.. automethod:: WorkflowInterface.set_Flow()
.. automethod:: WorkflowInterface.set_Turbulence()


*******************
Boundary Conditions
*******************

.. automethod:: WorkflowInterface.set_BoundaryConditions()


********************
Numerical parameters
********************
.. automethod:: WorkflowInterface.set_Numerics()


**************
Initialization
**************
.. automethod:: WorkflowInterface.set_Initialization()


***********
Extractions
***********

Extractions are defined with the workflow attribute **Extractions** as a :class:`list`. 
Each element is a :class:`dict` and corresponds to an extraction.

For each extraction, at least one key is mandatory: Type (:class:`str`). 
Available extraction types are reported below.

Extractions of 3D fields
========================

All data that strictly needed for restart are automatically extracted.

.. automethod:: WorkflowInterface.add_to_Extractions_3D

Extractions of 2D surfaces
==========================

.. automethod:: WorkflowInterface.add_to_Extractions_BC
.. automethod:: WorkflowInterface.add_to_Extractions_IsoSurface

Extractions of 1D signals
=========================

Residuals, memory consumption and time monitoring are extracted by default.

.. automethod:: WorkflowInterface.add_to_Extractions_Integral


****************************************************************
Splitting and distribution of computational domain on processors
****************************************************************
.. automethod:: WorkflowInterface.set_SplittingAndDistribution()


********************
Convergence criteria
********************
.. automethod:: WorkflowInterface.set_ConvergenceCriteria()


*****************************
Information on job submission
*****************************
.. automethod:: WorkflowInterface.set_RunManagement()

***********************************
Parameters specific to the Workflow
***********************************

They are set using attribute **ApplicationContext**. See documentation of the specific Workflow.


*********************************
Parameters specific to the solver
*********************************

They are set using attribute **SolverParameters**.

