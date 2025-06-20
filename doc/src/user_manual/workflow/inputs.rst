###############
Workflow inputs
###############

.. py:currentmodule::  mola.workflow.interface

.. contents:: Table of contents

*********************************
Read and transform the input mesh 
*********************************
.. automethod:: WorkflowInterface.add_to_RawMeshComponents()


***********************************************
Reference values for fluid, flow and turbulence
***********************************************

Use attributes `Fluid`, `Flow` and `Turbulence`
to parametrize the reference values for the simulation. 
All these attributes are dictionaries.
They might be used for flow initialization, default values 
for boundary conditions and so on.

Fluid
=====

For now, only ideal gases are implemented. By deafult, the gas is dry air.
The following parameters (all are floats) can be modified:

+-----------------------+-------------------+
|       Parameter       |   Default value   |
+=======================+===================+
|        Gamma          |       1.4         |
+-----------------------+-------------------+
|   IdealGasConstant    |     287.053       |
+-----------------------+-------------------+
|       Prandtl         |       0.72        |
+-----------------------+-------------------+
|   PrandtlTurbulent    |       0.9         |
+-----------------------+-------------------+
|  SutherlandConstant   |     110.4         |
+-----------------------+-------------------+
|  SutherlandViscosity  |    1.78938e-05    |
+-----------------------+-------------------+
| SutherlandTemperature |     288.15        |
+-----------------------+-------------------+


Flow
====

Flow setting depends on the parameter `Generator`. Several generators are implemented:

* ``'External_rho_T_V'``: the flow is generated as a function of Density, Temperature and Velocity.

* ``'External_Mach_P_T'``: the flow is generated as a function of Mach, Pressure and Temperature.

* ``'External_Mach_Pt_Tt'``: the flow is generated as a function of Mach, PressureStagnation and TemperatureStagnation

* ``'Internal'``: the flow is generated from PressureStagnation, TemperatureStagnation, and MassFlow or Mach.

The flow direction can be set with the parameter `Direction`, by default `[1.,0,0]`.

Turbulence
==========

The following parameters are editable:

* ``Viscosity_EddyMolecularRatio`` (float):
  Ratio of :math:`\mu_t/\mu` used at freestream in order to set the 
  dissipation scale of turbulence models accordingly.

* ``Level`` (float):
  Level of freestream turbulence :math:`T_u`, typically used to set
  the first scale of turbulence models accordingly

* ``Model`` (str):
  Choose the turbulence modeling strategy. This will set appropriate
  values for each solver. If more solver-specific adjustments are 
  desired, these shall be done using **SolverParameters** attribute.
  For RANS turbulence models, please note that we tend to use the same
  name as NASA's convention https://turbmodels.larc.nasa.gov.
  The covered models are (availability depends on the employed solver):  
  * ``'Euler'``: The Euler equations are solved  
  * ``'DNS'`` or ``'ILES'`` or ``'Laminar'``: The Navier-Stokes laminar equations are solved  
  * ``'LES'``: Use Large Eddy Simulation  
  * ``'ZDES-1'``  
  * ``'ZDES-2'``  
  * ``'ZDES-3'``  
  * ``'Wilcox2006-klim'``  
  * ``'Wilcox2006-klim-V'``  
  * ``'Wilcox2006'``  
  * ``'Wilcox2006-V'``  
  * ``'SST-2003'``  
  * ``'SST-V2003'``  
  * ``'SST'``  
  * ``'SST-V'``  
  * ``'BSL'``  
  * ``'BSL-V'``  
  * ``'SST-2003-LM2009'``  
  * ``'SST-V2003-LM2009'``  
  * ``'SSG/LRR-RSM-w2012'``  
  * ``'smith'``  
  * ``'SA'``  

* ``TurbulenceCutOffRatio`` (float):
  The minimum allowed value of the turbulence quantities based upon 
  the turbulence level :math:`T_u`



*******************
Boundary Conditions
*******************

Boundary conditions are defined with the workflow attribute **BoundaryConditions** as a :class:`list`. 
Each element is a :class:`dict` and corresponds to the boundary condition imposed on one given Family.

For each :class:`dict`, at least two keys are mandatory for all types of conditions:
    * Family (:class:`str`): Name of the Family on which the boundary condition is applied.
    * Type (:class:`str`): Type of condition. Available conditions are: 
        * Farfield
        * InflowStagnation
        * InflowMassFlow 
        * OutflowPressure 
        * OutflowSupersonic  
        * OutflowMassFlow 
        * OutflowRadialEquilibrium  
        * WallViscous  
        * WallViscousIsothermal      
        * WallInviscid        
        * Wall: depending the context (Euler or Navier-Stokes), it redirects to WallInviscid or WallViscous 
        * SymmetryPlane 
        * MixingPlane     
        * UnsteadyRotorStatorInterface 
        * ChorochronicInterface    

Other arguments depends on the Type of boundary condition. 
Please see the dedicated page: :doc:`boundary_conditions`


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

Some optional parameters are common to most extraction types. They are listed in :ref:`Common parameters for extractions`.


Extractions of 3D fields
========================

.. note:: 
    
    All data that strictly needed for restart are automatically extracted.
    You have only to think about other quantities (or ``GridLocation``, or ``Frame``)
    you want to extract.

Here is an example to extract the 3D fields of Pressure and Entropy at mesh nodes:

.. code-block:: python

    dict(
        Type = '3D',
        Fields = ['Pressure', 'Entropy'],
        GridLocation = 'Vertex',  # CellCenter by default
        Container = 'FlowSolution#MyExtraction',  # Default value is 'FlowSolution#Output'
    )

.. admonition:: Default values

    * The filename (`File`) is :mola_name:`FILE_OUTPUT_3D`.
    * Extraction period (`ExtractionPeriod`) is 5000.
    * Saving period (`SavePeriod`) is 5000.



Extractions of 2D surfaces
==========================

Boundary conditions
-------------------

Here is an example to extract the Pressure on the boundary tagged with the Family 'INFLOW':

.. code-block:: python

    dict(Type = 'BC', Source = 'INFLOW', Fields = ['Pressure'])

It is also possible to extract all boundary conditions of a specific type. For instance, 
to extract pressure and friction on all walls (of type `WallViscous`), we write:

.. code-block:: python

    dict(Type = 'BC', Source = 'WallViscous', Fields = ['Pressure', 'Friction']) 

.. admonition:: Default values

    * The filename (`File`) is :mola_name:`FILE_OUTPUT_2D`.
    * If `Name` is not given, data will be sorted by Family in the output file, even if `Source` is a BC type.
    * Extraction period (`ExtractionPeriod`) is 100.
    * Saving period (`SavePeriod`) is 100.

Isosurfaces
-----------

Here is an example to extract the Pressure on the boundary tagged with the Family 'INFLOW':

.. code-block:: python

    dict(
        Type = 'IsoSurface', 
        IsoSurfaceField = 'CoordinateX', # a coordinate or a field or a Container/field
        IsoSurfaceValue = 0.5, # 
        Fields = ['Pressure']
    )

`Fields` is the list of quantities that will be present in the generated iso-surface. 
If that list is empty, the output will contain only the surface geometry, without any quantities.
It might be useful in some specific cases, e.g. to extract a Q-criterion contour or surface for Mach number equal to one. 

.. admonition:: Default values

    * The filename (`File`) is :mola_name:`FILE_OUTPUT_2D`.
    * If `Name` is not given, extraction will be named according `IsoSurfaceField` and `IsoSurfaceValue`. 
      For instance:

      * if `IsoSurfaceField='CoordinateX'` and `IsoSurfaceValue=0.5`, the default name is `Iso_X_0.5`.
      * if `IsoSurfaceField='ChannelHeight` and `IsoSurfaceValue=0.9`, the default name is `Iso_H_0.9`.
      * if `IsoSurfaceField='Mach'` and `IsoSurfaceValue=1.`, the default name is `Iso_Mach_1`.
      
    * Extraction period (`ExtractionPeriod`) is 100.
    * Saving period (`SavePeriod`) is 100.




Extractions of 1D signals
=========================

Residuals, memory consumption and time monitoring are extracted by default.

Integral quantities
-------------------

This type of extraction is very close to extractions of Type 'BC', except that `Fields` are integrated value
on the surface.

The following lines show examples to extract the massflow on the boundary tagged with the Family 'INFLOW'
and to extract the force exerted on the boundary tagged with the Family 'BLADE':

.. code-block:: python

    dict(Type = 'Integral', Source = 'INFLOW', Fields = ['MassFlow'])
    dict(Type = 'Integral', Source = 'BLADE', Fields = ['Force'])

.. admonition:: Default values

    * The filename (`File`) is :mola_name:`FILE_OUTPUT_1D`.
    * If `Name` is not given, data will be sorted by Family in the output file, even if `Source` is a BC type.
    * Extraction period (`ExtractionPeriod`) is 1.
    * Saving period (`SavePeriod`) is 100.


Probes
------

The following lines show examples to extract the massflow on the boundary tagged with the Family 'INFLOW'
and to extract the force exerted on the boundary tagged with the Family 'BLADE':

.. code-block:: python

    dict(
        Type = 'Probe', 
        Position = (0.1, 3., 1.),  # position (x,y,z) in 3d space 
        Fields = ['Pressure', 'VelocityX'],
    )

When the simulation starts, the closest cell to the given position will be identified by its index. 
During the simulation, each time values are extracted (depending on `ExtractionPeriod`), values at 
center of the previously identified cell are registered without any interpolation.

The maximum distance between `Position` and the closest cell center must be less than `Tolerance` (by default 0.01).
If this distance is greater, the probe will be ignored, considering it is out of the computational domain.
`Tolerance` may be modified if needed.

.. admonition:: Default values

    * The filename (`File`) is :mola_name:`FILE_OUTPUT_1D`.
    * If `Name` is not given, it will be based on `Position`. 
      For instance, if `Position=(0.1, 3., 1.)`, the name will be ``Probe_0.1_3._1.``.
    * Extraction period (`ExtractionPeriod`) is 1.
    * Saving period (`SavePeriod`) is 100.



Common parameters for extractions
=================================

Some optional parameters are common to most extraction types. They are listed below:

.. code-block:: python

    dict(
        File = 'other_file.cgns',  # to extract data in a separated file
        Name = 'NameOfExtraction',  # To force the name of extraction (base or zone in the CGNS file, depending on extraction Type)
        ExtractionPeriod = 50,  # Number of iterations between two extractions
        SavePeriod = 100,  # Number of iterations between two data savings. 
        Override = False,  # default is True
        Frame = 'relative',  # Default value is 'relative', the other available choice is 'absolute'
        GridLocation = 'CellCenter',  # default value depends on extraction Type
    )

If `Override` is :py:obj:`False`, then each time that the file is saved, 
its name is suffixed with `_AfterIter<Iteration>`. 
For instance, if `File='signals.cgns'` and `SavePeriod=100`, the first file will be 
`signals_AfterIter100.cgns`, the second file will be `signals_AfterIter200.cgns`, etc.

For 1D data, it may be useful having `SavePeriod` greater than `ExtractionPeriod`. 


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

They are set using attribute **SolverParameters**. This attribute allow overriding default parameters 
specific a solver and that are not plugged into MOLA interface.
Notice that you might check what MOLA has written into the attribute **SolverParameters** after preprocess
in :mola_name:`CONTAINER_WORKFLOW_PARAMETERS`.

Examples are given below for different solvers:

With elsA
=========

.. code-block:: python

    SolverParameters = dict(
        # cfdpb = dict(),
        model = dict(walldistcompute='mininterf_ortho2'),
        numerics = dict(limiter='venkata', viscous_fluxes='5p_cor'),
    )

With SoNICS
===========

.. code-block:: python

    SolverParameters = dict(
        features = ["viscous_flux/vf5p_cor"],
        parameters = dict(pctrad=0.02),
    )

