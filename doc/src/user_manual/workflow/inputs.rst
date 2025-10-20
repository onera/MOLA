###############
Workflow inputs
###############

.. py:currentmodule::  mola.workflow.interface

.. contents:: Table of contents

*********************************
Read and transform the input mesh 
*********************************

The attribute `RawMeshComponents` allows providing one or several mesh components to MOLA. 
It is a list of dict, each one corresponding to one provided mesh.

A mesh is given as a dict of mandatory and optional parameters.

The following parameters are mandatory:

* ``Name`` (str):
  Name of the component, which defines also the name of the CGNS Base.

* ``Source`` (str, or a treelab Tree, Base or Zone):
  Name of the mesh file. Source can also be directly a Tree, Base or Zone read by treelab.


The following parameters are optional and specify how the mesh should be read:

* ``Mesher`` (str):
  Name of the tool used to generate the Mesh. Available values are: 

  #. `None` or `'default'`: in that case, nothing is done.

  #. `'autogrid'`: make standard operations to clean and rotate a mesh generated with Autogrid5.

* ``Unit`` (str):
  Unit for mesh coordinates, to convert to meters if needed. 
  Available units are: 'm', 'dm', 'cm', 'mm', 'inches'.
  By default 'm'.


The following optional parameters can be used to modify or add elements in the read mesh:

* ``Families`` (list of dict):
  Each dict corresponds to an operation to create a new family (of boundary conditions) in the mesh.
  It could be useful to complete a simple mesh, but it is not recommended using this 
  for a complex multi-block mesh.

  One should prescribe the ``Name`` of the new family to create and the `Location` of the targeted boundary, 
  by using one of these keywords: `'imin'`, `'imax'`, `'jmin'`, `'jmax'`,  `'kmin'`, `'kmax'` and `'remaining'`.

  For instance:

  .. code-block:: python

    Families = [
        dict(Name='Wall', Location='kmin'),  # tag all BCs that correspond to k=0 (for all zones) with Family 'Wall'
        dict(Name='Farfield', Location='remaining'),  # tag all remaining BCs with Family 'Farfield'
    ]

  .. warning::

    It is also possible to use the argument **planeTag** (possibly with a **tolerance**):

    >>> Families = [dict(Name='SymmetryPlane', planeTag='planeXZ', tolerance=1e-8)]

    Careful, the feature has not been tested yet.

* ``Positioning`` (list of dict):
  Each dict corresponds to an operation to apply on transformation on the mesh.
  Two kinds of operation are available for now:

  * Scaling (which is redundant with the `Unit` parameter):

    >>> Positioning=[dict(Type='Scale', Scale=1e-3)]

  * Translation and rotation, by giving an initial frame of reference and a requested frame:

    .. code-block:: python  
        
        Positioning=[
            dict(
                Type='TranslationAndRotation',
                InitialFrame   = dict(Point=[0,0,0], Axis1=[0,0,1], Axis2=[1,0,0], Axis3=[0,1,0]),
                RequestedFrame = dict(Point=[0,0,0], Axis1=[1,0,0], Axis2=[0,1,0], Axis3=[0,0,1]),
                )
        ]

* ``Connection`` (list of dict):
  Each dict corresponds to an operation to add grid connectivities in the mesh.
  The `Type` of connectivity could be `Match`, `PeriodicMatch` or `NearMatch`. 
  A specific absolute `Tolerance` can be given.

  >>> Connection = [dict(Type='Match', Tolerance=1e-6)]

  For `PeriodicMatch` connectivity, a translation vector `Translation` must be given for linear periodicity:

  >>> Connection = [dict(Type='PeriodicMatch', Translation=[0,1,0])]

  A rotation vector `RotationAngle` must be given for annular configurations, with angles defined in degrees:
  
  >>> Connection = [dict(Type='PeriodicMatch', RotationAngle=[45., 0., 0.])]

  For `NearMatch` connectivity, a `Ratio` (int) should be given (2 by defaults).

  >>> Connection = [dict(Type='NearMatch', Ratio=2)]


* ``DefaultToleranceForConnection`` (float): 
  default (absolute) tolerance used for each element in ``Connection``, if ``Tolerance`` is not given. 
  The default value is 1e-8.




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

.. _FlowGenerator-Internal:

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

  .. note:: 
    
    The correspondence between these names and parameters in elsA may be found on this page: 
    https://elsa-doc.onera.fr/MU_tuto/latest/MU-98057/Textes/turbmods.html#nasa-named-turbulence-models-mapping

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

The main numerical parameters are common to different solvers and are 
defined in the `Numerics` attribute of Workflow. 
Keep in mind that default values of these parameters could depend on the chosen applicative Workflow.

`Scheme` (str, optional):
    Spatial scheme. Available schemes are: 'Jameson', 'Roe', 'ausm+'.
    By default 'Jameson' (for the basic Workflow). 

`TimeMarching` (str, optional):
    Type of simulation, available choices are: 'Steady', 'Unsteady'.
    By default 'Steady'.

`NumberOfIterations` (int, optional): by default 10000

`MinimumNumberOfIterations` (int, optional):
    Number of iterations that will be done in all cases, 
    even if convergence criteria have been already reached.
    By default 1000.

`IterationAtInitialState` (int, optional): by default 1

`TimeAtInitialState` (float, optional): by default 0.0

`TimeMarchingOrder` (int, optional): by default 2

`TimeStep` (float, optional): Useful only for unsteady simulation.

`CFL` (float or dict, optional):
    CFL number, by default 10.0.
    It could be a scalar or a linear ramp given as a dict. For example:

    >>> CFL = dict(EndIteration=300, StartValue=1., EndValue=30.)

    defines a ramp with CFL=1 at iteration 1 (could be modified with `StartIteration`)
    until CFL=30 at iteration 300.

For other parameters that are specific to the solver, it is still possible to use 
the workflow attribute **SolverParameters**, see :ref:`Parameters specific to the solver`. 


**************
Initialization
**************

Different initialization methods can be used to generate the 3D field to start the simulation.
The `Initialization` attribute of the Workflow is a dictionnary with the foloowing parameters:

``Method`` (str, optional):
    Available methods are: 

    * `'uniform'`: initialize flow with reference values as computed 
        from **Fluid**, **Flow** and **Turbulence** attributes.

    * `'copy'`: initialize flow by copying the flow in the file given by **Source**.
        Both meshes must be exactly the same.
        
    * `'interpolate'`: initialize flow by interpolating the flow from the file given by **Source**.

    By default 'uniform'

`Source` (str or a trelab Tree, Base, Zone, optional):
    Source mesh for `copy` or `interpolate` methods, given as a file name or as a treelab Tree.

`SourceContainer` (str, optional):
    Container to consider in the source mesh, by default 'FlowSolution#Init'

`ComputeWallDistanceAtPreprocess` (bool, optional):
    If True, compute distances to walls during preprocess.
    By default False


.. _inputs-extractions:

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

    dict(Type = 'BC', Source = 'WallViscous', Fields = ['Pressure', 'SkinFriction']) 

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

The attribute `SplittingAndDistribution` sets how to split mesh and how the 
computational domain will be distributed among processors.

To use Cassiopée:

.. code-block:: python 

    SplittingAndDistribution = dict(
        Strategy='AtPreprocess',
        Splitter='Cassiopee', 
        Distributor='Cassiopee', 
        ComponentsToSplit='all', # or None or ['first', 'second'...]
        NumberOfParts=4,  # If not given, based on RunManagement['NumberOfProcessors']
        )


To use PyPart:

.. code-block:: python 

    SplittingAndDistribution=dict(Strategy='AtComputation', Splitter='PyPart')


To use Maia:

.. code-block:: python 

    SplittingAndDistribution=dict(Strategy='AtComputation', Splitter='maia')




********************
Convergence criteria
********************

The attribute ``ConvergenceCriteria`` is a list of dict that allow stopping the simulation before 
that all iterations have been performed (set with `Numerics['NumberOfIterations']`).

Each element of the list is a dictionary representing one convergence criterion, defined by:

#. an `ExtractionName` (str), being the name of the 1D extraction to monitor.

#. a `Variable` (str) to monitor into that extraction. The variable has been extracted by the user 
   or by default (depending on the Workflow) for the extraction named `ExtractionName`.
   It is also possible to compute a sliding statistic on an existing variable by using 
   a prefix. For instance, `std-ForceX` is the standard deviation of `ForceX` (estimating an absolute convergence error), 
   and `rsd-MassFlow` is the relative standard deviation of `MassFlow` 
   (standard deviation divided by average, so estimating relative error in percent).

#. a `Threshold` (float) below which convergence is reached.

For instance, if the following criterion is defined:

.. code-block:: python 

    ConvergenceCriteria = [
        dict(ExtractionName='Outflow', Variable='rsd-MassFlow', Threshold=1e-4)
    ]

Then the simulation will end either when the maximum iteration is reached, or before when the `MassFlow` 
has converged within a relative error of 1e-4, that's to say 0.01%.

A convergence criterion could be necessary (if `Necessary=True`) and/or sufficient (`Sufficient=True`). 
To reach convergence, all necessary criteria (if they are defined) should be reached 
and at least one sufficient criterion (if it exists) should be reached.

By default, a criterion is sufficient but not necessary. In other words, if three criteria are defined 
without specifying the keys `Necessary` and `Sufficient`, the simulation stops as soon as one of these 
criteria is reached.



*****************************
Information on job submission
*****************************

The Workflow attribute `RunManagement` handles job submission. 
It is a dict with the following parameters:

`JobName` (str, optional):
    Name of the job (useful only for using a Scheduler), by default 'mola'

`RunDirectory` (str or Path from pathlib package, optional):
    Path where the simulation will be done,
    by default '.' (simulation is prepared in the current directory)

`NumberOfProcessors` (int, optional):
    Number of processors used to run the simulation with MPI, by default `MPI.COMM_WORLD.Get_size()`

`NumberOfThreads` (int, optional):
    Number of threads used to run the simulation with OpenMP, by default 1.

    .. note:: only useful with `Solver='fast'`.

`Machine` (str, optional):
    Name of the machine where the simulation will be run.
    **RunDirectory** is relative to that machine.
    If not given, an attempt to guess the destination machine will be done
    using **RunDirectory**, the current directory and environment setting.
    If the machine cannot be guessed, localhost is taken by default.

`User` (str, optional):
    Username on the destination **Machine**, by default the same user than currently on localhost.

`TimeLimit` (Union[str, float], optional):
    Time limit for the simulation, either in seconds (:class:`float`) or as a :class:`str`
    like '00:30:00' (30min), '15:00' (15min), '1-10:00:00' (34h).
    The default value depends on the **Machine** and environment parameters.

`QuitMarginBeforeTimeOutInSeconds` (int, optional):
    Margin in seconds before quitting the simulation, by default 300.
    When the simulation has run for **TimeLimit** - **QuitMarginBeforeTimeOutInSeconds**,
    it won't make new iterations and the simulation try ending safely performing final extractions.
    It will be automatically submitted again.

`LauncherCommand` (str, optional):
    Command that will be executed after preprocess to run the simulation (on the destination **Machine**).
    If not providing, the default value 'auto' corresponds to:

        * with `Scheduler='bash'`:  cd <RunDirectory>; sbatch :mola_name:`FILE_JOB`

        * with `Scheduler='SLURM'`: cd <RunDirectory>; sbatch :mola_name:`FILE_JOB`

`FilesAndDirectories` (list, optional):
    Files and directories to copy in **RunDirectory**, by default []

`mola_target_path` (str, optional):
    When the simulation is launched on a remote **Machine** that has no acces to the local MOLA installation directory, 
    you can provide the path for MOLA sources on this remote **Machine**.

`Scheduler` (str, optional):
    Job scheduler, like SLURM, to use to run the simulation.
    The default value depends on the **Machine** and environment parameters.

`AER` (str, optional):
    AER number for simulation on sator

`RemovePreviousRunDirectory` (bool, optional):
    Only used for a simulation on a remote machine. If True, remove the previous `RunDirectory` before
    preprocessing the case. Default value is False.




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

