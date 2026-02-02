.. _inputs-extractions:

***********
Extractions
***********

Extractions are defined with the workflow attribute **Extractions** as a :class:`list`. 
Each element is a :class:`dict` and corresponds to an extraction.

For each extraction, at least one key is mandatory: Type (:class:`str`). 
Available extraction types are reported below.

Some optional parameters are common to most extraction types. They are listed in :ref:`Common parameters for extractions`.

.. contents:: Table of contents

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

.. hint::

  It is possible to extract an iso-surface on radius with `IsoSurfaceField='Radius'` (or `'radius'` or `'CoordinateR'`), 
  even if radius is not already present in the input file :mola_name:`FILE_INPUT_SOLVER`. 
  The radius will be computed considering that the main axis is X : :math:`r=\sqrt{y^2+z^2}`

  With the :ref:`Workflow turbomachinery` and the :ref:`Workflow linear cascade`, it is also possible to extract an 
  iso-surface on channel height with `IsoSurfaceField='ChannelHeight'`. If at least one extraction on channel height 
  is asked by the user, MOLA automatically computes it during preprocess and add a node 
  `FlowSolution#Height/ChannelHeight` in the file :mola_name:`FILE_INPUT_SOLVER`.

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



