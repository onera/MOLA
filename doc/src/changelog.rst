#########
Changelog
#########

This page contains information about what has changed in each new version of MOLA.

.. To update this page: keep the following order: 💡 New features, 🚀 Feature improvements, 🚧 API changes, 🐞 Fixes, 💻 For developpers

******
v2.0.1 
******

===============
💡 New features
===============

* Allow 2D simulations with SoNICS (automatic if the input mesh is 2D)
* Allow vertex-centered simulation with SoNICS (using ``Numerics['GridLocation']='Vertex'``)

=======================
🚀 Feature improvements
=======================

* Make more robust duplication process

========
🐞 Fixes
========

* Fix a bug for iso-surface with SoNICS for multi-zones domains.
* Fix a bug to allow the copy or move of the run directory.
* Fix `bug #283 <https://gitlab.onera.net/numerics/mola/-/issues/283>`_: 
  PyPart output file writing issue in elsA simulations, 
  specifically when extracting boundary conditions data.
* Fix `bug #296 <https://gitlab.onera.net/numerics/mola/-/issues/296>`_: 
  Incomplete or missing data extraction on a boundary condition.
* Fix `bug #273 <https://gitlab.onera.net/numerics/mola/-/issues/273>`_:
  Correction on force and torque extraction with SoNICS.
* Fix bugs `#288 <https://gitlab.onera.net/numerics/mola/-/issues/288>`_
  and `#292 <https://gitlab.onera.net/numerics/mola/-/issues/292>`_:
  issues to restart a simulation.
* Fix `bug #304 <https://gitlab.onera.net/numerics/mola/-/issues/304>`_:
  Correction on elsa simulation interruption when using 2nd order time marching.

*****************
v2 compared to v1 
*****************

===============
💡 New features
===============

* MOLA may be now used with the solvers SoNICS and Fast  
* Parametric studies are easy to launch with the WorkflowManager 

=======================
🚀 Feature improvements
=======================

* Extractions are more modular and user-friendly (➥ :ref:`inputs-extractions`)
* Error management has been improved for clarity 

==============
🚧 API changes
==============

* refer to sections :doc:`user_manual/workflow/introduction` and :doc:`user_manual/workflow/inputs`.

================================================================
:fas:`person-digging;sd-text-warning` Not yet transfered from v1
================================================================

* Body force method
* 2D airfoils polars
* interfaces for the following low or mid-fidelty tools: VPM, XFoil, BEMT
* Post-processing functions (except the *turbo* interface with the Workflow **turbomachinery** that is already available)

==================
💻 For developpers
==================

* better structure of the source code
* better code quality
* test base with unit, integration and user_case tests
* source code more easily extended in the future
