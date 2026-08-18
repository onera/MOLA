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
* Write automatically a postprocess script to gather results after using a WorkflowManager (depending on workflow)
* Select automatically the adapted Flow Generator from the given input data in **Flow**.
* Perform automatic second order restart in the context of an unsteady 2nd order simulation with elsA.

=======================
🚀 Feature improvements
=======================

* Make more robust duplication process
* Allow modifying stage parameters for elsA (for MixingPlane, UnsteadyRotorStatorInterface and ChorochronicInterface)
* Optimize WorkflowManager

========
🐞 Fixes
========

* Fix `bug #326 <https://gitlab.onera.net/numerics/mola/-/work_items/326>`_: 
  correction for chorochronic simulations with elsA.
* Fix a bug for iso-surface with SoNICS for multi-zones domains.
* Fix a bug to allow the copy or move of the run directory.
* Fix `bug #283 <https://gitlab.onera.net/numerics/mola/-/work_items/283>`_: 
  PyPart output file writing issue in elsA simulations, 
  specifically when extracting boundary conditions data.
* Fix `bug #296 <https://gitlab.onera.net/numerics/mola/-/work_items/296>`_: 
  Incomplete or missing data extraction on a boundary condition.
* Fix `bug #273 <https://gitlab.onera.net/numerics/mola/-/work_items/273>`_:
  Correction on force and torque extraction with SoNICS.
* Fix bug `#286 <https://gitlab.onera.net/numerics/mola/-/work_items/286>`_:
  issues to write a tree in parallel with PyPart.
* Fix bugs `#288 <https://gitlab.onera.net/numerics/mola/-/work_items/288>`_
  and `#292 <https://gitlab.onera.net/numerics/mola/-/work_items/292>`_:
  issues to restart a simulation.
* Fix `bug #304 <https://gitlab.onera.net/numerics/mola/-/work_items/304>`_:
  Correction on elsa simulation interruption when using 2nd order time marching.
* Fix bugs in WorkflowManager
* Fix bugs in postprocess interface for turbomachinery

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
