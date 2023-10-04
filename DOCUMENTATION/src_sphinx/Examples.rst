.. _Examples:

Examples
========

This page summarizes all examples that can be found on ``EXAMPLES`` directory.

BEMT
----

Design of a rotor in hover
**************************

.. code-block:: bash

   $MOLA/EXAMPLES/BEMT/ROTOR_HOVER_DESIGN

**Short description:** computation of the geometrical laws of the blade
(and associated ``LiftingLine.cgns`` file)  using Minimum Induced Loss theory
based on Blade Element Momentum Theory (BEMT), for a single condition

.. figure:: ../../EXAMPLES/BEMT/ROTOR_HOVER_DESIGN/rotor_laws.svg
    :width: 60%
    :align: center

    Geometrical laws of the blade


Optimum design of a propeller
*****************************

.. code-block:: bash

   $MOLA/EXAMPLES/BEMT/PROPELLER_DESIGN

**Short description:** computation of the geometrical laws of the blade
(and associated ``LiftingLine.cgns`` file)  using Minimum Induced Loss theory
based on Blade Element Momentum Theory (BEMT), using an enveloppe of conditions 
(first a nominal condition and second a maximum thrust constraint).

Analysis of a propeller
***********************

.. code-block:: bash

   $MOLA/EXAMPLES/BEMT/PROPELLER_ANALYSIS

**Short description:** make a BEMT computation of a propeller.

Airfoil Polars computation using XFoil
**************************************

.. code-block:: bash

   $MOLA/EXAMPLES/BEMT/POLARS_XFOIL

**Short description:** generate a ``Polars.cgns`` file of the flow around an airoil 
using XFoil solver.

MESHING
-------

Modification of an airfoil geometry
***********************************

.. code-block:: bash

   $MOLA/EXAMPLES/MESHING/AIRFOIL_DESIGN/

**Short description:** modify an existing airfoil using new airfoil geometrical
characteristics

.. figure:: ../../EXAMPLES/MESHING/AIRFOIL_DESIGN/ModifyAirfoil.svg
    :width: 80%
    :align: center

    Comparison of original airfoil (gray) and modified (black), including their
    mean camber lines


Generation of a O-H mesh surface
********************************

.. code-block:: bash

   $MOLA/EXAMPLES/MESHING/PERIODIC_O-H

**Short description:** create a O-H surface grid with arbitrary orientation

.. figure:: ../../EXAMPLES/MESHING/PERIODIC_O-H/mesh3D.png
    :width: 80%
    :align: center

    OH grid around an airfoil, with imposed 3D boundaries and projection onto
    a cylinder surface

WORKFLOW COMPRESSOR
-------------------

.. _Rotor37:

Rotor 37
********

.. code-block:: bash

    $MOLA/EXAMPLES/WORKFLOW_COMPRESSOR/rotor37_SingleCase/


**Short description**:

The NASA 37 transonic rotor is a well known turbomachinery open test case.
The rotor has 36 blades and a nominal speed of 17188.7 rpm.
This test case is interesting to evaluate the robustness of a CFD solver since
it presents a shock-wave/boundary-layer interaction leading to a flow separation.

At design point, the mass flow rate is 20.5114 kg/s, the stagnation pressure ratio is 2.106,
and the polytropic efficiency is 0.889.

For more information on the configuration and experimental data,
see:

 * Agard-AR-355 , "CFD Validation for Propulsion System Components", May 1998:
   https://apps.dtic.mil/sti/pdfs/ADA349027.pdf

The mesh is very light (around 500 000 cells), which allow quick tests on few CPU.

.. figure:: ../../EXAMPLES/WORKFLOW_COMPRESSOR/rotor37_SingleCase/flow_r37.png
    :width: 80%
    :align: center

    flow around the rotor37 blade


Rotor 37 - iso-speed line
*************************

.. code-block:: bash

    $MOLA/EXAMPLES/WORKFLOW_COMPRESSOR/rotor37_IsoSpeedLine/

.. figure:: ../../EXAMPLES/WORKFLOW_COMPRESSOR/rotor37_IsoSpeedLine/isoSpeedLine_with_annotation.png
    :width: 80%
    :align: center

    Performance on the iso-speed line

**Short description**:

This case is identical to :ref:`Rotor37`, except that several operating points
are simulated for the design rotational speed, by varying the outflow condition.



LMFA linear cascade
*******************

.. code-block:: bash

    $MOLA/EXAMPLES/WORKFLOW_COMPRESSOR/LMFAcascade_NACA65009/

.. figure:: ../../EXAMPLES/WORKFLOW_COMPRESSOR/LMFAcascade_NACA65009/flow_lmfa.png
    :width: 80%
    :align: center

    LMFAcascade_NACA65009 flow

**Short description**:

This is a linear cascade of NACA 65-009 profiles, previously installed in
LMFA facilities.
The width of the domain is 0.134m, with a periodicity by translation.
The blade leading edge angle is 54.31 degrees and the flow incidence in the
example is 4 degrees.
The blade chord-based Reynolds number is :math:`3.8 \times 10^5`.

The mesh has around 1.7 million cells.

An experimental reference for this configuration is:

* Zambonini, G., Ottavy, X., and Kriegseis, J. (March 22, 2017). "Corner Separation Dynamics in a Linear Compressor Cascade." ASME. J. Fluids Eng. June 2017; 139(6): 061101. https://doi.org/10.1115/1.4035876


WORKFLOW AIRFOIL
----------------

Very light single case
**********************

.. code-block:: bash

   $MOLA/EXAMPLES/WORKFLOW_AIRFOIL/LIGHT_SINGLE_CASE

**Short description:** very light case of a 2D flow computation around an airfoil
showing the main steps of a MOLA computation using elsA, from mesh construction
up to simple post-processed ``OUTPUT`` files. This case is very light, as it can
run in a local machine.

.. figure:: ../../EXAMPLES/WORKFLOW_AIRFOIL/LIGHT_SINGLE_CASE/flow_airfoil.png
    :width: 80%
    :align: center

    Contour of Mach number around NACA0012 airfoil


Airfoil Polar computation using light mesh
******************************************

.. code-block:: bash

   $MOLA/EXAMPLES/WORKFLOW_AIRFOIL/LIGHT_POLAR

**Short description:** this example is employed in Tutorial :ref:`AirfoilPolars`.

.. figure:: FIGURES/PolarsCL_OA309_original.svg
    :width: 80%
    :align: center

    :math:`c_L(\alpha)` of around OA309 airfoil



.. WORKFLOW AEROTHERMAL COUPLING
.. -----------------------------

.. Channel with two heated walls
.. *****************************

.. .. code-block:: bash

..     $MOLA/EXAMPLES/WORKFLOW_AEROTHERMAL_COUPLING/channel_2HeatedWalls_structured/

.. .. figure:: ../../EXAMPLES/WORKFLOW_AEROTHERMAL_COUPLING/channel_2HeatedWalls_structured/Temperature.png
..     :width: 100%
..     :align: center

..     Temperature inside the flow and the solid (top and bottom walls)

.. **Short description**

.. This test case is a 2D (one cell in Z direction) flow channel with heated walls on both sides.
.. elsA is used for the fluid domain (structured mesh), and Zset is used for the solid domain.

.. Upstream the heated walls, walls are adiabatic and viscous.
.. For the heated walls, a constant temperature (1500K for the bottom wall, 1300K
.. for the top wall) is imposed for the boudary conditions of the solid domain
.. (other than interfaces with the fluid domain).
.. At the interfaces between the fluid and the solid domains, the coupling is done
.. with a Dirichlet condition on the fluid side and a Robin condition on the solid side.

.. Correspondance between elsA and Zset Families :

.. ==========   =======
.. elsA         Zset
.. ==========   =======
.. BottomWall   nord
.. TopWall      sud1
.. ==========   =======

WORKFLOW STANDARD
-----------------

.. _LightWing:

Light wing case
***************

.. code-block:: bash

   $MOLA/EXAMPLES/WORKFLOW_STANDARD/LIGHT_WING


**Short description:** light case of the 3D flow computation around a wing
showing the main steps of a MOLA computation using elsA, from mesh construction
up to simple post-processed ``OUTPUT`` files. This case is very light, as it can
run in a local machine. Thus, mesh is *VERY* coarse and must be refined for
practical usage. This case can be used for rapid testing of MOLA functionalities.

.. figure:: ../../EXAMPLES/WORKFLOW_STANDARD/LIGHT_WING/flow_wing.png
    :width: 80%
    :align: center

    LIGHT_WING

Light wing case (overset)
*************************

.. code-block:: bash

   $MOLA/EXAMPLES/WORKFLOW_STANDARD/LIGHT_OVERSET


**Short description:** This case is identical to :ref:`LightWing`, except that
a cartesian octree-type grid is employed around a bodyfitted mesh component around
the wing. This small example simply shows an overset type (a.k.a. chimera technique)
preprocessing. Mesh is *VERY* coarse and must be refined for practical usage.

.. figure:: ../../EXAMPLES/WORKFLOW_STANDARD/LIGHT_OVERSET/flow_wing_ovt.png
    :width: 80%
    :align: center

    LIGHT_OVERSET


Light Propeller using Bodyforce
*******************************

.. code-block:: bash

   $MOLA/EXAMPLES/WORKFLOW_STANDARD/LIGHT_BODYFORCE


**Short description:** Light case of the CFD computation of a propeller using
the Bodyforce Method. Mesh is *VERY* coarse and must be refined for practical
usage. Only an octree grid is employed, with no overset components.

.. figure:: ../../EXAMPLES/WORKFLOW_STANDARD/LIGHT_BODYFORCE/flow_bfm.png
    :width: 80%
    :align: center

    LIGHT_BODYFORCE flow


Propeller using Bodyforce and an Overset mesh refinement technique
******************************************************************

.. code-block:: bash

   $MOLA/EXAMPLES/WORKFLOW_STANDARD/OVERSET_BODYFORCE


**Short description:** CFD computation of a propeller using
the Bodyforce Method using a local refinment mesh using the overset mesh technique.

.. figure:: ../../EXAMPLES/WORKFLOW_STANDARD/OVERSET_BODYFORCE/flow_bfm_ovt.png
    :width: 80%
    :align: center

    OVERSET_BODYFORCE flow

Aircraft components with several propellers and rotors
******************************************************

.. code-block:: bash

    $MOLA/EXAMPLES/WORKFLOW_STANDARD/HEAVY_OVERSET_BODYFORCE

.. code-block:: bash

    $MOLASATOR/EXAMPLES/WORKFLOW_STANDARD/HEAVY_OVERSET_BODYFORCE

**Short description:** Simulation of an aircraft represented
by only two solids (wing and horizontal stabilizer) which includes a propulsive
propeller on the wing-tip and two rotors for hovering. Only half configuration is
simulated.

.. figure:: ../../EXAMPLES/WORKFLOW_STANDARD/HEAVY_OVERSET_BODYFORCE/flow_bfm_hvy.png
    :width: 80%
    :align: center

    View of two slices of *MomentumX* including the solid walls and bodyforce
    disks

Light Helicopter Rotor
**********************

.. code-block:: bash

    $MOLA/EXAMPLES/WORKFLOW_STANDARD/LIGHT_ROTOR


.. figure:: ../../EXAMPLES/WORKFLOW_STANDARD/LIGHT_ROTOR/FRAMES/frame014400.png
    :width: 80%
    :align: center

    Simulation of the HVAB rotor using MOLA standard workflow


**Short description:** Simulation of a light rotor of a helicopter using unsteady
overset technique. This uses a very coarse mesh.

WORKFLOW PROPELLER
------------------

HAD-1 Propeller
***************

.. code-block:: bash

    $MOLA/EXAMPLES/WORKFLOW_PROPELLER/HAD-1

.. code-block:: bash

    $MOLASATOR/EXAMPLES/WORKFLOW_PROPELLER/HAD-1

.. figure:: ../../EXAMPLES/WORKFLOW_PROPELLER/HAD-1/flow_had1.png
    :width: 80%
    :align: center

    HAD-1 flow

**Short description:** Simulation of HAD-1 propeller in axial flight conditions.
This case features automatic full-match grid generation.
The input data for grid generation are the sections (airfoils) of the propeller
and the spinner profile curve.

Own designed propeller
**********************

.. code-block:: bash

    $MOLA/EXAMPLES/WORKFLOW_PROPELLER/BLADE_NACA_AIRFOIL

.. code-block:: bash

    $MOLASATOR/EXAMPLES/WORKFLOW_PROPELLER/BLADE_NACA_AIRFOIL

.. figure:: ../../EXAMPLES/WORKFLOW_PROPELLER/BLADE_NACA_AIRFOIL/flow_bladenaca.png
    :width: 80%
    :align: center

    BLADE_NACA_AIRFOIL flow


**Short description:** Simulation of a totally custom propeller in axial flight
conditions.
The case features automatic full-match grid generation.
Blade geometry can be defined either by geometrical laws, an existing LiftingLine,
or by passing sections interpolation.
The spinner profile is automatically generated using geometrical parameters.


WORKFLOW ORAS
-------------

ORAS case
*********

.. code-block:: bash

    $MOLA/EXAMPLES/WORKFLOW_ORAS/USF_NEXTAIR_SE


.. code-block:: bash

    $MOLASATOR/EXAMPLES/WORKFLOW_ORAS/USF_NEXTAIR_SE

.. figure:: ../../EXAMPLES/WORKFLOW_ORAS/USF_NEXTAIR_SE/flow_usf.png
    :width: 80%
    :align: center

    USF_NEXTAIR_SE flow


**Short description:** This example presents an Open Rotor and Stator (ORAS)
configuration for steady RANS computations with mixing-plane.



VPM
---

Several VPM examples of wings, rotors and propellers are available here:

.. code-block:: bash

    $MOLA/EXAMPLES/VPM/LIFTING_LINE

.. figure:: ../../EXAMPLES/VPM/LIFTING_LINE/ROTORS/KDE_QuadRotor/flow_kde_quad.png
    :width: 80%
    :align: center

    Flow around the KDE quadcopter drone