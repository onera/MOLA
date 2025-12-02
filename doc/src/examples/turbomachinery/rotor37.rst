.. _example-rotor37:
  
########
Rotor 37 
########

*****************
Short description
*****************

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

In the following, the mesh has been generated with Autogrid5 from Cadence. 

.. figure:: //stck/mola/data/open/mesh/rotor37/mesh_view.png
  :width: 70%
  :align: center

  Input mesh generated with Autogrid5

.. figure:: flow_r37.png
  :width: 100%
  :align: center

  Relative Mach number at 90% span

***************
RANS simulation
***************

.. literalinclude:: ../../../../examples/open/workflow/rotating_component/turbomachinery/rotor37/run_sator.py
    :language: python

