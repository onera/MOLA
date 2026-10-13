######
SPLEEN 
######

*****************
Short description
*****************

This is an open-access database of experimental data of the flow in a high-speed low-pressure turbine cascade. 
The data have been collected at the von Karman Institute for Fluid Dynamics in the period 2018-2022 
within the H2020 Clean Sky 2 project SPLEEN – Secondary and Leakage Flow Effects in High-Speed Low-Pressure Turbines, 
a project in collaboration with Safran Aircraft Engines.

The experimental database and the geometry are available:

* S. LAVAGNOLI, G. LOPES, L. SIMONASSI, AND A. F. M. TORRE, Spleen - high speed turbine
  cascade – test case database, Sept. 2024.
  https://doi.org/10.5281/zenodo.13712768

For more information on the configuration and experimental data,
see:

* L. SIMONASSI, G. LOPES, S. GENDEBIEN, A. F. M. TORRE, M. PATINIOS, S. LAVAGNOLI,
  N. ZELLER, AND L. PINTAT, An Experimental Test Case for Transonic Low-Pressure Turbines
  – Part I: Rig Design, Instrumentation and Experimental Methodology, American Society of
  Mechanical Engineers Digital Collection, Oct. 2022.

* G. LOPES, L. SIMONASSI, A. F. M. TORRE, M. PATINIOS, AND S. LAVAGNOLI, An Experimental
  Test Case for Transonic Low-Pressure Turbines - Part II: Cascade Aerodynamics at On- and
  Off-Design Reynolds and Mach Numbers, American Society of Mechanical Engineers Digital
  Collection, Oct. 2022


.. figure:: flow_spleen.png
  :width: 100%
  :align: center

  Mach number around the blade



***************
RANS simulation
***************

.. literalinclude:: ../../../../examples/open/workflow/fixed_component/linear_cascade/SPLEEN/run.py
    :language: python

