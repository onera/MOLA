#############
Isolated wing 
#############

*****************
Short description
*****************

A simple subsonic wing.

.. figure:: //stck/data/open/mesh/isolated_wing/isolated_wing_mesh.png
  :width: 70%
  :align: center

  Input mesh - Full domain on the left, zoom on the wing on the right


.. .. figure:: isolated_wind_CL_convergence_comparison.png
..   :width: 70%
..   :align: center

..   Convergence of the lift coefficient


***************
RANS simulation
***************

.. literalinclude:: ../../../../examples/open/workflow/fixed_component/airplane/isolated_wing/run_sator.py
    :language: python

