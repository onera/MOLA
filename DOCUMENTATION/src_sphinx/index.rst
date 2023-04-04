MOLA documentation
==================

Welcome to **MO**\ dular workf\ **L**\ ows for **A**\ erodynamics (MOLA)
documentation website.

Current documentation version is: ``v1.15``

Please note the `GitLab page <https://gitlab.onera.net/numerics/mola>`_ of MOLA
and its associated `Issues <https://gitlab.onera.net/numerics/mola/-/issues>`_ section where
you can make suggestions or report bugs.


MOLA is an `ONERA <https://www.onera.fr>`_ code [#f1]_
that implements user-level workflows and tools for aerodynamic analysis. These
tools are essentially interfaces of multiple simulation techniques such as
computational fluid dynamics (CFD), blade-element momentum theory (BEMT) and
vortex particle method (VPM).

Some CFD worfklows include automatic mesh generation, and all of them include
preprocessing using `Cassiopee <https://elsa.onera.fr/Cassiopee>`_ and computation
using `elsA <https://elsa.onera.fr>`_ solver.


.. toctree::
  :maxdepth: 1
  :caption: Contents:

  StarterGuide
  Tutorials
  Examples
  Modules

Major changes
=============

Major changes with respect to previous version (``v1.14``):

General
-------
* MOLA is now released using `GNU GPL v3 open-source license <http://www.gnu.org/licenses/>`_

CFD
---
* use of elsA v5.1.03
* cell-volume correction on lifting-line bodyforce method
* major enhancements on internal aerodynamics bodyforce method
* automatic turbomachinery post-processing using *turbo* module
* dynamic overset suitable for unsteady rotor computations
* chorochronic simulations on Workflow Compressor and ORAS
* expansion of existing Airfoil Polars
* automatic restart of any CFD run upon timeout
* injection of user-defined radial profiles in turbomachinery simulations

Miscellaneous
-------------
* Release of Vortex Particle Method (VPM) stable version v0.2, including multiple examples
* major enhancements of TreeLab 
* enhanced visualisation techniques (Visu module)
* new tools accessible from MOLA environment: turbo, ErsatZ, maia
* bugs correction and improved general robustness

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. rubric:: Footnotes

.. [#f1] Registered code ``IDDN.FR.001.240036.000.S.X.2022.000.31235``
