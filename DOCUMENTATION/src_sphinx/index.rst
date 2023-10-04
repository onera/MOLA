MOLA documentation
==================

Welcome to **MO**\ dular workf\ **L**\ ows for **A**\ erodynamics (MOLA)
documentation website.

Current documentation version is: ``Dev``

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
  Commands

Major changes
=============

Major changes with respect to previous version (``v1.15``):

General
-------
* Included mola command lines. See available commands using `mola_available`
* ONERA CentOS 6 machines not supported anymore

CFD
---
* use of elsA v5.2.02
* enhanced unstructured compatibility using `maia <https://numerics.gitlab-pages.onera.net/mesh/maia/1.2/>`_ functions during preprocess (sequential only)
* automatic handling of volume and surface fields time-averaging and restart procedure
* allowed for multi-container behavior (extractions, slices, post-processing)
* Standard workflow now accepts turbomachinery type boundary conditions
* Standard workflow now accepts PyPart splitting (if no overset is done)
* included simple probes extractions

MESHING
-------

* enhanced automatic propeller meshing procedure, now allowing for narrow USF-type monochannel grids

VPM
---
* use of elsA VPM v0.3 (still beta)
* included multiple diffusion methods (DVM is current best choice)
* included velocity perturbation technique (see new NREL5 wind turbine example)
* included enstrophy-based dynamic control of turbulence viscosity
* lifting-line self-induction modeled using particles


Miscellaneous
-------------
* enhanced visualisation techniques (Visu module)
* bugs correction and improved general robustness

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. rubric:: Footnotes

.. [#f1] Registered code ``IDDN.FR.001.240036.000.S.X.2022.000.31235``
