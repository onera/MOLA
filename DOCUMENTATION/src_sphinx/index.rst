MOLA documentation
==================

Welcome to **MO**\ dular workf\ **L**\ ows for **A**\ erodynamics (MOLA)
documentation website.

Current documentation version is: ``v1.18``.

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

Major changes with respect to previous version (``v1.17``):

CFD
---
* fixed bug impacting Menter-Langtry turbulence model limiters
* included ``injrot`` elsA condition
* allow using 2D map for ``OutflowPressure`` (``outpres``) and ``Farfield`` (``nref``) boundary conditions
* fixed bug on turbulence cutoff in WorfkflowAirfoil
* bug fixes on unsteady masking


LIFTING-LINE
------------
* fixed minor bug on computation of lifting-line length

VPM
---
* use of VULCAINS v0.5
* new documentation
* new factorized user interface
* new enstrophy-based stabilizing algorithm
* new examples now available on Juno and Sator ONERA machines


TreeLab
-------
* use of v0.2.0
* allows for reading children of nodes with links
* simultaneous modification of node value (or name) for multiple selected nodes
* simultaneous modification of numpy values for multiple cells selected in table
* new reimplmentation of merge function

More details
------------
* See `here <https://gitlab.onera.net/numerics/mola/-/issues/?sort=created_date&state=closed&milestone_title=v1.18&first_page_size=100>`_ a full list of relevant actions related to v1.18 release


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. rubric:: Footnotes

.. [#f1] Registered code ``IDDN.FR.001.240036.000.S.X.2022.000.31235``
