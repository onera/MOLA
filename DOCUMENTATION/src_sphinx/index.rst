MOLA documentation
==================

Welcome to **MO**\ dular workf\ **L**\ ows for **A**\ erodynamics (MOLA)
documentation website.

Current documentation version is: ``v1.17``.

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

Major changes with respect to previous version (``v1.16``):

CFD
---
* use of elsA v5.2.03
* corrected major bug involving the restart procedure when time-averaging
* many new examples in `$MOLA/EXAMPLES` and `$MOLA/EXAMPLES_RESTRICTED` directories
* new inlet and outlet Giles conditions including relevant examples
* residuals are now accessible during the simulation in file arrays.cgns


LIFTING-LINE
------------
* DISCLAIMER: LiftingLines files generated with MOLA < v1.17 are not compatible. They shall be regenerated.
* new sweep and dihedral corrections implemented
* enhancements on the collective pitch variations
* included new examples showing lifting-line generation from a blade scan in `$MOLA/EXAMPLES/LIFTING_LINE`
* multiple new and more readable spanwise loads contained in FlowSolution

BODY-FORCE
----------
* the API for body-force applied to internal aerodynamics has changed. See the :ref:`dedicated tutorial <TutorialWorkflowCompressorBodyForce>`.

VPM
---
* use of VPM v0.4 now called VULCAINS
* fixed memory leaks
* multiple lifting-line resolution approach implemented
* enhanced lifting-line circulation subiterations algorithm
* included new examples in `$MOLA/EXAMPLES/VPM`

TreeLab
-------
* Extracted the repository of `TreeLab <https://github.com/Luispain/treelab>`_, which is now a fully independent package
* multi-tab approach, allowing for copying nodes from one tree to another
* new look with selectable themes (light and dark modes)
* fully portable and cross-platform, included in `PyPi <https://pypi.org/project/mola-treelab>`_

More details
------------
* See `here <https://gitlab.onera.net/numerics/mola/-/issues/?sort=created_date&state=closed&milestone_title=v1.17&first_page_size=100>`_ a full list of relevant actions related to v1.17 release


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. rubric:: Footnotes

.. [#f1] Registered code ``IDDN.FR.001.240036.000.S.X.2022.000.31235``
