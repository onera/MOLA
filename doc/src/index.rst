##################
MOLA documentation
##################

Welcome to **MO**\ dular workf\ **L**\ ows for **A**\ erodynamics (MOLA)
documentation website.

Current documentation version is: |version|.

.. todo:: Make a link to Treelab documentation

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
   :hidden:

   user_manual/index
   examples/index
   tutorials/index
   developer_manual/index
   changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. rubric:: Footnotes

.. [#f1] Registered code ``IDDN.FR.001.240036.000.S.X.2022.000.31235``