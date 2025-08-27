.. _how-to-debug:

####################
How to debug ? 🆘 🔧
####################

Although it is not possible to anticipate all causes of errors, 
this page aims to give some advice to debug your case with MOLA.

=============================
Switch to debug verbose level
=============================

If you launch your MOLA Python script with:

>>> python prepare.py

the verbose level will be 'INFO'. That means that all messages with at least the level 'INFO'
will be written in the terminal (including levels 'WARNING', 'ERROR' and 'CRITICAL').

You can lower the verbose level to 'DEBUG' with the following option:

>>> python prepare.py -v DEBUG

New information starting with 'DEBUG: ' will be written in the terminal.
It allows the user checking more operations and computed values.


===================================
Write the tree during preprocessing
===================================

When you are running the Workflow method `prepare()`, 
in fact you are applying the following steps:

.. literalinclude:: ../../../../src/mola/workflow/workflow.py
    :language: python
    :pyobject: Workflow.prepare

Thus, if during preprocessing you encounter a bug when boundary conditions are set,
you can write the tree just before this step to try to understand what is wrong:

.. code-block:: python

    workflow = Workflow(...)
    workflow.prepare_job()
    workflow.process_mesh()
    workflow.process_overset()
    workflow.compute_flow_and_turbulence()
    workflow.set_motion()
    workflow.tree.save('debug.cgns')  # save the tree to be able to inspect it
    workflow.set_boundary_conditions()  # this line raises an error

.. note:: 

    You may write the tree with different methods:

    * `workflow.tree.save('debug.cgns')` writes the tree sequentially with treelab.
    * `workflow.write_tree('debug.cgns')` writes the tree depending on tree state. 
      MOLA chooses if it is better to write it with treelab, Cassiopée or Maia.
    * `C.convertPyTree2File(workflow.tree, 'debug.cgns')` writes the tree with Cassiopée (after `import Converter.Pytree as C`).
    * `maia.pytree.dist_tree_to_file(workflow.tree, 'debug.cgns', comm)` writes the distributed tree with Maia.
      (after `import maia` and `from mpi4py.MPI import COMM_WORLD as comm`).
      You need to be sure that the tree is effectively distributed at this stage, 
      and that probably won't be the case. Hence, this method is not recommended.

