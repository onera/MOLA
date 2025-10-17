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

============
Known issues
============

-----------------------
Mesh file in ADF format
-----------------------

What's the error
^^^^^^^^^^^^^^^^

If the mesh file that you want to read is not in HDF format but in ADF format, you will have this error message:

.. code-block::

  File "/stck/mola/treelab/v0.4.3/ld_elsA/lib/python3.8/site-packages/treelab/cgns/read_write/h5py2cgns.py", line 52, in load
    f = load_h5(filename)
  File "/stck/mola/treelab/v0.4.3/ld_elsA/lib/python3.8/site-packages/treelab/cgns/read_write/h5py2cgns.py", line 72, in load_h5
    f = h5py.File(filename, permission, track_order=True)
  File "/opt/tools/python/3.8.14-gnu831/lib/python3.8/site-packages/h5py/_hl/files.py", line 533, in __init__
    fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
  File "/opt/tools/python/3.8.14-gnu831/lib/python3.8/site-packages/h5py/_hl/files.py", line 226, in make_fid
    fid = h5f.open(name, flags, fapl=fapl)
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "h5py/h5f.pyx", line 106, in h5py.h5f.open
  OSError: Unable to open file (file signature not found)

Possible workaround
^^^^^^^^^^^^^^^^^^^

Just use the following command to convert your mesh in HDF format:

>>> adf2hdf mesh.cgns

You may also look at the settings of your mesher to export the mesh file directly in the right format.

