##########
Deployment
##########

************
Dependencies
************

Depending on the solver you want to use.

********************
Installation of MOLA
********************

.. warning::

  There could be a conflict regarding the Python package h5py. Indeed, one of MOLA dependencies (Maia)
  use a non-official version of this package. As Maia is part of the solver environment, the best 
  solution is not to install h5py to be sure to use the one used by Maia.

MOLA depends on the open-source software `treelab <https://github.com/Luispain/treelab>`_. 
To install it, follow these steps:

#. Choose the path to install treelab (one repository by machine **and** solver): 
   
   .. code-block:: 
     
     export TREELAB_INSTALL_PATH=<YOUR_CHOICE>

#. Source elsA: 

   .. code-block:: 
    
     source <ELSA_INTALL_PATH>/.env_elsA

#. Install treelab with pip. The following line are recommanded to prevent conflicts between packages: 

   .. code-block:: 
    
     pip install --no-deps --prefix=$TREELAB_INSTALL_PATH mola-treelab>=0.4.4
     pip install --prefix=$TREELAB_INSTALL_PATH qtvscodestyle
     

Then, install MOLA itself.

* Source the solver environment.

* get MOLA sources: 

  .. code-block::

      git clone https://gitlab.onera.net/numerics/mola.git  # for ONERA users
      git clone https://github.com/onera/MOLA.git  # for others

  Or download directly the raw sources.

* Append the `src` directory in MOLA sources to you PYTHONPATH.

* Complete the environment directory (`src/mola/env/`). You can mimic what is done in `src/mola/env/template`.

