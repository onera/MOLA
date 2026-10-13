###########
Environment
###########

Let's assume that MOLA is installed on your machine at path *<MOLA_ROOT_PATH>*. 

.. hint::

    This *<MOLA_ROOT_PATH>* may be something like `/home/user/mola/src/` (if you got the sources)
    or `/home/user/mola/dist/lib/python3.12/site-packages/` (if it was deployed with pip).
    In all cases, in the following, *<MOLA_ROOT_PATH>* refers to the directory where 
    the Python package "mola" will be found during a Python import.

.. code::  
    
    source <MOLA_ROOT_PATH>/mola/env/<NETWORK>/env.sh <SOLVER>

*<NETWORK>* is the name of the network where MOLA has been installed (e.g. `onera`).
*<SOLVER>* is the name of the chosen solver for your computation (note that this name is case-insensitive). 
It is mandatory to choose it when loading the environment because each solver may have different requirements.

For instance, on ONERA network and for a simulation with elsA, you would source:

.. code-block:: bash
    
    source /stck/mola/v2.0.0/mola/env/onera/env.sh elsa

This command finds on which machine of the network your are working on, and source the corresponding environment file. 
The previous line should have the following output in the console:

.. code-block:: bash

    source /stck/mola/v2.0.0/mola/env/onera/ld/elsa.sh

That line could also have been sourced directly, but hence there is also a dependance to the machine (e.g. `ld` here).

.. hint::

  For ONERA users, there is also a shortcut here : 
  
  .. code-block:: bash
  
    source /stck/mola/v2.0.0/env_mola.sh <SOLVER>


.. important::

  If you encounter problems for sourcing the environment, this may be due to the
  use of an incompatible *bashrc* file. Please retry using a bashrc file with
  no module loadings or hard environment settings such as:

  .. code-block:: bash

      mv ~/.bashrc ~/.bashrcBACKUP && cp /stck/lbernard/.bashrc ~/.bashrc

  it is strongly recommended to have an **empty bash user profile**
  in SATOR. The reason is that a user profile with strong modifications with
  respect to the default environment may cause unexpected incompatibilities
  with MOLA Workflows.

  .. code-block:: bash

      ssh sator
      mv /tmp_user/sator/$USER/.bashrc /tmp_user/sator/$USER/.bashrcBACKUP

After having sourced the environment:

* the environment variable `$MOLA` should point on *<MOLA_ROOT_PATH>* (it has also be added to your `$PYTHONPATH`).
* many MOLA commands are accessible, refer to :ref:`commands`.
* you are ready to use MOLA !

===============================
What's inside that environment?
===============================

The command `mola_version` print a message indicating the main available libraries:

.. code-block:: text

    MOLA version 2.0.0 at ld (avx512)
    --> Python 3.8.14
    --> treelab 0.4.3
    --> Cassiopee 4.0
    --> maia 1.6
    --> turbo 1.3.1
    --> Ersatz 1.6.3
    --> elsA v5.4.01
        with ETC v0.337a 
    You are using the latest version of MOLA

=============================
Autologin to a remote machine
=============================

Most Worfklows require autologin to SATOR machine. Please make sure that you
have configured your ssh keys accordingly.

If you never allowed for autologin to SATOR machine, then you may be interested
in configuring ssh keys before proceding further. In order to do so, you may
follow one of the many `tutorials <https://www.thegeekstuff.com/2008/11/3-steps-to-perform-ssh-login-without-password-using-ssh-keygen-ssh-copy-id/>`_
available on the net. These instructions are summarized here:

.. tip::
  in order to make modifications to your ssh configuration files, you
  may need to make your hidden ``.ssh`` folder readable like this:

  .. code-block:: bash

    chmod 755 /home/$USER
    chmod 755 /home/$USER/.ssh

  Then you create public and private keys locally:

  .. code-block:: bash

    ssh-keygen

  Then you will be prompted to enter a key and passwords. You can simply
  let it blank and you type **[ENTER]** keyboard three (3) times.

  A message will show up indicating that you have successfully created
  your private and public keys, including a fingerprint.

  Then you copy your local key into remote host, like this:

  .. code-block:: bash

    ssh-copy-id -i ~/.ssh/id_rsa.pub $USER@sator

  You will be prompted to enter your password one last time.

  Finally, you can access to the remote host without entering again your
  password

  .. code-block:: bash

    ssh sator

