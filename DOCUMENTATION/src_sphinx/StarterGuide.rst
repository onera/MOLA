.. _StarterGuide:

Starter Guide
=============

Environment and machine
-----------------------

MOLA environment is available on the following ONERA machines:

* sator

* ld (local linux in CentOS 7 or CentOS 8)

* visung

* visio

* celeste

* spiro (partition ``spiro-daaa``)


You may connect to one of these machines, and source the unified MOLA
environment file found at :

.. code-block:: bash

    source /stck/lbernard/MOLA/v1.14/env_MOLA.sh

You will see a message indicating the main available libraries:

.. code-block:: text

    MOLA version v1.14 at sator-new
     --> Python 3.7.4
     --> elsA v5.1.01
     --> ETC v0.330f
     --> Cassiopee 3.3
     --> VPM 0.1
     --> TreeLab dev
     --> PUMA r337
    You are using the latest version of MOLA

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
      rm /tmp_user/sator/$USER/.bashrc


Autologin to SATOR machine
--------------------------

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

Make functional checkings
-------------------------

Now it is time to make a final functional checking in order to make sure
everything is configured correctly. In order to make this verification, you
shall start a python console from your ``stck`` space:


.. code-block:: bash

   cd /stck/$USER
   python

then, in the Python console, you import the module :mod:`MOLA.WorkflowAirfoil` and
launch function :mod:`~MOLA.WorkflowAirfoil.checkDependencies`

>>> import MOLA.WorkflowAirfoil as WF
>>> WF.checkDependencies()

the correct output of the call of :mod:`~MOLA.WorkflowAirfoil.checkDependencies` is:


.. code-block:: text

  Checking numpy...
  used version: 1.16.6
  minimum required: 1.16.6
  numpy version OK
  Checking scipy...
  used version: 1.2.3
  minimum required: 1.2.3
  scipy version OK

  Checking interpolations...
  interpolation OK

  Attempting file/directories operations on SATOR...
  Repatriating /tmp_user/sator/lbernard/MOLAtest/testfile.txt by COPY...
  Waiting for testfile.txt ...
  ok
  /tmp_user/sator/lbernard/MOLAtest/
  Attempting file/directories operations on SATOR... done

  Checking XFoil...
  XFoil OK
  Checking matplotlib...
  used version: 2.2.5
  minimum required: 2.2.5
  matplotlib version OK
  producing figure...
  saving figure...
  showing figure... (close figure to continue)

  VERIFICATIONS TERMINATED

.. _matplotlib: https://matplotlib.org/

.. _XFoil: https://web.mit.edu/drela/Public/web/xfoil/

.. attention:: The checking procedure produces **graphic output**. If you do not
  allow for graphic output in the used machine, then `XFoil`_ and `matplotlib`_
  operations will fail.
