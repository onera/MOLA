.. _Commands:

Commands
========

MOLA comes with several commands accessible from any directory, once MOLA environment is sourced. 
These commands are shortcuts for often simple but frequent operations. They may be used directly in 
the terminal. 

The list of available commands may be displayed (with a short description for each) by using:

.. code-block:: bash

    mola_available

The output of this command is:

.. code-block:: text

    mola_available        : Display all available MOLA commands
    mola_clean            : Remove all MOLA log files, cache files, debug files, status files and plots
    mola_jobsqueue_sator  : Show jobs currently in queue on sator
    mola_merge_containers : Merge fields containers to facilitate visualization
    mola_repatriate       : Repatriate directories after submitting several simulations on sator at once
    mola_seelog           : Display the updated content of coprocess.log
    mola_version          : Display versions of modules loaded with MOLA
    mola_watchdir         : watch the current directory files
    mola_watchqueue       : watch the SLURM job queue
    treelab               : Open the Treelab interface to read and manipulate CGNS tree

You may notice that each of them starts with the prefix `mola_*`. The only exception is the 
graphical CGNS interface `treelab`, which is kind of an enhanced version of `cgnsview`.

.. note::
    
    Some of the commands may be used with options. You may try to call a command with the 
    help option (``-h`` or ``--help``) to display help, but it may be not available for all commands yet.

.. note:: 

    All these executable files are located in $MOLA/bin

 