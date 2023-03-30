Simple Sphere Tutorial
======================

The purpose of this tutorial is to progressively introduce new MOLA users to the external aerodynamics CFD workflow.

For this purpose, we will make a computation of a flow around a very simple aerodynamic body: a *sphere*.

Before proceeding further, please make sure you have carefully followed the instructions of :ref:`StarterGuide`. 

In this tutorial, you will learn how to run a CFD elsA computation using MOLA interface, from mesh creation to visualization of results. To follow this tutorial, you will need a local machine with MOLA and elsA available. 

.. note::
    if you are using **SPIRO** machine, then please carefully read the :ref:`interactive session guidelines <spiroadvices>`

Before start, please copy the full example to a local directory of your choice, for instance:

.. code-block:: text

    cp -r $MOLA/EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE /stck/$USER/TUTO_SPHERE
    cd /stck/$USER/TUTO_SPHERE

Then, you can reset the case files by running:

.. code-block:: text

    ./reset.sh


Now you are ready to start this Tutorial!

1. Creation of the mesh
-----------------------

First step of a CFD computation consists in creating the mesh (or grid). This may be done using commercial software such as AutoGrid, Pointwise or ICEM. 

For simple meshes, you can use MOLA meshing techniques, powered by Cassiopee. Since the mesh around a sphere is very simple, we make use of Cassiopee and MOLA for creating the surface of the sphere, and then extrude it, obtaining a structured volume mesh. This is implemented in the script :download:`MESHING/build.py <../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/MESHING/build.py>`:

.. literalinclude:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/MESHING/build.py
    :language: python

.. important::
    It is very common that during MOLA Tutorials like the present one we explicitly show `python <https://www.python.org>`_ scripts. These scripts employ multiple functions from the different imported `modules <https://docs.python.org/3/tutorial/modules.html#modules>`_. We do **not** exhaustively explain in these tutorials the entire range of parameters and possibilities of each function. It is hence recommended that you read the available documentation associated to each function.

    For example, in this script we have made use of the Cassiopee functions `sphere6 <http://elsa.onera.fr/Cassiopee/Geom.html#Geom.sphere6>`_ and `getNodeFromName <http://elsa.onera.fr/Cassiopee/Internal.html#Converter.Internal.getNodeFromName>`_. We have also used MOLA functions :py:func:`~MOLA.GenerativeVolumeDesign.newExtrusionDistribution`, :py:func:`~MOLA.GenerativeVolumeDesign.extrude` and :py:func:`~MOLA.InternalShortcuts.save`. 

    **Remember:** each time there is a function in a script, you are invited to consult its documentation in order to know more details about such function.

Now you can build your mesh by executing:

.. code-block:: bash

    cd MESHING
    python3 build.py

This will produce the raw mesh file ``MESHING/sphere.cgns``, which looks like this:

.. figure:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/mesh.png
    :width: 60%
    :align: center

    raw mesh file ``MESHING/sphere.cgns`` produced by ``MESHING/build.py``

Next step consists in preparing all the data required for running the CFD simulation such as adding boundary-conditions, setting physical and numerical parameters, stating an initialization, etc...

2. Preprocess
-------------

Next step is *preprocess* (or *preparation*) of the simulation required data. This step is accomplished using the script :download:`prepare.py <../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/prepare.py>`. Please, take a moment to read it carefully:

.. literalinclude:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/prepare.py
    :language: python

The previous script employs :py:func:`~MOLA.Preprocess.prepareMesh4ElsA` and :py:func:`~MOLA.Preprocess.prepareMainCGNS4ElsA` functions of the module :py:mod:`~MOLA.Preprocess`. This is the *Workflow Standard*, adapted to external aerodynamics computations. 

.. seealso::
    :py:func:`MOLA.Preprocess.prepareMesh4ElsA`, :py:func:`MOLA.Preprocess.prepareMainCGNS4ElsA`

Essentially, :py:func:`~MOLA.Preprocess.prepareMesh4ElsA` function is employed for:

* making the `connection <https://cgns.github.io/CGNS_docs_current/sids/cnct.html#GridConnectivity1to1>`_ between blocks of the mesh
* setting `simple <https://cgns.github.io/CGNS_docs_current/sids/bc.html#t:BCTypeSimple>`_ `boundary-conditions <https://cgns.github.io/CGNS_docs_current/sids/bc.html>`_ (classical of external aerodynamics). In this case, *BCWall* and *BCFarfield* are both specified through `families <https://cgns.github.io/CGNS_docs_current/sids/misc.html#FamilyBC>`_ named ``WALL`` and ``FARFIELD``, respectively.
* making overset-oriented operations (not covered in this tutorial)
* splitting (or *partitioning*) and setting the processor's distribution of the final mesh. In this case, we impose ``4`` processors.

Next, :py:func:`~MOLA.Preprocess.prepareMainCGNS4ElsA` is basically employed for:

* setting physical parameters (airflow direction and properties).
* setting numerical parameters (scheme, time-marching...)
* setting requested extractions (output surfaces)
* producing simulation required files

Finally, you can run the preprocess using the command:

.. code-block:: bash

    python3 prepare.py



The output message will be:

.. code-block:: text

    assembling meshes...
    Reading MESHING/sphere.cgns (bin_hdf)...done.
    connecting type Match at base SPHERE
    setting boundary conditions...
    setting BC at base SPHERE
    setting boundary conditions... done
    splitting and distributing mesh...
    User requested NumberOfProcessors=4, switching to mode=="imposed"
    connecting type Match at base SPHERE

    Total number of processors is 4
    Total number of zones is 8
    Proc 0 has lowest nb. of points with 12152
    Proc 0 has highest nb. of points with 12152

    Node 1 has 48608 points
    TOTAL NUMBER OF POINTS: 48 608

    adding families...
    Setting BCName WALL of BCType BCWall at base SPHERE
    Setting BCName FARFIELD of BCType BCFarfield at base SPHERE
    adapting NearMatch to elsA
    No undefined BC found on PyTree
    prepareMesh took 0 hours 00 minutes and 00.237360 seconds
    setting .Solver#Output to FamilyNode WALL
    Initialize FlowSolution with uniform reference values
    invoking FlowSolution#Init with uniform fields using ReferenceState
    gathering links between main CGNS and fields
    saving PyTrees with links
    Writing OUTPUT/fields.cgns (bin_cgns)...done.
    Writing main.cgns (bin_cgns)...done.
    REMEMBER : configuration shall be run using 4 procs
    Copying templates of Workflow Standard (compute.py, coprocess.py) to the current directory
    prepareMainCGNS took 0 hours 00 minutes and 02.638992 seconds



And the following **case-specific** files will be created:

* ``setup.py``: ultralight python file including general information about the simulation. It includes the CFD solver parameters. This file is used for running the CFD simulation and may provide useful information for post-processing.

* ``main.cgns``:  `CGNS <https://cgns.github.io/CGNS_docs_current/sids/index.html>`_ file containing the initial mesh and *almost* all the information that CFD solver requires to properly run the simulation.

.. hint::
    you can visualize CGNS files using the MOLA integrated tree visualization graphical tool called **TreeLab**:

    .. code-block:: bash

        treelab main.cgns &


    .. figure:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/main.png
        :width: 60%
        :align: center

        CGNS file ``main.cgns`` produced by ``prepare.py`` script



.. important::
    ``main.cgns`` file contains links towards ``OUTPUT/fields.cgns`` *(see next)*

* ``OUTPUT/fields.cgns``: `CGNS <https://cgns.github.io/CGNS_docs_current/sids/index.html>`_ file, which contains the flowfields required for restarting the simulation (or initialization). This file is updated after each CFD run.

Also, the following **workflow-specific** files will appear:

* ``compute.py``: main python script employed for launching elsA solver.

* ``coprocess.py``: python script employed for making operations during the elsA computation (extractions, monitoring, control...)

* ``job_template.sh``: bash script employed for launching the CFD simulation.


.. danger::
    The aforementioned files should **not** be directly modified by the user (except ``job_template.sh``). If you want to modify simulation settings, please adapt and run again ``prepare.py`` script. If you **absolutely** need to modify these files (python or CGNS), please proceed with extreme caution. Before attempting modifications to these files, please read carefully the documentation of :py:func:`~MOLA.Preprocess.prepareMesh4ElsA` and :py:func:`~MOLA.Preprocess.prepareMesh4ElsA`, since your needs may already be satisfied by these functions (or equivalent functions of other MOLA Workflows). If after reading the documentation your need is still not covered, then you are invited to inform us about your specific need through the `MOLA Issues section <https://gitlab.onera.net/numerics/mola/-/issues>`_.

We are done with preprocessing! now we are ready to launch our simulation!

3. Computation
--------------

Next step is launch the CFD solver. This example is very light and can be run in your local machine. However, you will need at least ``4`` logical CPUs in order to run this example as presented here. 

Please check how many CPUs your machine has using:

.. code-block:: bash

    echo Available CPUs: $NPROCMPI


If your number of CPUs shown is lower than ``4``, then adjust **NumberOfProcessors** parameter of ``prepare.py`` to fit your available number of processors.

Then, you can run the simulation in your local machine using:

.. code-block:: bash

    ./job_template.sh



This will launch elsA solver and the simulation will start.

.. hint::
    You can check the iteration progress of the simulation using:

    .. code-block:: bash

        less -r coprocess.log

    followed by keyboard touches ``Shift`` + ``F``. This will show:

    .. code-block:: text

        COPROCESS LOG FILE STARTED AT 22/02/2023 10:41:17
        [0]: launch compute
        [0]: iteration 0
        [0]: iteration 1
        [0]: iteration 2
        [0]: iteration 3
        [0]: iteration 4
        ...
        [0]: iteration 1195
        [0]: iteration 1196
        [0]: iteration 1197
        [0]: iteration 1198
        [0]: iteration 1199
        [0]: iteration 1200 -> end of run
        [0]: updating setup.py ...
        [0]: updating setup.py ... OK
        [0]: will save OUTPUT/arrays.cgns ...
        [0]: ... saved OUTPUT/arrays.cgns
        [0]: will save OUTPUT/surfaces.cgns ...
        [0]: ... saved OUTPUT/surfaces.cgns
        [0]: will save OUTPUT/bodyforce.cgns ...
        [0]: ... saved OUTPUT/bodyforce.cgns
        [0]: will save OUTPUT/tmp-fields.cgns ...
        [0]: ... saved OUTPUT/tmp-fields.cgns
        [0]: deleting OUTPUT/fields.cgns ...
        [0]: deleting OUTPUT/fields.cgns ... OK
        [0]: moving OUTPUT/tmp-fields.cgns to OUTPUT/fields.cgns ...
        [0]: moving OUTPUT/tmp-fields.cgns to OUTPUT/fields.cgns ... OK
        [0]: END OF compute.py



    you can quit the file using keyboard touches ``ctrl`` + ``C`` and then touch ``Q``.

4. Visualization
----------------

After a while, your simulation will be completed. Now it is time to visualize the results. 

Essentially, you will have three different kind of result files contained in the ``OUTPUT`` directory:

* ``arrays.cgns``: includes 0-D data (scalars that depend on time, like Lift or Drag coefficients).

* ``surfaces.cgns``: includes surfaces (walls, slices, iso-surfaces, etc...).

* ``fields.cgns``: includes 3D volume data *(file may have very big size)*.

In the following paragraphs we will cite some strategies for visualizing the results.

MOLA Visualization technique
****************************

This technique makes use of :py:mod:`MOLA.Visu` module, which combines Cassiopee's `CPlot <http://elsa.onera.fr/Cassiopee/CPlot.html>`_ utility together with famous `matplotlib <https://matplotlib.org/>`_ library.

Its usage is script-based. Please take a moment to see an example of usage for this case in :download:`visualize.py <../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/visualize.py>`:

.. literalinclude:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/visualize.py
    :language: python

This example shows the generation of an image of the Mach flowfield, including the wall surface of the sphere, and a curve plot of the Drag evolution over the simulation iterations:

.. figure:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/flow.png
    :width: 100%
    :align: center

    flow visualization produced by ``visualize.py``


One of the main difficulties of this technique is correctly setting the **camera** settings (*posCam*, *posEye*, *dirCam*). In the next paragraph, you will learn how to visualize results using tkCassiopee interface, and we will also show how to get the aforementioned camera parameters in next :ref:`hint box <hintCamera>`.

tkCassiopee GUI
***************

You can visualize and explore data using `Cassiopee GUI <http://elsa.onera.fr/Cassiopee/Tutorials/Tutorials.html>`_ (graphical user interface). Please, use the following command to open the surfaces file:

.. code-block::

    cassiopee OUTPUT/surfaces.cgns &


This will open two windows: the main toolbar interface, and the CPlot window. Please take a moment to read Casiopee tutorials in order to understand `navigation controls <http://elsa.onera.fr/Cassiopee/Tutorials/Controls/Controls.html>`_, the `toolbar buttons <http://elsa.onera.fr/Cassiopee/Tutorials/ToolBar/ToolBar.html>`_, the `menu options <http://elsa.onera.fr/Cassiopee/Tutorials/OpeningApplets/OpeningApplets.html>`_ and the `view controls <http://elsa.onera.fr/Cassiopee/Tutorials/View/View.html>`_ including `field views <http://elsa.onera.fr/Cassiopee/Tutorials/ViewScalar/ViewScalar.html>`_.

Now, we are going to explain briefly how to show a field and explore data.

* Once the file is open, please click on CPlot window and press keyboard touch ``M``, in order to switch to 2D (:math:`OXY` plane) view/navigation mode.

* left-click on ``Visu`` tab. 

* on ``Mesh`` scroll-bar menu, select ``Scalar``

.. figure:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/tkCassiopeeScalarSelection.png
    :width: 50%
    :align: center

    ``Visu`` > ``tkView`` > ``Mesh`` select ``Scalar``

* Next to ``Field:`` label, select ``Mach`` on scroll-down menu.

* In order to precisely know the field value at a given point, you can press ``Ctrl`` + ``left-click`` at desired point. 

.. figure:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/tkCassiopeeMach.png
    :width: 100%
    :align: center

    Exploration of fields data contained in ``OUTPUT/surfaces.cgns`` using tkCassiopee

.. _hintCamera:

.. hint::
    in order to obtain **camera** parameters, as stated in previous paragraph, you can at any moment ``right-click`` on ``State`` tab, select ``tkCamera`` tools and press ``Get`` button. This will show the *posCam*, *posEye* and *dirCam* values that you want to use into :download:`visualize.py <../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/visualize.py>` script:

    .. figure:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/tkCassiopeeCamera.png
        :width: 50%
        :align: center

        get camera parameters for usage in ``visualize.py``

Finally, please note that you can also visualize 3D flowfields using tkCassiopee:

.. code-block:: bash

    cassiopee OUTPUT/fields.cgns &

.. figure:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/tkCassiopeeFields.png
    :width: 100%
    :align: center

    visualize data of ``OUTPUT/fields.cgns`` using tkCassiopee

.. tip::
    you can hide blocks by pressing ``Shift`` + ``right-click`` on the block that you want to hide, either on CPlot or on the tree nodes.


Paraview
********

Results fields can also be visualized using `Paraview <https://www.paraview.org/>`_ tool. Let us open our surfaces file:

.. code-block::

    paraview surfaces.cgns &


Please consult the detailed `Paraview documentation <https://www.paraview.org/resources/>`_ to learn how to use this software. However, in this example we will briefly explain the steps to follow in order to produce the following visualization:


.. figure:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/paraviewMachPressure.png
    :width: 100%
    :align: center

    visualize Mach and Pressure data of ``OUTPUT/surfaces.cgns`` using Paraview


Follow these steps:

* Check **only** ``Iso_Z_1e-06`` data on the Hierarchy list in Properties Tab (usually found in bottom left of the window)

.. image:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/paraviewHierarchy.png
    
* Then scroll-down in Properties tab and check **only** Mach field in ``Point Arrays`` list:

.. image:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/paraviewPointArray.png


.. |apply| image:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/paraviewApply.png

* Apply options by clicking on |apply| button

.. |vtk| image:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/paraviewvtkBlockColors.png

.. |mach| image:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/paraviewMachButton.png


* by default, your view will be set to ``vtkBlockColors`` mode on the toolbar (top side of the window |vtk|). You will need to click on it and explicitly select ``Mach`` point field |mach|. Now your Mach field become visible (use mouse scroll to zoom-in):

.. figure:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/paraviewMachPoor.png
    :width: 50%
    :align: center

    visualization of |mach|

* now we are going to add the Pressure visualization at the sphere walls. For this, we open a new instance of our file ``File`` > ``Open...`` and select ``OUTPUT/surfaces.cgns``.

* We make sure to select the **second** element in our Pipeline Browser and activate visibility (eye opened):

.. image:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/paraviewPipeline.png

* we select **only** *WALL* in our Hierarchy panel:

.. image:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/paraviewHierarchyWall.png

* we select ``Pressure`` field in our *Cell Arrays* panel:

.. image:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/paraviewCellArray.png

.. |pressure| image:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/paraviewPressureButton.png

* we select ``Pressure`` in our Coloring scroll-down menu 

.. |preset| image:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/paraviewPreset.png

* we select a different colormap for our Pressure by pressing into the *Choose preset* button |preset|

and that's all!

.. tip::
    you may save your visualization settings into a file using ``File`` > ``Save State...``. Then you will be able to open again your visualization by simply doing:

    .. code-block:: bash

        paraview state.pvsm &

.. note::
    you can also use Paraview for visualization of ``OUTPUT/fields.cgns`` file:

    .. code-block:: bash

        paraview OUTPUT/fields.cgns &


Tecplot
*******

You can also use `Tecplot <https://www.tecplot.com/>`_ software to visualize ``OUTPUT/surfaces.cgns`` or ``OUTPUT/fields.cgns`` data.

.. important:: 
    make sure you have tecplot available on your system. You may need to load it explicitly using for example:

    .. code-block:: bash 

        module load tecplot/2021R2-360ex


You can open your surfaces file from command line simply like this:

.. code-block:: bash 

    tec360 OUTPUT/surfaces.cgns &


Or, once Tecplot is opened, by using the menu ``File`` > ``Load Data`` > select ``OUTPUT/surfaces.cgns`` >> ``Add to List`` > ``Open``


.. figure:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/FIGURES/tecplot.png
    :width: 100%
    :align: center

    visualization of file ``OUTPUT/surfaces.cgns`` using Tecplot

.. note::
    You can also use Tecplot to read ``OUTPUT/fields.cgns`` file. 

.. tip::
    Sometimes Tecplot has some incompatibilities with CGNS files as used in MOLA. For this reason, in this occasions it is useful to 
    convert the ``OUTPUT`` files into binary teclplot format, which is better read by Tecplot:

    ::

        import MOLA.InternalShortcuts as J
        t = J.load('OUTPUT/surfaces.cgns')
        J.save(t,'OUTPUT/surfaces.plt')


    This applies to the main three OUTPUT files: ``arrays.cgns``, ``surfaces.cgns`` and ``fields.cgns``


matplotlib
**********

Visualization of 1D data is easily done using matplotlib and ``arrays.cgns``` file. For example, you may use script :download:`monitor_loads.py <../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/monitor_loads.py>` for plotting the :math:`C_D`:

.. literalinclude:: ../../EXAMPLES/WORKFLOW_STANDARD/SPHERE/SIMPLE/monitor_loads.py
    :language: python 