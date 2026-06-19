#######################
Workflow Architecture
#######################

MOLA uses a three-tier architecture for workflow management: **Workflow**, **WorkflowInterface**, and **WorkflowManager**. This section explains each component and how they interact.

.. contents:: Table of Contents
   :depth: 2
   :local:

Workflow
========

The :py:class:`Workflow` class (located at :file:`src/mola/workflow/workflow.py`) is the core orchestrator of the CFD simulation pipeline.

Core Responsibilities
---------------------

- Orchestrate the preprocessing pipeline
- Manage mesh operations (assembly, positioning, connection, families)
- Coordinate flow generation, boundary conditions, and initialization
- Handle job submission and remote execution

Key Methods
-----------

The main workflow methods follow a sequential pipeline:

.. code-block:: python

    def prepare(self):
        """Main preprocessing entry point"""
        self.prepare_job()           # Run management setup
        self.process_mesh()          # Mesh assembly and processing
        self.process_overset()       # Overset mesh configuration
        self.compute_flow_and_turbulence()  # Flow field generation
        self.set_motion()            # Motion parameters
        self.set_boundary_conditions()      # BC setup
        self.set_cfd_parameters()    # Numerical parameters
        self.initialize_flow()       # Flow initialization
        self.set_extractions()       # Data extraction setup
        self.check_preprocess()      # Consistency checks
        self.finalize_preprocess()   # Final preparation

    def compute(self):
        """Execute the CFD solver"""
        compute.apply(self)

    def write_cfd_files(self):
        """Write CGNS input files for the solver"""
        write_cfd_files.apply(self)

    def remove_cfd_files(self):
        """Clean up CGNS files"""
        remove_cfd_files.apply(self)

Mesh Processing Methods
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def process_mesh(self):
        self.assemble()        # Read and assemble meshes
        self.positioning()     # Apply positioning transformations
        self.connect()         # Set up grid connections
        self.define_families() # Define family groups
        self.duplicate()       # Handle mesh duplication
        self.split_and_distribute()  # Prepare for parallel execution

WorkflowInterface
=================

The :py:class:`WorkflowInterface` class (located at :file:`src/mola/workflow/interface.py`) 
provides the user-facing API for configuring workflows.

Design Pattern
--------------

The interface uses a **property delegation** pattern where interface attributes are linked 
to Workflow attributes through Python properties. This ensures that modifications through 
the interface directly affect the underlying Workflow object.

.. code-block:: python

    class WorkflowInterface(object):
        def __init__(self, workflow, **kwargs):
            # Link attributes to workflow using properties
            for attr_name in attr_names:
                self._create_property(attr_name)
        
        def _create_property(self, attr_name):
            def getter(self):
                return getattr(self.workflow, attr_name)
            
            def setter(self, value):
                setattr(self.workflow, attr_name, value)
            
            setattr(WorkflowInterface, attr_name, property(getter, setter))

Configuration Methods
---------------------

The interface provides type-safe configuration methods following the pattern ``set_<attribute>()``:

.. code-block:: python

    def set_Solver(self, solver_name: str = solver):
        """Set the CFD solver (elsa, fast, sonics)"""
    
    def set_Fluid(self, Gamma=1.4, IdealGasConstant=287.053, ...):
        """Set fluid properties"""
    
    def set_Flow(self, Generator='External_rho_T_V', **kwargs):
        """Set flow parameters using a flow generator"""
    
    def set_Turbulence(self, Model='Wilcox2006-klim', ...):
        """Set turbulence model parameters"""
    
    def set_BoundaryConditions(self, user_list: list):
        """Set boundary conditions from a list of dicts"""
    
    def set_RunManagement(self, RunDirectory='.', Machine=None, ...):
        """Set job execution parameters"""

Adding Components
~~~~~~~~~~~~~~~~~

For list-based attributes, use ``add_to_<Attribute>()`` methods:

.. code-block:: python

    def add_to_BoundaryConditions(self, Family: str, Type: str, **kwargs):
        """Add a boundary condition to the list"""
    
    def add_to_Extractions_BC(self, Source: str, Fields: list, ...):
        """Add a boundary extraction"""
    
    def add_to_ConvergenceCriteria(self, ExtractionName: str, ...):
        """Add a convergence criterion"""

Type Validation
~~~~~~~~~~~~~~~

The interface performs automatic type validation based on function signatures:

.. code-block:: python

    def set_Fluid(self,
            Gamma: float = 1.4,
            IdealGasConstant: float = 287.053,
            Prandtl: float = 0.72,
            ...):
        self.Fluid = self._get_comp(
            WorkflowInterface.set_Fluid, 
            self.get_default_values_from_local_signature()
        )

For previous example, default values are set for `Gamma`, `IdealGasConstant` and `Prandtl` 
if not provided by user in the **Flow** attribute. 
If user gives a value with the wrong type 
(e.g. an :py:class:`int` for `IdealGasConstant`, like `IdealGasConstant=287`)
the interface raises an error and prints expected parameters (names, types and default values).

WorkflowManager
===============

The :py:class:`WorkflowManager` class (located at :file:`src/mola/workflow/manager.py`) 
manages parametric studies and batch job execution.

Main Classes
------------

WorkflowManager
~~~~~~~~~~~~~~~

It is the only class that must be called directly by user. 
It coordinates multiple workflow executions:

.. code-block:: python

    from mola.workflow import Workflow
    from mola.workflow import manager as WM

    workflow = Workflow(...)
    manager = WM.WorkflowManager(workflow, '/path/to/study/')

    # Define variations
    manager.new_job('case_1')
    manager.add_variations([
        ('RunManagement|JobName', 'case_1'),
        ('Flow|Velocity', 50.0),
    ])

    manager.prepare()  # Deploy directories and prepare jobs
    manager.submit()   # Submit jobs to scheduler

When specifying variations, use the pipe separator to navigate nested structures:

- ``'RunManagement|JobName'`` → ``workflow.RunManagement['JobName']``
- ``'BoundaryConditions|Family=Wall|Type'`` → Find BC where ``Family=='Wall'``, then get ``Type``
- ``'Numerics|CFL|StartValue'`` → ``workflow.Numerics['CFL']['StartValue']``

Other internal classes
~~~~~~~~~~~~~~~~~~~~~~

- WorkflowDispatcher: Generates workflow variations from a base workflow:
- WorkflowSequentialManager: Handles sequential execution of workflow cases:
- WorkflowSender: handles remote file synchronization:


Adding a New Workflow
---------------------

Workflow Inheritance
~~~~~~~~~~~~~~~~~~~~

Workflows inherit from either the base :py:class:`Workflow` class or from an 
intermediate class like :py:class:`WorkflowRotatingComponent`. The choice of 
parent class depends on the application type and kinematics:

- **Base Workflow**: Use for static geometries or simple configurations
- **Intermediate Classes**: Use for specific kinematic behaviors (e.g., rotating components, 
  moving bodies, flexible structures)

Module Structure
~~~~~~~~~~~~~~~~

A new workflow must be created as a new module in :file:`src/mola/workflow/` with the 
following structure:

.. code-block:: text

    src/mola/workflow/
    └── my_workflow/
        ├── __init__.py
        ├── workflow.py      # Mandatory: Custom Workflow class
        ├── interface.py     # Mandatory: Custom WorkflowInterface class
        └── manager.py       # Optional: Custom manager functionality

**workflow.py** - Contains the custom Workflow class:

.. code-block:: python

    from mola.workflow.workflow import Workflow

    class MyCustomWorkflow(Workflow):
        """
        Custom workflow for specific application.
        
        Inherit from Workflow base class or from an intermediate class
        like WorkflowRotatingComponent depending on kinematics.
        """
        
        def initialize_flow(self):
            # Custom preprocessing logic
            super().initialize_flow()
            self.custom_operation()
        
        def custom_operation(self):
            # Your custom implementation
            pass

**interface.py** - Contains the corresponding WorkflowInterface class:

.. code-block:: python

    from mola.workflow.interface import WorkflowInterface

    class MyCustomWorkflowInterface(WorkflowInterface):
        """
        Interface for MyCustomWorkflow.
        """
        
        def __init__(self, workflow, **kwargs):
            super().__init__(workflow, **kwargs)
        
        def set_CustomParameter(self, value: float = 1.0):
            """Set custom parameter."""
            self.CustomParameter = self._get_comp(
                MyCustomWorkflowInterface.set_CustomParameter,
                self.get_default_values_from_local_signature()
            )


**manager.py** - For custom workflow management functionality:

.. code-block:: python

    from mola.workflow.manager import WorkflowManager

    class MyCustomWorkflowManager(WorkflowManager):
        """
        Custom workflow manager with specialized functionality.
        """
        
        def prepare(self):
            # Custom preparation logic
            super().prepare()

