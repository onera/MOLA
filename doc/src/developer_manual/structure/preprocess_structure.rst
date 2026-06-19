####################################
Preprocess Architecture
####################################

MOLA's preprocessing architecture follows a **dispatcher pattern** where generic workflow methods delegate to solver-specific implementations. This section explains how workflow methods connect to preprocess modules and how solver specialization is achieved.

.. contents:: Table of Contents
   :depth: 2
   :local:

Architecture Overview
=====================

The preprocessing pipeline follows this pattern:


.. graphviz::
   :align: center
   :caption: Preprocessing Architecture Flow

   digraph PreprocessArchitecture {
       rankdir=TB; 

       node [shape=box, style=filled, fillcolor="#333333", fontname="Helvetica"];
       edge [fontname="Helvetica", fontsize=10];
       
       // Workflow Layer
       workflow [label="Workflow Method\n(e.g., set_extractions())", fillcolor="#A0C4FF"];
       
       // Generic Layer  
       dispatcher [label="Preprocess module\n(e.g., cfd/preprocess/extractions/extractions.py)\napply()", fillcolor="#FFD93D"];
       
       // Core Dispatcher
       core [label="src/mola/cfd/__init__.py\napply_to_solver()", fillcolor="#FF6B6B"];
       
       // Solver Implementations
       solver_elsa [label="cfd/preprocess/extractions/\nsolver_elsa.py\napply_on_solver()", fillcolor="#6BCB77"];
       solver_fast [label="cfd/preprocess/extractions/\nsolver_fast.py\napply_on_solver()", fillcolor="#6BCB77"];
       solver_sonics [label="cfd/preprocess/extractions/\nsolver_sonics.py\napply_on_solver()", fillcolor="#6BCB77"];
       
       // Connections
       workflow -> dispatcher [label="calls apply()"];
       dispatcher -> core [label="calls apply_on_solver()"];
       core -> solver_elsa [label="if solver='elsa'"];
       core -> solver_fast [label="if solver='fast'"];
       core -> solver_sonics [label="if solver='sonics'"];
   }


This design allows MOLA to support multiple CFD solvers (elsA, Fast, SoNICS) 
while maintaining a unified workflow interface.

Generic Methods: apply and apply_on_solver
------------------------------------------

The preprocessing architecture uses two key generic methods to connect workflow
methods to solver-specific implementations.

**apply** (Generic Implementation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Located in the generic file 
(e.g., :file:`src/mola/cfd/preprocess/extractions/extractions.py`),
the ``apply`` method serves as the entry point for preprocessing operations:

.. code-block:: python

    # In apply_to_solver(workflow)/boundary_conditions.py
    def apply(workflow):
        # Generic operations (relevant for all solvers)
        ...
        # Operations specific to the chosen solver
        apply_to_solver(workflow)

Key characteristics:

- Takes ``workflow`` as unique parameter
- Does not contain solver-specific logic
- Calls ``apply_to_solver`` from :file:`src/mola/cfd/__init__.py`. 
  That function applies the function also called ``apply_to_solver`` in the file
  specific to the solver and located in the same repository. For the current example, 
  if the chosen solver is elsA, it calls ``apply_on_solver`` in 
  :file:`src/mola/cfd/preprocess/extractions/solver_elsa.py`.


**apply_on_solver** (Solver-Specific Implementation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Located in solver-specific files 
(e.g., :file:`src/mola/cfd/preprocess/extractions/solver_elsa.py`),
the ``apply_on_solver`` method:

- Contains solver-specific logic (e.g. CGNS tree manipulation, parameter formatting, etc)
- Uses solver-specific helper functions


Where to Extend
===============

Adding a New Preprocess Module
------------------------------

1. Create new directory in :file:`src/mola/cfd/preprocess/`

2. Create generic dispatcher:

.. code-block:: python

    # In my_module/my_module.py
    def apply(workflow):
        """Generic apply function"""
        from mola.cfd import call_solver_specific_function
        call_solver_specific_function(workflow, 'apply', 3, workflow)

3. Create solver-specific implementations:

.. code-block:: python

    # In my_module/solver_elsa.py
    def apply(workflow):
        """elsA-specific implementation"""
        # Your implementation here
        pass

    # In my_module/solver_fast.py
    def apply(workflow):
        """FAST-specific implementation"""
        pass

    # In my_module/solver_sonics.py
    def apply(workflow):
        """SONICS-specific implementation"""
        pass

4. Register the function in :file:`src/mola/cfd/__init__.py`:

.. code-block:: python

    def call_solver_specific_function(workflow, function_name, min_depth, *args):
        solver = workflow.Solver
        module_path = f'mola.cfd.preprocess.my_module.solver_{solver}'
        module = importlib.import_module(module_path)
        func = getattr(module, function_name)
        return func(*args)

Adding a New Boundary Condition Type
------------------------------------

1. Add function to solver-specific file:

.. code-block:: python

    # In boundary_conditions/solver_elsa.py
    def my_custom_bc(workflow, Family, CustomParam=1.0):
        """
        Custom boundary condition for elsA.
        
        Parameters
        ----------
        workflow : Workflow
        Family : str
        CustomParam : float
            Custom parameter
        """
        # Implementation
        define_bc_family(workflow.tree, Family, 'BCFarfield')
        # Set custom parameters
        workflow.tree.setParameters('.Solver#BC', custom_param=CustomParam)

2. Register in dispatcher:

.. code-block:: python

    # In boundary_conditions/boundary_conditions_dispatcher.py
    self.boundary_functions['MyCustomBC'] = my_custom_bc

3. Add interface method:

.. code-block:: python

    # In workflow/interface.py
    def add_to_BoundaryConditions_MyCustomBC(self, Family: str, CustomParam: float = 1.0):
        """Add custom boundary condition"""
        self.BoundaryConditions.append(self._get_comp(
            WorkflowInterface.add_to_BoundaryConditions_MyCustomBC,
            self.get_default_values_from_local_signature()
        ))

Adding Solver Support
---------------------

To add a new solver (e.g., 'my_solver'):

1. Create solver-specific files in each preprocess module:

.. code-block:: text

    src/mola/cfd/preprocess/
    ├── boundary_conditions/solver_my_solver.py
    ├── motion/solver_my_solver.py
    ├── cfd_parameters/solver_my_solver.py
    └── ...

2. Implement required functions in each file

3. Update :file:`src/mola/cfd/__init__.py` to recognize the solver

4. Add compute implementation in :file:`src/mola/cfd/compute/solver_my_solver.py`

5. Add coprocess implementation in :file:`src/mola/cfd/coprocess/solver_my_solver.py`