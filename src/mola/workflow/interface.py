#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

import os
import copy
import pathlib
import numpy as np
import copy

from mpi4py import MPI
from treelab import cgns
from treelab.cgns.tree import Tree
from treelab.cgns.base import Base
from treelab.cgns.zone import Zone
import inspect
from typing import Union, Callable, Dict, get_type_hints
from mola.logging import (mola_logger,
                       MolaException,
                       MolaUserError,
                       MolaUserAttributeError,
                       get_signature)
from mola.logging.formatters import BOLD, RED, CYAN, PINK, YELLOW, ENDC
import mola.naming_conventions as names
from mola import solver


class WorkflowInterface(object):

    _fake_attributes = ['self','tree','workflow','Mesh']

    def __init__(self, workflow,
            tree=None,
            Solver : str = None,
            Mesh : Union[dict, str] = None,
            RawMeshComponents : list = None,
            Fluid : dict = None,
            Flow : dict = None,
            Turbulence : dict = None,
            BoundaryConditions : list = None,
            SplittingAndDistribution : dict = None,
            Overset : dict = None,
            Numerics : dict = None,
            BodyForceModeling : list = None,
            Motion : dict = None, # make list of dicts
            Initialization : dict = None,
            ExtractionsDefaults : list = None,
            Extractions : list = None,
            ConvergenceCriteria : list = None,
            RunManagement : dict = None,
            ApplicationContext : dict = None,
            SolverParameters : dict = None,
            ):
            
        attributes = self.get_default_values_from_local_signature()
        self.workflow = workflow
        
        # If Mesh is given, use it to initialize RawMeshComponents and delete it
        if Mesh is not None:
            if isinstance(Mesh, dict):
                Mesh.setdefault('Name','Base')
                attributes['RawMeshComponents'] = [Mesh]
            elif isinstance(Mesh, str):
                attributes['RawMeshComponents'] = [dict(Source=Mesh, Name='Base')]
            else:
                raise MolaUserError(f"Input parameter Mesh must be either a dict or str but is {type(Mesh)}")
        attributes.pop('Mesh')
    
        # Link attributes of WorkflowInterface to them of Workflow.
        # Hence, a modification of the attribute in WorkflowInterface
        # modifies the attribute of Workflow with the same name.
        attr_names = list(attributes) + ['_workflow_parameters_container_', 'Name', 'tree']
        for attr_name in attr_names:
            if attr_name in ['self', 'workflow']:
                continue
            self._create_property(attr_name)

        self._workflow_parameters_container_ = names.CONTAINER_WORKFLOW_PARAMETERS
        self.Name = self.workflow.__class__.__name__
        self.tree = tree

        if self.tree is None: 
            self.set_attributes(attributes)
        else:
            self.get_workflow_parameters_from_tree()
        
    def _create_property(self, attr_name):
        def getter(self):
            return getattr(self.workflow, attr_name)

        def setter(self, value):
            setattr(self.workflow, attr_name, value)

        setattr(WorkflowInterface, attr_name, property(getter, setter))

    
    def get_workflow_parameters_from_tree(self):
        
        if isinstance(self.tree, str):
            workflow_parameters = cgns.load_workflow_parameters(self.tree)
        elif isinstance(self.tree, cgns.Tree):
            workflow_parameters = self.tree.getParameters(self._workflow_parameters_container_, transform_numpy_scalars=True)
        else:
            raise MolaUserError(f'The given tree must be either a filename or a Tree read by treelab.')
        
        for parameter in workflow_parameters:
            setattr(self, parameter, workflow_parameters[parameter])

        # for attributes appearing in constructor signature
        expected_types = self.get_argument_types(WorkflowInterface.__init__)
        for attribute_name, expected_type in expected_types.items():
            if attribute_name in self._fake_attributes: continue
            if getattr(self, attribute_name) is None:
                setattr(self, attribute_name, expected_type())

        if self.SolverParameters is None: self.SolverParameters = dict()

        # HACK treelab writes str with spaces as list of str
        if isinstance(self.RunManagement['LauncherCommand'], list):
            self.RunManagement['LauncherCommand'] = ' '.join(self.RunManagement['LauncherCommand'])
    
    def set_attributes(self, attributes):

        expected_attribute_types = self.get_argument_types(WorkflowInterface.__init__)

        for attribute_name, user_input in attributes.items():
            if attribute_name in self._fake_attributes: continue
        
            try:
                expected_type = expected_attribute_types[attribute_name]
            except KeyError:
                raise MolaException(f'attribute_name={attribute_name} not implemented from {self.Name} (expected {list(expected_attribute_types)})')

            if user_input is None:
                user_input = expected_type()
            
            if not isinstance(user_input, expected_type):
                raise MolaUserError(f'attribute {attribute_name} must be of type {expected_type}')
            
            method = getattr(self, 'set_'+attribute_name)

            if expected_type is dict:

                # since BUG incompatible with Motion defined by user (a dict)
                if attribute_name == "Motion":
                    method(user_input)

                else:
                    try: method(**user_input)
                    except TypeError as e: raise MolaUserAttributeError(method, e)

            elif expected_type in [list, str]:
                try: method(user_input)
                except TypeError as e: raise MolaUserAttributeError(method, e)


    def check_consistency_between_solver_and_environment(self):
        requested_solver = self.Solver
        env_solver = os.environ.get('MOLA_SOLVER')
        if requested_solver != env_solver:
            mola_logger.warning(
                f'The requested solver "{requested_solver}" does not '
                f'match the type of environment "{env_solver}"'
                )

    def set_Solver(self, solver_name : str = solver):
        if solver_name is None or solver_name == '':
            self.Solver = solver 
        else:
            self.Solver = solver_name.lower()
        

    def add_to_RawMeshComponents(self,
        Mesher           : str  = None,
        Unit             : str  = 'm',
        CleaningMacro    : str  = None,
        Families         : list = None,
        Positioning      : list = None,
        Connection       : list = None,
        DefaultToleranceForConnection : float = 1e-8,
        OversetOptions   : dict = None,
        OversetMotion    : dict = None,
        *,
        Name             : str,
        Source           : Union[ str, Tree, Base, Zone],
        ):
        '''
        Set workflow attribute **RawMeshComponents**

        Parameters
        ----------
        Name : str
            Name of the component, which defines also the name of the CGNS Base.
        Source : Union[ str, Tree, Base, Zone]
            Name of the mesh file. Source can also be directly a Tree, Base or Zone read by treelab.
        Mesher : str, optional
            Name of the tool used to generate the Mesh. Available values are: 
                #. `None` or `'default'`: in that case, nothing is done.
                #. `'autogrid'`: make standard operations to clean and rotate a mesh generated with Autogrid5.
        Unit : str, optional
            Unit for mesh coordinates, to convert to meters if needed. 
            Available units are: 'm', 'dm', 'cm', 'mm', 'inches'.
            By default 'm'.
        CleaningMacro : str, optional
            :fas:`person-digging;sd-text-warning`
        Families : list, optional
            _description_, by default None
        Positioning : list, optional
            _description_, by default None
        Connection : list, optional
            _description_, by default None
        DefaultToleranceForConnection : float, optional
            1e-8 by default
        OversetOptions : dict, optional
            :fas:`person-digging;sd-text-warning`
        '''
        Positioning = self._add_scaling_according_to_unit(Positioning, Unit)    
        self.RawMeshComponents.append(self._get_comp(
            WorkflowInterface.add_to_RawMeshComponents, self.get_default_values_from_local_signature()))
    
    @staticmethod
    def _add_scaling_according_to_unit(Positioning, Unit):
        SCALE_DICT = dict(
            mm = 0.001,
            cm = 0.01,
            dm = 0.1,
            m  = 1.,
            inches = 0.0254,
        )
        if Unit != 'm':
            if Positioning is None:
                Positioning = []
            if not any([item['Type'] == 'Scale' for item in Positioning]):
                Positioning.append(dict(Type='Scale', Scale=SCALE_DICT[Unit]))
        return Positioning

    def set_RawMeshComponents(self, user_list : list):
        self._set_by_user_list(self._method_name(), user_list)
    
    def set_ApplicationContext(self):
        '''
        this method is virtual and shall be reimplemented in inherited workflows
        '''
        self.ApplicationContext = self._get_comp(self.set_ApplicationContext, self.get_default_values_from_local_signature())

    def set_Fluid(self,
            Gamma                      : float =  1.4,
            IdealGasConstant           : float =  287.053,
            Prandtl                    : float =  0.72,
            PrandtlTurbulent           : float =  0.9,
            SutherlandConstant         : float = 110.4,
            SutherlandViscosity        : float = 1.78938e-05,
            SutherlandTemperature      : float = 288.15):
        '''
        Set workflow attribute **Fluid**

        Parameters
        ----------
        Gamma : float, optional
            Specific heat ratio (or adiabatic index), by default 1.4
        IdealGasConstant : float, optional
            by default 287.053
        Prandtl : float, optional
            by default 0.72
        PrandtlTurbulent : float, optional
            by default 0.9
        SutherlandConstant : float, optional
            by default 110.4
        SutherlandViscosity : float, optional
            by default 1.78938e-05
        SutherlandTemperature : float, optional
            by default 288.15
        '''
        self.Fluid = self._get_comp(WorkflowInterface.set_Fluid, self.get_default_values_from_local_signature())


    def set_Flow(self,
            Generator : str = 'External_rho_T_V',
            # NOTE kwargs are here not to raise an error due to specific arguments for the Generator
            # This function has a specific behavior to raise appropriated errors.
            # The interface is delegated to the method set_defaults of the Generator class 
            **kwargs  
            ):

        from  mola.cfd.preprocess import flow_generators
        FlowGen = flow_generators.get_flow_generator(Generator) 
        signature = inspect.signature(FlowGen.set_Flow_defaults)
        default_kwargs = dict((name, param.default) for name, param in signature.parameters.items() if name != 'self')
        for name, value in kwargs.items():
            if name not in default_kwargs:
                error_msg = (f"set_Flow() got an unexpected keyword argument '{name}'. " 
                             f"The following arguments are for the currently selected "
                             f"{BOLD}Generator{ENDC}{RED}: {PINK}{Generator}{ENDC}{RED} "
                             f"(another one may be selected in {BOLD}Flow{ENDC}{RED} if needed)")
                raise MolaUserAttributeError(FlowGen.set_Flow_defaults, error_msg)
            else:
                default_kwargs[name] = value
                
        self.Flow = self._get_comp(FlowGen.set_Flow_defaults, default_kwargs) 
        self.Flow['Generator'] = Generator

    def set_Turbulence(self,
        Viscosity_EddyMolecularRatio : float = 0.1,
        Level                        : float = 0.001,
        Model                        :   str = 'Wilcox2006-klim',
        TurbulenceCutOffRatio        : float = 1e-8,
        TransitionMode               :   str = None,
                       ):
        r'''
        Set workflow attribute **Turbulence**, used for turbulence modeling parameters.

        Parameters
        ----------

        Viscosity_EddyMolecularRatio : float
            Ratio of :math:`\mu_t/\mu` used at freestream in order to set the 
            dissipation scale of turbulence models accordingly

        Level : float
            Level of freestream turbulence :math:`T_u`, typically used to set
            the first scale of turbulence models accordingly

        Model : str
            Choose the turbulence modeling strategy. This will set appropriate
            values for each solver. If more solver-specific adjustments are 
            desired, these shall be done using **SolverParameters** attribute.
            For RANS turbulence models, please note that we tend to use the same
            name as NASA's convention <https://turbmodels.larc.nasa.gov/>`__ .
            The covered models are (availability depends on the employed solver):

            * ``'Euler'``
                The Euler equations are solved

            * ``'DNS'`` or ``'ILES'`` or ``'Laminar'``
                The Navier-Stokes laminar equations are solved

            * ``'LES'``
                Use Large Eddy Simulation

            * ``'ZDES-1'``

            * ``'ZDES-2'``

            * ``'ZDES-3'``

            * ``'Wilcox2006-klim'``

            * ``'Wilcox2006-klim-V'``

            * ``'Wilcox2006'``

            * ``'Wilcox2006-V'``

            * ``'SST-2003'``

            * ``'SST-V2003'``

            * ``'SST'``

            * ``'SST-V'``

            * ``'BSL'``

            * ``'BSL-V'``

            * ``'SST-2003-LM2009'``

            * ``'SST-V2003-LM2009'``

            * ``'SSG/LRR-RSM-w2012'``

            * ``'smith'``

            * ``'SA'``

        TurbulenceCutOffRatio : float
            The minimum allowed value of the turbulence quantities based upon 
            the turbulence level :math:`T_u`

        '''
        self.Turbulence = self._get_comp(WorkflowInterface.set_Turbulence, self.get_default_values_from_local_signature())

    def set_BoundaryConditions(self, user_list : list):
        '''
        Set workflow attribute **BoundaryConditions** as a :class:`list`. 
        Each element is a :class:`dict` and corresponds to the boundary condition imposed on one given Family.

        For each :class:`dict`, the following keys are mandatory for all types of conditions:
            * Family (:class:`str`): Name of the Family on which the boundary condition is applied.
            * Type (:class:`str`): Type of condition. Available conditions are: 
                * Farfield
                * InflowStagnation
                * InflowMassFlow 
                * OutflowPressure 
                * OutflowSupersonic  
                * OutflowMassFlow 
                * OutflowRadialEquilibrium  
                * WallViscous  
                * WallViscousIsothermal      
                * WallInviscid        
                * Wall: depending the context (Euler or Navier-Stokes), it redirects to WallInviscid or WallViscous 
                * SymmetryPlane 
                * MixingPlane     
                * UnsteadyRotorStatorInterface 
                * ChorochronicInterface    

        Other arguments depends on the Type of boundary condition.
        '''
        self._set_by_user_list(self._method_name(), user_list)

    def add_to_BoundaryConditions(self,
        *,
        Family        : str   = None,
        Type          : str   = None,
        **kwargs
        ):
        parameters  = self.get_default_values_from_local_signature()
        parameters['kwargs'] = kwargs
        self.BoundaryConditions.append(self._get_comp(
            WorkflowInterface.add_to_BoundaryConditions, parameters))
 

    def set_SplittingAndDistribution(self,
        Strategy                         : str = 'AtPreprocess',
        Splitter                         : str = 'Cassiopee',
        Distributor                      : str = 'Cassiopee',
        ComponentsToSplit                : Union[ str,
                                                 None,
                                                 list ] = 'all',
        NumberOfParts                    : int = None,
        CoresPerNode                     : int = 48):
        self.SplittingAndDistribution = self._get_comp(
            WorkflowInterface.set_SplittingAndDistribution, self.get_default_values_from_local_signature())


    def set_Overset(self,
        depth                :  int = 2,
        optimizeOverlap      : bool = False,
        prioritiesIfOptimize : list = [],
        double_wall          :  int = 0,
        saveMaskBodiesTree   : bool = True,
        overset_in_CGNS      : bool = False, # see elsA #10545
        CHECK_OVERSET        : bool = True):
        self.Overset = self._get_comp(
            WorkflowInterface.set_Overset, self.get_default_values_from_local_signature())



    def set_Numerics(self,
        Scheme                    : str   = 'Jameson',
        TimeMarching              : str   = 'Steady',
        NumberOfIterations        : int   = 10000,
        MinimumNumberOfIterations : int   = 1000, 
        IterationAtInitialState   : int   = 1,
        TimeAtInitialState        : float = 0.0,
        TimeMarchingOrder         : int   = 2,
        TimeStep                  : float = None,
        CFL                       : Union[ float,
                                            dict] = 10.0,
                       ):
        '''
        Set workflow attribute **Numerics**

        Parameters
        ----------
        Scheme : str, optional
            Spatial scheme. 
            Available schemes are: 'Jameson', 'Roe'.
            By default 'Jameson' (for the basic Workflow). 
        TimeMarching : str, optional
            Type of simulation, available choices are: 'Steady', 'Unsteady'.
            By default 'Steady'
        NumberOfIterations : int, optional
            by default 10000
        MinimumNumberOfIterations : int, optional
            Number of iterations that will be done in all cases, 
            even if convergence criteria have been already reached.
            By default 1000
        IterationAtInitialState : int, optional
            by default 1
        TimeAtInitialState : float, optional
            by default 0.0
        TimeMarchingOrder : int, optional
            by default 2
        TimeStep : float, optional
            Useful only for unsteady simulation.
        CFL : Union[ float, dict], optional
            CFL number, by default 10.0.
            It could be a scalar or a linear ramp given as a dict. For example:

            >>> CFL = dict(EndIteration=300, StartValue=1., EndValue=30.)

            defines a ramp with CFL=1 at iteration 1 (could be modified with `StartIteration`)
            until CFL=30 at iteration 300.

        '''
        self.Numerics = self._get_comp(
            WorkflowInterface.set_Numerics, self.get_default_values_from_local_signature())
        self.check_time_marching()
        self.check_cfl()

    def check_time_marching(self):
        time_marching = self.Numerics['TimeMarching']
        if time_marching != 'Steady':
            if 'TimeStep' not in self.Numerics:
                msg = ('TimeStep must be defined to perform a simulation '
                    f'with TimeMarching={time_marching}')
                raise MolaUserAttributeError(msg)

    def check_cfl(self):
        cfl = self.Numerics['CFL'] if 'CFL' in self.Numerics else None
        if isinstance(cfl,dict): self.set_cfl(**cfl)

    def set_cfl(self,
            StartIteration : int   = None,
            *,
            EndIteration   : int   = 500,
            StartValue     : float = 1.0, 
            EndValue       : float = 10.0,
            ):
        cfl = dict(EndIteration=EndIteration,
                   StartValue=StartValue,
                   EndValue=EndValue)
        if StartIteration is None:
            cfl['StartIteration'] = self.Numerics['IterationAtInitialState'] 

        self.Numerics['CFL'].update(cfl)

    def set_BodyForceModeling(self, user_list : list):
        self._set_by_user_list(self._method_name(), user_list)

    def add_to_BodyForceModeling(self,
            ToBeImplmented : str = 'NotYetImplemented'):
        self.BodyForceModeling.append(self._get_comp(
            WorkflowInterface.add_to_BodyForceModeling, self.get_default_values_from_local_signature()))

    def set_Motion(self,
            motion_per_family_dict    : dict  = None):
        self.Motion = motion_per_family_dict
        # BUGGED
        # self.Motion = self._get_comp(WorkflowInterface.set_Motion, self.get_default_values_from_local_signature())

    def set_Initialization(self,
            Method    : str  = 'uniform',
            Source    : Union[     str,
                                  Tree,
                                  Base,
                                  Zone ]  = None,
            SourceContainer : str = None,
            ComputeWallDistanceAtPreprocess : bool = False,
            WallDistanceComputingTool : str = 'maia'):
        '''
        Set workflow attribute **Initialization**

        Parameters
        ----------
        Method : str, optional
            Available methods are: 
                * `'uniform'`: initialize flow with reference values as computed 
                  from **Fluid**, **Flow** and **Turbulence** attributes.
                * `'copy'`: initialize flow by copying the flow in the file given by **Source**.
                  Both meshes must be exactly the same.
                * `'interpolate'`: initialize flow by interpolating the flow from the file given by **Source**.

            By default 'uniform'
        Source : Union[     str, Tree, Base, Zone ], optional
            Source mesh, given as a file name or as a treelab Tree.
        SourceContainer : str, optional
            Container to consider in the source mesh, by default 'FlowSolution#Init'
        ComputeWallDistanceAtPreprocess : bool, optional
            If True, compute distances to walls during preprocess.
            By default False
        '''
        self.Initialization = self._get_comp(
            WorkflowInterface.set_Initialization, self.get_default_values_from_local_signature())

    def set_Extractions(self, user_list : list):
        '''
        Extractions are defined with the workflow attribute **Extractions** as a :class:`list`. 
        Each element is a :class:`dict` and corresponds to an extraction.

        For each extraction, at least one key is mandatory:
            * Type (:class:`str`)
        '''
        self._set_by_user_list(self._method_name(), user_list, several_add_tos=True,
            external_defaults=self.ExtractionsDefaults)

    def _set_by_user_list(self, method_name, user_list, several_add_tos=False,
            external_defaults=[]):
        attribute = method_name.replace('set_','')
        setattr(self, attribute, [])
        if user_list is None: return
        add_tos = self._get_add_to_methods_of_attribute(attribute)
        for user_component in user_list:
            if several_add_tos:
                try: Type = user_component['Type']
                except KeyError: raise MolaUserError(f'Must provide parameter "Type" for using {method_name}')
            else: 
                Type = attribute
            try: add_tos[Type](**user_component)
            except TypeError as e: raise MolaUserAttributeError(add_tos[Type], e)
            if external_defaults:
                workflow_component = getattr(self, attribute)[-1]
                default_component = self._get_default_component(external_defaults,
                                                                workflow_component)
                for key, value in default_component.items():
                    if key not in user_component:
                        workflow_component[key] = value

    def _get_default_component(self, DefaultComponents, UserComponent):
        for comp in DefaultComponents:
            if comp['ReferenceParameter'] in UserComponent: return copy.deepcopy(comp)
        return {}
    
    def add_to_Extractions_Residuals(self,
            Name : str = "Residuals",
            File : str = names.FILE_OUTPUT_1D,
            ExtractionPeriod : int = 1,
            SavePeriod : int = 100,
            Override : bool = True, # if False, will tag with iteration
            ExtractAtEndOfRun : bool = True,  # if True, extract and save when the simulation ends, whatever ExtractionPeriod and SavePeriod
            *,
            Type : str = 'Residuals',
            ):
        '''
        Extraction of global or local residuals
        '''
        self.Extractions.append(self._get_comp(
            WorkflowInterface.add_to_Extractions_Residuals, self.get_default_values_from_local_signature()))

    def add_to_Extractions_Integral(self,
            File : str = names.FILE_OUTPUT_1D,
            Name : str = 'ByFamily',  # if None, will be based on Source
            ExtractionPeriod : int = 1,
            SavePeriod : int = 100,
            Override : bool = True, # if False, will tag with iteration
            ExtractAtEndOfRun : bool = True,  # if True, extract and save when the simulation ends, whatever ExtractionPeriod and SavePeriod
            Frame : str = 'absolute',
            TimeAveragingFirstIteration : int = 1000,
            TimeAveragingIterations : int = 1000,
            PostprocessOperations : list = None,
            OtherOptions : dict = None,
            *,
            Type : str = 'Integral', # 'Integral',
            Source : str, # "BCWall", "MyFamilyBC"... TODO accept regex &| ?
            Fields : list, # accepts prefix avg- or std- Accept MOLA keywords "Force" and "Torque"
            ):
        '''
        Summation over a given source of the mesh, providing a scalar integral value
        '''
        if not Name: Name = Source
        if len(Fields) == 0: return
        self.Extractions.append(self._get_comp(
            WorkflowInterface.add_to_Extractions_Integral, self.get_default_values_from_local_signature()))

    def add_to_Extractions_Probe(self,
            Tolerance : float = 1e-2,
            Method : str = 'getNearestPointIndex',
            Fields : list = None, # accepts prefix avg- or std-
            File : str = names.FILE_OUTPUT_1D,
            Name : str = None, # if None, will be based on Position
            ExtractionPeriod : int = 1,
            SavePeriod : int = 100,
            Override : bool = True, # if False, will tag with iteration
            ExtractAtEndOfRun : bool = True,  # if True, extract and save when the simulation ends, whatever ExtractionPeriod and SavePeriod
            Frame : str = 'relative',
            TimeAveragingFirstIteration : int = 1000,
            TimeAveragingIterations : int = 1000,
            PostprocessOperations : list = None,
            OtherOptions : dict = None,
            *,
            Type : str = 'Probe',
            Position : Union [ list,
                              tuple,
                              np.ndarray ] = [0,0,0],
            ):
        '''
        Probe extraction 
        '''
        if not Name: 
            Name = f'Probe_{Position[0]:.4g}_{Position[1]:.4g}_{Position[2]:.4g}'
        if not Method in ['getNearestPointIndex', 'nearestNodes']:
            raise MolaUserError(f'The argument Method for Extraction of Type="Probe" must be either "getNearestPointIndex" or "nearestNodes"')
        self.Extractions.append(self._get_comp(
            WorkflowInterface.add_to_Extractions_Probe, self.get_default_values_from_local_signature()))

    def add_to_Extractions_BC(self,
            Fields : list = [],
            File : str = names.FILE_OUTPUT_2D,
            Name : str = 'ByFamily',  # if None, will be based on Source
            ExtractionPeriod : int = 100,
            SavePeriod : int = 100,
            Override : bool = True, # if False, will tag with iteration
            ExtractAtEndOfRun : bool = True,  # if True, extract and save when the simulation ends, whatever ExtractionPeriod and SavePeriod
            GridLocation : str = 'CellCenter',
            ContainersToTransfer : Union[ str, # accepts "all"
                                         list ] = names.CONTAINER_OUTPUT_FIELDS_AT_CENTER, 

            Frame : str = 'relative',
            TimeAveragingFirstIteration : int = 1000,
            TimeAveragingIterations : int = 1000,
            PostprocessOperations : list = None,
            OtherOptions : dict = None,
            *,
            Type : str = 'BC',
            Source : str # Family, BC... TODO accept regex ?
            ):
        '''
        Extraction at boundaries of the mesh
        '''
        self.Extractions.append(self._get_comp(
            WorkflowInterface.add_to_Extractions_BC, self.get_default_values_from_local_signature()))

    def add_to_Extractions_IsoSurface(self,
            Fields : list = None,
            File : str = names.FILE_OUTPUT_2D,
            Name : str = None, # if None, will be based on Source
            ExtractionPeriod : int = 100,
            SavePeriod : int = 100,
            Override : bool = True, # if False, will tag with iteration
            ExtractAtEndOfRun : bool = True,  # if True, extract and save when the simulation ends, whatever ExtractionPeriod and SavePeriod
            GridLocation : str = 'Vertex',
            ContainersToTransfer : Union[ str, # accepts "all"
                                         list ] = names.CONTAINER_OUTPUT_FIELDS_AT_VERTEX, 
            Frame : str = 'relative',
            TimeAveragingFirstIteration : int = 1000,
            TimeAveragingIterations : int = 1000,
            PostprocessOperations : list = None,
            OtherOptions : dict = None,
            *,
            Type : str = 'IsoSurface',
            IsoSurfaceField : str = 'CoordinateX', # a coordinate or a field or a Container/field
            IsoSurfaceValue : float = 0.0, 
            IsoSurfaceContainer : str = 'auto', 
            ):
        '''
        Extraction using an iso-surface operation
        '''
        if not Name:
            FieldName = IsoSurfaceField.replace('Coordinate','').replace('Radius', 'R').replace('ChannelHeight', 'H')
            Name = f"Iso_{FieldName}_{IsoSurfaceValue:.4g}"
        self.Extractions.append(self._get_comp(
            WorkflowInterface.add_to_Extractions_IsoSurface, self.get_default_values_from_local_signature()))

    def add_to_Extractions_Interpolation(self,
            Fields : list = None,
            File : str = names.FILE_OUTPUT_2D,
            Name : str = None, # if None, will be based on Source
            ExtractionPeriod : int = 100,
            SavePeriod : int = 100,
            InterpolationOrder : int = 0,
            Override : bool = True, # if False, will tag with iteration
            ExtractAtEndOfRun : bool = False,  # if True, extract and save when the simulation ends, whatever ExtractionPeriod and SavePeriod
            PostprocessOperations : list = None,
            OtherOptions : dict = None,
            *,
            Type : str = 'Interpolation',
            Source : Union[str,
                           Tree, 
                           Base,
                           Zone,
                           ]  = 'my_source_mesh.cgns',
            ):
        '''
        Extraction using an interpolation on a user-provided grid by file or in memory
        '''
        self.Extractions.append(self._get_comp(
            WorkflowInterface.add_to_Extractions_Interpolation, self.get_default_values_from_local_signature()))

    def add_to_Extractions_3D(self,
            Fields : list = None,
            File : str = names.FILE_OUTPUT_3D, 
            Name : str = None, # if None, will be based on Position
            ExtractionPeriod : int = 5000,
            SavePeriod : int = 5000,
            Frame : str = 'relative', # TODO add warning for fast (only 'absolute' possible)
            Override : bool = True, # if False, will tag with iteration
            ExtractAtEndOfRun : bool = True,  # if True, extract and save when the simulation ends, whatever ExtractionPeriod and SavePeriod
            Container : str = None,  # Default value depends on GridLocation
            GridLocation : str = 'Vertex',
            GhostCells : bool = False,
            TimeAveragingFirstIteration : int = 1000,
            TimeAveraging : bool = False,
            PostprocessOperations : list = None,
            OtherOptions : dict = None,
            *,
            Type : str = '3D',
            ):
        '''
        Fields (or sub-fields) extraction 
        '''
        if Container is None:
            if GridLocation == 'Vertex':
                Container = names.CONTAINER_OUTPUT_FIELDS_AT_VERTEX
            else:
                Container = names.CONTAINER_OUTPUT_FIELDS_AT_CENTER
        self.Extractions.append(self._get_comp(
            WorkflowInterface.add_to_Extractions_3D, self.get_default_values_from_local_signature()))
    
    def add_to_Extractions_Restart(self,
            Fields : list = None,
            File : str = names.FILE_INPUT_SOLVER, 
            ExtractionPeriod : int = 1000000000, # Only done at the end of the simulation
            SavePeriod : int = 1000000000,
            ExtractAtEndOfRun : bool = True,  # if True, extract and save when the simulation ends, whatever ExtractionPeriod and SavePeriod
            Frame : str = 'relative',
            Container : str = None, # if None will define automatic container names
            GridLocation : str = 'CellCenter',
            GhostCells : bool = False,
            *,
            Type : str = 'Restart',
            ):
        '''
        Fields used to restart a simulation 
        '''
        self.Extractions.append(self._get_comp(
            WorkflowInterface.add_to_Extractions_Restart, self.get_default_values_from_local_signature()))

    def add_to_Extractions_MemoryUsage(self,
            File : str = names.FILE_OUTPUT_1D,
            ExtractionPeriod : int = 100,
            SavePeriod : int = 1000,
            Override : bool = True, # if False, will tag with iteration
            ExtractAtEndOfRun : bool = True,  # if True, extract and save when the simulation ends, whatever ExtractionPeriod and SavePeriod
            *,
            Type : str = 'MemoryUsage',
            ):
        '''
        Extraction of memory usage
        '''
        self.Extractions.append(self._get_comp(
            WorkflowInterface.add_to_Extractions_MemoryUsage, self.get_default_values_from_local_signature()))
        
    def add_to_Extractions_TimeMonitoring(self,
            File : str = names.FILE_OUTPUT_1D,
            ExtractionPeriod : int = 1000000000, # Only done at the end of the simulation
            SavePeriod : int = 1000000000, # Only done at the end of the simulation
            Override : bool = True, # if False, will tag with iteration
            ExtractAtEndOfRun : bool = True,  # if True, extract and save when the simulation ends, whatever ExtractionPeriod and SavePeriod
            *,
            Type : str = 'TimeMonitoring',
            ):
        '''
        Extraction of time monitoring
        '''
        self.Extractions.append(self._get_comp(
            WorkflowInterface.add_to_Extractions_TimeMonitoring, self.get_default_values_from_local_signature()))
        
    def set_ExtractionsDefaults(self, user_list : list = None):
        self._set_by_user_list(self._method_name(), user_list)

    def add_to_ExtractionsDefaults(self,
        Type : str = None,
        File : str = None,
        ExtractionsPeriod : int = None,
        SavePeriod : int = None,
        Frame : str = None,
        Override : bool = None,
        Container : str = None,
        GridLocation : str = None,
        GhostCells : bool = None,
        TimeAveragingFirstIteration : int = None,
        TimeAveraging : bool = None,
        *,
        ReferenceParameter : str = 'File',
        ):
        self.ExtractionsDefaults.append(self._get_comp(
            WorkflowInterface.add_to_ExtractionsDefaults, self.get_default_values_from_local_signature()))

    def set_ConvergenceCriteria(self, user_list : list):
        self._set_by_user_list(self._method_name(), user_list)

    def add_to_ConvergenceCriteria(self,
        Necessary     : bool  = False,
        Sufficient    : bool  = True,
        *,
        ExtractionName: str,
        Variable      : str,
        Threshold     : float,
        ):
        self.ConvergenceCriteria.append(self._get_comp(
            WorkflowInterface.add_to_ConvergenceCriteria, self.get_default_values_from_local_signature()))

    def set_RunManagement(self,
        JobName : str = None,
        RunDirectory : Union[str, pathlib.PosixPath] = '.',
        NumberOfProcessors : int = MPI.COMM_WORLD.Get_size(),
        NumberOfThreads : int = 1,
        Machine : str = None,
        User : str = None,
        TimeLimit : Union[str, float] = None,
        QuitMarginBeforeTimeOutInSeconds : int = 300,
        LauncherCommand : str = 'auto',
        FilesAndDirectories : list = [],
        mola_target_path : str = None,
        Scheduler : str = None, # None : chosed auto. "SLURM": will launch sbatch; "local" will launch ./job
        SchedulerOptions : dict = None,
        AER : str = None,
        RemovePreviousRunDirectory : bool = False,
        ):
        '''
        Set workflow attribute **RunManagement** to handle job submission.

        Parameters
        ----------
        JobName : str, optional
            Name of the job (useful only for using a Scheduler), by default 'mola'
        RunDirectory : Union[str, pathlib.PosixPath], optional
            Path where the simulation will be done, 
            by default '.' (simulation is prepared in the current directory).
        NumberOfProcessors : int, optional
            Number of processors used to run the simulation, by default MPI.COMM_WORLD.Get_size()
        NumberOfThreads : int, optional
            :fas:`person-digging;sd-text-warning`
        Machine : str, optional
            Name of the machine where the simulation will be run.
            **RunDirectory** is relative to that machine. 
            If not given, an attempt to guess the destination machine will be done
            using **RunDirectory**, the current directory and environment setting. 
            If the machine cannot be guessed, localhost is taken by default.
        User : str, optional
            Username on the destination **Machine**, by default the same user than currently on localhost.
        TimeLimit : Union[str, float], optional
            Time limit for the simulation, either in seconds (:class:`float`) or as a :class:`str`
            like '00:30:00' (30min), '15:00' (15min), '1-10:00:00' (34h).
            The default value depends on the **Machine** and environment parameters. 
        QuitMarginBeforeTimeOutInSeconds : int, optional
            Margin in seconds before quitting the simulation, by default 300.
            When the simulation has run for **TimeLimit** - **QuitMarginBeforeTimeOutInSeconds**, 
            it won't make new iterations and the simulation try ending safely performing final extractions.
            It will be automatically submitted again.
        LauncherCommand : str, optional
            Command that will be executed after preprocess to run the simulation (on the destination **Machine**).
            If not providing, the default value 'auto' corresponds to:
                * with `Scheduler='bash'`:  cd <RunDirectory>; sbatch :mola_name:`FILE_JOB`
                * with `Scheduler='SLURM'`: cd <RunDirectory>; sbatch :mola_name:`FILE_JOB`

            It is possible to run a more sophisticated command if needed with this attribute **LauncherCommand**.            
        FilesAndDirectories : list, optional
            Files and directories to copy in **RunDirectory**, by default []
        mola_target_path : str, optional
            :fas:`person-digging;sd-text-warning`
        Scheduler : str, optional
            Job scheduler, like SLURM, to use to run the simulation. 
            The default value depends on the **Machine** and environment parameters. 
        SchedulerOptions : dict, optional
            Parameters for the scheduler, with scheduler specific names.
        AER : str, optional
            AER number for simulation on sator
        RemovePreviousRunDirectory : bool, optional
            Only used for a simulation on a remote machine. If True, remove the previous RunDirectory before 
            preprocessing the case. Default value is False.
        '''
        RunDirectory = str(RunDirectory)
        self.RunManagement = self._get_comp(
            WorkflowInterface.set_RunManagement, self.get_default_values_from_local_signature())
        
    def set_SolverParameters(self, **kwargs):
        # no check on this attribute, because it is solver dependent. 
        # It allows to replace a solver parameter by a user defined value, without checking.
        self.SolverParameters = kwargs
            
    def __str__(self, keep_args=None, maxlevel=1000):
        
        def get_interface_text(cls, indent="    ", maxlevel=maxlevel):

            def process_signature_per_class_to_text(signature_per_class):
                txt = ''
                parent_signature = ''
                for class_txt, signature_txt in reversed(list(signature_per_class.items())):
                    if parent_signature == signature_txt:
                        signature_per_class[class_txt] = ''
                    parent_signature = signature_txt

                level = 0
                for class_txt, signature_txt in signature_per_class.items():
                    if level == maxlevel: break
                    if signature_txt:
                        level += 1
                        indent_local = " " * (level * len(indent))
                        if txt.endswith('\n'):
                            txt += indent_local[:-2]+ '↳ which is a specialization of ' + class_txt +':\n'
                        elif not txt.endswith(' → '):
                            txt += indent_local + 'parameters provided by ' + class_txt +' (highest priority):\n'
                        else:
                            txt += class_txt +':\n'

                        for line in signature_txt.split('\n'):
                            txt += indent_local + line + '\n'
                    else:
                        if txt.endswith('\n') or txt == '':
                            indent_local = " " * (level * len(indent))
                            txt += indent_local + 'parameters provided by ' + class_txt + ' → '
                        else:
                            txt += class_txt + ' → '

                return txt

            def get_signature_per_class_of_setters(param_name) -> dict:
                setter_name = 'set_'+param_name
                try:
                    setter_method = getattr(self,setter_name)
                except:
                    raise MolaException(f'Must implement interface for argument "{param_name}" using method "{setter_name}" in {self.Name}')
                queue = [(cls, 0)]
                signature_per_class = {}
                while queue:
                    current_cls, level = queue.pop(0)
                    setter_method = getattr(current_cls, setter_name)
                    signature = get_signature(setter_method)
                    signature_non_empty = not not signature.split()
                    if signature_non_empty:
                        key = current_cls.__name__
                        signature_per_class[key] = ''
                        for line in signature.split('\n'):
                            signature_per_class[key] += line + '\n'

                    for base_cls in current_cls.__bases__:
                        if len(base_cls.__bases__) > 0:
                            queue.append((base_cls, level + 1))
                return signature_per_class
            
            def get_signature_per_class_of_add_to(add_to_method) -> dict:
                queue = [(cls, 0)]
                signature_per_class = {}
                while queue:
                    current_cls, level = queue.pop(0)
                    add_to_method_of_current_cls = getattr(current_cls, add_to_method.__name__)
                    signature = get_signature(add_to_method_of_current_cls)
                    signature_non_empty = not not signature.split()
                    if signature_non_empty:
                        key = current_cls.__name__
                        signature_per_class[key] = ''
                        for line in signature.split('\n'):
                            signature_per_class[key] += line + '\n'

                    for base_cls in current_cls.__bases__:
                        if len(base_cls.__bases__) > 0:
                            queue.append((base_cls, level + 1))
                return signature_per_class
            
            txt = ''
            signature = inspect.signature(WorkflowInterface.__init__)
            for param in signature.parameters.values():
                param_name = param.name
                if param_name in self._fake_attributes: continue
                if keep_args is not None and param_name not in keep_args: continue

                txt += f'Attribute \033[4m\033[1m{param_name}\033[0m is set using:\n'

                signature_per_class = get_signature_per_class_of_setters(param_name)
                txt += process_signature_per_class_to_text(signature_per_class)


                add_to_methods_from_type = self._get_add_to_methods_of_attribute(param_name)
                several_add_to_methods = len(add_to_methods_from_type) > 1

                for Type, add_to_method in add_to_methods_from_type.items():
                    txt += f"where each item is a {CYAN}dict{ENDC} with these authorized keys:\n"
                    if several_add_to_methods:
                        txt += f'if {BOLD}Type{ENDC} ({CYAN}str{ENDC}) == {PINK}"{Type}"{ENDC}\n'

                    signature_per_class = get_signature_per_class_of_add_to(add_to_method)
                    txt += process_signature_per_class_to_text(signature_per_class)

            return txt

        banner = '='*(18+len(self.Name)) + '\n'
        txt = banner
        txt += f'User interface of {BOLD}{self.Name}{ENDC}\n'
        txt += banner
        txt += 'Parameters documentation is given with the following template:\n'
        txt += f'   {BOLD}name{ENDC} ({CYAN}allowed types{ENDC}) : {PINK}default value{ENDC}\n\n'

        return txt + get_interface_text(type(self))

    def _get_add_to_methods_of_attribute(self, attribute):
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        add_to_methods_from_type = dict()
        for method_name, method in methods:
            if not method_name.startswith('add_to_'): continue
            split_name = method_name.replace('add_to_','').split('_')
            if split_name[0] == attribute:
                add_to_methods_from_type[split_name[-1]] = method
        return add_to_methods_from_type
    
    def transfer_attributes_to_workflow(self, workflow):
        if not workflow: return
        for attr_name, attr_value in vars(self).items():
            if not callable(attr_value):
                setattr(workflow, attr_name, attr_value)

    @staticmethod
    def _method_name():
        return inspect.currentframe().f_back.f_code.co_name

    @staticmethod
    def _get_comp(fun, kwargs):
        signature = inspect.signature(fun)
        parameter_annotations = get_type_hints(fun)
        new_component = dict()
        for name in list(signature.parameters):
            if name == 'self': 
                continue
            elif name == 'kwargs' and kwargs:
                # last possible parameters in the signature
                # --> update new_component with all that remains in kwargs
                new_component.update(kwargs['kwargs'])
                break

            try:
                value = kwargs[name]
            except KeyError:
                raise MolaException(f'parameter {name} was not implemented in interface {fun.__name__}. \nkwargs={kwargs}\nnew_component={new_component}')
            if value is not None:
                expected_type = parameter_annotations.get(name)
                if expected_type:
                    if hasattr(expected_type, '__origin__') and \
                        expected_type.__origin__ is Union:
                        expected_types = expected_type.__args__
                        if not any(isinstance(value, t) for t in expected_types):
                            raise TypeError(f'argument {name} was expected to be one of the following types: {expected_types}, but got {type(value)}')
                    elif not isinstance(value, expected_type):
                        raise TypeError(f'argument {name} was expected to be type: {expected_type}, but got {type(value)}')
                new_component[name] = value        
       
        return new_component

    @staticmethod
    def get_default_values_from_local_signature():
        # Get the current frame (frame where this function is called)
        frame = inspect.currentframe().f_back
        # Get the arguments from the calling frame
        locals_dict = frame.f_locals
        locals_dict.pop("self", None)
        locals_dict.pop("kwargs", None)
        locals_dict.pop("__class__", None)
        locals_dict.pop("self.get_default_values_from_local_signature", None)
        kwargs = {key: locals_dict[key] for key in locals_dict if key not in locals_dict.get("args", [])}
        return kwargs

    @staticmethod
    def get_method_kwargs_with_defaults(method):
        """
        Returns a dictionary containing all keyword arguments of a method,
        including their default values.
        """
        signature = inspect.signature(method)
        kwargs_with_defaults = {}
        for param_name, param in signature.parameters.items():
            if param.default != inspect.Parameter.empty:
                kwargs_with_defaults[param_name] = param.default
        return kwargs_with_defaults

    @staticmethod
    def get_argument_types(func: Union[Callable, type]) -> Dict[str, Union[type, Union[type, None]]]:
        """
        Get argument types of a function or method.

        Args:
            func (Union[Callable, type]): Function or method.

        Returns:
            Dict[str, Union[type, Union[type, None]]]: Dictionary mapping argument names to their expected types.
        """
        if isinstance(func, type):
            sig = inspect.signature(func.__init__)
        else:
            sig = inspect.signature(func)

        arg_types = {}
        for param in sig.parameters.values():
            if param.annotation != param.empty:
                arg_types[param.name] = param.annotation
            else:
                arg_types[param.name] = None

        return arg_types
    