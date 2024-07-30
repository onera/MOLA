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
import multiprocessing
from mpi4py import MPI
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


class WorkflowInterface(object):

    def __init__(self, workflow=None,
            tree=None,
            Solver : str = os.environ.get('MOLA_SOLVER'),
            RawMeshComponents : list = None,
            Fluid : dict = None,
            Flow : dict = None,
            Turbulence : dict = None,
            BoundaryConditions : list = None,
            SplittingAndDistribution : dict = None,
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

        self._workflow_parameters_container_ = names.CONTAINER_WORKLFOW_PARAMETERS

        self.Name = workflow.Name if workflow else self.__class__.__name__
        self.set_attributes(attributes)
        self.transfer_attributes_to_workflow(workflow)



    def set_attributes(self, attributes, skip_attributes=['self','tree','workflow']):

        expected_attribute_types = self.get_argument_types(WorkflowInterface.__init__)
        
        for attribute_name, user_input in attributes.items():
            if attribute_name in skip_attributes: continue
        
            try:
                expected_type = expected_attribute_types[attribute_name]
            except KeyError:
                raise MolaException(f'attribute_name={attribute_name} not implemented from {self.Name} (expected {list(expected_attribute_types)})')

            if user_input is None: user_input = expected_type()
            
            if not isinstance(user_input, expected_type):
                raise MolaUserError(f'attribute {attribute_name} must be of type {expected_type}')
            
            method = getattr(self, 'set_'+attribute_name)

            if expected_type is dict:
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

    def set_Solver(self, solver_name : str):
        self.Solver = solver_name.lower()
        

    def add_to_RawMeshComponents(self,
        Mesher           : str  = None,
        Unit             : str  = 'm',
        CleaningMacro    : str  = None,
        Families         : list = None,
        Positioning      : list = None,
        Connection       : list = None,
        OversetOptions   : dict = None,
        *,
        Name             : str,
        Source           : Union[ str, Tree, Base, Zone],
        ):
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
        self.Fluid = self._get_comp(WorkflowInterface.set_Fluid, self.get_default_values_from_local_signature())


    def set_Flow(self,
            Generator : str = 'External_rho_V_T',
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
        self.Turbulence = self._get_comp(WorkflowInterface.set_Turbulence, self.get_default_values_from_local_signature())

    def set_BoundaryConditions(self, user_list : list):
        self._set_by_user_list(self._method_name(), user_list)

    def add_to_BoundaryConditions(self,
        Pressure      : float = None,
        MassFlow      : float = None,
        Motion        : dict  = None, # TODO check this
        LinkedFamily  : str   = None,
        *,
        Family        : str   = None,
        Type          : str   = None,
        ):
        self.BoundaryConditions.append(self._get_comp(
            WorkflowInterface.add_to_BoundaryConditions, self.get_default_values_from_local_signature()))

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

        self.Numerics['CFL'] = cfl

    def set_BodyForceModeling(self, user_list : list):
        self._set_by_user_list(self._method_name(), user_list)

    def add_to_BodyForceModeling(self,
            ToBeImplmented : str = 'NotYetImplemented'):
        self.BodyForceModeling.append(self._get_comp(
            WorkflowInterface.add_to_BodyForceModeling, self.get_default_values_from_local_signature()))

    def set_Motion(self,
            motion_per_family_dict    : dict  = None):
        self.Motion = self._get_comp(WorkflowInterface.set_Motion, self.get_default_values_from_local_signature())

    def set_Initialization(self,
            Method    : str  = 'uniform',
            Source    : Union[     str,
                                  Tree,
                                  Base,
                                  Zone ]  = None,
            KeepTurbulentDistance    : bool  = False):
        self.Initialization = self._get_comp(
            WorkflowInterface.set_Initialization, self.get_default_values_from_local_signature())

    def set_Extractions(self, user_list : list):
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
            Fields : list = None, # accepts prefix avg- or std-
            File : str = names.FILE_OUTPUT_1D,
            Name : str = None, # if None, will be based on Source
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
            Type : str = 'Integral',
            Source : str = 'BCWall', # "BCWall", "MyFamilyBC"... TODO accept regex &| ?
            ):
        '''
        Summation over a given source of the mesh, providing a scalar integral value
        '''
        if not Name: Name = Source
        self.Extractions.append(self._get_comp(
            WorkflowInterface.add_to_Extractions_Integral, self.get_default_values_from_local_signature()))

    def add_to_Extractions_Probe(self,
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
        self.Extractions.append(self._get_comp(
            WorkflowInterface.add_to_Extractions_Probe, self.get_default_values_from_local_signature()))

    def add_to_Extractions_BC(self,
            Fields : list = [],
            File : str = names.FILE_OUTPUT_2D,
            Name : str = 'ByFamily',  #None, # if None, will be based on Source
            ExtractionPeriod : int = 100,
            SavePeriod : int = 100,
            Override : bool = True, # if False, will tag with iteration
            ExtractAtEndOfRun : bool = False,  # if True, extract and save when the simulation ends, whatever ExtractionPeriod and SavePeriod
            GridLocation : str = 'CellCenter',
            Frame : str = 'relative',
            TimeAveragingFirstIteration : int = 1000,
            TimeAveragingIterations : int = 1000,
            PostprocessOperations : list = None,
            OtherOptions : dict = None,
            *,
            Type : str = 'BC',
            Source : str = 'MyFamily', # Family, BC... TODO accept regex ?
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
            ExtractAtEndOfRun : bool = False,  # if True, extract and save when the simulation ends, whatever ExtractionPeriod and SavePeriod
            GridLocation : str = 'Vertex',
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
            Frame : str = 'relative',
            Override : bool = True, # if False, will tag with iteration
            ExtractAtEndOfRun : bool = False,  # if True, extract and save when the simulation ends, whatever ExtractionPeriod and SavePeriod
            Container : str = 'FlowSolution#Output', 
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
        ExtractionName: str   = 'MyFamily',
        Variable      : str   = 'std-MyVariable',
        Threshold     : float = 1e-3,
        ):
        self.ConvergenceCriteria.append(self._get_comp(
            WorkflowInterface.add_to_ConvergenceCriteria, self.get_default_values_from_local_signature()))

    def set_RunManagement(self,
        JobName : str = None,
        RunDirectory : Union[str, pathlib.PosixPath] = '.',
        NumberOfProcessors : int = MPI.COMM_WORLD.Get_size(),
        NumberOfThreads : int = multiprocessing.cpu_count(),
        Machine : str = None,
        User : str = None,
        TimeOutInSeconds : float = None,
        SecondsMarginForQuitBeforeTimeOut : float = None,
        LauncherCommand : str = 'auto',
        FilesAndDirectories : list = [],
        mola_target_path : str = None,
        AER : str = None,
        ):
        RunDirectory = str(RunDirectory)
        self.RunManagement = self._get_comp(
            WorkflowInterface.set_RunManagement, self.get_default_values_from_local_signature())
        
    def set_SolverParameters(self, **kwargs):
        # no check on this attribute, because it is solver dependent. 
        # It allows to replace a solver parameter by a user defined value, without checking.
        self.SolverParameters = kwargs
            
    def __str__(self, maxlevel=1000):
        
        def get_interface_text(cls, indent="    ", skip_args=['self','tree','workflow'], maxlevel=maxlevel):

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
                    raise MolaException(f'Must implement interface for argument "{param_name}" using method "{setter_name}" in {self.Name}\n{skip_args}')
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
                if param_name in skip_args: continue

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

        txt = f'User interface of {BOLD}{self.Name}{ENDC}:\n'
        txt += f'{BOLD}name{ENDC} ({CYAN}allowed types{ENDC}) : {PINK}default value{ENDC}\n\n'

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
        for name, param in signature.parameters.items():
            if name == 'self': continue
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
    