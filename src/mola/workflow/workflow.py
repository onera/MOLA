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
import numpy as np
import copy
from treelab import cgns
import inspect
from typing import Union, get_type_hints
from mola.logging import (mola_logger,
                       MolaException,
                       MolaUserError,
                       MolaUserAttributeError,
                       redirect_streams_to_logger,
                       get_signature)
from mola.logging.formatters import BOLD, RED, CYAN, PINK, YELLOW, ENDC
from  mola.cfd.preprocess.mesh import (reader,
                                    positioning,
                                    connect,
                                    split,
                                    families)
from  mola.cfd.preprocess import (flow_generators,
                               boundary_conditions,
                               initialization,
                               motion,
                               cfd_parameters,
                               extractions,
                               write_cfd_files)
from mola import server as SV 
from mola.cfd.postprocess import remove_cfd_files
from mola.cfd.compute import compute


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        elif isinstance(v, list):
            d[k].extent(v)
        else:
            d[k] = v
    return d

def show_signature_if_TypeError(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError as e:
            raise MolaUserAttributeError(func, e)
    return wrapper

class Workflow(object):

    def __init__(self, 
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
            ExtractionsDefaults : dict = None,
            Extractions : list = None,
            ConvergenceCriteria : list = None,
            RunManagement : dict = None,
            ApplicationContext : dict = None,
            ):
            # Monitoring=dict(SaveSignalsPeriod=30,
            #                 SaveExtractionsPeriod=30,
            #                 SaveFieldsPeriod=30,
            #                 SaveBodyForcePeriod=2000,
            #                 TagExtractionsWithIteration='auto'),


        try: self.set_Solver(solver_name=Solver)
        except TypeError as e: raise MolaUserAttributeError(self.set_Solver, e)

        self._workflow_parameters_container_ = 'WorkflowParameters'

        self.Name = self.__class__.__name__
        self.tree = tree

        if self.tree is not None:
            self.get_workflow_parameters_from_tree()

        else:

            try: self.set_RawMeshComponents(RawMeshComponents)
            except TypeError as e: raise MolaUserAttributeError(self.set_RawMeshComponents, e)
            
            if ApplicationContext is None: ApplicationContext = dict()
            try: self.set_ApplicationContext(**ApplicationContext)
            except TypeError as e: raise MolaUserAttributeError(self.set_ApplicationContext, e)

            if Fluid is None: Fluid = dict()
            try: self.set_Fluid(**Fluid)
            except TypeError as e: raise MolaUserAttributeError(self.set_Fluid, e)

            if Flow is None: Flow = dict()
            try: self.set_Flow(**Flow)
            except TypeError as e: raise MolaUserAttributeError(self.set_Flow, e)

            if Turbulence is None: Turbulence = dict()
            try: self.set_Turbulence(**Turbulence)
            except TypeError as e: raise MolaUserAttributeError(self.set_Turbulence, e)

            try: self.set_BoundaryConditions(BoundaryConditions)
            except TypeError as e: raise MolaUserAttributeError(self.set_BoundaryConditions, e)

            if SplittingAndDistribution is None: SplittingAndDistribution = dict()
            try: self.set_SplittingAndDistribution(**SplittingAndDistribution)
            except TypeError as e: raise MolaUserAttributeError(self.set_SplittingAndDistribution, e)

            if Numerics is None: Numerics = dict()
            try: self.set_Numerics(**Numerics)
            except TypeError as e: raise MolaUserAttributeError(self.set_Numerics, e)

            try: self.set_BodyForceModeling(BodyForceModeling)
            except TypeError as e: raise MolaUserAttributeError(self.set_BodyForceModeling, e)

            if Motion is None: Motion = dict()
            try: self.set_Motion(**Motion)
            except TypeError as e: raise MolaUserAttributeError(self.set_Motion, e)

            if Initialization is None: Initialization = dict()
            try: self.set_Initialization(**Initialization)
            except TypeError as e: raise MolaUserAttributeError(self.set_Initialization, e)

            try: self.set_ExtractionsDefaults(ExtractionsDefaults)
            except TypeError as e: raise MolaUserAttributeError(self.set_ExtractionsDefaults, e)

            try: self.set_Extractions(Extractions)
            except TypeError as e: raise MolaUserAttributeError(self.set_Extractions, e)

            try: self.set_ConvergenceCriteria(ConvergenceCriteria)
            except TypeError as e: raise MolaUserAttributeError(self.set_ConvergenceCriteria, e)

            if RunManagement is None: RunManagement = dict()
            try: self.set_RunManagement(**RunManagement)
            except TypeError as e: raise MolaUserAttributeError(self.set_RunManagement, e)

            self.SolverParameters = dict()

        self._FlowGenerator = self.get_flow_generator(self.Flow['Generator'])

    def check_consistency_between_solver_and_environment(self):
        requested_solver = self.Solver
        env_solver = os.environ.get('MOLA_SOLVER')
        if requested_solver != env_solver:
            raise MolaException((f'the requested solver "{requested_solver}" does not'
                f'match the type of environment "{env_solver}"'))

    def set_Solver(self, solver_name : str):
        self.Solver = solver_name.lower()
        self.check_consistency_between_solver_and_environment()

    def add_to_RawMeshComponents(self,
        Mesher        : str  = None,
        CleaningMacro : str  = None,
        Families      : list = None,
        Positioning   : list = None,
        Connection    : list = None,
        OversetOptions: dict = None,
        *,
        Name          : str,
        Source        : Union[ str,
                               cgns.tree.Tree,
                               cgns.base.Base,
                               cgns.zone.Zone ],
        ):
        self.RawMeshComponents.append(self._get_comp(self.add_to_RawMeshComponents, self.repack_kwargs()))

    def set_RawMeshComponents(self, user_list : list):
        self._set_by_user_list(self._method_name(), user_list)
    
    def set_ApplicationContext(self):
        '''
        this method is virtual and shall be reimplemented in inherited workflows
        '''
        self.ApplicationContext = self._get_comp(self.set_ApplicationContext, self.repack_kwargs())


    def set_Fluid(self,
            Gamma                      : float =  1.4,
            IdealGasConstant           : float =  287.053,
            Prandtl                    : float =  0.72,
            PrandtlTurbulent           : float =  0.9,
            SutherlandConstant         : float = 110.4,
            SutherlandViscosity        : float = 1.78938e-05,
            SutherlandTemperature      : float = 288.15):
        self.Fluid = self._get_comp(self.set_Fluid, self.repack_kwargs())


    def set_Flow(self,
            Generator : str = 'External_rho_V_T',
            Velocity               : float = 1.0,
            # Parameters relevant to InternalFlowGenerator
            MassFlow               : float = None,
            Mach                   : float = None,
            PressureStagnation     : float = None,
            TemperatureStagnation  : float = None,
            IdealGasConstant       : float = None,
            Gamma                  : float = None,
            # Parameters relevant to ExternalFlowGenerator
            Direction              : Union [ list,
                                            tuple,
                                       np.ndarray ] = [1, 0, 0],
            Density                : float = 1.225,
            Temperature            : float = 288.15,
            VelocityUsedForScalingAndTurbulence : float = None
            ):

        self.Flow = self._get_comp(self.set_Flow, self.repack_kwargs())

        if 'Direction' in self.Flow:
            self.Flow['Direction'] = np.array(self.Flow['Direction'], dtype=float)
            if len(self.Flow['Direction']) != 3:
                raise MolaUserAttributeError('Direction argument must be a 3-float list, tuple or numpy')
            
        if 'VelocityUsedForScalingAndTurbulence' not in self.Flow:
            V = np.abs(self.Flow['Velocity'])
            if V < 1e-5:
                raise MolaUserError('Velocity is very low. You must set a positive value for VelocityUsedForScalingAndTurbulence')
            else:
                self.Flow['VelocityUsedForScalingAndTurbulence'] = V
        elif self.Flow['VelocityUsedForScalingAndTurbulence'] <= 0:
            raise MolaUserError('You must provide positive value for VelocityUsedForScalingAndTurbulence')

    def set_Turbulence(self,
        Viscosity_EddyMolecularRatio : float = 0.1,
        Level                        : float = 0.001,
        Model                        :   str = 'Wilcox2006-klim',
        TurbulenceCutOffRatio        : float = 1e-8,
        TransitionMode               :   str = None,
                       ):
        self.Turbulence = self._get_comp(self.set_Turbulence, self.repack_kwargs())

    def set_BoundaryConditions(self, user_list : list):
        self._set_by_user_list(self._method_name(), user_list)


    def add_to_BoundaryConditions(self,
        Pressure      : float = None,
        Motion        : dict  = None, # TODO check this
        LinkedFamily  : str   = None,
        *,
        Family        : str   = None,
        Type          : str   = None,
        ):
        self.BoundaryConditions.append(self._get_comp(self.add_to_BoundaryConditions, self.repack_kwargs()))


    def set_SplittingAndDistribution(self,
        Strategy                         : str = 'AtPreprocess',
        Splitter                         : str = 'Cassiopee',
        Distributor                      : str = 'Cassiopee',
        ComponentsToSplit                : Union[ str,
                                                 None,
                                                 list ] = 'all',
        NumberOfProcessors               : Union[ str,
                                                  int]  = 'auto',
        MinimumAllowedNodes              : int = 1,
        MaximumAllowedNodes              : int = 20,
        MaximumNumberOfPointsPerNode     : int = int(1e9),
        CoresPerNode                     : int = 48,
        DistributeExclusivelyOnFullNodes : bool = True,
                       ):
        self.SplittingAndDistribution = self._get_comp(self.set_SplittingAndDistribution, self.repack_kwargs())


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
        self.Numerics = self._get_comp(self.set_Numerics, self.repack_kwargs())
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
        self.BodyForceModeling.append(self._get_comp(self.add_to_BodyForceModeling, self.repack_kwargs()))

    def set_Motion(self,
            motion_per_family_dict    : dict  = None):
        self.Motion = self._get_comp(self.set_Motion, self.repack_kwargs())

    def set_Initialization(self,
            Method    : str  = 'uniform',
            Source    : Union[     str,
                        cgns.tree.Tree,
                        cgns.base.Base,
                        cgns.zone.Zone ]  = None,
            KeepTurbulentDistance    : bool  = False):
        self.Initialization = self._get_comp(self.set_Initialization, self.repack_kwargs())

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

    def add_to_Extractions_Integral(self,
            Fields : list = None, # accepts prefix avg- or std-
            File : str = 'signals.cgns', # if None will use signals.cgns
            Name : str = None, # if None, will be based on Source
            ExtractionPeriod : int = 1,
            SavePeriod : int = 100,
            Override : bool = True, # if False, will tag with iteration
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
        self.Extractions.append(self._get_comp(self.add_to_Extractions_Integral, self.repack_kwargs()))

    def add_to_Extractions_Probe(self,
            Fields : list = None, # accepts prefix avg- or std-
            File : str = 'signals.cgns',
            Name : str = None, # if None, will be based on Position
            ExtractionPeriod : int = 1,
            SavePeriod : int = 100,
            Override : bool = True, # if False, will tag with iteration
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
        self.Extractions.append(self._get_comp(self.add_to_Extractions_Probe, self.repack_kwargs()))

    def add_to_Extractions_BC(self,
            Fields : list = None,
            File : str = 'surfaces.cgns',
            Name : str = None, # if None, will be based on Source
            ExtractionPeriod : int = 100,
            SavePeriod : int = 100,
            Override : bool = True, # if False, will tag with iteration
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
        self.Extractions.append(self._get_comp(self.add_to_Extractions_BC, self.repack_kwargs()))


    def add_to_Extractions_IsoSurface(self,
            Fields : list = None,
            File : str = 'surfaces.cgns', # if None will use surfaces.cgns
            Name : str = None, # if None, will be based on Source
            ExtractionPeriod : int = 100,
            SavePeriod : int = 100,
            Override : bool = True, # if False, will tag with iteration
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
            ):
        '''
        Extraction using an iso-surface operation
        '''
        self.Extractions.append(self._get_comp(self.add_to_Extractions_IsoSurface, self.repack_kwargs()))


    def add_to_Extractions_Interpolation(self,
            Fields : list = None,
            File : str = 'surfaces.cgns', # if None will use surfaces.cgns
            Name : str = None, # if None, will be based on Source
            ExtractionPeriod : int = 100,
            SavePeriod : int = 100,
            InterpolationOrder : int = 0,
            Override : bool = True, # if False, will tag with iteration
            PostprocessOperations : list = None,
            OtherOptions : dict = None,
            *,
            Type : str = 'Interpolation',
            Source : Union[str,
                           cgns.tree.Tree, 
                           cgns.base.Base,
                           cgns.zone.Zone,
                           ]  = 'my_source_mesh.cgns',
            ):
        '''
        Extraction using an interpolation on a user-provided grid by file or in memory
        '''
        self.Extractions.append(self._get_comp(self.add_to_Extractions_Interpolation, self.repack_kwargs()))


    def add_to_Extractions_3D(self,
            Fields : list = None,
            File : str = 'fields.cgns', # if None will use fields.cgns
            Name : str = None, # if None, will be based on Position
            ExtractionPeriod : int = 5000,
            SavePeriod : int = 5000,
            Frame : str = 'relative',
            Override : bool = True, # if False, will tag with iteration
            Container : str = None, # if None will define automatic container names
            GridLocation : str = 'CellCenter',
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
        self.Extractions.append(self._get_comp(self.add_to_Extractions_3D, self.repack_kwargs()))


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
        self.ExtractionsDefaults.append(self._get_comp(self.add_to_ExtractionsDefaults, self.repack_kwargs()))

    def set_ConvergenceCriteria(self, user_list : list):
        self._set_by_user_list(self._method_name(), user_list)

    def add_to_ConvergenceCriteria(self,
        Necessary     : bool  = False,
        Sufficient    : bool  = True,
        *,
        Family        : str   = 'MyFamily',
        Variable      : str   = 'std-MyVariable',
        Threshold     : float = 1e-3,
        ):
        self.ConvergenceCriteria.append(self._get_comp(self.add_to_ConvergenceCriteria, self.repack_kwargs()))


    def set_RunManagement(self,
        JobName : str = None,
        RunDirectory : str = '.',
        NumberOfProcessors : int = 1,
        Machine : int = None,
        TimeOutInSeconds : float = None,
        SecondsMarginForQuitBeforeTimeOut : float = None,
        LauncherCommand : str = 'auto',
        FilesAndDirectories : list = [],
        mola_target_path : str = None,
                          ):
        self.RunManagement = self._get_comp(self.set_RunManagement, self.repack_kwargs())


    def write_tree(self, filename='main.cgns'):
        if not self.tree: 
            self.tree = cgns.Tree()
        with redirect_streams_to_logger(mola_logger):
            self.tree.save(filename)

    def convert_to_dict(self):
        params= dict()
        for a in list(self.__dict__):
            if not a.startswith('_') and a != 'tree':
                att = getattr(self,a)
                if not callable(att):
                    params[a] = att
        return params

    def print(self):
        print(self.__str__())
    
    def __str__(self):
        params= self.convert_to_dict()
        import pprint
        return pprint.pformat(params)

    def get_workflow_parameters_from_tree(self):
        
        self.tree = cgns.load(self.tree)
        
        workflow_parameters = self.tree.getParameters(
            self._workflow_parameters_container_, transform_numpy_scalars=True)
        
        for parameter in workflow_parameters:
            setattr(self, parameter, workflow_parameters[parameter])

        try:
            self._FlowGenerator = self.get_flow_generator(self.Flow['Generator'])
        except:
            self._FlowGenerator = None

        if self.RawMeshComponents is None: self.RawMeshComponents = []
        if self.ApplicationContext is None: self.ApplicationContext = dict()
        if self.Fluid is None: self.Fluid = dict()
        if self.Flow is None: self.Flow = dict()
        if self.Turbulence is None: self.Turbulence = dict()
        if self.BoundaryConditions is None: self.BoundaryConditions = []
        if self.SplittingAndDistribution is None: self.SplittingAndDistribution = dict()
        if self.Numerics is None: self.Numerics = dict()
        if self.BodyForceModeling is None: self.BodyForceModeling = []
        if self.Motion is None: self.Motion = dict()
        if self.Initialization is None: self.Initialization = dict(method='uniform')
        if self.ExtractionsDefaults is None: self.ExtractionsDefaults = []
        if self.Extractions is None: self.Extractions = []
        if self.ConvergenceCriteria is None: self.ConvergenceCriteria = []
        if self.RunManagement is None: self.RunManagement = dict()
        if self.SolverParameters is None: self.SolverParameters = dict()

    def set_workflow_parameters_in_tree(self):
        if not self.tree: self.tree = cgns.Tree()

        params= self.convert_to_dict()
        self.tree.setParameters(self._workflow_parameters_container_,
                                **params)
    
    def set_workflow_parameters_in_file(self, filename='setup.py'):

        import mola
        import pprint
        Lines = '#!/usr/bin/env python3\n'
        Lines+= f"'''\nMOLA {mola.__version__} setup.py file automatically generated in PREPROCESS\n"
        Lines+= f"Path to MOLA: {mola.__MOLA_PATH__}\n"
        Lines+= f"Commit SHA: {mola.__SHA__}\n'''\n\n"

        params = self.convert_to_dict()
        for key, value in params.items():
            Lines += f"{key}={pprint.pformat(value)}\n\n"

        with open(filename,'w') as f: f.write(Lines)

        try: os.remove(filename+'c')
        except: pass
            
    def prepare(self):
        self.assemble()
        self.positioning()
        self.connect()
        self.define_families()
        self.split_and_distribute()
        self.process_overset()
        self.compute_flow_and_turbulence()
        self.set_motion()
        self.set_boundary_conditions()
        self.set_cfd_parameters()  # model, numerics, others...
        self.initialize_flow()  # eventually + distance to wall
        self.set_extractions()
        # self.check_preprocess() # empty BCs... maybe solver-specific
        self.set_workflow_parameters_in_tree()
        # self.set_workflow_parameters_in_file()

    def assemble(self):
        self.read_meshes()
        self.set_workflow_parameters_in_tree()

    def positioning(self):
        positioning.apply(self)

    def connect(self):
        connect.apply(self)

    def define_families(self):
        families.apply(self)

    def read_meshes(self):
        meshes = []
        for component in self.RawMeshComponents:
            base = reader.apply(component)
            meshes += [base]
        
        self.tree = cgns.merge(meshes)

        dimOfBases = set(base.dim() for base in self.tree.bases())
        if len(dimOfBases) != 1:
            raise MolaUserError('All bases must have the same physical dimension')
        self.ProblemDimension = int(list(dimOfBases)[0])


    def split_and_distribute(self):
        split.apply(self)

    def process_overset(self):
        pass

    def get_flow_generator(self, fg):
        if isinstance(fg, str):
            return flow_generators.AvailableFlowGenerators[fg]
        else:
            return fg
        
    def compute_flow_and_turbulence(self):
        # mola-generic set of parameters
        FlowGen = self._FlowGenerator(self)
        FlowGen.generate()
        self.Fluid = FlowGen.Fluid
        self.Flow = FlowGen.Flow
        self.Turbulence = FlowGen.Turbulence

    def initialize_flow(self):
        initialization.apply(self)
    
    def set_boundary_conditions(self):
        boundary_conditions.apply(self)

    def set_motion(self):
        motion.apply(self)

    def set_cfd_parameters(self):
        cfd_parameters.apply(self)

    def set_extractions(self):
        extractions.apply(self)

    def write_cfd_files(self):
        write_cfd_files.apply(self)
    
    def remove_cfd_files(self):
        remove_cfd_files.apply(self)

    def compute(self):
        compute.apply(self)

    def visu(self):
        pass

    def get_component(self, base_name):
        for component in self.RawMeshComponents:
            if component['Name']==base_name:
                return component

    def has_overset_component(self):
        for component in self.RawMeshComponents:
            if 'OversetOptions' not in component: 
                continue
            if component['OversetOptions']:
                return True
        return False
    
    def submit(self, command=None):
        mola_logger.info(f"Submit job on machine {self.RunManagement['Machine']}")
        if command is None:
            command = self.RunManagement['LauncherCommand']
        user = self.RunManagement.get('User')
        SV.submit_command(command, self.RunManagement['Machine'], user=user)

    def write_tree_remote(self, data_directory=None):
        from . import workflow_manager as WM
        sender = WM.WorkflowSender(self, data_directory=data_directory)
        sender.apply()

    def merge(self, other_workflow):
        # TODO Still in development, not validated
        mola_logger.warning(f'Merge workflows')

        self._merge_trees(other_workflow)

        self._update_interfaces_between_workflows(other_workflow)
        
        # merge attributes
        self.RawMeshComponents += other_workflow.RawMeshComponents
        # Handle ApplicationContext ?
        self.BoundaryConditions += other_workflow.BoundaryConditions
        self.BodyForceModeling += other_workflow.BodyForceModeling
        self.Extractions += other_workflow.Extractions
        self.ConvergenceCriteria += other_workflow.ConvergenceCriteria

    def _merge_trees(self, other_workflow):
        other_tree = other_workflow.tree
        other_tree.findAndRemoveNode(Name=self._workflow_parameters_container_, Depth=1)
        main_base = self.tree.get(Type='CGNSBase', Depth=1)
        secondary_basename = other_tree.get(Type='CGNSBase', Depth=1).name()

        self.tree.merge(other_tree)

        # Move children of secondary base to the main base if they don't already exists in main base
        secondary_base = self.tree.get(Name=secondary_basename, Type='CGNSBase', Depth=1)
        children_to_move = copy.copy(secondary_base.children())
        for child in children_to_move:
            if not main_base.get(Name=child.name(), Type=child.type(), Depth=1):
                child.moveTo(main_base)
        secondary_base.remove()

    def _update_interfaces_between_workflows(self, other_workflow):
        updated_boundary_conditions = []
        for bc in self.BoundaryConditions + other_workflow.BoundaryConditions:
            if bc['type'] != 'InterfaceBetweenWorkflows':
                continue

            if not 'interface_type' in bc:
                raise MolaException(
                    f"The boundary condition on Family {bc['Family']} is of type {bc['type']},"
                    "and for this type the key 'interface_type' must be defined."
                    )
            
            elif isinstance(bc['interface_type'], str):
                assert bc['interface_type'] in ['Match']
                raise NotImplementedError

            elif isinstance(bc['interface_type'], dict):
                bc.update(bc['interface_type'])
                bc.pop('interface_type')
                if bc['type'] in boundary_conditions.turbomachinery_interfaces:
                    bc.pop('Family')
                updated_boundary_conditions.append(bc)

            else:
                raise MolaException(
                    f"For BC on Family {bc['Family']}, the value of 'interface_type' must be of type str or dict."
                    )
        
        # set again boundary conditions because it have changed
        boundary_conditions.apply(self, updated_boundary_conditions)

    def show_interface(self):
        def show_interface_of_method(method, skip_args=['self'], indentation=2):
            indent1 = ' '*indentation
            signature = inspect.signature(method)
            for param in signature.parameters.values():
                if param.name in skip_args: continue
                setter_name = 'set_'+param.name
                try:
                    setter_method = getattr(self,setter_name)
                except:
                    raise MolaException(f'Must implement interface for argument {param.name} using method "{setter_name}" in {self.Name}')
                print(indent1+f'Attribute \033[4m\033[1m{param.name}\033[0m is set using:')
                signature = get_signature(setter_method)
                for line in signature.split('\n'):
                    print(indent1 + line)

                add_to_methods_from_type = self._get_add_to_methods_of_attribute(param.name)
                if not add_to_methods_from_type: continue 
                indent2 = indent1+' '*2
                several_add_to_methods = len(add_to_methods_from_type) > 1
                for Type, add_to_method in add_to_methods_from_type.items():
                    print(indent2+f"where each item is a {CYAN}dict{ENDC} with these authorized keys:")
                    if several_add_to_methods:
                        print(indent2+ f'if {BOLD}Type{ENDC} ({CYAN}str{ENDC}) == {PINK}"{Type}"{ENDC}')
                    signature = get_signature(add_to_method)
                    for line in signature.split('\n'):
                        print(indent2 + line)


        print(f'User interface of {BOLD}{self.Name}{ENDC}:')
        print(f'{BOLD}name{ENDC} ({CYAN}allowed types{ENDC}) : {PINK}default value{ENDC}\n')
        show_interface_of_method(self.__init__, skip_args=['self','tree'])

    def _get_add_to_methods_of_attribute(self, attribute):
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        add_to_methods_from_type = dict()
        for method_name, method in methods:
            if not method_name.startswith('add_to_'): continue
            split_name = method_name.replace('add_to_','').split('_')
            if split_name[0] == attribute:
                add_to_methods_from_type[split_name[-1]] = method
        return add_to_methods_from_type


    @staticmethod
    def _method_name():
        return inspect.currentframe().f_back.f_code.co_name

    @staticmethod
    def _get_comp(fun, kwargs):
        signature = inspect.signature(fun)
        parameter_annotations = get_type_hints(fun)
        new_component = dict()
        for name, param in signature.parameters.items():
            try:
                value = kwargs[name]
            except KeyError:
                raise MolaException(f'parameter {name} was not implemented in interface {fun.__name__}. \n{kwargs=}\n{new_component=}')
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
    def repack_kwargs(**kwargs):
        # Get the current frame (frame where this function is called)
        frame = inspect.currentframe().f_back
        # Get the arguments from the calling frame
        locals_dict = frame.f_locals
        locals_dict.pop("self", None)
        locals_dict.pop("kwargs", None)
        locals_dict.pop("__class__", None)
        locals_dict.pop("self.repack_kwargs", None)
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
