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

from typing import Union
import numpy as np
from . import WorkflowRotatingComponent
from . import workflow_manager as WM
from mola.logging import mola_logger, MolaAssertionError
from mola.cfd.preprocess.mesh import tools as mesh_tools

class WorkflowTurbomachinery(WorkflowRotatingComponent):

    def __init__(self, 
                 SplittingAndDistribution=None,
                 Flow=None,
                 **kwargs
                 ):
        
        super().__init__(SplittingAndDistribution=SplittingAndDistribution,
                         Flow=Flow,
                         **kwargs)

        if self.tree is None:
            for meshInfo in self.RawMeshComponents:
                meshInfo.setdefault('mesher', 'Autogrid')

            self.Extractions.extend([
                dict(Type='BC', Source='BCWall*', Fields=['Pressure', 'BoundaryLayer', 'yPlus']),
                dict(Type='BC', Source='BCInflow*', Fields=['MassFlow']),
                dict(Type='BC', Source='BCOutflow*', Fields=['MassFlow']),
            ])

    def submit_iso_speed_line(self, ThrottleValues, ParallelMode=False, initialize_from_previous=True):

        try:
            RunDirectory = self.RunManagement['RunDirectory']
        except:
            raise MolaAssertionError("unknown RunDirectory. It must be provided in RunManagement['RunDirectory']")
        
        if ParallelMode and initialize_from_previous:
            mola_logger.warning('Because ParallelMode=True, initialize_from_previous is set to False.')
            initialize_from_previous = False
                        
        job_name = self.RunManagement.get('JobName', 'isospeed')
        RPM = self.ApplicationContext.get('RPM', 30/np.pi*self.ApplicationContext.get('ShaftRotationSpeed'))

        outflow_bc = mesh_tools.get_bc_from_bc_type(self, 'Outflow*')

        THROTTLE_KEY = dict(
            OutflowPressure = 'Pressure', 
            OutflowMassFlow = 'MassFlow',
        )
        # # The following lines are specific to elsA
        # if outflow_bc['valve_type'] == 0:
        #     if 'prespiv' in outflow_bc: 
        #         THROTTLE_KEY['OutflowRadialEquilibrium'] = 'prespiv'
        #     elif 'valve_ref_pres' in outflow_bc: 
        #         THROTTLE_KEY['OutflowRadialEquilibrium'] = 'valve_ref_pres'
        # elif outflow_bc['valve_type'] in [1, 5]:
        #     THROTTLE_KEY['OutflowRadialEquilibrium'] = 'valve_ref_pres' 
        # elif outflow_bc['valve_type'] == 2:
        #     THROTTLE_KEY['OutflowRadialEquilibrium'] = 'valve_ref_mflow'
        # elif outflow_bc['valve_type'] in [3, 4]:
        #     THROTTLE_KEY['OutflowRadialEquilibrium'] = 'valve_relax' 

        throttle_key = THROTTLE_KEY[outflow_bc["type"]]

        dispatcher = WM.WorkflowDispatcher(self)
        if not ParallelMode:
                dispatcher.new_job(f'isospeed_{RPM}rpm')
        for throttle in ThrottleValues:
            if ParallelMode:
                dispatcher.new_job(f'{throttle_key}_{throttle}')
            dispatcher.add_variations(
                [
                    ('RunManagement|JobName', f'{job_name}_{throttle}'),
                    ('RunManagement|RunDirectory', f'{throttle_key}_{throttle}'),
                    (f'BoundaryConditions|Family={outflow_bc["Family"]}|{throttle_key}', throttle),
                ], 
                initialize_from_previous=initialize_from_previous
                )
            
        scheduler = WM.WorkflowParallelScheduler(dispatcher, RunDirectory, skip_if_exists=True)
        scheduler.prepare()
        scheduler.submit()

    def set_Flow(self,
            Generator : str = 'Internal',
            Velocity  : float = 1.0,
            # Parameters relevant to InternalFlowGenerator
            MassFlow               : float = None,
            Mach                   : float = None,
            PressureStagnation     : float = None,
            TemperatureStagnation  : float = None,
            IdealGasConstant       : float = None,
            Gamma                  : float = None,
            ):
        return super().set_Flow(**self.repack_kwargs())

    def set_SplittingAndDistribution(self, 
            Strategy                         : str = 'AtComputation',
            Splitter                         : str = 'PyPart',
            Distributor                      : str = 'PyPart',
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
        return super().set_SplittingAndDistribution(**self.repack_kwargs())
        

    def set_ApplicationContext(self,
            ShaftAxis : Union[list,
                             tuple,
                             np.ndarray] = [1,0,0],
            
            # TODO : redefine as Workflow's attributes with set_* and add_to_* ?
            Rows : dict = None,
            HubRotationSpeed : list = None,
            ShaftRotationSpeed : float = None,
            NormalizationCoefficient : dict = None):
        '''
        
        '''
        # shall make _get_comp accessible (staticmethod?)
        self.ApplicationContext = self._get_comp(self.set_ApplicationContext, self.repack_kwargs())

        self.ApplicationContext['ShaftAxis'] = np.array(self.ApplicationContext['ShaftAxis'],dtype=float)