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
from typing import Union
import numpy as np
from .rotating_component import WorkflowRotatingComponent
from .turbomachinery_interface import WorkflowTurbomachineryInterface
from . import workflow_manager as WM
from mola.logging import mola_logger, MolaAssertionError
from mola.cfd.preprocess.mesh import tools as mesh_tools

class WorkflowTurbomachinery(WorkflowRotatingComponent):

    def __init__(self, tree=None, **kwargs):
        
        self._workflow_parameters_container_ = 'WorkflowParameters'
        self.Name = self.__class__.__name__
        self.tree = tree
        self._interface = WorkflowTurbomachineryInterface(workflow=self, **kwargs)
        if tree is not None:
            self.get_workflow_parameters_from_tree()
        else:
            self.Extractions.extend([
                dict(Type='BC', Source='BCWall*', Fields=['Pressure',
                                                          'BoundaryLayer',
                                                          'yPlus']),
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

        throttle_key = THROTTLE_KEY[outflow_bc["Type"]]

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
 