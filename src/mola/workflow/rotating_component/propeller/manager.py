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
from mola.logging import mola_logger, MolaAssertionError
from mola.cfd.preprocess.mesh import tools as mesh_tools
from mola.workflow import WorkflowManager

class WorkflowPropellerManager(WorkflowManager):

    _units = dict(
        Thrust = 'N',
        Power = 'W', 
        Torque = 'N.m',
        FigureOfMeritHover = '-',
        PropulsiveEfficiency = '-',
    )
    _default_quantities_to_plot = ['Thrust', 'Power', 'PropulsiveEfficiency']

    def add_operating_point(self, operating_point, Density, Velocity, Temperature, RPM_range):
        self.new_job(operating_point) 
        for RPM in RPM_range:
            self.add_variations(
                [
                    ('RunManagement|JobName', operating_point),
                    ('RunManagement|RunDirectory', f'{operating_point}_{RPM}rpm'),
                    # ShaftRotationSpeed in rad/s, because WorkflowManager does not apply conversion function (applied only first time the Workflow is instanciated!)
                    ('ApplicationContext|ShaftRotationSpeed', -abs(RPM/30*np.pi)),
                    ('ApplicationContext|ShaftRotationSpeedUnit', 'rad/s'),
                    ('Flow|Density', Density),
                    ('Flow|Velocity', Velocity),
                    ('Flow|Temperature', Temperature),
                ],
            )

    def gather_performance(self, 
                           variables:Union[list, None]=None, 
                           blade_family_name: str='BLADE', 
                           filename:Union[str, None]=None, 
                           update_from_remote_machine=True) -> dict:

        if variables is None:
            variables = self._default_quantities_to_plot

        queries = [
            f'CGNSTree/Integral/{blade_family_name}/FlowSolution/{var}'
            for var in variables
        ]
        perfo_data = self.gather_signals(
            queries, 
            filename=filename, 
            keep_last_point=True, 
            update_from_remote_machine=update_from_remote_machine
            )

        perfo_data = self._rearange_performance(perfo_data)

        return perfo_data
    
    def _rearange_performance(self, perfo_on_op):
        ordered_perfo = dict()
        for op, perfo_on_rpm in self.rearange_signals(perfo_on_op).items():
            rpm = np.array([float(case.split('_')[-1].replace('rpm', '')) for case in perfo_on_rpm['case']])
            ordered_perfo[op] = perfo_on_rpm
            ordered_perfo[op]['RPM'] = rpm

        return ordered_perfo

    def plot_performance_on_rpm(self, perfo_on_op, variables=None, fmt='png', show=True):
        import matplotlib.pyplot as plt

        if variables is None:
            variables = self._default_quantities_to_plot

        for op, perfo in perfo_on_op.items():

            for var in variables:

                plt.figure()
                plt.xlabel('Rotation speed (RPM)')
                if var in self._units:
                    plt.ylabel(f'{var} ({self._units[var]})')
                else:
                    plt.ylabel(var)
                plt.title(op)
                plt.plot(perfo['RPM'], perfo[var], 'o-')
                plt.tight_layout()
                plt.savefig(f'{op}_{var}.{fmt}', dpi=300)

        if show: plt.show()

    def export_perfo_to_dat(self, perfo_on_op, variables=None, fmt='%.4e'):
        if variables is None:
            a_op = list(perfo_on_op)[0]
            variables = list(perfo_on_op[a_op])
            variables.remove('RPM')

        for op in perfo_on_op:  
            # export to .dat
            data = np.vstack([perfo_on_op[op]['RPM']] + [perfo_on_op[op][var] for var in variables]).T
            np.savetxt(f'{op}.dat', data, header='RPM '+' '.join(variables), fmt=fmt)

            ## To read these data:
            # data = np.loadtxt(f'{op}.dat')
            # RPM    = data[:, 0]
            # Thrust = data[:, 1]
            # Power  = data[:, 2]

