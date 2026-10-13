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

from treelab import cgns
from mola.logging import mola_logger

def apply_to_solver(workflow):
    add_info_in_reference_state(workflow)

def get_init_solution_container_name(workflow):
    return 'FlowSolution#Centers'

def add_info_in_reference_state(workflow):

    additional_data = dict()

    for var in ['Mach','Pressure','Temperature']:
        additional_data[var] = workflow.Flow[var]

    namesForCassiopee = dict(
        cv                    = 'Cv',
        Gamma                 = 'Gamma',
        SutherlandViscosity   = 'Mus',
        SutherlandConstant    = 'Cs',
        SutherlandTemperature = 'Ts',
        Prandtl               = 'Pr',
    )
    # Needed by Fast, but not other solvers 
    for key, value in namesForCassiopee.items():
        additional_data[value] = workflow.Fluid[key]

    for base in workflow.tree.bases():
        reference_state = base.get(Type='ReferenceState', Depth=1)
        for name, value in additional_data.items():
            cgns.Node(Name=name, Value=value, Type='DataArray', Parent=reference_state)
