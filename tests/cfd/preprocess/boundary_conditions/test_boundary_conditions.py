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

import pytest
import numpy as np

from treelab import cgns
from mola.workflow.workflow import Workflow

def get_workflow_prepared_to_test_bcs(BoundaryConditions):

    x, y, z = np.meshgrid( np.linspace(0,1,5),
                           np.linspace(0,1,5),
                           np.linspace(0,1,5), indexing='ij')
    mesh = cgns.newZoneFromArrays( 'block', ['x','y','z'], [ x,  y,  z ])

    params = dict(

        RawMeshComponents=[
            dict(
                Name='cartesian',
                Source=mesh,
                Families=[
                    dict(Name='imin', Location='imin'),
                    dict(Name='imax', Location='imax'),
                    dict(Name='jmin', Location='jmin'),
                    dict(Name='jmax', Location='jmax'),
                    dict(Name='kmin', Location='kmin'),
                    dict(Name='kmax', Location='kmax'),
                ],
                )
        ],

        Flow=dict(Velocity = 100.),

        Turbulence = dict(Model = 'SA',),

        Numerics = dict(
            NumberOfIterations = 2,
            CFL=1.0,
        ),

        BoundaryConditions=BoundaryConditions,

        )
    workflow = Workflow(**params)
    workflow.process_mesh()
    workflow.process_overset()
    workflow.compute_flow_and_turbulence()
    workflow.set_motion()
    return workflow

