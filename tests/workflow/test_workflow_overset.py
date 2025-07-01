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
pytestmark = pytest.mark.elsa

import numpy as np

from mola.workflow import Workflow


@pytest.mark.integration
@pytest.mark.cost_level_4
def test_overset_sphere(tmp_path):

    Dpsi = 2.0
    RPM = 100.0
    dt = Dpsi/(6*RPM)
    omega = np.pi*RPM/30.

    w = Workflow(
        RawMeshComponents=[
            dict(
                Name='BLADE',
                Source='/stck/mola/v1.19/EXAMPLES/WORKFLOW_STANDARD/SPHERE/OVERSET_ROTATION/MESHING/sphere.cgns',
                Families=[
                    dict(Name='BLADEwall', Location='kmin'),
                    dict(Name='Overlap', Location='kmax'),
                ],
                Connection = [ dict(Type='Match', Tolerance=1e-8),],

                OversetOptions = dict( NCellsOffset=8, OnlyMaskedByWalls=True), 

                OversetMotion = dict(
                    NumberOfBlades=1,
                    InitialFrame=dict(
                        RotationCenter=[0,0,0],
                        RotationAxis=[0,0,1],
                        BladeDirection=[0,1,0],
                        RightHandRuleRotation=True),
                    RequestedFrame=dict(
                        RotationCenter=[0,0,0],
                        RotationAxis=[0,0,1],
                        BladeDirection=[0,1,0],
                        RightHandRuleRotation=True),
                                    ),
                ),

            dict(
                Name='BACKGROUND',
                Source='/stck/mola/v1.19/EXAMPLES/WORKFLOW_STANDARD/SPHERE/OVERSET_ROTATION/MESHING/background.cgns',
                Families=[ dict(Name='Farfield', Location='remaining'), ],
            ),
        ],

        Motion=dict(
            BLADE=dict(
                RPM=RPM,
                Function=dict(
                    type='rotor_motion',
                    tet_pnt=[0.0,0.0,0.0],
                    tet_vct=[0.0,1.0,0.0],
                    tet0=10.0,
                    pre_lag_pnt=[0.0,0.0,0.0],
                    pre_lag_vct=[0.0, 0.0, 1.0],
                    pre_lag_ang=0.0,
                    pre_con_pnt=[0.0,0.0,0.0],
                    pre_con_vct=[1.0, 0.0, 0.0],
                    pre_con_ang=0.0)
                ),
        ),

        SplittingAndDistribution=dict(
            Strategy='AtPreprocess',
            Splitter='Cassiopee',
            Distributor='Cassiopee', 
            ComponentsToSplit=None,
            ),

        Flow=dict(
            Density = 0.2,
            Temperature = 100.,
            Velocity = 50.,
                 ),

        Turbulence = dict(
            Model = 'SA',
        ),

        Numerics = dict(
            TimeMarching='Unsteady',
            # NumberOfIterations=int( (15/60) * np.round(360/Dpsi) ),
            NumberOfIterations=2,
            TimeStep=dt,
        ),

        BoundaryConditions=[
            dict(Family='BLADEwall', Type='Wall'),
            dict(Family='Farfield', Type='Farfield'),
        ],

        Extractions = [
            dict(Type="IsoSurface",
                 IsoSurfaceField="CoordinateZ",
                 IsoSurfaceValue=0.0,
                 Fields=['Density',
                         'MomentumX',
                         'MomentumY',
                         'MomentumZ',
                         'Viscosity_EddyMolecularRatio'],
                 Frame='absolute'),
        ],


        RunManagement=dict(
            NumberOfProcessors=1,
            RunDirectory=str(tmp_path),
            Scheduler = 'local',
            ),
        )

    w.prepare()
    w.write_cfd_files()
    w.submit()
    w.assert_completed_without_errors()
