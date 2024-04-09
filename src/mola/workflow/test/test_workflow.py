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
import numpy as np

import treelab.cgns as cgns

from mola.workflow import Workflow
from mola.logging import mola_logger, MolaException, mute_stdout
from mola import server as SV

import pytest
onera_only = pytest.mark.skipif(SV.get_network() != 'onera', reason="test on ONERA machines")


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_init():
    w = Workflow()


def get_workflow1():

    x, y, z = np.meshgrid( np.linspace(0,1,21),
                           np.linspace(0,1,21),
                           np.linspace(0,1,21), indexing='ij')
    mesh = cgns.newZoneFromArrays( 'block', ['x','y','z'],
                                            [ x,  y,  z ])

    w = Workflow(
        RawMeshComponents=[
            dict(
                Name='cartesian',
                Source=mesh,
                Mesher=None,
                CleaningMacro=None,
                Families=[
                    dict(Name='Ground',
                         Location='kmin'),
                    dict(Name='Farfield',
                         Location='remaining'),
                ],
                Positioning=[
                    dict(
                        Type='TranslationAndRotation',
                        InitialFrame=dict(
                            Point=[0,0,0],
                            Axis1=[1,0,0],
                            Axis2=[0,1,0],
                            Axis3=[0,0,1]),
                        RequestedFrame=dict(
                            Point=[0,0,0],
                            Axis1=[1,0,0],
                            Axis2=[0,1,0],
                            Axis3=[0,0,1]),
                        ),
                    # dict(
                    #     Type='DuplicateByRotation',
                    #     RotationPoint=[0,0,0],
                    #     RotationAxis=[0,0,1],
                    #     RightHandRuleRotation=True,
                    #     NumberOfInstances=4,
                    #     AddInstancesAsNewComponents=True,
                    #     ),
                ],
                Connection = [
                    dict(Type='Match', Tolerance=1e-8),
                ],
                OversetOptions=dict(),
                )
        ],

        SplittingAndDistribution=dict(
            Strategy='AtPreprocess', # "AtPreprocess" or "AtComputation"
            Splitter='Cassiopee', # or 'maia', 'PyPart' etc..
            Distributor='Cassiopee', 
            ComponentsToSplit='all', # 'all', or None or ['first', 'second'...]
            NumberOfProcessors=4, 
            # MinimumAllowedNodes=1,
            # MaximumAllowedNodes=20,
            # MaximumNumberOfPointsPerNode=1e9,
            # CoresPerNode=48,
            # DistributeExclusivelyOnFullNodes=True,
            ),


        )
    return w

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_write_tree():
    w = Workflow()
    w.write_tree('main.cgns')
    os.unlink('main.cgns')

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_workflow_parameters_in_tree(filename=''):
    w = Workflow()
    w.set_workflow_parameters_in_tree()
    if filename: w.write_tree(filename)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_workflow_parameters_from_tree(filename=''):
    w = Workflow()
    w.set_workflow_parameters_in_tree()
    w.write_tree('test.cgns')
    w.tree = 'test.cgns'
    w.get_workflow_parameters_from_tree()
    os.unlink('test.cgns')
    if filename: w.write_tree(filename)

@pytest.mark.cost_level_1
@pytest.mark.integration
def test_prepare_workflow1():
    w = get_workflow1()
    w.assemble()
    w.positioning()
    w.connect()
    w.define_families()
    w.split_and_distribute()
    w.write_tree()

def get_workflow2():

    x, y, z = np.meshgrid( np.linspace(0,1,21),
                           np.linspace(0,1,21),
                           np.linspace(0,1,21), indexing='ij')
    mesh = cgns.newZoneFromArrays( 'block', ['x','y','z'],
                                            [ x,  y,  z ])

    w = Workflow(
        RawMeshComponents=[
            dict(
                Name='cartesian',
                Source=mesh,
                Families=[
                    dict(Name='Ground',
                         Location='kmin'),
                    dict(Name='Farfield',
                         Location='remaining'),
                ],
                Positioning=[
                    dict(
                        Type='TranslationAndRotation',
                        InitialFrame=dict(
                            Point=[0,0,0],
                            Axis1=[1,0,0],
                            Axis2=[0,1,0],
                            Axis3=[0,0,1]),
                        RequestedFrame=dict(
                            Point=[0,0,0],
                            Axis1=[1,0,0],
                            Axis2=[0,1,0],
                            Axis3=[0,0,1]),
                        ),
                ],
                Connection = [
                    dict(Type='Match', Tolerance=1e-8),
                ],
                OversetOptions=dict(),
                )
        ],

        SplittingAndDistribution=dict(
            Strategy='AtPreprocess', # "AtPreprocess" or "AtComputation"
            Splitter='Cassiopee', # or 'maia', 'PyPart' etc..
            Distributor='Cassiopee', 
            ComponentsToSplit='all', # 'all', or None or ['first', 'second'...]
            NumberOfProcessors=4, 
            ),

        Flow=dict(
            Velocity = 100.,
        ),

        Turbulence = dict(
            Model = 'SA',
        ),

        Solver=os.environ.get('MOLA_SOLVER'),

        Numerics = dict(
            CFL=1.,
        ),

        BoundaryConditions=[
            dict(Family='Ground', type='Wall'),
            dict(Family='Farfield', type='Farfield'),
        ],

        Extractions=[
            dict(type='signals', name='Integrals', fields=['CL', 'std-CL'], Period=10),
            dict(type='probe', name='probe1', fields=['std-Pressure'], Period=5),
            dict(type='probe', name='probe2', fields=['std-Density'], Period=5),
            dict(type='3D', fields=['Mach', 'q_criterion']),
            dict(type='bc', BCType='BCWall*', storage='ByFamily', fields=['normalvector', 'frictionvector']),
            dict(type='bc', BCType='*', storage='ByFamily', fields=['Pressure']),
            dict(type='IsoSurface', name='MySurface', field='CoordinateY', value=1.e-6, AllowedFields=['Mach','cellN']),
            ],


        )
    return w


def get_workflow_sphere_struct():
    w = Workflow(
        RawMeshComponents=[
            dict(
                Name='sphere',
                Source='/stck/mola/data/mesh/sphere/sphere_struct.cgns',
                Families=[
                    dict(Name='Wall', Location='kmin'),
                    dict(Name='Farfield', Location='remaining'),
                ],
                )
        ],

        SplittingAndDistribution=dict(
            Strategy='AtPreprocess', # "AtPreprocess" or "AtComputation"
            Splitter='Cassiopee', # or 'maia', 'PyPart' etc..
            Distributor='Cassiopee', 
            ComponentsToSplit='all', # 'all', or None or ['first', 'second'...]
            NumberOfProcessors=1, 
            ),

        Flow=dict(
            Density = 0.2,
            Temperature = 100.,
            Velocity = 50.,
                 ),

        Turbulence = dict(
            Model = 'SA',
        ),

        Solver=os.environ.get('MOLA_SOLVER'),

        Numerics = dict(
            NumberOfIterations=10,
            CFL=1.,
        ),

        BoundaryConditions=[
            dict(Family='Wall', type='Wall'),
            dict(Family='Farfield', type='Farfield'),
        ],

        Extractions=[
            dict(type='bc', BCType='*', storage='ByFamily', fields=['Pressure']),
            dict(type='bc', BCType='BCWall*', storage='ByFamily', fields=['NormalVector', 'Friction', 'BoundaryLayer']),
            dict(type='IsoSurface', name='MySurface', field='CoordinateZ', value=1.e-6),
            ],

        RunManagement=dict(
            NumberOfProcessors=1,
            RunDirectory=os.path.dirname(os.path.realpath(__file__)),
            ),
        )
    
    return w

@pytest.mark.integration
@pytest.mark.cost_level_1
def test_prepare_workflow2():
    w = get_workflow2()
    w.prepare()
    w.write_cfd_files()
    w.remove_cfd_files()

@pytest.mark.integration
@pytest.mark.cost_level_3
def test_workflow_sphere_struct():
    w = get_workflow_sphere_struct()
    w.prepare()
    w.write_cfd_files()
    w.submit()
    COMPLETED_PATH = os.path.join(w.RunManagement['RunDirectory'],'COMPLETED')
    if not os.path.exists(COMPLETED_PATH):
        raise MolaException('simulation did not ended as expected')
    w.remove_cfd_files()

# @onera_only
# @pytest.mark.integration
# @pytest.mark.cost_level_3
# def test_workflow_sphere_struct_remote_sator():
#     w = get_workflow_sphere_struct()
#     w.RunManagement['RunDirectory'] = f'/tmp_user/sator/$USER/.test/tmp_MOLA_test/'
#     w.RunManagement['mola_target_path'] = f'/tmp_user/sator/$USER/MOLA/mola_v2/src/'
#     w.RunManagement['AER'] = '34790002F' # PDEV MOLA 2024
#     w.RunManagement['TimeLimit'] = '00:30:00'

#     SV.remove_path(w.RunManagement['RunDirectory'], machine='sator', file_only=False)

#     w.prepare()
#     w.write_cfd_files()
#     w.submit()

#     COMPLETED_PATH = os.path.join(w.RunManagement['RunDirectory'],'COMPLETED')
#     SV.wait_until(SV.is_existing_path, path=COMPLETED_PATH, machine='sator', timeout=30)
#     SV.remove_path(w.RunManagement['RunDirectory'], machine='sator', file_only=False)
