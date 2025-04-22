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
from pathlib import Path

from treelab import cgns
from mola.workflow import Workflow
from mola.cfd.postprocess import iso_surface, extract_bc

def compute_workflow(run_directory):

    x, y, z = np.meshgrid( np.linspace(0,1,5),
                           np.linspace(0,1,5),
                           np.linspace(0,1,5), indexing='ij')
    mesh = cgns.newZoneFromArrays( 'block', ['x','y','z'], [ x,  y,  z ])

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
                )
        ],

        SplittingAndDistribution=dict(Strategy='AtComputation', Splitter='maia'),

        Flow=dict(Velocity = 100.),

        Turbulence = dict(Model = 'SA'),

        Numerics = dict(NumberOfIterations = 2, CFL=1.0),

        BoundaryConditions=[
            dict(Family='Ground', Type='Wall'),
            dict(Family='Farfield', Type='Farfield'),
        ],

        Extractions=[
            dict(Type='IsoSurface', Name='MySurface', IsoSurfaceField='CoordinateY', IsoSurfaceValue=0.5, Fields=['Mach']),
            dict(Type='BC', Source='BCWall*', Fields=['Pressure']),
            ],

        RunManagement=dict(
            Scheduler='local',
            RunDirectory = run_directory,
            NumberOfProcessors = 2,
            ),

        )
    
    w.prepare()
    w.write_cfd_files()
    w.submit()

    ## NOTE to gererate in test data, the function get_elsa_output_tree must be followed by:
    # import maia.pytree as PT
    # lines = PT.yaml.to_yaml(t)
    # with open(f'/stck/tbontemp/softs/MOLA/Dev/src/mola/cfd/postprocess/test/elsa_cassiopee_output_tree_{rank}.yaml', 'w') as fic:
    #     for line in lines:
    #         print(line)
    #         fic.write(line+'\n')
    # comm.barrier()
    # exit(0)
    

@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.elsa
@pytest.mark.mpi
@pytest.mark.parametrize('splitter', ['cassiopee', 'pypart'])
def test_iso_surface_elsa(splitter):

    import maia.pytree as PT

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size == 1:
        return

    # Read tree from yaml files (2 procs)
    script_path = Path(__file__).resolve().parent / 'snippets'
    with open(script_path / f'elsa_{splitter}_output_tree_{rank}.yaml', 'r') as open_file:
        yaml_tree = open_file.read()
    output_tree = cgns.castNode(PT.yaml.to_node(yaml_tree))

    extraction = dict(Type='IsoSurface', Name='MySurface', 
                      IsoSurfaceField='CoordinateY', IsoSurfaceValue=0.5, 
                      IsoSurfaceContainer='FlowSolution#Output', Fields=['Mach'])
    
    isosurface = iso_surface(
        output_tree, 
        IsoSurfaceField = extraction['IsoSurfaceField'], 
        IsoSurfaceValue = extraction['IsoSurfaceValue'], 
        IsoSurfaceContainer = extraction['IsoSurfaceContainer'],
        Name = extraction['Name'],
        tool = 'cassiopee',
        )
    
    # Open reference
    with open(script_path / f'iso_{rank}.yaml', 'r') as open_file:
        yaml_tree = open_file.read()
    iso_ref = cgns.castNode(PT.yaml.to_node(yaml_tree))

    predicate = lambda n: PT.get_label(n) not in ['CGNSLibraryVersion_t', 'ReferenceState_t', 'Ordinal_t']
    PT.rm_nodes_from_predicate(isosurface, predicate)
    PT.rm_nodes_from_predicate(iso_ref, predicate)

    condition_to_rm = lambda n: PT.get_label(n) in ['CGNSLibraryVersion_t', 'ReferenceState_t', 'FlowEquationSet_t', 'Ordinal_t'] \
                            or PT.get_name(n) in [':CGNS#Ppart', ':CGNS#GlobalNumbering', 'ELSA_TRIGGER', '.Solver#ownData', '.Solver#Param'] 
    PT.rm_nodes_from_predicate(isosurface, predicate=condition_to_rm)
    PT.rm_nodes_from_predicate(iso_ref, predicate=condition_to_rm)
        
    assert PT.is_same_tree(isosurface, iso_ref)


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.elsa
@pytest.mark.mpi
@pytest.mark.parametrize('splitter', ['cassiopee', 'pypart'])
def test_extract_bc_elsa(splitter):

    import maia.pytree as PT

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size == 1:
        return

    # Read tree from yaml files (2 procs)
    script_path = Path(__file__).resolve().parent / 'snippets'
    with open(script_path / f'elsa_{splitter}_output_tree_{rank}.yaml', 'r') as open_file:
        yaml_tree = open_file.read()
    output_tree = cgns.castNode(PT.yaml.to_node(yaml_tree))

    extraction = dict(Type='BC', Source='Ground', Name='ByFamily')
    
    bc = extract_bc(output_tree, extraction['Source'], tool='cassiopee')

    # import maia.pytree as PT
    # lines = PT.yaml.to_yaml(bc)
    # with open(script_path / f'bc_{rank}.yaml', 'w') as fic:
    #     for line in lines:
    #         fic.write(line+'\n')
    # comm.barrier()

    # Open reference
    with open(script_path / f'bc_{rank}.yaml', 'r') as open_file:
        yaml_tree = open_file.read()
    bc_ref = cgns.castNode(PT.yaml.to_node(yaml_tree))

    condition_to_rm = lambda n: PT.get_label(n) in ['CGNSLibraryVersion_t', 'ReferenceState_t', 'FlowEquationSet_t', 'Ordinal_t'] \
                            or PT.get_name(n) in [':CGNS#Ppart', ':CGNS#GlobalNumbering', 'ELSA_TRIGGER', '.Solver#ownData', '.Solver#Param'] 
    PT.rm_nodes_from_predicate(bc, predicate=condition_to_rm)
    PT.rm_nodes_from_predicate(bc_ref, predicate=condition_to_rm)
        
    assert PT.is_same_tree(bc, bc_ref)


if __name__ == '__main__':
    # compute_workflow('test')
    from mpi4py import MPI
    test_extract_bc_elsa('maia', MPI.COMM_WORLD)
