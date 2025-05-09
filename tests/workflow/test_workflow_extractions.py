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
import os
import numpy as np

from treelab import cgns
from mola import naming_conventions as names
from mola.workflow import Workflow, read_workflow
from .test_workflow import  get_workflow_cart_monoproc

def assert_file_with_relevant_zone_and_fields(filename, zonename, fieldnames,
        path=None, expected_number_of_items=None):
    
    if path:
        expected_file = os.path.join(path, names.DIRECTORY_OUTPUT, filename)
    else:
        expected_file = os.path.join(names.DIRECTORY_OUTPUT, filename)
    
    assert os.path.isfile(expected_file)

    tree = cgns.load(expected_file)

    assert tree

    base = tree.get(Name="Integral", Type="CGNSBase_t", Depth=1)
    assert base
    
    zone = base.get(Name=zonename, Type="Zone_t", Depth=1)
    assert zone

    container = zone.get(Name='FlowSolution', Type='FlowSolution_t', Depth=1)
    assert container

    iterations = container.get(Name='Iteration', Type='DataArray_t', Depth=1)
    assert iterations

    if not isinstance(fieldnames,list):
        if not isinstance(fieldnames,str): raise AttributeError("wrong fieldnames attribute")
        fieldnames = [fieldnames]

    for fieldname in fieldnames:
        field_node = container.get(Name=fieldname, Type='DataArray_t', Depth=1)
        assert field_node 

        if expected_number_of_items is not None:
            assert len(field_node.value()) == expected_number_of_items

@pytest.mark.integration
@pytest.mark.cost_level_1
def test_found_requested_extraction():

    dist = np.linspace(0,1,5)
    x, y, z = np.meshgrid( dist, dist, dist, indexing='ij')
    zone = cgns.newZoneFromArrays( 'block', ['x','y','z'], [ x,  y,  z ])
    tree = cgns.Tree(Base=zone)

    w = Workflow(
        Solver=os.environ.get('MOLA_SOLVER'),

        RawMeshComponents=[
            dict(
                Name='CART',
                Source=tree,
                Families=[
                    dict(Name='Ground',
                         Location='kmin'),
                    dict(Name='Farfield',
                         Location='remaining'),
                ],
            )
        ],
        
        SplittingAndDistribution=dict(
            Strategy='AtComputation',
            Splitter='maia',
            Distributor='maia', 
        ),

        Turbulence=dict(
            Model='SA'
        ),

        BoundaryConditions=[
            dict(Family='Ground', Type='Wall'),
            dict(Family='Farfield', Type='Farfield'),
        ],

        Extractions = [
            dict(Type='Integral', Name='TOTO', Fields=['Force', 'Torque'], Source='Ground')
        ]
    )

    found_extract = any([e["Name"] == "TOTO" for e in w.Extractions if "Name" in e])
    assert found_extract

    if w.Solver == 'sonics':
        from mola.cfd.preprocess.boundary_conditions.solver_sonics import adapt_workflow_for_sonics
        adapt_workflow_for_sonics(w)
    w.prepare()
    
    still_found_extract = any([e["Name"] == "TOTO" for e in w.Extractions if "Name" in e])
    assert still_found_extract



@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_2
def test_integrals_one_run(tmp_path, niter=10):
    
    separated_filename = 'test_integrals.cgns'

    w = get_workflow_cart_monoproc(tmp_path)
    if w.Solver == 'sonics':
        from mola.cfd.preprocess.boundary_conditions.solver_sonics import adapt_workflow_for_sonics
        adapt_workflow_for_sonics(w)

    w._interface.add_to_Extractions_Integral(
        Name='TestSeparatedFile',
        Fields=['Force','Torque'],
        File=separated_filename,
        Source='Ground',
    )

    w._interface.add_to_Extractions_Integral(
        Name='TestIntoSignals',
        Fields=['Force','Torque'],
        File=names.FILE_OUTPUT_1D,
        Source='Inlet',
    )

    w._interface.add_to_Extractions_Integral(
        Name='TestIntoSignals2',
        Fields=['MassFlow'],
        File=names.FILE_OUTPUT_1D,
        Source='Inlet',
    )
    
    w.Numerics['NumberOfIterations'] = niter
    w.RunManagement['Scheduler'] = 'local'
    w.prepare()

    
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()
    
    expected_number_of_items = niter + 1

    def assert_all():
        assert_file_with_relevant_zone_and_fields(separated_filename, "TestSeparatedFile",
        ['ForceX','ForceY','ForceZ','TorqueX','TorqueY','TorqueZ'], tmp_path, expected_number_of_items)
    
        assert_file_with_relevant_zone_and_fields(names.FILE_OUTPUT_1D, "TestIntoSignals",
            ['ForceX','ForceY','ForceZ','TorqueX','TorqueY','TorqueZ'], tmp_path, expected_number_of_items)
        
        assert_file_with_relevant_zone_and_fields(names.FILE_OUTPUT_1D, "TestIntoSignals2",
            "MassFlow", tmp_path, expected_number_of_items)


    assert_all() # TODO test also here the residuals output


@pytest.mark.integration
@pytest.mark.elsa
# @pytest.mark.fast  # FIXME Bug at restart in FastS.display_temporal_criteria
@pytest.mark.cost_level_3
def test_integrals_two_runs(tmp_path, niter_first_run=5, niter_second_run=7):

    w = get_workflow_cart_monoproc(tmp_path)
    if w.Solver == 'sonics':
        from mola.cfd.preprocess.boundary_conditions.solver_sonics import adapt_workflow_for_sonics
        adapt_workflow_for_sonics(w)

    w._interface.add_to_Extractions_Integral(
        Name='TestIntoSignals',
        Fields=['Force','Torque'],
        File=names.FILE_OUTPUT_1D,
        Source='Ground',
    )

    w.Numerics['NumberOfIterations'] = niter_first_run
    w.RunManagement['Scheduler'] = 'local'
    w.set_workflow_parameters_in_tree()
    w.prepare()

    # First run
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()

    # update NumberOfIterations, it was 0 at the end of the first run
    os.system(f'cd {tmp_path}; mola_update --NumberOfIterations={niter_second_run}')

    # Second run: we must read the updated file main.cgns with workflow reader
    w = read_workflow(os.path.join(tmp_path,names.FILE_INPUT_SOLVER))
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()

    expected_number_of_items = niter_first_run + niter_second_run + 1

    # TODO test also here the residuals output
    assert_file_with_relevant_zone_and_fields(names.FILE_OUTPUT_1D, "TestIntoSignals",
        ['ForceX','ForceY','ForceZ','TorqueX','TorqueY','TorqueZ'],
          tmp_path, expected_number_of_items)


@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_2
def test_bc_one_run(tmp_path, niter=10):
    
    separated_filename = 'test_bc.cgns'

    w = get_workflow_cart_monoproc(tmp_path)
    if w.Solver == 'sonics':
        from mola.cfd.preprocess.boundary_conditions.solver_sonics import adapt_workflow_for_sonics
        adapt_workflow_for_sonics(w)

    w._interface.add_to_Extractions_BC(
        Name='TestSeparatedFile',
        Fields=['Pressure'],
        File=separated_filename,
        Source='Ground',
    )
    
    w.Numerics['NumberOfIterations'] = niter
    w.RunManagement['Scheduler'] = 'local'
    w.prepare()

    
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()
    
@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_2
def test_integral_with_postprocess(tmp_path, niter=10):
    
    w = get_workflow_cart_monoproc(tmp_path)
    if w.Solver == 'sonics':
        from mola.cfd.preprocess.boundary_conditions.solver_sonics import adapt_workflow_for_sonics
        adapt_workflow_for_sonics(w)

    w._interface.add_to_Extractions_Integral(
        Source='Ground',
        Fields=['Force'],
        PostprocessOperations=[dict(Type='rsd', Variable='ForceX')]
    )
    
    w.Numerics['NumberOfIterations'] = niter
    w.RunManagement['Scheduler'] = 'local'
    w.prepare()

    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()

    expected_number_of_items = niter + 1 

    assert_file_with_relevant_zone_and_fields(names.FILE_OUTPUT_1D, "Ground",
            ['ForceX', 'ForceY', 'ForceZ', 'rsd-ForceX'], tmp_path, expected_number_of_items)

@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_2
def test_convergence_on_criterion(tmp_path, niter=20):
    
    w = get_workflow_cart_monoproc(tmp_path)
    if w.Solver == 'sonics':
        from mola.cfd.preprocess.boundary_conditions.solver_sonics import adapt_workflow_for_sonics
        adapt_workflow_for_sonics(w)

    w.Flow['Velocity'] = 0.01  # With that, the solution is already converged when the run begins. Not 0 otherwise elsA stops at iteration 1
    w.Numerics['MinimumNumberOfIterations'] = 10  # to allow stopping the simulation as soon as the convergence criterion is reached
    w._interface.add_to_ConvergenceCriteria(ExtractionName='Ground', Variable='rsd-ForceX', Threshold=0.1)
    
    w.Numerics['NumberOfIterations'] = niter
    w.RunManagement['Scheduler'] = 'local'
    w.prepare()

    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()

    if w.Solver == 'elsa':
        expected_number_of_items = 12 
    elif w.Solver == 'fast':
        expected_number_of_items = 11
    else:
        raise AssertionError

    assert_file_with_relevant_zone_and_fields(names.FILE_OUTPUT_1D, "Ground",
            ['rsd-ForceX'], tmp_path, expected_number_of_items)

if __name__ == '__main__':
    test_integrals_one_run('extract_integrals_one_run_'+os.environ.get("MOLA_SOLVER"))
    # test_integrals_two_runs('extract_integrals_two_runs_'+os.environ.get("MOLA_SOLVER"))
    # test_bc_one_run('test_bc_one_run_'+os.environ.get("MOLA_SOLVER"))