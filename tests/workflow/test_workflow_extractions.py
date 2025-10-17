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
from glob import glob

from treelab import cgns
from mola import solver
from mola import naming_conventions as names
from mola.workflow import Workflow, read_workflow
from .test_workflow import  get_workflow_cart_monoproc, get_workflow2_parameters

def assert_file_with_relevant_zone_and_fields(filename, basename, zonename, fieldnames,
        path=None, expected_number_of_items=None):
    
    if path:
        expected_file = os.path.join(path, names.DIRECTORY_OUTPUT, filename)
    else:
        expected_file = os.path.join(names.DIRECTORY_OUTPUT, filename)
    
    assert os.path.isfile(expected_file)

    tree = cgns.load(expected_file)

    assert tree

    base = tree.get(Name=basename, Type="CGNSBase_t", Depth=1)
    assert base
    
    zone = base.get(Name=zonename, Type="Zone_t", Depth=1)
    assert zone

    container = zone.get(Name='FlowSolution', Type='FlowSolution_t', Depth=1)
    assert container

    iterations = container.get(Name='Iteration', Type='DataArray_t', Depth=1)
    assert iterations
    assert np.allclose(iterations.value(), np.arange(1, expected_number_of_items+1))

    if not isinstance(fieldnames,list):
        if not isinstance(fieldnames,str): raise AttributeError("wrong fieldnames attribute")
        fieldnames = [fieldnames]

    for fieldname in fieldnames:
        field_node = container.get(Name=fieldname, Type='DataArray_t', Depth=1)
        assert field_node 
        field_value = field_node.value()

        assert not np.any(np.isnan(field_value)), f'nan found in {fieldname} in {basename}/{zonename}'

        if expected_number_of_items is not None:
            assert len(field_value) == expected_number_of_items

def assert_file_containing_expected_field_at_expected_container(filename, basename: str, fieldnames: list,
        expected_container : str, path: str=None, exclusive : bool = False ):
    
    if path:
        expected_file = os.path.join(path, names.DIRECTORY_OUTPUT, filename)
    else:
        expected_file = os.path.join(names.DIRECTORY_OUTPUT, filename)
    
    assert os.path.isfile(expected_file)

    tree = cgns.load(expected_file)

    assert tree

    base = tree.get(Name=basename, Type="CGNSBase_t", Depth=1)
    assert base
    
    for zone in base.zones():

        container = zone.get(Name=expected_container, Depth=1)
        assert container, f"expected container {expected_container}"

        if exclusive:
            existing_field_names = [field.name() for field in container.group(Type='DataArray_t',Depth=1)]
            assert set(existing_field_names) == set(fieldnames), f"did not get the expected field names, found: {existing_field_names} but expected: {fieldnames}"

        for fieldname in fieldnames:
            field_node = container.get(Name=fieldname, Type='DataArray_t', Depth=2)
            assert field_node 
            field_value = field_node.value()
            assert not np.any(np.isnan(field_value)), f'nan found in {fieldname} in {basename}'

def assert_tecplot_containing_expected_field(filename, fieldnames: list, path: str=None):
    import Converter.PyTree as C

    if path:
        expected_file = os.path.join(path, names.DIRECTORY_OUTPUT, filename)
    else:
        expected_file = os.path.join(names.DIRECTORY_OUTPUT, filename)
    
    
    trees = [ cgns.castNode(C.convertFile2PyTree(f)) for f in glob(expected_file) ] # allows wildcards, used for one-file-per-rank saving
    tree = cgns.merge(trees)

    assert tree

    base = tree.get(Type="CGNSBase_t", Depth=1)
    assert base
    
    for zone in base.zones():
        container = zone.get(Name='FlowSolution')
        assert container

        for fieldname in fieldnames:
            field_node = container.get(Name=fieldname, Type='DataArray_t', Depth=2)
            assert field_node 
            field_value = field_node.value()
            assert not np.any(np.isnan(field_value)), f'nan found in {field_node.path()}'


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

    w.prepare()
    
    still_found_extract = any([e["Name"] == "TOTO" for e in w.Extractions if "Name" in e])
    assert still_found_extract



@pytest.mark.integration
@pytest.mark.cost_level_2
def test_integrals_one_run(tmp_path, niter=10):
    
    separated_filename = 'test_integrals.cgns'

    w = get_workflow_cart_monoproc(tmp_path)

    w._interface.add_to_Extractions_Integral(
        Name='TestSeparatedFile',
        Fields=['Force'], #,'Torque'],  # TODO Torque not available in SoNICS for now
        File=separated_filename,
        Source='Ground',
    )

    w._interface.add_to_Extractions_Integral(
        Name='TestIntoSignals',
        Fields=['Force'], #,'Torque'], 
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
    
    expected_number_of_items = niter

    def assert_all():
        assert_file_with_relevant_zone_and_fields(separated_filename, "Integral", "TestSeparatedFile",
            ['ForceX','ForceY','ForceZ'],#'TorqueX','TorqueY','TorqueZ'], 
            tmp_path, expected_number_of_items)
    
        assert_file_with_relevant_zone_and_fields(names.FILE_OUTPUT_1D, "Integral", "TestIntoSignals",
            ['ForceX','ForceY','ForceZ'],#'TorqueX','TorqueY','TorqueZ'], 
            tmp_path, expected_number_of_items)
        
        assert_file_with_relevant_zone_and_fields(names.FILE_OUTPUT_1D, "Integral", "TestIntoSignals2",
            "MassFlow", tmp_path, expected_number_of_items)

        # FIXME 
        if w.Solver != 'fast':
            assert_file_with_relevant_zone_and_fields(names.FILE_OUTPUT_1D, "Residuals", "Residuals",
                [], tmp_path, expected_number_of_items)  # names of residuals depend on solver

    assert_all()


@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.skipif(solver=='sonics', reason="FIXME allow restart runs with sonics")
@pytest.mark.cost_level_3
def test_integrals_two_runs(tmp_path, niter_first_run=5, niter_second_run=7):

    w = get_workflow_cart_monoproc(tmp_path)

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

    expected_number_of_items = niter_first_run + niter_second_run

    assert_file_with_relevant_zone_and_fields(names.FILE_OUTPUT_1D, "Integral", "TestIntoSignals",
        ['ForceX','ForceY','ForceZ','TorqueX','TorqueY','TorqueZ'],
          tmp_path, expected_number_of_items)
    
    # FIXME 
    if w.Solver != 'fast':
        assert_file_with_relevant_zone_and_fields(names.FILE_OUTPUT_1D, "Residuals", "Residuals",
            [], tmp_path, expected_number_of_items)  # names of residuals depend on solver


@pytest.mark.integration
@pytest.mark.cost_level_2
def test_bc_one_run(tmp_path, niter=10):
    
    separated_filename = 'test_bc.cgns'
    basename = 'TestSeparatedFile'

    w = get_workflow_cart_monoproc(tmp_path)

    w._interface.add_to_Extractions_BC(
        Name=basename,
        Fields=['Pressure', 'SkinFriction'],
        File=separated_filename,
        Source='Ground',
    )
    
    w.Numerics['NumberOfIterations'] = niter
    w.RunManagement['Scheduler'] = 'local'
    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()

    if w.Solver == 'fast':
        expected_fields = ['Pressure']
    else:
        expected_fields = ['Pressure', 'SkinFrictionX', 'SkinFrictionY', 'SkinFrictionZ']

    assert_file_containing_expected_field_at_expected_container(
        separated_filename, basename, expected_fields,
        names.CONTAINER_OUTPUT_FIELDS_AT_CENTER, tmp_path)


@pytest.mark.integration
@pytest.mark.cost_level_2
def test_iso_surface_only(tmp_path, niter=10):
    
    w = get_workflow_cart_monoproc(tmp_path) 

    separated_filename = 'test_iso.cgns'

    w._interface.add_to_Extractions_IsoSurface(
        Fields=['Density',
                'Pressure' # FIXME not working with SONICS
                ],
        IsoSurfaceField='CoordinateZ',
        IsoSurfaceValue=0.5,
        File=separated_filename,
    )
    w.Numerics['NumberOfIterations'] = niter
    w.RunManagement['Scheduler'] = 'local'
    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()

    assert_file_containing_expected_field_at_expected_container(
        separated_filename, 'Iso_Z_0.5', ['Density'],
        names.CONTAINER_OUTPUT_FIELDS_AT_VERTEX, tmp_path)
    

@pytest.mark.integration
@pytest.mark.cost_level_2
def test_exclusive_fields_in_iso_surface_and_bc(tmp_path, niter=10):
    
    w = get_workflow_cart_monoproc(tmp_path)

    requested_fields_in_bc = ['Pressure','MomentumX']
    requested_fields_in_iso_surface = ['Density']

    w._interface.add_to_Extractions_BC(
        Fields=requested_fields_in_bc,
        Source='Ground',
    )


    w._interface.add_to_Extractions_IsoSurface(
        Fields=requested_fields_in_iso_surface,
        IsoSurfaceField='CoordinateZ',
        IsoSurfaceValue=0.5,
    )


    w.Numerics['NumberOfIterations'] = niter
    w.RunManagement['Scheduler'] = 'local'
    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()

    assert_file_containing_expected_field_at_expected_container(
        'extractions.cgns', 'Iso_Z_0.5', requested_fields_in_iso_surface,
        names.CONTAINER_OUTPUT_FIELDS_AT_VERTEX, tmp_path, exclusive=True)

    assert_file_containing_expected_field_at_expected_container(
        'extractions.cgns', 'Ground', requested_fields_in_bc,
        names.CONTAINER_OUTPUT_FIELDS_AT_CENTER, tmp_path, exclusive=True)


@pytest.mark.integration
@pytest.mark.cost_level_2
def test_exclusive_fields_in_3D(tmp_path, niter=10):
    
    w = get_workflow_cart_monoproc(tmp_path)

    requested_fields = ['Density','MomentumX','MomentumY','MomentumZ']

    w._interface.add_to_Extractions_3D(
        Fields=requested_fields,
    )

    w.Numerics['NumberOfIterations'] = niter
    w.RunManagement['Scheduler'] = 'local'
    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()

    assert_file_containing_expected_field_at_expected_container(
        'fields.cgns', 'cart', requested_fields,
        names.CONTAINER_OUTPUT_FIELDS_AT_VERTEX, tmp_path, exclusive=True)


@pytest.mark.integration
@pytest.mark.cost_level_2
@pytest.mark.skipif(solver=='sonics', reason="FIXME allow cgns transform and write in tecplot")
def test_write_extractions_in_tecplot_format_co1(tmp_path, niter=10):
    
    params = get_workflow2_parameters()
    params["RunManagement"]["RunDirectory"] = str(tmp_path)
    params["RunManagement"]["NumberOfProcessors"] = 1
    params["RunManagement"]['Scheduler'] = 'local'
    params["Numerics"]['NumberOfIterations'] = niter
    w = Workflow(**params)

    requested_fields_in_signals = ['Force']
    requested_fields_in_bc = ['Pressure','MomentumX']
    requested_fields_in_iso_surface = ['Density']
    requested_fields_in_3d = ['Density','MomentumX','MomentumY','MomentumZ']

    w._interface.add_to_Extractions_Integral(
        File='signals.plt',
        Name='TestSeparatedFile',
        Fields=requested_fields_in_signals, 
        Source='Ground',
    )

    w._interface.add_to_Extractions_BC(
        File = 'extractions.plt',
        Fields=requested_fields_in_bc,
        Source='Ground',
    )

    w._interface.add_to_Extractions_IsoSurface(
        File = 'extractions.plt',
        Fields=requested_fields_in_iso_surface,
        IsoSurfaceField='CoordinateZ',
        IsoSurfaceValue=0.5,
    )

    w._interface.add_to_Extractions_3D(
        File = 'fields.plt',
        Fields=requested_fields_in_3d,
    )


    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()

    assert_tecplot_containing_expected_field('signals.plt',
        ['ForceX','ForceY','ForceZ'], tmp_path)

    assert_tecplot_containing_expected_field('extractions.plt',
        requested_fields_in_iso_surface+requested_fields_in_bc, tmp_path)

    assert_tecplot_containing_expected_field('fields.plt',
        requested_fields_in_3d, tmp_path)


@pytest.mark.integration
@pytest.mark.cost_level_2
@pytest.mark.skipif(solver=='sonics', reason="FIXME allow cgns transform and write in tecplot")
def test_write_extractions_in_tecplot_format_co2(tmp_path, niter=10):
    
    params = get_workflow2_parameters()
    params["RunManagement"]["RunDirectory"] = str(tmp_path)
    params["RunManagement"]["NumberOfProcessors"] = 2
    params["RunManagement"]['Scheduler'] = 'local'
    params["Numerics"]['NumberOfIterations'] = niter
    w = Workflow(**params)

    requested_fields_in_signals = ['Force']
    requested_fields_in_bc = ['Pressure','MomentumX']
    requested_fields_in_iso_surface = ['Density']
    requested_fields_in_3d = ['Density','MomentumX','MomentumY','MomentumZ']

    w._interface.add_to_Extractions_Integral(
        File='signals.plt',
        Name='TestSeparatedFile',
        Fields=requested_fields_in_signals, 
        Source='Ground',
    )

    w._interface.add_to_Extractions_BC(
        File = 'extractions.plt',
        Fields=requested_fields_in_bc,
        Source='Ground',
    )

    w._interface.add_to_Extractions_IsoSurface(
        File = 'extractions.plt',
        Fields=requested_fields_in_iso_surface,
        IsoSurfaceField='CoordinateZ',
        IsoSurfaceValue=0.5,
    )

    w._interface.add_to_Extractions_3D(
        File = 'fields.plt',
        Fields=requested_fields_in_3d,
    )


    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()

    assert_tecplot_containing_expected_field('signals/*.plt',
        ['ForceX','ForceY','ForceZ'], tmp_path)

    assert_tecplot_containing_expected_field('extractions/*.plt',
        requested_fields_in_iso_surface+requested_fields_in_bc, tmp_path)

    assert_tecplot_containing_expected_field('fields/*.plt',
        requested_fields_in_3d, tmp_path)


@pytest.mark.integration
@pytest.mark.cost_level_2
def test_integral_with_postprocess(tmp_path, niter=10):
    
    w = get_workflow_cart_monoproc(tmp_path)

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

    expected_number_of_items = niter

    assert_file_with_relevant_zone_and_fields(names.FILE_OUTPUT_1D, "Integral", "Ground",
            ['ForceX', 'ForceY', 'ForceZ', 'rsd-ForceX'], tmp_path, expected_number_of_items)

@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_2
def test_convergence_on_criterion(tmp_path, niter=20):
    
    w = get_workflow_cart_monoproc(tmp_path)

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
        expected_number_of_items = 11 
    elif w.Solver == 'fast':
        expected_number_of_items = 11
    else:
        raise AssertionError

    assert_file_with_relevant_zone_and_fields(names.FILE_OUTPUT_1D, "Integral", "Ground",
            ['rsd-ForceX'], tmp_path, expected_number_of_items)

if __name__ == '__main__':
    test_integrals_one_run('extract_integrals_one_run_'+os.environ.get("MOLA_SOLVER"))
    # test_integrals_two_runs('extract_integrals_two_runs_'+os.environ.get("MOLA_SOLVER"))
    # test_bc_one_run('test_bc_one_run_'+os.environ.get("MOLA_SOLVER"))