import pytest

import os
import shutil
from pathlib import Path

import numpy as np
from dataclasses import dataclass

from treelab import cgns

import mola.naming_conventions as names
from mola.logging import check_error_message, MolaException
from mola import server as SV
from mola.workflow import Workflow
import mola.workflow.manager as WM

def get_fake():
    @dataclass
    class Fake(Workflow):

        BoundaryConditions = [
            dict(Family='INFLOW'),
            dict(Family='OUTFLOW', Type='OutflowPressure', Pressure=10),
        ]

        RunManagement = dict(
            RunDirectory = 'test',
            other = 3,
        )
    return Fake()

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_value_on_leaf_1():
    d = dict(x=1, y=dict(z=3))
    WM.set_value_on_leaf(d, ['y', 'z'], 2)
    assert d['y']['z'] == 2

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_value_on_leaf_2():
    l = [dict(x=1), dict(y=2, z=3)]
    WM.set_value_on_leaf(l, ['y=2', 'z'], 5)
    assert l[1]['z'] == 5


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_value_on_leaf_3():
    fake = get_fake()
    WM.set_value_on_leaf(fake, ['RunManagement', 'RunDirectory'], 'other_test')
    assert fake.RunManagement['RunDirectory'] == 'other_test'


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_value_on_leaf_4():
    fake = get_fake()
    WM.set_value_on_leaf(fake, ['BoundaryConditions', 'Family=OUTFLOW', 'Pressure'], 20)
    assert fake.BoundaryConditions[1]['Pressure'] == 20


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_value_on_leaf():
    fake = get_fake()
    value = WM.get_value_on_leaf(fake, ['BoundaryConditions', 'Family=OUTFLOW', 'Pressure'])
    assert value == 10

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_find_matching_leaves():
    class Fake(Workflow):

        def __init__(self):

            self.RawMeshComponents = [
                dict(
                    Name = 'test',
                    Source = 'source.cgns',
                )
            ]

            self.BoundaryConditions = [
                dict(Family='INFLOW', File='inflow_map.cgns'),
                dict(Family='OUTFLOW', Type='OutflowPressure', Pressure=10),
            ]

            self.Initialization = dict(
                Method = 'copy',
                Source = 'init.cgns',
            )

            self.Extractions = [
                dict(Type='BC', File='extractions.cgns'),
            ]

    def add_preffix(name):
        return 'new_'+name

    workflow = Fake()
    filenames = WM.find_matching_leaves(workflow, ['*.cgns'], operation=add_preffix, excluded_attributes=['Extractions'])
    assert set(filenames) == {'source.cgns', 'inflow_map.cgns', 'init.cgns'}
    assert workflow.RawMeshComponents[0]['Source'] == 'new_source.cgns'
    assert workflow.BoundaryConditions[0]['File'] == 'new_inflow_map.cgns'
    assert workflow.Initialization['Source'] == 'new_init.cgns'
    assert workflow.Extractions[0]['File'] == 'extractions.cgns'

def get_fake_workflow():
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
            ),
        ],

        Flow=dict(
            Density = 0.2,
            Temperature = 100.,
            Velocity = 50.,

        ),

        Solver='elsa',

        Numerics = dict(
            NumberOfIterations=10,
            CFL=1.,
        ),

        BoundaryConditions=[
            dict(Family='Wall', Type='Wall'),
            dict(Family='Farfield', Type='Farfield', Pressure=10.0),
        ],

        RunManagement=dict(
            RunDirectory='test_10',
            NumberOfProcessors=4,
            ),

        )

    return w

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_dispatcher_error_new_job_empty():

    w = object()
    dispatcher = WM.WorkflowDispatcher(w)
    err_msg = 'Before calling `add_variations`, `new_job` must be called first to declare directory.'
    check_error_message(err_msg, dispatcher.add_variations, ['fake'])


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_dispatcher_directories():

    w = get_fake_workflow()
    dispatcher = WM.WorkflowDispatcher(w)

    dispatcher.new_job('root')
    for index in [10, 20, 30]:
        dispatcher.add_variations([('RunManagement|RunDirectory', f'test_{index}')])

    current_directories = dispatcher.get_directories_in_current_job()
    assert current_directories == ['test_10', 'test_20', 'test_30']

    dispatcher.new_job('root2')
    for index in [40, 50]:
        dispatcher.add_variations([('RunManagement|RunDirectory', f'test_{index}')])

    current_directories = dispatcher.get_directories_in_current_job()
    assert current_directories == ['test_40', 'test_50']
    directories = dispatcher.get_directories()
    assert directories == ['root/test_10', 'root/test_20', 'root/test_30', 'root2/test_40', 'root2/test_50']

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_dispatcher_convert_to_dict_and_read():

    w = get_fake_workflow()
    dispatcher = WM.WorkflowDispatcher(w)

    dispatcher.new_job('root')
    for index in [10, 20, 30]:
        dispatcher.add_variations([('RunManagement|RunDirectory', f'test_{index}')])

    d = dispatcher.convert_to_dict()

    tree = cgns.Tree()
    tree.setParameters(
        ContainerName = 'WorkflowManager',
        WorkflowDispatcher = d,
        )

    dispatcher_read = WM.read_workflow_dispatcher_from_tree(tree.get(Name='WorkflowDispatcher', Depth=2))
    assert dispatcher == dispatcher_read

@pytest.mark.integration
@pytest.mark.cost_level_1
def test_WorkflowManager_prepare(tmp_path):

    test_dir = str(tmp_path)
    w = get_fake_workflow()
    manager_file_path = str(tmp_path/names.FILE_WORKFLOW_MANAGER)
    manager = WM.WorkflowManager(w, root_directory=test_dir, manager_file_path=manager_file_path)

    for model in ['model1', 'model2']:
        manager.new_job(model)
        for pressure in [10, 20, 30]:
            manager.add_variations([('RunManagement|RunDirectory', f'test_{pressure}')])

    manager.prepare()

    root_dirs = []
    files_list = []
    for root, dirs, files in os.walk(test_dir):
        root_dirs.append(root)
        files_list.append(files)

    assert set(Path(p) for p in root_dirs) == {
        tmp_path, 
        tmp_path/'model1',
        tmp_path/'model1'/'test_10',
        tmp_path/'model1'/'test_20',
        tmp_path/'model1'/'test_30',
        tmp_path/'model2',
        tmp_path/'model2'/'test_10',
        tmp_path/'model2'/'test_20',
        tmp_path/'model2'/'test_30',
        }

    assert files_list == [
        [names.FILE_WORKFLOW_MANAGER],
        [names.FILE_JOB_SEQUENCE], 
        [names.FILE_INPUT_WORKFLOW], 
        [names.FILE_INPUT_WORKFLOW], 
        [names.FILE_INPUT_WORKFLOW], 
        [names.FILE_JOB_SEQUENCE], 
        [names.FILE_INPUT_WORKFLOW], 
        [names.FILE_INPUT_WORKFLOW], 
        [names.FILE_INPUT_WORKFLOW],
        ]
    
    for workflows in manager.dispatcher.table_of_workflows:
        init_files = []
        for workflow in workflows:
            try:
                source = workflow.Initialization['Source']
                init_files.append(source)
            except:
                init_files.append('no_file')
        assert init_files == ['no_file', '../test_10/main.cgns', '../test_20/main.cgns']
    
@pytest.mark.integration
@pytest.mark.cost_level_1
def test_WorkflowManager_prepare_remote_sator(tmp_path):

    test_dir = str(tmp_path)
    w = get_fake_workflow()
    w.RunManagement['Machine'] = 'sator'
    manager_file_path = str(tmp_path/names.FILE_WORKFLOW_MANAGER)
    manager = WM.WorkflowManager(w, root_directory=test_dir, manager_file_path=manager_file_path)

    for model in ['model1', 'model2']:
        manager.new_job(model)
        for pressure in [10, 20, 30]:
            manager.add_variations([('RunManagement|RunDirectory', f'test_{pressure}')])

    manager.prepare()
    
    for sequence_of_workflows in manager.sequential_managers:
        init_files = []
        for workflow in sequence_of_workflows.workflows:
            try:
                source = workflow.Initialization['Source']
                init_files.append(source)
            except:
                init_files.append('no_file')
        assert init_files == ['no_file', '../test_10/main.cgns', '../test_20/main.cgns']
    
# TODO adapt to run it with sonics too. For now, it won't work because the mesh is strucutred
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.integration
@pytest.mark.cost_level_4
def test_WorkflowManager_cart_local(tmp_path):

    if isinstance(tmp_path,str): os.makedirs(tmp_path,exist_ok=True)

    from .test_workflow import  get_workflow_cart_monoproc
    w = get_workflow_cart_monoproc(tmp_path)
    w.RunManagement["Scheduler"] = "local"

    # since mesh is built in memory, manager requires to save it in a file
    mesh_path = os.path.join(tmp_path,'mesh.cgns')
    w.RawMeshComponents[0]['Source'].save(mesh_path)
    w.RawMeshComponents[0]['Source'] = os.path.join('..','..','mesh.cgns') # CAUTION: path is relative to launch case

    test_dir = str(tmp_path)
    manager_file_path = str(tmp_path/names.FILE_WORKFLOW_MANAGER)
    manager = WM.WorkflowManager(w, test_dir, manager_file_path=manager_file_path)
    for BCWall in ['WallViscous',]:
        manager.new_job(BCWall)
        for velocity in [50., 20.]:
            manager.add_variations(
                [
                    ('RunManagement|RunDirectory', f'Velocity_{velocity}'),
                    ('Flow|Velocity', velocity),
                    ('BoundaryConditions|Family=Ground|Type', BCWall),
                ], 
                initialize_from_previous=False
                )
    
    manager.prepare()
    manager.submit()

    # this requires job to have finished, which is the case only if we 
    # impose w.RunManagement["Scheduler"] = "local". Otherwise we have a 
    # synchronicity issue (jobs are submitted, and the following checks are
    # done before the simulations are run)
    for BCWall in ['WallViscous',]:
        for velocity in [50., 20.]:
            COMPLETED_PATH = os.path.join(manager.root_directory, BCWall, f'Velocity_{velocity}', names.FILE_JOB_COMPLETED)
            if not os.path.exists(COMPLETED_PATH):
                raise MolaException(f'simulation did not end as expected: unable to find file {COMPLETED_PATH}')

@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.integration
@pytest.mark.cost_level_1
def test_WorkflowManager_write_local(tmp_path):

    if isinstance(tmp_path,str): os.makedirs(tmp_path,exist_ok=True)

    from .test_workflow import  get_workflow_cart_monoproc
    w = get_workflow_cart_monoproc(tmp_path)
    w.RunManagement["Scheduler"] = "local"

    # since mesh is built in memory, manager requires to save it in a file
    mesh_path = os.path.join(tmp_path,'mesh.cgns')
    w.RawMeshComponents[0]['Source'].save(mesh_path)
    w.RawMeshComponents[0]['Source'] = os.path.join('..','..','mesh.cgns') # CAUTION: path is relative to launch case

    test_dir = str(tmp_path)
    manager_file_path = str(tmp_path/names.FILE_WORKFLOW_MANAGER)
    written_manager = WM.WorkflowManager(w, test_dir, manager_file_path=manager_file_path)
    for BCWall in ['WallViscous',]:
        written_manager.new_job(BCWall)
        for velocity in [50., 20.]:
            written_manager.add_variations(
                [
                    ('RunManagement|RunDirectory', f'Velocity_{velocity}'),
                    ('Flow|Velocity', velocity),
                    ('BoundaryConditions|Family=Ground|Type', BCWall),
                ], 
                initialize_from_previous=False
                )
    
    written_manager.write()
    read_manager = WM.WorkflowManager(manager_file_path)
    assert read_manager == written_manager
    
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.network_onera
@pytest.mark.integration
@pytest.mark.cost_level_4
def test_WorkflowManager_sphere_remote_sator(tmp_path):

    from .test_workflow import  get_workflow_sphere_struct
    w = get_workflow_sphere_struct('.')
    scheduler_defaults = SV.get_scheduler_defaults('sator')
    w.RunManagement['AER'] = scheduler_defaults.AER_FOR_TEST
    w.RunManagement['TimeLimit'] = '00:30:00'

    test_dir = f'/tmp_user/sator/{os.getenv("USER")}/.test_WorkflowManager_sphere_remote_sator_{os.getenv("MOLA_SOLVER")}/tmp_MOLA_test/'
    try:
        # remove this directory in case it exists already (e.g. because of a previous error)
        SV.remove_path(test_dir, machine='sator', file_only=False)
    except FileNotFoundError:
        pass

    manager_file_path = str(tmp_path/names.FILE_WORKFLOW_MANAGER)
    manager = WM.WorkflowManager(w, test_dir,manager_file_path=manager_file_path)
    for BCWall in ['WallViscous', 'WallInviscid']:
        manager.new_job(BCWall)
        for velocity in [50., 20., 80.]:
            manager.add_variations(
                [
                    ('RunManagement|JobName', f'test_{BCWall}'),
                    ('RunManagement|RunDirectory', f'Velocity_{velocity}'),
                    ('Flow|Velocity', velocity),
                    ('BoundaryConditions|Family=Wall|Type', BCWall),
                ], 
                initialize_from_previous=False
                )
            
    manager.prepare()
    manager.submit()

    # NOTE: do not wait for job to end, since that approach would provoke
    # too important delays (waiting for resources of SLURM)
    # for BCWall in ['WallViscous', 'WallInviscid']:
    #     for velocity in [50., 20., 80.]:
    #         COMPLETED_PATH = os.path.join(test_dir, BCWall, f'Velocity_{velocity}', names.FILE_JOB_COMPLETED)
    #         SV.wait_until(SV.is_existing_path, path=COMPLETED_PATH, machine='sator', timeout=180)

    # SV.remove_path(test_dir, machine='sator', file_only=False)



if __name__ == '__main__':
    # test_show_interface_1()
    # test_prepare_workflow1()
    test_WorkflowManager_cart_local('cart_scheduler_'+os.environ.get("MOLA_SOLVER"))
    # test_wip()
