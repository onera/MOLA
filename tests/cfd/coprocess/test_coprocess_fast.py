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

pytestmark = pytest.mark.fast

import os

from treelab import cgns
from mola.cfd.coprocess import solver_fast
import mola.naming_conventions as names

def get_rans_tree():

    import Converter.PyTree as C
    import Generator.PyTree as G
    import Fast.PyTree as Fast
    import Initiator.PyTree as Init

    npts = 9
    dx = 0.5
    z = G.cart((0.0,0.0,0.0), (dx,dx,dx), (npts,npts,npts))
    C._addBC2Zone(z, 'WALL', 'FamilySpecified:WALL', 'imin')
    C._fillEmptyBCWith(z, 'FARFIELD', 'FamilySpecified:FARFIELD', dim=3)
    C._addState(z, 'GoverningEquations', 'NSTurbulent')

    Init._initConst(z, MInf=0.4, loc='centers')
    C._addState(z, MInf=0.4)
    t = C.newPyTree(['Base', z])
    C._tagWithFamily(t,'FARFIELD')
    C._tagWithFamily(t,'WALL')
    C._addFamily2Base(t, 'FARFIELD', bndType='BCFarfield')
    C._addFamily2Base(t, 'WALL', bndType='BCWall')

    import Dist2Walls.PyTree as DTW
    walls = C.extractBCOfType(t, 'BCWall')
    DTW._distance2Walls(t, walls, loc='centers', type='ortho')

    numb = { 'temporal_scheme': 'implicit', 'ss_iteration':3, 'modulo_verif':1}
    numz = { 'scheme':'roe', 'slope':'minmod',
        'time_step':0.0007,'time_step_nature':'local', 'cfl':4}
    Fast._setNum2Zones(t, numz); Fast._setNum2Base(t, numb)

    return t

def get_euler_tree():

    import Converter.PyTree as C
    import Generator.PyTree as G
    import Fast.PyTree as Fast
    import Initiator.PyTree as Init

    npts = 9
    dx = 0.5
    z = G.cart((0.0,0.0,0.0), (dx,dx,dx), (npts,npts,npts))
    C._fillEmptyBCWith(z, 'FARFIELD', 'FamilySpecified:FARFIELD', dim=3)
    C._addState(z, 'GoverningEquations', 'Euler')
    Init._initConst(z, MInf=0.4, loc='centers')
    C._addState(z, MInf=0.4)
    t = C.newPyTree(['Base', z])
    C._tagWithFamily(t,'FARFIELD')
    C._addFamily2Base(t, 'FARFIELD', bndType='BCFarfield')

    numb = { 'temporal_scheme': 'implicit', 'ss_iteration':3, 'modulo_verif':1}
    numz = { 'scheme':'roe', 'slope':'minmod',
        'time_step':0.0007,'time_step_nature':'local', 'cfl':4}
    Fast._setNum2Zones(t, numz); Fast._setNum2Base(t, numb)

    return t

def get_laminar_tree():

    import Converter.PyTree as C
    import Generator.PyTree as G
    import Fast.PyTree as Fast
    import Initiator.PyTree as Init

    npts = 9
    dx = 0.5
    z = G.cart((0.0,0.0,0.0), (dx,dx,dx), (npts,npts,npts))
    C._addBC2Zone(z, 'WALL', 'FamilySpecified:WALL', 'imin')
    C._fillEmptyBCWith(z, 'FARFIELD', 'FamilySpecified:FARFIELD', dim=3)
    C._addState(z, 'GoverningEquations', 'NSLaminar')
    Init._initConst(z, MInf=0.4, loc='centers')
    C._addState(z, MInf=0.4)
    t = C.newPyTree(['Base', z])
    C._tagWithFamily(t,'FARFIELD')
    C._tagWithFamily(t,'WALL')
    C._addFamily2Base(t, 'FARFIELD', bndType='BCFarfield')
    C._addFamily2Base(t, 'WALL', bndType='BCWall')

    numb = { 'temporal_scheme': 'implicit', 'ss_iteration':3, 'modulo_verif':1}
    numz = { 'scheme':'roe', 'slope':'minmod',
        'time_step':0.0007,'time_step_nature':'local', 'cfl':4}
    Fast._setNum2Zones(t, numz); Fast._setNum2Base(t, numb)

    return t


def get_fake_workflow_with_coprocess_manager(RunDirectory, type_of_tree='rans',
        create_convergence_nodes=False):
    import FastS.PyTree as FastS
    import Converter.Internal as I

    if type_of_tree == 'rans':
        t = get_rans_tree()

    elif type_of_tree == 'euler':
        t = get_euler_tree()

    elif type_of_tree == 'laminar':
        t = get_laminar_tree()


    else:
        raise ValueError(f'wrong type_of_tree={type_of_tree}')
    

    class FakeWorkflow():
        def __init__(self):                          
            self._status = 'BEFORE_FIRST_ITERATION'
            self.Numerics = dict(IterationAtInitialState=1,
                                      NumberOfIterations=5,
                                      TimeAtInitialState=0.0,
                                            TimeMarching='Steady')
            self.Extractions = []
            self.RunManagement = dict(RunDirectory=RunDirectory)
            self._expected_field_names = [
                'Density',
                'VelocityX',
                'VelocityY',
                'VelocityZ',
                'Temperature',
                'TurbulentSANuTilde']

            self.Fluid = dict(Gamma                  =  1.4,
                              IdealGasConstant       =  287.053,
                              Prandtl                =  0.72,
                              PrandtlTurbulent       =  0.9,
                              SutherlandConstant     = 110.4,
                              SutherlandViscosity    = 1.78938e-05,
                              SutherlandTemperature  = 288.15)

            from mola.cfd.coprocess.manager import CoprocessManager
            self._coprocess_manager = CoprocessManager(self)

    workflow = FakeWorkflow()
    I._addGhostCells(t,t,2,adaptBCs=1,fillCorner=0)
    t = cgns.castNode(t)
    for FlowEq in t.group(Type='FlowEquationSet_t'):
        cgns.Node(Name='EquationDimension',
                  Type='EquationDimension_t',
                  Value=3, Parent=FlowEq)

    t, tc, metrics = FastS.warmup(t, None)

    workflow._fast_metrics = metrics
    workflow.tree = cgns.castNode(t)
    if create_convergence_nodes:
        FastS._createConvergenceHistory(workflow.tree, workflow.Numerics['NumberOfIterations']+1)

        workflow.tree = cgns.castNode(workflow.tree)

    return workflow


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_output_tree(tmp_path):
    
    workflow = get_fake_workflow_with_coprocess_manager(tmp_path)
    solver_fast.get_output_tree(workflow, workflow._coprocess_manager)

    existing_field_names = solver_fast.get_field_names(workflow.tree)
    for expected_field_name in workflow._expected_field_names:
        assert expected_field_name in existing_field_names

    workflow._coprocess_manager._status = 'COMPLETED'


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("extract_field", solver_fast.post_fields_using_fast + \
        solver_fast.post_fields_using_cassiopee_computeVariables + 
        solver_fast.post_fields_using_cassiopee_computeExtraVariable)
def test_compute_missing_fields_at_cell_centers(tmp_path, extract_field):
    
    workflow = get_fake_workflow_with_coprocess_manager(tmp_path)

    if extract_field == 'ShearStress':
        solver_fast.compute_missing_fields_at_cell_centers( workflow, workflow.tree,
                                                       ['ViscosityMolecular'])
    solver_fast.compute_missing_fields_at_cell_centers( workflow, workflow.tree,
                                                       [extract_field])

    existing_field_names = solver_fast.get_field_names(workflow.tree)
    expected_field_names = workflow._expected_field_names[:]
    
    if extract_field == 'Vorticity':
        for c in 'XYZ':
            expected_field_names.append(extract_field+c)
    
    elif extract_field == 'ShearStress':
        for c in ('XX','YY','ZZ','XY','XZ','YZ'):
            expected_field_names.append(extract_field+c)
    
    else:
        expected_field_names.append(extract_field)

    zone = workflow.tree.zones()[0]
    zone_dim = zone.value()
    fs = workflow.tree.zones()[0].get('FlowSolution#Centers',Depth=1)
    for expected_field_name in expected_field_names:

        assert expected_field_name in existing_field_names

        field = fs.get(expected_field_name).value()

        assert zone_dim[0,1] == field.shape[0]
        assert zone_dim[1,1] == field.shape[1]
        assert zone_dim[2,1] == field.shape[2]

    workflow._coprocess_manager._status = 'COMPLETED'



@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("GridLocation", ["Vertex", "CellCenter"])
@pytest.mark.parametrize("GhostCells", [True, False])
def test_extract_fields(tmp_path, GridLocation, GhostCells):


    workflow = get_fake_workflow_with_coprocess_manager(tmp_path)
    
    workflow._coprocess_manager.Extractions = [
        dict(Fields=['Density','MomentumX','MomentumY','MomentumZ',
                    'Mach','ViscosityMolecular','Vorticity'],
             GridLocation=GridLocation, GhostCells=GhostCells,
             Container='FlowSolution#Output',
             Type='3D')]
    workflow.Extractions = workflow._coprocess_manager.Extractions
    
    output_tree = solver_fast.get_output_tree(workflow, workflow._coprocess_manager)
    
    for extraction in workflow._coprocess_manager.Extractions:
        tRef = solver_fast.extract_fields(output_tree, extraction)
        
        computed_fields = solver_fast.get_field_names(tRef, container=extraction['Container'])
        for expected_field_name in extraction['Fields']:
            if expected_field_name == 'Vorticity':
                for c in 'XYZ':
                    assert expected_field_name+c in computed_fields
            else:
                assert expected_field_name in computed_fields

    workflow._coprocess_manager._status = 'COMPLETED'


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_extract_isosurface(tmp_path):

    workflow = get_fake_workflow_with_coprocess_manager(tmp_path)
    
    workflow._coprocess_manager.Extractions = [
        dict(Fields=['Density','VelocityX'],
             IsoSurfaceField='CoordinateX',
             Name='MySlice',
             IsoSurfaceContainer='auto',
             IsoSurfaceValue=0.1,
             ContainersToTransfer=[names.CONTAINER_OUTPUT_FIELDS_AT_VERTEX],
             Type='IsoSurface')]
    workflow.Extractions = workflow._coprocess_manager.Extractions
    
    output_tree = solver_fast.get_output_tree(workflow, workflow._coprocess_manager)
    
    for extraction in workflow._coprocess_manager.Extractions:
        extraction['Data'] = solver_fast.extract_isosurface(output_tree, extraction)

        tRef = extraction['Data']

        zone = tRef.zones()[0]
        container_names = [n.name() for n in zone.group(Type="FlowSolution_t", Depth=1)]
        assert names.CONTAINER_OUTPUT_FIELDS_AT_VERTEX in container_names
        fs = zone.get(Name=names.CONTAINER_OUTPUT_FIELDS_AT_VERTEX, Depth=1)
        computed_fields = [n.name() for n in fs.group(Type='DataArray_t', Depth=1)]
        for expected_field_name in extraction['Fields']:
            if expected_field_name == 'Vorticity':
                for c in 'XYZ':
                    assert expected_field_name+c in computed_fields
            else:
                assert expected_field_name in computed_fields

    workflow._coprocess_manager._status = 'COMPLETED'


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_extract_bc(tmp_path):

    workflow = get_fake_workflow_with_coprocess_manager(tmp_path)
    
    workflow._coprocess_manager.Extractions = [
        dict(Type='BC', Source='WALL', Fields=['Pressure', 'Temperature'], Name='ByFamily',ContainersToTransfer=[names.CONTAINER_OUTPUT_FIELDS_AT_CENTER],),
        dict(Type='BC', Source='FARFIELD', Fields=['Pressure', 'MomentumX'], Name='ByFamily',ContainersToTransfer=[names.CONTAINER_OUTPUT_FIELDS_AT_CENTER],)]

    workflow.Extractions = workflow._coprocess_manager.Extractions
    
    output_tree = solver_fast.get_output_tree(workflow, workflow._coprocess_manager)
    families_to_bctype = solver_fast.get_family_to_BCType(output_tree)

    for extraction in workflow._coprocess_manager.Extractions:
        tRef = solver_fast.extract_bc(output_tree, extraction, families_to_bctype,
                                      workflow._fast_metrics)
        src = extraction["Source"]
        tRef.save(os.path.join(tmp_path,f'extraction_{src}.cgns'))
        
        computed_fields = solver_fast.get_field_names(tRef,
                                    container=names.CONTAINER_OUTPUT_FIELDS_AT_CENTER)
        for expected_field_name in extraction['Fields']:
            assert expected_field_name in computed_fields

    workflow._coprocess_manager._status = 'COMPLETED'


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("modeling", ['euler', 'rans'])
def test_extract_residuals(tmp_path,modeling):

    workflow = get_fake_workflow_with_coprocess_manager(tmp_path, modeling, create_convergence_nodes=True)
    workflow._coprocess_manager.Extractions = [dict(Type='Residuals')]
    workflow.Extractions = workflow._coprocess_manager.Extractions

    import FastS.PyTree as FastS

    t = workflow.tree
    tc = None
    metrics = workflow._fast_metrics
    graph = None

    inititer = workflow.Numerics['IterationAtInitialState']
    niter = workflow.Numerics['NumberOfIterations']
    
    for it in range( inititer-1, inititer+niter ):
        FastS._compute(t, metrics, it, tc, graph)
        FastS.display_temporal_criteria(t, metrics, it, format='store')
    
    workflow.tree = cgns.castNode(t)
    
    output_tree = solver_fast.get_output_tree(workflow, workflow._coprocess_manager)
    
    extraction = None
    for e in workflow._coprocess_manager.Extractions:
        if e["Type"] == "Residuals":
            extraction = e
    if extraction is None:
        raise AttributeError("residuals extraction not found")

    solver_fast.extract_residuals(output_tree, extraction)

    workflow._coprocess_manager._status = 'COMPLETED'


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_stress_and_state(tmp_path):
    workflow = get_fake_workflow_with_coprocess_manager(tmp_path, 'laminar')
    
    workflow._coprocess_manager.Extractions = [
        dict(Type='Integral', Source='WALL', Name='WALL_LOADS')]
    workflow.Extractions = workflow._coprocess_manager.Extractions

    output_tree = solver_fast.get_output_tree(workflow, workflow._coprocess_manager)
    stress, state = solver_fast.get_stress_and_state(output_tree, 'WALL',
                                                     workflow._fast_metrics)
    
    expected_stress_keys=('fx','fy','fz','t0x','t0y','t0z','S','m','ForceX','ForceY','ForceZ')
    for k in expected_stress_keys: assert k in stress

    expected_state_keys=('Density','MomentumX','MomentumY','MomentumZ')
    for k in expected_state_keys: assert k in state

    workflow._coprocess_manager._status = 'COMPLETED'


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_dimensionalize_torque():

    stress = dict(t0x=1.0, t0y=1.0, t0z=1.0, S=0.1)
    state = dict(Density=1.0, MomentumX=1.0, MomentumY=1.0, MomentumZ=1.0)

    solver_fast.dimensionalize_torque(stress, state)

    expected_stress_keys = ('Torque0X', 'Torque0Y', 'Torque0Z')
    for k in expected_stress_keys: assert k in stress


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_extract_integral(tmp_path):
    
    workflow = get_fake_workflow_with_coprocess_manager(tmp_path, 'laminar')
    
    extraction = dict(Type='Integral', Source='WALL', Name='WALL_LOADS',
                      Fields=['Force','Torque','MassFlow'],
                      FluxCoef=1.0)
    workflow._coprocess_manager.Extractions = [ extraction ]
    workflow.Extractions = workflow._coprocess_manager.Extractions

    import FastS.PyTree as FastS

    niter = 3
    for it in range( niter ):
        FastS._compute(workflow.tree, workflow._fast_metrics, it)
    
        workflow._coprocess_manager.iteration = it
        workflow.tree = cgns.castNode(workflow.tree)

        output_tree = solver_fast.get_output_tree(workflow, workflow._coprocess_manager)
        
        solver_fast.extract_integral(output_tree, extraction, workflow)

    flow_sol = extraction['Data'].get('FlowSolution')
    
    assert flow_sol

    expected_integrals = ('Iteration','ForceX',  'ForceY',   'ForceZ',
                          'MassFlow',     'TorqueX','TorqueY', 'TorqueZ')
    for k in expected_integrals: 
        expected_node = flow_sol.get(k, Type='DataArray_t')
        assert expected_node
        assert len(expected_node.value()) == niter

    workflow._coprocess_manager._status = 'COMPLETED'



if __name__ == '__main__':
    # test_compute_missing_fields_at_cell_centers('wkflw_fields','Vorticity')
    # test_get_output_tree('wkflw_'+os.environ.get("MOLA_SOLVER"))
    # test_extract_fields('wkflw2', 'CellCenter', True)
    # test_remove_not_requested_fields()
    # test_extract_bc('bc')
    test_extract_residuals('residuals','rans')
    # test_extract_integral('integral')