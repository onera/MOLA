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
import copy
import numpy as np
from treelab import cgns
from mola.cfd.preprocess.extractions import solver_elsa
import mola.naming_conventions as names
from mola.workflow import Workflow
from ....workflow.test_workflow import get_workflow2, get_workflow2_parameters

pytestmark = pytest.mark.elsa


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_add_extractions_for_overset_components():

    def add_overset_component(w):
        w.RawMeshComponents = [dict(OversetOptions=dict(option=True))]

    workflow = get_workflow2()
    workflow.compute_flow_and_turbulence()

    # Check that nothing happend if not overset components
    ref_Extractions = copy.copy(workflow.Extractions)
    solver_elsa.add_extractions_for_overset_components(workflow)
    assert workflow.Extractions == ref_Extractions

    # Check behavior with overset components
    add_overset_component(workflow)
    solver_elsa.add_extractions_for_overset_components(workflow)

    last_extraction = workflow.Extractions[-1]
    assert last_extraction['Type'] == '3D'
    assert last_extraction['Fields'] == ['CoordinateX', 'CoordinateY', 'CoordinateZ']
    assert last_extraction['Container'] == 'FlowSolution#EndOfRun#Coords'
    assert last_extraction['Frame'] == 'absolute'
    assert last_extraction['File'] == names.FILE_OUTPUT_3D


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_add_trigger():
    ref_trigger = ['ELSA_TRIGGER', None, [
        ['.Solver#Trigger', None, [
            ['next_state', np.array([16], dtype=np.int32), [], 'DataArray_t'], 
            ['next_iteration', np.array([1], dtype=np.int32), [], 'DataArray_t'], 
            ['file', np.array([b'c', b'o', b'p', b'r', b'o', b'c', b'e', b's', b's', b'.', b'p', b'y'], dtype='|S1'), [], 'DataArray_t']
        ], 'UserDefinedData_t']], 'Family_t']

    workflow = get_workflow2()
    workflow.assemble()
    solver_elsa.add_trigger(workflow.tree)

    for zone in workflow.tree.zones():
        assert zone.get(Name='ELSA_TRIGGER', Type='AdditionalFamilyName', Value='ELSA_TRIGGER')

    for base in workflow.tree.bases():
        trigger_fam_node = base.get(Name='ELSA_TRIGGER', Type='Family')
        assert str(trigger_fam_node) == str(ref_trigger)
    


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_global_convergence_history():
    ref_node = ['GlobalConvergenceHistory', np.array([0], dtype=np.int32), [
        ['NormDefinitions', np.array([b'C', b'o', b'n', b'v', b'e', b'r', b'g', b'e', b'n', b'c', b'e', b'H', b'i', b's', b't', b'o', b'r', b'y'], dtype='|S1'), [], 'Descriptor_t'], 
        ['.Solver#Output', None, [
            ['period', np.array([1], dtype=np.int32), [], 'DataArray_t'], 
            ['writingmode', np.array([0], dtype=np.int32), [], 'DataArray_t'], 
            ['var', np.array([b'r', b'e', b's', b'i', b'd', b'u', b'a', b'l', b'_', b'c', b'o', b'n', b's', b' ', b'r', b'e', b's', b'i', b'd', b'u', b'a', b'l', b'_', b't', b'u', b'r', b'b'], dtype='|S1'), [], 'DataArray_t']
        ], 'UserDefinedData_t']
        ], 'UserDefinedData_t']
    
    workflow = get_workflow2()
    workflow.assemble()
    solver_elsa.add_global_convergence_history(workflow)
    for base in workflow.tree.bases():
        conv_node = base.get(Name='GlobalConvergenceHistory')   
        assert str(conv_node) == str(ref_node)  


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_process_extractions_of_type_field_base():
    params = get_workflow2_parameters()
    params['Extractions'] = [dict(Type='3D', Fields=['Density', 'MomentumX'], Container='FlowSolution#Output')]
    workflow = Workflow(**params)
    workflow.assemble()
    solver_elsa.process_extractions_of_type_field(workflow)

    zone = workflow.tree.zones()[0]
    FS = zone.get(Name='FlowSolution#Output', Type='FlowSolution')

    assert FS is not None
    assert FS.get(Name='period').value() == 1
    assert FS.get(Name='writingmode').value() == 2
    assert FS.get(Name='writingframe').value() == 'relative'
    assert FS.get(Name='loc').value() == 'node'
    assert FS.get(Name='var').value() == ['ro', 'rovx']

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_process_extractions_3d_additional_variables():
    params = get_workflow2_parameters()
    params['Extractions'] = [
        dict(Type='3D', Fields=['Density', 'MomentumX'], Container='FS#Output3D'),
        dict(Type='3D', Fields=['MomentumX', 'Mach', 'Pressure'], Container='FS#Output3D'),
        ]
    workflow = Workflow(**params)
    workflow.assemble()
    solver_elsa.process_extractions_of_type_field(workflow)

    zone = workflow.tree.zones()[0]
    FS = zone.get(Name='FS#Output3D', Type='FlowSolution')
   
    assert FS is not None
    assert FS.get(Name='period').value() == 1
    assert FS.get(Name='writingmode').value() == 2
    assert FS.get(Name='writingframe').value() == 'relative'
    assert FS.get(Name='loc').value() == 'node'
    assert FS.get(Name='var').value() == ['ro', 'rovx', 'mach', 'psta']



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_process_extractions_3d_coords():
    params = get_workflow2_parameters()
    params['Extractions'] = [
        dict(Type='3D', Container='FlowSolution#EndOfRun#Coords', 
             Fields=['CoordinateX', 'CoordinateY', 'CoordinateZ'], 
             GridLocation='Vertex', Frame='absolute')
             ]
    workflow = Workflow(**params)
    workflow.assemble()
    solver_elsa.process_extractions_of_type_field(workflow)

    zone = workflow.tree.zones()[0]
    FS = zone.get(Name='FlowSolution#EndOfRun#Coords', Type='FlowSolution')

    assert FS is not None
    assert FS.get(Name='period').value() == 1
    assert FS.get(Name='writingmode').value() == 2
    assert FS.get(Name='writingframe').value() == 'absolute'
    assert FS.get(Name='loc').value() == 'node'
    assert FS.get(Name='var').value() == ['x', 'y', 'z']

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_process_extractions_3d_average():
    params = get_workflow2_parameters()
    params['Extractions'] = [dict(Type='3D', Container='FlowSolution#Average', Fields=['Density', 'MomentumX'], OtherOptions=dict(average='time', period_init='inactive'))]
    workflow = Workflow(**params)
    workflow.assemble()
    solver_elsa.process_extractions_of_type_field(workflow)

    zone = workflow.tree.zones()[0]
    FS = zone.get(Name='FlowSolution#Average', Type='FlowSolution')

    assert FS is not None
    assert FS.get(Name='period').value() == 1
    assert FS.get(Name='writingmode').value() == 2
    assert FS.get(Name='writingframe').value() == 'relative'
    assert FS.get(Name='loc').value() == 'node'
    assert FS.get(Name='var').value() == ['ro', 'rovx']
    assert FS.get(Name='average').value() == 'time'
    assert FS.get(Name='period_init').value() == 'inactive'


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_adapt_variables_for_2d_extraction_wall():

    workflow = get_workflow2()
    workflow.assemble()
    workflow.compute_flow_and_turbulence()
    for zone in workflow.tree.zones():
        cgns.Node(Name='ZoneType', Type='ZoneType', Value='Structured', Parent=zone)

    Extraction = dict(Type='bc', Source='BCWall', Fields=['Pressure', 'BoundaryLayer', 'yPlus', 
                                                          'geomdepdom','delta_cell_max','delta_compute',
                                                          'vortratiolim','shearratiolim','pressratiolim'])  # BCType is not used by adapt_variables_for_2d_extraction
    ExtractBCType = 'BCWall'
    ExtractVariablesList = solver_elsa.adapt_variables_for_2d_extraction(workflow, Extraction, ExtractBCType)

    assert ExtractVariablesList == ['Pressure', 'BoundaryLayer', 'yPlus', 
                                    'geomdepdom','delta_cell_max','delta_compute',
                                    'vortratiolim','shearratiolim','pressratiolim']


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_adapt_variables_for_2d_extraction_BCWallInviscid():

    workflow = get_workflow2()
    workflow.assemble()
    workflow.compute_flow_and_turbulence()
    for zone in workflow.tree.zones():
        cgns.Node(Name='ZoneType', Type='ZoneType', Value='Structured', Parent=zone)

    Extraction = dict(Type='BC', Source='BCWall', Fields=['Pressure', 'BoundaryLayer', 'yPlus', 
                                                          'geomdepdom','delta_cell_max','delta_compute',
                                                          'vortratiolim','shearratiolim','pressratiolim'])  # BCType is not used by adapt_variables_for_2d_extraction
    ExtractBCType = 'BCWallInviscid'
    ExtractVariablesList = solver_elsa.adapt_variables_for_2d_extraction(workflow, Extraction, ExtractBCType)

    assert ExtractVariablesList == ['Pressure']



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_adapt_variables_for_2d_extraction_unstructured():

    workflow = get_workflow2()
    workflow.assemble()
    workflow.compute_flow_and_turbulence()
    for zone in workflow.tree.zones():
        ZoneType = cgns.Node(Name='ZoneType', Type='ZoneType', Value='Unstructured') 
        zone.addChild(ZoneType)

    Extraction = dict(Type='bc', BCType='BCWall', Fields=['Pressure', 'BoundaryLayer', 'yPlus', 
                                                          'geomdepdom','delta_cell_max','delta_compute',
                                                          'vortratiolim','shearratiolim','pressratiolim'])  # Source is not used by adapt_variables_for_2d_extraction
    ExtractBCType = 'BCWall'
    ExtractVariablesList = solver_elsa.adapt_variables_for_2d_extraction(workflow, Extraction, ExtractBCType)

    assert ExtractVariablesList == ['Pressure', 'yPlus', 
                                    'geomdepdom','delta_cell_max','delta_compute',
                                    'vortratiolim','shearratiolim','pressratiolim']



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_adapt_variables_for_2d_extraction_TransitionMode_NonLocalCriteria_LSTT():

    workflow = get_workflow2()
    workflow.assemble()
    workflow.compute_flow_and_turbulence()
    for zone in workflow.tree.zones():
        cgns.Node(Name='ZoneType', Type='ZoneType', Value='Structured', Parent=zone)
    workflow.Turbulence = dict(TransitionMode='NonLocalCriteria-LSTT')

    Extraction = dict(Type='bc', BCType='BCWall', Fields=['Pressure'])  # Source is not used by adapt_variables_for_2d_extraction
    ExtractBCType = 'BCWall'
    ExtractVariablesList = solver_elsa.adapt_variables_for_2d_extraction(workflow, Extraction, ExtractBCType)
    
    assert ExtractVariablesList == ['Pressure', 'intermittency', 'clim', 'how', 
                                    'origin','lambda2', 'turb_level', 'n_tot_ag', 
                                    'n_crit_ag', 'r_tcrit_ahd', 'r_theta_t1', 
                                    'line_status', 'crit_indicator']



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_adapt_variables_for_2d_extraction_TransitionMode_imposed():

    workflow = get_workflow2()
    workflow.assemble()
    workflow.compute_flow_and_turbulence()
    for zone in workflow.tree.zones():
        cgns.Node(Name='ZoneType', Type='ZoneType', Value='Structured', Parent=zone)
    workflow.Turbulence = dict(TransitionMode='Imposed')

    Extraction = dict(Type='BC', BCType='BCWall', Fields=['Pressure'])  # BCType is not used by adapt_variables_for_2d_extraction
    ExtractBCType = 'BCWall'
    ExtractVariablesList = solver_elsa.adapt_variables_for_2d_extraction(workflow, Extraction, ExtractBCType)
    
    assert ExtractVariablesList == ['Pressure', 'intermittency', 'clim']


def get_workflow_1():

    class FakeWorkflow():
        def __init__(self,):

            self.SolverParameters = dict(
                model = dict(
                    delta_compute   = 'first_order_bl',
                    vortratiolim    = 1e-3,
                    shearratiolim   = 2e-2,
                    pressratiolim   = 1e-3,
                )
            )

            self.Flow = dict(Pressure=1e5)

            self.Turbulence = dict(Model='SA')

            self.tree = cgns.Tree()

    w = FakeWorkflow()
    return w



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_add_2d_extractions_in_SolverOutput_wall():

    workflow = get_workflow_1()

    FamilyNode = cgns.Node(Name='FamilyA', Type='Family')
    cgns.Node(Name='FamilyBC', Type='FamilyBC', Value='BCWall', Parent=FamilyNode)

    Extraction = dict(Type="BC",
                      Fields=['Pressure', 'BoundaryLayer'],
                      Source="FamilyA",
                      Name="FamilyA",
                      ExtractionPeriod=1,
                      GridLocation="CellCenter",
                      Frame="absolute")

    
    solver_elsa.add_2d_extractions_in_SolverOutput(FamilyNode, Extraction, workflow)

    solver_output = FamilyNode.getParameters('.Solver#Output',transform_numpy_scalars=True)
    solver_output_ref = dict(
        period=Extraction["ExtractionPeriod"],
        writingmode=2,
        loc = "interface",
        fluxcoeff = 1.0,
        writingframe=Extraction["Frame"], 
        pinf=workflow.Flow['Pressure'],
        torquecoeff=1.0,
        xtorque=0.0,
        ytorque=0.0,
        ztorque=0.0,
        delta_compute=workflow.SolverParameters['model']['delta_compute'],
        vortratiolim=workflow.SolverParameters['model']['vortratiolim'],
        shearratiolim=workflow.SolverParameters['model']['shearratiolim'],
        pressratiolim=workflow.SolverParameters['model']['pressratiolim'],
        geomdepdom=2,
        delta_cell_max=300,
        var=['psta', 'bl_quantities_2d', 'bl_quantities_3d', 'bl_ue_vector']
    )

    assert solver_output == solver_output_ref

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_add_multiple_2d_extractions_in_SolverOutput_wall():

    workflow = get_workflow_1()

    FamilyNode = cgns.Node(Name='FamilyA', Type='Family')
    cgns.Node(Name='FamilyBC', Type='FamilyBC', Value='BCWall', Parent=FamilyNode)

    # First request of extraction
    Extraction = dict(Type="BC",
                      Fields=['Pressure'],
                      Source="FamilyA",
                      Name="SameName",
                      ExtractionPeriod=1,
                      GridLocation="CellCenter",
                      Frame="absolute")
    solver_elsa.add_2d_extractions_in_SolverOutput(FamilyNode, Extraction, workflow)

    # Second request of extraction with another frame of reference
    Extraction2 = dict(Type="BC",
                      Fields=['Temperature'],
                      Source="FamilyA",
                      Name="SameName",
                      ExtractionPeriod=1,
                      GridLocation="CellCenter",
                      Frame="relative")
    solver_elsa.add_2d_extractions_in_SolverOutput(FamilyNode, Extraction2, workflow)

    # Third request of extraction with the same name that the first one
    Extraction2 = dict(Type="BC",
                      Fields=['BoundaryLayer', 'Pressure'],
                      Source="FamilyA",
                      Name="SameName",
                      ExtractionPeriod=1,
                      GridLocation="CellCenter",
                      Frame="absolute")
    solver_elsa.add_2d_extractions_in_SolverOutput(FamilyNode, Extraction2, workflow)

    solver_output = FamilyNode.getParameters('.Solver#Output',transform_numpy_scalars=True)
    solver_output_ref = dict(
        period=Extraction["ExtractionPeriod"],
        writingmode=2,
        loc = "interface",
        fluxcoeff = 1.0,
        writingframe=Extraction["Frame"], 
        pinf=workflow.Flow['Pressure'],
        torquecoeff=1.0,
        xtorque=0.0,
        ytorque=0.0,
        ztorque=0.0,
        delta_compute=workflow.SolverParameters['model']['delta_compute'],
        vortratiolim=workflow.SolverParameters['model']['vortratiolim'],
        shearratiolim=workflow.SolverParameters['model']['shearratiolim'],
        pressratiolim=workflow.SolverParameters['model']['pressratiolim'],
        geomdepdom=2,
        delta_cell_max=300,
        var=['psta', 'bl_quantities_2d', 'bl_quantities_3d', 'bl_ue_vector']
    )
    assert solver_output == solver_output_ref

    solver_output2 = FamilyNode.getParameters('.Solver#Output#2',transform_numpy_scalars=True)
    solver_output_ref2 = dict(
        period=Extraction["ExtractionPeriod"],
        writingmode=2,
        loc = "interface",
        fluxcoeff = 1.0,
        writingframe="relative", 
        pinf=workflow.Flow['Pressure'],
        torquecoeff=1.0,
        xtorque=0.0,
        ytorque=0.0,
        ztorque=0.0,
        var='tsta'
    )
    assert solver_output2 == solver_output_ref2


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("field_name",['Pressure','Temperature','PressureStagnation'])
def test_add_2d_extractions_in_SolverOutput_inflow(field_name):

    to_elsa = dict(Temperature='tsta', Pressure='psta', PressureStagnation='pgen')

    workflow = get_workflow_1()

    FamilyNode = cgns.Node(Name='FamilyA', Type='Family')
    cgns.Node(Name='FamilyBC', Type='FamilyBC', Value='BCInflow', Parent=FamilyNode)

    Extraction = dict(Type="BC",
                      Fields=[field_name],
                      Source="BCInflow",
                      Name="FamilyA",
                      ExtractionPeriod=1,
                      GridLocation="CellCenter",
                      Frame="absolute")

    
    solver_elsa.add_2d_extractions_in_SolverOutput(FamilyNode, Extraction, workflow)

    solver_output = FamilyNode.getParameters('.Solver#Output', transform_numpy_scalars=True)
    solver_output_ref = dict(
        period=Extraction["ExtractionPeriod"],
        writingmode=2,
        loc = "interface",
        fluxcoeff = 1.0,
        writingframe=Extraction["Frame"],
        var=to_elsa[field_name]
    )

    assert solver_output == solver_output_ref


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_add_integral_extractions_in_wall():

    workflow = get_workflow_1()

    FamilyNode = cgns.Node(Name='FamilyA', Type='Family')
    cgns.Node(Name='FamilyBC', Type='FamilyBC', Value='BCWall', Parent=FamilyNode)

    Extraction = dict(Type="Integral",
                      Fields=['Force', 'Torque'],
                      Source="FamilyA",
                      Name="FamilyA",
                      ExtractionPeriod=1,
                      Frame="absolute")

    
    solver_elsa.add_2d_extractions_in_SolverOutput(FamilyNode, Extraction, workflow)

    solver_output = FamilyNode.getParameters('.Solver#Output',transform_numpy_scalars=True)
    solver_output_ref = dict(
        period=Extraction["ExtractionPeriod"],
        writingmode=2,
        fluxcoeff = 1.0,
        writingframe=Extraction["Frame"], 
        pinf=workflow.Flow['Pressure'],
        loc='interface',
        torquecoeff=1.0,
        xtorque=0.0,
        ytorque=0.0,
        ztorque=0.0,
        var=['flux_rou', 'flux_rov', 'flux_row', 'torque_rou', 'torque_rov', 'torque_row']
    )

    assert solver_output == solver_output_ref


if __name__ == '__main__':
    test_add_2d_extractions_in_SolverOutput_wall()