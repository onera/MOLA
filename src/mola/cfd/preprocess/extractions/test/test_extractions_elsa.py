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

import copy
import numpy as np
from treelab import cgns
from mola.cfd.preprocess.extractions import solver_elsa

class FakeWorkflow():

    def __init__(self):

        self.Flow = dict(
            Conservatives = dict(Density=1.2,Momentum=10.,Energy=5.)
        )

        self.Turbulence = dict(TransitionMode=None)

        self.Extractions = [
            dict(type='fake'),
            dict(type='bc', BCType='BCWallViscous', fields=['Pressure']),
            dict(type='3D', fields=['Density', 'Momentum', 'Energy']),
        ]

        self.tree = cgns.Tree()
        base1 = cgns.Base(Name='Base1', Parent=self.tree)
        base2 = cgns.Base(Name='Base2', Parent=self.tree)
        zone = cgns.Zone(Parent=base1)

        self.overset_flag = False
    
    def has_overset_component(self):
        return self.overset_flag




def test_add_extractions_for_overset_components():
    workflow = FakeWorkflow()

    # Check that nothing happend if not overset components
    ref_Extractions = copy.copy(workflow.Extractions)
    solver_elsa.add_extractions_for_overset_components(workflow)
    assert workflow.Extractions == ref_Extractions

    # Check behavior with overset components
    workflow = FakeWorkflow()
    workflow.overset_flag = True
    ref_Extractions.append(
        dict(
            type      = '3D', 
            fields    = dict(Density=1.2,Momentum=10.,Energy=5.), 
            Container = 'FlowSolution#EndOfRun#Relative', 
            Frame     = 'relative'
        )
    )
    solver_elsa.add_extractions_for_overset_components(workflow)
    assert workflow.Extractions == ref_Extractions


def test_add_trigger():
    ref_trigger = ['ELSA_TRIGGER', None, [
        ['.Solver#Trigger', None, [
            ['next_state', np.array([16], dtype=np.int32), [], 'DataArray_t'], 
            ['next_iteration', np.array([1], dtype=np.int32), [], 'DataArray_t'], 
            ['file', np.array([b'c', b'o', b'p', b'r', b'o', b'c', b'e', b's', b's', b'.', b'p', b'y'], dtype='|S1'), [], 'DataArray_t']
        ], 'UserDefinedData_t']], 'Family_t']

    workflow = FakeWorkflow()
    solver_elsa.add_trigger(workflow.tree)

    for zone in workflow.tree.zones():
        assert zone.get(Name='ELSA_TRIGGER', Type='AdditionalFamilyName', Value='ELSA_TRIGGER')

    for base in workflow.tree.bases():
        trigger_fam_node = base.get(Name='ELSA_TRIGGER', Type='Family')
        assert str(trigger_fam_node) == str(ref_trigger)
    

def test_global_convergence_history():
    ref_node = ['GlobalConvergenceHistory', np.array([0], dtype=np.int32), [
        ['NormDefinitions', np.array([b'C', b'o', b'n', b'v', b'e', b'r', b'g', b'e', b'n', b'c', b'e', b'H', b'i', b's', b't', b'o', b'r', b'y'], dtype='|S1'), [], 'Descriptor_t'], 
        ['.Solver#Output', None, [
            ['period', np.array([1], dtype=np.int32), [], 'DataArray_t'], 
            ['writingmode', np.array([0], dtype=np.int32), [], 'DataArray_t'], 
            ['var', np.array([b'r', b'e', b's', b'i', b'd', b'u', b'a', b'l', b'_', b'c', b'o', b'n', b's', b' ', b'r', b'e', b's', b'i', b'd', b'u', b'a', b'l', b'_', b't', b'u', b'r', b'b'], dtype='|S1'), [], 'DataArray_t']
        ], 'UserDefinedData_t']
        ], 'UserDefinedData_t']
    
    workflow = FakeWorkflow()
    solver_elsa.add_global_convergence_history(workflow)
    for base in workflow.tree.bases():
        conv_node = base.get(Name='GlobalConvergenceHistory')   
        assert str(conv_node) == str(ref_node)  


def test_process_extractions_3d():
    # assert False, 'Not implemented yet'
    workflow = FakeWorkflow()
    workflow.Extractions = [dict(type='3D', fields=['Density', 'Momentum', 'Energy'])]
    solver_elsa.process_extractions_3d(workflow)

    zone = workflow.tree.zones()[0]
    FS = zone.get(Name='FlowSolution#EndOfRun', Type='FlowSolution')
    assert FS

    FS_ref = ['FlowSolution#EndOfRun', None, [
                ['Density', None, [], 'DataArray_t'], 
                ['Momentum', None, [], 'DataArray_t'], 
                ['Energy', None, [], 'DataArray_t'], 
                ['GridLocation', np.array([b'C', b'e', b'l', b'l', b'C', b'e', b'n', b't', b'e', b'r'], dtype='|S1'), [], 'GridLocation_t'], 
                ['.Solver#Output', None, [
                    ['period', np.array([1], dtype=np.int32), [], 'DataArray_t'], 
                    ['writingmode', np.array([2], dtype=np.int32), [], 'DataArray_t'], 
                    ['writingframe', np.array([b'r', b'e', b'l', b'a', b't', b'i', b'v', b'e'], dtype='|S1'), [], 'DataArray_t']
                ], 'UserDefinedData_t']], 'FlowSolution_t']

    assert str(FS) == str(FS_ref)

def test_process_extractions_3d_additional_variables():
    # assert False, 'Not implemented yet'
    workflow = FakeWorkflow()
    workflow.Extractions = [
        dict(type='3D', fields=['Density', 'Momentum', 'Energy']),
        dict(type='3D', fields=['Mach', 'Pressure']),
        ]
    solver_elsa.process_extractions_3d(workflow)

    zone = workflow.tree.zones()[0]
    FS = zone.get(Name='FlowSolution#EndOfRun', Type='FlowSolution')
    assert FS

    FS_ref = ['FlowSolution#EndOfRun', None, [
                ['Density', None, [], 'DataArray_t'], 
                ['Momentum', None, [], 'DataArray_t'], 
                ['Energy', None, [], 'DataArray_t'], 
                ['GridLocation', np.array([b'C', b'e', b'l', b'l', b'C', b'e', b'n', b't', b'e', b'r'], dtype='|S1'), [], 'GridLocation_t'], 
                ['.Solver#Output', None, [
                    ['period', np.array([1], dtype=np.int32), [], 'DataArray_t'], 
                    ['writingmode', np.array([2], dtype=np.int32), [], 'DataArray_t'], 
                    ['writingframe', np.array([b'r', b'e', b'l', b'a', b't', b'i', b'v', b'e'], dtype='|S1'), [], 'DataArray_t']
                ], 'UserDefinedData_t'],
                ['Mach', None, [], 'DataArray_t'], 
                ['Pressure', None, [], 'DataArray_t'], 
            ], 'FlowSolution_t']
    
    assert str(FS) == str(FS_ref)

def test_process_extractions_3d_coords():
    # assert False, 'Not implemented yet'
    workflow = FakeWorkflow()
    workflow.Extractions = [dict(type='3D', Container='FlowSolution#EndOfRun#Coords', fields=['CoordinateX', 'CoordinateY', 'CoordinateZ'], GridLocation='Vertex', Frame='absolute')]
    solver_elsa.process_extractions_3d(workflow)

    zone = workflow.tree.zones()[0]
    FS = zone.get(Name='FlowSolution#EndOfRun#Coords', Type='FlowSolution')
    assert FS

    FS_ref = ['FlowSolution#EndOfRun#Coords', None, [
                ['CoordinateX', None, [], 'DataArray_t'], 
                ['CoordinateY', None, [], 'DataArray_t'], 
                ['CoordinateZ', None, [], 'DataArray_t'], 
                ['GridLocation', np.array([b'V', b'e', b'r', b't', b'e', b'x'], dtype='|S1'), [], 'GridLocation_t'], 
                ['.Solver#Output', None, [
                    ['period', np.array([1], dtype=np.int32), [], 'DataArray_t'], 
                    ['writingmode', np.array([2], dtype=np.int32), [], 'DataArray_t'], 
                    ['writingframe', np.array([b'a', b'b', b's', b'o', b'l', b'u', b't', b'e'], dtype='|S1'), [], 'DataArray_t']
                ], 'UserDefinedData_t']], 'FlowSolution_t']

    assert str(FS) == str(FS_ref)

def test_process_extractions_3d_average():
    # assert False, 'Not implemented yet'
    workflow = FakeWorkflow()
    workflow.Extractions = [dict(type='3D', Container='FlowSolution#Average', fields=['Density', 'Momentum'], options=dict(average='time', period_init='inactive'))]
    solver_elsa.process_extractions_3d(workflow)

    zone = workflow.tree.zones()[0]
    FS = zone.get(Name='FlowSolution#Average', Type='FlowSolution')
    assert FS

    FS_ref = ['FlowSolution#Average', None, [
                ['Density', None, [], 'DataArray_t'], 
                ['Momentum', None, [], 'DataArray_t'], 
                ['GridLocation', np.array([b'C', b'e', b'l', b'l', b'C', b'e', b'n', b't', b'e', b'r'], dtype='|S1'), [], 'GridLocation_t'], 
                ['.Solver#Output', None, [
                    ['period', np.array([1], dtype=np.int32), [], 'DataArray_t'], 
                    ['writingmode', np.array([2], dtype=np.int32), [], 'DataArray_t'], 
                    ['writingframe', np.array([b'r', b'e', b'l', b'a', b't', b'i', b'v', b'e'], dtype='|S1'), [], 'DataArray_t'],
                    ['average', np.array([b't', b'i', b'm', b'e'], dtype='|S1'), [], 'DataArray_t'],
                    ['period_init', np.array([b'i', b'n', b'a', b'c', b't', b'i', b'v', b'e'], dtype='|S1'), [], 'DataArray_t'],
                ], 'UserDefinedData_t']], 'FlowSolution_t']

    assert str(FS) == str(FS_ref)



def test_adapt_variables_for_2d_extraction_wall():

    workflow = FakeWorkflow()
    for zone in workflow.tree.zones():
        cgns.Node(Name='ZoneType', Type='ZoneType', Value='Structured', Parent=zone)

    Extraction = dict(type='bc', BCType='BCWall', fields=['Pressure', 'BoundaryLayer', 'yPlus', 
                                                          'geomdepdom','delta_cell_max','delta_compute',
                                                          'vortratiolim','shearratiolim','pressratiolim'])  # BCType is not used by adapt_variables_for_2d_extraction
    ExtractBCType = 'BCWall'
    ExtractVariablesList = solver_elsa.adapt_variables_for_2d_extraction(workflow, Extraction, ExtractBCType)

    assert ExtractVariablesList == ['Pressure', 'BoundaryLayer', 'yPlus', 
                                    'geomdepdom','delta_cell_max','delta_compute',
                                    'vortratiolim','shearratiolim','pressratiolim']

def test_adapt_variables_for_2d_extraction_BCWallInviscid():

    workflow = FakeWorkflow()
    for zone in workflow.tree.zones():
        cgns.Node(Name='ZoneType', Type='ZoneType', Value='Structured', Parent=zone)

    Extraction = dict(type='bc', BCType='BCWall', fields=['Pressure', 'BoundaryLayer', 'yPlus', 
                                                          'geomdepdom','delta_cell_max','delta_compute',
                                                          'vortratiolim','shearratiolim','pressratiolim'])  # BCType is not used by adapt_variables_for_2d_extraction
    ExtractBCType = 'BCWallInviscid'
    ExtractVariablesList = solver_elsa.adapt_variables_for_2d_extraction(workflow, Extraction, ExtractBCType)

    assert ExtractVariablesList == ['Pressure']

def test_adapt_variables_for_2d_extraction_unstructured():

    workflow = FakeWorkflow()
    for zone in workflow.tree.zones():
        ZoneType = cgns.Node(Name='ZoneType', Type='ZoneType', Value='Unstructured') 
        zone.addChild(ZoneType)

    Extraction = dict(type='bc', BCType='BCWall', fields=['Pressure', 'BoundaryLayer', 'yPlus', 
                                                          'geomdepdom','delta_cell_max','delta_compute',
                                                          'vortratiolim','shearratiolim','pressratiolim'])  # BCType is not used by adapt_variables_for_2d_extraction
    ExtractBCType = 'BCWall'
    ExtractVariablesList = solver_elsa.adapt_variables_for_2d_extraction(workflow, Extraction, ExtractBCType)

    assert ExtractVariablesList == ['Pressure', 'yPlus', 
                                    'geomdepdom','delta_cell_max','delta_compute',
                                    'vortratiolim','shearratiolim','pressratiolim']
    
def test_adapt_variables_for_2d_extraction_TransitionMode_NonLocalCriteria_LSTT():

    workflow = FakeWorkflow()
    for zone in workflow.tree.zones():
        cgns.Node(Name='ZoneType', Type='ZoneType', Value='Structured', Parent=zone)
    workflow.Turbulence = dict(TransitionMode='NonLocalCriteria-LSTT')

    Extraction = dict(type='bc', BCType='BCWall', fields=['Pressure'])  # BCType is not used by adapt_variables_for_2d_extraction
    ExtractBCType = 'BCWall'
    ExtractVariablesList = solver_elsa.adapt_variables_for_2d_extraction(workflow, Extraction, ExtractBCType)
    
    assert ExtractVariablesList == ['Pressure', 'intermittency', 'clim', 'how', 
                                    'origin','lambda2', 'turb_level', 'n_tot_ag', 
                                    'n_crit_ag', 'r_tcrit_ahd', 'r_theta_t1', 
                                    'line_status', 'crit_indicator']

def test_adapt_variables_for_2d_extraction_TransitionMode_imposed():

    workflow = FakeWorkflow()
    for zone in workflow.tree.zones():
        cgns.Node(Name='ZoneType', Type='ZoneType', Value='Structured', Parent=zone)
    workflow.Turbulence = dict(TransitionMode='Imposed')

    Extraction = dict(type='bc', BCType='BCWall', fields=['Pressure'])  # BCType is not used by adapt_variables_for_2d_extraction
    ExtractBCType = 'BCWall'
    ExtractVariablesList = solver_elsa.adapt_variables_for_2d_extraction(workflow, Extraction, ExtractBCType)
    
    assert ExtractVariablesList == ['Pressure', 'intermittency', 'clim']


def get_test_parameters_1():
    SolverParameters = dict(
        model = dict(
            delta_compute   = 'first_order_bl',
            vortratiolim    = 1e-3,
            shearratiolim   = 2e-2,
            pressratiolim   = 1e-3,
        )
    )
    pinf = 1e5
    return solver_elsa.get_default_parameters_for_2d_extractions(SolverParameters, pinf)
    
def test_add_2d_extractions_in_SolverOutput_1():
    default_bc_parameters, default_bc_wall_parameters = get_test_parameters_1()
    
    ExtractBCType = 'BCWall'
    FamilyNode = cgns.Node(Name='Family', Type='Family')
    cgns.Node(Name='FamilyBC', Type='FamilyBC', Value=ExtractBCType, Parent=FamilyNode)

    ExtractVariablesList = ['Pressure', 'BoundaryLayer']

    solver_elsa.add_2d_extractions_in_SolverOutput(FamilyNode, ExtractBCType, ExtractVariablesList, default_bc_parameters, default_bc_wall_parameters)

    solver_output = FamilyNode.getParameters('.Solver#Output')
    solver_output_ref = dict(**default_bc_wall_parameters)
    solver_output_ref['var'] = ['psta', 'bl_quantities_2d', 'bl_quantities_3d', 'bl_ue']

    assert solver_output == solver_output_ref

def test_add_2d_extractions_in_SolverOutput_2():
    default_bc_parameters, default_bc_wall_parameters = get_test_parameters_1()

    ExtractBCType = 'BCInflow'
    FamilyNode = cgns.Node(Name='Family', Type='Family')
    cgns.Node(Name='FamilyBC', Type='FamilyBC', Value=ExtractBCType, Parent=FamilyNode)

    ExtractVariablesList = ['Pressure']

    solver_elsa.add_2d_extractions_in_SolverOutput(FamilyNode, ExtractBCType, ExtractVariablesList, default_bc_parameters, default_bc_wall_parameters)

    solver_output = FamilyNode.getParameters('.Solver#Output')
    solver_output_ref = dict(**default_bc_parameters)
    solver_output_ref['var'] = 'psta'

    assert solver_output == solver_output_ref

def test_add_2d_extractions_in_SolverOutput_3():
    default_bc_parameters, default_bc_wall_parameters = get_test_parameters_1()

    ExtractBCType = 'BCInflow'
    FamilyNode = cgns.Node(Name='Family', Type='Family')
    cgns.Node(Name='FamilyBC', Type='FamilyBC', Value=ExtractBCType, Parent=FamilyNode)
    solver_output = {'period': 1, 'writingmode': 2, 'loc': 'interface', 'fluxcoeff': 1.0, 'writingframe': 'absolute', 'geomdepdom': 2, 'delta_cell_max': 300, 'var': 'psta'}
    FamilyNode.setParameters('.Solver#Output', **solver_output)

    solver_output_ref = copy.deepcopy(FamilyNode.getParameters('.Solver#Output'))
    solver_output_ref['var'] = ['psta', 'tsta']

    ExtractVariablesList = ['Temperature']

    solver_elsa.add_2d_extractions_in_SolverOutput(FamilyNode, ExtractBCType, ExtractVariablesList, default_bc_parameters, default_bc_wall_parameters)
    solver_output = FamilyNode.getParameters('.Solver#Output')
    assert solver_output == solver_output_ref

def test_add_2d_extractions_in_SolverOutput_4():
    default_bc_parameters, default_bc_wall_parameters = get_test_parameters_1()
    
    ExtractBCType = 'BCInflow'
    FamilyNode = cgns.Node(Name='Family', Type='Family')
    cgns.Node(Name='FamilyBC', Type='FamilyBC', Value=ExtractBCType, Parent=FamilyNode)
    solver_output = {'period': 1, 'writingmode': 2, 'loc': 'interface', 'fluxcoeff': 1.0, 'writingframe': 'absolute', 'geomdepdom': 2, 'delta_cell_max': 300, 'var': ['psta', 'pgen']}
    FamilyNode.setParameters('.Solver#Output', **solver_output)

    solver_output_ref = copy.deepcopy(FamilyNode.getParameters('.Solver#Output'))
    solver_output_ref['var'] = ['psta', 'pgen', 'tsta']

    ExtractVariablesList = ['Temperature']

    solver_elsa.add_2d_extractions_in_SolverOutput(FamilyNode, ExtractBCType, ExtractVariablesList, default_bc_parameters, default_bc_wall_parameters)
    solver_output = FamilyNode.getParameters('.Solver#Output')
    assert solver_output == solver_output_ref
    