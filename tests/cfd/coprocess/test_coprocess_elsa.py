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

from treelab import cgns
from mola.cfd.coprocess import solver_elsa

def get_tree():

    import Converter.PyTree as C
    import Converter.Internal as I
    import Generator.PyTree as G
    import Initiator.PyTree as Init

    I.__FlowSolutionCenters__ = 'FlowSolution#Output'

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

    t = cgns.castNode(t)
    I.__FlowSolutionCenters__ = 'FlowSolution#Centers'

    return t


def get_fake_workflow_with_coprocess_manager(RunDirectory):

    t = get_tree()

    class FakeWorkflow():
        def __init__(self):                          
            self._status = 'BEFORE_FIRST_ITERATION'
            self.Numerics = dict(IterationAtInitialState=1,
                                NumberOfIterations=5,
                                TimeAtInitialState=0.0,
                                TimeMarching='Steady')
            self.Extractions = []
            self.RunManagement = dict(RunDirectory=RunDirectory)
  
            from mola.cfd.coprocess.manager import CoprocessManager
            self._coprocess_manager = CoprocessManager(self)

    workflow = FakeWorkflow()
 
    workflow.tree = cgns.castNode(t)

    return workflow



@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("GridLocation", ["Vertex", "CellCenter"])
def test_extract_fields(tmp_path, GridLocation):


    workflow = get_fake_workflow_with_coprocess_manager(tmp_path)
    
    workflow._coprocess_manager.Extractions = [
        dict(Fields=['Density','MomentumX','MomentumY','MomentumZ'],
             GridLocation=GridLocation,
             Container='FlowSolution#Output',
             Type='3D')]
    workflow.Extractions = workflow._coprocess_manager.Extractions
    
    output_tree = workflow.tree
    
    for extraction in workflow._coprocess_manager.Extractions:
        tRef = solver_elsa.extract_fields(output_tree, extraction)

        output_FS = tRef.get(Name='FlowSolution#Output', Type='FlowSolution')
        try:
            existing_field_names = [n.name() for n in output_FS.group(Type='DataArray', Depth=1)]
        except:
            existing_field_names = []

        expected_field_names = ['Density','MomentumX','MomentumY','MomentumZ']
        assert set(existing_field_names) == set(expected_field_names)

    workflow._coprocess_manager._status = 'COMPLETED'


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_extract_isosurface(tmp_path):

    workflow = get_fake_workflow_with_coprocess_manager(tmp_path)
    
    workflow._coprocess_manager.Extractions = [
        dict(Fields=['MomentumX','MomentumY'],
             IsoSurfaceField='CoordinateX',
             Name='MySlice',
             IsoSurfaceContainer='auto',
             ContainersToTransfer='all',
             IsoSurfaceValue=0.1,
             Type='IsoSurface')]
    workflow.Extractions = workflow._coprocess_manager.Extractions
    
    output_tree = workflow.tree
    
    for extraction in workflow._coprocess_manager.Extractions:
        extraction['Data'] = solver_elsa.extract_isosurface(output_tree, extraction)
        solver_elsa.remove_not_needed_fields(extraction)
        tRef = extraction['Data']

        existing_field_names = []
        for FS in tRef.group(Type='FlowSolution'):
            for node in FS.group(Type='DataArray', Depth=1):
                existing_field_names.append(node.name())
        existing_field_names = set(existing_field_names)

        expected_field_names = set(['MomentumX','MomentumY'])
        assert existing_field_names == expected_field_names

    workflow._coprocess_manager._status = 'COMPLETED'


# @pytest.mark.unit
# @pytest.mark.cost_level_0
# def test_extract_bc(tmp_path):

#     workflow = get_fake_workflow_with_coprocess_manager(tmp_path)
    
#     workflow._coprocess_manager.Extractions = [
#         dict(Type='BC', Source='WALL', Fields=['Pressure', 'Temperature']),
#         dict(Type='BC', Source='FARFIELD', Fields=['Pressure', 'MomentumX'])]

#     workflow.Extractions = workflow._coprocess_manager.Extractions
    
#     output_tree = workflow.tree
#     families_to_bctype = solver_elsa.get_family_to_BCType(output_tree)

#     for extraction in workflow._coprocess_manager.Extractions:
#         tRef = solver_elsa.extract_bc(output_tree, extraction, families_to_bctype)

#     workflow._coprocess_manager._status = 'COMPLETED'



# @pytest.mark.unit
# @pytest.mark.cost_level_0
# @pytest.mark.parametrize("modeling", ['euler', 'rans'])
# def test_extract_residuals(tmp_path,modeling):

#     workflow = get_fake_workflow_with_coprocess_manager(tmp_path, modeling, create_convergence_nodes=True)
#     output_tree = workflow.tree

#     workflow._interface.add_to_Extractions_Residuals()
#     extraction = workflow.Extractions[-1]
    
#     solver_elsa.extract_residuals(output_tree, extraction)

#     workflow._coprocess_manager._status = 'COMPLETED'


# @pytest.mark.unit
# @pytest.mark.cost_level_0
# def test_extract_integral(tmp_path):
    
#     workflow = get_fake_workflow_with_coprocess_manager(tmp_path, 'laminar')
    
#     extraction = dict(Type='Integral', Source='WALL', Name='WALL_LOADS',
#                       Fields=['Force','Torque','MassFlow'])
#     workflow._coprocess_manager.Extractions = [ extraction ]
#     workflow.Extractions = workflow._coprocess_manager.Extractions

#     import FastS.PyTree as FastS

#     niter = 3
#     for it in range( niter ):
#         FastS._compute(workflow.tree, workflow._fast_metrics, it)
    
#         workflow._coprocess_manager.iteration = it
#         workflow.tree = cgns.castNode(workflow.tree)

#         output_tree = solver_fast.get_output_tree(workflow, workflow._coprocess_manager)
        
#         solver_fast.extract_integral(output_tree, extraction, workflow)

#     flow_sol = extraction['Data'].get('FlowSolution')
    
#     assert flow_sol

#     expected_integrals = ('Iteration','ForceX',  'ForceY',   'ForceZ',
#                           'MassFlow',     'TorqueX','TorqueY', 'TorqueZ')
#     for k in expected_integrals: 
#         expected_node = flow_sol.get(k, Type='DataArray_t')
#         assert expected_node
#         assert len(expected_node.value()) == niter

#     workflow._coprocess_manager._status = 'COMPLETED'

if __name__ == '__main__':
    # t = get_tree()
    # t.save('test_tree.cgns')
    test_extract_fields('.', 'CellCenter')
