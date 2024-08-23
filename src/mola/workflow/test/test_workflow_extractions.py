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

from treelab import cgns
from mola import naming_conventions as names
from mola.workflow.test.test_workflow import get_workflow_cart_monoproc


@pytest.mark.integration
@pytest.mark.cost_level_2
def test_integrals(tmp_path):
    
    def assert_file_with_relevant_zone_and_fields(filename, zonename, fieldnames, expected_number_of_items):
        
        expected_file = os.path.join(tmp_path, names.DIRECTORY_OUTPUT, filename)
        
        assert os.path.isfile(expected_file)

        tree = cgns.load(expected_file)

        assert tree

        base = tree.get(Name="Integral", Type="CGNSBase_t", Depth=1)
        assert base
        
        zone = base.get(Name=zonename, Type="Zone_t", Depth=1)
        assert zone

        container = zone.get(Name='FlowSolution', Type='FlowSolution_t', Depth=1)
        assert container

        iterations = container.get(Name='IterationNumber', Type='DataArray_t', Depth=1)
        assert iterations

        if not isinstance(fieldnames,list):
            if not isinstance(fieldnames,str): raise AttributeError("wrong fieldnames attribute")
            fieldnames = [fieldnames]

        for fieldname in fieldnames:
            field_node = container.get(Name=fieldname, Type='DataArray_t', Depth=1)
            assert field_node 

            assert len(field_node.value()) == expected_number_of_items

    separated_filename = 'test_integrals.cgns'


    w = get_workflow_cart_monoproc(tmp_path)
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
    
    niter = 10
    w.Numerics['NumberOfIterations'] = niter
    w.RunManagement['Scheduler'] = 'local'
    w.prepare()

    
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.simulation_status()
    
    expected_number_of_items = niter + 1

    assert_file_with_relevant_zone_and_fields(separated_filename, "TestSeparatedFile",
        ['ForceX','ForceY','ForceZ','TorqueX','TorqueY','TorqueZ'], expected_number_of_items)
    
    assert_file_with_relevant_zone_and_fields(names.FILE_OUTPUT_1D, "TestIntoSignals",
        ['ForceX','ForceY','ForceZ','TorqueX','TorqueY','TorqueZ'], expected_number_of_items)
    
    assert_file_with_relevant_zone_and_fields(names.FILE_OUTPUT_1D, "TestIntoSignals2",
        "MassFlow", expected_number_of_items)

    

if __name__ == '__main__':
    test_integrals('extract_integral_'+os.environ.get("MOLA_SOLVER"))
