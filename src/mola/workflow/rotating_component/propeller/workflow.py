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

import numpy as np
from treelab import cgns
from packaging.version import Version

from mola.logging import mola_logger, MolaUserError, MolaException
from ..workflow import WorkflowRotatingComponent
from .interface import WorkflowPropellerInterface
from mola.cfd.postprocess import extract_bc
from mola import solver
from mola.pytree.user.checker import is_partitioned_for_use_in_maia, is_distributed_for_use_in_maia
from mola.cfd.postprocess.signals.propeller_coefficients_computer import add_aerodynamic_coefficients_to
from mola.pytree.user import checker 

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class WorkflowPropeller(WorkflowRotatingComponent):

    def __init__(self, **kwargs):
        super().__init__(_skip_interface=True) # used to recover the private attributes of WorkflowRotatingComponent
        self._interface = WorkflowPropellerInterface(self, **kwargs)
        self._blade_radius = None

        # CAVEAT
        # since extract_bc does not verify the Dependency Rule by polymorphism,
        # we need to violate the Rule at this high-level by explicitly choosing
        # the default tool depending on the Solver context. This should have been
        # prevented by automatically using polymorphism. Alternatively, choosing
        # the default tool could have been done in a low-level factory step.
        # TODO review design of extract_bc in order to use DIP and avoid Solver
        # filtering inside the Workflow


        self._extract_bc_default_tool = 'cassiopee'
        # requires being more general https://gitlab.onera.net/numerics/mesh/maia/-/issues/201
        # if self.Solver == 'sonics':
        #     self._extract_bc_default_tool = 'maia' 
        # else: 
        #     self._extract_bc_default_tool = 'cassiopee'

    def assemble(self):
        super().assemble()
        self.force_tagging_all_zones_with_name_propeller()
        self.set_workflow_parameters_in_tree()

    def force_tagging_all_zones_with_name_propeller(self):
        for base in self.tree.bases():
            family_under_base = cgns.Node(Name='Propeller',Type='Family_t')
            base.addChild(family_under_base)

            for zone in base.zones():
                cgns.Node(Name='FamilyName',
                          Type='FamilyName_t',
                          Value='Propeller',
                          Parent=zone,
                          override_sibling_by_name=True)


    def compute_flow_and_turbulence(self):
        self.set_velocity_for_scaling_and_turbulence()
        super().compute_flow_and_turbulence()


    def set_velocity_for_scaling_and_turbulence(self):
        omega_units = self.ApplicationContext["ShaftRotationSpeedUnit"]
        if omega_units == 'rpm':
            omega = (np.pi / 30.0) * self.ApplicationContext["ShaftRotationSpeed"]
        elif omega_units == 'rad/s':
            omega = self.ApplicationContext["ShaftRotationSpeed"]
        else:
            raise MolaUserError(f'got wrong ShaftRotationSpeedUnit "{omega_units}", shall be "rpm" or "rad/s"')
        
        
        r_max = self.blade_radius()
        assert r_max > 0
        r_rel_ref = self.ApplicationContext["TurbulenceSetAtRelativeRadius"]
        axial_velocity = self.Flow['Velocity']
        tangential_velocity = omega * r_rel_ref * r_max

        turb_velocity = np.sqrt( axial_velocity**2 + tangential_velocity**2 )

        self.Flow['VelocityForScalingAndTurbulence'] = turb_velocity

    def blade_radius(self):
        if self._blade_radius is None:
            if self.is_blade_radius_in_application_context():
                return self.set_blade_radius_from_application_context()

            elif self.tree is None:
                raise MolaUserError('did not find a tree, hence cannot retrieve blade radius. Maybe you forgot to process_mesh?')
            
            else:
                return self._compute_maximum_blade_radius()
        
        else:
            return self._blade_radius

    def is_blade_radius_in_application_context(self):
        return '_BladeRadius' in self.ApplicationContext

    def set_blade_radius_from_application_context(self):
        blade_radius = self.ApplicationContext['_BladeRadius']
        self._blade_radius = blade_radius
        return blade_radius
         


    def _compute_maximum_blade_radius(self, imposed_tool : str = None):
        if imposed_tool:
            tool = imposed_tool
        else: 
            tool = self._extract_bc_default_tool

        mola_logger.info(f"    📐 measuring propeller blade radius with {tool}...", rank=0)

        if not self.tree.isStructured():
            if tool == 'cassiopee':
                raise MolaUserError('cannot compute blade radius on unstructured grids using Cassiopee:\n'
                                    'see https://gitlab.onera.net/numerics/mola/-/work_items/308#note_82633\n'
                                    'HINT: provide _BladeRadius into ApplicationContext')
            elif tool.startswith('maia'):
                from maia import __version__ as current_maia_version
                if Version(current_maia_version) < Version('1.8'):
                    raise MolaUserError(
                        f'cannot extract propeller blade radius using maia v{current_maia_version}.\n'
                        'It requires maia >=1.8 (https://gitlab.onera.net/numerics/mesh/maia/-/issues/201).\n'
                        'HINT: provide _BladeRadius into ApplicationContext')


        # if not, go ahead with default tool
        # if is NGON: forbid Cassiopee and if maia<=1.8 then raise user exception requiring ApplicationContext=_BladeRadius

        tree = self.tree
        if tool == 'maia':
            tree = self._get_partitioned_tree_for_use_in_maia()

        blade_family_name = self.get_blade_family_names(must_be_unique=True)[0]
        blade_surface = extract_bc(tree, blade_family_name, tool=tool)
        self._blade_radius = self._compute_maximum_distance_to_axis_from(blade_surface)
        self.ApplicationContext['_BladeRadius'] = self._blade_radius
        return self._blade_radius
    
    def _get_partitioned_tree_for_use_in_maia(self):
        tree = self.tree.copy()
        if not is_partitioned_for_use_in_maia(tree):
            from mpi4py import MPI
            import maia
            
            if not is_distributed_for_use_in_maia(tree):
                tree = maia.factory.full_to_dist_tree(tree, MPI.COMM_WORLD)

            tree = maia.factory.partition_dist_tree(tree, MPI.COMM_WORLD, preserve_orientation=True)
            tree = cgns.castNode(tree)
        
        return tree

    def compute_propeller_coefficients(self, extraction : dict, **operation):

        if extraction["Type"] != "Integral" or not extraction.get("Data"):
            return

        diameter = 2*self.blade_radius()
        add_aerodynamic_coefficients_to(extraction, self.ApplicationContext,
                                        diameter,
                                        self.Flow['Density'],
                                        self.Flow['Velocity'])
