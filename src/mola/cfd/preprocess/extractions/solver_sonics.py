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

from treelab import cgns
from mola.logging import mola_logger, MolaUserError
from mola.cfd.preprocess.extractions.extractions import get_familiesBC_nodes, get_bc_families_names_to_extract
from mola.cfd.preprocess.solver_specific_tools.solver_sonics import translate_extraction_variables_to_sonics

def apply_to_solver(workflow):

    add_extractions_for_restart(workflow)
    add_AllZones_family(workflow.tree)
    adapt_extractions(workflow.Extractions)
    add_parameters_for_torque_extraction(workflow)

def adapt_extractions(Extractions):
    for ext in Extractions:
        if ext['Type'] in  ['BC', 'Residuals']:
            ext['ExtractionPeriod'] = 1000000000 # Only done at the end of the simulation
            ext['SavePeriod'] = 1000000000 # Only done at the end of the simulation
            ext['ExtractAtEndOfRun'] = True
        # elif ext['Type'] == '3D':
        #     mola_logger.user_warning('output container for extraction 3D is changed to Fields@Vertex@End')
        #     ext['Container'] = 'Fields@Vertex@End'
            
def add_AllZones_family(tree):
    # HACK The current implementation of residual extraction requires to tag zones we want to 
    # integrate into the global residual computation into the "AllZones" family
    for base_node in tree.bases():
        family_all_zones_node = base_node.get(Name='AllZones', Type='Family', Depth=1)
        if not family_all_zones_node:
            family_all_zones_node = cgns.Node(Name='AllZones', Type='Family', Parent=base_node)

        for zone_node in base_node.zones():
            family_all_zones_node = zone_node.get(Value='AllZones', Type='FamilyName', Depth=1)
            if not family_all_zones_node:
                cgns.Node(Name='FamilyAllZones', Type='AdditionalFamilyName', Value='AllZones', Parent=zone_node)
    
def add_extractions_for_restart(workflow):
    workflow._interface.add_to_Extractions_Restart(
        # Container='FlowSolution#EndOfRun', 
        Fields=['Conservatives'],
        )

def add_fields_and_bc_extractions(workflow):
    import sonics
    from sonics.toolkit.graph_utils import DataFactory

    familiesBC = get_familiesBC_nodes(workflow.tree)

    def compute_extracts_from_terms(conf, solver, topology):

        treg = solver.terms
        df = DataFactory(solver, topology)
        elt_location = treg.cell if sonics.spl.guards.cell_center in conf else treg.vertex
        # dual_location = treg.face if sonics.spl.guards.cell_center in conf else treg.edge

        if sonics.spl.guards.space_dim_3d not in conf:
            # 2D case
            elt_location = treg.face if sonics.spl.guards.cell_center in conf else treg.vertex
            bc_location  = treg.edge if sonics.spl.guards.cell_center in conf else treg.dual_facet
        else:
            # 3D case
            elt_location = treg.cell if sonics.spl.guards.cell_center in conf else treg.vertex
            bc_location  = treg.face if sonics.spl.guards.cell_center in conf else treg.dual_facet

        extracts = []

        # HACK Add "dummy" variables to prevent deadlock
        extracts = df.create_zones(treg.dummy(treg.conservatives(treg.full)), elt_location)
        # extracts += df.create_zones(treg.dummy(treg.primitives(treg.mean_flow)), elt_location)
        extracts += df.create_zones(treg.dummy(treg.grad(treg.primitives(treg.mean_flow))), elt_location)
        # extracts += df.create_zones(treg.dummy(treg.interp_on(treg.grad(treg.Velocity), treg.cell)), elt_location)

        # Real extractions
        extracts += df.create_zones(treg.conservatives(treg.full), elt_location)
        # extracts += df.create_zones(treg.SurfaceNormal, treg.face)
        # extracts += df.create_zones(treg.primitives(treg.full), elt_location)
        # extracts += df.create_zones(treg.Mach, elt_location)
        # extracts += df.create_zones(treg.grad(treg.Velocity), elt_location)

        if (sonics.spl.guards.nslam in conf) or (sonics.spl.guards.nstur in conf):
            extracts += df.create_zones(treg.LaminarViscosity, elt_location)

        if (sonics.spl.guards.nstur in conf):
            extracts += df.create_zones(treg.TurbulentViscosity, elt_location)
            extracts += df.create_zones(treg.TurbulentDistance,  elt_location)

        for extraction in workflow.Extractions:
            if extraction['Type'] != 'BC' or len(extraction['Fields'])==0:
                continue

            if extraction['GridLocation'] == 'CellCenter':
                if not sonics.spl.guards.cell_center:
                    raise MolaUserError('Cannot extract a BC at "Vertex" because SoNICS will run at CellCenter')
            else:
                if sonics.spl.guards.cell_center:
                    raise MolaUserError('Cannot extract a BC at "CellCenter" because SoNICS will run at Vertex')

            families = get_bc_families_names_to_extract(workflow, extraction, familiesBC)
            fields = translate_extraction_variables_to_sonics(extraction['Fields'], solver)
            for family in families:                
                for field in fields:
                    extracts += df.create_bcs_from_family_name(field, bc_location, family)

        return extracts
    
    return compute_extracts_from_terms

def add_parameters_for_torque_extraction(workflow):
    params_for_torque_extraction = dict(
        ReferencePressure = workflow.Flow['Pressure'],
        TorqueOriginX = 0.0,
        TorqueOriginY = 0.0,
        TorqueOriginZ = 0.0,
    )
    for base in workflow.tree.bases():
        ReferenceState = base.get(Type='ReferenceState', Depth=1)
        for name, value in params_for_torque_extraction.items():
            cgns.Node(Name=name, Value=value, Type='DataArray', Parent=ReferenceState)
