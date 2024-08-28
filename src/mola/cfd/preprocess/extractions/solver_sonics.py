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

from fnmatch import fnmatch
from treelab import cgns
from mola.logging import mola_logger, MolaException
from mola.cfd.preprocess.extractions.extractions import get_familiesBC_nodes

def apply_to_solver(workflow):

    mola_logger.warning('No custom extractions available with SoNICS for now.')
    add_extractions_for_restart(workflow)
    add_AllZones_family(workflow.tree)
    # process_extractions(workflow)

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
                cgns.Node(Name='FamilyAllZones', Type='FamilyName', Value='AllZones', Parent=zone_node)
    
def add_extractions_for_restart(workflow):
    workflow._interface.add_to_Extractions_Restart(
        # Container='FlowSolution#EndOfRun', 
        Fields=['conservatives'],
        )

# def process_extractions(workflow):
#     import sonics.toolkit.triggers as triggers

#     extractions_merged = []
#     for extraction in workflow.Extractions:
#         family = extraction.get('Family', '*')
#         if family not in extractions_merged:
#             extractions_merged[family] = extraction['Fields']
#         else:
#             extractions_merged[family] += extraction['Fields']

#     trigger = triggers.ExtractTrigger(
#         workflow.SolverParameters['configuration']['conf'], 
#         extractions_merged, 
#         workflow.SolverParameters['configuration']['hpc_conf']['hardware_target']
#         ) 
    
#     workflow._pytriggers += trigger

def add_extractions_for_families(workflow):
    import sonics
    from sonics.toolkit.graph_utils import DataFactory

    # bc_families_to_extract = []
    # bc_families = get_bc_families(workflow)
    # for extraction in workflow.Extractions:
    #     if extraction['Type'] == 'BC' and extraction['Source'] in bc_families:
    #         bc_families_to_extract.append(extraction['Source'])

    bc_families_to_extract = []
    familiesBC = get_familiesBC_nodes(workflow)
    for Extraction in workflow.Extractions:
        if Extraction['Type'] != 'BC': 
            continue 
        requested_source = Extraction['Source']

        for familyBC in familiesBC:

            family = familyBC.parent()
            family_name = family.name()
            bc_type = familyBC.value() 
            
            family_match_requirement = fnmatch(family_name, requested_source) or fnmatch(bc_type, requested_source) 
            if family_match_requirement and 'Fields' in Extraction:
                if family_name not in bc_families_to_extract:
                    bc_families_to_extract.append(family_name) 

    def compute_extracts_from_terms(conf, solver, topology):

        treg = solver.terms
        df = DataFactory(solver, topology)
        elt_location = treg.cell if sonics.spl.guards.cell_center in conf else treg.vertex
        # dual_location = treg.face if sonics.spl.guards.cell_center in conf else treg.edge
        bc_location = treg.face if sonics.spl.guards.cell_center in conf else treg.dual_facet

        extracts = []
        extracts += df.create_zones(treg.conservatives(treg.full), elt_location)
        extracts += df.create_zones(treg.SurfaceNormal, treg.face)
        extracts += df.create_zones(treg.primitives(treg.full), elt_location)
        extracts += df.create_zones(treg.Mach, elt_location)
        extracts += df.create_zones(treg.grad(treg.primitives(treg.full)), elt_location)
        extracts += df.create_zones(treg.grad(treg.Velocity), elt_location)
        extracts += df.create_zones(treg.grad(treg.Temperature), elt_location)

        if (sonics.spl.guards.nslam in conf) or (sonics.spl.guards.nstur in conf):
            extracts += df.create_zones(treg.LaminarViscosity, elt_location)

        if (sonics.spl.guards.nstur in conf):
            extracts += df.create_zones(treg.TurbulentViscosity, elt_location)
            extracts += df.create_zones(treg.TurbulentDistance,  elt_location)

        for family in bc_families_to_extract:
            extracts += df.create_bcs_from_family_name(treg.conservatives(treg.full), bc_location, family)
            extracts += df.create_bcs_from_family_name(treg.primitives(treg.full), bc_location, family)

        # if (sonics.spl.guards.nslam in conf) or (sonics.spl.guards.nstur in conf):
        #     for family in ['HUB', 'SHROUD']:
        #         extracts += df.create_bcs_from_family_name(treg.SkinFriction,     treg.face,   family)
        #         extracts += df.create_bcs_from_family_name(treg.XYZPlusMeshSize,  bc_location, family)
        #         extracts += df.create_bcs_from_family_name(treg.NormalHeatFlux,   treg.face,   family)
        #         extracts += df.create_bcs_from_family_name(treg.LaminarViscosity, bc_location, family)

        return extracts
    
    return compute_extracts_from_terms

def get_bc_families(workflow):

    families = workflow.tree.group(Type='Family', Depth=2)
    familiesBC = []
    for family in families:
        familyBC = family.get(Type='FamilyBC', Depth=1)
        if familyBC:
            familiesBC.append(family.name())

    return familiesBC
