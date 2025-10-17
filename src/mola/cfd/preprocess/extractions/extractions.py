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
from fnmatch import fnmatch
from pprint import pformat as pretty
from mola.cfd import apply_to_solver
from mola.cfd.postprocess.signals import AVAILABLE_OPERATIONS_ON_SIGNALS
from mola.logging import mola_logger
from mola.logging.exceptions import MolaUserError, MolaException
from mola.naming_conventions import FILE_INPUT_SOLVER

def apply(workflow):

    replace_shortcuts(workflow)
    add_residuals_extraction(workflow)
    add_memory_usage_extraction(workflow)
    add_time_monitoring_extraction(workflow)
    split_bc_and_integral_extractions_by_family(workflow)
    update_extractions_from_convergence_criteria(workflow)
    apply_to_solver(workflow)
    print_extractions(workflow.Extractions)

def add_residuals_extraction(workflow):
    if not any([ext['Type'] == 'Residuals' for ext in workflow.Extractions]):
        workflow._interface.add_to_Extractions_Residuals()

def add_memory_usage_extraction(workflow):
    if not any([ext['Type'] == 'MemoryUsage' for ext in workflow.Extractions]):
        workflow._interface.add_to_Extractions_MemoryUsage()

def add_time_monitoring_extraction(workflow):
    if not any([ext['Type'] == 'TimeMonitoring' for ext in workflow.Extractions]):
        workflow._interface.add_to_Extractions_TimeMonitoring()

def split_bc_and_integral_extractions_by_family(workflow):
    familiesBC = get_familiesBC_nodes(workflow.tree)

    if not familiesBC:
        raise ValueError("did not have any family in tree")

    Extractions = []
    for Extraction in workflow.Extractions:
        if Extraction['Type'] in ['BC', 'Integral']:
            Extraction.setdefault('Fields', [])
            if isinstance(Extraction['Fields'], str):
                # NOTE Despite the check of the interface, Fields may be a str
                # when workflow.cgns is read directly, in the context of WorkflowManager
                Extraction['Fields'] = [Extraction['Fields']]
            
            fam_names = get_bc_families_names_to_extract(workflow, Extraction, familiesBC)
            if not fam_names:
                errmsg = "did not have any family associated to Extraction:\n"
                errmsg+= pretty(Extraction)
                raise ValueError(errmsg)

            for fam_name in fam_names:
                ext = copy.deepcopy(Extraction)
                ext['Source'] = fam_name
                if ext['Name'] == 'ByFamily':
                    ext['Name'] = fam_name
                try:
                    ext['FluxCoef'] = workflow.ApplicationContext['NormalizationCoefficient'][fam_name]['FluxCoef']
                except:
                    ext['FluxCoef'] = 1.
                Extractions.append(ext)
        
        else:
            Extractions.append(Extraction)

    assert_no_extraction_named_by_family_is_left(Extractions)
    workflow.Extractions = Extractions


def assert_no_extraction_named_by_family_is_left(Extractions : dict):
    unsplit_extractions = []
    for extraction in Extractions:
        if "Name" in extraction and extraction["Name"] == "ByFamily":
            unsplit_extractions += [ extraction ]

    if unsplit_extractions:        
        msg = f'Some "ByFamily" extractions where not correctly split:\n'
        msg+= f'{pretty(unsplit_extractions)}'
        raise MolaException(msg)


def replace_shortcuts(workflow):
    shortcuts = dict(
        Conservatives = workflow.Flow['Conservatives'],
    )
    
    for extraction in workflow.Extractions:
        if 'Fields' not in extraction: 
            continue
        if isinstance(extraction['Fields'], str):
            extraction['Fields'] = [extraction['Fields']]
        for shortcut, variables in shortcuts.items():
            if shortcut in extraction['Fields']:
                extraction['Fields'].remove(shortcut)
                extraction['Fields'].extend(variables)


def get_bc_families_to_extract(workflow, Extraction, familiesBC=None):
    tree = workflow.tree
    bc_families_to_extract = []
    if familiesBC is None:
        familiesBC = get_familiesBC_nodes(tree)
    requested_source = Extraction['Source']
    
    bc_dispatcher = workflow.get_bc_dispatcher()

    registered_family_names = []
    registered_bc_types = set()
    for familyBC in familiesBC:

        family = familyBC.parent()
        family_name = family.name()
        registered_family_names += [ family_name ]

        # CAVEAT this entire logic depends on a CGNS structure, and
        # specifically on values contained in Family nodes.
        # It should have depended only on abstractions of solvers like:
        # <label/type> <-> <Extraction> in order to be acceptable also for FSDM
        # or other conventions
        bc_type = familyBC.value()
        registered_bc_types.add(bc_type)
        
        bc_type_generic = bc_dispatcher.to_generic_name(bc_type)
        registered_bc_types.add(bc_type_generic)

        family_match_requirement = fnmatch(family_name, requested_source) or \
                                   fnmatch(bc_type, requested_source) or \
                                   fnmatch(bc_type_generic, requested_source)

        if family_match_requirement \
            and 'Fields' in Extraction \
            and len(Extraction['Fields']) > 0 \
            and family not in bc_families_to_extract:

            bc_families_to_extract.append(family) 

    if len(bc_families_to_extract) == 0: 
        try:
            extraction_name = Extraction["Name"]
        except:
            if 'Data' in Extraction: del Extraction['Data']
            extraction_name = "\n" + pretty(Extraction) + "\n"

        extraction_type = Extraction["Type"]
        raise MolaUserError((f'requested Source="{requested_source}" in'
            f' Extraction named "{extraction_name}" with type "{extraction_type}" does not match'
            f' any family from names {pretty(registered_family_names)} nor'
            f' from types {registered_bc_types}'+ "\n" + pretty(Extraction) + "\n"))

    return bc_families_to_extract

def get_familiesBC_nodes(tree):

    families = tree.group(Type='Family', Depth=2)
    familiesBC = []
    for family in families:
        familyBC = family.get(Type='FamilyBC', Depth=1)
        if familyBC:
            familiesBC += [ familyBC ]

    return familiesBC


def get_bc_families_names_to_extract(workflow, Extraction, familiesBC=None):
    bc_families_to_extract = get_bc_families_to_extract(workflow, Extraction, familiesBC=familiesBC)
    fam_names = [fam.name() for fam in bc_families_to_extract]
    return fam_names

def update_extractions_from_convergence_criteria(workflow):
    # TODO PostprocessOperations has to be handle with 
    # workflow._interface.add_PostprocessOperations

    def _get_extraction_from_source(source):
        for Extraction in workflow.Extractions:
            if Extraction['Type'] not in [
                # 'Residuals', # FIXME do not have Source, so find out another way to detect it
                # 'Probe', # TODO
                'Integral']: 
                continue

            if source == Extraction['Source']:
                return Extraction
        return None

    def _append_var_to_extraction_if_needed(extraction, var):
        vector_name = None
        if var.endswith('X') or var.endswith('Y') or var.endswith('Z'):
            # var is a vector component
            vector_name = var[:-1]
        if var not in extraction['Fields'] or (vector_name and vector_name not in extraction['Fields']):
            extraction['Fields'].append(var)

    def _split_operations_on_variable(var: str, operations=None, full_name=None) -> tuple:
        '''
        Parameters
        ----------
        var : str
            input variable name, for instance 'std-avg-MassFlow'
        prefixes : str, optional
            accumulator used by the recursive function. User must not use it. By default None

        Returns
        -------
        tuple

        Example
        -------
        _split_operations_on_variable('std-avg-MassFlow') returns ('std-avg', 'MassFlow')
        '''
        if full_name is None:
            full_name = var
        if operations is None: 
            operations = []

        for op in AVAILABLE_OPERATIONS_ON_SIGNALS:
            prefix = op + '-'
            if var.startswith(prefix):
                operations.append(op)
                var = var[len(prefix):]
                operations, var = _split_operations_on_variable(var, operations, full_name=full_name)

        # Check if there is still a dash
        if '-' in var:
            raise MolaUserError((
                f'There is still a dash in the variable name "{var}" extracted from the requirement "{full_name}". '
                'There should be an syntax error, check the input in ConvergenceCriteria.'
            ))

        operations.reverse()  # to have operations to apply for var
        return operations, var

    for criterion in workflow.ConvergenceCriteria:
        operations, var = _split_operations_on_variable(criterion['Variable'])
        
        # Search the extraction to modify
        found_extraction = _get_extraction_from_source(criterion['ExtractionName'])

        if found_extraction is not None:   
            _append_var_to_extraction_if_needed(found_extraction, var)

            for op in operations:
                PostprocessOperation = dict(Type=op, Variable=var, AtEndOfRunOnly=False)
                var = f'{op}-{var}'

                if not 'PostprocessOperations' in found_extraction:
                    found_extraction['PostprocessOperations'] = [PostprocessOperation]
                elif not PostprocessOperation in found_extraction['PostprocessOperations']:
                    found_extraction['PostprocessOperations'].append(PostprocessOperation)
        
        else:
            PostprocessOperations = []
            var_tmp = var
            for op in operations:
                PostprocessOperations.append(dict(Type=op, Variable=var_tmp, AtEndOfRunOnly=False))
                var_tmp = f'{op}-{var_tmp}'

            workflow._interface.add_to_Extractions_Integral(
                Name=criterion['ExtractionName'],
                Source=criterion['ExtractionName'],
                Fields=[var],
                PostprocessOperations=PostprocessOperations
            )
            try:
                workflow.Extractions[-1]['FluxCoef'] = workflow.ApplicationContext['NormalizationCoefficient'][criterion['ExtractionName']]['FluxCoef']
            except:
                workflow.Extractions[-1]['FluxCoef'] = 1.

def print_extractions(Extractions: list):

    def sort_by_file_and_type(Extractions):
        extraction_files = dict()
        for ext in Extractions:
            if ext['File'] not in extraction_files:
                extraction_files[ext['File']] = [ext]
            else:
                extraction_files[ext['File']].append(ext)
        
        for filename in extraction_files:
            extraction_files[filename] = sorted(extraction_files[filename], key=lambda e: e['Type'])

        return extraction_files
    
    for filename, extractions in sort_by_file_and_type(Extractions).items():
        if filename == FILE_INPUT_SOLVER:
            continue
        mola_logger.info(f'  To write in {filename}:', rank=0)
        for ext in extractions:
            msg = False
            try:
                fields = ', '.join(ext['Fields'])
            except:
                fields = 'no fields'
            if ext['Type'] == 'Restart':
                continue
            elif ext['Type'] in ['Residuals', 'TimeMonitoring', 'MemoryUsage']:
                msg = f"    - {ext['Type']}"
            elif ext['Type'] in ['3D', 'Interpolation']:
                msg = f"    - {ext['Type']} extraction at {ext['GridLocation']} in {ext['Frame']} Frame for {fields} "
            elif ext['Type'] == 'Probe':
                msg = f"    - Probe {ext['Name']} at {ext['Position']} for {fields}"
            elif ext['Type'] == 'Integral':
                msg = f"    - {fields} on {ext['Source']}"
            elif ext['Type'] == 'IsoSurface':
                msg = f"    - {ext['Name']} with {fields}"
            elif ext['Type'] == 'BC':
                msg = f"    - BC {ext['Source']} with {fields}"
        
            if msg: mola_logger.info(msg, rank=0)
    