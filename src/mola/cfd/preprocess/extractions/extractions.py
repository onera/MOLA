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
from mola.logging.exceptions import MolaUserError

def apply(workflow):

    replace_shortcuts(workflow)
    add_residuals_extraction(workflow)
    add_memory_usage_extraction(workflow)
    add_time_monitoring_extraction(workflow)
    split_bc_and_integral_extractions_by_family(workflow)
    update_extractions_from_convergence_criteria(workflow)
    apply_to_solver(workflow)
    
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
            
            fam_names = get_bc_families_names_to_extract(workflow.tree, Extraction, familiesBC)

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

    workflow.Extractions = Extractions

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

def get_familiesBC_nodes(tree):

    families = tree.group(Type='Family', Depth=2)
    familiesBC = []
    for family in families:
        familyBC = family.get(Type='FamilyBC', Depth=1)
        if familyBC:
            familiesBC += [ familyBC ]

    return familiesBC

def get_bc_families_to_extract(tree, Extraction, familiesBC=None):
    bc_families_to_extract = []
    if familiesBC is None:
        familiesBC = get_familiesBC_nodes(tree)
    requested_source = Extraction['Source']

    family_node_matched = None
    registered_family_names = []
    registered_bc_types = []
    for familyBC in familiesBC:

        family = familyBC.parent()
        family_name = family.name()
        registered_family_names += [ family_name ]
        bc_type = familyBC.value()
        registered_bc_types += [ bc_type ]
        
        family_match_requirement = fnmatch(family_name, requested_source) or fnmatch(bc_type, requested_source) 

        if family_match_requirement and 'Fields' in Extraction and len(Extraction['Fields']) > 0:
            family_node_matched = family
            break

    if not family_node_matched:
        try:
            extraction_name = Extraction["Name"]
        except:
            if 'Data' in Extraction: del Extraction['Data']
            extraction_name = "\n" + pretty(Extraction) + "\n"
        raise MolaUserError((f'requested Source="{requested_source}" in'
            f' Extraction named "{extraction_name}" does not match'
            f' any family from names {pretty(registered_family_names)} nor'
            f' from types {pretty(registered_bc_types)}'))

    if family not in bc_families_to_extract:
        bc_families_to_extract.append(family) 

    return bc_families_to_extract

def get_bc_families_names_to_extract(tree, Extraction, familiesBC=None):
    bc_families_to_extract = get_bc_families_to_extract(tree, Extraction, familiesBC=familiesBC)
    fam_names = [fam.name() for fam in bc_families_to_extract]
    return fam_names

def update_extractions_from_convergence_criteria(workflow):
    # TODO PostprocessOperations has to be handle with 
    # workflow._interface.add_PostprocessOperations
    for criterion in workflow.ConvergenceCriteria:
        operation, var = _split_operations_on_variable(criterion['Variable'])
        PostprocessOperation = dict(Type=operation, Variable=var, AtEndOfRunOnly=False)
        
        extraction_ok = False
        for Extraction in workflow.Extractions:
            if Extraction['Type'] not in ['BC', 'Integral']: 
                continue

            if criterion['ExtractionName'] == Extraction['Source']:
                extraction_ok = True
                vector_name = None
                if var.endswith('X') or var.endswith('Y') or var.endswith('Z'):
                    # var is a vector component
                    vector_name = var[:-1]
                if var not in Extraction['Fields'] and (vector_name and vector_name not in Extraction['Fields']):
                    Extraction['Fields'].append(var)
                if len(operation) > 0:
                    if not 'PostprocessOperations' in Extraction:
                        Extraction['PostprocessOperations'] = [PostprocessOperation]
                    else:
                        Extraction['PostprocessOperations'].append(PostprocessOperation)
        
        if not extraction_ok:
            workflow._interface.add_to_Extractions_Integral(
                Name=criterion['ExtractionName'],
                Source=criterion['ExtractionName'],
                Fields=[var],
                PostprocessOperations=[PostprocessOperation]
            )
            try:
                workflow.Extractions[-1]['FluxCoef'] = workflow.ApplicationContext['NormalizationCoefficient'][criterion['ExtractionName']]['FluxCoef']
            except:
                workflow.Extractions[-1]['FluxCoef'] = 1.

def _split_operations_on_variable(var: str, prefixes=None) -> tuple:
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
    if prefixes is None: 
        prefixes = ''

    for op in AVAILABLE_OPERATIONS_ON_SIGNALS:
        prefix = op + '-'
        if var.startswith(prefix):
            prefixes += prefix
            var = var[len(prefix):]
            prefixes, var = _split_operations_on_variable(var, prefixes)

    # Remove final '-' if prefixes is not empty
    if prefixes[-1] == '-':
        prefixes = prefixes[:-1]
    return prefixes, var
