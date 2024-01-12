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

import os
import shutil 
from mola import (cgns, misc)
from mola import __MOLA_PATH__
from mola.cfd.preprocess.write_cfd_files import write_cfd_files

def adapt_to_solver(workflow):

    add_reference_state(workflow)
    add_governing_equations(workflow)
    if hasattr(workflow, '_FULL_CGNS_MODE'):
        add_elsa_keys_to_cgns(workflow)

    write_run_scripts(workflow)
    write_data_files(workflow)

def add_reference_state(workflow):
    '''
    Add ``ReferenceState`` node to CGNS using user-provided conditions
    '''

    ReferenceState = dict(**workflow.Flow['ReferenceState'])

    for var in ['Reynolds','Mach','Pressure','Temperature']:
        ReferenceState[var] = workflow.Flow[var]
 
    namesForCassiopee = dict(
        cv                    = 'Cv',
        Gamma                 = 'Gamma',
        SutherlandViscosity   = 'Mus',
        SutherlandConstant    = 'Cs',
        SutherlandTemperature = 'Ts',
        Prandtl               = 'Pr',
    )
    for var in ['cv','Gamma','SutherlandViscosity','SutherlandConstant','SutherlandTemperature','Prandtl']:
        ReferenceState[namesForCassiopee[var]] = workflow.Fluid[var]

    for base in workflow.tree.bases():
        base.setParameters('ReferenceState', ContainerType='ReferenceState', **ReferenceState)

def add_governing_equations(workflow):
    '''
    Add the nodes corresponding to `FlowEquationSet_t`
    '''
    FlowEquationSet = cgns.Node(Name='FlowEquationSet', Type='FlowEquationSet')
    cgns.Node(Parent=FlowEquationSet, Name='GoverningEquations', Type='GoverningEquations', Value='NSTurbulent')
    cgns.Node(Parent=FlowEquationSet, Name='EquationDimension', Type='EquationDimension', Value=workflow.ProblemDimension)

    workflow.tree.findAndRemoveNodes(Type='FlowEquationSet', Depth=2)
    for base in workflow.tree.bases():
        base.addChild(FlowEquationSet)

def add_elsa_keys_to_cgns(workflow):
    '''
    Include node ``.Solver#Compute`` , where elsA keys are set in full CGNS mode.
    '''
    workflow.tree.findAndRemoveNodes(Name='.Solver#Compute', Depth=2)

    # Put all solver keys in a unique and flat dictionary
    AllElsAKeys = dict()
    for keySet in workflow.SolverParameters.values():
        AllElsAKeys.update(keySet)
      
    for base in workflow.tree.bases(): 
        base.setParameters('.Solver#Compute', **AllElsAKeys)

def write_data_files(workflow):

    t = workflow.tree

    # HACK required in order to avoid AssertionError at line 771 in
    # etc/pypart/PpartCGNS/LayoutsS.pxi, Layouts.splitBCDataSet 
    for node in t.group(Name='BCDataSet#Average', Type='BCDataSet'):
        node.setType('UserDefinedData')

    # Save fields.cgns with the 3D fields
    os.makedirs(os.path.join(workflow.RunManagement['RunDirectory'], 'OUTPUT'), exist_ok=True)
    t.save(os.path.join(workflow.RunManagement['RunDirectory'], 'OUTPUT', 'fields.cgns'))

    # Save main.cgns with links to OUTPUT/fields.cgns for 
    NodesToLink = t.group(Name='FlowSolution#Init*', Type='FlowSolution', Depth=3) # for initial field(s) (possible second order restart)
    NodesToLink += t.group(Name='FlowSolution#Average', Type='FlowSolution', Depth=3) 
    NodesToLink += t.group(Name='BCDataSet#Average') 
    
    for FlowSolutionInit in NodesToLink:
        path = FlowSolutionInit.path()
        FlowSolutionInit.remove()
        t.addLink(path=path, target_file='OUTPUT/fields.cgns', target_path=path)
    t.save(os.path.join(workflow.RunManagement['RunDirectory'], 'main.cgns'))

def saveMainCGNSwithLinkToOutputFields(t, DIRECTORY_OUTPUT='OUTPUT',
                               MainCGNSFilename='main.cgns',
                               FieldsFilename='fields.cgns',
                               writeOutputFields=True):
    '''
    Saves the ``main.cgns`` file including linsk towards ``OUTPUT/fields.cgns``
    file, which contains ``FlowSolution#Init`` fields.

    Parameters
    ----------

        t : PyTree
            fully preprocessed PyTree

        DIRECTORY_OUTPUT : str
            folder containing the file ``fields.cgns``

            .. note:: it is advised to use ``'OUTPUT'``

        MainCGNSFilename : str
            name for main CGNS file.

            .. note:: it is advised to use ``'main.cgns'``

        FieldsFilename : str
            name of CGNS file containing initial fields

            .. note:: it is advised to use ``'fields.cgns'``

        writeOutputFields : bool
            if :py:obj:`True`, write ``fields.cgns`` file

    Returns
    -------

        None - None
            files ``main.cgns`` and eventually ``OUTPUT/fields.cgns`` are written
    '''
    print('gathering links between main CGNS and fields')
    AllCGNSLinks = []
    include_zone_bc_link = I.getNodeFromName(t,'.Solver#Output#Average') is not None
    for b in I.getBases(t):
        for z in b[2]:
            if z[3] != 'Zone_t': continue
            for fs in I.getNodesFromName(z, 'FlowSolution#Init*') + I.getNodesFromName(z, 'FlowSolution#Average'):
                currentNodePath='/'.join([b[0], z[0], fs[0]])
                targetNodePath=currentNodePath
                AllCGNSLinks += [['.',
                                DIRECTORY_OUTPUT+'/'+FieldsFilename,
                                '/'+targetNodePath,
                                currentNodePath]]

            if include_zone_bc_link:
                zbc = I.getNodeFromType1(z,'ZoneBC_t')
                if zbc:
                    for bc in I.getNodesFromType1(zbc, 'BC_t'):
                        currentNodePath='/'.join([b[0], z[0], zbc[0], bc[0], 'BCDataSet#Average'])
                        bcdsavg = I.createNode('BCDataSet#Average', 'BCDataSet_t', parent=bc)

                        targetNodePath=currentNodePath
                        AllCGNSLinks += [['.',
                                        DIRECTORY_OUTPUT+'/'+FieldsFilename,
                                        '/'+targetNodePath,
                                        currentNodePath]]

    print('saving PyTrees with links')
    to = I.copyRef(t)
    I._renameNode(to, 'FlowSolution#Centers', 'FlowSolution#Init')

    # HACK required in order to avoid AssertionError at line 771 in
    # etc/pypart/PpartCGNS/LayoutsS.pxi, Layouts.splitBCDataSet 
    for b in I.getBases(to):
        for z in b[2]:
            if z[3] != 'Zone_t': continue
            zbc = I.getNodeFromType1(z,'ZoneBC_t')
            if zbc:
                for bc in I.getNodesFromType1(zbc, 'BC_t'):
                    bcdsavg = I.getNodeFromName1(bc, 'BCDataSet#Average')
                    if bcdsavg: bcdsavg[3] = 'UserDefinedData_t'

    if writeOutputFields:
        try: os.makedirs(DIRECTORY_OUTPUT)
        except: pass
        C.convertPyTree2File(to, os.path.join(DIRECTORY_OUTPUT, FieldsFilename))
    C.convertPyTree2File(t, MainCGNSFilename, links=AllCGNSLinks)


def write_run_scripts(workflow):
    write_compute(workflow.RunManagement)
    write_coprocess(workflow.RunManagement)
    write_job_launcher(workflow.RunManagement)

def write_compute(RunManagement):

    txt = '''
from mola.workflow.workflow import Workflow

workflow = Workflow('main.cgns')
workflow.print()
workflow.compute()
'''
    compute_filename = os.path.join(RunManagement['RunDirectory'], 'compute.py')
    with open(compute_filename, 'w') as File:
        File.write(txt)
    os.chmod(compute_filename, 0o777)

def write_coprocess(RunManagement):
    with open(os.path.join(RunManagement['RunDirectory'], 'coprocess.py'), 'w') as File:
        File.write('# do nothing')

def write_job_launcher(RunManagement, jobFile='job.sh'):

    # shutil.copy2(f'{__MOLA_PATH__}/TEMPLATES/job_template.sh', 'job.sh')
    job_text = write_cfd_files.get_job_text(RunManagement, 'elsa')
    job_text += f'mpirun $OPENMPIOVERSUBSCRIBE -np {RunManagement["NumberOfProcessors"]} elsA.x -C xdt-runtime-tree compute.py 1>stdout.log 2>stderr.log\n'
    
    # Write job file
    job_filename = os.path.join(RunManagement['RunDirectory'], jobFile)
    with open(job_filename, 'w') as f:
        f.write(job_text)
    os.chmod(job_filename, 0o777)
