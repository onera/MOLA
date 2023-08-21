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
from mola import misc, cgns

import Converter.PyTree as C
import Converter.Internal as I


def apply(workflow):
    '''
    Initialize the flow solution.

    #. Compute FlowSolution#Init in all zones
    
    #. Adapt this node to the solver
    '''

    if workflow.Initialization['method'] is None:
        pass
    elif workflow.Initialization['method'] == 'uniform':
        print(misc.CYAN + 'Initialize FlowSolution with uniform reference values' + misc.ENDC)
        workflow.tree.newFields(workflow.Flow['ReferenceState'], Container='FlowSolution#Init')

    elif workflow.Initialization['method'] == 'interpolate':
        print(misc.CYAN + f"Initialize FlowSolution by interpolation from {workflow.Initialization['file']}" + misc.ENDC)
        initialize_flow_from_file_by_interpolation(workflow)
        
    elif workflow.Initialization['method'] == 'copy':
        print(misc.CYAN + f"Initialize FlowSolution by copy of {workflow.Initialization['file']}" + misc.ENDC)
        initialize_flow_from_file_by_copy(workflow)
    else:
        raise Exception(misc.RED+'The key "method" of the dictionary workflow.Initialization is mandatory'+misc.ENDC)

    for zone in workflow.tree.zones():
        if not zone.get(Name='FlowSolution#Init', Type='FlowSolution', Depth=1):
            MSG = 'FlowSolution#Init is missing in zone {}'.format(zone.name)
            raise ValueError(misc.RED + MSG + misc.ENDC)
    
    current_path = os.path.dirname(os.path.realpath(__file__))
    solverModule = misc.load_source('solverModule', os.path.join(current_path, f'solver_{workflow.Solver}.py'))
    solverModule.adapt_to_solver(workflow)



def initialize_flow_from_file_by_interpolation(t, ReferenceValues, sourceFilename, container='FlowSolution#Init'):
    '''
    Initialize the flow solution of **t** from the flow solution in the file
    **sourceFilename**.
    Modify the tree **t** in-place.

    Parameters
    ----------

        t : PyTree
            Tree to initialize

        ReferenceValues : dict
            as produced by :py:func:`computeReferenceValues`

        sourceFilename : str
            Name of the source file for the interpolation.

        container : str
            Name of the ``'FlowSolution_t'`` node use for the interpolation.
            Default is 'FlowSolution#Init'

    '''
    sourceTree = C.convertFile2PyTree(sourceFilename)
    OLD_FlowSolutionCenters = I.__FlowSolutionCenters__
    I.__FlowSolutionCenters__ = container
    sourceTree = C.extractVars(sourceTree, ['centers:{}'.format(var) for var in ReferenceValues['Fields']])

    I._rmNodesByType(sourceTree, 'BCDataSet_t')
    I._rmNodesByNameAndType(sourceTree, '*EndOfRun*', 'FlowSolution_t')
    P._extractMesh(sourceTree, t, mode='accurate', extrapOrder=0)
    if container != 'FlowSolution#Init':
        I._rmNodesByName(t, 'FlowSolution#Init')
        I.renameNode(t, container, 'FlowSolution#Init')
    I.__FlowSolutionCenters__ = OLD_FlowSolutionCenters

def initialize_flow_from_file_by_copy(workflow):
    '''
    Initialize the flow solution of **t** by copying the flow solution in the file
    **sourceFilename**.
    Modify the tree **t** in-place.

    Parameters
    ----------

        t : PyTree
            Tree to initialize

        ReferenceValues : dict
            as produced by :py:func:`computeReferenceValues`

        sourceFilename : str
            Name of the source file.

        container : str
            Name of the ``'FlowSolution_t'`` node to copy.
            Default is 'FlowSolution#Init'

        keepTurbulentDistance : bool
            if :py:obj:`True`, copy also fields ``'TurbulentDistance'`` and
            ``'TurbulentDistanceIndex'``.

            .. danger::
                The restarted simulation must be submitted with the same
                CPU distribution that the previous one ! It is due to the field
                ``'TurbulentDistanceIndex'`` that indicates the index of the
                nearest wall, and this index varies with the distribution.

    '''
    keepTurbulentDistance = workflow.Initialization.get('keepTurbulentDistance', False)

    sourceTree = cgns.load(workflow.Initialization['file'])

    varNames = list(workflow.Flow['ReferenceState'])
    if keepTurbulentDistance:
        varNames += ['TurbulentDistance', 'TurbulentDistanceIndex']

    for zone in workflow.tree.zones():
        FSpath = zone.path() + '/FlowSolution#Init'
        try:
            FlowSolutionInSourceTree = sourceTree.getAtPath(FSpath)
            zone.addChild(FlowSolutionInSourceTree, override_brother_by_name=True)
        except AttributeError:
            ERROR_MSG = f"The node {FSpath} is not found in {workflow.Initialization['file']}"
            raise Exception(misc.RED+ERROR_MSG+misc.ENDC)
