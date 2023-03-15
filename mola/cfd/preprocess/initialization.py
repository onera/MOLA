#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

import Converter.PyTree as C
import Converter.Internal as I


def initializeFlowSolution(t, Initialization, ReferenceValues):
    '''
    Initialize the flow solution in tree **t**.

    Parameters
    ----------

        t : PyTree
            preprocessed tree as performed by :py:func:`prepareMesh4ElsA`

        Initialization : dict
            dictionary defining the type of initialization, using the key
            **method**. This latter is mandatory and should be one of the
            following:

            * **method** = :py:obj:`None` : the Flow Solution is not initialized.

            * **method** = ``'uniform'`` : the Flow Solution is initialized uniformly
              using the **ReferenceValues**.

            * **method** = ``'copy'`` : the Flow Solution is initialized by copying
              the FlowSolution container of another file. The file path is set by
              using the key **file**. The container might be set with the key
              **container** (``'FlowSolution#Init'`` by default).

            * **method** = ``'interpolate'`` : the Flow Solution is initialized by
              interpolating the FlowSolution container of another file. The file
              path is set by using the key **file**. The container might be set
              with the key **container** (``'FlowSolution#Init'`` by default).

            Default method is ``'uniform'``.

        ReferenceValues : dict
            dictionary as got from :py:func:`computeReferenceValues`

    '''
    if not 'container' in Initialization:
        Initialization['container'] = 'FlowSolution#Init'

    if Initialization['method'] is None:
        pass
    elif Initialization['method'] == 'uniform':
        print(J.CYAN + 'Initialize FlowSolution with uniform reference values' + J.ENDC)
        initializeFlowSolutionFromReferenceValues(t, ReferenceValues)
    elif Initialization['method'] == 'interpolate':
        print(J.CYAN + 'Initialize FlowSolution by interpolation from {}'.format(Initialization['file']) + J.ENDC)
        initializeFlowSolutionFromFileByInterpolation(t, ReferenceValues,
            Initialization['file'], container=Initialization['container'])
    elif Initialization['method'] == 'copy':
        print(J.CYAN + 'Initialize FlowSolution by copy of {}'.format(Initialization['file']) + J.ENDC)
        if not 'keepTurbulentDistance' in Initialization:
            Initialization['keepTurbulentDistance'] = False
        initializeFlowSolutionFromFileByCopy(t, ReferenceValues, Initialization['file'],
            container=Initialization['container'],
            keepTurbulentDistance=Initialization['keepTurbulentDistance'])
    else:
        raise Exception(J.FAIL+'The key "method" of the dictionary Initialization is mandatory'+J.ENDC)

    for zone in I.getZones(t):
        if not I.getNodeFromName1(zone, 'FlowSolution#Init'):
            MSG = 'FlowSolution#Init is missing in zone {}'.format(I.getName(zone))
            raise ValueError(J.FAIL + MSG + J.ENDC)
        

def initializeFlowSolutionFromReferenceValues(t, ReferenceValues):
    '''
    Invoke ``FlowSolution#Init`` fields using information contained in
    ``ReferenceValue['ReferenceState']`` and ``ReferenceValues['Fields']``.

    .. note:: This is equivalent as a *uniform* initialization of flow.

    Parameters
    ----------

        t : PyTree
            main CGNS PyTree where fields are going to be initialized

            .. note:: tree **t** is modified

        ReferenceValues : dict
            as produced by :py:func:`computeReferenceValues`

    '''
    print('invoking FlowSolution#Init with uniform fields using ReferenceState')
    I._renameNode(t,'FlowSolution#Centers','FlowSolution#Init')
    FieldsNames = ReferenceValues['Fields']
    I.__FlowSolutionCenters__ = 'FlowSolution#Init'
    for i in range(len(ReferenceValues['ReferenceState'])):
        FieldName = FieldsNames[i]
        FieldValue= ReferenceValues['ReferenceState'][i]
        C._initVars(t,'centers:%s'%FieldName,FieldValue)

    # TODO : This should not be required.
    # FieldsNamesAdd = ReferenceValues['FieldsAdditionalExtractions'].split(' ')
    # for fn in FieldsNamesAdd:
    #     try:
    #         ValueOfField = ReferenceValues[fn]
    #     except KeyError:
    #         ValueOfField = 1.0
    #     C._initVars(t,'centers:%s'%fn,ValueOfField)
    I.__FlowSolutionCenters__ = 'FlowSolution#Centers'

    FlowSolInit = I.getNodesFromName(t,'FlowSolution#Init')
    I._rmNodesByName(FlowSolInit, 'ChimeraCellType')

def initializeFlowSolutionFromFileByInterpolation(t, ReferenceValues, sourceFilename, container='FlowSolution#Init'):
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

def initializeFlowSolutionFromFileByCopy(t, ReferenceValues, sourceFilename,
        container='FlowSolution#Init', keepTurbulentDistance=False):
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
    sourceTree = C.convertFile2PyTree(sourceFilename)
    OLD_FlowSolutionCenters = I.__FlowSolutionCenters__
    I.__FlowSolutionCenters__ = container
    varNames = copy.deepcopy(ReferenceValues['Fields'])
    if keepTurbulentDistance:
        varNames += ['TurbulentDistance', 'TurbulentDistanceIndex']

    sourceTree = C.extractVars(sourceTree, ['centers:{}'.format(var) for var in varNames])

    for base in I.getBases(t):
        basename = I.getName(base)
        for zone in I.getNodesFromType1(base, 'Zone_t'):
            zonename = I.getName(zone)
            zonepath = '{}/{}'.format(basename, zonename)
            FSpath = '{}/{}'.format(zonepath, container)
            FlowSolutionInSourceTree = I.getNodeFromPath(sourceTree, FSpath)
            if FlowSolutionInSourceTree:
                I._rmNodesByNameAndType(zone, container, 'FlowSolution_t')
                I._append(t, FlowSolutionInSourceTree, zonepath)
            else:
                ERROR_MSG = 'The node {} is not found in {}'.format(FSpath, sourceFilename)
                raise Exception(J.FAIL+ERROR_MSG+J.ENDC)

    if container != 'FlowSolution#Init':
        I.renameNode(t, container, 'FlowSolution#Init')

    I.__FlowSolutionCenters__ = OLD_FlowSolutionCenters
