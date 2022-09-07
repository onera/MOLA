#!/usr/bin/python

# Python general packages
import os

# Cassiopee packages
import Converter.PyTree   as C
import Converter.Internal as I
import Converter.Mpi      as Cmpi
import Post.PyTree        as P
import Transform.PyTree   as T

import MOLA.InternalShortcuts as J

########################
import turbo.fields   as TF
import turbo.height   as TH
import turbo.machis   as TMis
import turbo.meridian as TM
import turbo.perfos   as TP
import turbo.radial_future  as TR
import turbo.slicesAt as TS
import turbo.user     as TUS
import turbo.utils    as TU

setup = J.load_source('setup', 'setup.py')

class RefState(object):
    def __init__(self, setup):
      self.Gamma = setup.FluidProperties['Gamma']
      self.Rgaz  = setup.FluidProperties['IdealGasConstant']
      self.Pio   = setup.ReferenceValues['PressureStagnation']
      self.Tio   = setup.ReferenceValues['TemperatureStagnation']
      self.roio  = self.Pio / self.Tio / self.Rgaz
      self.aio   = (self.Gamma * self.Rgaz * self.Tio)**0.5
      self.Lref  = 1.

def getExtractionInfo(surface):
    '''
    Get information into ``.ExtractionInfo`` of **surface**.

    Parameters
    ----------

        surface : PyTree
            Base corresponding to a surface, with a ``.ExtractionInfo`` node

    Returns
    -------

        info : dict
            dictionary with the template:

            >>> info[nodeName] = nodeValue

    '''
    ExtractionInfo = I.getNodeFromName1(surface, '.ExtractionInfo')
    info = dict((I.getName(node), str(I.getValue(node))) for node in I.getChildren(ExtractionInfo))
    return info

def getSurfacesFromInfo(surfaces, breakAtFirst=False, **kwargs):
    '''
    Inside a top tree **surfaces**, search for the nodes with a ``.ExtractionInfo``
    matching the requirements in **kwargs**

    Parameters
    ----------

        surfaces : PyTree
            top tree or base

        kwargs : unwrapped :py:class:`dict`
            parameters required in the ``.ExtractionInfo`` node of the searched
            zone

    Returns
    -------

        extractedSurfaces : list
            bases that matches with **kwargs**
    '''
    for key, value in kwargs.items():
        kwargs[key] = str(value)
    topType = I.getType(surfaces)
    if topType == 'CGNSTree_t':
        getChildren = I.getBases
    elif topType == 'CGNSBase_t':
        getChildren = I.getZones
    else:
        raise TypeError('surfaces must be eiter a CGNSTree_t or a CGNSBase_t')
    extractedSurfaces = []
    for surface in getChildren(surfaces):
        info = getExtractionInfo(surface)
        if all([key in info and info[key] == value for key, value in kwargs.items()]):
            extractedSurfaces.append(surface)
            if breakAtFirst:
                break
    return extractedSurfaces

def getSurfaceFromInfo(surfaces, **kwargs):
    '''
    Inside a top tree **surfaces**, search for the node with a ``.ExtractionInfo``
    matching the requirements in **kwargs**

    Parameters
    ----------

        surfaces : PyTree
            top tree or base

        kwargs : unwrapped :py:class:`dict`
            parameters required in the ``.ExtractionInfo`` node of the searched
            zone

    Returns
    -------

        surface : PyTree or :py:obj:`None`
            node that matches with **kwargs**
    '''
    extractedSurfaces = getSurfacesFromInfo(surfaces, breakAtFirst=True, **kwargs)
    if len(extractedSurfaces) == 0:
        return None
    else:
        return extractedSurfaces[0]

def getSurfaceArea(surface):
    base =  C.convertArray2Tetra(surface)
    I._rmNodesByName(base, '.*')
    # I._rmNodesByName(base, I.__FlowSolutionNodes__)
    C._initVars(base, 'ones=1')
    # area = abs(P.integNorm(base, var='ones')[0][0])
    area = 0
    for zone in I.getZones(base):
        area += abs(P.integNorm(zone, var='ones')[0][0])
    return area

@J.mute_stdout
def _computeAeroVariables(surfaces, filename=None, useSI=True, velForm='absolute'):
    for surface in I.getNodesFromType(surfaces, 'CGNSBase_t'):
        info = getExtractionInfo(surface)
        if info['type'] != 'IsoSurface': continue
        tmp_surface = C.node2Center(surface, I.__FlowSolutionNodes__)
        for fsname in [I.__FlowSolutionNodes__, I.__FlowSolutionCenters__]:
            TF._computeOtherFields(tmp_surface, RefState(setup), TUS.getFields(),
                fsname=fsname, useSI=useSI, velocity=velForm, specific_tv_dir_name=True)
        I.rmNode(surfaces, surface)
        I.addChild(surfaces, tmp_surface)
    if filename: C.convertPyTree2File(surfaces, filename)

def computeAeroVariables(surfaces, filename=None, useSI=True, velForm='absolute',
                        var2saveCell=None, var2saveVertex=None):
    new_surfaces = I.copyTree(surfaces)
    _computeAeroVariables(new_surfaces, filename=filename, useSI=useSI, velForm=velForm)
    mergeFlowSolutionOfTrees(surfaces, new_surfaces, var2save=var2saveCell, container=I.__FlowSolutionCenters__)
    mergeFlowSolutionOfTrees(surfaces, new_surfaces, var2save=var2saveVertex, container=I.__FlowSolutionNodes__)
    return new_surfaces

def computePerfosFromPlanes(upstream, downstream, downstream2=None, var2avgByMassflow=[], var2avgBySurface=[], var4comp=[]):
    '''
    Compute row performance (massflow in/out, total pressure ratio,
    total temperature ratio, isentropic efficiency) between two surfaces.
    Results are written into a file.

    Parameters
    ----------

        upstream : PyTree
            Top PyTree or base corresponding to the upstream surface for
            performance computation.

        downstream : PyTree
            Top PyTree or base corresponding to the downstream surface for
            performance computation.

        downstream2 : PyTree

    '''
    TurboConfiguration = setup.TurboConfiguration

    if downstream2 is not None:
        plane_names = ['upstream','downstream', 'downstream2']
        plane_trees = [upstream, downstream, downstream2]
    else:
        plane_names = ['upstream','downstream']
        plane_trees = [upstream, downstream]

    perfos = dict()
    for plane, slices in zip(plane_names, plane_trees):
        ExtractionInfo = I.getNodeFromName(slices, '.ExtractionInfo')
        ReferenceRow = I.getValue(I.getNodeFromName(ExtractionInfo, 'ReferenceRow'))

        nBlades = TurboConfiguration['Rows'][ReferenceRow]['NumberOfBlades']
        nBladesSimu = TurboConfiguration['Rows'][ReferenceRow]['NumberOfBladesSimulated']
        fluxcoeff = nBlades / float(nBladesSimu)
        perfTreeMassflow = TP.computePerformances(slices, plane,
            variables=var2avgByMassflow,
            average='massflow',
            compute_massflow=True,
            fluxcoef=fluxcoeff,
            fsname=I.__FlowSolutionCenters__)
        perfTreeSurface = TP.computePerformances(slices, plane,
            variables=var2avgBySurface,
            average='surface',
            compute_massflow=True,
            fluxcoef=fluxcoeff,
            fsname=I.__FlowSolutionCenters__)

        perfos[plane] = I.merge([perfTreeMassflow, perfTreeSurface])

    tAvg = I.merge(perfos.values())

    if downstream2 is not None:
        tBilan = TP.comparePerformancesPlane2Plane(perfos['upstream'], perfos['downstream'],
                                          ['upstream','downstream', 'downstream2'],
                                          'Bilan',
                                          t3=perfos['downstream2'],
                                          config='compressor',
                                          variables=var4comp)
    else:
        tBilan = TP.comparePerformancesPlane2Plane(perfos['upstream'], perfos['downstream'],
                                          ['upstream','downstream'],
                                          'Bilan',
                                          config='compressor',
                                          variables=var4comp)

    tPerfos = I.merge([tAvg, tBilan])

    return tPerfos

def sortVariablesByAverage(variables):
    '''
    Sort variables in a dictionnary by average type.
    Currently, every variable that contains 'Stagnation' or 'Entropy' in
    its name is appended to the 'massflow' list. Every other variable is appended
    to the 'surface' list.

    Parameters
    ----------

        variables : list
            list of variable names to sort

    Examples
    --------

        input:

        .. code-block:: python

            variables = ['Mach', 'Pressure', 'StagnationPressure', 'Entropy']
            sortedVariables = sortVariablesByAverage(variables)
            print(sortedVariables)

        output:

        .. code-block:: python

            {'massflow': ['StagnationPressure', 'Entropy'], 'surface': ['Mach', 'Pressure']}

    '''
    averages = dict(massflow=[], surface=[])
    for var in variables:
        if any([pattern in var for pattern in ['Stagnation', 'Entropy']]):
            averages['massflow'].append(var)
        else:
            averages['surface'].append(var)
    return averages

def mergeFlowSolutionOfTrees(t1, t2, var2save=None, container=I.__FlowSolutionCenters__):
    for base1 in I.getBases(t1):
        for zone1 in I.getZones(base1):
            for FS1 in I.getNodesFromName1(zone1, container):
                FS1Path = '{}/{}/{}'.format(I.getName(base1), I.getName(zone1), I.getName(FS1))
                FS2 = I.getNodeFromPath(t2, FS1Path)
                if not FS2:
                    continue
                for data2 in I.getNodesFromType1(FS2, 'DataArray_t'):
                    var = I.getName(data2)
                    data1 = I.getNodesFromNameAndType(FS1, var, 'DataArray_t')
                    if not data1:
                        if var2save is None or var in var2save:
                            I.addChild(FS1, data2)
    return t1

def cleanSurfaces(surfaces, var2keep=[]):
    '''
    Clean the tree **surfaces** to keep only useful data:

    * keep only conservatives variables at nodes for IsoSurfaces, plus variables
      in **var2keep**

    * for surface corresponding to a BC extraction , delete all variables at
      nodes except the following: ['ChannelHeight', 'IsentropicMachNumber',
      'Pressure', 'StagnationPressureRelDim', 'RefStagnationPressureRelDim',
      'SkinFrictionX', 'SkinFrictionY', 'SkinFrictionZ']

    Parameters
    ----------

        surfaces : PyTree
            tree read from ``'surfaces.cgns'``

        var2keep : list
            variables to keep at nodes for IsoSurfaces (in addition to
            conservative variables)

    '''
    coordinates = ['CoordinateX', 'CoordinateY', 'CoordinateZ', 'ChannelHeight']
    conservatives = setup.ReferenceValues['Fields']
    var2keepOnBlade = ['ChannelHeight', 'IsentropicMachNumber',
        'Pressure', 'StagnationPressureRelDim', 'RefStagnationPressureRelDim',
        'SkinFrictionX', 'SkinFrictionY', 'SkinFrictionZ'
        ]

    surfacesIso = getSurfacesFromInfo(surfaces, type='IsoSurface')
    for surface in surfacesIso:
        for zone in I.getZones(surface):
            I._rmNodesByName1(zone, I.__FlowSolutionCenters__)
            C._extractVars(zone, coordinates+conservatives+var2keep)

    surfacesBC = getSurfacesFromInfo(surfaces, type='BC', BCType='BCWallViscous')
    for surface in surfacesBC:
        for zone in I.getZones(surface):
            FSnodes = I.getNodeFromName1(zone, I.__FlowSolutionNodes__)
            for node in I.getNodesFromType(FSnodes, 'DataArray_t'):
                varname = I.getName(node)
                if varname not in var2keepOnBlade:
                    I._rmNode(FSnodes, node)

@J.mute_stdout
def computeVariablesOnIsosurface(surfaces):
    I._renameNode(surfaces, 'FlowSolution#Init', I.__FlowSolutionCenters__)
    surfacesIso = getSurfacesFromInfo(surfaces, type='IsoSurface')
    varAtNodes = C.getVarNames(I.getNodeFromType1(surfacesIso[0], 'Zone_t'), loc='nodes')[0]
    for v in varAtNodes: C._node2Center__(surfacesIso, v)

    for surface in surfacesIso:
        for fsname in [I.__FlowSolutionNodes__, I.__FlowSolutionCenters__]:
            TF._computeOtherFields(surface, RefState(setup), TUS.getFields(),
                                    fsname=fsname, useSI=True, velocity='absolute')


def compute0DPerformances(surfaces, variablesByAverage, var4comp_perf):
    surfacesIsoX = getSurfacesFromInfo(surfaces, type='IsoSurface', field='CoordinateX')
    for surface in surfacesIsoX:
        ExtractionInfo = I.getNodeFromName(surface, '.ExtractionInfo')
        ReferenceRow = I.getValue(I.getNodeFromName(ExtractionInfo, 'ReferenceRow'))
        nBlades = setup.TurboConfiguration['Rows'][ReferenceRow]['NumberOfBlades']
        nBladesSimu = setup.TurboConfiguration['Rows'][ReferenceRow]['NumberOfBladesSimulated']
        fluxcoeff = nBlades / float(nBladesSimu)

        perfTreeMassflow = TP.computePerformances(surface, '.Average',
            variables=variablesByAverage['massflow'], average='massflow',
            compute_massflow=False, fluxcoef=fluxcoeff, fsname=I.__FlowSolutionCenters__)
        perfTreeSurface = TP.computePerformances(surface, '.Average',
            variables=variablesByAverage['surface'], average='surface',
            compute_massflow=True, fluxcoef=fluxcoeff, fsname=I.__FlowSolutionCenters__)

        perfos = I.merge([perfTreeMassflow, perfTreeSurface])
        perfos = I.getNodeFromType2(perfos, 'Zone_t')
        # I.setType(perfos, 'UserDefinedData_t')  # must be a zone to be detected by comparePerformancesPlane2Plane
        J.set(perfos, '.averageType', **variablesByAverage)
        I.addChild(surface, perfos)

    for row in setup.TurboConfiguration['Rows']:

        InletPlane = getSurfaceFromInfo(surfaces, ReferenceRow=row, tag='InletPlane')
        OutletPlane = getSurfaceFromInfo(surfaces, ReferenceRow=row, tag='OutletPlane')
        if not(InletPlane and OutletPlane): continue

        comparePerfoPlane2Plane(InletPlane, OutletPlane, var4comp_perf)


def comparePerfoPlane2Plane(InletPlane, OutletPlane, var4comp_perf):
    for node in I.getNodesFromName1(InletPlane, '.Average') + I.getNodesFromName1(OutletPlane, '.Average'):
        I.setType(node, 'Zone_t')

    ComparisonNodeName = '.Average#Comparison01'
    ExistingComparisonNodes = [I.getName(n) for n in I.getNodesFromName(InletPlane, '.Average#Comparison*')]
    i = 1
    while ComparisonNodeName in ExistingComparisonNodes:
        i += 1
        ComparisonNodeName = '.Average#Comparison{:02d}'.format(i)

    tBilan = TP.comparePerformancesPlane2Plane(InletPlane, OutletPlane,
                    '.Average', ComparisonNodeName,
                    config='compressor', variables=var4comp_perf)
    tBilan = I.getNodeFromType2(tBilan, 'Zone_t')
    I.setType(tBilan, 'UserDefinedData_t')
    I.createChild(tBilan, '.comparedTo', 'UserDefinedData_t', value=I.getName(OutletPlane), pos=0)
    I.addChild(InletPlane, tBilan)

    # Convert average nodes in 'UserDefinedData_t'
    # Must be done at the end, because TP.comparePerformancesPlane2Plane need
    # that these nodes are 'Zone_t'
    for node in I.getNodesFromName2(InletPlane, '.Average') + I.getNodesFromName2(OutletPlane, '.Average'):
        I.setType(node, 'UserDefinedData_t')


@J.mute_stdout
def compute1DRadialProfiles(surfaces, variablesByAverage, var4comp_repart):
    surfacesIsoX = getSurfacesFromInfo(surfaces, type='IsoSurface', field='CoordinateX')
    for surface in surfacesIsoX:
        info = getExtractionInfo(surface)
        tmp_surface = C.convertArray2NGon(surface, recoverBC=0)
        # I._rmNodesByName(tmp_surface, '.*')
        radial_surf = TR.computeRadialProfile_future(tmp_surface, '.RadialProfile',
            variablesByAverage['surface'], 'surface', fsname=I.__FlowSolutionCenters__)
        radial_massflow = TR.computeRadialProfile_future(tmp_surface, '.RadialProfile',
            variablesByAverage['massflow'], 'massflow', fsname=I.__FlowSolutionCenters__)
        t_radial = I.merge([radial_surf, radial_massflow])
        t_radial = I.getNodeFromType2(t_radial, 'Zone_t')
        J.set(t_radial, '.averageType', **variablesByAverage)
        I.addChild(surface, t_radial)

    for row in setup.TurboConfiguration['Rows']:

        InletPlane = getSurfaceFromInfo(surfaces, ReferenceRow=row, tag='InletPlane')
        OutletPlane = getSurfaceFromInfo(surfaces, ReferenceRow=row, tag='OutletPlane')
        if not(InletPlane and OutletPlane): continue

        tBilan = TR.compareRadialProfilePlane2Plane(InletPlane, OutletPlane,
                        '.RadialProfile', '.RadialProfile#Comparison',
                        config='compressor', variables=var4comp_repart)
        tBilan = I.getNodeFromType2(tBilan, 'Zone_t')
        I.setType(tBilan, 'UserDefinedData_t')
        I.createChild(tBilan, '.comparedTo', 'UserDefinedData_t', value=I.getName(OutletPlane), pos=0)
        I.addChild(InletPlane, tBilan)

    for node in I.getNodesFromName2(surfaces, '.RadialProfile'):
        I.setType(node, 'UserDefinedData_t')

@J.mute_stdout
def computeVariablesOnBladeProfiles(surfaces, allVariables, hList='all'):

    def searchBladeInTree(row):
        famnames = ['{}_*BLADE*'.format(row), '{}_*Blade*'.format(row), '{}_*AUBE*'.format(row), '{}_*Aube*'.format(row)]
        blade = None
        for famname in famnames:
            blade_families = I.getNodesFromNameAndType(surfaces, famname, 'CGNSBase_t')
            if len(blade_families) == 1:
                blade = blade_families[0]
                break
        if not blade: raise Exception('No blade family (or more than one) has been found for row {}'.format(row))
        return blade
    
    if hList == 'all':
        hList = []
        surfacesIsoH = getSurfacesFromInfo(surfaces, type='IsoSurface', field='ChannelHeight')
        for surface in surfacesIsoH:
            ExtractionInfo = I.getNodeFromName(surface, '.ExtractionInfo')
            valueH = I.getValue(I.getNodeFromName(ExtractionInfo, 'value'))
            hList.append(valueH)

    for row in setup.TurboConfiguration['Rows']:

        InletPlane = getSurfaceFromInfo(surfaces, ReferenceRow=row, tag='InletPlane')
        if not InletPlane: continue
        TF._computeOtherFields(InletPlane, RefState(setup), allVariables, fsname=I.__FlowSolutionNodes__, useSI=True, velocity='absolute')

        blade = searchBladeInTree(row)

        I._renameNode(blade, 'FlowSolution#Centers', I.__FlowSolutionCenters__)
        C._initVars(blade, 'Radius=sqrt({CoordinateY}**2+{CoordinateZ}**2)')
        blade = C.center2Node(blade, I.__FlowSolutionCenters__)
        C._initVars(blade, 'StaticPressureDim={Pressure}')
        #TODO: compute a new InletPlane: profiles of ptmax on the blade
        blade = TMis.computeIsentropicMachNumber(InletPlane, blade, RefState(setup))

        for h in hList:
            bladeIsoH = T.join(P.isoSurfMC(blade, 'ChannelHeight', h))
            I.setName(bladeIsoH, '.Iso_H_{}'.format(h))
            I.setType(bladeIsoH, 'UserDefinedData_t')
            I._addChild(blade, bladeIsoH)

        I._rmNodesByName1(surfaces, I.getName(blade))
        I.addChild(surfaces, blade)


def postprocess_turbomachinery(FILE_SURFACES='OUTPUT/surfaces.cgns', 
                               FILE_SURFACES_NEW='OUTPUT/surfaces.cgns', 
                               var4comp_repart=None, var4comp_perf=None, var2keep=None, 
                               stages=[]):
    '''
    Perform a series of classical postprocessings for a turbomachinery case : 

    #. Compute extra variables, in relative and absolute frames of reference

    #. Compute averaged values for all iso-X planes (results are in the `.Average` node), and
       compare inlet and outlet planes for each row if available, to get row performance (total 
       pressure ratio, isentropic efficiency, etc) (results are in the `.Average#ComparisonXX` of
       the inlet plane, `XX` being the numerotation starting at `01`)

    #. Compute radial profiles for all iso-X planes (results are in the `.RadialProfile` node), and
       compare inlet and outlet planes for each row if available, to get row performance (total 
       pressure ratio, isentropic efficiency, etc) (results are in the `.RadialProfile#ComparisonXX` of
       the inlet plane, `XX` being the numerotation starting at `01`)

    #. Compute isentropic Mach number on blades, slicing at constant height, for all values of height 
       already extracted as iso-surfaces. Results are in the `.Iso_H_XX` nodes.

    Parameters
    ----------
        FILE_SURFACES : str, optional
            Name of the input file with surfaces, by default 'OUTPUT/surfaces.cgns'

        FILE_SURFACES_NEW : str, optional
            Name of the output file with surface, by default 'OUTPUT/surfaces.cgns' (same as input)

        var4comp_repart : :py:class:`list`, optional
            List of variables computed for radial distributions. If not given, all possible variables are computed.

        var4comp_perf : :py:class:`list`, optional
            List of variables computed for row performance (plane to plane comparison). If not given, 
            the same variables as in **var4comp_repart** are computed, plus `Power`.

        var2keep : :py:class:`list`, optional
            List of variables to keep in the saved file. If not given, the following variables are kept:
            
            .. code-block:: python

                var2keep = [
                    'Pressure', 'Temperature', 'PressureStagnation', 'TemperatureStagnation',
                    'Entropy',
                    'Viscosity_EddyMolecularRatio',
                    'VelocitySoundDim', 'StagnationEnthalpyAbsDim',
                    'MachNumberAbs', 'MachNumberRel',
                    'AlphaAngleDegree',  'BetaAngleDegree', 'PhiAngleDegree',
                    'VelocityXAbsDim', 'VelocityRadiusAbsDim', 'VelocityThetaAbsDim',
                    'VelocityMeridianDim', 'VelocityRadiusRelDim', 'VelocityThetaRelDim',
                    ]

        stages : :py:class:`list` of :py:class:`tuple`, optional
            List of row stages, of the form:

            >>> stages = [('rotor1', 'stator1'), ('rotor2', 'stator2')] 

            For each tuple of rows, the inlet plane of row 1 is compared with the outlet plane of row 2.
    '''
    #______________________________________________________________________________
    # Variables
    #______________________________________________________________________________
    allVariables = TUS.getFields()
    if not var4comp_repart:
        var4comp_repart = ['StagnationEnthalpyDelta',
                        'StagnationPressureRatio', 'StagnationTemperatureRatio',
                        'StaticPressureRatio', 'Static2StagnationPressureRatio',
                        'IsentropicEfficiency', 'PolytropicEfficiency',
                        'StaticPressureCoefficient', 'StagnationPressureCoefficient',
                        'StagnationPressureLoss1', 'StagnationPressureLoss2',
                        ]
    if not var4comp_perf:
        var4comp_perf = var4comp_repart + ['Power']  
    if not var2keep:
        var2keep = [
            'Pressure', 'Temperature', 'PressureStagnation', 'TemperatureStagnation',
            'Entropy',
            'Viscosity_EddyMolecularRatio',
            'VelocitySoundDim', 'StagnationEnthalpyAbsDim',
            'MachNumberAbs', 'MachNumberRel',
            'AlphaAngleDegree',  'BetaAngleDegree', 'PhiAngleDegree',
            'VelocityXAbsDim', 'VelocityRadiusAbsDim', 'VelocityThetaAbsDim',
            'VelocityMeridianDim', 'VelocityRadiusRelDim', 'VelocityThetaRelDim',
            ]
        
    variablesByAverage = sortVariablesByAverage(allVariables)

    #______________________________________________________________________________
    Cmpi.barrier()
    surfaces = Cmpi.convertFile2PyTree(FILE_SURFACES)
    Cmpi.barrier()
    #______________________________________________________________________________#
    computeVariablesOnIsosurface(surfaces)
    compute0DPerformances(surfaces, variablesByAverage, var4comp_perf)
    compute1DRadialProfiles(surfaces, variablesByAverage, var4comp_repart)
    computeVariablesOnBladeProfiles(surfaces, allVariables)
    #______________________________________________________________________________#

    for (row1, row2) in stages:
        InletPlane = getSurfaceFromInfo(surfaces, ReferenceRow=row1, tag='InletPlane')
        OutletPlane = getSurfaceFromInfo(surfaces, ReferenceRow=row2, tag='OutletPlane')
        comparePerfoPlane2Plane(InletPlane, OutletPlane, var4comp_perf)

    cleanSurfaces(surfaces, var2keep=var2keep)

    Cmpi.barrier()
    Cmpi.convertPyTree2File(surfaces, FILE_SURFACES_NEW)
    Cmpi.barrier()



if __name__ == '__main__':

    import MOLA.PostprocessTurbo as PostTurbo

    # > USER DATA
    FILE_SETUP        = 'setup.py'
    FILE_FIELDS       = 'fields.cgns'
    FILE_SURFACES     = 'surfaces.cgns'
    FILE_SURFACES_NEW = 'surfaces.cgns'
    DIRECTORY_OUTPUT  = 'OUTPUT'

    #______________________________________________________________________________
    # Variables
    #______________________________________________________________________________
    allVariables = TUS.getFields()
    var4comp_repart = ['StagnationEnthalpyDelta',
                       'StagnationPressureRatio', 'StagnationTemperatureRatio',
                       'StaticPressureRatio', 'Static2StagnationPressureRatio',
                       'IsentropicEfficiency', 'PolytropicEfficiency',
                       'StaticPressureCoefficient', 'StagnationPressureCoefficient',
                       'StagnationPressureLoss1', 'StagnationPressureLoss2',
                       ]
    var4comp_perf = var4comp_repart + ['Power']
    variablesByAverage = PostTurbo.sortVariablesByAverage(allVariables)
    var2keep = [
        'Pressure', 'Temperature', 'PressureStagnation', 'TemperatureStagnation',
        'Entropy',
        'Viscosity_EddyMolecularRatio',
        'VelocitySoundDim', 'StagnationEnthalpyAbsDim',
        'MachNumberAbs', 'MachNumberRel',
        'AlphaAngleDegree',  'BetaAngleDegree', 'PhiAngleDegree',
        'VelocityXAbsDim', 'VelocityRadiusAbsDim', 'VelocityThetaAbsDim',
        'VelocityMeridianDim', 'VelocityRadiusRelDim', 'VelocityThetaRelDim',
        ]
    #______________________________________________________________________________

    PostTurbo.setup = J.load_source('setup', FILE_SETUP)
    surfaces = C.convertFile2PyTree(os.path.join(DIRECTORY_OUTPUT, FILE_SURFACES))
    #______________________________________________________________________________#
    PostTurbo.computeVariablesOnIsosurface(surfaces)
    PostTurbo.compute0DPerformances(surfaces, variablesByAverage, var4comp_perf)
    PostTurbo.compute1DRadialProfiles(surfaces, variablesByAverage, var4comp_repart)
    PostTurbo.computeVariablesOnBladeProfiles(surfaces, allVariables, [0.1, 0.5, 0.9])
    #______________________________________________________________________________#

    InletPlane = PostTurbo.getSurfaceFromInfo(surfaces, ReferenceRow='row_1', tag='InletPlane')
    OutletPlane = PostTurbo.getSurfaceFromInfo(surfaces, ReferenceRow='row_2', tag='OutletPlane')
    PostTurbo.comparePerfoPlane2Plane(InletPlane, OutletPlane, var4comp_perf)

    PostTurbo.cleanSurfaces(surfaces, var2keep=var2keep)

    C.convertPyTree2File(surfaces, os.path.join(DIRECTORY_OUTPUT, FILE_SURFACES_NEW))
