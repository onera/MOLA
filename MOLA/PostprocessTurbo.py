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

#!/usr/bin/python
import MOLA

import MOLA.InternalShortcuts as J
if not MOLA.__ONLY_DOC__:
    # Python general packages
    import os

    # Cassiopee packages
    import Converter.PyTree   as C
    import Converter.Internal as I
    import Converter.Mpi      as Cmpi
    import Post.PyTree        as P
    import Transform.PyTree   as T


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

try:
    setup = J.load_source('setup', 'setup.py')
except:
    setup = None
    
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

        dict
            dictionary with the template:

            >>> info[nodeName] = nodeValue

    '''
    ExtractionInfo = I.getNodeFromName2(surface, '.ExtractionInfo')
    if not ExtractionInfo: 
        return dict()
    else:
        return dict((I.getName(node), str(I.getValue(node))) for node in I.getChildren(ExtractionInfo))


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
            if not FSnodes: continue
            for node in I.getNodesFromType(FSnodes, 'DataArray_t'):
                varname = I.getName(node)
                if varname not in var2keepOnBlade:
                    I._rmNode(FSnodes, node)

# @J.mute_stdout
def computeVariablesOnIsosurface(surfaces, variables, config='annular', lin_axis='XZ'):
    '''
    Compute extra variables for all isoSurfaces, using **turbo** function `_computeOtherFields`.

    Parameters
    ----------
    
        surfaces : PyTree
            as produced by :py:func:`extractSurfaces`
    '''
    surfacesIso = getSurfacesFromInfo(surfaces, type='IsoSurface')
    varAtNodes = None
    prev_fs_vertex = I.__FlowSolutionNodes__
    I.__FlowSolutionNodes__ = 'FlowSolution'
    for surface in surfacesIso:
        firstZone = I.getNodeFromType1(surface, 'Zone_t')
        if firstZone:
            varAtNodes = C.getVarNames(firstZone, loc='nodes')[0]
            break

    if not varAtNodes:
        # There is no zone in any iso-surface on this proc
        # Caution: cannot do a return here, because it seems to be a barrier hidden inside _computeOtherFields
        variables = []
    else:
        for v in varAtNodes: C._node2Center__(surfacesIso, v)
    
    for surface in surfacesIso:
        for fsname in [I.__FlowSolutionNodes__, I.__FlowSolutionCenters__]:
            # BUG https://gitlab.onera.net/numerics/analysis/turbo/-/issues/1
            TF._computeOtherFields(surface, RefState(setup), variables,
                                        fsname=fsname, useSI=True, velocity='absolute',
                                        config=config, lin_axis=lin_axis) # FIXME: to be adapted if user can perform relative computation (vel_formulation)

    I.__FlowSolutionNodes__ = prev_fs_vertex

def compute0DPerformances(surfaces, variablesByAverage):
    '''
    Compute averaged values for all variables for all iso-X surfaces

    Parameters
    ----------
    
        surfaces : PyTree
            as produced by :py:func:`extractSurfaces`

        variablesByAverage : dict
            Lists of variables sorted by type of average (as produced by :py:func:`sortVariablesByAverage`)

    '''
    surfacesToProcess = getSurfacesFromInfo(surfaces, type='IsoSurface', field='CoordinateX')
    # Add eventual non axial InletPlanes or OutletPlanes for centrifugal configurations
    InletPlanes = getSurfacesFromInfo(surfaces, type='IsoSurface', tag='InletPlane')
    OutletPlanes = getSurfacesFromInfo(surfaces, type='IsoSurface', tag='OutletPlane')
    surfacesToProcessNames = [I.getName(surf) for surf in surfacesToProcess]
    for plane in InletPlanes + OutletPlanes:
        if I.getName(plane) not in surfacesToProcessNames:
            surfacesToProcess.append(plane)

    def getFluxCoeff(surface):
        RowFamilies = []
        for Family in I.getNodesFromType1(surface, 'Family_t'):
            # if I.getNodeFromName1(Family, '.Solver#Motion'):
            #     RowFamilies.append(I.getName(Family))
            if not I.getNodeFromType1(Family, 'FamilyBC_t'):
                RowFamilies.append(I.getName(Family))
        if len(RowFamilies) == 0:
            raise Exception(f'There is no zone family detected in {I.getName(surface)}')
        elif len(RowFamilies) > 1:
            raise Exception(f'There are more than 1 zone family in {I.getName(surface)}')
        ReferenceRow = RowFamilies[0]
        try:
            nBlades = setup.TurboConfiguration['Rows'][ReferenceRow]['NumberOfBlades']
            nBladesSimu = setup.TurboConfiguration['Rows'][ReferenceRow]['NumberOfBladesSimulated']
            fluxcoeff = nBlades / float(nBladesSimu)
        except:
            # Linear cascade with a periodicity by translation
            fluxcoeff = 1.
        return fluxcoeff

    AveragesName = 'Averages0D'+I.__FlowSolutionNodes__.replace('FlowSolution','')
    Averages = I.newCGNSBase(AveragesName, cellDim=0, physDim=3, parent=surfaces)

    for surface in surfacesToProcess:
        surfaceName = I.getName(surface)
        fluxcoeff = getFluxCoeff(surface)
        info = getExtractionInfo(surface)

        perfTreeMassflow = TP.computePerformances(surface, surfaceName,
                                                  variables=variablesByAverage['massflow'], average='massflow',
                                                  compute_massflow=False, fluxcoef=fluxcoeff, fsname=I.__FlowSolutionCenters__)
        perfTreeSurface = TP.computePerformances(surface, surfaceName,
                                                 variables=variablesByAverage['surface'], average='surface',
                                                 compute_massflow=True, fluxcoef=fluxcoeff, fsname=I.__FlowSolutionCenters__)

        perfos = I.merge([perfTreeMassflow, perfTreeSurface])
        perfos = I.getNodeFromType2(perfos, 'Zone_t')
        PostprocessInfo = {'averageType': variablesByAverage,
                           'surfaceName': surfaceName,
                           '.ExtractionInfo': info
                           }
        J.set(perfos, '.PostprocessInfo', **PostprocessInfo)
        I.addChild(Averages, perfos)


def comparePerfoPlane2Plane(surfaces, var4comp_perf, stages=[]):
    '''
    Compare averaged values between the **InletPlane** and the **OutletPlane**.

    Parameters
    ----------

        surfaces : PyTree
            as produced by :py:func:`extractSurfaces`
    
        var4comp_perf : list 
            Names of variables to compare between planes tagged with 'InletPlane' and 'OutletPlane'.
        
        stages : :py:class:`list` of :py:class:`tuple`, optional
            List of row stages, of the form:

            >>> stages = [('rotor1', 'stator1'), ('rotor2', 'stator2')] 

            For each tuple of rows, the inlet plane of row 1 is compared with the outlet plane of row 2.

    '''
    avg_name = 'Averages0D'+I.__FlowSolutionNodes__.replace('FlowSolution','')
    Averages0D = I.getNodeFromName1(surfaces, avg_name)

    for row in setup.TurboConfiguration['Rows']:
        if (row, row) not in stages:
            stages.append((row, row))
    
    for (row1, row2) in stages:
        InletPlane = getSurfaceFromInfo(Averages0D, ReferenceRow=row1, tag='InletPlane')
        OutletPlane = getSurfaceFromInfo(Averages0D, ReferenceRow=row2, tag='OutletPlane')
        if not(InletPlane and OutletPlane): 
            continue

        tBudget = TP.comparePerformancesPlane2Plane(InletPlane, OutletPlane,
                                                    [I.getName(InletPlane), I.getName(OutletPlane)],
                                                    f'Comparison',
                                                    fsname=I.__FlowSolutionNodes__,
                                                    config='compressor', variables=var4comp_perf)
            
        fsBudget = I.getNodeFromType(tBudget, 'FlowSolution_t')
        I.createUniqueChild(fsBudget, 'GridLocation', 'GridLocation_t', 'CellCenter', pos=0)
        I.setName(fsBudget, f'Comparison#{I.getName(InletPlane)}')
        I.addChild(OutletPlane, fsBudget)


def compute1DRadialProfiles(surfaces, variablesByAverage, config='annular', lin_axis='XY'):
    '''
    Compute radial profiles for all iso-X surfaces

    Parameters
    ----------
    
        surfaces : PyTree
            as produced by :py:func:`extractSurfaces`

        variablesByAverage : dict
            Lists of variables sorted by type of average (as produced by :py:func:`sortVariablesByAverage`)

        config : str
            ‘annular’ or ‘linear’ configuration

        lin_axis : str
            For ‘linear’ configuration, streamwise and spanwise directions. 
            ‘XZ’ means: streamwise = X-axis, spanwise = Z-axis

    '''
    RadialProfiles = I.getNodeFromName1(surfaces,'RadialProfiles')
    if not RadialProfiles:
        RadialProfiles = I.newCGNSBase('RadialProfiles', cellDim=1, physDim=3,
                                        parent=surfaces)
    surfacesIsoX = getSurfacesFromInfo(surfaces, type='IsoSurface', field='CoordinateX')

    for surface in surfacesIsoX:
        surfaceName = I.getName(surface)
        tmp_surface = C.convertArray2NGon(surface, recoverBC=0)
        radial_surf = TR.computeRadialProfile_future(
            tmp_surface, surfaceName, variablesByAverage['surface'], 'surface',
            fsname=I.__FlowSolutionCenters__, config=config, lin_axis=lin_axis)
        radial_massflow = TR.computeRadialProfile_future(
            tmp_surface, surfaceName, variablesByAverage['massflow'], 'massflow',
            fsname=I.__FlowSolutionCenters__, config=config, lin_axis=lin_axis)
        t_radial = I.merge([radial_surf, radial_massflow])
        z_radial = I.getNodeFromType2(t_radial, 'Zone_t')
        previous_z_radial = I.getNodeFromName1(RadialProfiles, z_radial[0])
        if not previous_z_radial:
            PostprocessInfo = {'averageType': variablesByAverage, 
                                'surfaceName': surfaceName,
                                '.ExtractionInfo': getExtractionInfo(surface)
                                }
            J.set(z_radial, '.PostprocessInfo', **PostprocessInfo)
            I.addChild(RadialProfiles, z_radial)
        else:
            flowSolsToAdd = I.getNodesFromType1(z_radial, 'FlowSolution_t')
            flowSolsToAddNames = [n[0] for n in flowSolsToAdd]
            flowSolsPrev = I.getNodesFromType1(previous_z_radial, 'FlowSolution_t')
            flowSolsPrevNames = [n[0] for n in flowSolsPrev]
            for fs, fsn in zip(flowSolsToAdd, flowSolsToAddNames):
                if fsn not in flowSolsPrevNames:
                    previous_z_radial[2] += [fs]
                


def compareRadialProfilesPlane2Plane(surfaces, var4comp_repart, stages=[], config='compressor'):
    '''
    Compare radial profiles between the **InletPlane** and the **OutletPlane**.

    Parameters
    ----------
    
        surfaces : PyTree
            as produced by :py:func:`extractSurfaces`
    
        var4comp_repart : list 
            Names of variables to compare between planes tagged with 'InletPlane' and 'OutletPlane'.
        
        stages : :py:class:`list` of :py:class:`tuple`, optional
            List of row stages, of the form:

            >>> stages = [('rotor1', 'stator1'), ('rotor2', 'stator2')] 

            For each tuple of rows, the inlet plane of row 1 is compared with the outlet plane of row 2.
        
        config : str
            Must be ‘compressor’ or ‘turbine’. Useful to compute efficency.
    '''
    RadialProfiles = I.getNodeFromName1(surfaces, 'RadialProfiles')

    for row in setup.TurboConfiguration['Rows']:
        if (row, row) not in stages:
            stages.append((row, row))

    for (row1, row2) in stages:
        InletPlane = getSurfaceFromInfo(RadialProfiles, ReferenceRow=row1, tag='InletPlane')
        OutletPlane = getSurfaceFromInfo(RadialProfiles, ReferenceRow=row2, tag='OutletPlane')
        
        if not(InletPlane and OutletPlane): continue

        fsname = I.getNodeFromType(InletPlane, 'FlowSolution_t')[0]

        extractionInfoInlet = getExtractionInfo(InletPlane)
        extractionInfoOutlet = getExtractionInfo(OutletPlane)
        if extractionInfoInlet['field'] == 'CoordinateX' and extractionInfoOutlet['field'] == 'CoordinateX':
            tBudget = TR.compareRadialProfilePlane2Plane(InletPlane, OutletPlane,
                                                        [I.getName(InletPlane), I.getName(OutletPlane)],
                                                        f'Comparison',
                                                        config=config,
                                                        fsname=fsname,
                                                        variables=var4comp_repart)
            zBudget = I.getNodeFromType3(tBudget,'Zone_t')
            fsBudget = I.getNodeFromType(zBudget, 'FlowSolution_t')
            I.createUniqueChild(fsBudget, 'GridLocation', 'GridLocation_t', 'CellCenter', pos=0)
            I.setName(fsBudget, f'Comparison#{I.getName(InletPlane)}')
            I.addChild(OutletPlane, fsBudget)


def computeVariablesOnBladeProfiles(surfaces, hList='all'):
    '''
    Make height-constant slices on the blades to compute the isentropic Mach number and other
    variables at blade wall.

    Parameters
    ----------
    
        surfaces : PyTree
            as produced by :py:func:`extractSurfaces`

        hList : list or str, optional
            List of heights to make slices on blades. 
            If 'all' (by default), the list is got by taking the values of the existing 
            iso-height surfaces in the input tree.
    '''

    def searchBladeInTree(row):
        famnames = ['*BLADE*'.format(row), '*Blade*'.format(row),
                    '*AUBE*'.format(row), '*Aube*'.format(row)]
        for famname in famnames:
            for bladeSurface in I.getNodesFromNameAndType(surfaces, famname, 'CGNSBase_t'):
                if I.getNodeFromNameAndType(bladeSurface, row, 'Family_t') and I.getZones(bladeSurface) != []:
                    return bladeSurface
        return

    if hList == 'all':
        hList = []
        surfacesIsoH = getSurfacesFromInfo(surfaces, type='IsoSurface', field='ChannelHeight')
        for surface in surfacesIsoH:
            ExtractionInfo = I.getNodeFromName(surface, '.ExtractionInfo')
            valueH = I.getValue(I.getNodeFromName(ExtractionInfo, 'value'))
            hList.append(valueH)
        
    RadialProfiles = I.getNodeByName1(surfaces, 'RadialProfiles')

    for row in setup.TurboConfiguration['Rows']:

        InletPlane = getSurfaceFromInfo(RadialProfiles, ReferenceRow=row, tag='InletPlane')
        if not InletPlane:
            continue

        blade = searchBladeInTree(row)
        if not blade:
            print(f'No blade family (or more than one) has been found for row {row}')
            continue

        I._renameNode(blade, 'FlowSolution#Centers', I.__FlowSolutionCenters__)
        C._initVars(blade, 'Radius=sqrt({CoordinateY}**2+{CoordinateZ}**2)')
        blade = C.center2Node(blade, I.__FlowSolutionCenters__)
        C._initVars(blade, 'StaticPressureDim={Pressure}')

        blade = TMis.computeIsentropicMachNumber(InletPlane, blade, RefState(setup))

        BladeSlices = I.newCGNSBase(f'{row}_Slices', cellDim=1, physDim=3, parent=surfaces)
        for h in hList:
            bladeIsoH = T.join(P.isoSurfMC(blade, 'ChannelHeight', h))
            # bladeIsoH = P.isoSurfMC(blade, 'ChannelHeight', h)
            I.setName(bladeIsoH, 'Iso_H_{}'.format(h))
            I._addChild(BladeSlices, bladeIsoH)

