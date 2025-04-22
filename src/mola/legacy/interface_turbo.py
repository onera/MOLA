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

'''
Creation by recycling PostprocessTurbo.py of v1.18.1
'''

from treelab import cgns
from mola.naming_conventions import CGNS_NODE_EXTRACTION_LOG

# Cassiopee packages
import Converter.PyTree   as C
import Converter.Internal as I
import Converter.Mpi      as Cmpi
import Post.PyTree        as P
import Transform.PyTree   as T

########################
import turbo.user     as TUS
import turbo.fields   as TF
import turbo.perfos   as TP
import turbo.radial   as TR
import turbo.machis   as TMis

RADIAL_PROFILES_BASE = 'RadialProfiles'
AVERAGES_0D_BASE = 'Averages0D'

class RefState():

    def __init__(self, w):
        self.Gamma = w.Fluid['Gamma']
        self.Rgaz  = w.Fluid['IdealGasConstant']
        self.Pio   = w.Flow['PressureStagnation']
        self.Tio   = w.Flow['TemperatureStagnation']
        self.roio  = self.Pio / self.Tio / self.Rgaz
        self.aio   = (self.Gamma * self.Rgaz * self.Tio)**0.5
        self.Lref  = 1.

def postprocess_turbomachinery(w, surfaces, signals, stages=[], 
                                var4comp_repart=None, var4comp_perf=None, var2keep=None, 
                                computeRadialProfiles=True, 
                                heightListForIsentropicMach='all',
                                config='annular', 
                                lin_axis='XY',
                                RowType='compressor',
                                container_at_vertex='FlowSolution#EndOfRunV'):
    '''
    Perform a series of classical postprocessings for a turbomachinery case : 

    #. Compute extra variables, in relative and absolute frames of reference

    #. Compute averaged values for all iso-X planes, and
       compare inlet and outlet planes for each row if available, to get row performance (total 
       pressure ratio, isentropic efficiency, etc) 

    #. Compute radial profiles for all iso-X planes (results are in the `RadialProfiles` base), and
       compare inlet and outlet planes for each row if available, to get row performance (total 
       pressure ratio, isentropic efficiency, etc) 

    #. Compute isentropic Mach number on blades, slicing at constant height, for all values of height 
       already extracted as iso-surfaces.

    Parameters
    ----------

        surfaces : PyTree
            extracted surfaces

        stages : :py:class:`list` of :py:class:`tuple`, optional
            List of row stages, of the form:

            >>> stages = [('rotor1', 'stator1'), ('rotor2', 'stator2')] 

            For each tuple of rows, the inlet plane of row 1 is compared with the outlet plane of row 2.

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
                    'StagnationPressureRelDim', 'StagnationTemperatureRelDim',
                    'Entropy',
                    'Viscosity_EddyMolecularRatio',
                    'VelocitySoundDim', 'StagnationEnthalpyAbsDim',
                    'MachNumberAbs', 'MachNumberRel',
                    'AlphaAngleDegree',  'BetaAngleDegree', 'PhiAngleDegree',
                    'VelocityXAbsDim', 'VelocityRadiusAbsDim', 'VelocityThetaAbsDim',
                    'VelocityMeridianDim', 'VelocityRadiusRelDim', 'VelocityThetaRelDim',
                    ]
        
        computeRadialProfiles : bool
            Choose or not to compute radial profiles.
        
        heightListForIsentropicMach : list or str, optional
            List of heights to make slices on blades. 
            If 'all' (by default), the list is got by taking the values of the existing 
            iso-height surfaces in the input tree.
        
        config : str
            see :py:func:`MOLA.PostprocessTurbo.compute1DRadialProfiles`

        lin_axis : str
            see :py:func:`MOLA.PostprocessTurbo.compute1DRadialProfiles`

        RowType : str
            see parameter 'config' of :py:func:`MOLA.PostprocessTurbo.compareRadialProfilesPlane2Plane`
        
        container_at_vertex : :py:class:`str` or :py:class:`list` of :py:class:`str`
            specifies the *FlowSolution* container located at 
            vertex where postprocess will be applied. 

            .. hint::
                provide a :py:class:`list` of :py:class:`str` so that the 
                postprocess will be applied to each of the provided containers.
                This is useful for making post-processing on e.g. both
                instantaneous and averaged flow fields
        
    '''
    # NOTE BE CAREFUL!!! This function needs to be run on all ranks, 
    # and MPI handling is done internally

    # prepare auxiliary surfaces tree, with flattened FlowSolution container
    # located at Vertex including ChannelHeight
    previous_vertex_container = I.__FlowSolutionNodes__
    turbo_required_vertex_container = 'FlowSolution'
    turbo_new_centers_container = 'FlowSolution#Centers'

    if isinstance(container_at_vertex, str):
        containers_at_vertex = [container_at_vertex]
    elif not isinstance(container_at_vertex, list):
        raise TypeError('container_at_vertex must be str or list of str')
    else:
        containers_at_vertex = container_at_vertex

    suffixes = [c.replace('FlowSolution','') for c in containers_at_vertex]

    for container_at_vertex in containers_at_vertex:
        I.__FlowSolutionNodes__ = container_at_vertex
        channel_height = None
        for zone in I.getZones(surfaces):
            fs_container = I.getNodeFromName1(zone, container_at_vertex)
            if not fs_container: 
                continue
            fs_container[0] = turbo_required_vertex_container

            channel_height = I.getNodeFromName2(zone, 'ChannelHeight')
            if not channel_height: 
                continue
            fs_container[2] += [ channel_height ]

        #______________________________________________________________________________
        # Variables
        #______________________________________________________________________________
        allVariables = TUS.getFields(config=config)
        if not channel_height and 'ChannelHeight' in allVariables:
            allVariables.remove('ChannelHeight')
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
                'StagnationPressureRelDim', 'StagnationTemperatureRelDim',
                'Entropy',
                'Viscosity_EddyMolecularRatio',
                'VelocitySoundDim', 'StagnationEnthalpyAbsDim',
                'MachNumberAbs', 'MachNumberRel',
                'AlphaAngleDegree',  'BetaAngleDegree', 'PhiAngleDegree',
                'VelocityXAbsDim', 'VelocityRadiusAbsDim', 'VelocityThetaAbsDim',
                'VelocityMeridianDim', 'VelocityRadiusRelDim', 'VelocityThetaRelDim',
            ]

        variablesByAverage = sortVariablesByAverage(allVariables)

        # COMPUTE ON A SINGLE SURFACE _______________________________________________________________#
        computeVariablesOnIsosurface(w, surfaces, allVariables, config=config, lin_axis=lin_axis)
        compute0DPerformances(w, surfaces, variablesByAverage)
        if computeRadialProfiles: 
            compute1DRadialProfiles(
                surfaces, variablesByAverage, config=config, lin_axis=lin_axis)
        # if config == 'annular' and heightListForIsentropicMach:
        #     # TODO compute Machis also for linear cascade. Is this available in turbo ? 
        #     computeVariablesOnBladeProfiles(w, surfaces, height_list=heightListForIsentropicMach)
        
        # COMPUTE BY COMPARING TWO SURFACES _________________________________________________________#
        move_0D_and_1D_data_to_rank_0(surfaces)
        if Cmpi.rank == 0:
            comparePerfoPlane2Plane(w, surfaces, var4comp_perf, stages)
            if computeRadialProfiles: 
                if I.getNodeFromName(surfaces, 'ChannelHeight'):
                    compareRadialProfilesPlane2Plane(
                        w, surfaces, var4comp_repart, stages, config=RowType)

        #____________________________________________________________________________________________#
        cleanSurfaces(w, surfaces, var2keep=var2keep)

        suffix = container_at_vertex.replace('FlowSolution','')
        for zone in I.getZones(surfaces):
            for fs_container in I.getNodesFromType1(zone, 'FlowSolution_t'):
                fs_name = fs_container[0]
                is_turbo_container = fs_name in [turbo_required_vertex_container,
                                                turbo_new_centers_container]
                is_new_comparison = fs_name.startswith('Comparison') and not \
                                    fs_name.endswith(suffix)

                if is_turbo_container or is_new_comparison: 
                    if not any([fs_container[0].endswith(s) for s in suffixes]):
                        fs_container[0] += suffix
                        if fs_container[0].startswith(turbo_new_centers_container):
                            fs_container[0]=fs_container[0].replace(turbo_new_centers_container,
                                                                    'FlowSolution')

        I.__FlowSolutionNodes__ = previous_vertex_container

    surfaces = cgns.castNode(surfaces)

    if Cmpi.rank == 0:
        move_scalar_outputs_to_signals(surfaces, signals)

    return surfaces, signals

def move_scalar_outputs_to_signals(surfaces: cgns.Tree, signals: cgns.Tree) -> None:
    # TODO Add IterationNumber in signals, and update bases to concatenate data for each iteration
    base = surfaces.get(Name=AVERAGES_0D_BASE, Type='CGNSBase', Depth=1)
    if base:
        base.dettach()
        base.attachTo(signals)

def move_0D_and_1D_data_to_rank_0(surfaces):
    Cmpi.barrier()
    for base_name in [RADIAL_PROFILES_BASE, AVERAGES_0D_BASE]:
        base = I.getNodeFromName1(surfaces, base_name)
        bases = Cmpi.gather(base, root=0)
        if Cmpi.rank == 0:
            base_gathered = I.merge(bases)
            I._rmNode(surfaces, base)
            I._addChild(surfaces, base_gathered)
        else:
            I._rmNode(surfaces, base)
    Cmpi.barrier()

def getExtractionInfo(surface):
    '''
    Get information into :mola_name:`CGNS_NODE_EXTRACTION_LOG` of **surface**.

    Parameters
    ----------

        surface : PyTree
            Base corresponding to a surface, with a :mola_name:`CGNS_NODE_EXTRACTION_LOG` node

    Returns
    -------

        dict
            dictionary with the template:

            >>> info[nodeName] = nodeValue

    '''
    surface = cgns.castNode(surface)
    try:
        return surface.getParameters(CGNS_NODE_EXTRACTION_LOG)
    except ValueError:
        return dict()

def getSurfacesFromInfo(surfaces, breakAtFirst=False, **kwargs):
    '''
    Inside a top tree **surfaces**, search for the nodes with a :mola_name:`CGNS_NODE_EXTRACTION_LOG`
    matching the requirements in **kwargs**

    Parameters
    ----------

        surfaces : PyTree
            top tree or base

        kwargs : unwrapped :py:class:`dict`
            parameters required in the :mola_name:`CGNS_NODE_EXTRACTION_LOG` node of the searched
            zone

    Returns
    -------

        extractedSurfaces : list
            bases that matches with **kwargs**
    '''
    def _flatten_dict(d):
        res = dict()
        for key, value in d.items():
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    res[key2] = str(value2)
            else:
                res[key] = str(value)
        return res
    
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
        info = _flatten_dict(info)
        if all([key in info and info[key] == value for key, value in kwargs.items()]):
            extractedSurfaces.append(surface)
            if breakAtFirst:
                break
    return extractedSurfaces

def getSurfaceFromInfo(surfaces, **kwargs):
    '''
    Inside a top tree **surfaces**, search for the node with a :mola_name:`CGNS_NODE_EXTRACTION_LOG`
    matching the requirements in **kwargs**

    Parameters
    ----------

        surfaces : PyTree
            top tree or base

        kwargs : unwrapped :py:class:`dict`
            parameters required in the :mola_name:`CGNS_NODE_EXTRACTION_LOG` node of the searched
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

def cleanSurfaces(w, surfaces, var2keep=[]):
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
    conservatives = list(w.Flow['Conservatives'])
    var2keepOnBlade = ['ChannelHeight', 'IsentropicMachNumber',
        'Pressure', 'StagnationPressureRelDim', 'RefStagnationPressureRelDim',
        'SkinFrictionX', 'SkinFrictionY', 'SkinFrictionZ'
        ]
    var2keepOnRadialProfiles = conservatives + var2keep + ['ChannelHeight', 'Radius']
    var2keepOnRadialProfiles += [f'centers:{var}' for var in var2keepOnRadialProfiles]

    surfacesIso = getSurfacesFromInfo(surfaces, Type='IsoSurface')
    for surface in surfacesIso:
        for zone in I.getZones(surface):
            I._rmNodesByName1(zone, I.__FlowSolutionCenters__)
            C._extractVars(zone, coordinates+conservatives+var2keep)

    radProfiles = I.getNodeFromName1(surfaces, RADIAL_PROFILES_BASE)  
    if radProfiles:
        C._extractVars(radProfiles, coordinates+var2keepOnRadialProfiles)

    surfacesBC = getSurfacesFromInfo(surfaces, Type='BC', BCType='BCWallViscous')
    for surface in surfacesBC:
        for zone in I.getZones(surface):
            FSnodes = I.getNodeFromName1(zone, I.__FlowSolutionNodes__)
            if not FSnodes: continue
            for node in I.getNodesFromType(FSnodes, 'DataArray_t'):
                varname = I.getName(node)
                if varname not in var2keepOnBlade:
                    I._rmNode(FSnodes, node)

def computeVariablesOnIsosurface(w, surfaces, variables, config='annular', lin_axis='XZ'):
    '''
    Compute extra variables for all isoSurfaces, using **turbo** function `_computeOtherFields`.

    Parameters
    ----------
    
        surfaces : PyTree
            as produced by :py:func:`extractSurfaces`
    '''
    surfacesIso = getSurfacesFromInfo(surfaces, Type='IsoSurface')
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
            filtered_variables = TUS.getFilteredFields(surface, variables, fsname=fsname)
            TF._computeOtherFields(surface, RefState(w), filtered_variables,
                                        fsname=fsname, useSI=True, velocity='absolute',
                                        config=config, lin_axis=lin_axis) # FIXME: to be adapted if user can perform relative computation (vel_formulation)

    I.__FlowSolutionNodes__ = prev_fs_vertex

def compute0DPerformances(w, surfaces, variablesByAverage):
    '''
    Compute averaged values for all variables for all iso-X surfaces

    Parameters
    ----------
    
        surfaces : PyTree
            as produced by :py:func:`extractSurfaces`

        variablesByAverage : dict
            Lists of variables sorted by type of average (as produced by :py:func:`sortVariablesByAverage`)

    '''
    Averages = I.getNodeFromName1(surfaces, AVERAGES_0D_BASE)
    if not Averages: 
        Averages = I.newCGNSBase(AVERAGES_0D_BASE, cellDim=0, physDim=3, parent=surfaces)

    surfacesToProcess = getSurfacesFromInfo(surfaces, Type='IsoSurface', IsoSurfaceField='CoordinateX')
    # Add eventual non axial InletPlanes or OutletPlanes for centrifugal configurations
    InletPlanes = getSurfacesFromInfo(surfaces, Type='IsoSurface', tag='InletPlane')
    OutletPlanes = getSurfacesFromInfo(surfaces, Type='IsoSurface', tag='OutletPlane')
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

        ## NOTE Commented below, a new way to get ReferenceRow from zones and not from Family_t nodes
        ## with MPI, but still not working. 
        # ReferenceRow = None
        # for zone in I.getZones(surface):
        #     try:
        #         FamilyName = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
        #         if ReferenceRow is None:
        #             ReferenceRow = FamilyName
        #         else:
        #             assert FamilyName == ReferenceRow, f'There are at least 2 zone families in {I.getName(surface)}: {ReferenceRow} and {FamilyName}'
        #     except TypeError:
        #         continue

        # # Some ranks may not have zones, and for them ReferenceRow=None
        # ReferenceRows_gathered = set(Cmpi.allgather(ReferenceRow))
        # ReferenceRows_gathered.discard(None)  # remove None element if present
        # if len(ReferenceRows_gathered) == 0:
        #     raise Exception(f'There is no zone family detected in {I.getName(surface)}')
        # # check that all remaining values are the same
        # elif len(ReferenceRows_gathered) > 1:
        #     raise Exception(f'There are several zone families in {I.getName(surface)}: {ReferenceRows_gathered}')
        # ReferenceRow = list(ReferenceRows_gathered)[0]
        # Cmpi.barrier()

        try:
            nBlades = w.ApplicationContext['Rows'][ReferenceRow]['NumberOfBlades']
            nBladesSimu = w.ApplicationContext['Rows'][ReferenceRow]['NumberOfBladesSimulated']
            fluxcoeff = nBlades / float(nBladesSimu)
        except:
            # Linear cascade with a periodicity by translation
            fluxcoeff = 1.
        return fluxcoeff

    for surface in surfacesToProcess:
        surfaceName = I.getName(surface)
        fluxcoeff = getFluxCoeff(surface)
        info = getExtractionInfo(surface)

        filtered_variables = TUS.getFilteredFields(surface, variablesByAverage['massflow'], fsname=I.__FlowSolutionCenters__)
        perfTreeMassflow = TP.computePerformances(surface, surfaceName,
                                                  variables=filtered_variables, average='massflow',
                                                  compute_massflow=False, fluxcoef=fluxcoeff, fsname=I.__FlowSolutionCenters__)
        filtered_variables = TUS.getFilteredFields(surface, variablesByAverage['surface'], fsname=I.__FlowSolutionCenters__)
        perfTreeSurface = TP.computePerformances(surface, surfaceName,
                                                 variables=filtered_variables, average='surface',
                                                 compute_massflow=True, fluxcoef=fluxcoeff, fsname=I.__FlowSolutionCenters__)

        perfos = I.merge([perfTreeMassflow, perfTreeSurface])
        perfos = I.getNodeFromType2(perfos, 'Zone_t')
        PostprocessInfo = {'averageType': variablesByAverage,
                           'surfaceName': surfaceName,
                           **info
                           }
        perfos = cgns.castNode(perfos)
        perfos.setParameters(CGNS_NODE_EXTRACTION_LOG, **PostprocessInfo)                   
        I.addChild(Averages, perfos)


def comparePerfoPlane2Plane(w, surfaces, var4comp_perf, stages=[]):
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
    Averages0D = I.getNodeFromName1(surfaces, AVERAGES_0D_BASE)

    for row in w.ApplicationContext['Rows']:
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


def compute1DRadialProfiles(surfaces, variablesByAverage, config='annular', lin_axis='XY', NumberOfRadialPoints=121, tipRadius=None):
    '''
    Compute radial profiles for all iso-X surfaces

    Parameters
    ----------
    
        surfaces : PyTree
            as produced by :py:func:`extractSurfaces`

        variablesByAverage : dict
            Lists of variables sorted by type of average (as produced by :py:func:`sortVariablesByAverage`)

        config : str
            ‘annular’ or ‘linear’ or 'oras' configuration

        lin_axis : str
            For ‘linear’ configuration, streamwise and spanwise directions. 
            ‘XZ’ means: streamwise = X-axis, spanwise = Z-axis
        
        NumberOfRadialPoints : int
            Number of radial crowns used to compute radial profile
        
        tipRadius: :py:class:`dict`
            Dictionary providing the value of the blade tip radius (in meters)
            for each row of the considered Open-fan.
            
            .. note::
                Only relevant when using WorkflowORAS.

            .. hint:: for example 
                
                >>>  tipRadius = dict(Rotor = 2.1, Stator = 1.9)

    '''
    RadialProfiles = I.getNodeFromName1(surfaces, RADIAL_PROFILES_BASE)
    if not RadialProfiles:
        RadialProfiles = I.newCGNSBase(RADIAL_PROFILES_BASE, cellDim=1, physDim=3, parent=surfaces)
    surfacesIsoX = getSurfacesFromInfo(surfaces, Type='IsoSurface', IsoSurfaceField='CoordinateX')

    for surface in surfacesIsoX:
        surfaceName = I.getName(surface)
        tmp_surface = C.convertArray2NGon(surface, recoverBC=0)

        if config == 'oras':
            radial_extend = 1.5
            radial_point = 31  #int(NumberOfRadialPoints*(1-1/radial_extend))
            if tipRadius == None:
                radial_dist = TR.defineRadialDistribution4USF(NumberOfRadialPoints, slice4auto=tmp_surface, tip_radius='auto', radial_extend=radial_extend, radial_point=radial_point)
            else:
                row = I.getValue(I.getNodeFromName(surface,'FamilyName'))
                radial_dist = TR.defineRadialDistribution4USF(NumberOfRadialPoints, slice4auto=tmp_surface, tip_radius= tipRadius[row], radial_extend=radial_extend, radial_point=radial_point)
            radial_dist_arr = I.getValue(I.getNodeFromName(radial_dist, 'Radius'))
            # Change config value to pass it to turbo functions
            config = 'axial'
        else:
            radial_dist_arr = None

        filtered_variables = TUS.getFilteredFields(tmp_surface, variablesByAverage['surface'], fsname=I.__FlowSolutionCenters__)
        radial_surf, radius_dist = TR.computeRadialProfile(
            tmp_surface, surfaceName, filtered_variables, 'surface',
            fsname=I.__FlowSolutionCenters__, config=config, lin_axis=lin_axis, 
            save_radius='return', load_radius=radial_dist_arr)

        filtered_variables = TUS.getFilteredFields(tmp_surface, variablesByAverage['massflow'], fsname=I.__FlowSolutionCenters__)
        radial_massflow = TR.computeRadialProfile(
            tmp_surface, surfaceName, filtered_variables, 'massflow',
            fsname=I.__FlowSolutionCenters__, config=config, lin_axis=lin_axis, 
            load_radius=radius_dist)
        
        t_radial = I.merge([radial_surf, radial_massflow])
        z_radial = I.getNodeFromType2(t_radial, 'Zone_t')
        previous_z_radial = I.getNodeFromName1(RadialProfiles, surfaceName)
        if previous_z_radial:
            I._rmNode(RadialProfiles, previous_z_radial)

        PostprocessInfo = {'averageType': variablesByAverage, 
                            'surfaceName': surfaceName,
                            **getExtractionInfo(surface)
                            }
        z_radial = cgns.castNode(z_radial)
        z_radial.setParameters(CGNS_NODE_EXTRACTION_LOG, **PostprocessInfo)   
        I.addChild(RadialProfiles, z_radial)


def compareRadialProfilesPlane2Plane(w, surfaces, var4comp_repart, stages=[], config='compressor'):
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
    RadialProfiles = I.getNodeFromName1(surfaces, RADIAL_PROFILES_BASE)

    for row in w.ApplicationContext['Rows']:
        if (row, row) not in stages:
            stages.append((row, row))

    for (row1, row2) in stages:
        InletPlane = getSurfaceFromInfo(RadialProfiles, ReferenceRow=row1, tag='InletPlane')
        OutletPlane = getSurfaceFromInfo(RadialProfiles, ReferenceRow=row2, tag='OutletPlane')
        
        if not(InletPlane and OutletPlane): continue

        fsname = I.getNodeFromType(InletPlane, 'FlowSolution_t')[0]

        extractionInfoInlet = getExtractionInfo(InletPlane)
        extractionInfoOutlet = getExtractionInfo(OutletPlane)
        if extractionInfoInlet['IsoSurfaceField'] == 'CoordinateX' and extractionInfoOutlet['IsoSurfaceField'] == 'CoordinateX':
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


def computeVariablesOnBladeProfiles(w, surfaces, height_list='all', kind='rotor'):
    '''
    Make height-constant slices on the blades to compute the isentropic Mach number and other
    variables at blade wall.

    Parameters
    ----------
    
        surfaces : PyTree
            as produced by :py:func:`extractSurfaces`

        height_list : list or str, optional
            List of heights to make slices on blades. 
            If 'all' (by default), the list is got by taking the values of the existing 
            iso-height surfaces in the input tree.
    '''

    def searchBladeInTree(row):
        for famname in ['*BLADE*', '*Blade*', '*AUBE*', '*Aube*']:
            for bladeSurface in I.getNodesFromNameAndType(surfaces, famname, 'CGNSBase_t'):
                if I.getNodeFromNameAndType(bladeSurface, row, 'Family_t') and len(I.getZones(bladeSurface)) != 0:
                    return bladeSurface
        return

    if height_list == 'all':
        height_list = []
        surfacesIsoH = getSurfacesFromInfo(surfaces, Type='IsoSurface', IsoSurfaceField='ChannelHeight')
        for surface in surfacesIsoH:
            ExtractionInfo = I.getNodeFromName(surface, CGNS_NODE_EXTRACTION_LOG)
            valueH = I.getValue(I.getNodeFromName(ExtractionInfo, 'value'))
            height_list.append(valueH)
        
    RadialProfiles = I.getNodeByName1(surfaces, RADIAL_PROFILES_BASE)

    for row in w.ApplicationContext['Rows']:

        InletPlane = getSurfaceFromInfo(RadialProfiles, ReferenceRow=row, tag='InletPlane')
        if not InletPlane:
            continue

        blade_ref = searchBladeInTree(row)
        if not blade_ref:
            print(f'No blade family (or more than one) has been found for row {row}')
            continue

        blade = I.copyRef(blade_ref)
        I._renameNode(blade, 'BCDataSet', I.__FlowSolutionCenters__)
        if not I.getNodeFromName(blade, 'Radius'):
            C._initVars(blade, 'Radius=sqrt({CoordinateY}**2+{CoordinateZ}**2)')
        if not I.getNodeFromName(InletPlane, 'Radius'):
            C._initVars(InletPlane, 'Radius=sqrt({CoordinateY}**2+{CoordinateZ}**2)')
        blade = C.center2Node(blade, I.__FlowSolutionCenters__)
        C._initVars(blade, 'StaticPressureDim={Pressure}')

        blade_with_Mis = TMis.computeIsentropicMachNumber(InletPlane, blade, RefState(w), kind=kind, fsname=I.__FlowSolutionNodes__ )

        for zone in I.getZones(blade_with_Mis):
            FS = I.getNodeFromName(zone, I.__FlowSolutionNodes__)
            I.newGridLocation(value='Vertex', parent=FS)
            zone_ref = I.getNodeFromName1(blade_ref, zone[0])
            I.addChild(zone_ref, FS)

        BladeSlices = I.newCGNSBase(f'{I.getName(blade_ref)}_Slices', cellDim=1, physDim=3, parent=surfaces)
        for h in height_list:
            bladeIsoH = T.join(POST.isoSurface(blade_with_Mis, fieldname='ChannelHeight', value=h, container='FlowSolution#Height'))
            if bladeIsoH == []:
                # empty slice
                continue
            I.setName(bladeIsoH, f'Iso_H_{h}')
            I._addChild(BladeSlices, bladeIsoH)

