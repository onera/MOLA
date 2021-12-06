#!/usr/bin/python

# Need to source /stck/jmarty/TOOLS/turbo/mySource_dev_etc_py2.me

# Python general packages
import os
import numpy as np
import copy
from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
NProcs = comm.Get_size()

# Cassiopee packages
import Converter.PyTree   as C
import Converter.Mpi      as Cmpi
import Converter.Internal as I
import Post.PyTree        as P
import Transform.PyTree   as T

import MOLA.InternalShortcuts as J

from etc.post        import isoSurfMChH, computeSurfaceAverage, computeMassflowAverage, computeScalarProductSum, IsentropicMachNumber
# Turbo post processing toolbox
from turbo.utils    import load_file, save_file
# from turbo.height   import computeHeightFromMask
# from turbo.slicesAt import sliceAt
from turbo.fields   import computeOtherFields
from turbo.perfos   import computePerformances, comparePerformancesPlane2Plane
from turbo.user     import getFields, getData4Comparison
from turbo.radial_future import computeRadialProfil_future, compareRadialProfilPlane2Plane

I.__FlowSolutionCenters__ = 'FlowSolution#Init'
I.__FlowSolutionNodes__   = 'FlowSolution'

import setup
Gamma = setup.FluidProperties['Gamma']
Rgaz = setup.FluidProperties['IdealConstantGas']
PressureStagnationRef = setup.ReferenceValues['PressureStagnation']
TemperatureStagnationRef = setup.ReferenceValues['TemperatureStagnation']
TurbulenceModel = setup.ReferenceValues['TurbulenceModel']
ShaftRotationSpeed = setup.TurboConfiguration['ShaftRotationSpeed']
TurboConfiguration = setup.TurboConfiguration

class RefState(object):
    def __init__(self):
      self.Gamma = Gamma
      self.Rgaz  = Rgaz
      self.Pio   = PressureStagnationRef
      self.Tio   = TemperatureStagnationRef
      self.roio  = self.Pio / self.Tio / self.Rgaz
      self.aio   = (self.Gamma * self.Rgaz * self.Tio)**0.5
      self.Lref  = 1.

state = RefState()

def computeAeroVariables(surfaces, filename=None, useSI=True, velForm='absolute'):
    new_surfaces = I.newCGNSTree()
    for surface in I.getNodesFromType(surfaces, 'CGNSBase_t'):
        info = getExtractionInfo(surface)
        if info['type'] != 'IsoSurface':
            continue
        surface = C.node2Center(surface, I.__FlowSolutionNodes__)
        surface = computeOtherFields(surface, state, getFields(),
            fsname=I.__FlowSolutionNodes__, useSI=useSI, velocity=velForm,
            specific_tv_dir_name=True)
        surface = computeOtherFields(surface, state, getFields(),
            fsname=I.__FlowSolutionCenters__, useSI=useSI, velocity=velForm,
            specific_tv_dir_name=True)
        surface = C.node2Center(surface, I.__FlowSolutionNodes__) # Because Gamma is only at nodes...
        I._addChild(new_surfaces, surface)
    surfaces = new_surfaces
    if filename:
        C.convertPyTree2File(surfaces, filename)
    surfaces = C.convertArray2NGon(surfaces, recoverBC=0)
    return surfaces

def computePerfosFromPlanes(upstream, downstream, variables):
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

        variables : list
    '''

    var2avg = copy.deepcopy(variables)
    if not 'Gamma' in var2avg:
        var2avg.append('Gamma')

    perfos = dict(massflow=dict(), surface=dict())
    for plane, slices in zip(['upstream','downstream'], [upstream, downstream]):
        ExtractionInfo = I.getNodeFromName(slices, '.ExtractionInfo')
        ReferenceRow = I.getValue(I.getNodeFromName(ExtractionInfo, 'ReferenceRow'))

        nBlades = TurboConfiguration['Rows'][ReferenceRow]['NumberOfBlades']
        nBladesSimu = TurboConfiguration['Rows'][ReferenceRow]['NumberOfBladesSimulated']
        fluxcoeff = nBlades / float(nBladesSimu)
        perfos['massflow'][plane] = computePerformances(slices, plane+'_avg-massflow',
            variables=var2avg,
            average='debit',
            compute_massflow=True,
            fluxcoef=fluxcoeff,
            fsname=I.__FlowSolutionCenters__)

    tAvg = I.merge([perfos['massflow']['upstream'], perfos['massflow']['downstream']])

    tBilan = comparePerformancesPlane2Plane(perfos['massflow']['upstream'],
                                           perfos['massflow']['downstream'],
                                          ['upstream_avg-massflow','downstream_avg-massflow'],
                                          'massflow',
                                          config='compressor',
                                          variables=var4comp)

    tPerfos = I.merge([tAvg, tBilan])

    return tPerfos

def computeRadialProfile(surfaces, fldnames, filename=None):
    radialProfiles = []
    for surface in I.getNodesFromType(surfaces, 'CGNSBase_t'):
        info = getExtractionInfo(surface)
        if info['type'] != 'IsoSurface' or info['field'] != 'CoordinateX':
            continue
        ExtractionInfo = I.getNodeFromName(surface, '.ExtractionInfo')

        for avg in ['surface', 'massflow']:
            var2avg = fldnames[avg]
            if not 'ChannelHeight' in var2avg: var2avg.append('ChannelHeight')
            radialProfile = computeRadialProfil_future(surface,
                '{}_avg-{}'.format(I.getName(surface), avg),
                var2avg,
                avg, config='annular',
                fsname=I.__FlowSolutionCenters__,
                nbband=101, c=0.1)

            zones = I.getZones(radialProfile)
            assert len(zones) == 1
            zone = zones[0]
            ExtractionInfoLocal = I.copyTree(ExtractionInfo)
            I.newDataArray('average', value=avg, parent=ExtractionInfoLocal)
            I._addChild(zone, ExtractionInfoLocal)

            radialProfiles.append(radialProfile)

    radialProfiles = I.merge(radialProfiles)
    I.setName(I.getBases(radialProfiles)[0], 'RadialProfiles')

    if filename:
        C.convertPyTree2File(radialProfiles, filename)

    return radialProfiles

# def radialProfileFromSlices(slices, fldnames_massflow=[], fldnames_surface=[]):
#     '''
#     Compute radial profiles for variables in **fldnames_massflow** and
#     **fldnames_surface**.
#
#     Parameters
#     ----------
#
#         slices : PyTree
#             Base corresponding to the surface to process
#
#         fldnames_massflow : :py:class:`list` of :py:class:`str`
#             Variables to average by massflow
#
#         fldnames_surface : :py:class:`list` of :py:class:`str`
#             Variables to average by surface
#
#     Returns
#     -------
#
#         averages : PyTree
#             Tree with results. The name of the base is the same that **slices**.
#             This base contains two zones:
#
#             * ``Average1D-Surface-`` with surface averaged profiles
#
#             * ``Average1D-Massflow-`` with massflow averaged profiles
#
#     '''
#     if fldnames_massflow == [] and fldnames_surface == []:
#         return None
#
#     extraVariables = ['Radius', 'ChannelHeight', 'Density', 'MomentumX']
#     for var in extraVariables:
#         if var not in fldnames_massflow:
#             fldnames_massflow.append(var)
#         if var not in fldnames_surface:
#             fldnames_surface.append(var)
#     C._initVars(slices, '{Radius}=numpy.sqrt({CoordinateY}**2+{CoordinateZ}**2)')
#
#     ExtractionInfo = I.getNodeFromNameAndType(slices, '.ExtractionInfo', 'UserDefinedData_t')
#     field = I.getValue(I.getNodeFromName(ExtractionInfo, 'field'))
#     value = I.getValue(I.getNodeFromName(ExtractionInfo, 'value'))
#     name = '{}_{}'.format(field, value)
#
#     slices = C.convertArray2NGon(slices, recoverBC=0)
#     slices = C.node2Center(slices, I.__FlowSolutionNodes__)
#     slices = Cmpi.setProc(slices, comm.Get_rank())
#     (averages1DS,averages1DQ) = computeRadialProfil(slices, '',
#         fldnames_massflow=fldnames_massflow, fldnames_surface=fldnames_surface,
#         fsname=I.__FlowSolutionCenters__)
#     averages = I.merge([averages1DS, averages1DQ])
#     base = I.getNodeFromType(averages, 'CGNSBase_t')
#     I.addChild(base, ExtractionInfo)
#     I._renameNode(averages, 'FlowSolution#Centers', I.__FlowSolutionCenters__)
#     I._renameNode(averages, 'FlowSolution', I.__FlowSolutionNodes__)
#     I.setName(base, I.getName(slices))
#     # if comm.Get_rank() == 0:
#     #     C.convertPyTree2File(averages, '{}/radialProfile_{}.cgns'.format(POST_DIR, name))
#     return averages
#
# def radialProfilePerfoFromSlices(radialProfiles, upstream, downstream):
    '''
    Compute radial profiles of performance fields.

    Parameters
    ----------

        radialProfiles : PyTree
            Tree that contains upstream and downstream radial profiles.

        upstream : dict
            parameters to match in ``.ExtractionInfo`` for the upstream surface

        downstream : dict
            Idem than **upstream** but for the downstream surface.

    Returns
    -------

        averages : PyTree
            modified tree with results. A new base is added, with the zone
            ``Average1D-Massflow-`` inside. This zone contains profiles of
            total pressure ratio, total temperature ratio, isentropic efficiency,
            total enthapy increase and phi and psi coefficients.
            An ``.ExtractionInfo`` node is also added.

    '''
    repartUps = getBaseFromExtractionInfo(radialProfiles, **upstream)
    repartDowns = getBaseFromExtractionInfo(radialProfiles, **downstream)
    if not repartUps:
        raise Exception('Cannot find the upstream base: {}'.format(upstream))
    if not repartDowns:
        raise Exception('Cannot find the downstream base: {}'.format(downstream))
    repartUps = I.copyTree(repartUps)
    repartDowns = I.copyTree(repartDowns)

    I._rmNodesByName(repartUps, 'Average1D-Surface-*')
    I._rmNodesByName(repartDowns, 'Average1D-Surface-*')
    I._renameNode(repartUps,'PressureStagnation','StagnationPressureAbsDim')
    I._renameNode(repartUps,'TemperatureStagnation','StagnationTemperatureAbsDim')
    I._renameNode(repartDowns,'PressureStagnation','StagnationPressureAbsDim')
    I._renameNode(repartDowns,'TemperatureStagnation','StagnationTemperatureAbsDim')
    avg_isEff = computeIsEff(repartUps, repartDowns, state)
    # total enthapy increase, phi and psi coefficients
    eqn_dh  = '{{centers:deltaHt}}={0}*{1}/({0}-1)*{{centers:StagnationTemperatureAbsDim}}*({{centers:RTI}}-1)'.format(Gamma, Rgaz)
    eqn_phi = '{{centers:phiCoeff}}={{centers:MomentumX}}/{{centers:Density}}/{{centers:Radius}}/{0}'.format(ShaftRotationSpeed)
    eqn_psi = '{{centers:psiCoeff}}={{centers:deltaHt}}/({{centers:Radius}}*{0})**2'.format(ShaftRotationSpeed)
    for eqn in [eqn_dh, eqn_phi, eqn_psi]:
        C._initVars(avg_isEff, eqn)

    I._renameNode(avg_isEff,'StagnationPressureAbsDim', 'PressureStagnation')
    I._renameNode(avg_isEff,'StagnationTemperatureAbsDim','TemperatureStagnation')
    I._renameNode(avg_isEff,'RPI', 'PressureStagnationRatio')
    I._renameNode(avg_isEff,'RTI', 'TemperatureStagnationRatio')
    I._renameNode(avg_isEff,'eta', 'EfficiencyIsentropic')

    # Change base name
    name1 = I.getName(repartUps)
    name2 = I.getName(repartDowns)
    base = I.getNodeFromType(avg_isEff, 'CGNSBase_t')
    I.setName(base, '{}_{}'.format(name1, name2.split('CoordinateX_')[-1]))

    infoUpstream   = getExtractionInfo(repartUps)
    infoDownstream = getExtractionInfo(repartDowns)
    newInfo = dict(type='perfo', field=infoUpstream['field'])
    for key in ['value', 'ReferenceRow', 'tag']:
        if key in infoUpstream:
            newInfo[key+'Upstream'] = infoUpstream[key]
        if key in infoDownstream:
            newInfo[key+'Downstream'] = infoDownstream[key]

    I._rmNodesByName(base, '.ExtractionInfo')
    J.set(base, '.ExtractionInfo', **newInfo)

    I.addChild(radialProfiles, base)

def banner(txt, sep='=', n=80):
    if txt != "": txt = ' {} '.format(txt)
    print('\n'+''.center(n, sep)+'\n'+txt.center(n, sep)+'\n'+''.center(n, sep))

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
    ExtractionInfo = I.getNodeFromNameAndType(surface, '.ExtractionInfo', 'UserDefinedData_t')
    info = dict((I.getName(node), str(I.getValue(node))) for node in I.getChildren(ExtractionInfo))
    return info

def getBaseFromExtractionInfo(surfaces, **kwargs):
    '''
    Inside a top tree **surfaces**, search for the base with a ``.ExtractionInfo``
    matching the requirements in **kwargs**

    Parameters
    ----------

        surfaces : PyTree
            top tree

        kwargs : unwrapped :py:class:`dict`
            parameters required in the ``.ExtractionInfo`` node of the searched
            base

    Returns
    -------

        surface : PyTree or :py:obj:`None`
            base that matches with **kwargs**
    '''
    for surface in I.getNodesFromType1(surfaces, 'CGNSBase_t'):
        info = getExtractionInfo(surface)
        if all([key in info and info[key] == value for key, value in kwargs.items()]):
            return surface
    return None

def getZoneFromExtractionInfo(surfaces, **kwargs):
    '''
    Inside a top tree **surfaces**, search for the zone with a ``.ExtractionInfo``
    matching the requirements in **kwargs**

    Parameters
    ----------

        surfaces : PyTree
            top tree

        kwargs : unwrapped :py:class:`dict`
            parameters required in the ``.ExtractionInfo`` node of the searched
            zone

    Returns
    -------

        surface : PyTree or :py:obj:`None`
            zone that matches with **kwargs**
    '''
    for surface in I.getZones(surfaces):
        info = getExtractionInfo(surface)
        if all([key in info and info[key] == value for key, value in kwargs.items()]):
            return surface
    return None


if __name__ == '__main__':

    # > USER DATA
    POST_DIR = 'OUTPUT'

    surfaces = C.convertFile2PyTree('OUTPUT/surfaces.cgns')

    surfaces = computeAeroVariables(surfaces, os.path.join(POST_DIR, 'surfaces_computed.cgns'))

    banner('PERFOS')

    variables=['StagnationPressureAbsDim','StagnationTemperatureAbsDim',
               'StaticPressureDim','StaticTemperatureDim',
               'StagnationEnthalpyAbsDim','StaticEnthalpyDim']

    var4comp = [
                'StagnationPressureRatio',
                'StagnationTemperatureRatio',
                'StaticPressureRatio',
                'Static2StagnationPressureRatio',
                'IsentropicEfficiency',
                'PolytropicEfficiency',
                'StagnationEnthalpyDelta',
                'Power',
                'StaticPressureDelta',
                'StagnationPressureDelta',
                'StagnationPressureLoss1',
                'StagnationPressureLoss2',
                ] # cf. getData4Comparison(config='compressor').keys()
    upstream = getBaseFromExtractionInfo(surfaces, ReferenceRow='R37', tag='InletPlane')
    downstream = getBaseFromExtractionInfo(surfaces, ReferenceRow='R37', tag='OutletPlane')
    tPerfos = computePerfosFromPlanes(upstream, downstream, variables=variables)

    fldnames_massflow = ['StagnationPressureAbsDim','StagnationTemperatureAbsDim',
               'StaticPressureDim','StaticTemperatureDim',
               'StagnationEnthalpyAbsDim','StaticEnthalpyDim',
               'EntropyDim', 'Viscosity_EddyMolecularRatio',
               'VelocityMeridianDim', 'VelocityThetaRelDim', 'VelocityThetaAbsDim',
               'MachNumberRel', 'MachNumberAbs',
               'AlphaAngleDegree', 'BetaAngleDegree', 'PhiAngleDegree',
               'Gamma']

    banner('RADIAL PROFILES')
    fldnames = dict(
        massflow = fldnames_massflow,
        surface  = fldnames_massflow
        )

    radialProfiles = computeRadialProfile(surfaces, fldnames)

    var4comp_repart = [
                       'StagnationPressureRatio',
                       'StagnationTemperatureRatio',
                       'StaticPressureRatio',
                       'Static2StagnationPressureRatio',
                       'IsentropicEfficiency',
                       'PolytropicEfficiency',
                       'StagnationEnthalpyDelta',
                       'StaticPressureDelta',
                       'StagnationPressureDelta',
                       'StagnationPressureLoss1',
                       'StagnationPressureLoss2',
                       ]

    avg = 'massflow'
    repartUps = getZoneFromExtractionInfo(radialProfiles, ReferenceRow='R37',  tag='InletPlane', average=avg)
    repartDowns = getZoneFromExtractionInfo(radialProfiles,ReferenceRow='R37', tag='OutletPlane', average=avg)
    compareRadialProfile = compareRadialProfilPlane2Plane(repartUps, repartDowns,
                             [I.getName(repartUps), I.getName(repartDowns)],
                             avg,
                             config='compressor',
                             variables=var4comp_repart)

    t = I.merge([tPerfos, radialProfiles, compareRadialProfile])
    Cmpi.barrier()
    C.convertPyTree2File(t, os.path.join(POST_DIR, 'all_post.cgns'))
