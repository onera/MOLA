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

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from treelab import cgns
from mola.naming_conventions import DIRECTORY_OUTPUT, FILE_OUTPUT_2D, CGNS_NODE_EXTRACTION_LOG
from mola.logging import mola_logger
import mola.naming_conventions as names

def pretty_name(var):
    '''
    Return a pretty label to write on a plot axis.

    Parameters
    ----------
    var : str
        Name of the variable in the CGNS tree

    Returns
    -------
    str
        Depending on **var**, try to return a pretty name 'Variable [unit]'
    '''
    unit = None
    if any([v in var for v in ['MachNumber', 'Efficiency', 'Ratio', 'Coefficient', 'Loss']]):
        unit = '-'
    elif 'Pressure' in var:
        unit = 'Pa'
    elif 'Velocity' in var:
        unit = 'm/s'
    elif 'AngleDegree' in var:
        unit = 'deg'
    elif 'Enthalpy' in var:
        unit = r'J/kg'
    elif 'Entropy' in var:
        unit = r'J/kg/K'
    elif 'Efficiency' in var:
        unit = r'J.m$^3$/K/kg'
    elif 'Entropy' in var:
        unit = r'J.m$^3$/K/kg'
    remplacements = [
        ('StagnationPressureAbsDim', '$P_{t,abs}$'),
        ('StagnationTemperatureAbsDim', '$T_{t,abs}$'),
        ('StagnationEnthalpyAbsDim', '$h_{t,abs}$'),
        ('StagnationPressureRelDim', '$P_{t,rel}$'),
        ('StagnationTemperatureRelDim', '$T_{t,rel}$'),
        ('StagnationEnthalpyRelDim', '$h_{t,rel}$'),
        ('StaticPressureDim', '$P_s$'),
        ('StaticTemperatureDim', '$T_s$'),
        ('StaticEnthalpyDim', '$h_s$'),
        ('EntropyDim',  '$s$'),
        ('Viscosity_EddyMolecularRatio', '$\mu_t / \mu$'),
        ('VelocityMeridianDim',  '$V_m$'),
        ('VelocityThetaRelDim',  '$W_\\theta$'),
        ('VelocityThetaAbsDim', '$V_\\theta$'),
        ('MachNumberRel',  '$M_{rel}$'),
        ('MachNumberAbs', '$M_{abs}$'),
        ('IsentropicMachNumber', '$M_{is}$'),
        ('AlphaAngleDegree', r'$\alpha$'),
        ('BetaAngleDegree', r'$\beta$'),
        ('PhiAngleDegree', r'$\phi$'),
        ('ChannelHeight', 'h'),
        ('StagnationPressureRatio', '$P_{t, abs, 2}/P_{t, abs, 1}$'),
        ('StagnationTemperatureRatio', '$T_{t, abs, 2}/T_{t, abs, 1}$'),
        ('StaticPressureRatio', '$P_{s, 2}/P_{s, 1}$'),
        ('Static2StagnationPressureRatio', '$P_{s, 2}/P_{t, abs, 1}$'),
        ('IsentropicEfficiency', r'$\eta_{is}$'),
        ('PolytropicEfficiency',  r'$\eta_{pol}$'),
        ('StagnationEnthalpyDelta', r'$\Delta h_{t, abs}$'),
        ('StaticPressureCoefficient', '$c_p$'),
        ('StagnationPressureCoefficient', '$c_{P_t}$'),
        ('StagnationPressureLoss1', r'$\omega_{P_t}$'),
        ('StagnationPressureLoss2', r'$\tilde{\omega}_{P_t}$'),
    ]
    for LongName, ShortName in remplacements:
        var = var.replace(LongName, ShortName)
    if unit:
        var += ' ({})'.format(unit)
    return var


class RadialProfilesPlotter():

    def __init__(self):
        self.base_name_with_profiles = 'RadialProfiles'
        self.container_profiles = names.CONTAINER_OUTPUT_FIELDS_AT_VERTEX
        self.profiles = dict()
        self.profiles_on_single_surface = dict()
        self.profiles_comparison = dict()

    def read(self, source):
        warning_source = ''
        if isinstance(source, str):
            warning_source = f' in {source}'
            source = cgns.load(source)

        RadialProfilesBase = source.get(Name=self.base_name_with_profiles, Depth=1)
        if RadialProfilesBase is None:
            mola_logger.warning(f'Base {self.base_name_with_profiles} is not found{warning_source}')
            return
        
        for zone in RadialProfilesBase.zones():
            surfaceName = zone.name()
            self.profiles[surfaceName] = self._fields_to_dict(zone, container=self.container_profiles)
            extractionInfo = zone.get(Name=CGNS_NODE_EXTRACTION_LOG)
            if extractionInfo:
                self.profiles[surfaceName][CGNS_NODE_EXTRACTION_LOG] = extractionInfo
            
            for FS in zone.group(Name='Comparison#*', Type='FlowSolution'):
                comparedPlane = FS.name().split('#')[1]
                comparisonName = f'{surfaceName}#{comparedPlane}'
                self.profiles[comparisonName] = self._fields_to_dict(zone, container=FS.name())

    def sort(self):
        '''
        Sort radial profiles in two groups:
        
        #. radial profiles on a single surface

        #. radial profiles resulting of the comparison of two surfaces (difference, quotient). 
           For instance, an isentropic efficiency profile.
        '''
        for surface, RadialProfilesOnSurface in self.profiles.items():
            if '#' in surface:
                self.profiles_comparison[surface] = RadialProfilesOnSurface
            else:
                self.profiles_on_single_surface[surface] = RadialProfilesOnSurface

    def plot(self, profiles, filename='RadialProfiles.pdf', assemble=False, variables=None):
        '''
        Plot radial profiles

        Parameters
        ----------
        filename : str, optional
            generic name of the file, by default 'RadialProfiles.pdf'

        assemble : bool, optional
            if True, write a unique PDF file with all the plots. By default False
        
        variables : list
            Among the available data, plot only variables in this list. If :py:obj:`None`, plot all available data.
        '''
        
        try:
            os.makedirs(os.path.dirname(filename))
        except:
            pass

        if not filename.endswith('.pdf'):
            assemble = False

        if not assemble:
            self._plot(profiles, variables, filename)
        else:
            self._plot_and_assemble(profiles, variables, filename)

    @staticmethod
    def _get_text_for_first_page(profiles):
        '''Generate the text to fill the first page of the PDF file, with information on planes

        Parameters
        ----------
        profiles : dict

        Returns
        -------
        str
            text to write on the first page of the PDF file
        '''
        # First pages with infomation
        txt = 'RADIAL PROFILES\n\n'
        for plane, RadialProfilesOnPlane in profiles.items():
            ExtractionInfo = RadialProfilesOnPlane.get(CGNS_NODE_EXTRACTION_LOG, {})
            PlotParameters = RadialProfilesOnPlane.get('.PlotParameters', dict(label=plane))
            if 'label' in PlotParameters:
                txt += PlotParameters['label'] + '\n'
            try:
                txt += f"IsoSurface {ExtractionInfo['IsoSurfaceField']} = {ExtractionInfo['IsoSurfaceValue']}\n"
                try:
                    txt += f"{ExtractionInfo['ReferenceRow']} {ExtractionInfo['tag']}\n"
                except:
                    pass
            except: 
                pass
            txt += '\n'
        return txt
    
    @staticmethod
    def _plot(profiles, variables, filename):
        filename_split = filename.split('.')
        filename_root = '.'.join(filename_split[:-1])
        extension = filename_split[-1]

        for plane, RadialProfilesOnPlane in profiles.items():
            if plane.startswith('.'):
                continue
            mola_logger.info(f'Plot data in {filename_root}_<varname>_{plane}.{extension}:')
            variables2plot = []
            for var in RadialProfilesOnPlane:
                if not var in ['ChannelHeight', 'Gamma'] and not var.startswith('.'):
                    if (variables is None) or (var in variables):
                        variables2plot.append(var)
            
            PlotParameters = RadialProfilesOnPlane.get('.PlotParameters', dict(label=plane))

            for var in variables2plot:
                filename_tmp = f'{filename_root}_{var}_{plane}.{extension}'
                mola_logger.info(f'  > plot {var}')
                plt.figure()
                plt.plot(RadialProfilesOnPlane[var], RadialProfilesOnPlane['ChannelHeight'] * 100., **PlotParameters)
                plt.xlabel(pretty_name(var))
                plt.ylabel('h (%)')
                plt.grid()
                plt.savefig(filename_tmp)
                plt.close()

    def _plot_and_assemble(self, profiles, variables, filename):
        mola_logger.info(f'Plot data in {filename}:')
        with PdfPages(filename) as pdf:

            textFirstPage = self._get_text_for_first_page(profiles)
            
            firstPage = plt.figure()
            firstPage.clf()
            firstPage.text(0.5, 0.5, textFirstPage, transform=firstPage.transFigure, size=12, ha="center", va="center")
            pdf.savefig()
            plt.close()

            # Assumption: same data on all plane
            variables2plot = []
            firstPlaneData = next(iter(profiles.values())) 
            for var in firstPlaneData:
                if not var in ['ChannelHeight', 'Gamma'] and not var.startswith('.'):
                    if (variables is None) or (var in variables):
                        variables2plot.append(var)
            
            AxisProperties = profiles.get('.AxisProperties', dict())

            for var in variables2plot:
                mola_logger.info(f'  > plot {var}')
                plt.figure()
                for plane, RadialProfilesOnPlane in profiles.items():
                    if plane.startswith('.'): 
                        continue
                    PlotParameters = RadialProfilesOnPlane.get('.PlotParameters', dict(label=plane))
                    if not var in RadialProfilesOnPlane: 
                        mola_logger.warning(f'    {var} not found on {plane}')
                        continue
                    plt.plot(RadialProfilesOnPlane[var], RadialProfilesOnPlane['ChannelHeight'] * 100., **PlotParameters)
                plt.xlabel(pretty_name(var))
                plt.ylabel('h (%)')
                plt.grid()
                if len(profiles) > 1: 
                    plt.legend()
                if var in AxisProperties:
                    plt.gca().set(**AxisProperties[var])
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()

    @staticmethod
    def _fields_to_dict(zone, container):
        fields = dict()
        fs_node = zone.get(Name=container)
        if fs_node:        
            for node in fs_node.group(Type='DataArray', Depth=1):
                fields[node.name()] = node.value()
        return fields

def plot_radial_profiles(
        surfaces=None, 
        filename_single=None,
        filename_comparison=None,
        variables_single=None,
        variables_comparison=None,
        ):
    if surfaces is None:
        surfaces = os.path.join(DIRECTORY_OUTPUT, FILE_OUTPUT_2D)
    if filename_single is None:
        filename_single = os.path.join(DIRECTORY_OUTPUT, 'RadialProfiles.pdf')
    if filename_comparison is None:
        filename_comparison = os.path.join(DIRECTORY_OUTPUT, 'RadialProfilesComparison.pdf')
    if variables_single is None:
        variables_single = [
            'StaticPressureDim', 'StagnationPressureAbsDim', 'StagnationTemperatureAbsDim', 
            'VelocityMeridianDim', 'VelocityXAbsDim', 'VelocityThetaAbsDim', 
            'AlphaAngleDegree', 'MachNumberAbs', 'EntropyDim',
            'StagnationPressureRelDim', 'StagnationTemperatureRelDim', 
            'VelocityThetaRelDim', 'BetaAngleDegree', 'MachNumberRel',
            ]
    if variables_comparison is None:
        variables_comparison = [
            'StagnationEnthalpyDelta', 'StagnationPressureRatio', 'StagnationTemperatureRatio', 
            'StaticPressureRatio', 'IsentropicEfficiency',
            ]

    plotter = RadialProfilesPlotter()
    plotter.read(surfaces)
    plotter.sort()
    assemble = True if filename_single.endswith('.pdf') else False
    if len(plotter.profiles_on_single_surface) > 0 and variables_single is not None:
        plotter.plot(plotter.profiles_on_single_surface, filename_single, assemble=assemble, variables=variables_single)
    if len(plotter.profiles_comparison) > 0 and variables_comparison is not None:
        plotter.plot(plotter.profiles_comparison, filename_comparison, assemble=assemble, variables=variables_comparison)

