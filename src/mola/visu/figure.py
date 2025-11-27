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
from typing import Union

from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size >2:
    import os
    os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from matplotlib import cm

import Converter.PyTree as C
import Converter.Internal as I  # only manipulations of __FlowSolutionNodes__ and __FlowSolutionCenters__
import Post.PyTree as P
import CPlot.PyTree as CPlot

from treelab import cgns

from mola.logging import mola_logger
from mola import naming_conventions as names
from .helpers import xyz_to_pixel


class Figure():

    def __init__(self, window_in_pixels=(1200,800), dpi=100, camera={}, filename=None, background='white', Elements=[]):
        self.window_in_pixels = window_in_pixels
        self.dpi = dpi
        self.camera = camera
        self.filename = filename
        self.Elements = Elements
        self.signals = None
        self.background = background
        
        self.fig = None

    def plot_surfaces(
            self, 
            surfaces=os.path.join(names.DIRECTORY_OUTPUT, names.FILE_OUTPUT_2D), 
            filename=None, 
            Elements=None,
            default_vertex_container=names.CONTAINER_OUTPUT_FIELDS_AT_VERTEX,
            default_centers_container='BCDataSet',
            offscreen=5 # https://elsa.onera.fr/issues/10948#note-14 
            ):
                    
        if filename:
            self.filename = filename
            self.fig = None  # init fig to call _create_overlap next time
        if Elements is not None:
            no_blending_elts = []
            blending_elts = []
            for i, elt in enumerate(Elements):
                if 'blending' in elt:
                    blending_elts += [elt]
                else:
                    no_blending_elts += [elt]
            self.Elements = no_blending_elts + blending_elts

        external_centers_container = I.__FlowSolutionCenters__
        external_vertex_container = I.__FlowSolutionNodes__

        cmap2int = dict(Blue2Red=1, Diverging=15, Black2White=15,
                        Viridis=17, Inferno=19, Magma=21, Plasma=23, Jet=25,
                        Greys=27, NiceBlue=29, Greens=31)

        t = cgns.load(surfaces)

        DIRECTORY_FRAMES = self.filename.split(os.path.sep)[:-1]    
        try: os.makedirs(os.path.join(*DIRECTORY_FRAMES))
        except: pass


        DisplayOptions = dict(mode='Render', displayInfo=0, displayIsoLegend=0, 
                            win=self.window_in_pixels, export=self.filename, shadow=1,
                            exportResolution='%gx%g'%self.window_in_pixels)
        DisplayOptions.update(self.camera)

        def hasBlending(elt):
            try:
                blend = elt['blending'] < 1
            except:
                return False
            try:
                mesh = elt['meshOverlay'] == True
            except:
                return False
            return blend and not mesh

        def get_zones_for_element(t, elt):
            field_name = elt['color'].replace('Iso:','')
            if elt['color'].startswith('Iso:'):
                zones = [z for z in t.get(Type='CGNSBase', Name=elt['extraction_name'], Depth=1).zones() if 
                    _fieldExistsAtNodesOrCentersAtZone(z, field_name, elt['vertex_container'], elt['centers_container'])]                    
            else:
                zones = t.get(Type='CGNSBase', Name=elt['extraction_name'], Depth=1).zones() 

            if not zones:
                if elt['color'].startswith('Iso:'):
                    warning_msg = (
                        f'visualization element [{i}]: '
                        f'field "{field_name}" does not exist in container {elt["vertex_container"]} nor {elt["centers_container"]} for extraction_name {elt["extraction_name"]}. '
                        'Please adjust the vertex_container and centers_container options'
                    )
                    mola_logger.warning(warning_msg)
                    return None
                else:
                    t.save('debug.cgns')
                    raise ValueError(f'could not find extraction_name={elt["extraction_name"]}, check debug.cgns')

            if hasBlending(elt): 
                zones = [cgns.castNode(z) for z in C.convertArray2Hexa(zones)] # see cassiopee #8740

            for z in zones:
                CPlot._addRender2Zone(z, material=elt['material'],
                    color=elt['color'], blending=elt['blending'],
                    meshOverlay=elt['meshOverlay'], shaderParameters=elt['shaderParameters'])
            
            # Add iso lines if required
            for value in elt['iso_line']: 
                isoLine = P.isoLine(zones, field_name, value)
                isoLine = cgns.castNode(isoLine)
                CPlot._addRender2Zone(isoLine, material='Solid', color=elt['iso_line_color'])
                zones.append(isoLine)
        
            return zones


        Trees = []
        TreesBlending = []
        baseName2elt = {}
        for i, elt in enumerate(self.Elements):

            # set default values
            elt.setdefault('extraction_name', None)
            elt.setdefault('blending', 1)
            elt.setdefault('material', 'Solid')
            elt.setdefault('color', 'White')
            elt.setdefault('meshOverlay', None)
            elt.setdefault('shaderParameters', None)
            elt.setdefault('vertex_container', default_vertex_container)
            elt.setdefault('centers_container', default_centers_container)
            elt.setdefault('iso_line', [])
            if not isinstance(elt['iso_line'], (list, np.ndarray)):
                elt['iso_line'] = [elt['iso_line']]
            elt.setdefault('iso_line_color', 'Black')

            I.__FlowSolutionNodes__ = elt['vertex_container']
            I.__FlowSolutionCenters__ = elt['centers_container']

            zones = get_zones_for_element(t, elt)
            if zones is None:
                continue

            if hasBlending(elt):
                base_name = f'blend.{i}'
                tree = cgns.Tree()
                cgns.Base(Name=base_name, Children=zones, Parent=tree)
                TreesBlending.append(tree)
            else:
                base_name = f'elt.{i}'
                baseName2elt[base_name] = elt
                tree = cgns.Tree()
                cgns.Base(Name=base_name, Children=zones, Parent=tree)
                Trees.append(tree)

        # requires to append blended zones (see cassiopee #8740 and #8748)
        # BEWARE of cassiopee #11311
        if TreesBlending:
            bases_blending = TreesBlending.bases()
            for t in Trees:
                t.addChildren(bases_blending)

        isoScales = extract_isoscales(Trees, baseName2elt)

        backgroundFile = self._get_backgroud_file()

        for i in range(len(Trees)):
            increment_offscreen = len(Trees)==1 or (i>0 and i == len(Trees)-1 and offscreen > 1)
            if increment_offscreen: 
                offscreen += 1
            self._display_element(Trees[i], DisplayOptions, backgroundFile, offscreen, 
                                  cmap2int, isoScales, default_vertex_container, default_centers_container)
        
        I.__FlowSolutionCenters__ = external_centers_container
        I.__FlowSolutionNodes__ = external_vertex_container

    def _get_backgroud_file(self):
        backgroundFile = None
        MOLAloc = os.getenv('MOLA')
        path_background = os.path.join(MOLAloc,'mola', 'visu', 'backgrounds',f'background_{self.background}.png')
        if os.path.exists(path_background):
            backgroundFile = path_background
        return backgroundFile

    def _display_element(self, tree, DisplayOptions, backgroundFile, offscreen, 
                         cmap2int, isoScales, default_vertex_container, default_centers_container):
        prefix = 'elt.'
        try:
            elt_base = tree.get(Name=f'{prefix}*', Depth=1)
        except IndexError:
            prefix = 'blend.'
            try:
                elt_base = tree.get(Name=f'{prefix}*', Depth=1)
            except:
                tree.save('debug.cgns')
                raise ValueError('FATAL: expected bases starting with "elt.*" or "blend.*", check debug.cgns')
        elt_index = int(elt_base.name().replace(prefix, ''))
        elt = self.Elements[elt_index]

        try: vertex_container = elt['vertex_container']
        except KeyError: vertex_container = default_vertex_container
        try: centers_container = elt['centers_container']
        except KeyError: centers_container = default_centers_container
        I.__FlowSolutionNodes__ = vertex_container
        I.__FlowSolutionCenters__ = centers_container

        try: additionalDisplayOptions = elt['additionalDisplayOptions']
        except: additionalDisplayOptions = {}
        DisplayOptions.update(additionalDisplayOptions)

        if  backgroundFile and \
            'backgroundFile' not in additionalDisplayOptions and \
            'bgColor' not in additionalDisplayOptions:
            DisplayOptions['backgroundFile'] = backgroundFile
            DisplayOptions['bgColor'] = 13


        try: cmap = cmap2int[elt['colormap']]
        except KeyError: cmap=0
        try:
            if 'shadow' not in elt: elt['shadow'] = True
            if not elt['shadow']: cmap -= 1
        except: pass
        
        DisplayOptions['offscreen'] = offscreen
        DisplayOptions['colormap'] = cmap
        DisplayOptions['isoScales'] = isoScales

        CPlot.display(tree, **DisplayOptions)
        CPlot.finalizeExport(offscreen)

    def _create_overlap(self):
        img = plt.imread(self.filename)

        fig, ax = plt.subplots(figsize=(img.shape[1]/float(self.dpi),
                                         img.shape[0]/float(self.dpi)), dpi=self.dpi)

        self.fig = fig
        self.axes = [ax]
        
        self._build_CPlot_colormaps()

        ax.imshow(img)
        ax.plot([],[])
        ax.set_axis_off()
        plt.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0., hspace=0.)
        return img

    def _build_CPlot_colormaps(self):
        from .colormaps import COLORMAPS
        self.colormaps = COLORMAPS

    def add_colorbar(self, field_name='', orientation='vertical', center=(0.90,0.5),
                          width=0.025, length=0.8, number_of_ticks=5, extend='neither',
                          font_color='black', colorbar_title='',
                          ticks_opposed_side=False, ticks_format='%g', 
                          ticks_size='medium', title_size='large'):
        # ticks_size and title_size can be: float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
        if not self.fig: 
            self._create_overlap()
        
        levels = None
        cmap = None
        for elt in self.Elements:
            try: field = elt['color'].replace('Iso:','')
            except KeyError: continue
            if field == field_name:
                levels = elt['levels']
                cmap = elt['colormap']
                break
        if not levels or isinstance(levels[1],str) or isinstance(levels[2],str)  :
            raise ValueError('element with color=Iso:%s bad defined.'%field_name)
    
        if orientation not in ('vertical','horizontal'):
            raise ValueError("orientation must be 'vertical' or 'horizontal'")

        if orientation == 'horizontal':
            xmin = center[0]-length/2.0
            xmax = center[0]+length/2.0
            ymin = center[1]-width/2.0
            ymax = center[1]+width/2.0
        else:
            xmin = center[0]-width/2.0
            xmax = center[0]+width/2.0
            ymin = center[1]-length/2.0
            ymax = center[1]+length/2.0

        cbar_ticks = np.linspace(levels[1],levels[2],number_of_ticks)
        cbaxes = self.fig.add_axes([xmin,ymin,xmax-xmin,ymax-ymin])

        colormap = self.colormaps[cmap]
        norm = mplcolors.BoundaryNorm(boundaries=np.linspace(levels[1],levels[2],levels[0]), ncolors=colormap.N, extend=extend) 
        cset = cm.ScalarMappable(norm=norm, cmap=colormap)
        cbar = self.fig.colorbar(cset, cax=cbaxes, orientation=orientation, ticks=cbar_ticks, format=ticks_format)
        cbar.ax.tick_params(which='major', length=4.0, width=0.5, color=font_color, labelsize=ticks_size)
        cbar.ax.tick_params(which='minor', length=0.)
        cbar.ax.xaxis.label.set_color(font_color)
        cbar.ax.yaxis.label.set_color(font_color)
        cbar.ax.tick_params(axis='x',colors=font_color)
        cbar.ax.tick_params(axis='y',colors=font_color)

        if orientation == 'vertical':
            if ticks_opposed_side: cbar.ax.yaxis.set_ticks_position("left")
            else: cbar.ax.yaxis.set_ticks_position("right")

            for value in elt['iso_line']:
                cbar.ax.axhline(y=value, c=elt['iso_line_color'])
        else:
            if ticks_opposed_side: cbar.ax.xaxis.set_ticks_position("top")
            else: cbar.ax.xaxis.set_ticks_position("bottom")

            for value in elt['iso_line']:
                cbar.ax.axvline(x=value, c=elt['iso_line_color'])

        if colorbar_title: cbar.ax.set_title(colorbar_title, color=font_color, fontsize=title_size)
        else:              cbar.ax.set_title(field_name,     color=font_color, fontsize=title_size)
        cbar.update_ticks()
        
        return cbar

    def plot_signals(self, signals: Union[str, cgns.Tree, None]=None, left=0.05, right=0.5, bottom=0.05, top=0.4,
            xlim=None, ylim=None, xmax=None, xlabel=None, ylabel=None, figure_name=None,
            background_opacity=1.0, font_color='black', 
            curves=None,
            iterationTracer=None):
        
        if curves is None:
            return
        elif isinstance(curves, dict):
            curves = [curves]
        else:
            assert isinstance(curves, list)
        
        if not self.fig: 
            self._create_overlap()
     
        if signals is None:
            if not self.signals:
                self.signals = cgns.load(os.path.join(names.DIRECTORY_OUTPUT, names.FILE_OUTPUT_1D))
        else:
            self.signals = cgns.load(signals)
        
        ax = self.fig.add_axes([left,bottom,right-left,top-bottom])

        for curve in curves:
            # Available parameters in curve: zone_name, x, y, include_last_point_label, plot_params
            curve.setdefault('include_last_point_label', False)
            curve.setdefault('multiply_by', 1)
            curve.setdefault('plot_params', {})
            if len(curves) > 1:
                curve['plot_params'].setdefault('label', curve['zone_name'])

            zone = self.signals.group(Type='Zone_t', Name=curve['zone_name'])
            if not zone: raise ValueError(f'zone {curve["zone_name"]} not found in arrays')
            if len(zone) > 1:
                mola_logger.info(f'found {len(zone)} zones with name {curve["zone_name"]}. Will use first found.')
            zone = zone[0]

            x, y = zone.fields([curve['x'],curve['y']], BehaviorIfNotFound='raise')
            y *= curve['multiply_by']

            if xmax is not None:
                interval = x <= xmax
                x = x[interval]
                y = y[interval]

            ax.plot(x,y,**curve['plot_params'])
            if iterationTracer:
                try:
                    if isinstance(iterationTracer, int):
                        iterationTracer = dict(iteration=iterationTracer)
                    iterations = zone.fields(['Iteration'])[0]
                    # On the following line: -1 because quantities correspond to the previous iteration
                    index = np.where(iterations == iterationTracer['iteration'] - 1)[0]
                    if not 'plot_params' in iterationTracer:
                        iterationTracer['plot_params'] = dict(marker='o', color='red')
                    ax.plot(x[index], y[index], **iterationTracer['plot_params'])
                except:
                    pass
            
            if curve['include_last_point_label']:
                ax.text(x[-1], y[-1], "%g"%y[-1],
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        color=plt.gca().lines[-1].get_color())


        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        if isinstance(xlabel,str): ax.set_xlabel(xlabel)
        else: ax.set_xlabel(curve['x'])
        if isinstance(ylabel,str): ax.set_ylabel(ylabel)
        else:
            ylabels = [c['y'] for c in curves]
            if ylabels.count(ylabels[0]) == len(ylabels):
                ax.set_ylabel(ylabels[0])
        if isinstance(figure_name,str): ax.set_title(figure_name)
        ax.patch.set_alpha(background_opacity)
        self.axes += [ ax ]

        if len(curves) > 1:
            ax.legend()

        ax.spines['bottom'].set_color(font_color)
        ax.spines['top'].set_color(font_color) 
        ax.spines['right'].set_color(font_color)
        ax.spines['left'].set_color(font_color)
        ax.tick_params(axis='x', colors=font_color)
        ax.tick_params(axis='y', colors=font_color)
        ax.yaxis.label.set_color(font_color)
        ax.xaxis.label.set_color(font_color)
        ax.title.set_color(font_color)

        return ax

    def plot(self, x, y, z, *args, **kwargs):
        if not self.fig: 
            self._create_overlap()

        x = np.array(x, ndmin=1)
        if isinstance(y, (float, int)):
            y = y * np.ones(x.size)
        if isinstance(z, (float, int)):
            z = z * np.ones(x.size)
        points = zip(x, y, z)
        pixels = xyz_to_pixel(points, self.window_in_pixels, **self.camera)
        pixels = np.array(pixels)
        u, v = pixels[:,0], pixels[:,1]
        self.axes[0].plot(u, v, *args, **kwargs)


    def save(self, output_filename=''):
        if not self.fig: 
            self._create_overlap()
            
        if not output_filename:
            output_filename = self.filename

        DIRECTORY_FRAMES = output_filename.split(os.path.sep)[:-1]    
        try: os.makedirs(os.path.join(*DIRECTORY_FRAMES))
        except: pass

        mola_logger.info(f'saving {output_filename} ...')
        plt.savefig(output_filename, dpi=self.dpi)
        mola_logger.info('done')
        for ax in self.axes: ax.clear()
        self.fig.clear()
        plt.close('all')

    def show(self):
        plt.show()

def extract_isoscales(Trees, baseName2elt):
    all_TreesMerged = cgns.merge(Trees)
    isoScales = [] # must be the same for all composite calls of CPlot.display
    for base_name, elt in baseName2elt.items():
        tree = all_TreesMerged.get(Name=base_name, Depth=1)
        if not tree:
            all_TreesMerged.save('debug.cgns')
            raise ValueError(f'could not find base "{base_name}", check debug.cgns')

        I.__FlowSolutionNodes__ = elt['vertex_container']
        I.__FlowSolutionCenters__ = elt['centers_container']

        if elt['color'].startswith('Iso:'):
            field_name = elt['color'].replace('Iso:','')
            levels = elt.get('levels', [200,'min','max'])

            levels[1] = _getMin(tree, field_name) if levels[1] == 'min' else float(levels[1])
            levels[2] = _getMax(tree, field_name) if levels[2] == 'max' else float(levels[2])
            elt['levels'] = levels
            iso_nodes = [field_name, levels[0], levels[1], levels[2]]
            iso_centers = ['centers:'+field_name, levels[0], levels[1], levels[2]]
            if len(levels) == 5:
                iso_nodes.extend([levels[3],levels[4]])
                iso_centers.extend([levels[3],levels[4]])
            isoScales += iso_nodes, iso_centers
        return isoScales
        
def _getMax(t: cgns.Node, field_name: str): return _getMinOrMax(t, field_name, Min=False)

def _getMin(t: cgns.Node, field_name: str): return _getMinOrMax(t, field_name, Min=True)

def _getMinOrMax(t: cgns.Node, field_name: str, Min: bool=True):
    if Min:
        sign = 1
        Fun = np.minimum
        fun = np.min
    else:
        sign = -1
        Fun = np.maximum
        fun = np.max

    actual = sign * np.inf
    for z in t.zones():
        for fs in z.group(Type='FlowSolution_t', Depth=1):
            if fs.name() not in [I.__FlowSolutionNodes__, I.__FlowSolutionCenters__]:
                continue
            node = fs.get(Name=field_name, Type='DataArray_t', Depth=1)
            if not node: 
                continue
            actual = Fun(actual, fun(node.value()))            

    if not np.isfinite(actual): 
        t.save('debug.cgns')
        raise ValueError(f'could not find min/max for {field_name} at {[I.__FlowSolutionNodes__, I.__FlowSolutionCenters__]}, check debug.cgns')
    return actual

def _fieldExistsAtNodesOrCentersAtZone(z: cgns.Zone, field_name: str, 
                                       vertex_container_name: str,
                                       centers_container_name: str):
    for fsname in [vertex_container_name, centers_container_name]:
        fs = z.get(Type='FlowSolution', Name=fsname, Depth=1)
        if not fs: 
            continue
        if fs.get(Name=field_name, Type='DataArray', Depth=1):
            return True
    return False


if __name__ == '__main__':

    from mola import visu

    from mpi4py import MPI
    comm   = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size >2:
        import os
        os.environ['OMP_NUM_THREADS'] = '1'

    first_iteration = 10
    delta_iteration = 10
    last_iteration  = 1200

    iterations = np.arange(first_iteration, last_iteration+delta_iteration, delta_iteration)
    iterations_per_rank = np.array_split(iterations, size)

    for i in iterations_per_rank[rank]:

        # set the different field elements to include in the image
        Elements = [
                dict(extraction_name='Iso_Z*',
                    color='Iso:Mach', colormap='Diverging',shadow=False),

                dict(extraction_name='WALL',
                    color='Iso:Pressure', colormap='Jet',shadow=False, levels=[40, 'min', 'max']),
                ]

        fig = visu.Figure(
            window_in_pixels = (1200, 800),
            dpi = 150,
            # set the camera view, the image size (in pixels) and the output image filename
            camera = dict(
                posCam  = (-0.008621,-0.000514, -0.032904),
                posEye = (-0.008621, -0.000514, 0.000001),
                dirCam = (0.000000, 1.000000, 0.000000)
                ),
            Elements = Elements, 
            filename = f'FRAMES/frame{i:06d}.png',
            )

        # generate the CPlot image of the field elements (will write an image)
        fig.plot_surfaces('OUTPUT/surfaces_AfterIter%d.cgns'%i)

        # include matplotlib components, such as colorbar and a 2D curve plot
        ax = fig.add_colorbar(field_name='Mach', orientation='vertical', center=(0.87,0.5),
                    width=0.025, length=0.8, font_color='black',
                    colorbar_title=r'$\bf{Mach}$')

        ax = fig.add_colorbar(field_name='Pressure', orientation='horizontal', center=(0.45,0.9),
                    width=0.025, length=0.7, font_color='black',
                    colorbar_title=r'$\bf{Pressure}$ (Pa)')

        # # plot the sphere drag coefficient
        # signals = cgns.load('OUTPUT/signals.cgns')
        # ax = fig.plot_signals(signals, left=0.10, right=0.40, bottom=0.08, top=0.28,
        #         xlabel='iteration', ylabel=r'$\bf{C_D}$',
        #         xlim=(first_iteration, i), ylim=(0,0.8),
        #         background_opacity=0.0, font_color='black',
        #         curves=[dict(zone_name='WALL',x='Iteration',y='CD',
        #                         plot_params={'color':'C0'}),])
        # for b in 'top', 'right': ax.spines[b].set_visible(False)

        # # plot the probe pressure
        # ax = fig.plot_signals(signals, left=0.55, right=0.85, bottom=0.08, top=0.28,
        #         xlabel='iteration', ylabel=r'$\bf{Pressure}$ at Probe (Pa)',
        #         xlim=(first_iteration, i),
        #         background_opacity=0.0, font_color='black',
        #         curves=[dict(zone_name='ProbeDownstream',x='Iteration',y='Pressure',
        #                         plot_params={'color':'magenta'}),])

        # probe = signals.get(Name='ProbeDownstream', Type='Zone_t')
        # probe_coords = probe.get(Name='position').value()
        # x, y, z = probe_coords[0], probe_coords[1], probe_coords[2]
        # fig.plot(x, y, z, 'o', color='magenta' )
        # for b in 'top', 'right': ax.spines[b].set_visible(False)

        # save image
        fig.save()

    comm.barrier()
    if rank==0:
        print('making movie...')
        visu.makeMovie(gif_filename='animation.gif', fps=24, width=800)
        