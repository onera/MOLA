'''
MOLA - Visu.py

VISU MODULE

Collection of routines and functions designed for visualization
techniques.

USEFUL NOTES:

Command lines for animations:

# first, resize your images to the width size desired for final video (e.g. 600 px)
cd FRAMES
for img in frame*.png; do convert -resize 600 -quality 100 "$img" "resized-$img"; done

# second, create movie with (increase fps for having faster motion)
mencoder  mf://FRAMES/resized-frame*.png -mf fps=24  -ovc x264 -x264encopts subq=6:partitions=all:8x8dct:me=umh:frameref=5:bframes=3:b_pyramid=normal:weight_b -o movie.avi

# then convert movie to gif, by scaling to desired pixels (e.g. width 400 px)
ffmpeg -i movie.avi -vf "fps=10,scale=400:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 animation.gif

File history:
12/06/2020 - v1.8.01 - L. Bernardos - Creation
'''

import MOLA
from . import InternalShortcuts as J

if not MOLA.__ONLY_DOC__:
    import sys
    import os
    import numpy as np
    from time import sleep

    import matplotlib.pyplot as plt
    import matplotlib.colors as mplcolors
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    from matplotlib.backends.backend_pdf import PdfPages


    import Converter.PyTree as C
    import Converter.Internal as I
    import Transform.PyTree as T



def xyz2Pixel(points,win,posCam,posEye,dirCam,viewAngle=50.0):
    '''
    Returns the two-component image-pixel positions of a set of points
    located in the 3D world of CPlot.

    Parameters
    ----------

        points : :py:class:`list` of 3 :py:class:`float` :py:class:`tuple`
            :math:`(x,y,z)` coordinates of points in 3D world

        win : :py:class:`tuple` of 2 :py:class:`int`
            Window resolution in pixels

        posCam : :py:class:`tuple` of 3 :py:class:`float`
            position of Camera (see CPlot doc)

        posEye : :py:class:`tuple` of 3 :py:class:`float`
            position of eye (see CPlot doc)

        dirCam : :py:class:`tuple` of 3 :py:class:`float`
            direction of Camera (see CPlot doc)

        viewAngle : float
            angle of Camera (see CPlot doc)

    Returns
    -------

        width, height : :py:class:`tuple` of 2 :py:class:`float`
            width and height in pixels using the convention of origin located at
            upper left side of image

    '''

    # ------------------------------- #
    # BUILD FRENET UNIT FRAME (b,n,c) #
    # ------------------------------- #
    # <c> is the Camera axes unit vector
    c =  np.array(posCam) - np.array(posEye)
    R = np.sqrt(c.dot(c)) # R is distance between posCam and posEye
    c /= R

    # <b> is binormal
    b = np.cross(np.array(dirCam),c)
    b /= np.sqrt(b.dot(b))

    # <n> is normal
    n = np.cross(c,b)
    n /= np.sqrt(b.dot(b))

    # <h> is the constant total height of the curvilinear window
    va = np.deg2rad(viewAngle)
    h = R * va
    h = 2 * R * np.tan(va/2.)

    # used to transform curvilinear unit to pixel
    crv2Pixel = float(win[1]) / h

    Pixels = []

    # The window plane is defiend as a set of three points (p0, p1, p2)
    p0 = np.array(posEye)
    p1 = p0+b
    p2 = p0+n
    p01 = p1 - p0 # segment
    p02 = p2 - p0 # segment

    for point in points:
        # ----------------------------------- #
        # COMPUTE pixel-position of point <p> #
        # ----------------------------------- #
        p = np.array(point)

        # Shall compute the intersection of the view of point <p> with the window plane

        # Such line is defined through two points (la, lb) as
        la, lb = np.array(posCam), p
        lab = lb - la # segment

        # Intersection is given by equation x = la + lab*t
        den = -lab.dot(np.cross(p01,p02))

        # Only for information (computation not required):
        # t = np.cross(p01,p02).dot(la-p0) / den
        # x = la + lab*t

        # parametric components (u, v) are actually what we look for
        u = np.cross(p02,-lab).dot(la-p0) / den
        v = np.cross(-lab,p01).dot(la-p0) / den

        # width and height in pixels are expressed in terms of (u, v)
        # Pixel components relative to Figure origin (upper left)
        pxP_w =  u*crv2Pixel + 0.5*float(win[0])
        pxP_h = -v*crv2Pixel + 0.5*float(win[1])

        Pixels += [[pxP_w, pxP_h]]
    return Pixels


def plotSurfaces(surfaces, frame='FRAMES/frame.png', camera={}, 
        window_in_pixels=(1200,800),
        Elements=[dict(selection=dict(baseName='BCWall'),
                       material='Solid', color='White', blending=0.8,
                       colormap='Viridis', levels=[200,'min','max'], shadow=True,
                       additionalDisplayOptions={},
                       additionalStateOptions={})]):

    machine = os.getenv('MAC', 'ld')
    if machine in ['spiro', 'sator']:
        offscreen=5 # TODO solve bug https://elsa.onera.fr/issues/10536
    elif machine in ['ld', 'visung', 'visio']:
        offscreen=3
    else:
        raise SystemError('machine "%s" not supported.'%machine)

    cmap2int = dict(Blue2Red=1, Green2Red=3, Diverging=9, Black2White=15,
                    Viridis=17, Inferno=19, Magma=21, Plasma=23, NiceBlue=25)

    import CPlot.PyTree as CPlot

    if isinstance(surfaces,str):
        t = C.convertFile2PyTree(surfaces)
    elif I.isTopTree(surfaces):
        t = surfaces
    else:
        raise ValueError('parameter mesh must be either a filename or a PyTree')

    DIRECTORY_FRAMES = frame.split(os.path.sep)[:-1]    
    try: os.makedirs(os.path.join(*DIRECTORY_FRAMES))
    except: pass

    DisplayOptions = dict(mode='Render', displayInfo=0, displayIsoLegend=0, 
                          win=window_in_pixels, export=frame, shadow=1,
                          exportResolution='%gx%g'%window_in_pixels)
    DisplayOptions.update(camera)

    def hasBlending(elt):
        try: return elt['blending'] < 1
        except: return False

    Trees = []
    TreesBlending = []
    for i, elt in enumerate(Elements):
        try: selection = elt['selection']
        except KeyError: selection = {}
        try: blending = elt['blending']
        except KeyError: blending = 1
        try: material = elt['material']
        except KeyError: material = 'Solid'
        try: color = elt['color']
        except KeyError: color = 'White'
        zones = J.selectZones(t, **selection)

        if hasBlending(elt): zones = C.convertArray2Hexa(zones) # see cassiopee #8740

        

        for z in zones:
            CPlot._addRender2Zone(z, material=material,color=color,
                                     blending=blending)

        if hasBlending(elt):
            TreesBlending += [ C.newPyTree(['blend.%d'%i, zones]) ]
        else:
            Trees += [ C.newPyTree(['elt.%d'%i, zones]) ]

    # requires to append blended zones (see cassiopee #8740 and #8748)
    if TreesBlending:
        for i in range(len(Trees)):
            Trees[i] = I.merge([Trees[i]]+TreesBlending)
    


    for i in range(len(Trees)):
        tree = Trees[i]
        elt = Elements[i]
        
        if elt['color'].startswith('Iso:'):
            field_name = elt['color'].replace('Iso:','')
            levels = elt['levels']
            levels[1] = C.getMinValue(tree, field_name) if levels[1] == 'min' else float(levels[1])
            levels[2] = C.getMaxValue(tree, field_name) if levels[2] == 'max' else float(levels[2])
            isoScales = [[field_name, levels[0], levels[1], levels[2]],
              ['centers:'+field_name, levels[0], levels[1], levels[2]]]
        else:
            isoScales = []

        if i>0 and i == len(Trees)-1 and offscreen > 1: offscreen += 1

        try: additionalDisplayOptions = elt['additionalDisplayOptions']
        except: additionalDisplayOptions = {}

        try: additionalStateOptions = elt['additionalStateOptions']
        except: additionalStateOptions = {}

        if  'backgroundFile' not in additionalDisplayOptions and \
            'bgColor' not in additionalDisplayOptions:
            MOLA = os.getenv('MOLA')
            MOLASATOR = os.getenv('MOLASATOR')
            for MOLAloc in [MOLA, MOLASATOR]:
                backgroundFile = os.path.join(MOLAloc,'MOLA','GUIs','background.png')
                if os.path.exists(backgroundFile):
                    CPlot.setState(backgroundFile=backgroundFile)
                    CPlot.setState(bgColor=13)
                    break
        if additionalStateOptions: CPlot.setState(**additionalStateOptions)


        try: cmap = cmap2int[elt['colormap']]
        except KeyError: cmap=0
        try:
            if not elt['shadow']: cmap -= 1
        except: pass

        CPlot.display(tree, offscreen=offscreen, colormap=cmap,
            isoScales=isoScales, **DisplayOptions, **additionalDisplayOptions)
        CPlot.finalizeExport(offscreen)
    
        sleep(0.5)


class matplotlipOverlap():
    """docstring"""

    def __init__(self, image_file='', Elements=[], dpi=100):
        self.image_file = image_file
        self.dpi = dpi
        self.Elements = Elements
        self.arrays = None
        

        img = plt.imread( image_file )

        fig, ax = plt.subplots(figsize=(img.shape[1]/float(dpi),
                                         img.shape[0]/float(dpi)), dpi=dpi)

        self.fig = fig
        self.axes = [ax]
        
        self._buildCPlotColormaps()

        ax.imshow(img)
        ax.plot([],[])
        ax.set_axis_off()
        plt.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0., hspace=0.)

    def _buildCPlotColormaps(self):
        f = [0,0.03125,0.0625,0.09375, 0.125, 0.15625, 0.1875, 0.21875, 0.25,
            0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875, 0.50,
            0.53125, 0.5625, 0.59375, 0.625, 0.65625, 0.6875, 0.71875, 0.75,
            0.78125, 0.8125, 0.84375, 0.875, 0.900625, 0.9375, 0.96875, 1.0]

        # keys must be the same as available in plotSurfaces
        self.color_codes = dict(
            Blue2Red=[(0.00,[0,0,1]),
                      (0.25,[0,1,1]),
                      (0.50,[0,1,0]),
                      (0.75,[1,1,0]),
                      (1.00,[1,0,0])],
            Green2Red=[(0.00,[0,1,0]),
                       (0.25,[0,1,1]),
                       (0.50,[0,0,1]),
                       (0.75,[1,0,1]),
                       (1.00,[1,0,0])],
            Diverging=[(f[0], [0.23137254902, 0.298039215686,0.752941176471]),
                       (f[1], [0.23137254902+1.12941176471*f[1],0.298039215686+1.7568627451*f[1],0.7980392156854992]),
                       (f[2], [0.23137254902+1.12941176471*f[2],0.298039215686+1.7568627451*f[2],15.6588235294-237.050980392*f[2]]),
                       (f[3], [0.223529411765+1.25490196078*f[3], 0.305882352941+1.63137254902*f[3],  0.764705882353+1.25490196078*f[3]]),
                       (f[4], [0.211764705882+1.38039215686*f[4], 0.305882352941+1.63137254902*f[4],  0.776470588235+1.12941176471*f[4]]),
                       (f[5], [0.227450980392+1.25490196078*f[5], 0.321568627451+1.50588235294*f[5], 0.807843137255+0.878431372549*f[5]]),
                       (f[6], [0.207843137255+1.38039215686*f[6], 0.321568627451+1.50588235294*f[6], 0.827450980392+0.752941176471*f[6]]),
                       (f[7], [0.207843137255+1.38039215686*f[7], 0.345098039216+1.38039215686*f[7], 0.874509803922+0.501960784314*f[7]]),
                       (f[8], [0.207843137255+1.38039215686*f[8], 0.345098039216+1.38039215686*f[8], 0.901960784314+0.376470588235*f[8]]),
                       (f[9], [0.207843137255+1.38039215686*f[9], 0.407843137255+1.12941176471*f[9], 0.964705882353+0.125490196078*f[9]]),
                       (f[10],[0.207843137255+1.38039215686*f[10],0.407843137255+1.12941176471*f[10],1.0]),
                       (f[11],[0.207843137255+1.38039215686*f[11], 0.486274509804+0.878431372549*f[11], 1.07843137255-0.250980392157*f[11]]),
                       (f[12],[0.250980392157+1.25490196078*f[12], 0.486274509804+0.878431372549*f[12], 1.16470588235-0.501960784314*f[12]]),
                       (f[13],[0.250980392157+1.25490196078*f[13], 0.580392156863+0.627450980392*f[13], 1.21176470588-0.627450980392*f[13]]),
                       (f[14],[0.250980392157+1.25490196078*f[14],  0.63137254902+0.501960784314*f[14], 1.26274509804-0.752941176471*f[14]]),
                       (f[15],[0.305882352941+1.12941176471*f[15], 0.741176470588+0.250980392157*f[15],  1.37254901961-1.00392156863*f[15]]),
                       (f[16],[0.364705882353+1.00392156863*f[16], 0.858823529412, 1.43137254902-1.12941176471*f[16]  ]),
                       (f[17],[0.364705882353+1.00392156863*f[17],0.733333333333+0.250980392157*f[17],1.61960784314-1.50588235294*f[17]]),
                       (f[18],[0.43137254902+0.878431372549*f[18], 1.2-0.627450980392*f[18], 1.61960784314-1.50588235294*f[18]]),
                       (f[19],[0.572549019608+0.627450980392*f[19],1.2-0.627450980392*f[19],1.61960784314-1.50588235294*f[19]]),
                       (f[20],[0.647058823529+0.501960784314*f[20], 1.34901960784-0.878431372549*f[20], 1.61960784314-1.50588235294*f[20]]),
                       (f[21],[0.803921568627+0.250980392157*f[21], 1.42745098039-1.00392156863*f[21], 1.69803921569-1.63137254902*f[21]]),
                       (f[22],[0.96862745098, 1.50980392157-1.12941176471*f[22], 1.61568627451-1.50588235294*f[22]]),
                       (f[23],[0.96862745098, 1.59607843137-1.25490196078*f[23], 1.70196078431-1.63137254902*f[23]]),
                       (f[24],[1.23921568627-0.376470588235*f[24], 1.6862745098-1.38039215686*f[24], 1.61176470588-1.50588235294*f[24]]),
                       (f[25],[1.23921568627-0.376470588235*f[25],1.78039215686-1.50588235294*f[25],1.61176470588-1.50588235294*f[25]]),
                       (f[26],[1.43529411765-0.627450980392*f[26], 1.87843137255-1.63137254902*f[26], 1.61176470588-1.50588235294*f[26]]),
                       (f[27],[1.63921568627-0.878431372549*f[27],1.98039215686-1.7568627451*f[27],1.50980392157-1.38039215686*f[27]]),
                       (f[28],[1.63921568627-0.878431372549*f[28], 2.0862745098-1.88235294118*f[28], 1.50980392157-1.38039215686*f[28]]),
                       (f[29],[1.85882352941-1.12941176471*f[29],2.19607843137-2.00784313725*f[29],1.50980392157-1.38039215686*f[29]]),
                       (f[30],[1.97254901961-1.25490196078*f[30], 4.2431372549-4.26666666667*f[30], 1.39607843137-1.25490196078*f[30]]),
                       (f[31],[2.09019607843-1.38039215686*f[31], 2.83137254902-2.76078431373*f[31], 1.27843137255-1.12941176471*f[31]]),
                       (f[32],[2.21176470588-1.50588235294*f[32], 4.53333333333-4.51764705882*f[32], 1.27843137255-1.12941176471*f[32]])],
            Black2White=[(0.00,[0.,0.,0.]),
                         (1.00,[1.,1.,1.])],
            Viridis=[(0.00,[253./255.,231./255.,37./255.]),
                     (0.50,[33./255.,145./255.,140./255.]),
                     (1.00,[68./255.,1./255.,84./255.])],
            Inferno=[(0.00,[252./255.,255./255.,164./255.]),
                     (0.50,[188./255.,55./255.,84./255.]),
                     (1.00,[0./255.,0./255.,4./255.])],
            Magma=[(0.00,[252./255.,253./255.,191./255.]),
                   (0.50,[183./255.,55./255.,121./255.]),
                   (1.00,[0./255.,0./255.,4./255.])],
            Plasma=[(0.00,[240./255.,249./255.,33./255.]),
                    (0.50,[204./255.,71./255.,120./255.]),
                    (1.00,[13./255.,8./255.,135./255.])],
            NiceBlue=[(0.00,[255./255.,255./255.,255./255.]),
                      (0.50,[0./255.,97./255.,165./255.]),
                      (1.00,[0./255.,0./255.,0./255.])]) 

        self.colormaps = dict()
        for color_code in self.color_codes:
            self.colormaps[color_code] = LinearSegmentedColormap.from_list(color_code,
                self.color_codes[color_code])

    def addColorbar(self, field_name='', orientation='vertical', center=(0.90,0.5),
                          width=0.025, length=0.8, number_of_ticks=5,
                          font_color='black', colorbar_title='',
                          ticks_opposed_side=False, ticks_format='%g'):
        
        levels = None
        cmap = None
        for elt in self.Elements:
            try: field = elt['color'].replace('Iso:','')
            except KeyError: continue
            if field == field_name:
                levels = elt['levels']
                cmap = elt['colormap']
                break
        if not levels:
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

        cset = cm.ScalarMappable(norm=mplcolors.Normalize(levels[1],
                                                          levels[2],
                                                        clip=False),
                                 cmap=self.colormaps[cmap].reversed())
        cset.set_array(np.linspace(levels[1],levels[2],levels[0]))
        cbar = self.fig.colorbar(cset, cax=cbaxes, orientation=orientation,
                                       ticks=cbar_ticks, format=ticks_format)
        cbar.ax.tick_params(which='major', length=4.0, width=0.5, color=font_color)
        cbar.ax.xaxis.label.set_color(font_color)
        cbar.ax.yaxis.label.set_color(font_color)
        cbar.ax.tick_params(axis='x',colors=font_color)
        cbar.ax.tick_params(axis='y',colors=font_color)

        if orientation == 'vertical':
            if ticks_opposed_side: cbar.ax.yaxis.set_ticks_position("left")
            else: cbar.ax.yaxis.set_ticks_position("right")
        else:
            if ticks_opposed_side: cbar.ax.xaxis.set_ticks_position("top")
            else: cbar.ax.xaxis.set_ticks_position("bottom")

        if colorbar_title: cbar.ax.set_title(colorbar_title, color=font_color)
        else:              cbar.ax.set_title(field_name,     color=font_color)
        cbar.update_ticks()
        
        return cbar

    def _loadArrays(self, arrays_file_or_tree):
        if isinstance(arrays_file_or_tree,str):
            self.arrays = C.convertFile2PyTree( arrays_file_or_tree )
        else:
            self.arrays = arrays_file_or_tree

    def plotArrays(self, arrays_file_or_tree, left=0.05, right=0.5, bottom=0.05, top=0.4,
            xlim=None, ylim=None, xmax=None, xlabel=None, ylabel=None, figure_name=None,
            background_opacity=1.0, font_color='black', 
            curves=[dict(zone_name='BLADES',x='IterationNumber',y='MomentumXFlux',
                         plot_params={})], 
            iterationTracer=None):

        if not self.arrays: self._loadArrays(arrays_file_or_tree)
        
        ax = self.fig.add_axes([left,bottom,right-left,top-bottom])

        for curve in curves:
            zone = [z for z in I.getZones(self.arrays) if z[0]==curve['zone_name']]
            if not zone: raise ValueError('zone %s not found in arrays'%curve['zone_name'])
            if len(zone) > 1:
                print('found %d zones with name %s. Will use first found.'%(len(zone),curve['zone_name']))
            zone = zone[0]

            x, y = J.getVars(zone,[curve['x'],curve['y']])
            if x is None: raise ValueError('x variable "%s" not found in %s'%(curve['x'],zone[0]))
            if y is None: raise ValueError('y variable "%s" not found in %s'%(curve['y'],zone[0]))

            if xmax is not None:
                interval = x <= xmax
                x = x[interval]
                y = y[interval]

            ax.plot(x,y,**curve['plot_params'])
            if iterationTracer:
                try:
                    if isinstance(iterationTracer, int):
                        iterationTracer = dict(iteration=iterationTracer)
                    iterations = J.getVars(zone, ['IterationNumber'])[0]
                    # On the following line: -1 because quantities correspond to the previous iteration
                    index = np.where(iterations == iterationTracer['iteration'] - 1)[0]
                    if not 'plot_params' in iterationTracer:
                        iterationTracer['plot_params'] = dict(marker='o', color='red')
                    ax.plot(x[index], y[index], **iterationTracer['plot_params'])
                except:
                    pass

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

    def save(self, output_filename=''):
        if not output_filename:
            output_filename = self.image_file

        DIRECTORY_FRAMES = output_filename.split(os.path.sep)[:-1]    
        try: os.makedirs(os.path.join(*DIRECTORY_FRAMES))
        except: pass

        print('saving %s ...'%output_filename,end=' ')
        plt.savefig(output_filename, dpi=self.dpi)
        print('done')
        for ax in self.axes: ax.clear()
        self.fig.clear()
        plt.close('all')


def makeShaftRotate(t, iteration):
    '''
    For a turbomachinery case, make rotate all the domains depending on their `.Solver#Motion` node in 
    their family. The angle of rotation is computed as:
    :math:`\theta = t_0 + \Omega (i-i_0)`
    where :math:`t_0` corresponds to the `itime` elsA key, :math:`t_0` is the rotationnal speed of the 
    row (depending on the current zone), :math:`i` is the current iteration (input argument **iteration**)
    and :math:`i_0` corresponds to the `inititer` elsA key.

    Parameters
    ----------
    t : PyTree
        Should be the PyTree read from a ``surfaces_AfterIteration*.cgns``.

    iteration : int
        iteration of extraction of **t**. Should correspond to the iteration in the name 
        of the file ``surfaces_AfterIteration*.cgns``.
    '''
    # TODO: Is this iteration or iteration - 1 to consider ? 

    # TODO: rotate also BCDataSet, ZoneSubRegion, etc, and others vectors
    # typically what Maia already

    setup = J.load_source('setup', 'setup.py')

    ekn = setup.elsAkeysNumerics
    currentTime = ekn['itime'] + ekn['timestep'] * (iteration - ekn['inititer'])

    vectors2rotate = [['VelocityX', 'VelocityY', 'VelocityZ'], ['MomentumX', 'MomentumY', 'MomentumZ']]
    vectors = []
    for vec in vectors2rotate:
        vectors.append(vec)
        vectors.append(['centers:'+v for v in vec])

    for base in I.getBases(t):
        # Fill families information
        rowFrame = dict()
        for Family in I.getNodesFromType1(base, 'Family_t'):
            solverMotion = I.getNodeFromName(Family, '.Solver#Motion')
            if not solverMotion: 
                # Not a zone family or a zone family without movement
                continue
            solverMotionDict = dict((I.getName(node), I.getValue(node))
                                    for node in I.getNodesFromType(solverMotion, 'DataArray_t'))
            rowFrame[I.getName(Family)] = dict(
                omega =solverMotionDict['omega'],
                center=(solverMotionDict['axis_pnt_x'], solverMotionDict['axis_pnt_y'], solverMotionDict['axis_pnt_z']),
                axis  =(solverMotionDict['axis_vct_x'], solverMotionDict['axis_vct_y'], solverMotionDict['axis_vct_z'])   
            )
        for zone in I.getZones(base):
            familyNode = I.getNodeFromType1(zone, 'FamilyName_t')
            if not familyNode: continue
            row = I.getValue(familyNode)
            angleDeg = rowFrame[row]['omega'] * currentTime * 180/np.pi
            T._rotate(zone, rowFrame[row]['center'], 
                            rowFrame[row]['axis'], 
                            angleDeg, 
                            vectors=vectors)


def makeMovie(FRAMES_DIRECTORY='.', filename='animation.gif', fps=24, width=400):
    '''
    Make an gif animation easily from pre-existing frames (must be named 'frame*.png')

    Parameters
    ----------
    FRAMES_DIRECTORY : str, optional
        Directory where the frames are, by default '.'
    filename : str, optional
        Name of the output file, by default 'animation.gif'
    fps : int, optional
        Number of frames per second, by default 24
    width : int, optional
        Width in pixels of the output animation file, by default 400
    '''

    # first, resize your images to the width size desired for final video (e.g. 600 px)
    os.system(
        f'for img in frame*.png; do convert -resize 600 -quality 100 "{FRAMES_DIRECTORY}/$img" "{FRAMES_DIRECTORY}/resized-$img"; done')
    
    # second, create movie with (increase fps for having faster motion)
    os.system(
        f'mencoder  mf://{FRAMES_DIRECTORY}/resized-frame*.png -mf fps={fps}  -ovc x264 -x264encopts subq=6:partitions=all:8x8dct:me=umh:frameref=5:bframes=3:b_pyramid=normal:weight_b -o movie.avi')

    # then convert movie to gif, by scaling to desired pixels (e.g. width 400 px)
    os.system(
        f'ffmpeg -i movie.avi -vf "fps=10,scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 {filename}')

def duplicateRows(t, setup=None, **RowsToDuplicate):
    '''
    Duplicate tree for visualization, for turbomachinery applications.

    Parameters
    ----------
    t : PyTree
        Input tree. It must be either a top tree or a base.

    setup : module, optional
        Read from ``setup.py``. If not given, the file ``setup.py`` is read in the current directory.

    RowsToDuplicate : kwargs
        Row families to duplicate. For each key, the value corresponds to the wished number of duplications.

    Examples
    --------

    .. code-block:: python

        import Converter.PyTree as C
        import MOLA.Visu as Visu 

        surfaces = C.convertFile2PyTree('OUTPUT/surfaces.cgns')
        Visu.duplicateRows(t, Rotor=3, Stator=5)

    Raises
    ------
    TypeError
        The input tree must be either a top tree or a base.
    '''

    import MOLA.WorkflowCompressor as WF

    if not setup:
        setup = J.load_source('setup', 'setup.py')

    if I.isTopTree(t) or I.getType(t) == 'CGNSBase_t':
        for row, nDupli in RowsToDuplicate.items():
            WF.duplicate(t, rowFamily=row, nBlades=setup.TurboConfiguration['Rows'][row]['NumberOfBlades'], 
                         nDupli=nDupli, verbose=1)
    else:
        raise TypeError('The input tree must be either a top tree or a base')


def prettyName(var):
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
        ('VelocityThetaRelDim',  '$W_\theta$'),
        ('VelocityThetaAbsDim', '$V_\theta$'),
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


def getRadialProfiles(surfaces='OUTPUT/surfaces.cgns'):
    '''
    Get the radial profiles into a dictionary. This function may feed :py:func:`getRadialProfiles`. 

    Parameters
    ----------
    surfaces : str or PyTree
        name of the CGNS file with a 'RadialProfiles' node (default is 'OUTPUT/surfaces.cgns'),
        or already read PyTree. 

    Returns
    -------
        RadialProfiles : dict
            Dictionary with all radial profiles in **surfaces**. The structure is 
            the following:

            .. code-block:: python
            
                RadialProfiles['Iso_X_0.1'] = {'MachNumberAbs': numpy.array(...), 
                                               'EntropyDim': numpy.array(...), 
                                               ...
                                               '.ExtractionInfo': {...},
                                               '.PlotParameters': {...}
                                            }
            
            Each iso-surface has an arbitrary number of quantities, plus additional optional 
            data with a key name starting with a dot '.':

                * '.ExtractionInfo': it is fill automatically if the corresponding node is found.
                
                * '.PlotParameters': the function lets it empty, but you may fill this node with
                  parameters used in the 'plot' function of matplotlib if you give the returned
                  **RadialProfiles** to :py:func:`getRadialProfiles`.
    '''

    if isinstance(surfaces, str):
        surfaces = C.convertFile2PyTree(surfaces)

    RadialProfiles = dict()
    RadialProfilesBase = I.getNodeFromName1(surfaces, 'RadialProfiles')
    for zone in I.getZones(RadialProfilesBase):
        surfaceName = I.getName(zone)
        RadialProfiles[surfaceName] = J.getVars2Dict(zone, Container='FlowSolution#Centers')
        extractionInfo = J.get(zone, '.ExtractionInfo')
        if extractionInfo:
            RadialProfiles[surfaceName]['.ExtractionInfo'] = extractionInfo
        
        for FS in I.getNodesFromNameAndType(zone, 'Comparison#*', 'FlowSolution_t'):
            comparedPlane = I.getName(FS).split('#')[-1]
            comparisonName = f'{surfaceName}#{comparedPlane}'
            RadialProfiles[comparisonName] = J.getVars2Dict(zone, Container=I.getName(FS))

    return RadialProfiles


def sortRadialProfiles(RadialProfiles):
    '''
    Sort radial profiles in two groups: 

    #. radial profiles on a single surface

    #. radial profiles resulting of the comparison of two surfaces (difference, quotient). For instance, 
       an isentropic efficiency profile.

    Parameters
    ----------
    RadialProfiles : dict
        as got from :py:func:`getRadialProfiles`

    Returns
    -------
    RadialProfilesSingleSurface, RadialProfilesComparison : :py:class:`tuple` of :py:class:`dict`
    '''

    RadialProfilesSingleSurface = dict()
    RadialProfilesComparison = dict()
    for surface, RadialProfilesOnSurface in RadialProfiles.items():
        if '#' in surface:
            RadialProfilesComparison[surface] = RadialProfilesOnSurface
        else:
            RadialProfilesSingleSurface[surface] = RadialProfilesOnSurface
    return RadialProfilesSingleSurface, RadialProfilesComparison

def plotRadialProfiles(RadialProfiles, filename='RadialProfiles.pdf', assemble=False, variablesToPlot=None):
    '''
    Plot radial profiles

    Parameters
    ----------
    RadialProfiles : dict
        as got from :py:func:`getRadialProfiles`.

        .. code-block:: python
            
                RadialProfiles['Iso_X_0.1'] = {'MachNumberAbs': numpy.array(...), 
                                               'EntropyDim': numpy.array(...), 
                                               ...
                                               '.ExtractionInfo': {...},
                                               '.PlotParameters': {...}
                                            }
            
        Each iso-surface has an arbitrary number of quantities, plus additional optional 
        data with a key name starting with a dot '.':

            * '.ExtractionInfo': information taken from a `.ExtractionInfo` node.
            
            * '.PlotParameters': additional parameters to be given to `matplotlib.pyplot.plot` function.
              '.PlotParameters' must be a :py:class:`dict`, it will be given as 'kwargs'.

        For each kind of variable, there may be a '.AxisProperty' argument that will be given to 
        the matplotlib axis object. For example, to set the limits of IsentropicEfficiency to 
        [0.7, 1.0], it must be set as:

        .. code-block:: python
            
                RadialProfiles['.AxisProperties']['IsentropicEfficiency'] = dict(xlim=[0.7, 1.0])
        
    filename : str, optional
        generic name of the file, by default 'RadialProfiles.pdf'

    assemble : bool, optional
        if True, write a unique PDF file with all the plots. By default False
    
    variablesToPlot : list
        Among the available data, plot only variables in this list. If :py:obj:`None`, plot all available data.

    '''
    
    try:
        os.makedirs(os.path.dirname(filename))
    except:
        pass

    filename_split = filename.split('.')
    filename_root = '.'.join(filename_split[:-1])
    extension = filename_split[-1]
    if not extension == 'pdf':
        assemble = False

    def getTextForFirstPage(RadialProfiles):
        '''Generate the text to fill the first page of the PDF file, with information on planes

        Parameters
        ----------
        RadialProfiles : dict
            as got from :py:func:`getRadialProfiles`

        Returns
        -------
        str
            text to write on the first page of the PDF file
        '''
        # First pages with infomation
        txt = 'RADIAL PROFILES\n\n'
        for plane, RadialProfilesOnPlane in RadialProfiles.items():
            ExtractionInfo = RadialProfilesOnPlane.get('.ExtractionInfo', {})
            PlotParameters = RadialProfilesOnPlane.get('.PlotParameters', dict(label=plane))
            if 'label' in PlotParameters:
                txt += PlotParameters['label'] + '\n'
            if ExtractionInfo:
                txt += f"IsoSurface {ExtractionInfo['field']} = {ExtractionInfo['value']}\n"
                try:
                    txt += f"{ExtractionInfo['ReferenceRow']} {ExtractionInfo['tag']}\n"
                except: 
                    pass
            txt += '\n'
        return txt

    if not assemble:

        for plane, RadialProfilesOnPlane in RadialProfiles.items():
            if plane.startswith('.'):
                continue
            variables2plot = []
            for var in RadialProfilesOnPlane:
                if not var in ['ChannelHeight', 'Gamma'] and not var.startswith('.'):
                    if (variablesToPlot is None) or (var in variablesToPlot):
                        variables2plot.append(var)
            
            PlotParameters = RadialProfilesOnPlane.get('.PlotParameters', dict(label=plane))

            for var in variables2plot:
                print(f'  > plot {var}')
                plt.figure()
                plt.plot(RadialProfilesOnPlane[var], RadialProfilesOnPlane['ChannelHeight'] * 100., **PlotParameters)
                plt.xlabel(prettyName(var))
                plt.ylabel('h (%)')
                plt.grid()
                plt.savefig(f'{filename_root}_{var}_{plane}.{extension}')
                plt.close()

    else:

        with PdfPages(filename) as pdf:

            textFirstPage = getTextForFirstPage(RadialProfiles)
            
            firstPage = plt.figure()
            firstPage.clf()
            firstPage.text(0.5, 0.5, textFirstPage, transform=firstPage.transFigure, size=12, ha="center", va="center")
            pdf.savefig()
            plt.close()

            # Assumption: same data on all plane
            variables2plot = []
            firstPlaneData = next(iter(RadialProfiles.values())) 
            for var in firstPlaneData:
                if not var in ['ChannelHeight', 'Gamma'] and not var.startswith('.'):
                    if (variablesToPlot is None) or (var in variablesToPlot):
                        variables2plot.append(var)
            
            AxisProperties = RadialProfiles.get('.AxisProperties', dict())

            for var in variables2plot:
                print(f'  > plot {var}')
                plt.figure()
                for plane, RadialProfilesOnPlane in RadialProfiles.items():
                    if plane.startswith('.'): 
                        continue
                    PlotParameters = RadialProfilesOnPlane.get('.PlotParameters', dict(label=plane))
                    if not var in RadialProfilesOnPlane: 
                        print(J.WARN + f'    not found on {plane}' + J.ENDC)
                        continue
                    plt.plot(RadialProfilesOnPlane[var], RadialProfilesOnPlane['ChannelHeight'] * 100., **PlotParameters)
                plt.xlabel(prettyName(var))
                plt.ylabel('h (%)')
                plt.grid()
                if len(RadialProfiles) > 1: 
                    plt.legend()
                if var in AxisProperties:
                    plt.gca().set(**AxisProperties[var])
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
