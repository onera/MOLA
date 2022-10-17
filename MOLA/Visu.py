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

import sys
import os
import numpy as np
from time import sleep

import Converter.PyTree as C
import Converter.Internal as I

from . import InternalShortcuts as J


def xyz2Pixel(points,win,posCam,posEye,dirCam,viewAngle):
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
                       colormap='Viridis', levels=[200,'min','max'],
                       additionalDisplayOptions={})]):

    machine = os.getenv('MAC', 'ld')
    if machine in ['spiro', 'sator']:
        offscreen=1 # TODO solve bug https://elsa.onera.fr/issues/10536
    elif machine in ['ld', 'visung', 'visio']:
        offscreen=3
    else:
        raise SystemError('machine "%s" not supported.'%machine)

    cmap2int = dict(Blue2Red=1, Green2Red=3, Diverging=9, Grey2White=15,
                    Viridis=17, Inferno=19, Magma=21, Plasma=23, Blue=25)

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
        try: return elt['blending'] > 0
        except: return False

    Trees = []
    TreesBlending = []
    for i, elt in enumerate(Elements):
        try: selection = elt['selection']
        except KeyError: selection = {}
        zones = J.selectZones(t, **selection)

        if hasBlending(elt): zones = C.convertArray2Hexa(zones) # see cassiopee #8740

        for z in zones:
            CPlot._addRender2Zone(z, material=elt['material'],color=elt['color'],
                                     blending=elt['blending'])

        if hasBlending(elt):
            TreesBlending += [ C.newPyTree(['blend.%d'%i, zones]) ]
        else:
            Trees += [ C.newPyTree(['elt.%d'%i, zones]) ]

    # requires to append blended zones (see cassiopee #8740 and #8748)
    if TreesBlending:
        for i in range(Trees):
            Trees[i] = I.merge([Trees[i]]+TreesBlending)


    for i in range(len(Trees)):
        tree = Trees[i]
        elt = Elements[i]
        
        if elt['color'].startswith('Iso:'):
            field_name = elt['color'].replace('Iso:','')
            levels = elt['levels']
            levels[2] = C.getMinValue(tree, field_name) if levels[2] == 'min' else float(levels[2])
            levels[3] = C.getMaxValue(tree, field_name) if levels[3] == 'max' else float(levels[3])
            isoScales = [[field_name, levels[1], levels[2], levels[3]]]
        else:
            isoScales = []

        if i == len(Trees)-1: offscreen += 1

        try: additionalDisplayOptions = elt['additionalDisplayOptions']
        except: additionalDisplayOptions = {}

        CPlot.display(tree, offscreen=offscreen, colormap=cmap2int[elt['colormap']],
                        isoScales=isoScales, **DisplayOptions, **additionalDisplayOptions)
        CPlot.finalizeExport(offscreen)

        if 'backgroundFile' not in additionalDisplayOptions:
            MOLA = os.getenv('MOLA')
            MOLASATOR = os.getenv('MOLASATOR')
            for MOLAloc in [MOLA, MOLASATOR]:
                backgroundFile = os.path.join(MOLAloc,'MOLA','GUIs','background.png')
                if os.path.exists(backgroundFile):
                    CPlot.setState(backgroundFile=backgroundFile)
                    CPlot.setState(bgColor=13)
                    break
    
        sleep(0.5)