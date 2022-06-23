'''
MOLA - Visu.py

VISU MODULE

Collection of routines and functions designed for visualization
techniques.

File history:
12/06/2020 - v1.8.01 - L. Bernardos - Creation
'''

import sys
import os
import numpy as np

'''
USEFUL NOTES:

Command lines for animations:


# By Ronan
mencoder  mf://@'+str(Imglist)  -mf fps=24  -ovc x264 -x264encopts subq=6:partitions=all:8x8dct:me=umh:frameref=5:bframes=3:b_pyramid=normal:weight_b -o fileout.avi

# By Rocco
mencoder mf://FRAMES/FrameIsoY*.png -mf w=800:h=600:fps=10:type=png -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o animIsoY.avi


# Resize frames
for f in Frames:
    os.system('convert "%s" -resize %dx%d -quality 100 "%s"'%(f,WidthPixels,WidthPixels,f))

# Make animation
convert -delay 50 -loop 0 frame* animation.gif

# Using ffmpeg
os.system('ffmpeg -i Frame%06d.png -vf palettegen palette.png -y') # make palette
os.system('ffmpeg -framerate 25 -i Frame%06d.png -i palette.png -lavfi paletteuse Animation.gif -y') # make animation

'''


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

def makeAnimation(Frames, WidthPixels=700, FrameWildCard='Frame*', delay=10,
                  OutputFilename='Animation.gif', RemoveFramesAtEnd=True):
    '''
    Make an animated gif file using a set of previously saved frames.

    Parameters
    ----------

        Frames : :py:class:`list` of :py:class:`str`
            List of paths where frames are located

        WidthPixels : int
            Width pixels of final movie (keeping aspect ratio)

        FrameWildCard : str
            Wildcard used for detecting the frame files

        delay : float
            delay between frames

        OutputFilename : str
            final file produced by this function

        RemoveFramesAtEnd : bool
            if :py:obj:`True`, systematically remove Frames after movie creation
    '''
    for f in Frames:
        os.system('convert "%s" -resize %dx%d -quality 100 "%s"'%(f,WidthPixels,WidthPixels,f))
    os.system('convert   -delay %d   -loop 0 %s  %s'%(delay,FrameWildCard,OutputFilename))
    if RemoveFramesAtEnd:
        for f in Frames:
            try: os.remove(f)
            except OSError: pass
