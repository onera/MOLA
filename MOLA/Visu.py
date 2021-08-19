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
NOTES:

Command lines for animations :

# By Ronan
mencoder  mf://@'+str(Imglist)  -mf fps=24  -ovc x264 -x264encopts subq=6:partitions=all:8x8dct:me=umh:frameref=5:bframes=3:b_pyramid=normal:weight_b -o fileout.avi

# Resize frames
for f in Frames:
    os.system('convert "%s" -resize %dx%d -quality 100 "%s"'%(f,WidthPixels,WidthPixels,f))

# Make animation
os.system('convert   -delay %d   -loop 0 %s  %s'%(delay,FrameWildCard,OutputFilename))

# Using ffmpeg
os.system('ffmpeg -i Frame%06d.png -vf palettegen palette.png -y') # make palette
os.system('ffmpeg -framerate 25 -i Frame%06d.png -i palette.png -lavfi paletteuse Animation.gif -y') # make animation
'''


def xyz2Pixel(points,win,posCam,posEye,dirCam,viewAngle):
    '''
    Returns the two-component image-pixel positions of a set of points
    located in the 3D world of CPlot.

    INPUTS

    points (list of 3-float tuples) - x,y,z coordinates of point in 3D world

    win (2-int tuple) - Window resolution in pixels

    posCam (3-float tuple) - position of Camera (see CPlot doc)

    posEye (3-float tuple) - position of eye (see CPlot doc)

    dirCam (3-float tuple) - direction of Camera (see CPlot doc)

    viewAngle (float) - angle of Camera (see CPlot doc)

    OUTPUTS

    width, height - (2-float tuple) - width and height in pixels using
        the convention of origin located at upper left side of image

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

    INPUTS
    Frames - (list of strings) - List of paths where frames are located

    WidthPixels - (int) - Width pixels of final movie (keeping aspect ratio)

    FrameWildCard (string) - Wildcard used for detecting the frame files

    delay (float) delay between frames

    OutputFilename (string) - final file produced by this function

    RemoveFramesAtEnd (boolean) - if True, systematically remove Frames
        after movie creation

    OUTPUTS
    None
    '''
    for f in Frames:
        os.system('convert "%s" -resize %dx%d -quality 100 "%s"'%(f,WidthPixels,WidthPixels,f))
    os.system('convert   -delay %d   -loop 0 %s  %s'%(delay,FrameWildCard,OutputFilename))
    if RemoveFramesAtEnd:
        for f in Frames:
            try: os.remove(f)
            except OSError: pass
