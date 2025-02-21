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

# TODO replace command os.system by subprocess.run
import os
import numpy as np

def xyz_to_pixel(points,win,posCam,posEye,dirCam,viewAngle=50.0):
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

def make_movie(FRAMES_DIRECTORY='FRAMES', fps=24, width=400,
        resize_images=True, movie_filename='movie.avi', gif_filename=''):
    '''
    Make an gif animation easily from pre-existing frames (must be named 'frame*.png')

    Parameters
    ----------
    FRAMES_DIRECTORY : str, optional
        Directory where the frames are, by default 'FRAMES'
    filename : str, optional
        Name of the output file, by default 'animation.gif'
    fps : int, optional
        Number of frames per second, by default 24
    width : int, optional
        Width in pixels of the output animation file, by default 400
    '''

    # first, resize your images to the width size desired for final video 
    if resize_images:
        os.system(
            f'for img in {FRAMES_DIRECTORY}/frame*.png; do convert -resize {width} -quality 100 "$img" "{FRAMES_DIRECTORY}/resized-${{img##*/}}"; done')
            # f'for img in {FRAMES_DIRECTORY}/frame*.png; do echo ${{img##*/}}; done')
        resized_image_name='resized-frame'
    else:
        resized_image_name=''

    # second, create movie with (increase fps for having faster motion)
    if gif_filename and not movie_filename:
        movie_filename = gif_filename.replace('.gif','.avi')
    os.system(
        f'mencoder  mf://{FRAMES_DIRECTORY}/{resized_image_name}*.png -mf fps={fps}  -ovc x264 -x264encopts subq=6:partitions=all:8x8dct:me=umh:frameref=5:bframes=3:b_pyramid=normal:weight_b -o {movie_filename}')

    if gif_filename:
        # then convert movie to gif, by scaling to desired pixels (e.g. width 400 px)
        os.system(
            f'ffmpeg -i {movie_filename} -vf "fps=10,scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 {gif_filename}')

