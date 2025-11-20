
import numpy as np
from mola import visu

from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size >2:
    import os
    os.environ['OMP_NUM_THREADS'] = '1'

# set the different field elements to include in the image
Elements = [
        dict(extraction_name='MySurface', color='Iso:VelocityX', colormap='Diverging', shadow=False),
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
    # filename = f'FRAMES/frame{i:06d}.png',
    filename = 'flow.png',
    )

# generate the CPlot image of the field elements (will write an image)
# fig.plotSurfaces('OUTPUT/surfaces_AfterIter%d.cgns'%i)
fig.plot_surfaces(default_vertex_container='FlowSolution#CentersV')

# include matplotlib components, such as colorbar and a 2D curve plot
ax = fig.add_colorbar(field_name='VelocityX', orientation='vertical', center=(0.87,0.5),
            width=0.025, length=0.8, font_color='black',
            colorbar_title=r'$\bf{V_x}$')

# save image
fig.save()

