
import numpy as np
from mola import visu

first_iteration = 5000
delta_iteration = 10
last_iteration  = 5500

iterations = np.arange(first_iteration, last_iteration+delta_iteration, delta_iteration)

for i in iterations:

    # set the different field elements to include in the image
    Elements = [
            dict(extraction_name='Iso_Z_0.0005', color='Iso:VelocityY', colormap='Diverging', 
                 levels=[21, -50, 50], shadow=False),
            ]

    fig = visu.Figure(
        window_in_pixels = (1200, 800),
        dpi = 150,
        # set the camera view, the image size (in pixels) and the output image filename
        camera = dict(
            posCam=(0.003890372502600852, -8.089132321008139e-05, 0.009174394542279805), 
            posEye=(0.003890372502600852, -8.089132321008139e-05, 0.0005), 
            dirCam=(0.0, 1.0, 0.0), 
            ),
        Elements = Elements, 
        filename = f'FRAMES/frame{i:06d}.png',
        )

    # generate the CPlot image of the field elements (will write an image)
    fig.plot_surfaces(f'OUTPUT/extractions_AfterIter{i}.cgns', default_vertex_container='FlowSolution#Output')

    # include matplotlib components, such as colorbar and a 2D curve plot
    ax = fig.add_colorbar(field_name='VelocityY', orientation='vertical', center=(0.87,0.5),
                width=0.025, length=0.8, font_color='black',
                colorbar_title=r'$\bf{V_x}$')

    # save image
    fig.save()


print('making movie...')
visu.make_movie(gif_filename='animation.gif', fps=24, width=800)
