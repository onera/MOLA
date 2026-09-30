from mola import visu

from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

solver = 'elsa'
example_name = 'example_vega2_'+solver


if size >2:
    import os
    os.environ['OMP_NUM_THREADS'] = '1'

# set the different field elements to include in the image
Elements = [
        dict(extraction_name='Iso_H_0.9', color='Iso:MachNumberRel', colormap='Diverging', shadow=False,
             levels=[12, 0, 1.1], iso_line=1),
        ]

fig = visu.Figure(
    window_in_pixels = (1200, 800),
    dpi = 300,
    # set the camera view, the image size (in pixels) and the output image filename
    camera = dict( 
        posCam=(0.072, 0.017, 0.45),
        posEye=(0.072, 0.006, 0.2), 
        dirCam=(0.0, 1.0, 0.0), 
    ),
    Elements = Elements, 
    filename = f'flow_{example_name}.png',
    )

# generate the CPlot image of the field elements (will write an image)
# fig.plotSurfaces('OUTPUT/surfaces_AfterIter%d.cgns'%i)
fig.plot_surfaces(
    f'{example_name}/OUTPUT/extractions.cgns',
    default_vertex_container='FlowSolution#Output@Vertex')

# include matplotlib components, such as colorbar and a 2D curve plot
ax = fig.add_colorbar(
    field_name='MachNumberRel', 
    # orientation='vertical', center=(0.87,0.5), width=0.025, length=0.8, 
    orientation='horizontal', center=(0.5,0.9), width=0.025, length=0.8, 
    font_color='black',
    colorbar_title=r'$\bf{M_{rel}}$'
    )

# save image
fig.save()
