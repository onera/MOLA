from mola import visu

from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# set the different field elements to include in the image
Elements = [
            dict(extraction_name='Iso_H_0.9',
                color='Iso:MachNumberRel', colormap='Diverging', shadow=False, levels=[21, 0.4, 1.6]) #, iso_line=1),
            ]

# Create a figure
fig = visu.Figure(
    filename='flow_r37.png',
    camera=dict( 
        posCam=(0.03, 0.38, -0.01), 
        posEye=(0.02, 0.2, -0.01), 
        dirCam=(0.0, 0.0, -1.0), 
    ),
    Elements=Elements
    )


# generate the CPlot image of the field elements (will write an image)
fig.plot_surfaces()
# include matplotlib components, such as colorbar and a 2D curve plot
fig.add_colorbar(
    field_name='MachNumberRel', colorbar_title = r'$\bf{M_{rel}}$ (-)',
    orientation='horizontal', center=(0.5,0.9), width=0.025, length=0.55
    )

# Plot convergence
ax = fig.plot_signals(left=0.6, right=0.92, bottom=0.5, top=0.8,
            ylabel=r'$\dot{m}$ (kg/s)',
            background_opacity=0.0,
            curves=[
                dict(zone_name='R37_INFLOW',x='Iteration',y='MassFlow', multiply_by=-1),
                dict(zone_name='R37_OUTFLOW',x='Iteration',y='MassFlow'),
                ],
            ylim=[19,22]
            )

fig.save()
