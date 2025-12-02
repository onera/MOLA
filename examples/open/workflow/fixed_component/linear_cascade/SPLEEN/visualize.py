from mola import visu

from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# set the different field elements to include in the image
Elements = [
            dict(extraction_name='MidSpan',
                color='Iso:MachNumberAbs', colormap='Diverging', shadow=False, levels=[16, 0.15, 1], iso_line=1),
            ]

# Create a figure
fig = visu.Figure(
    filename='flow_spleen.png',
    camera=dict( 
        posCam=(0.033, -0.1, -0.004), 
        posEye=(0.033, 0.001, -0.004), 
        dirCam=(0.0, 0.0, 1.0)
    ),
    Elements=Elements
    )


# generate the CPlot image of the field elements (will write an image)
fig.plot_surfaces() #default_vertex_container='FlowSolution#EndOfRunV')
# include matplotlib components, such as colorbar and a 2D curve plot
fig.add_colorbar(
    field_name='MachNumberAbs', colorbar_title = r'$\bf{M}$ (-)',
    orientation='horizontal', center=(0.5,0.9), width=0.025, length=0.55
    )

# Plot convergence
ax = fig.plot_signals(left=0.7, right=0.98, bottom=0.5, top=0.8,
            ylabel=r'$\dot{m}$ (kg/s)',
            background_opacity=0.0,
            curves=[
                dict(zone_name='SPLEEN_INFLOW',x='Iteration',y='MassFlow', multiply_by=-1),
                dict(zone_name='SPLEEN_OUTFLOW',x='Iteration',y='MassFlow'),
                ]
            )

fig.save()
