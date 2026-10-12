from mola import visu

for solver in [
    'elsa',
    # 'fast', # case crashes
    # 'sonics' # known BUG : Cassiopee is having trouble loading NFACE surface from sonics
    ]:
    example_name = 'example_had1_'+solver

    # set the different field elements to include in the image
    Elements = [
                dict(extraction_name='Iso_Z_0',
                    color='Iso:MomentumX', colormap='Diverging', shadow=False, levels=[16, 0, 150]),
                dict(extraction_name='BLADE', 
                    color='Iso:Pressure', colormap='Viridis', shadow=False, levels=[12, 60e3, 110e3]),
                ]

    # Create a figure
    fig = visu.Figure(
        filename=f'flow_{example_name}.png',
        camera=dict( 
            posCam=(0.2266, 0.4293, 1.459), 
            posEye=(0.2266, 0.4293, -0.0536), 
            dirCam=(0.0, 1.0, 0.0),
        ),
        Elements=Elements
        )


    # generate the CPlot image of the field elements (will write an image)
    fig.plot_surfaces(
        surfaces=f'{example_name}/OUTPUT/extractions.cgns',
        default_vertex_container='FlowSolution#Output@Vertex',
        default_centers_container='FlowSolution#Output@Center')
    # include matplotlib components, such as colorbar and a 2D curve plot
    fig.add_colorbar(
        field_name='MomentumX', colorbar_title = r'$\bf{\rho V_x}$ (m/s)',
        orientation='horizontal', center=(0.5,0.1), width=0.025, length=0.55
        )
    fig.add_colorbar(
        field_name='Pressure', colorbar_title = r'$\bf{P_s}$ (Pa)',
        orientation='vertical', center=(0.9,0.5), width=0.025, length=0.5
        ) 

    # Plot convergence
    ax = fig.plot_signals(
        signals=f'{example_name}/OUTPUT/signals.cgns',
        left=0.06, right=0.3, bottom=0.6, top=0.9,
        ylabel='Thrust (N)',
        background_opacity=0.0,
        curves=[
            dict(zone_name='BLADE',x='Iteration',y='Thrust', include_last_point_label=True),
            ]
        )

    fig.save()
