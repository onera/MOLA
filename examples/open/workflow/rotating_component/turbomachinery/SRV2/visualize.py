import os
from treelab import cgns
from mola import naming_conventions as names
from mola import visu

solver = 'elsa'
example_name = 'example_srv2_'+solver


# set the different field elements to include in the image
Elements = [
            dict(extraction_name='HUB', material='Chrome'), 
            dict(extraction_name='Impeller_Main_Blade', material='Chrome'), 
            dict(extraction_name='Impeller_Splitter_Blade', material='Chrome'), 
            dict(extraction_name='Iso_H_0.5', color='Iso:Pressure', colormap='Diverging', levels=[22, 90e3, 300e3]),  #, blending=0.5),
            ]

# Create a figure
fig = visu.Figure(
    filename=f'flow_{example_name}.png',
    camera=dict( 
        posCam=(-0.24592463703457942, 0.5121130031377165, -0.0030930765364429007), 
        posEye=(0.10981445197145112, 0.14571377306643096, 0.03191791003196333), 
        dirCam=(0.21665287388276538, 0.11774582223679872, -0.9691220013930758)
    ),
    Elements=Elements
    )

# Read and duplicate the solution
surfaces = cgns.load(os.path.join(example_name,names.DIRECTORY_OUTPUT, names.FILE_OUTPUT_2D))
surfaces = visu.duplicate_rows(surfaces, Impeller=dict(NumberOfDuplications=12, NumberOfBlades=13))

# generate the CPlot image of the field elements (will write an image)
fig.plot_surfaces(surfaces)
fig.add_colorbar(
    field_name='Pressure', colorbar_title = r'$\bf{P_s}}$ (Pa)',
    orientation='vertical', center=(0.6,0.5), width=0.025, length=0.55
    )

# Plot convergence
ax = fig.plot_signals(
    signals=os.path.join(example_name,names.DIRECTORY_OUTPUT, names.FILE_OUTPUT_1D),
    left=0.75, right=0.98, bottom=0.3, top=0.7,
    ylabel=r'$\dot{m}$ (kg/s)',
    background_opacity=0.0,
    curves=[
        dict(zone_name='Impeller_INFLOW',x='Iteration',y='MassFlow', multiply_by=-1),
        dict(zone_name='Impeller_OUTFLOW',x='Iteration',y='MassFlow'),
        ],
    )

fig.save()
