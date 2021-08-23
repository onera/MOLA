'''
             --------------------------------
               BEMT PROPELLER POLAR EXAMPLE             
             --------------------------------

This script describes the procedure for performing BEMT 
propeller's polar* computation.

*: polar in the sense of propeller (change in pitch), not 
   in the airfoil's sense !

DISCLAIMER:
This script is intended to be applied AFTER successful 
execution of script BEMT_MinimalEx.py, because this example
also shows how to re-use Polar and LiftingLine previously
saved data.

Thus, several steps are described in this script:

STEP 1 - Load Lifting Line and Polars CGNS objects

STEP 2 - Prepare the sweep of parameters

STEP 3 - Perform BEMT sweep computations

STEP 4 - Save the results

STEP 5 - Plot the results

Previous knowledge required by the user:
-> BEMT_MinimalEx.py

File history
19/04/2020 - v1.1 - L. Bernardos - Minor enhancements
13/04/2020 - v1.0 - L. Bernardos - Adapted to MOLA v1.7
'''

# System modules
import numpy as np
import sys

# TODO: continue
print("WORK IN PROGRESS")
sys.exit()

# Cassiopee modules
import Converter.PyTree   as C
import Converter.Internal as I

# MOLA modules (recent version of Scipy is required)
import MOLA.InternalShortcuts  as J
import MOLA.PropellerAnalysis  as PA
import MOLA.LiftingLine  as LL

# -------------------------------------------------- #
#  STEP 1: LOAD LIFTING LINE AND POLARS CGNS OBJECTS
# -------------------------------------------------- #

'''
We recycle the data generated in BEMTminimalExample.py
'''

# Load the polars
PyZonePolars = C.convertFile2PyTree('MyPolars.cgns')
PyZonePolars = I.getZones(PyZonePolars)

# Build the Polars interpolator functions
PolarsInterpFuns = LL.buildPolarsInterpolatorDict(PyZonePolars)

# Load the Lifting Line object
LiftingLine = C.convertFile2PyTree('MyBladeLL.cgns')
LiftingLine = I.getZones(LiftingLine)[0]

# We make sure LiftingLine twist is set conventional
LL.resetPitch(LiftingLine,ZeroPitchRelativeSpan=0.75)

# -------------------------------------------------- #
#  STEP 2: PREPARE THE SWEEP OF PARAMETERS
# -------------------------------------------------- #
'''
In this step, we prepare the sweep of parameters.

Let us suppose that we want to make polars based on three
different variables: Pitch, Velocity and MachTip. We are 
interested in storing and plotting, say, Thrust and the 
Propulsive Efficiency.

Thus, we will be handling 3D matrices (3-variables).
In this step, we allocate them:
'''

# We declare the 1D vectors of the sweep variables ranges
PitchRange    = np.linspace(5., 40., 36)  # Three different ways
VelocityRange = np.arange(10., 40.,  5.)  # of declaring the
MachTipRange  = np.array([0.5, 0.6, 0.7]) # variables is shown

# We construct the 3-dimensional arrays
Pitches, Velocities, MachTips = np.meshgrid(PitchRange, VelocityRange, MachTipRange)

# We allocate the result variables:
DesiredVariables = ('Thrust','PropulsiveEfficiency')
AllResults = {}
for dv in DesiredVariables: AllResults[dv] = Pitches*0. 

# We are almost done. Now, we construct convenient flat VIEWS
# of the previous arrays:
P_ = Pitches.ravel(order='K')    # | These are VIEWS, not new
V_ = Velocities.ravel(order='K') # | arrays !!
M_ = MachTips.ravel(order='K')   # |


# -------------------------------------------------- #
#  STEP 3: PERFORM BEMT SWEEP COMPUTATIONS
# -------------------------------------------------- #

# For computing Mach Tip we need to know the blade span and
# the temperature, so:
r, = J.getVars(LiftingLine,['Span'])
Rmax = r.max()
Temperature = 288.

# Let us launch the computationS:
Ncomp = len(P_)
for i in range(Ncomp):

    # Translate MachTip to RPM
    RPM = (30/np.pi)*M_[i]*np.sqrt(1.4*287.*Temperature)/Rmax

    Sol = PA.computeBEMT(
    LiftingLine, PolarsInterpFuns, # Mandatory inputs

    Velocity=V_[i],
    Temperature=Temperature,
    RPM = RPM,
    Constraint='Pitch',
    ConstraintValue=P_[i],

    # VERY IMPORTANT: We recommend using FailedAsNaN=True.
    # In this way, if a computation yield poor convergence
    # (due to extreme brake or deep stall, the computation
    # returns a NaN value)
    FailedAsNaN=True, 
    )

    print ("(%d/%d) Pitch=%0.4f  Mtip=%0.3f V=%0.3f | T=%g, eta=%g"%(i+1,Ncomp,P_[i],M_[i],V_[i],Sol['Thrust'],Sol['PropulsiveEfficiency']))

    # Store the results
    for var in AllResults:
        RavelView_ = AllResults[var].ravel(order='K')
        RavelView_[i] = Sol[var]


# -------------------------------------------------- #
#  STEP 4: SAVE THE RESULTS
# -------------------------------------------------- #
'''
All results are contained in the dictionary <AllResuts>.
In this step, we show how to easily save the results in 
a file.

For this, we use an auxiliary PyTree Zone.
'''

Arrays = [AllResults[var] for var in AllResults]
Names  = [var for var in AllResults]

# We add the sweep variables in the zone
Arrays += [Pitches, Velocities, MachTips]
Names += ['Pitch', 'Velocity', 'MachTip']

ResultsZone = J.createZone('AllSweepResults',Arrays,Names)

# Save the results in a file
C.convertPyTree2File(ResultsZone,'SweepResults.cgns')


# -------------------------------------------------- #
#  STEP 5: PLOT THE RESULTS
# -------------------------------------------------- #
'''
In this step we show how to read a previously saved file
containing the sweep results, and an example of a plot.
'''

# First, we load the sweep results file
t = C.convertFile2PyTree('SweepResults.cgns')
ResultsZone = I.getZones(t)[0]

# We extract the variables contained in ResultsZone as a
# Python dictionary named <Sol>
VarNames = C.getVarNames(ResultsZone)[0]
Sol = J.getVars2Dict(ResultsZone, VarNames)

# Now we make, for instance, a sequence of plots of eta(Thrust)
# containing all MachTips, and that, for each advance velocity

# print(Sol['Velocity'][:,0,0]) # Varies with i-index
# print(Sol['Pitch'][0,:,0])    # Varies with j-index
# print(Sol['MachTip'][0,0,:])  # Varies with k-index

Velocities  = Sol['Velocity'][:,0,0]
MachTips    = Sol['MachTip'][0,0,:]

nVels  = len(Velocities)
nMachs = len(MachTips)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

for k in range(nMachs):
    
    plt.cla() # clear the axis

    for i in range(nVels):
        # Find corresponding Thrust vector
        Thrust = Sol['Thrust'][i,:,k]

        # Keep not-NaN values
        notNaNValues = np.logical_not(np.isnan(Thrust))
        Thrust = Thrust[notNaNValues]

        # Keep Thrust>0 values
        PositiveVals = Thrust>0
        Thrust = Thrust[PositiveVals]

        # Find Propulsive Efficiency vector
        eta = Sol['PropulsiveEfficiency'][i,:,k][notNaNValues][PositiveVals]

        # plot the curve
        ax.plot(Thrust, eta, '.-', label='V=%0.2f m/s'%Velocities[i])

    ax.set_xlabel('Thrust (N)')
    ax.set_ylabel('Prop. efficiency $\\eta$')
    ax.legend(loc='best')
    plt.tight_layout()
    figurefilename = 'Sweep_Mtip%0.2f.pdf'%MachTips[k]
    print("Saving figure %s ..."%figurefilename)
    fig.savefig('Sweep_Mtip%0.2f.pdf'%MachTips[k])
    print("ok")



