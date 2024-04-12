#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import MOLA.Postprocess as POST

iterations, dualIterations, residuals = POST.getSubIterations()
firstIteration = iterations[0]

########## USER PARAMETERS #################
# Change the range of iterations to plot
iterationsSubRange = iterations[::100]
#####  END OF USER PARAMETERS ##############

# DISPLAY
cmap = matplotlib.colormaps['copper']
norm = matplotlib.colors.Normalize(vmin=np.min(iterationsSubRange), vmax=np.max(iterationsSubRange))

plt.figure()
for iteration in iterationsSubRange:
    plt.plot(dualIterations[iteration-firstIteration], residuals[iteration-firstIteration], color=cmap(norm(iteration)))

plt.xlabel('dual iteration')
plt.ylabel('residual')
plt.xlim(left=1)
plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

plt.grid()
plt.show()  
    