import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from treelab import cgns

component_name = 'BLADE'
y_name = 'Thrust'
x_name = 'Iteration'
include_cost = True


signal_files = sorted(Path('.').glob('example_*/OUTPUT/signals.cgns'))
for signal_file in signal_files:
    case = signal_file.parent.parent.name[len('example_'):]
    tree = cgns.load(str(signal_file))
    integrals = tree.get("Integral")
    component = integrals.get(component_name)
    x, y = component.fields([x_name, y_name])

    if include_cost:
        total_real_time = sum([n.value() for n in tree.group('TotalRealTime')])        
        hours = total_real_time/3600.0
        case += " (%0.2f h)"%hours

    plt.plot(x,y,label=case)

plt.legend(loc='best')
plt.xlabel(x_name)
plt.ylabel(y_name)
plt.grid()
plt.tight_layout()
plt.savefig(f'light_propeller_{y_name}_convergence_comparison.png')
if os.environ.get('MOLA_HEADLESS', '').lower() not in {'1', 'true', 'yes'}:
    plt.show()
