import matplotlib.pyplot as plt
from treelab import cgns

component_name = 'WING'
y_name = 'CL'
x_name = 'Iteration'
include_cost = True


for case in ['elsa','fast','sonics']:
    tree = cgns.load(f'example_{case}/OUTPUT/signals.cgns')
    integrals = tree.get("Integral")
    component = integrals.get(component_name)
    x, y = component.fields([x_name, y_name])

    if include_cost:
        total_real_time = sum([n.value() for n in tree.group('TotalRealTime')])        
        minutes = total_real_time/60.0
        case += " (%0.2f min)"%minutes

    plt.plot(x,y,label=case)

plt.legend(loc='best')
plt.xlabel(x_name)
plt.ylabel(y_name)
plt.ylim([0, 1])
plt.grid()
plt.tight_layout()
plt.savefig(f'isolated_wind_{y_name}_convergence_comparison.png')
plt.show()


