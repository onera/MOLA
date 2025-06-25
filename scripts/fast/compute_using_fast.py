import Fast.PyTree as Fast
import FastS.PyTree as FastS

inititer = 1 
niter = 2000

theta = 0.0
omega = 209.43951023931953 # rad/s

t, tc, ts, graph = Fast.load('t.cgns', 'tc.cgns')
t, tc, metrics = FastS.warmup(t, tc, graph, infos_ale=(theta, omega))

stress_tree = FastS.createStressNodes(t, ["BLADE"])
stresses = [[],[],[],[],[],[],[],[],[],[],[]]
for it in range( inititer-1, inititer+niter-1 ):
    print(f'iteration {it}/{inititer+niter-2}')
    FastS._compute(t, metrics, it, tc, graph)
    FastS.display_temporal_criteria(t, metrics, it, format='store') 
    FastS._calc_global_convergence(t)
    current_stress = FastS._computeStress(t, stress_tree, metrics)
    for j in range(len(current_stress)):
        stresses[j].append(current_stress[j])
    print(f'Force = ({current_stress[8]}, {current_stress[9]}, {current_stress[10]}) N')


Fast.save(t, fileName='restart.cgns', tc=tc, fileNameC='tc_restart.cgns', ts=ts, fileNameS='tstat_restart.cgns', split='single', compress=0)

fx, fy, fz, t0x, t0y, t0z, S, m, ForceX, ForceY, ForceZ = stresses

print(f'ForceX = {ForceX}')
print(f'ForceY = {ForceY}')
print(f'ForceZ = {ForceZ}')
print(f't0x = {t0x}')
print(f't0y = {t0y}')
print(f't0z = {t0z}')
print(f'S = {S[0]}')