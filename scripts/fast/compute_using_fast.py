import Fast.PyTree as Fast
import FastS.Mpi as FastS
import FastC.PyTree as FastC
import Converter.Mpi as Cmpi
rank =Cmpi.rank

FastC.MX_SYNCHRO= 25000

inititer = 1 
niter = 2000

theta = 0.0
omega = 209.43951023931953 # rad/s

rho = 1.0
U = 100
nb_blades = 8

PressureDynamic = 0.5 * rho * U**2
LengthReference = 1.0


t = FastC.loadTree(fileName='main.cgns', split='single')
tc, graph  = FastC.loadTree(fileName='tc.cgns', split='single', graph=True)

numb = { 'temporal_scheme':'implicit', 'ss_iteration':5,'omp_mode':1}
numz = {
        'prandtltb':0.9,
        'ransmodel':'SA',
        'ratiom':10000.0,
        'scheme':'roe_min',
        'psiroe':0.01,
        'slope':'o3',
        'nb_relax':1,
        'epsi_newton': 0.01,
        'time_step':4.166666666666667e-05,
        'time_step_nature': 'global',
        'motion':'rigid',
        'rotation':[-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        }

Fast._setNum2Zones(t, numz) ; Fast._setNum2Base(t, numb)

(t, tc, metrics) = FastS.warmup(t, tc, graph=graph, verbose=1,infos_ale=(theta, omega))

stress_tree = FastS.createStressNodes(t, ["BLADE"])
stresses = [[],[],[],[],[],[],[],[],[],[],[]]
Powers=[]
Thrusts=[]
for it in range( inititer-1, inititer+niter-1 ):
    if rank==0: print(f'iteration {it}/{inititer+niter-2}')
    FastS._compute(t, metrics, it, tc=tc, graph=graph, vtune=True)

    FastS.display_temporal_criteria(t, metrics, it, format='store') 
    current_stress = FastS._computeStress(t, stress_tree, metrics)
    for j in range(len(current_stress)):
        stresses[j].append(current_stress[j])
    if rank==0:
        fx, fy, fz, t0x, t0y, t0z, S, m, ForceX, ForceY, ForceZ = stresses

        torque_coef = PressureDynamic * S[0] * LengthReference

        Power = nb_blades *  torque_coef * t0x[-1] * omega
        Thrust = -nb_blades * ForceX[-1]

        print(f"Thrust = {Thrust} N    Power = {Power} W")

        Thrusts += [Thrust]
        Powers += [Power]
    

if rank==0:
    fx, fy, fz, t0x, t0y, t0z, S, m, ForceX, ForceY, ForceZ = stresses
    print(f'ForceX = {ForceX[-1]}')
    print(f'ForceY = {ForceY[-1]}')
    print(f'ForceZ = {ForceZ[-1]}')
    print(f't0x = {t0x[-1]}')
    print(f't0y = {t0y[-1]}')
    print(f't0z = {t0z[-1]}')
    print(f'S = {S[0]}')

    print(f"Thrust = {Thrust} N    Power = {Power} W")

    with open('loads.dat','w') as f:
        f.write('# Thrust(N) Power(W)\n')
        for th, po in zip(Thrusts, Powers):
            f.write(f'{th} {po}\n')

Cmpi.barrier()