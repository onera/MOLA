import MOLA.LiftingLine as LL

t = LL.C.convertFile2PyTree('Polar.cgns')
LL.plotStructPyZonePolars(t, addiationalQuantities=['std-CL','Cd'],
                             filesuffix='_original')
