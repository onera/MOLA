import Converter.PyTree as C
import MOLA.LiftingLine as LL

t = C.convertFile2PyTree('Polar.cgns')
LL.plotStructPyZonePolars(t, addiationalQuantities=['std-CL','Cd'],
                             filesuffix='_orig')