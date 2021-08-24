from timeit import default_timer as tic

import Converter.PyTree as C
import Generator.PyTree as G

import MOLA.Fields as Fields

zone = G.cart((0,0,0),(1,1,1),(1000,1000,1000))
toc = tic()
for i in range(20):
    print(i)
    Fields.new(zone,['field1','field2','field3','field4'])
ET = tic() - toc
print('Initialization of fields using Fields.new(): %g s'%ET)
C.convertPyTree2File(zone, 'FieldsNew.cgns')

zone = G.cart((0,0,0),(1,1,1),(1000,1000,1000))
toc = tic()
for i in range(20):
    print(i)
    for fn in ['field1','field2','field3','field4']:
        C._initVars(zone,fn,0)
ET = tic() - toc
print('Initialization of fields using C._initVars(): %g s'%ET)
C.convertPyTree2File(zone, 'initVars.cgns')
