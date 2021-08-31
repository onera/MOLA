from timeit import default_timer as tic

import Converter.PyTree as C
import Generator.PyTree as G

from MOLA import Fields

zone = G.cart((0,0,0),(1,1,1),(10,10,10))
Fields.new(zone,['field1','field2','field3','field4'])

v = Fields.get(zone, OutputObject='dict')

v['field1'] += 1.
v['field2'] += 2.
v['field3'] += 3.
v['field4'] += 4.

v2 = Fields.coordinates(zone, OutputObject='dict', AtCenters=True)

v.update(v2)

for i in v:
    print('field: %s shape %s'%(i,str(v[i].shape)))

C.convertPyTree2File(zone, 'test.cgns')
