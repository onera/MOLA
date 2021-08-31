import Converter.PyTree as C
import Transform.PyTree as T
import Converter.Internal as I
import Geom.PyTree as D
import Post.PyTree as P
import Generator.PyTree as G

foils = []
for i in xrange(10): foils += [T.translate(D.naca('44%02d'%(10+i)),(0,0,i))]

wing = G.stack(foils)

wing = C.convertArray2Tetra(wing)
eF = P.exteriorFaces(wing)
eF = T.splitManifold(eF)
Tips = []
for e in eF:
    try: Tips += [G.delaunay(e)]
    except: pass


Volume = G.tetraMesher(Tips+[wing],algo=1)
C.convertPyTree2File(Volume,'out.cgns')