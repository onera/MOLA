from mola.workflow import read_workflow
import maia.pytree as PT


# w = read_workflow('main.cgns')
# w.postprocess(RowType='turbine')

# w.read_tree()
from mola.cfd.postprocess import iso_surface
from treelab import cgns

rank = 0
tree = cgns.load(f'output_tree_{rank}.cgns')

for GL in tree.group(Type='GridLocation', Value='?FaceCenter'):
    GL.setValue('FaceCenter')
# w.tree.findAndRemoveNode(Depth=1, Name=w._interface._workflow_parameters_container_)
# tree.findAndRemoveNodes(Name='.Solver#Param')
for z in tree.zones():
    if z.get(Name='.Solver#Param').get(Name='proc').value() != rank:
        z.remove()

PT.print_tree(tree)

iso = iso_surface(
    tree, 
    IsoSurfaceField='CoordinateX', 
    IsoSurfaceValue=0.1, 
    IsoSurfaceContainer='GridCoordinates', 
    Name='test'
    )
iso.save('iso.cgns')

PT.print_tree(iso)
