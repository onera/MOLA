import maia 
from mpi4py import MPI

filename = 'mesh.cgns'
base_and_zone_names = 'Base/Propeller_Zone'

t = maia.io.file_to_dist_tree(filename, MPI.COMM_WORLD)
maia.io.fix_tree.fix_point_ranges(t)
maia.algo.dist.convert_s_to_ngon(t, MPI.COMM_WORLD)
maia.algo.pe_to_nface(t, MPI.COMM_WORLD)
maia.algo.dist.convert_elements_to_ngon(t, MPI.COMM_WORLD)
maia.algo.dist.merge_zones(t, ["*/*"], MPI.COMM_WORLD,
    concatenate_jns=False, # HACK https://gitlab.onera.net/numerics/mesh/maia/-/issues/230
    output_path=base_and_zone_names, subset_merge='family')
maia.io.dist_tree_to_file(t, filename.replace('.cgns','_ngon.cgns'),MPI.COMM_WORLD)