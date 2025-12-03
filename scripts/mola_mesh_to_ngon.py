#!/usr/bin/env python3
#
# convert a mesh to NGON and merge blocks

import sys
import mola.naming_conventions as names

from mola.cfd.preprocess.mesh.io.unstructured import convert_elements_to_ngon, merge_all_unstructured_zones_from_families
from treelab import cgns

import maia 
from mpi4py import MPI
from maia.io.fix_tree import fix_point_ranges


err1 = f'must specify an input file name, e.g.: "{__file__} {names.FILE_INPUT_WORKFLOW}"'
if len(sys.argv) != 2: raise AttributeError(err1)
filename = str(sys.argv[1])

t = maia.io.file_to_dist_tree(filename, MPI.COMM_WORLD)
t = cgns.castNode(t)
fix_point_ranges(t)
maia.algo.dist.convert_s_to_ngon(t, MPI.COMM_WORLD)
maia.algo.pe_to_nface(t, MPI.COMM_WORLD)
t = cgns.castNode(t)
t = convert_elements_to_ngon(t)
t = cgns.castNode(t)
t = merge_all_unstructured_zones_from_families(t)
maia.io.dist_tree_to_file(t, filename.replace('.cgns','_ngon.cgns'),MPI.COMM_WORLD)