#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from treelab import cgns
from mola.cfd import apply_to_solver
from mola.logging import mola_logger
from ...tools import to_distributed

def apply(workflow):
    apply_to_solver(workflow)

def convert_elements_to_ngon(t):
    from mpi4py import MPI
    import maia

    mola_logger.user_warning('Some cells are not NGon: converting to NGon')
    
    t = to_distributed(t)

    try:
        maia.algo.dist.convert_elements_to_ngon(t, MPI.COMM_WORLD)
    except BaseException as e:
        mola_logger.warning(f'Could not convert to NGon using maia, received error:\n{e}\nattempting using Cassiopee...')
        import Converter.Internal as I
        import Converter.PyTree as C

        def get_unstructured_zones(t):
            zones = []
            for z in I.getZones(t):
                if I.getZoneType(z) == 2:
                    zones += [ z ]
            return zones

        try:
            uns = get_unstructured_zones(t)[0]
            I.printTree(uns, 'tree1.txt')
            C._convertArray2NGon(uns, recoverBC=1)
        except BaseException as e2:
            C.convertPyTree2File(t, 'debug.cgns')
            mola_logger.error('could not convert to NGon using Cassiopee, check debug.cgns')
            uns = get_unstructured_zones(t)[0]
            I.printTree(uns, 'tree2.txt')
            C._convertArray2NGon(uns, recoverBC=1)
        C._signNGonFaces(uns)
        I._adaptNGon32NGon4(uns)

    return cgns.castNode(t)

def merge_all_unstructured_zones_from_families(t):
    from mpi4py import MPI
    import maia 

    def correct_type_i8_to_i4(t):
        nodes = []
        for Elements in t.group(Type='Elements'):
            nodes += Elements.children()
        nodes += t.group(Type='IndexArray')  # PointList in BC
        for node in nodes:
            value = node.value()
            if isinstance(value, np.ndarray) and isinstance(value.ravel()[0], np.int64):
                node.setValue(value.astype(np.int32))

    mola_logger.info('Merge unstructured zones by family', rank=0)

    t = to_distributed(t)

    zonePathsByFamily = dict()
    for base in t.bases():
        for zone in base.zones():
            if zone.isStructured(): 
                continue 
            zone_path = f'{base.name()}/{zone.name()}'
            try:
                FamilyName = zone.get(Type='FamilyName', Depth=1).value()
            except:
                FamilyName = 'unspecified'
            if FamilyName not in zonePathsByFamily:
                zonePathsByFamily[FamilyName] = [zone_path]
            else:
                zonePathsByFamily[FamilyName].append(zone_path)

    for family, zone_paths in zonePathsByFamily.items():
        if len(zone_paths) < 2: continue
        base_name = zone_paths[0].split('/')[0]
        mola_logger.info(f' --> merging zones of family {family}', rank=0)

        maia.pytree.rm_nodes_from_name(t, 'NFaceElements')
        maia.algo.dist.merge_zones(t, zone_paths, MPI.COMM_WORLD, output_path=f'{base_name}/{family}_Zone', subset_merge='family')

    t = cgns.castNode(t)
    # HACK see https://elsa-e.onera.fr/issues/11725
    correct_type_i8_to_i4(t)
    
    # for elsA, change GC value from <base>/<zone> to <zone>
    for gc in t.group(Type='GridConnectivity'):
        value = gc.value()
        gc.setValue(value.split('/')[-1])

    mola_logger.info(f' --> after merge, zones names are: {", ".join([z.name() for z in t.zones()])}', rank=0)
    
    return t
