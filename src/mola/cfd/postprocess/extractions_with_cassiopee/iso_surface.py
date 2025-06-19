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

import os

import Converter.PyTree as C
import Converter.Internal as I
import Converter.Mpi as Cmpi
import Transform.PyTree as T
import Post.PyTree as P

from .tools import *
from mola.cfd.postprocess.compute import compute_radius

def iso_surface(t, fieldname=None, value=None, container='FlowSolution#Init'):
    '''
    This is a multi-container wrapper of Cassiopee Post.isoSurfMC function, as
    requested in https://elsa.onera.fr/issues/11221.

    .. attention::
        all zones contained in **t** must have the same containers. If this is
        not your case, you may want to first select the zones with same containers
        before using this function (see :py:func:`MOLA.InternalShortcuts.selectZones`)

    Parameters
    ----------

        t : PyTree
            input tree where iso-surface will be performed

        fieldname : str
            name of the field used for making the iso-surface. It can be the
            coordinates names such as ``'CoordinateX'``, ``'CoordinateY'`` or 
            ``'CoordinateZ'`` (in such cases, parameter **container** is ignored)

        value : float
            value used for computing the iso-surface of field **fieldname**

        container : str
            name of the *FlowSolution_t* CGNS container where the field
            **fieldname** is contained. This parameter is ignored if **fieldname**
            is a coordinate.

    Returns
    -------

        surfaces : :py:class:`list` of Zone_t
            list of zones with fields arranged at multiple containers following
            the original data structure.

    '''
    # HACK https://elsa.onera.fr/issues/11221
    t = Cmpi.convert2PartialTree(t, rank=Cmpi.rank)

    bases_children_except_zones = []
    for base in I.getBases(t):
        for n in base[2]:
            if n[3] != 'Zone_t': 
                bases_children_except_zones.append( n )
    if not t or not I.getNodeFromType3(t,'Zone_t'): return

    if fieldname in ['Radius', 'radius', 'CoordinateR']:
        # FIXME only if axis is the X-axis
        compute_radius(t, axis='x', fieldname=fieldname, container=I.__FlowSolutionNodes__)

    tPrev = I.copyRef(t)
    t = mergeContainers(t, FlowSolutionVertexName=I.__FlowSolutionNodes__,
                           FlowSolutionCellCenterName=I.__FlowSolutionCenters__)

    isosurfs = _iso_surface_on_merge_containers(t, fieldname, value, container, tPrev)

    t_merged = C.newPyTree(['Base', isosurfs])
    base = I.getBases(t_merged)[0]
    base[2].extend( bases_children_except_zones )
    surfs = I.getZones(t_merged)
    
    return surfs


def _iso_surface_on_merge_containers(t, fieldname=None, value=None, container='FlowSolution#Init', tPrev=None):
    isosurfs = []
    for zone in I.getZones(t):

        # NOTE slicing will provoque all containers to be located at Vertex
        tags_containers = I.getNodeFromName1(zone, 'tags_containers')
        locations_node = I.getNodeFromName1(tags_containers, 'locations')

        containers_names = I.getNodeFromName1(tags_containers, 'containers_names')
        if not containers_names:
            # this means that there is no FlowSolution_t and slice is done on coords.
            # so slice it and skip

            if I.getNodeFromType1(zone,'FlowSolution_t'):
                C.convertPyTree2File(t,'debug.cgns')
                raise ValueError(
                    f'missing node in zone:{zone[0]}/tags_containers/containers_names '
                    'and existing FlowSolution_t, which should not happen.\n'
                    'Check debug.cgns')

            if fieldname not in ['CoordinateX', 'CoordinateY', 'CoordinateZ']:
                C.convertPyTree2File(t,'debug.cgns')
                raise ValueError(
                    f'missing node in zone:{zone[0]}/tags_containers/containers_names '
                    f'and slice was requested for field {fieldname}, which should not happen.\n'
                    'Check debug.cgns')
            isosurfs +=  P.isoSurfMC(zone, fieldname, value) 
            continue
                
        if fieldname not in ['CoordinateX', 'CoordinateY', 'CoordinateZ']:
            fieldnameWithTag = None
            for cn in containers_names[2]:
                container_name = I.getValue(cn)
                tag = cn[0]
                if container_name == container:
                    fieldnameWithTag = fieldname + tag
                    break
            if fieldnameWithTag is None:
                from mpi4py import MPI
                rank = MPI.COMM_WORLD.Get_rank()
                C.convertPyTree2File(tPrev,f'debug_tPrev_{rank}.cgns')
                C.convertPyTree2File(zone,f'debug_zone_{rank}.cgns')
                raise ValueError(f'could not find tag <-> container "{container}" correspondance')
        else:
            fieldnameWithTag = fieldname

        for n in containers_names[2]:
            tag = n[0]
            loc = I.getValue(I.getNodeFromName1(locations_node, tag))
            if loc == 'CellCenter':
                cont_name = I.getValue(n)
                I.setValue(n,cont_name+'V') # https://gitlab.onera.net/numerics/mola/-/issues/146#note_20639

        for n in I.getNodeFromName1(tags_containers, 'locations')[2]:
            if I.getValue(n) == 'CellCenter':
                I.setValue(n,'Vertex')

        # HACK https://elsa.onera.fr/issues/11255
        if I.getZoneType(zone) == 2: # unstructured zone
            if I.getNodeFromName1(zone,I.__FlowSolutionCenters__):
                fieldnames = C.getVarNames(zone, excludeXYZ=True, loc='centers')[0]
                for f in fieldnames:
                    C._center2Node__(zone,f,0)
                I._rmNodesByName1(zone,I.__FlowSolutionCenters__)

            # HACK https://gitlab.onera.net/numerics/mola/-/issues/111
            # HACK https://elsa.onera.fr/issues/10997#note-6
            zone = T.breakElements(zone)

        surfs = P.isoSurfMC(zone, fieldnameWithTag, value)
        for surf in I.getZones(surfs):
            surf[2] += [ tags_containers ]
            isosurfs += [ recoverContainers(surf) ]

    return isosurfs
