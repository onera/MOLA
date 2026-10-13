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

import mola.mesh.ExtractSurfacesProcessor as ESP

import Converter.PyTree as C
import Converter.Internal as I
import Converter.Mpi as Cmpi
from .tools import * # BAD PRACTICE !!

from mola.pytree.user.checker import is_distributed_for_use_in_maia, assert_zones_have_zone_type_node
from mola.cfd.preprocess.mesh.families import _ungroupBCsByBCType

def extract_bc(tree, Family=None, Name=None, Type=None):
    '''
    This is a multi-container wrapper of Cassiopee C.extractBC* functions, 
    as requested in https://elsa.onera.fr/issues/10641. 

    Parameters
    ----------

        t : PyTree
            input tree where surfaces will be extracted

        Family : str
            (optional) family name of the BC to be extracted

        Name : str
            (optional) name of the BC to be extracted

        Type : str
            (optional) type of the BC to be extracted

    Returns
    -------

        surfaces : :py:class:`list` of Zone_t
            list of surfaces (zones) with multi-containers (including *BCData_t* 
            transformed into *FlowSolution_t* nodes)    
    '''
    # CAUTION https://elsa.onera.fr/issues/12070
    # CAUTION https://elsa.onera.fr/issues/12076
    # HACK    https://elsa.onera.fr/issues/10641

    t = I.copyTree(tree) # HACK https://gitlab.onera.net/numerics/solver/sonics/-/issues/180#note_51164
    
    if Cmpi.size > 1:
        if is_distributed_for_use_in_maia(t):
            raise TypeError("Cannot use Cassiopee for extracting a BC using a maia-distributed tree in a MPI parallel context https://elsa.onera.fr/issues/12070")
        t = Cmpi.convert2PartialTree(t, rank=Cmpi.rank)

    t = mergeContainers(t, FlowSolutionVertexName=I.__FlowSolutionNodes__,
                           FlowSolutionCellCenterName=I.__FlowSolutionCenters__)

    # CAUTION https://elsa.onera.fr/issues/12076
    t = I.adaptNGon42NGon3(t)
    I._adaptPE2NFace(t)

    args = [Family, Name, Type]
    if args.count(None) != len(args)-1:
        raise AttributeError('must provide only one of: Name, Type or Family')

    if Family is not None:
        if Family.startswith('FamilySpecified:'):
            Type = Family
        else:
            Type = 'FamilySpecified:'+Family
        Name = None

    elif Name is not None and Name.startswith('FamilySpecified:'):
        Type = Name
        Name = None


    if Name:
        extractBCarg = Name
        extractBCfun = C.extractBCOfName
    else:
        extractBCarg = Type
        extractBCfun = C.extractBCOfType
    


    bases_children_except_zones = []
    for base in I.getBases(t):
        for n in base[2]:
            if n[3] != 'Zone_t': 
                bases_children_except_zones.append( n )

    bcs = []
    for zone in I.getZones(t):

        extraction_output = extractBCfun(zone, extractBCarg)
        extracted_bcs = I.getZones( extraction_output )
        if not extracted_bcs: continue
        I._adaptZoneNamesForSlash(extracted_bcs)
        for surf in extracted_bcs:
            mergeBCtagContainerWithFlowSolutionTagContainer(zone, surf)
            bc_multi_container = recoverContainers(surf)
            bcs += [ bc_multi_container ]
        reshapeFieldsForStructuredGrid(bcs)

    t_merged = C.newPyTree(['Base',bcs])
    base = I.getBases(t_merged)[0]
    base[2].extend( bases_children_except_zones )
    zones = I.getZones(t_merged)

    if Cmpi.size == 0: 
        # we allow no zones in MPI?
        if not zones:
            raise TypeError("extract_bc produced no zones")

    return zones



def getWalls(t, SuffixTag=None):
    '''
    Get closed watertight surfaces from walls (defined using ``BCWall*``)

    Parameters
    ----------

        t : PyTree
            assembled tree

        SuffixTag : str
            if provided, include a tag on newly created zone names

    Returns
    -------

        walls - list
            
    '''


    if SuffixTag:
        walls = extract_bc(t, Family=SuffixTag)
        for w in I.getZones(walls): w[0] = SuffixTag
    else:
        walls = extract_bc(t, Type='BCWall')
    
    return walls

