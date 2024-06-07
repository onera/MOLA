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

import Converter.Internal as I
import Converter.Mpi as Cmpi

import elsAxdt

from mola.logging import MolaException
from mola.cfd.coprocess.tools import ravelBCDataSet, forceFamilyBCasFamilySpecified

def end_simulation(workflow):
    elsAxdt.safeInterrupt()

def perform_extractions(workflow, coprocess_manager):
    if coprocess_manager.extractions_to_perform:
        coprocess_manager.restart_fields = get_elsa_output_tree(coprocess_manager)

    # for extraction in coprocess_manager.extractions_to_perform:
    #     if extraction['Type'] in ['Restart', '3D']:
    #         from .solver_elsa import get_elsa_output_tree
    #         self.fields = get_elsa_output_tree(self)

def get_elsa_output_tree(coprocess_manager):
    '''
    Extract the coupling CGNS PyTree from elsAxdt *OUTPUT_TREE* and make
    necessary adaptions, including migration of coordinates fields to
    GridCoordinates_t nodes, renaming of conventional fields names and
    adding the tree's Skeleton.

    Returns
    -------

        t : PyTree
            Coupling adapted PyTree

    '''
    t = elsAxdt.get(elsAxdt.OUTPUT_TREE)
    adaptEndOfRun(t, coprocess_manager.iteration)
    for tree in [t, coprocess_manager.skeleton]: 
        ravelBCDataSet(tree) # HACK https://elsa.onera.fr/issues/11219
    # resumeFieldsAveraging(coprocess_manager, t)
    t = I.merge([coprocess_manager.skeleton, t])
    removeEmptyBCDataSet(t)
    forceFamilyBCasFamilySpecified(t) # HACK https://elsa.onera.fr/issues/10928

    return t


def update_elsa_input(new_tree):
    elsAxdt.xdt(elsAxdt.PYTHON,(elsAxdt.RUNTIME_TREE, new_tree, 1))

def adaptEndOfRun(to, CurrentIteration):
    '''
    This function is used to make adaptations of the coupling trigger tree
    provided by elsA. The following operations are performed:

    * ``GridCoordinates`` node is created from ``FlowSolution#EndOfRun#Coords``
    * adapt name of masking field (``cellnf`` is renamed as ``cellN``)
    * rename ``FlowSolution#EndOfRun`` as ``FlowSolution#Init``

    Parameters
    ----------

         to : PyTree
            Coupling tree as obtained from function

            >>> elsAxdt.get(elsAxdt.OUTPUT_TREE)

            .. note:: tree **to** is modified
    '''
    moveCoordsFromEndOfRunToGridCoords(to)
    I._renameNode(to, 'cellnf', 'cellN')
    I._renameNode(to, 'FlowSolution#EndOfRun', 'FlowSolution#Init')
    I._rmNodesByName(to, 'FlowSolution#Init-1')
    I._renameNode(to, f'FlowSolution#EndOfRun{CurrentIteration-1:04d}', 'FlowSolution#Init-1')

def moveCoordsFromEndOfRunToGridCoords(to):
    '''
    This function is used to make adaptations of the coupling trigger tree
    provided by elsA. The following operations are performed:

    * ``GridCoordinates`` node is created from ``FlowSolution#EndOfRun#Coords``


    Parameters
    ----------

         to : PyTree
            Coupling tree as obtained from function

            >>> elsAxdt.get(elsAxdt.OUTPUT_TREE)

            .. note:: tree **to** is modified
    '''
    FScoords = I.getNodeFromName(to, 'FlowSolution#EndOfRun#Coords')
    if FScoords:
        I._renameNode(to,'FlowSolution#EndOfRun#Coords','GridCoordinates')
        for GridCoordsNode in I.getNodesFromName3(to, 'GridCoordinates'):
            GridLocationNode = I.getNodeFromType1(GridCoordsNode, 'GridLocation_t')
            if I.getValue(GridLocationNode) != 'Vertex':
                zone = I.getParentOfNode(to, GridCoordsNode)
                ERRMSG = ('Extracted coordinates of zone '
                          '%s must be located in Vertex')%I.getName(zone)
                raise MolaException(ERRMSG)
            I.rmNode(to, GridLocationNode)
            I.setType(GridCoordsNode, 'GridCoordinates_t')
    Cmpi.barrier()

def resumeFieldsAveraging(coprocess_manager, t, container_name='FlowSolution#Average'):
    '''
    use any pre-existing average fields contained in ``FlowSolution#Average``
    nodes in order to resume the fields averaging process
    '''
    Skeleton  = coprocess_manager.skeleton
    inititer = coprocess_manager.workflow.Numerics['IterationAtInitialState']
    firstiter = coprocess_manager.workflow.ReferenceValues['CoprocessOptions']['FirstIterationForFieldsAveraging']
    if firstiter is None: 
        return
    firstiter -= 1
    cit = coprocess_manager.iteration

    # adapt 3D fields:
    old = _getDictofNodesFieldsPerZone(Skeleton, container_name)
    tot = _getDictofNodesFieldsPerZone(t, container_name)
    if cit == firstiter:
        ini = _getDictofNodesFieldsPerZone(t, 'FlowSolution#Init')
    for zone_name in tot:
        for field_name in tot[zone_name]:
            if field_name in ['cellN','indicm']: continue
            avg_old = old[zone_name][field_name] # BEWARE this is a CGNS node
            avg_tot = tot[zone_name][field_name] # BEWARE this is a CGNS node
            
            if cit == firstiter:
                avg_old[1] = np.copy(avg_tot[1], order='F')
                avg_tot[1] = np.copy(ini[zone_name][field_name][1], order='F')
                continue

            if cit < firstiter: 
                avg_old[1] = None
                avg_new    = None
            
            else:
                if avg_old[1] is None or avg_tot[1] is None: continue
                if inititer < firstiter:
                    avg_new =  (avg_tot[1]*(cit-inititer+1) \
                            -avg_old[1]*(firstiter-inititer+1))/(cit-firstiter)
                
                else:
                    avg_new =  (avg_old[1]*(inititer-(firstiter+1)) \
                            +avg_tot[1]*(cit-inititer+1))/(cit-firstiter)

                avg_tot[1] = avg_new # update of OUTPUT_TREE

    if cit < firstiter: return

    # adapt BC fields:
    tot = _getDictofNodesBCFieldsPerZone(t, 'BCDataSet#Average')
    Cmpi.barrier()
    old = _getDictofNodesBCFieldsPerZoneAtSkeleton(Skeleton, 'BCDataSet#Average', tot)
    Cmpi.barrier()
    if cit == firstiter:
        ini = _getDictofNodesBCFieldsPerZone(t, 'BCDataSet')
    for zone_name in tot:
        for bcfamily_name in tot[zone_name]:
            for field_name in tot[zone_name][bcfamily_name]:
                if field_name in ['cellN','indicm']: continue
                try:
                    avg_old = old[zone_name][bcfamily_name][field_name] # BEWARE this is a CGNS node
                except KeyError:
                    avg_old = [field_name,None,[],'DataArray_t']

                avg_tot = tot[zone_name][bcfamily_name][field_name] # BEWARE this is a CGNS node
                
                if cit == firstiter:
                    avg_old[1] = np.copy(avg_tot[1], order='F')
                    avg_tot[1] = np.copy(ini[zone_name][bcfamily_name][field_name][1], order='F')
                    continue

                if cit < firstiter:
                    avg_old[1] = None
                    avg_new    = None
                
                else:
                    if avg_old[1] is None or avg_tot[1] is None: continue
                    if inititer < firstiter:
                        avg_new =  (avg_tot[1]*(cit-inititer+1) \
                                -avg_old[1]*(firstiter-inititer+1))/(cit-firstiter)
                    
                    else:
                        avg_new =  (avg_old[1]*(inititer-(firstiter+1)) \
                                +avg_tot[1]*(cit-inititer+1))/(cit-firstiter)

                avg_tot[1] = avg_new # update of OUTPUT_TREE
                avg_old[1] = avg_new # update of OUTPUT_TREE

def _getDictofNodesFieldsPerZone(t, Container):
    fields = dict()
    for base in I.getNodesFromType1(t, 'CGNSBase_t'):
        for zone in I.getNodesFromType1(base, 'Zone_t'):
            zone_name = zone[0]
            fields[zone_name] = dict()
            fs = I.getNodeFromName1(zone, Container)
            if not fs:
                del fields[zone_name]
                continue
            for f in fs[2]:
                if f[3] != 'DataArray_t': continue
                fields[zone_name][f[0]] = f
    return fields


def _getDictofNodesBCFieldsPerZoneAtSkeleton(t, Container, tot):
    fields = dict()
    for base in I.getNodesFromType1(t, 'CGNSBase_t'):
        for zone in I.getNodesFromType1(base, 'Zone_t'):
            zone_name = zone[0]
            fields[zone_name] = dict()
            for bc in I.getNodesFromType(zone,'BC_t'):
                bcfamily_name = bc[0]
                if bcfamily_name not in fields[zone_name]:
                    fields[zone_name][bcfamily_name] = dict()
                bcds = I.getNodeFromName1(bc, Container)
                if bcds:
                    bcds[3] = 'BCDataSet_t'
                    data = I.getNodeFromName1(bcds,'NeumannData')
                    if data:
                        for f in data[2]:
                            if f[3] != 'DataArray_t': continue
                            fields[zone_name][bcfamily_name][f[0]] = f
                    else:
                        try: fields_tot = tot[zone_name][bcfamily_name]
                        except KeyError: continue
                        if not fields_tot: continue
                        nd = I.createUniqueChild(bcds,'NeumannData','BCData_t')
                        for field_name, f in fields_tot.items():
                            field_node = I.createUniqueChild(nd,f[0],'DataArray_t',np.copy(f[1],order='F'))
                            fields[zone_name][bcfamily_name][f[0]] = field_node
                else:
                    try: fields_tot = tot[zone_name][bcfamily_name]
                    except KeyError: continue
                    if not fields_tot: continue
                    bcds = I.createUniqueChild(bc,Container,'BCDataSet_t')
                    nd = I.createUniqueChild(bcds,'NeumannData','BCData_t')
                    for field_name, f in fields_tot.items():
                        field_node = I.createUniqueChild(nd,f[0],'DataArray_t',np.copy(f[1],order='F'))
                        fields[zone_name][bcfamily_name][f[0]] = field_node

    return fields


def _getDictofNodesBCFieldsPerZone(t, Container):
    fields = dict()
    for base in I.getNodesFromType1(t, 'CGNSBase_t'):
        for zone in I.getNodesFromType1(base, 'Zone_t'):
            zone_name = zone[0]
            fields[zone_name] = dict()
            for bc in I.getNodesFromType(zone,'BC_t'):
                bcfamily_name = bc[0]
                if bcfamily_name not in fields[zone_name]:
                    fields[zone_name][bcfamily_name] = dict()
                bcds = I.getNodeFromName1(bc, Container)
                if bcds:
                    data = I.getNodeFromName1(bcds,'NeumannData')
                    if data:
                        for f in data[2]:
                            if f[3] != 'DataArray_t': continue
                            fields[zone_name][bcfamily_name][f[0]] = f
    return fields

def removeEmptyBCDataSet(t):
    for z in I.getZones(t):
        for zbc in I.getNodesFromType1(z,'ZoneBC_t'):
            for bc in I.getNodesFromType1(zbc,'BC_t'):
                for n in bc[2]:
                    if n[0].startswith('BCDataSet'):
                        if not n[2]:
                            I._rmNode(t, n)

