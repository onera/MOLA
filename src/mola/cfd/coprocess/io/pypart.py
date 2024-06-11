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
import glob

import Converter.PyTree as C
import Converter.Internal as I

from . import rank, comm
from .utils import removeNonLocalZones, ravelBCDataSet, forceFamilyBCasFamilySpecified

def save_with_pypart(t, filename, PyPartBase):
    '''
    Function to save a PyTree **t** with PyPart. The PyTree must have been
    splitted with PyPart in ``compute.py``. An important point is the presence
    in every zone of **t** of the special node ``:CGNS#Ppart``.

    Use this function to save ``'fields.cgns'``.

    .. note:: For more details on PyPart, see the dedicated pages on elsA
        support:
        `PyPart alone <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/PreprocessTutorials/etc_pypart_alone.html>`_
        and
        `PyPart with elsA <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/PreprocessTutorials/etc_pypart_elsa.html>`_

    Parameters
    ----------

        t : PyTree
            tree to save

        filename : str
            Name of the file

        tagWithIteration : bool
            if :py:obj:`True`, adds a suffix ``_AfterIter<iteration>``
            to the saved filename (creates a copy)
    '''
    tpt = I.copyRef(t)
    removeNonLocalZones(tpt)
    I._rmNodesByName(tpt, '.Solver#Param')
    I._rmNodesByType(tpt, 'IntegralData_t')
    comm.barrier()
    PyPartBase.mergeAndSave(tpt, 'PyPart_fields')
    comm.barrier()
    if rank == 0:
        t_merged = C.convertFile2PyTree('PyPart_fields_all.hdf')
        # addLostFieldsExtractors(t_merged)
        migrateSolverOutputOfFlowSolutions(t, t_merged)
        I._rmNodesByName(t_merged, 'FlowSolution#EndOfRun*')
        ravelBCDataSet(t_merged)
        forceFamilyBCasFamilySpecified(t_merged) 
        C.convertPyTree2File(t_merged, filename)
        for fn in glob.glob('PyPart_fields_*.hdf'):
            try:
                os.remove(fn)
            except:
                pass
    comm.barrier()

def migrateSolverOutputOfFlowSolutions(t_dnr, t_rcv):
    # Required because of https://elsa.onera.fr/issues/11137
    zones = I.getZones(t_dnr)
    for zm in I.getZones(t_rcv):
        for z in zones:
            if z[0].startswith(zm[0]):
                all_fs  = I.getNodesFromType(z, 'FlowSolution_t')
                all_fsm = I.getNodesFromType(zm, 'FlowSolution_t')
                for fsm in all_fsm:
                    for fs in all_fs:
                        if fs[0] == fsm[0]:
                            SolverOutput = I.getNodeFromName(fs, '.Solver#Output')
                            if SolverOutput:
                                fsm[2].append( SolverOutput )
                                continue
