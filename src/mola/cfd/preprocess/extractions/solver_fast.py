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

def apply_to_solver(workflow):

    workflow._interface.add_to_Extractions_Restart(
        Container='FlowSolution#Centers'
        )

    for Extraction in workflow.Extractions: 
        if Extraction['Type'] == 'Residuals':
            add_convergence_history(workflow)
            Extraction['ExtractionPeriod'] = Extraction['SavePeriod']



def add_convergence_history(workflow):

    # # FIXME https://github.com/onera/Fast/issues/13 
    # import FastS.PyTree as FastS
    # FastS.createConvergenceHistory(workflow.tree, ExtactionPeriod)

    # HACK
    _createConvergenceHistory(workflow.tree,
        workflow.Numerics['IterationAtInitialState'],
        workflow.Numerics['NumberOfIterations']+1)
    
    # # WIP
    # _createConvergenceHistoryCass(workflow.tree, workflow.Numerics['NumberOfIterations'])

    cgns.castNode(workflow.tree)


def _createConvergenceHistory(t, inititer, niter,
            residuals_names = ['RSD_L2','RSD_oo','RSD_L2_diff','RSD_oo_diff']):
    """HACK create a node in tree to store convergence history."""
    for base in t.bases():
        cgns.Node(Name='GlobalConvergenceHistory',
                  Type='ConvergenceHistory_t',
                  Value=inititer, Parent=base)

        model='unknown'
        governing_eqns = t.get('GoverningEquations')
        if governing_eqns: model = governing_eqns.value()

        for zone in base.zones():
            different_governing_eqns = zone.get('GoverningEquations')
            if different_governing_eqns: model = different_governing_eqns.value()

            nb_of_equations = 6 if model in ('nsspalart','NSTurbulent') else 5

            conv_hist = cgns.Node(Name='ZoneConvergenceHistory',
                                  Type='ConvergenceHistory_t',
                                  Value=inititer-1, # FIXME this may produce segfault
                                  Parent=zone)

            cgns.Node(Name='IterationNumber',
                      Type='DataArray_t',
                      Value=np.zeros((niter),dtype=np.int32, order='F'),
                      Parent=conv_hist)
            
            for residual_name in residuals_names:
                cgns.Node(Name=residual_name,
                          Type='DataArray_t',
                          Value=np.zeros((niter*nb_of_equations),
                                          dtype=np.float64, order='F'),
                          Parent=conv_hist)


def _createConvergenceHistoryCass(t, nrec):
    import Converter.Internal as I
    import numpy
    varsR   = ['RSD_L2','RSD_oo','RSD_L2_diff','RSD_oo_diff']
    bases   = I.getNodesFromType1(t, 'CGNSBase_t')
    curIt   = 0
    for b in bases:
       I.createUniqueChild(b, 'GlobalConvergenceHistory',
                              'ConvergenceHistory_t', value=curIt)

       model='Nada'
       a = I.getNodeFromName2(t, 'GoverningEquations')
       if a is not None: model = I.getValue(a)

       for z in I.getZones(b):

          a = I.getNodeFromName2(z, 'GoverningEquations')
          if a is not None: model = I.getValue(a)
          neq = 5
          if model == 'nsspalart' or model =='NSTurbulent': neq = 6
         
          c = I.createUniqueChild(z, 'ZoneConvergenceHistory',
                                     'ConvergenceHistory_t', value=curIt)
          tmp = numpy.zeros((nrec), numpy.int32)
          I.createChild(c, 'IterationNumber', 'DataArray_t', tmp)
          for var in varsR:
            tmp = numpy.zeros((nrec*neq), numpy.float64)
            I.createChild(c, var ,'DataArray_t', tmp)