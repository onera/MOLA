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

from mola.logging import mola_logger
from treelab import cgns
from mola.cfd.coprocess.solver_fast import ALLOWED_EXTRACTIONS

def apply_to_solver(workflow):

    workflow._interface.add_to_Extractions_Restart(
        Container='FlowSolution#Centers'
        )

    for Extraction in workflow.Extractions: 
        if Extraction['Type'] == 'Residuals':
            Extraction['ExtractionPeriod'] = Extraction['SavePeriod']

        if Extraction['Type'] in ['BC','IsoSurface','3D']:
            if 'Fields' in Extraction:
                if isinstance(Extraction['Fields'],str):
                    Extraction['Fields'] = [ Extraction['Fields'] ]

                for field in Extraction['Fields'][:]:
                    if field not in ALLOWED_EXTRACTIONS:
                        mola_logger.user_warning(f'field "{field}" not supported in fast, skipping')
                        Extraction['Fields'].remove(field)

def add_convergence_history(t, niter):
    import Converter.Internal as I

    I._rmNodesByType(t, 'ConvergenceHistory_t')

    
    import FastS.PyTree as FastS
    FastS._createConvergenceHistory(t, niter+1) # https://github.com/onera/Fast/issues/13 
    


def _createConvergenceHistory(t, inititer, niter,
            residuals_names = ['RSD_L2','RSD_oo','RSD_L2_diff','RSD_oo_diff']):
    """HACK create a node in tree to store convergence history."""
    for base in t.bases():
        cgns.Node(Name='GlobalConvergenceHistory',
                  Type='ConvergenceHistory_t',
                  Value=0, Parent=base)

        model='unknown'
        governing_eqns = t.get('GoverningEquations')
        if governing_eqns: model = governing_eqns.value()

        for zone in base.zones():
            different_governing_eqns = zone.get('GoverningEquations')
            if different_governing_eqns: model = different_governing_eqns.value()

            nb_of_equations = 6 if model in ('nsspalart','NSTurbulent') else 5

            conv_hist = cgns.Node(Name='ZoneConvergenceHistory',
                                  Type='ConvergenceHistory_t',
                                  Value=0, # FIXME this may produce segfault
                                  Parent=zone)

            cgns.Node(Name='Iteration',
                      Type='DataArray_t',
                      Value=np.zeros((niter),dtype=np.int32, order='F'),
                      Parent=conv_hist)
            
            for residual_name in residuals_names:
                cgns.Node(Name=residual_name,
                          Type='DataArray_t',
                          Value=np.zeros((niter*nb_of_equations),
                                          dtype=np.float64, order='F'),
                          Parent=conv_hist)
