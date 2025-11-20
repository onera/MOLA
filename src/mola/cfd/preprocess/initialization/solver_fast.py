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

from mola.logging import mola_logger

def apply_to_solver(workflow):
    groupOfNodes = workflow.tree.group(Name='FlowSolution#Init')
    for node in groupOfNodes: node.setName('FlowSolution#Centers')


def adapt_workflow_for_fast(workflow):
    split_opts = workflow.SplittingAndDistribution
    _must_split_at_preprocess_with_cassiopee(split_opts)
    _must_distribute_with_cassiopee(split_opts)
    

def _must_split_at_preprocess_with_cassiopee(split_opts):
    strategy = split_opts['Strategy']
    if strategy != 'AtPreprocess':
        msg = f'fast solver requires splitting in preprocess, switching strategy from "{strategy}" to "AtPreprocess"'
        mola_logger.user_warning(msg)
        split_opts['Strategy'] = 'AtPreprocess'
        split_opts['Splitter'] = 'Cassiopee'

def _must_distribute_with_cassiopee(split_opts):
    if split_opts['Distributor'].lower() != 'cassiopee':
        distributor = split_opts['Distributor']
        msg = f'fast solver requires to pre-assign mpi ranks in preprocess, for doing this switching distributor from distributor "{distributor}" to "Cassiopee"'
        mola_logger.user_warning(msg)
        split_opts['Distributor'] = 'Cassiopee'

