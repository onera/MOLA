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

SCRIPT_DIR=$( \cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../network.sh

export MACHINE=spiro_el8
export CASSIOPEE=/stck/cassiope/git/Cassiopee/ 
source $CASSIOPEE/Dist/sh_Cassiopee_local &> /dev/null

# to avoid message:
# MPI startup(): Warning: I_MPI_PMI_LIBRARY will be ignored since the hydra process manager was found
# source : https://www.osc.edu/supercomputing/batch-processing-at-osc/slurm_migration/slurm_migration_issues
unset I_MPI_PMI_LIBRARY 

unset I_MPI_TCP_NETMASK
unset I_MPI_FABRICS_LIST

# Treelab
# NOTE installation hint:
# python3 -m pip install --force-reinstall --no-cache-dir --ignore-installed --prefix=/stck/mola/treelab/v0.1.0/ld_elsA mola-treelab
export TREELABPATH=/stck/mola/treelab/$TREELABVERSION/spiro_elsA
export PATH="$TREELABPATH/bin${PATH:+:${PATH}}"
export PYTHONPATH=$TREELABPATH/lib/python3.7/site-packages:$PYTHONPATH
export PYTHONPATH=/stck/lbernard/treelab/dev/src:$PYTHONPATH # ONLY DURING DEV

# maia
module use --append /scratchm/sonics/usr/modules/
module load maia/1.4-dsi-cfd5

# turbo
export PYTHONPATH=/stck/jmarty/TOOLS/turbo/install/$TURBOVERSION/env_elsA_$ELSAVERSION/spiro3_mpi/lib/python3.7/site-packages/:$PYTHONPATH

# ErstaZ
export EZPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/bin/spiro
export PYTHONPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/python_module:$PYTHONPATH

# external python packages
# export PYTHONPATH=$MOLAext/spiro_el8/lib/python3.8/site-packages/:$PYTHONPATH
# export PATH=$MOLAext/spiro_el8/bin:$PATH
# export LD_LIBRARY_PATH=$MOLAext/spiro_el8/lib/python3.8/site-packages/PyQt5/Qt5/lib/:$LD_LIBRARY_PATH


export PYTHONPATH=$MOLA:$PYTHONPATH
export PATH=$MOLA/mola/bin:$PATH

export PYTHONEXE=python3
alias python=python3

export MOLA_SOLVER=fast