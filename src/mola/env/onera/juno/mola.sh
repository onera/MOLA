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

source /etc/bashrc
module purge &>/dev/null
unset PYTHONPATH
shopt -s expand_aliases
ulimit -s unlimited # in order to allow arbitrary use of stack (required by VPM)

module load python/3.10.8-gnu831

# to avoid message:
# MPI startup(): Warning: I_MPI_PMI_LIBRARY will be ignored since the hydra process manager was found
# source : https://www.osc.edu/supercomputing/batch-processing-at-osc/slurm_migration/slurm_migration_issues
unset I_MPI_PMI_LIBRARY 

unset I_MPI_TCP_NETMASK
unset I_MPI_FABRICS_LIST

# for avoiding error bootstrap
# https://community.intel.com/t5/Intel-MPI-Library/Unable-to-run-bstrap-proxy-error-with-intel-oneapi-mpi-2021-8/td-p/1466543
# https://slurm.schedmd.com/mpi_guide.html
export I_MPI_HYDRA_BOOTSTRAP=ssh


# Treelab
# NOTE installation hint:
# python3 -m pip install --force-reinstall --no-cache-dir --ignore-installed --prefix=/stck/mola/treelab/v0.1.0/ld_elsA mola-treelab
export TREELABPATH=/stck/mola/treelab/$TREELABVERSION/spiro_elsA
export PATH="$TREELABPATH/bin${PATH:+:${PATH}}"
export PYTHONPATH=$TREELABPATH/lib/python3.7/site-packages:$PYTHONPATH
export PYTHONPATH=/tmp_user/juno/lbernard/treelab/dev/src:$PYTHONPATH # ONLY DURING DEV

# external python packages
export PYTHONPATH=$MOLAext/spiro_el8/lib/python3.8/site-packages/:$PYTHONPATH
export PATH=$MOLAext/spiro_el8/bin:$PATH
export LD_LIBRARY_PATH=$MOLAext/spiro_el8/lib/python3.8/site-packages/PyQt5/Qt5/lib/:$LD_LIBRARY_PATH


export PYTHONPATH=$MOLA:$PYTHONPATH
export PATH=$MOLA/mola/bin:$PATH

export PYTHONEXE=python3
alias python=python3