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

source /tmp_user/juno/elsa/Public/$ELSAVERSION/Dist/bin/juno_mpi/.env_elsA &>/dev/null

unset I_MPI_PMI_LIBRARY
unset I_MPI_TCP_NETMASK 
unset I_MPI_FABRICS_LIST

# # for avoiding error bootstrap
# # https://community.intel.com/t5/Intel-MPI-Library/Unable-to-run-bstrap-proxy-error-with-intel-oneapi-mpi-2021-8/td-p/1466543
# # https://slurm.schedmd.com/mpi_guide.html
# export I_MPI_HYDRA_BOOTSTRAP=ssh
export I_MPI_FABRICS=shm:tcp
export FI_PROVIDER=tcp

# TODO in future, but requires using "srun" instead of "mpirun":
# export I_MPI_HYDRA_BOOTSTRAP=pmix
# unset I_MPI_PMI_LIBRARY
# export I_MPI_FABRICS=shm:ofi
# export FI_PROVIDER=tcp



# # maia
# module use --append /tmp_user/juno/sonics/usr/modules/
# module load maia/$MAIAVERSION-dsi-cfd6

# Treelab
export TREELABPATH=/tmp_user/juno/mola/treelab/$TREELABVERSION/juno_elsA
# export TREELABPATH=/stck/lbernard/treelab/dev # ONLY DURING DEV (replaces stable version)
export PATH="$TREELABPATH/bin${PATH:+:${PATH}}"
export PYTHONPATH=$TREELABPATH/lib/python3.8/site-packages:$PYTHONPATH

# turbo 
export PYTHONPATH=/tmp_user/juno/jmarty/TOOLS/turbo/install/$TURBOVERSION/env_elsA_v5.3.01/juno_mpi/lib/python3.8/site-packages/:$PYTHONPATH

export PYTHONPATH=$MOLA:$PYTHONPATH
export PATH=$MOLA/mola/bin:$PATH

export PYTHONEXE=python3
alias python=python3

export MOLA_SOLVER=elsa