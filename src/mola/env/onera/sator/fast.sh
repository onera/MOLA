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

# TODO use stable release
export MACHINE=sator_sph
export CASSIOPEE=/tmp_user/sator/cassiope/git/releases/Cassiopee/$CASSIOPEE_VERSION
source $CASSIOPEE/Dist/sh_Cassiopee_local &> /dev/null

unset I_MPI_PMI_LIBRARY
unset I_MPI_TCP_NETMASK 
unset I_MPI_FABRICS_LIST

export OMPI_MCA_mca_base_component_show_load_errors=0

# Treelab
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp_user/sator/lbernard/treelab/lib # because required libxcb-cursor is missing in sator https://bugreports.qt.io/browse/PYSIDE-2306
export TREELABPATH=/tmp_user/sator/mola/treelab/$TREELABVERSION/sator_elsA
# export TREELABPATH=/tmp_user/sator/lbernard/treelab/dev/sator_elsA # ONLY DURING DEV (replaces stable version)
export PATH="$TREELABPATH/bin${PATH:+:${PATH}}"
export PYTHONPATH=$TREELABPATH/lib/python3.8/site-packages:$PYTHONPATH

# maia
module use --append /tmp_user/sator/sonics/usr/modules/
module load maia/$MAIAVERSION-dsi-cfd6 &> /dev/null

# trick to read pdf files due to conflict https://elsa.onera.fr/issues/11052
pdf()
{
    export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
    okular "$1" &
    export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH
}


export PYTHONPATH=$MOLA:$PYTHONPATH
export PATH=$MOLA/mola/bin:$PATH

export PYTHONEXE=python3
alias python=python3

export MOLA_SOLVER=fast