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


unset I_MPI_PMI_LIBRARY
unset I_MPI_TCP_NETMASK 
unset I_MPI_FABRICS_LIST

# Treelab
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp_user/sator/lbernard/treelab/lib # because required libxcb-cursor is missing in sator https://bugreports.qt.io/browse/PYSIDE-2306
export TREELABPATH=/tmp_user/sator/mola/treelab/$TREELABVERSION/sator_elsA
# export TREELABPATH=/tmp_user/sator/lbernard/treelab/dev/sator_elsA # ONLY DURING DEV (replaces stable version)
export PATH="$TREELABPATH/bin${PATH:+:${PATH}}"
export PYTHONPATH=$TREELABPATH/lib/python3.8/site-packages:$PYTHONPATH

export MACHINE=sator_sph
export CASSIOPEE=/tmp_user/sator/cassiope/git/releases/Cassiopee/$CASSIOPEE_VERSION
source $CASSIOPEE/Dist/sh_Cassiopee_local &> /dev/null

source /tmp_user/sator/sonics/usr/sonics/$SONICSVERSION/gcc/source.sh &>/dev/null
# export PYTHONPATH=/tmp_user/sator/tbontemp/miles:$PYTHONPATH

# # external python packages
# export MOLAext=/tmp_user/sator/mola/future_v2/ext
# export PYTHONPATH=$MOLAext/sator/lib/python3.8/site-packages/:$PYTHONPATH
# export PATH=$MOLAext/sator/bin:$PATH
# export LD_LIBRARY_PATH=$MOLAext/sator/lib/python3.8/site-packages/PyQt5/Qt5/lib/:$LD_LIBRARY_PATH

export PYTHONPATH=$MOLA:$PYTHONPATH
export PATH=$MOLA/mola/bin:$PATH

export PYTHONEXE=python3
alias python=python3

export MOLA_SOLVER=sonics