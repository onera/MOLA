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

# Save SLURM variables before sourcing Cassiopee
SLURM_VARS_TO_SAVE=$(env | grep '^SLURM_' | cut -d= -f1)
for var in $SLURM_VARS_TO_SAVE; do
    eval "SAVED_$var=\"\$$var\""
done

if [[ -v OMP_NUM_THREADS ]]; then
    _omp_was_set=1
    _omp_original_value=$OMP_NUM_THREADS
else
    _omp_was_set=0
fi


# Treelab
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp_user/sator/lbernard/treelab/lib # because required libxcb-cursor is missing in sator https://bugreports.qt.io/browse/PYSIDE-2306
export TREELABPATH=/tmp_user/sator/mola/treelab/$TREELABVERSION/sator_elsA
# export TREELABPATH=/tmp_user/sator/lbernard/treelab/dev/sator_elsA # ONLY DURING DEV (replaces stable version)
export PATH="$TREELABPATH/bin${PATH:+:${PATH}}"
export PYTHONPATH=$TREELABPATH/lib/python3.8/site-packages:$PYTHONPATH

export MACHINE=sator_sph
export CASSIOPEE=/tmp_user/sator/cassiope/git/releases/Cassiopee/v4.0
source $CASSIOPEE/Dist/sh_Cassiopee_local &> /dev/null # CAUTION Cassiopee clears SLURM vars

if (( _omp_was_set )); then
    export OMP_NUM_THREADS="$_omp_original_value"
fi

unset _omp_was_set _omp_original_value


source /tmp_user/sator/sonics/usr/sonics/$SONICSVERSION/gcc/source.sh &>/dev/null

unset I_MPI_PMI_LIBRARY


# Restore SLURM variables
for var in $SLURM_VARS_TO_SAVE; do
    eval "export $var=\"\${SAVED_$var}\""
done


export PYTHONPATH=$MOLA:$PYTHONPATH
export PATH=$MOLA/mola/bin:$PATH

export PYTHONEXE=python3
alias python=python3

export MOLA_SOLVER=sonics