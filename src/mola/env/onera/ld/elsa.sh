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

source /stck/elsa/Public/$ELSAVERSION/Dist/bin/local-os8_mpi/.env_elsA &>/dev/null
module load texlive/2021 # for LaTeX rendering in matplotlib with STIX font
module load vscode
module load pointwise
module load paraview
module load occt

export OPENMPIOVERSUBSCRIBE='--oversubscribe'

unset I_MPI_PMI_LIBRARY
export OMPI_MCA_mca_base_component_show_load_errors=0

# Treelab
export DIST="ld"
export TREELABPATH=/stck/mola/treelab/$TREELABVERSION/${DIST}_elsA
export PATH="$TREELABPATH/bin${PATH:+:${PATH}}"
export PYTHONPATH=$TREELABPATH/lib/python3.8/site-packages:$PYTHONPATH
# export PYTHONPATH=/stck/lbernard/treelab/dev/src:$PYTHONPATH # ONLY DURING DEV

# # maia
# module use --append /home/sonics/LD8/modules/
# module load maia/$MAIAVERSION-dsi-ompi405

# turbo 
export PYTHONPATH=/stck/jmarty/TOOLS/turbo/install/$TURBOVERSION/env_elsA_v5.3.01/local-os8_mpi/lib/python3.8/site-packages/:$PYTHONPATH
export TURBO_COMPILER='gcc'

# ErstaZ
export EZPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/bin/eos
export PYTHONPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/python_module:$PYTHONPATH

# external python dependencies
export PYTHONPATH=$MOLAext/ld8/lib/python3.8/site-packages/:$PYTHONPATH
export PATH=$MOLAext/ld8/bin:$PATH
export LD_LIBRARY_PATH=$MOLAext/ld8/lib/python3.8/site-packages/PyQt5/Qt5/lib/:$LD_LIBRARY_PATH

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

export MOLA_SOLVER=elsa