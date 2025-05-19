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

export MAIAVERSION=1.6.0 #dev #1.4

export MACHINE=ld
export CASSIOPEE_VERSION=main #main #v4.0a
export CASSIOPEE=/stck/cassiope/git/releases/Cassiopee/$CASSIOPEE_VERSION
source $CASSIOPEE/Dist/sh_Cassiopee_local &> /dev/null

module load texlive/2021 # for LaTeX rendering in matplotlib with STIX font
module load vscode/1.99.3

export OPENMPIOVERSUBSCRIBE='--oversubscribe'

unset I_MPI_PMI_LIBRARY
export OMPI_MCA_mca_base_component_show_load_errors=0

# Treelab
# NOTE installation hint:
# python3 -m pip install --force-reinstall --no-cache-dir --ignore-installed --prefix=/stck/mola/treelab/v0.1.0/ld_elsA mola-treelab
export DIST="ld"
MAC0=$(echo $KC | grep 'visung'); if [ "$MAC0" != "" ]; then export DIST="visung"; fi
export TREELABPATH=/stck/mola/treelab/$TREELABVERSION/${DIST}_elsA
export PATH="$TREELABPATH/bin${PATH:+:${PATH}}"
export PYTHONPATH=$TREELABPATH/lib/python3.8/site-packages:$PYTHONPATH
export PYTHONPATH=/stck/lbernard/treelab/dev/src:$PYTHONPATH # ONLY DURING DEV

# maia
module use --append /home/sonics/LD8/modules/
module load maia/$MAIAVERSION-dsi-ompi405 &> /dev/null

# # trick to read pdf files due to conflict https://elsa.onera.fr/issues/11052
# pdf()
# {
#     export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
#     export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
#     okular "$1" &
#     export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH
# }


export PYTHONPATH=$MOLA:$PYTHONPATH
export PATH=$MOLA/mola/bin:$PATH

export PYTHONEXE=python3
alias python=python3

export MOLA_SOLVER=fast