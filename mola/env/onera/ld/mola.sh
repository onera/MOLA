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

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export MOLA=${SCRIPT_DIR%/mola/env/*}  # retain the part before /mola/env/*
export MOLAext=$MOLA/ext # TODO check that !!

source /etc/bashrc
module purge &>/dev/null
unset PYTHONPATH
shopt -s expand_aliases
ulimit -s unlimited # in order to allow arbitrary use of stack (required by VPM)

export http_proxy=http://proxy.onera:80 https_proxy=http://proxy.onera:80 ftp_proxy=http://proxy.onera:80
export no_proxy=localhost,gitlab-dtis.onera,gitlab.onera.net

export FORT_BUFFERED=true
export MPI_GROUP_MAX=8192
export MPI_COMM_MAX=8192
export PYTHONUNBUFFERED=true # cf ticket 9685


module load python/3.10.8-gnu831

module load texlive/2021 # for LaTeX rendering in matplotlib with STIX font
module load vscode/1.74.3
module load pointwise/2022.1.2
# # module load paraview/5.11.0 # provokes python and libraries incompatibilities
module load occt/7.6.1-gnu831

export OPENMPIOVERSUBSCRIBE='--oversubscribe'

unset I_MPI_PMI_LIBRARY
export OMPI_MCA_mca_base_component_show_load_errors=0

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
export PATH=$MOLA/bin:$PATH

export PYTHONEXE=python3
alias python=python3