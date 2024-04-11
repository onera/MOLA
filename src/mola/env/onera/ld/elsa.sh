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


export FORT_BUFFERED=true
export MPI_GROUP_MAX=8192
export MPI_COMM_MAX=8192
export PYTHONUNBUFFERED=true # cf ticket 9685

export ELSAVERSION=v5.2.03
export ELSA_VERBOSE_LEVEL=0 # cf elsA ticket 9689
export ELSA_MPI_LOG_FILES=OFF
export ELSA_MPI_APPEND=FALSE # cf elsA ticket 7849
export ELSA_NOLOG=ON

export TREELABVERSION=v0.1.1
export VPMVERSION=Dev
export PUMAVERSION=v2.0.3
export TURBOVERSION=v1.3
export ERSTAZVERSION=v1.6.3
export MAIAVERSION=1.2

# architecture
if lscpu | grep -q 'avx512' ; then
    export ARCH='avx512'
elif lscpu | grep -q 'avx2' ; then
    export ARCH='avx2'
elif lscpu | grep -q 'avx' ; then
    export ARCH='avx'
elif lscpu | grep -q 'sse4_2' ; then
    export ARCH='sse4_2'
elif lscpu | grep -q 'sse4_1' ; then
    export ARCH='sse4_1'
elif lscpu | grep -q 'ssse3' ; then
    export ARCH='ssse3'
elif lscpu | grep -q 'sse3' ; then
    export ARCH='sse3'
else
    export ARCH='sse2'
fi


source /stck/elsa/Public/$ELSAVERSION/Dist/bin/local-os8_mpi/.env_elsA &>/dev/null
module load texlive/2021 # for LaTeX rendering in matplotlib with STIX font
module load vscode/1.85.2
module load pointwise/2022.1.2
# # module load paraview/5.11.0 # provokes python and libraries incompatibilities
module load occt/7.6.1-gnu831

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

# PUMA
export PumaRootDir=/stck/rboisard/bin/local/x86_64z/Puma_${PUMAVERSION}_os8
export PYTHONPATH=$PumaRootDir/lib/python3.8/site-packages:$PYTHONPATH
export PYTHONPATH=$PumaRootDir/lib/python3.8/site-packages/PUMA:$PYTHONPATH
export LD_LIBRARY_PATH=$PumaRootDir/lib/python3.8:$LD_LIBRARY_PATH
export PUMA_LICENCE=$PumaRootDir/pumalicence.txt

# turbo 
export PYTHONPATH=/stck/jmarty/TOOLS/turbo/install/$TURBOVERSION/env_elsA_$ELSAVERSION/local-os8_mpi/lib/python3.8/site-packages/:$PYTHONPATH

# ErstaZ
export EZPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/bin/eos
export PYTHONPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/python_module:$PYTHONPATH

# maia
module use --append /home/sonics/LD8/modules/
module load maia/$MAIAVERSION-dsi-ompi405

# VPM
export VPMPATH=/stck/lbernard/VPM/$VPMVERSION/ld/$ARCH
export PATH=$VPMPATH:$VPMPATH/lib:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/stck/benoit/lib
export LD_LIBRARY_PATH=$VPMPATH:$VPMPATH/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$VPMPATH:$PYTHONPATH
export PYTHONPATH=$VPMPATH/lib/python3.8/site-packages:$PYTHONPATH
# replaces module load intel/21.2.0 since this module
# brakes MPI https://elsa.onera.fr/issues/10933#note-16
export LD_LIBRARY_PATH=/opt/tools/intel/oneapi/compiler/2021.2.0/linux/compiler/lib/intel64_lin/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/tools/intel/oneapi/mpi/2021.6.0/lib/release:$LD_LIBRARY_PATH


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