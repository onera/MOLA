#! /bin/sh
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
export MOLA=${SCRIPT_DIR%/mola/env/*}  # retain the part before /mola/env/*
export MOLAext=/stck/mola/v2.0/ext
export MOLA_NETWORK=${SCRIPT_DIR##*/} # get only the last part of SCRIPT_DIR, so here 'onera'

export http_proxy=http://proxy.onera:80 https_proxy=http://proxy.onera:80 ftp_proxy=http://proxy.onera:80
export no_proxy=localhost,gitlab-dtis.onera,gitlab.onera.net

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

export FORT_BUFFERED=true
export MPI_GROUP_MAX=8192
export MPI_COMM_MAX=8192
export PYTHONUNBUFFERED=true # cf ticket 9685, but simulation is slower cf ticket 10472

export TREELABVERSION=v0.4.4
# NOTE installation hint:
# python3 -m pip install --force-reinstall --no-cache-dir --ignore-installed --prefix=/stck/mola/treelab/$TREELABVERSION/ld_elsA mola-treelab==$TREELABVERSION

export ELSAVERSION=v5.4.01 
export ELSA_VERBOSE_LEVEL=0 # cf elsA ticket 9689
export ELSA_MPI_LOG_FILES=OFF
export ELSA_MPI_APPEND=FALSE # cf elsA ticket 7849
export ELSA_NOLOG=ON
# ELSA_MEMORY_VERBOSE=TRUE https://elsa.onera.fr/issues/10621#note-19

export SONICSVERSION=0.6.19

export CASSIOPEE_VERSION=v4.0  # For Fast and SoNICS. For elsA, keep the version loaded with the solver
export MAIAVERSION=1.6  # For Fast. For elsA and SoNICS, keep the version loaded with the solver
export TURBOVERSION=v1.3.1
export ERSTAZVERSION=v1.6.3
