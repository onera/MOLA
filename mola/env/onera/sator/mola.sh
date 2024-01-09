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

if [ -n "$SLURM_NTASKS" ] ; then
    if [ $SLURM_NTASKS == 1 ] ; then
        if [ -n "$SLURM_CPUS_PER_TASK" ] ; then
            export NPROCMPI=$SLURM_CPUS_PER_TASK
        elif [ -n "$SLURM_CPUS_ON_NODE" ] ; then
            export NPROCMPI=$SLURM_CPUS_ON_NODE
        else
            export NPROCMPI=$(nproc)
        fi
    else
        export NPROCMPI=$SLURM_NTASKS
    fi
else
    export NPROCMPI=$(nproc)
fi


unset I_MPI_PMI_LIBRARY
export MOLA=$MOLASATOR

export PYTHONPATH=$MOLASATORext/sator/lib/python3.7/site-packages/:$PYTHONPATH
export PATH=$MOLASATORext/sator/bin:$PATH
export LD_LIBRARY_PATH=$MOLASATORext/sator/lib/python3.7/site-packages/PyQt5/Qt5/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/tmp_user/sator/lbernard/lib:$LD_LIBRARY_PATH



export PYTHONPATH=$MOLA:$PYTHONPATH
export PATH=$MOLA/bin:$PATH

export PYTHONEXE=python3
alias python=python3