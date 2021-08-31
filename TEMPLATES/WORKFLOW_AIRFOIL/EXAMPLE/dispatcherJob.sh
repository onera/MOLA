#!/bin/bash
#SBATCH -J mesher
#SBATCH --comment 31447034F
#SBATCH -o job.output.%j
#SBATCH -e job.error.%j
#SBATCH -t 0-0:30
#SBATCH -n 1

source /etc/bashrc

export ELSA_MPI_LOG_FILES=OFF
export ELSA_MPI_APPEND=FALSE # See ticket 7849
export FORT_BUFFERED=true
export MPI_GROUP_MAX=8192
export MPI_COMM_MAX=8192
source /tmp_user/sator/elsa/Public/v5.0.02/Dist/bin/sator/source.me

# NUMPY SCIPY
export PATH=$PATH:/tmp_user/sator/lbernard/.local/bin/
export PYTHONPATH=/tmp_user/sator/lbernard/.local/lib/python2.7/site-packages/:$PYTHONPATH

# MOLA
export MOLA=/tmp_user/sator/lbernard/MOLA/Dev
export PYTHONPATH=$PYTHONPATH:$MOLA

cd /tmp_user/sator/lbernard/POLARS/NEWTEST/DISPATCHER
python MeshAndDispatch.py 1>MeshAndDispatch-out.log 2>MeshAndDispatch-err.log
