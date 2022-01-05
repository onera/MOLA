#!/bin/bash
#SBATCH -J <JobName>
#SBATCH --comment <AERnumber>
#SBATCH -o output.%j
#SBATCH -e error.%j
#SBATCH -t 0-15:00
#SBATCH -n <NProcs>

module purge all

export http_proxy=proxy:80 https_proxy=proxy:80 ftp_proxy=proxy:80

# ELSA+CASSIOPEE
export ELSA_MPI_LOG_FILES=OFF
export ELSA_MPI_APPEND=FALSE # See ticket 7849
export FORT_BUFFERED=true
export MPI_GROUP_MAX=8192
export MPI_COMM_MAX=8192

export ELSAVERSION=v5.0.03
export ELSAPROD=spiro_mpi
export ELSAPATHPUBLIC=/stck/elsa/Public/$ELSAVERSION/Dist/bin/$ELSAPROD
source $ELSAPATHPUBLIC/.env_elsA

# NUMPY SCIPY
export PATH=$PATH:/tmp_user/sator/lbernard/.local/bin/
export PYTHONPATH=/tmp_user/sator/lbernard/.local/lib/python2.7/site-packages/:$PYTHONPATH

# MOLA
export MOLA=/stck/lbernard/MOLA/Dev
export PYTHONPATH=$PYTHONPATH:$MOLA

mpirun -np $SLURM_NTASKS elsA.x -C xdt-runtime-tree -- compute.py 1>stdout.log 2>stderr.log