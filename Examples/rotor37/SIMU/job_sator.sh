#!/bin/bash
#SBATCH -J IMOTHEP
#SBATCH --comment 28771019F  # AER IMOTHEP
#SBATCH -o output.%j.log
#SBATCH -e error.%j.log
#SBATCH -t 0-15:00
#SBATCH -n 8

# ELSA+CASSIOPEE
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
#export MOLA=/tmp_user/sator/lbernard/MOLA/v1.11
export MOLA=/tmp_user/sator/tbontemp/MOLA/Dev
export PYTHONPATH=$PYTHONPATH:$MOLA


mpirun -np $SLURM_NTASKS elsA.x -C xdt-runtime-tree -- compute.py 1>stdout.log 2>stderr.log
mv OUTPUT/tmp-fields.cgns OUTPUT/fields.cgns
