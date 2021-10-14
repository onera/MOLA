#!/bin/bash
#SBATCH -J <JobName>
#SBATCH --comment <AERnumber>
#SBATCH -o output.%j
#SBATCH -e error.%j
#SBATCH -t 0-15:00
#SBATCH -n <NProcs>

# ELSA+CASSIOPEE
export ELSA_MPI_LOG_FILES=OFF
export ELSA_MPI_APPEND=FALSE # See ticket 7849
export FORT_BUFFERED=true
export MPI_GROUP_MAX=8192
export MPI_COMM_MAX=8192
source /tmp_user/sator/elsa/Public/v5.0.03/Dist/bin/sator/source.me

# NUMPY SCIPY
export PATH=$PATH:/tmp_user/sator/lbernard/.local/bin/
export PYTHONPATH=/tmp_user/sator/lbernard/.local/lib/python2.7/site-packages/:$PYTHONPATH

# MOLA
export MOLA=/tmp_user/sator/tbontemp/MOLA/Dev
export PYTHONPATH=$PYTHONPATH:$MOLA

mpirun -np $SLURM_NTASKS elsA.x -C xdt-runtime-tree -- compute.py 1>stdout.log 2>stderr.log
mv OUTPUT/tmp-fields.cgns OUTPUT/fields.cgns
