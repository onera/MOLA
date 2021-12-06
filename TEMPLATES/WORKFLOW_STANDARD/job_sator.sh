#!/bin/bash
#SBATCH -J ID19
#SBATCH --comment 28754013F
#SBATCH -o output.ID19
#SBATCH -e error.ID19
#SBATCH -t 0-15:00
#SBATCH -n 245

# ELSA+CASSIOPEE
export ELSA_MPI_LOG_FILES=OFF
export ELSA_MPI_APPEND=FALSE # See ticket 7849
export FORT_BUFFERED=true
export MPI_GROUP_MAX=8192
export MPI_COMM_MAX=8192
export ELSA_NOLOG=ON
source /tmp_user/sator/elsa/Public/v5.0.03/Dist/bin/sator/source.me

# PUMA
export PumaRootDir=/tmp_user/sator/rboisard/TOOLS/Puma_r336
export PYTHONPATH=$PumaRootDir/lib/python2.7/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$PumaRootDir/lib/python2.7:$LD_LIBRARY_PATH
export PUMA_LICENCE=$PumaRootDir/pumalicence.txt

# export PYTHONPATH=/stck/lbernard/.local/lib/python2.7/site-packages/:$PYTHONPATH

# NUMPY SCIPY
export PATH=$PATH:/tmp_user/sator/lbernard/.local/bin/
export PYTHONPATH=/tmp_user/sator/lbernard/.local/lib/python2.7/site-packages/:$PYTHONPATH

# MOLA
export MOLA=/tmp_user/sator/lbernard/MOLA/Dev
export PYTHONPATH=$PYTHONPATH:$MOLA


mpirun -np $SLURM_NTASKS elsA.x -C xdt-runtime-tree -- compute.py 1>stdout.log 2>stderr.log
