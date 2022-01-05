#!/bin/bash
#SBATCH -J <JobName>
#SBATCH --comment <AERnumber> # only relevant for sator
#SBATCH -o output.%j.log
#SBATCH -e error.%j.log
#SBATCH -t 0-15:00
#SBATCH -n <NProcs>
#SBATCH --qos=c1_test_giga    # relevant for spiro (max time 6h)


###############################################################################
# -------------- THESE LINES MUST TO BE ADAPTED BY DEVELOPERS --------------- #
if [ -f "/tmp_user/sator/lbernard/MOLA/Dev/ENVIRONMENTS/env.sh" ]; then
    source /tmp_user/sator/lbernard/MOLA/Dev/ENVIRONMENTS/env.sh
else
    source /stck/lbernard/MOLA/Dev/ENVIRONMENTS/env.sh
fi
###############################################################################


if [ -n "$SLURM_CPUS_PER_TASK" ] ; then
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
elif [ -n "$SLURM_CPUS_ON_NODE" ] ; then
    export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
else
    export OMP_NUM_THREADS=$(nproc)
fi

if [ -n "$SLURM_NTASKS" ] ; then
    export NPROCMPI=$SLURM_NTASKS
else
    export NPROCMPI=$(nproc)
fi

mpirun -np $NPROCMPI elsA.x -C xdt-runtime-tree -- compute.py 1>stdout.log 2>stderr.log
