#!/bin/bash
#SBATCH -J <JobName>
#SBATCH --comment <AERnumber> # only relevant for sator
#SBATCH -o output.%j.log
#SBATCH -e error.%j.log
#SBATCH -t 0-15:00
#SBATCH -n <NProcs>

# NOTE : if job is used in SPIRO, then flag --qos (e.g. c1_test_giga)
#        must also be provided


###############################################################################
# -------------- THESE LINES MUST BE ADAPTED BY DEVELOPERS ------------------ #
if [ -f "/tmp_user/sator/lbernard/MOLA/Dev/env_MOLA.sh" ]; then
    source /tmp_user/sator/lbernard/MOLA/Dev/env_MOLA.sh
else
    source /stck/lbernard/MOLA/Dev/env_MOLA.sh
fi
###############################################################################


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

mpirun -np $NPROCMPI elsA.x -C xdt-runtime-tree -- compute.py 1>stdout.log 2>stderr.log
