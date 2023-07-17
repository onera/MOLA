#!/bin/bash
#SBATCH -J <JobName>
#SBATCH --comment <AERnumber> # only relevant for sator
#SBATCH -o output.%j.log
#SBATCH -e error.%j.log
#SBATCH -t <TimeLimit>
#SBATCH -n <NumberOfProcessors>
#SBATCH --constraint=<SlurmConstraint>
#SBATCH --qos=<SlurmQualityOfService>

###############################################################################
# -------------- THESE LINES MUST BE ADAPTED BY DEVELOPERS ------------------ #
if [ -f "/tmp_user/sator/lbernard/MOLA/Dev/env_MOLA.sh" ]; then
    source /tmp_user/sator/lbernard/MOLA/Dev/env_MOLA.sh
else
    source /stck/lbernard/MOLA/Dev/env_MOLA.sh
fi
###############################################################################

mpirun $OPENMPIOVERSUBSCRIBE -np $NPROCMPI elsA.x -C xdt-runtime-tree -- compute.py 1>stdout.log 2>stderr.log

if [ -f "NEWJOB_REQUIRED" ]; then
    rm -f  NEWJOB_REQUIRED
    job_filename="${0##*/}"
    echo "LAUNCHING ${job_filename} AGAIN"
    sbatch ${job_filename} --dependency=singleton
    exit 0
fi
