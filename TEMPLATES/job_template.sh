#!/bin/bash
#SBATCH -J <JobName>
#SBATCH --comment <AERnumber> # only relevant for sator
#SBATCH -o output.%j.log
#SBATCH -e error.%j.log
#SBATCH -t 0-15:00
#SBATCH -n <NumberOfProcessors>
# #SBATCH --qos <qos>
# #SBATCH --constraint="csl"
# NOTE : if job is used in SPIRO, then flag --qos (e.g. c1_test_giga)
#        must also be provided


###############################################################################
# -------------- THESE LINES MUST BE ADAPTED BY DEVELOPERS ------------------ #
export MOLAVER=Dev
if [ -f "/tmp_user/sator/lbernard/MOLA/$MOLAVER/env_MOLA.sh" ]; then
    source /tmp_user/sator/lbernard/MOLA/$MOLAVER/env_MOLA.sh
else
    source /stck/lbernard/MOLA/$MOLAVER/env_MOLA.sh
fi
if [ "$MAC" = "ld" ] ; then export OMP_NUM_THREADS=1; fi # ticket elsA 11143
###############################################################################

mpirun $OPENMPIOVERSUBSCRIBE -np $NPROCMPI elsA.x -C xdt-runtime-tree -- compute.py 1>stdout.log 2>stderr.log

if [ -f "NEWJOB_REQUIRED" ]; then
    # check if script is started via SLURM or bash
    if [ ! -z "$SLURM_JOB_ID" ];  then
        # check the original location through scontrol and $SLURM_JOB_ID
        SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
        echo "LAUNCHING ${SCRIPT_PATH} AGAIN USING SLURM"
        sbatch ${SCRIPT_PATH} --dependency=singleton
    else
        # otherwise: started with bash. Get the real location.
        SCRIPT_PATH=${0}
        echo "LAUNCHING ${SCRIPT_PATH} AGAIN USING BASH"
        ${SCRIPT_PATH} &
    fi
    rm -f NEWJOB_REQUIRED
    exit 0
fi
