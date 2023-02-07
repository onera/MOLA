SECONDS=0

for case in `ls -d 0*/ | sort -V`; do

    echo "entering $case"
    cd $case

    while [ ! -f "COMPLETED" ] && [ ! -f "FAILED" ]; do

        echo "preprocess case $case at $SECONDS s"
        python preprocess.py $SECONDS

        if [ -f "FAILED" ]; then
            echo "WARNING: case $case cannot be launched"
            break
        fi

        echo "compute case $case at $SECONDS s"
        mpirun -np $SLURM_NTASKS elsA.x -C xdt-runtime-tree -- compute.py 1>stdout.log 2>stderr.log
        python postprocess.py

        if [ -f "NEWJOB_REQUIRED" ]; then
            rm NEWJOB_REQUIRED
            echo "LAUNCHING THIS JOB AGAIN"
            cd ..
            sbatch jobP.sh --dependency=singleton
            exit 0
        fi
    done

    cd ..

done
