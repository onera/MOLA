export SONICSVERSION=0.6.4
source /stck/sonics/LD8/sonics/$SONICSVERSION/source.sh
mpirun -np 1 python3 compute_using_sonics.py 1>stdout.log 2>stderr.log