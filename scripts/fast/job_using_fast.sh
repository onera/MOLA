#!/bin/bash
export MACHINE=ld
export CASSIOPEE=/stck/cassiope/git/releases/Cassiopee/v4.1
source $CASSIOPEE/Dist/sh_Cassiopee_local &> /dev/null

export KMP_WARNINGS=FALSE
export OMP_NUM_THREADS=8

python3 compute_using_fast.py 1>stdout.log 2>stderr.log