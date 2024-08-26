#!/bin/bash

source /stck/lbernard/MOLA/Dev/src/mola/env/onera/ld/fast.sh
unset "${!OMPI_@}" "${!MPI_@}"

export KMP_WARNINGS=FALSE
export OMP_PLACES=cores
kpython -n 1 -t 8 compute.py 1>stdout.log 2>stderr.log