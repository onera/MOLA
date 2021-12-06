#!/bin/bash

module purge all

export http_proxy=proxy:80 https_proxy=proxy:80 ftp_proxy=proxy:80

# ELSA+CASSIOPEE
export ELSA_MPI_LOG_FILES=OFF
export ELSA_MPI_APPEND=FALSE # See ticket 7849
export FORT_BUFFERED=true
export MPI_GROUP_MAX=8192
export MPI_COMM_MAX=8192

export ELSAVERSION=v5.0.03
export ELSAPROD=spiro_mpi
export ELSAPATHPUBLIC=/stck/elsa/Public/$ELSAVERSION/Dist/bin/$ELSAPROD
source $ELSAPATHPUBLIC/.env_elsA

# MOLA
export MOLA=/stck/lbernard/MOLA/v1.12
export PYTHONPATH=$PYTHONPATH:$MOLA

mpirun -np 8 elsA.x -C xdt-runtime-tree -- compute.py 1>stdout.log 2>stderr.log

