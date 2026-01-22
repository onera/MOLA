#!/bin/bash

export ELSAVERSION=v5.4.01 
export ELSA_VERBOSE_LEVEL=0 # cf elsA ticket 9689
export ELSA_MPI_LOG_FILES=OFF
export ELSA_MPI_APPEND=FALSE # cf elsA ticket 7849
export ELSA_NOLOG=ON
# ELSA_MEMORY_VERBOSE=TRUE https://elsa.onera.fr/issues/10621#note-19

source /stck/elsa/Public/$ELSAVERSION/Dist/bin/local-os8_mpi/.env_elsA
# source /tmp_user/sator/elsa/Public/$ELSAVERSION/Dist/bin/sator_new22/.env_elsA

mpirun $OPENMPIOVERSUBSCRIBE -np 8 elsA.x -C xdt-runtime-tree compute_using_elsa.py 1>stdout.log 2>stderr.log
