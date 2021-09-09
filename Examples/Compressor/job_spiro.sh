#!/bin/bash

module purge all

export http_proxy=proxy:80 https_proxy=proxy:80 ftp_proxy=proxy:80

# ELSA+CASSIOPEE
export ELSA_MPI_LOG_FILES=OFF
export ELSA_MPI_APPEND=FALSE # See ticket 7849
export FORT_BUFFERED=true
export MPI_GROUP_MAX=8192
export MPI_COMM_MAX=8192

export ELSAVERSION=v5.0.02
export ELSAPROD=spiro_mpi
export ELSAPATHPUBLIC=/home/elsa/Public/$ELSAVERSION/Dist/bin/$ELSAPROD
source $ELSAPATHPUBLIC/.env_elsA

# PUMA
export PumaRootDir=/home/rboisard/bin/local/x86_64z/Puma_r336_spiro
export PYTHONPATH=$PumaRootDir/lib/python2.7/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$PumaRootDir/lib/python2.7:$LD_LIBRARY_PATH
export PUMA_LICENCE=$PumaRootDir/pumalicence.txt

# MOLA
export MOLA=/home/tbontemp/softs/MOLA/Dev
export PYTHONPATH=$PYTHONPATH:$MOLA



mpirun -np 8 elsA.x -C xdt-runtime-tree -- compute.py 1>stdout.log 2>stderr.log
mv OUTPUT/tmp-fields.cgns OUTPUT/fields.cgns