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
export ELSAPATHPUBLIC=/stck/elsa/Public/$ELSAVERSION/Dist/bin/$ELSAPROD
source $ELSAPATHPUBLIC/.env_elsA

export PYTHONPATH=/stck/lbernard/.local/lib/python2.7/site-packages/:$PYTHONPATH
export PATH=$PATH:/opt/tools/texlive/2016/bin/x86_64-linux/

# MOLA
export MOLA=/stck/tbontemp/softs/MOLA/Dev
export PYTHONPATH=$PYTHONPATH:$MOLA