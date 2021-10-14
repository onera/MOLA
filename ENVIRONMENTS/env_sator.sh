#!/bin/bash
module purge all

export http_proxy=proxy:80 https_proxy=proxy:80 ftp_proxy=proxy:80

# ELSA+CASSIOPEE
export ELSA_MPI_LOG_FILES=OFF
export ELSA_MPI_APPEND=FALSE # See ticket 7849
export FORT_BUFFERED=true
export MPI_GROUP_MAX=8192
export MPI_COMM_MAX=8192
source /tmp_user/sator/elsa/Public/v5.0.03/Dist/bin/sator/source.me

# PUMA
export PumaRootDir=/tmp_user/sator/rboisard/TOOLS/Puma_r336
export PYTHONPATH=$PumaRootDir/lib/python2.7/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$PumaRootDir/lib/python2.7:$LD_LIBRARY_PATH
export PUMA_LICENCE=$PumaRootDir/pumalicence.txt

# NUMPY SCIPY
export PATH=$PATH:/tmp_user/sator/lbernard/.local/bin/
export PYTHONPATH=/tmp_user/sator/lbernard/.local/lib/python2.7/site-packages/:$PYTHONPATH

# MOLA
export MOLA=/tmp_user/sator/lbernard/MOLA/Dev
export PYTHONPATH=$PYTHONPATH:$MOLA
