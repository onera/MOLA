#! /bin/sh
source /etc/bashrc
module purge
unset PYTHONPATH
shopt -s expand_aliases

###############################################################################
# ---------------- THESE LINES MUST BE ADAPTED BY DEVELOPERS ---------------- #
export MOLA=/stck/lbernard/MOLA/Dev
export TREELAB=/stck/lbernard/TreeLab/dev
export EXTPYLIB=$MOLA/ExternalPythonPackages
export MOLASATOR=/tmp_user/sator/lbernard/MOLA/Dev
export TREELABSATOR=/tmp_user/sator/lbernard/TreeLab/dev
export EXTPYLIBSATOR=$MOLASATOR/ExternalPythonPackages
###############################################################################


export http_proxy=proxy:80 https_proxy=proxy:80 ftp_proxy=proxy:80

export ELSA_MPI_LOG_FILES=OFF
export ELSA_MPI_APPEND=FALSE # See ticket 7849
export FORT_BUFFERED=true
export MPI_GROUP_MAX=8192
export MPI_COMM_MAX=8192
export ELSAVERSION=v5.0.04
export ELSA_NOLOG=ON

# Detection machine
KC=`uname -n`
MAC0=$(echo $KC | grep 'n'); if [ "$MAC0" != "" ]; then export MAC="sator"; fi
MAC0=$(echo $KC | grep 'sator1'); if [ "$MAC0" != "" ]; then export MAC="sator"; fi
MAC0=$(echo $KC | grep 'sator2'); if [ "$MAC0" != "" ]; then export MAC="sator"; fi
MAC0=$(echo $KC | grep 'sator3'); if [ "$MAC0" != "" ]; then export MAC="sator"; fi
MAC0=$(echo $KC | grep 'sator4'); if [ "$MAC0" != "" ]; then export MAC="sator"; fi
MAC0=$(echo $KC | grep 'sator5'); if [ "$MAC0" != "" ]; then export MAC="sator-new"; fi
MAC0=$(echo $KC | grep 'sator6'); if [ "$MAC0" != "" ]; then export MAC="sator-new"; fi
MAC0=$(echo $KC | grep 'ld'); if [ "$MAC0" != "" ]; then export MAC="ld"; fi
MAC0=$(echo $KC | grep 'eos'); if [ "$MAC0" != "" ]; then export MAC="ld"; fi
MAC0=$(echo $KC | grep 'clausius'); if [ "$MAC0" != "" ]; then export MAC="ld"; fi
MAC0=$(echo $KC | grep 'visung'); if [ "$MAC0" != "" ]; then export MAC="ld"; fi
MAC0=$(echo $KC | grep 'visio'); if [ "$MAC0" != "" ]; then export MAC="visio"; fi
MAC0=$(echo $KC | grep 'celeste'); if [ "$MAC0" != "" ]; then export MAC="visio"; fi
MAC0=$(echo $KC | grep 'elmer'); if [ "$MAC0" != "" ]; then export MAC="visio"; fi
MAC0=$(echo $KC | grep 'dumbo'); if [ "$MAC0" != "" ]; then export MAC="visio"; fi
MAC0=$(echo $KC | grep 'ganesh'); if [ "$MAC0" != "" ]; then export MAC="visio"; fi
MAC0=$(echo $KC | grep 'spiro'); if [ "$MAC0" != "" ]; then export MAC="spiro"; fi


if { [ "$MAC" = "sator" ] && [ -n "$SLURM_CPUS_ON_NODE" ]; } ; then
    if [ $(nproc) == 48 ] ; then
        export MAC="sator-new"
    fi
fi


if [ "$MAC" = "spiro" ]; then
    source /stck/elsa/Public/$ELSAVERSION/Dist/bin/spiro_mpi/.env_elsA
    export PYTHONPATH=$EXTPYLIB/lib/python2.7/site-packages/:$PYTHONPATH
    export PATH=$EXTPYLIB/bin:$PATH
    module load texlive/2016 # for LaTeX rendering in matplotlib with STIX font

    export PumaRootDir=/stck/rboisard/bin/local/x86_64z/Puma_r337_spiro
    export PYTHONPATH=$PumaRootDir/lib/python2.7/site-packages:$PYTHONPATH
    export LD_LIBRARY_PATH=$PumaRootDir/lib/python2.7:$LD_LIBRARY_PATH
    export PUMA_LICENCE=$PumaRootDir/pumalicence.txt

elif [ "$MAC" = "visio" ]; then
    export ELSAVERSION=v5.0.03 # TODO adapt this once #9666 fixed
    source /stck/elsa/Public/$ELSAVERSION/Dist/bin/centos6_mpi/.env_elsA
    export PYTHONPATH=$EXTPYLIB/lib/python2.7/site-packages/:$PYTHONPATH
    export PATH=$EXTPYLIB/bin:$PATH
    module load texlive/2016 # for LaTeX rendering in matplotlib with STIX font

    export PumaRootDir=/stck/rboisard/bin/local/x86_64z/Puma_r337_centos6
    export PYTHONPATH=$PumaRootDir/lib/python2.7/site-packages:$PYTHONPATH
    export LD_LIBRARY_PATH=$PumaRootDir/lib/python2.7:$LD_LIBRARY_PATH
    export PUMA_LICENCE=$PumaRootDir/pumalicence.txt


elif [ "$MAC" = "ld" ]; then
    export ELSAVERSION=v5.0.03 # TODO adapt this once #9671 fixed
    source /stck/elsa/Public/$ELSAVERSION/Dist/bin/eos-intel_mpi/.env_elsA
    export PYTHONPATH=$EXTPYLIB/lib/python2.7/site-packages/:$PYTHONPATH
    export PATH=$EXTPYLIB/bin:$PATH
    module load texlive/2016 # for LaTeX rendering in matplotlib with STIX font

    export PumaRootDir=/stck/rboisard/bin/local/x86_64z/Puma_r337_eos
    export PYTHONPATH=$PumaRootDir/lib/python2.7/site-packages:$PYTHONPATH
    export LD_LIBRARY_PATH=$PumaRootDir/lib/python2.7:$LD_LIBRARY_PATH
    export PUMA_LICENCE=$PumaRootDir/pumalicence.txt


elif [ "$MAC" = "sator" ]; then
    source /tmp_user/sator/elsa/Public/$ELSAVERSION/Dist/bin/sator3/.env_elsA
    export MOLA=$MOLASATOR
    export TREELAB=$TREELABSATOR
    export PYTHONPATH=$EXTPYLIBSATOR/lib/python3.7/site-packages/:$PYTHONPATH
    export PATH=$EXTPYLIBSATOR/bin:$PATH

    export PumaRootDir=/tmp_user/sator/rboisard/TOOLS/Puma_r337_sator3
    export PYTHONPATH=$PumaRootDir/lib/python3.7/site-packages:$PYTHONPATH
    export LD_LIBRARY_PATH=$PumaRootDir/lib/python3.7:$LD_LIBRARY_PATH
    export PUMA_LICENCE=$PumaRootDir/pumalicence.txt

    alias treelab='python3 $TREELAB/GUI/treelab.py '
    alias python='python3'

elif [ "$MAC" = "sator-new" ]; then
    source /tmp_user/sator/elsa/Public/$ELSAVERSION/Dist/bin/sator_new21/.env_elsA
    export MOLA=$MOLASATOR
    export TREELAB=$TREELABSATOR
    export PYTHONPATH=$EXTPYLIBSATOR/lib/python3.7/site-packages/:$PYTHONPATH
    export PATH=$EXTPYLIBSATOR/bin:$PATH

    export PumaRootDir=/tmp_user/sator/rboisard/TOOLS/Puma_r337_satornew
    export PYTHONPATH=$PumaRootDir/lib/python3.7/site-packages:$PYTHONPATH
    export LD_LIBRARY_PATH=$PumaRootDir/lib/python3.7:$LD_LIBRARY_PATH
    export PUMA_LICENCE=$PumaRootDir/pumalicence.txt

    alias treelab='python3 $TREELAB/GUI/treelab.py '
    alias python='python3'

else
    echo -e "\033[91mERROR: MACHINE $KC NOT INCLUDED IN MOLA ENVIRONMENT\033[0m"
    exit 0
fi

export PYTHONPATH=$MOLA:$TREELAB:$PYTHONPATH

echo "using MOLA environment for $MAC"
