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
export VPMVERSION=v0.1
export PUMAVERSION=r337
###############################################################################


export http_proxy=http://proxy.onera:80 https_proxy=http://proxy.onera:80 ftp_proxy=http://proxy.onera:80
export no_proxy=localhost,gitlab-dtis.onera,gitlab.onera.net

export ELSAVERSION=v5.1.01
export ELSA_VERBOSE_LEVEL=0 # see ticket #9689
export ELSA_MPI_LOG_FILES=OFF
export ELSA_MPI_APPEND=FALSE # See ticket 7849
export FORT_BUFFERED=true
export MPI_GROUP_MAX=8192
export MPI_COMM_MAX=8192
export ELSA_NOLOG=ON
export PYTHONUNBUFFERED=true # ticket 9685

# Detection machine
KC=`uname -n`
MAC0=$(echo $KC | grep 'n'); if [ "$MAC0" != "" ]; then export MAC="sator"; fi
MAC0=$(echo $KC | grep 'sator1'); if [ "$MAC0" != "" ]; then export MAC="sator-new"; fi
MAC0=$(echo $KC | grep 'sator2'); if [ "$MAC0" != "" ]; then export MAC="sator-new"; fi
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


if [ -n "$SLURM_NTASKS" ] ; then
    if [ $SLURM_NTASKS == 1 ] ; then
        if [ -n "$SLURM_CPUS_PER_TASK" ] ; then
            export NPROCMPI=$SLURM_CPUS_PER_TASK
        elif [ -n "$SLURM_CPUS_ON_NODE" ] ; then
            export NPROCMPI=$SLURM_CPUS_ON_NODE
        else
            export NPROCMPI=$(nproc)
        fi
    else
        export NPROCMPI=$SLURM_NTASKS
    fi
else
    export NPROCMPI=$(nproc)
fi



if [ "$MAC" = "sator" ] ; then
    echo nproc is $NPROCMPI
    if [ $NPROCMPI == 48 ] || [ $NPROCMPI == 44 ] ; then
        echo switching to sator-new
        export MAC="sator-new"
    fi

fi

echo Machine is $MAC

if [ "$MAC" = "spiro" ]; then
    source /stck/elsa/Public/$ELSAVERSION/Dist/bin/spiro3_mpi/.env_elsA
    export PYTHONPATH=$EXTPYLIB/lib/python3.7/site-packages/:$PYTHONPATH
    export PATH=$EXTPYLIB/bin:$PATH
    module load texlive/2016 # for LaTeX rendering in matplotlib with STIX font

    # PUMA
    export PumaRootDir=/stck/rboisard/bin/local/x86_64z/Puma_${PUMAVERSION}_spiro3
    export PYTHONPATH=$PumaRootDir/lib/python${PYTHONVR}/site-packages:$PYTHONPATH
    export PYTHONPATH=$PumaRootDir/lib/python${PYTHONVR}/site-packages/PUMA:$PYTHONPATH
    export LD_LIBRARY_PATH=$PumaRootDir/lib/python${PYTHONVR}:$LD_LIBRARY_PATH
    export PUMA_LICENCE=$PumaRootDir/pumalicence.txt

    # VPM
    export VPMPATH=/stck/lbernard/VPM/$VPMVERSION/$MAC
    export PATH=$VPMPATH:$PATH
    export LD_LIBRARY_PATH=$VPMPATH/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/stck/benoit/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/stck/benoit/opencascade/lib:/opt/tools/hdf5-1.10.5-intel-19-impi-19/lib
    export PYTHONPATH=$VPMPATH:$PYTHONPATH
    export PYTHONPATH=$VPMPATH/lib/python${PYTHONVR}/site-packages:$PYTHONPATH

    alias treelab='python3 $TREELAB/TreeLab/GUI/__init__.py'
    alias python='python3'
    source ~/.bashrc


elif [ "$MAC" = "visio" ]; then
    export ELSAVERSION=UNAVAILABLE # TODO adapt this once #10587 fixed
    # source /stck/elsa/Public/$ELSAVERSION/Dist/bin/centos6_mpi/.env_elsA
    # export PYTHONPATH=$EXTPYLIB/lib/python2.7/site-packages/:$PYTHONPATH
    # export PATH=$EXTPYLIB/bin:$PATH

    . /etc/profile.d/modules-dri.sh
    module load subversion/1.7.6
    module load python/3.6.1
    export PYTHONVR=3.6
    module unload $(module -t list 2>&1 | grep -i intel)
    module load gcc/4.8.1
    module load intel/17.0.4
    module load impi/17
    module load texlive/2016 # for LaTeX rendering in matplotlib with STIX font
    export OMP_NUM_THREADS=16
    alias python=python3
    export PYTHONEXE=python3


    # CAVEAT -> PUMA is installed in python v2 only in visio
    # export PumaRootDir=/stck/rboisard/bin/local/x86_64z/Puma_${PUMAVERSION}_centos6
    # export PYTHONPATH=$PumaRootDir/lib/python2.7/site-packages:$PYTHONPATH
    # export PYTHONPATH=$PumaRootDir/lib/python2.7/site-packages/PUMA:$PYTHONPATH
    # export LD_LIBRARY_PATH=$PumaRootDir/lib/python2.7:$LD_LIBRARY_PATH
    # export PUMA_LICENCE=$PumaRootDir/pumalicence.txt

    export VPMPATH=/stck/lbernard/VPM/$VPMVERSION/$MAC
    export PATH=$PATH:$VPMPATH
    # export PATH=/stck/lbernard/.local/bin:$PATH
    export PATH=$VPMPATH:$PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/stck/benoit/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/stck/benoit/opencascade2/lib:/usr/local/hdf5-intel-1.8.8/lib:/usr/local/gtk+3/lib
    export PYTHONPATH=$VPMPATH:$PYTHONPATH
    export PYTHONPATH=$VPMPATH/lib/python${PYTHONVR}/site-packages:$PYTHONPATH

    alias treelab='python3 $TREELAB/TreeLab/GUI/__init__.py'
    alias python='python3'


elif [ "$MAC" = "ld" ]; then
    EL8=`uname -r|grep el8`
    if [ "$EL8" ]; then
        echo 'loading MOLA environment for CentOS 8'
        source /stck/elsa/Public/v5.0.04dev/Dist/bin/local-os8_mpi/.env_elsA # TODO adapt this once #10587 fixed
        module load texlive/2016 # for LaTeX rendering in matplotlib with STIX font
    else
        echo 'loading MOLA environment for CentOS 7'
        source /stck/elsa/Public/$ELSAVERSION/Dist/bin/eos-intel3_mpi/.env_elsA
        module load texlive/2016 # for LaTeX rendering in matplotlib with STIX font

        # VPM
        export VPMPATH=/stck/lbernard/VPM/$VPMVERSION/$MAC
        export PATH=$VPMPATH:$PATH
        export LD_LIBRARY_PATH=$VPMPATH/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/stck/benoit/lib
        export LD_LIBRARY_PATH=/stck/benoit/opencascade/lib:/opt/tools/hdf5-1.10.5-intel-19-impi-19/lib:$LD_LIBRARY_PATH
        export PYTHONPATH=$VPMPATH:$PYTHONPATH
        export PYTHONPATH=$VPMPATH/lib/python${PYTHONVR}/site-packages:$PYTHONPATH

    fi

    alias treelab='python3 $TREELAB/TreeLab/GUI/__init__.py'
    alias python='python3'

elif [ "$MAC" = "sator" ]; then
    source /tmp_user/sator/elsa/Public/v5.0.04/Dist/bin/sator/.env_elsA # TODO adapt this once #10587 fixed
    export MOLA=$MOLASATOR
    export TREELAB=$TREELABSATOR
    export PYTHONPATH=$EXTPYLIBSATOR/lib/python2.7/site-packages/:$PYTHONPATH
    export PATH=$EXTPYLIBSATOR/bin:$PATH

    export PumaRootDir=/tmp_user/sator/rboisard/TOOLS/Puma_${PUMAVERSION}_sator
    export PYTHONPATH=$PumaRootDir/lib/python2.7/site-packages:$PYTHONPATH
    export PYTHONPATH=$PumaRootDir/lib/python2.7/site-packages/PUMA:$PYTHONPATH
    export LD_LIBRARY_PATH=$PumaRootDir/lib/python2.7:$LD_LIBRARY_PATH
    export PUMA_LICENCE=$PumaRootDir/pumalicence.txt

elif [ "$MAC" = "sator-new" ]; then
    source /tmp_user/sator/elsa/Public/$ELSAVERSION/Dist/bin/sator_new21/.env_elsA
    export MOLA=$MOLASATOR
    export TREELAB=$TREELABSATOR
    export PYTHONPATH=$EXTPYLIBSATOR/lib/python3.7/site-packages/:$PYTHONPATH
    export PATH=$EXTPYLIBSATOR/bin:$PATH

    # PUMA incompatible with intel21 ?
    export PumaRootDir=/tmp_user/sator/rboisard/TOOLS/Puma_${PUMAVERSION}_satornew
    export PYTHONPATH=$PumaRootDir/lib/python3.7/site-packages:$PYTHONPATH
    export PYTHONPATH=$PumaRootDir/lib/python3.7/site-packages/PUMA:$PYTHONPATH
    export LD_LIBRARY_PATH=$PumaRootDir/lib/python3.7:$LD_LIBRARY_PATH
    export PUMA_LICENCE=$PumaRootDir/pumalicence.txt

    # VPM
    export VPMPATH=/tmp_user/sator/lbernard/VPM/$VPMVERSION/$MAC
    export PATH=$VPMPATH:$PATH
    export LD_LIBRARY_PATH=$VPMPATH/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$VPMPATH:$LD_LIBRARY_PATH
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp_user/sator/lbernard/lib
    export PYTHONPATH=$VPMPATH:$PYTHONPATH
    export PYTHONPATH=$VPMPATH/lib/python${PYTHONVR}/site-packages:$PYTHONPATH


    alias treelab='python3 $TREELAB/TreeLab/GUI/__init__.py'
    alias python='python3'

else
    echo -e "\033[91mERROR: MACHINE $KC NOT INCLUDED IN MOLA ENVIRONMENT\033[0m"
    exit 0
fi

export PYTHONPATH=$MOLA:$TREELAB:$PYTHONPATH

alias mola_version="python -c 'import MOLA.InternalShortcuts as J;J.printEnvironment()'"

alias mola_jobsqueue_sator="python -c 'import MOLA.JobManager as JM;JM.getCurrentJobsStatus()'"

mola_version
