#! /bin/sh
source /etc/bashrc
module purge
unset PYTHONPATH
shopt -s expand_aliases
ulimit -s unlimited # in order to allow arbitrary use of stack (required by VPM)


###############################################################################
# ---------------- THESE LINES MUST BE ADAPTED BY DEVELOPERS ---------------- #
export MOLA=/stck/lbernard/MOLA/Dev
export MOLASATOR=/tmp_user/sator/lbernard/MOLA/Dev
export VPMVERSION=v0.2
export PUMAVERSION=r337
export TURBOVERSION=v1.2.2
export ERSTAZVERSION=vT
export MOLAext=/stck/lbernard/MOLA/Dev/ext # you should not modify this line
export MOLASATORext=/tmp_user/sator/lbernard/MOLA/Dev/ext # you should not modify this line
###############################################################################


export http_proxy=http://proxy.onera:80 https_proxy=http://proxy.onera:80 ftp_proxy=http://proxy.onera:80
export no_proxy=localhost,gitlab-dtis.onera,gitlab.onera.net

export ELSAVERSION=v5.1.03 # except CentOS < 7 (ld7, visung, visio) see elsA ticket 10870
export ELSA_VERBOSE_LEVEL=0 # cf elsA ticket 9689
export ELSA_MPI_LOG_FILES=OFF
export ELSA_MPI_APPEND=FALSE # cf elsA ticket 7849
export FORT_BUFFERED=true
export MPI_GROUP_MAX=8192
export MPI_COMM_MAX=8192
export ELSA_NOLOG=ON
export PYTHONUNBUFFERED=true # cf ticket 9685

# Detection machine
KC=`uname -n`
MAC0=$(echo $KC | grep 'n'); if [ "$MAC0" != "" ]; then export MAC="sator"; fi
MAC0=$(echo $KC | grep 'sator'); if [ "$MAC0" != "" ]; then export MAC="sator"; fi
MAC0=$(echo $KC | grep 'ld'); if [ "$MAC0" != "" ]; then export MAC="ld"; fi
MAC0=$(echo $KC | grep 'eos'); if [ "$MAC0" != "" ]; then export MAC="ld"; fi
MAC0=$(echo $KC | grep 'clausius'); if [ "$MAC0" != "" ]; then export MAC="ld"; fi
MAC0=$(echo $KC | grep 'visung'); if [ "$MAC0" != "" ]; then export MAC="visung"; fi
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


EL8=`uname -r|grep el8`


if [ "$MAC" = "spiro" ]; then
    if [ "$EL8" ]; then
            echo 'loading MOLA environment for CentOS 8'
    else
        echo -e "\033[91mERROR: YOU MUST CONNECT TO SPIRO CENTOS 8 (temporarily ssh -X spiro-el8)\033[0m"
        exit 0
    fi
    source /stck/elsa/Public/$ELSAVERSION/Dist/bin/spiro-el8_mpi/.env_elsA

    export PYTHONPATH=$MOLAext/spiro_el8/lib/python3.7/site-packages/:$PYTHONPATH
    export PATH=$MOLAext/spiro_el8/bin:$PATH


    # # PUMA
    # export PumaRootDir=/stck/rboisard/bin/local/x86_64z/Puma_${PUMAVERSION}_spiro3
    # export PYTHONPATH=$PumaRootDir/lib/python${PYTHONVR}/site-packages:$PYTHONPATH
    # export PYTHONPATH=$PumaRootDir/lib/python${PYTHONVR}/site-packages/PUMA:$PYTHONPATH
    # export LD_LIBRARY_PATH=$PumaRootDir/lib/python${PYTHONVR}:$LD_LIBRARY_PATH
    # export PUMA_LICENCE=$PumaRootDir/pumalicence.txt

    # VPM
    export VPMPATH=/stck/lbernard/VPM/$VPMVERSION/spiro_el8
    export PATH=$VPMPATH:$PATH
    export LD_LIBRARY_PATH=$VPMPATH/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/stck/benoit/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/stck/benoit/opencascade/lib:/opt/tools/hdf5-1.10.5-intel-19-impi-19/lib
    export PYTHONPATH=$VPMPATH:$PYTHONPATH
    export PYTHONPATH=$VPMPATH/lib/python${PYTHONVR}/site-packages:$PYTHONPATH

    # turbo
    export PYTHONPATH=/stck/jmarty/TOOLS/turbo/install/$TURBOVERSION/env_elsA_$ELSAVERSION/spiro3_mpi/lib/python3.7/site-packages/:$PYTHONPATH
    
    # ErstaZ
    export EZPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/bin/spiro
    export PYTHONPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/module_python/python3:$PYTHONPATH

    # MAIA
    export MAIA_HOME=/scratchm/jcoulet/aa_install_py3/maia/opt-cfd5_21/
    export LD_LIBRARY_PATH=$MAIA_HOME/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=$MAIA_HOME/lib/python3.7/site-packages:$PYTHONPATH


elif [ "$MAC" = "visio" ]; then
    export ELSAVERSION=UNAVAILABLE # TODO adapt this once #10587 fixed
    echo -e "\033[93mWARNING: elsA is not installed yet in VISIO CentOS 6\033[0m"
    echo -e "\033[93mwatch https://elsa.onera.fr/issues/10587 for more information\033[0m"

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
    export VPMPATH=/stck/lbernard/VPM/$VPMVERSION/vis_r8
    export PATH=$PATH:$VPMPATH
    # export PATH=/stck/lbernard/.local/bin:$PATH
    export PATH=$VPMPATH:$PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/stck/benoit/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/stck/benoit/opencascade2/lib:/usr/local/hdf5-intel-1.8.8/lib:/usr/local/gtk+3/lib
    export PYTHONPATH=$VPMPATH:$PYTHONPATH
    export PYTHONPATH=$VPMPATH/lib/python${PYTHONVR}/site-packages:$PYTHONPATH

    # turbo 
    export TURBOVERSION=UNAVAILABLE 

    # ErstaZ
    export EZPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/bin/visio
    export PYTHONPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/module_python/python3:$PYTHONPATH

elif [ "$MAC" = "visung" ]; then
    ELSAVERSION=v5.1.02
    source /stck/elsa/Public/$ELSAVERSION/Dist/bin/eos-intel3_mpi/.env_elsA
    module load texlive/2016 # for LaTeX rendering in matplotlib with STIX font
    module load freetype/2.10.2
    module load pointwise/18.6R1

    module load texlive/2016 # for LaTeX rendering in matplotlib with STIX font

    export PYTHONEXE=python3
    export PYTHONVR=3.6
    alias python=python3
    module load intel/17.0.4
    module load impi/17
    module load hdf5/1.8.8

    export PYTHONPATH=$MOLAext/ld7/lib/python3.7/site-packages/:$PYTHONPATH
    export PATH=$MOLAext/ld7/bin:$PATH


    # PUMA # not correctly installed !
    # export PumaRootDir=/stck/rboisard/bin/local/x86_64z/Puma_${PUMAVERSION}_eos3
    # export PYTHONPATH=$PumaRootDir/lib/python${PYTHONVR}/site-packages:$PYTHONPATH
    # export PYTHONPATH=$PumaRootDir/lib/python${PYTHONVR}/site-packages/PUMA:$PYTHONPATH
    # export LD_LIBRARY_PATH=$PumaRootDir/lib/python${PYTHONVR}:$LD_LIBRARY_PATH
    # export PUMA_LICENCE=$PumaRootDir/pumalicence.txt

    # VPM
    export VPMPATH=/stck/lbernard/VPM/$VPMVERSION/vis_r8
    export PATH=$VPMPATH:$VPMPATH/lib:$PATH
    export LD_LIBRARY_PATH=$VPMPATH:$VPMPATH/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/stck/benoit/lib
    export LD_LIBRARY_PATH=$VPMPATH:$VPMPATH/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=$VPMPATH:$PYTHONPATH
    export PYTHONPATH=$VPMPATH/lib/python${PYTHONVR}/site-packages:$PYTHONPATH

    # turbo
    export PYTHONPATH=/stck/jmarty/TOOLS/turbo/install/$TURBOVERSION/env_elsA_$ELSAVERSION/eos-intel3_mpi/lib/python3.6/site-packages/:$PYTHONPATH

    # ErstaZ
    export EZPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/bin/visio
    export PYTHONPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/module_python/python3:$PYTHONPATH

elif [ "$MAC" = "ld" ]; then
    if [ "$EL8" ]; then
        echo 'loading MOLA environment for CentOS 8'
        source /stck/elsa/Public/v5.1.02/Dist/bin/local-os8_mpi/.env_elsA
        module load hdf5/1.8.17-intel2120
        module load texlive/2016 # for LaTeX rendering in matplotlib with STIX font

        alias python=python3

        export PYTHONPATH=$MOLAext/ld8/lib/python3.6/site-packages/:$PYTHONPATH
        export PATH=$MOLAext/ld8/bin:$PATH


        # PUMA
        export PumaRootDir=/stck/rboisard/bin/local/x86_64z/Puma_${PUMAVERSION}_eos3
        export PYTHONPATH=$PumaRootDir/lib/python${PYTHONVR}/site-packages:$PYTHONPATH
        export PYTHONPATH=$PumaRootDir/lib/python${PYTHONVR}/site-packages/PUMA:$PYTHONPATH
        export LD_LIBRARY_PATH=$PumaRootDir/lib/python${PYTHONVR}:$LD_LIBRARY_PATH
        export PUMA_LICENCE=$PumaRootDir/pumalicence.txt

        # VPM
        export VPMPATH=/stck/lbernard/VPM/$VPMVERSION/${MAC}8
        export PATH=$VPMPATH:$VPMPATH/lib:$PATH
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/stck/benoit/lib
        export LD_LIBRARY_PATH=$VPMPATH:$VPMPATH/lib:$LD_LIBRARY_PATH
        export PYTHONPATH=$VPMPATH:$PYTHONPATH
        export PYTHONPATH=$VPMPATH/lib/python${PYTHONVR}/site-packages:$PYTHONPATH

        # turbo 
        export TURBOVERSION=UNAVAILABLE 

        # ErstaZ
        export EZPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/bin/centos8
        export PYTHONPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/module_python/python3:$PYTHONPATH

    else
        echo 'loading MOLA environment for CentOS 7'
        # ELSAVERSION=v5.1.02
        source /stck/elsa/Public/$ELSAVERSION/Dist/bin/eos-intel3_mpi/.env_elsA
        module load texlive/2016 # for LaTeX rendering in matplotlib with STIX font

        export PYTHONEXE=python3
        export PYTHONVR=3.6
        alias python=python3
        module load intel/17.0.4
        module load impi/17
        module load hdf5/1.8.8-intel-15

        export PYTHONPATH=$MOLAext/ld7/lib/python3.6/site-packages/:$PYTHONPATH
        export PATH=$MOLAext/ld7/bin:$PATH

        # PUMA # not correctly installed !
        # export PumaRootDir=/stck/rboisard/bin/local/x86_64z/Puma_${PUMAVERSION}_eos3
        # export PYTHONPATH=$PumaRootDir/lib/python${PYTHONVR}/site-packages:$PYTHONPATH
        # export PYTHONPATH=$PumaRootDir/lib/python${PYTHONVR}/site-packages/PUMA:$PYTHONPATH
        # export LD_LIBRARY_PATH=$PumaRootDir/lib/python${PYTHONVR}:$LD_LIBRARY_PATH
        # export PUMA_LICENCE=$PumaRootDir/pumalicence.txt

        # VPM
        export VPMPATH=/stck/lbernard/VPM/$VPMVERSION/${MAC}7
        export PATH=$VPMPATH:$VPMPATH/lib:$PATH
        export LD_LIBRARY_PATH=$VPMPATH:$VPMPATH/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/stck/benoit/lib
        export LD_LIBRARY_PATH=$VPMPATH:$VPMPATH/lib:$LD_LIBRARY_PATH
        export PYTHONPATH=$VPMPATH:$PYTHONPATH
        export PYTHONPATH=$VPMPATH/lib/python${PYTHONVR}/site-packages:$PYTHONPATH

        # turbo 
        export TURBOVERSION=UNAVAILABLE 

        # ErstaZ
        export EZPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/bin/eos
        export PYTHONPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/module_python/python3:$PYTHONPATH

    fi

elif [ "$MAC" = "sator" ]; then
    source /tmp_user/sator/elsa/Public/$ELSAVERSION/Dist/bin/sator_new21/.env_elsA
    export MOLA=$MOLASATOR
    export PATH=$PATH:/tmp_user/sator/lbernard/lib

    export PYTHONPATH=$MOLASATORext/sator/lib/python3.7/site-packages/:$PYTHONPATH
    export PATH=$MOLASATORext/sator/bin:$PATH


    # PUMA incompatible with intel21 ?
    export PumaRootDir=/tmp_user/sator/rboisard/TOOLS/Puma_${PUMAVERSION}_satornew
    export PYTHONPATH=$PumaRootDir/lib/python3.7/site-packages:$PYTHONPATH
    export PYTHONPATH=$PumaRootDir/lib/python3.7/site-packages/PUMA:$PYTHONPATH
    export LD_LIBRARY_PATH=$PumaRootDir/lib/python3.7:$LD_LIBRARY_PATH
    export PUMA_LICENCE=$PumaRootDir/pumalicence.txt

    export VPMPATH=/tmp_user/sator/lbernard/VPM/$VPMVERSION/sat_cas_r8
    export PATH=$VPMPATH:$PATH
    export LD_LIBRARY_PATH=$VPMPATH/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$VPMPATH:$LD_LIBRARY_PATH
    export PYTHONPATH=$VPMPATH:$PYTHONPATH
    export PYTHONPATH=$VPMPATH/lib/python${PYTHONVR}/site-packages:$PYTHONPATH

    # turbo
    export PYTHONPATH=/tmp_user/sator/jmarty/TOOLS/turbo/install/$TURBOVERSION/env_elsA_$ELSAVERSION/sator_new21/lib/python3.7/site-packages/:$PYTHONPATH

else
    echo -e "\033[91mERROR: MACHINE $KC NOT INCLUDED IN MOLA ENVIRONMENT\033[0m"
    exit 0
fi


STE=$PYTHONPATHL
STES='*'$STE'*'
if [ "$PYTHONPATH" = "" ]; then
    export PYTHONPATH=$STE
else
    case $PYTHONPATH in
	$STES)
#        echo '->var detected; not added'
        ;;
	*)
#        echo $PYTHONPATH, $STE
            export PYTHONPATH="$STE":"$PYTHONPATH"
            ;;
    esac
fi


export PYTHONPATH=$MOLA:$PYTHONPATH

alias python='python3'

alias treelab="python3 -c 'import MOLA.GUIs.TreeLab as t;import sys;t.launch(sys.argv)'"

alias mola_version="python3 -c 'import MOLA.InternalShortcuts as J;J.printEnvironment()'"

alias mola_jobsqueue_sator="python3 -c 'import MOLA.JobManager as JM;JM.getCurrentJobsStatus()'"

mola_version
