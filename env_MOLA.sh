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

export ELSAVERSION=v5.1.03
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
EL8=`uname -r|grep el8`
MAC0=$(echo $KC | grep 'n'); if [ "$MAC0" != "" ]; then export MAC="sator"; fi
MAC0=$(echo $KC | grep 'sator'); if [ "$MAC0" != "" ]; then export MAC="sator"; fi
MAC0=$(echo $KC | grep 'ld'); if [ "$MAC0" != "" ]; then export MAC="ld"; fi
MAC0=$(echo $KC | grep 'eos'); if [ "$MAC0" != "" ]; then export MAC="ld"; fi
MAC0=$(echo $KC | grep 'spiro'); if [ "$MAC0" != "" ]; then export MAC="spiro"; fi
MAC0=$(echo $KC | grep 'visung'); if [ "$MAC0" != "" ]; then export MAC="visung"; fi
MAC0=$(echo $KC | grep 'visio'); if [ "$MAC0" != "" ]; then export MAC="visio"; fi
MAC0=$(echo $KC | grep 'celeste'); if [ "$MAC0" != "" ]; then export MAC="visio"; fi

if [ "$MAC" = "ld" ] && [ ! "$EL8" ] ; then export MAC="visung"; fi


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




# architecture
if lscpu | grep -q 'avx512' ; then
    export ARCH='avx512'
elif lscpu | grep -q 'avx2' ; then
    export ARCH='avx2'
elif lscpu | grep -q 'avx' ; then
    export ARCH='avx'
elif lscpu | grep -q 'sse4_2' ; then
    export ARCH='sse4_2'
elif lscpu | grep -q 'sse4_1' ; then
    export ARCH='sse4_1'
elif lscpu | grep -q 'ssse3' ; then
    export ARCH='ssse3'
elif lscpu | grep -q 'sse3' ; then
    export ARCH='sse3'
else
    export ARCH='sse2'
fi


if [ "$MAC" = "sator" ]; then
    source /tmp_user/sator/elsa/Public/$ELSAVERSION/Dist/bin/sator_new21/.env_elsA 2>/dev/null
    export MOLA=$MOLASATOR
    export PATH=$PATH:/tmp_user/sator/lbernard/lib

    # PUMA incompatible with intel21 ?
    export PumaRootDir=/tmp_user/sator/rboisard/TOOLS/Puma_${PUMAVERSION}_satornew
    export PYTHONPATH=$PumaRootDir/lib/python3.7/site-packages:$PYTHONPATH
    export PYTHONPATH=$PumaRootDir/lib/python3.7/site-packages/PUMA:$PYTHONPATH
    export LD_LIBRARY_PATH=$PumaRootDir/lib/python3.7:$LD_LIBRARY_PATH
    export PUMA_LICENCE=$PumaRootDir/pumalicence.txt

    # VPM
    export VPMPATH=/tmp_user/sator/lbernard/VPM/$VPMVERSION/sat_cas_r8
    export PATH=$VPMPATH:$PATH
    export LD_LIBRARY_PATH=$VPMPATH/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$VPMPATH:$LD_LIBRARY_PATH
    export PYTHONPATH=$VPMPATH:$PYTHONPATH
    export PYTHONPATH=$VPMPATH/lib/python${PYTHONVR}/site-packages:$PYTHONPATH

    # turbo
    export PYTHONPATH=/tmp_user/sator/jmarty/TOOLS/turbo/install/$TURBOVERSION/env_elsA_$ELSAVERSION/sator_new21/lib/python3.7/site-packages/:$PYTHONPATH

    export PYTHONPATH=$MOLASATORext/sator/lib/python3.7/site-packages/:$PYTHONPATH
    export PATH=$MOLASATORext/sator/bin:$PATH
    export LD_LIBRARY_PATH=$MOLASATORext/sator/lib/python3.7/site-packages/PyQt5/Qt5/lib/:$LD_LIBRARY_PATH

elif [ "$MAC" = "spiro" ]; then
    if [ "$EL8" ]; then
            echo 'loading MOLA environment for CentOS 8'
    else
        echo -e "\033[91mERROR: SPIRO CENTOS 7 NOT SUPPORTED ANYMORE\033[0m"
        exit 0
    fi
    source /stck/elsa/Public/$ELSAVERSION/Dist/bin/spiro-el8_mpi/.env_elsA 2>/dev/null


    # to avoid message:
    # MPI startup(): Warning: I_MPI_PMI_LIBRARY will be ignored since the hydra process manager was found
    # source : https://www.osc.edu/supercomputing/batch-processing-at-osc/slurm_migration/slurm_migration_issues
    unset I_MPI_PMI_LIBRARY 

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
    export PYTHONPATH=$VPMPATH/lib/python3.7/site-packages:$PYTHONPATH

    # turbo
    export PYTHONPATH=/stck/jmarty/TOOLS/turbo/install/$TURBOVERSION/env_elsA_$ELSAVERSION/spiro3_mpi/lib/python3.7/site-packages/:$PYTHONPATH
    
    # ErstaZ
    export EZPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/bin/spiro
    export PYTHONPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/module_python/python3:$PYTHONPATH

    # MAIA
    export MAIA_HOME=/scratchm/jcoulet/aa_install_py3/maia/opt-cfd5_21/
    export LD_LIBRARY_PATH=$MAIA_HOME/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=$MAIA_HOME/lib/python3.7/site-packages:$PYTHONPATH

    # own Cassiopee
    module load occt/7.6.1-gnu831
    export OWNCASSREV=rev4386
    export OWNCASS=/stck/lbernard/Cassiopee/$OWNCASSREV/spiro
    export LD_LIBRARY_PATH=$OWNCASS/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=$OWNCASS/lib/python3.7/site-packages:$PYTHONPATH

    # external python packages
    export PYTHONPATH=$MOLAext/spiro_el8/lib/python3.7/site-packages/:$PYTHONPATH
    export PATH=$MOLAext/spiro_el8/bin:$PATH
    export LD_LIBRARY_PATH=$MOLAext/spiro_el8/lib/python3.7/site-packages/PyQt5/Qt5/lib/:$LD_LIBRARY_PATH

    alias mola_sinter='srun --export=ALL,SLURM_EXACT=1,SLURM_OVERLAP=1 --immediate=2 --pty --qos c1_inter_giga'

elif [ "$MAC" = "visung" ]; then

    source /stck/elsa/Public/$ELSAVERSION/Dist/bin/eos-intel3_mpi/.env_elsA 2>/dev/null
    module load texlive/2016 # for LaTeX rendering in matplotlib with STIX font
    module load freetype/2.10.2
    module load pointwise/2022.1.2

    export PYTHONEXE=python3
    alias python=python3

    # turbo
    # export PYTHONPATH=/stck/jmarty/TOOLS/turbo/install/$TURBOVERSION/env_elsA_$ELSAVERSION/eos-intel3_mpi/lib/python3.6/site-packages/:$PYTHONPATH

    # ErstaZ
    export EZPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/bin/visio
    export PYTHONPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/module_python/python3:$PYTHONPATH

    # VPM
    export VPMPATH=/stck/lbernard/VPM/$VPMVERSION/visung
    export LD_LIBRARY_PATH=$VPMPATH/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=$VPMPATH/lib/python3.7/site-packages:$PYTHONPATH

    # own Cassiopee
    export OWNCASSREV=rev4386
    export OWNCASS=/stck/lbernard/Cassiopee/$OWNCASSREV/visung
    export LD_LIBRARY_PATH=$OWNCASS/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=$OWNCASS/lib/python3.7/site-packages:$PYTHONPATH
    # requireed by OCC, not provided currently in CentOS 7 visung
    export LD_LIBRARY_PATH=/stck/benoit/opencascade/lib:$LD_LIBRARY_PATH
    

    # external python dependencies
    export PYTHONPATH=$MOLAext/visung/lib/python3.7/site-packages/:$PYTHONPATH
    export PATH=$MOLAext/visung/bin:$PATH
    export LD_LIBRARY_PATH=$MOLAext/visung/lib/python3.7/site-packages/PyQt5/Qt5/lib/:$LD_LIBRARY_PATH
    

elif [ "$MAC" = "ld" ]; then

    source /stck/elsa/Public/$ELSAVERSION/Dist/bin/local-os8_mpi/.env_elsA 2>/dev/null
    module load texlive/2021 # for LaTeX rendering in matplotlib with STIX font
    module load vscode/1.74.3
    module load pointwise/2022.1.2
    # # module load paraview/5.11.0 # provokes python and libraries incompatibilities
    module load occt/7.6.1-gnu831

    export OPENMPIOVERSUBSCRIBE='-oversubscribe'

    alias python=python3
    unset I_MPI_PMI_LIBRARY

    # turbo 
    export TURBOVERSION=UNAVAILABLE 

    # ErstaZ
    export EZPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/bin/centos8
    export PYTHONPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/module_python/python3:$PYTHONPATH

    # VPM
    export VPMPATH=/stck/lbernard/VPM/$VPMVERSION/ld8/$ARCH
    export PATH=$VPMPATH:$VPMPATH/lib:$PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/stck/benoit/lib
    export LD_LIBRARY_PATH=$VPMPATH:$VPMPATH/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=$VPMPATH:$PYTHONPATH
    export PYTHONPATH=$VPMPATH/lib/python3.8/site-packages:$PYTHONPATH
    # replaces module load intel/21.2.0 since this module
    # brakes MPI https://elsa.onera.fr/issues/10933#note-16
    export LD_LIBRARY_PATH=/opt/tools/intel/oneapi/compiler/2021.2.0/linux/compiler/lib/intel64_lin/:$LD_LIBRARY_PATH

    # own Cassiopee
    export OWNCASSREV=rev4386
    export OWNCASS=/stck/lbernard/Cassiopee/$OWNCASSREV/ld8
    export LD_LIBRARY_PATH=$OWNCASS/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=$OWNCASS/lib/python3.8/site-packages:$PYTHONPATH

    # external python dependencies
    export PYTHONPATH=$MOLAext/ld8/lib/python3.8/site-packages/:$PYTHONPATH
    export PATH=$MOLAext/ld8/bin:$PATH
    export LD_LIBRARY_PATH=$MOLAext/ld8/lib/python3.8/site-packages/PyQt5/Qt5/lib/:$LD_LIBRARY_PATH

elif [ "$MAC" = "visio" ]; then
    export ELSAVERSION=UNAVAILABLE # TODO adapt this once #10587 fixed
    echo -e "\033[93mWARNING: elsA is not installed yet in VISIO CentOS 6\033[0m"
    echo -e "\033[93mwatch https://elsa.onera.fr/issues/10587 for more information\033[0m"

    . /etc/profile.d/modules-dri.sh 2>/dev/null
    module load subversion/1.7.6
    module load python/3.6.1
    module unload $(module -t list 2>&1 | grep -i intel)
    module load gcc/4.8.1
    module load intel/17.0.4
    module load impi/17
    alias python=python3
    export PYTHONEXE=python3

    # turbo 
    export TURBOVERSION=UNAVAILABLE 

    # ErstaZ
    export EZPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/bin/visio
    export PYTHONPATH=/stck/rbarrier/PARTAGE/ersatZ_$ERSTAZVERSION/module_python/python3:$PYTHONPATH

    # own Cassiopee (includes OCC, Apps, VPM)
    export OWNCASSREV=rev4386
    export OWNCASS=/stck/lbernard/Cassiopee/$OWNCASSREV/visio
    export LD_LIBRARY_PATH=$OWNCASS/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/stck/benoit/opencascade2/lib:$CASSIOPEE/Dist/bin/"$ELSAPROD":$CASSIOPEE/Dist/bin/"$ELSAPROD"/lib:/usr/local/hdf5-intel-1.8.8/lib:/usr/local/gtk+3/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=$OWNCASS/lib/python3.6/site-packages:$PYTHONPATH

    # external python dependencies
    export PYTHONPATH=$MOLAext/visio/lib/python3.6/site-packages/:$PYTHONPATH
    export PATH=$MOLAext/visio/bin:$PATH
    export LD_LIBRARY_PATH=$MOLAext/visio/lib/python3.6/site-packages/PyQt5/Qt5/lib/:$LD_LIBRARY_PATH


else
    echo -e "\033[91mERROR: MACHINE $KC NOT INCLUDED IN MOLA ENVIRONMENT\033[0m"
    exit 0
fi

export PYTHONPATH=$MOLA:$PYTHONPATH

alias python='python3'

alias treelab="python3 -c 'import MOLA.GUIs.TreeLab as t;import sys;t.launch(sys.argv)'"

alias mola_version="mpirun -np 1 python3 -c 'import MOLA.InternalShortcuts as J;J.printEnvironment()'"

alias mola_jobsqueue_sator="python3 -c 'import MOLA.JobManager as JM;JM.getCurrentJobsStatus()'"

mola_version
