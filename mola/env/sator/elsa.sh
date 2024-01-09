#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

export ELSAVERSION=v5.2.03
export ELSA_VERBOSE_LEVEL=0 # cf elsA ticket 9689
export ELSA_MPI_LOG_FILES=OFF
export ELSA_MPI_APPEND=FALSE # cf elsA ticket 7849
export ELSA_NOLOG=ON

export VPMVERSION=Dev
export PUMAVERSION=v2.0.3
export TURBOVERSION=v1.3
export ERSTAZVERSION=v1.6.3
export OWNCASSREV=rev4670
export MAIAVERSION=1.2


source /tmp_user/sator/elsa/Public/$ELSAVERSION/Dist/bin/sator_new21/.env_elsA &>/dev/null
unset I_MPI_PMI_LIBRARY
export MOLA=$MOLASATOR

# PUMA
export PUMAVERSION=v2.0.3_mod
export PumaRootDir=/tmp_user/sator/rboisard/TOOLS/Puma_${PUMAVERSION}
export PYTHONPATH=$PumaRootDir/lib/python3.7/site-packages:$PYTHONPATH
export PYTHONPATH=$PumaRootDir/lib/python3.7/site-packages/PUMA:$PYTHONPATH
export LD_LIBRARY_PATH=$PumaRootDir/lib/python3.7:$LD_LIBRARY_PATH
export PUMA_LICENCE=$PumaRootDir/pumalicence.txt


# maia
module use --append /tmp_user/sator/sonics/usr/modules/
module load maia/$MAIAVERSION-dsi-cfd5_idx32

# VPM
export VPMPATH=/tmp_user/sator/lbernard/VPM/$VPMVERSION/sator/$ARCH
export PATH=$VPMPATH:$PATH
export LD_LIBRARY_PATH=$VPMPATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$VPMPATH:$LD_LIBRARY_PATH
export PYTHONPATH=$VPMPATH:$PYTHONPATH
export PYTHONPATH=$VPMPATH/lib/python${PYTHONVR}/site-packages:$PYTHONPATH

# turbo
export PYTHONPATH=/tmp_user/sator/jmarty/TOOLS/turbo/install/$TURBOVERSION/env_elsA_$ELSAVERSION/sator_new21/lib/python3.7/site-packages/:$PYTHONPATH

# ErstaZ
export EZPATH=/tmp_user/sator/rbarrier/ersatZ_$ERSTAZVERSION/bin/sator
export PYTHONPATH=/tmp_user/sator/rbarrier/ersatZ_$ERSTAZVERSION/python_module:$PYTHONPATH

# own Cassiopee
module load occt/7.6.1-gnu831
export OWNCASS=/tmp_user/sator/lbernard/Cassiopee/$OWNCASSREV/sator
export PATH=$PATH:$OWNCASS
export LD_LIBRARY_PATH=$OWNCASS/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$OWNCASS/lib/python3.7/site-packages:$PYTHONPATH


export PYTHONPATH=$MOLASATORext/sator/lib/python3.7/site-packages/:$PYTHONPATH
export PATH=$MOLASATORext/sator/bin:$PATH
export LD_LIBRARY_PATH=$MOLASATORext/sator/lib/python3.7/site-packages/PyQt5/Qt5/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/tmp_user/sator/lbernard/lib:$LD_LIBRARY_PATH
