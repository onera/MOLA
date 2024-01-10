#! /bin/sh
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

# ###############################################################################
# # ---------------- THESE LINES MUST BE ADAPTED BY DEVELOPERS ---------------- #
# export MOLAVER=Dev # looks to current directory name
# export MOLA=/stck/tbontemp/softs/MOLA/Dev
# export MOLASATOR=/tmp_user/sator/tbontemp/MOLA/Dev
# export MOLAext=/stck/lbernard/MOLA/Dev/ext # you should not modify this line
# export MOLASATORext=/tmp_user/sator/lbernard/MOLA/Dev/ext # you should not modify this line
# ###############################################################################

# Detection machine
KC=`uname -n`
EL8=`uname -r|grep el8`
MAC0=$(echo $KC | grep 'n'); if [ "$MAC0" != "" ]; then export MAC="sator"; fi
MAC0=$(echo $KC | grep 'sator'); if [ "$MAC0" != "" ]; then export MAC="sator"; fi
MAC0=$(echo $KC | grep 'ld'); if [ "$MAC0" != "" ]; then export MAC="ld"; fi
MAC0=$(echo $KC | grep 'eos'); if [ "$MAC0" != "" ]; then export MAC="ld"; fi
MAC0=$(echo $KC | grep 'spiro'); if [ "$MAC0" != "" ]; then export MAC="spiro"; fi
MAC0=$(echo $KC | grep 'visung'); if [ "$MAC0" != "" ]; then export MAC="visung"; fi

if [ "$MAC" = "ld" ] && [ ! "$EL8" ] ; then export MAC="visung"; fi

if [ "$MAC" = "visung" ] && [ "$EL8" ] ; then export MAC="ld"; fi

if [ "$1" = "" ]; then
    solver=mola
else
    solver=$1
fi

SCRIPT_DIR=$( \cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export MOLA=${SCRIPT_DIR%/mola/env/*}  # retain the part before /mola/env/*
export MOLAext=/stck/lbernard/MOLA/Dev/ext

# source the environment associated to the current machine and solver
echo "source $MOLA/mola/env/onera/$MAC/$solver.sh"
source $MOLA/mola/env/onera/$MAC/$solver.sh &>/dev/null
